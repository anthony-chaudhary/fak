package ggufload

// gguf_load_provenance_test.go — witnesses for the GGUF load-provenance
// producer (#4746, root incident #4273).
//
// The acceptance criteria these bind, in the issue's own words:
//   - "A Qwen3.6 load reports the ssm_a: negative-decay -> A_log transform and
//     how many layers/tensors used it" — TestLoadProvenanceReportsQwen36SSMADecayInversion.
//   - "The provenance artifact is deterministic for identical model bytes +
//     loader commit" — TestLoadProvenanceDeterministicPerBytesAndLoaderRev.
//   - "Invalid source-domain values fail with tensor name, transform ID, index,
//     and expected domain" — TestSSMADomainRefusalNamesAllFourFacts.
//   - "Operators can compare two runs and immediately see transform/path
//     differences" — TestLoadProvenanceDiffRoutesTransformDeltaToLoader.

import (
	"bytes"
	"errors"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// writeLoadProvenanceGGUF builds a header-only GGUF (no tensor data blob at all,
// so any path reaching for a weight payload would read past EOF) carrying:
//   - one blk.<n>.ssm_a per layer — the #4273 transform, per-layer;
//   - one identity tensor, which must never appear in the artifact;
//   - one model-global output_norm.weight, which is transformed but carries no
//     layer index, so it proves Layers counts layers and not tensors.
func writeLoadProvenanceGGUF(arch string, layers int) []byte {
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(layers)+2, 2)
	writeKVString(&b, "general.architecture", arch)
	writeKVUint32(&b, "general.alignment", 32)
	off := uint64(0)
	for i := 0; i < layers; i++ {
		writeTensorInfoForTest(&b, "blk."+strconv.Itoa(i)+".ssm_a", []uint64{4}, TensorF32, off)
		off += 64
	}
	writeTensorInfoForTest(&b, "blk.0.ffn_gate.weight", []uint64{8, 8}, TensorF32, off)
	writeTensorInfoForTest(&b, "output_norm.weight", []uint64{8}, TensorF32, off+64)
	return b.Bytes()
}

func loadProvenanceScopeForTest() LoadProvenanceScope {
	return LoadProvenanceScope{
		ModelDigest: "sha256:" + strings.Repeat("ab", 32),
		ModelBytes:  4096,
		LoaderRev:   "fak-loader@095101ee8",
		Quant:       "Q4_K",
		ForwardPath: model.ForwardQwen35GDN,
	}
}

func readProvenanceFixture(t *testing.T, arch string, layers int) *File {
	t.Helper()
	raw := writeLoadProvenanceGGUF(arch, layers)
	f, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Read header-only gguf: %v", err)
	}
	// The fixture really is payload-free: the header parse consumed the whole
	// file, so no weight byte exists to be read. Everything asserted below is
	// therefore proven to come from the header alone.
	if f.TensorDataOffset < int64(len(raw)) {
		t.Fatalf("fixture is not payload-free: data offset %d < file size %d", f.TensorDataOffset, len(raw))
	}
	return f
}

// TestLoadProvenanceReportsQwen36SSMADecayInversion is the headline acceptance
// witness: a Qwen3.6 checkpoint's provenance names the ssm_a decay inversion and
// says how much of the model it ran on. Before #4746 the runtime could report the
// model id, the quant mode, and tokens/sec — every one of which was IDENTICAL
// between the broken and the fixed #4273 load — but not this.
func TestLoadProvenanceReportsQwen36SSMADecayInversion(t *testing.T) {
	const layers = 3
	f := readProvenanceFixture(t, "qwen3.6", layers)

	p, err := f.LoadProvenance(loadProvenanceScopeForTest())
	if err != nil {
		t.Fatalf("LoadProvenance: %v", err)
	}

	// The vendor spelling canonicalizes onto the family fak actually runs.
	if p.GGUFArch != "qwen35" {
		t.Errorf("GGUFArch = %q, want the canonicalized %q for a qwen3.6 header", p.GGUFArch, "qwen35")
	}
	if p.GGUFVersion != uint32(Version) {
		t.Errorf("GGUFVersion = %d, want %d", p.GGUFVersion, Version)
	}

	tensors, layerCount, ok := p.TransformTensors(TransformInvertNegExpDecay)
	if !ok {
		t.Fatalf("provenance does not report %s at all; transforms = %+v", TransformInvertNegExpDecay, p.Transforms)
	}
	if tensors != layers || layerCount != layers {
		t.Errorf("%s ran on %d tensors / %d layers, want %d/%d",
			TransformInvertNegExpDecay, tensors, layerCount, layers, layers)
	}

	// The ssm_a record must carry the semantic domains, not just an id — the
	// domain pair is what tells an operator WHICH direction the loader read.
	var ssmA model.TransformRecord
	for _, r := range p.Transforms {
		if r.External == "ssm_a" {
			ssmA = r
		}
	}
	if ssmA.ID == "" {
		t.Fatalf("no ssm_a transform record in %+v", p.Transforms)
	}
	if ssmA.Canonical != "linear_attn.A_log" {
		t.Errorf("ssm_a canonical = %q, want linear_attn.A_log", ssmA.Canonical)
	}
	if !strings.Contains(ssmA.SourceDomain, "negative") {
		t.Errorf("ssm_a source domain %q must describe the negated-decay export domain", ssmA.SourceDomain)
	}
	if !ssmA.DomainValidated {
		t.Error("ssm_a must record DomainValidated: the loader validates the finite-negative source domain")
	}

	// A model-global transformed tensor contributes a tensor but no layer.
	var norm model.TransformRecord
	for _, r := range p.Transforms {
		if r.External == "output_norm.weight" {
			norm = r
		}
	}
	if norm.ID != TransformGainMinusOne {
		t.Fatalf("output_norm.weight transform = %q, want %q", norm.ID, TransformGainMinusOne)
	}
	if norm.Tensors != 1 || norm.Layers != 0 {
		t.Errorf("output_norm.weight recorded %d tensors / %d layers, want 1/0 (model-global has no layer)",
			norm.Tensors, norm.Layers)
	}

	// Identity mappings are the absence of provenance, never an entry in it.
	for _, r := range p.Transforms {
		if r.External == "ffn_gate.weight" {
			t.Errorf("identity tensor ffn_gate.weight recorded a transform %q", r.ID)
		}
	}

	// The header-only producer records no VALUE evidence. This is asserted, not
	// merely documented: a future change that synthesizes an all-clear
	// DomainCheck without running the guard would be claiming a validation that
	// never happened, which is the failure mode the artifact exists to prevent.
	if len(p.DomainChecks) != 0 {
		t.Errorf("header-only provenance must record no domain checks, got %+v", p.DomainChecks)
	}
	if len(p.TensorSummaries) != 0 {
		t.Errorf("header-only provenance must record no tensor summaries, got %+v", p.TensorSummaries)
	}
}

// TestLoadProvenanceDeterministicPerBytesAndLoaderRev pins the property that
// makes the digest usable as evidence: it is a function of the model bytes and
// the loader revision, and of nothing else. Two independent builds of the same
// checkpoint agree; a different loader revision or different bytes does not.
func TestLoadProvenanceDeterministicPerBytesAndLoaderRev(t *testing.T) {
	f := readProvenanceFixture(t, "qwen3.6", 2)
	scope := loadProvenanceScopeForTest()

	first, err := f.LoadProvenance(scope)
	if err != nil {
		t.Fatalf("LoadProvenance: %v", err)
	}
	// A second, independently parsed File over identical bytes.
	g := readProvenanceFixture(t, "qwen3.6", 2)
	second, err := g.LoadProvenance(scope)
	if err != nil {
		t.Fatalf("LoadProvenance (second parse): %v", err)
	}
	if first.Digest() != second.Digest() {
		t.Fatalf("identical bytes + loader rev produced different digests:\n %s\n %s",
			first.Digest(), second.Digest())
	}
	if !bytes.Equal(first.JSON(), second.JSON()) {
		t.Errorf("identical inputs serialized differently:\n%s\n%s", first.JSON(), second.JSON())
	}

	bumped := scope
	bumped.LoaderRev = "fak-loader@deadbeef"
	moved, err := f.LoadProvenance(bumped)
	if err != nil {
		t.Fatalf("LoadProvenance (bumped rev): %v", err)
	}
	if moved.Digest() == first.Digest() {
		t.Error("a different loader revision must change the artifact digest")
	}

	// Different STRUCTURE (a layer more) must move the manifest digest, so a
	// publication claim cannot be carried across an architecturally different
	// checkpoint.
	wider := readProvenanceFixture(t, "qwen3.6", 3)
	if wider.CanonicalManifestDigest() == f.CanonicalManifestDigest() {
		t.Error("checkpoints with different tensor directories must have different manifest digests")
	}
}

// TestLoadProvenanceRefusesIncompleteScope proves the producer is fail-closed:
// an artifact missing a required fact is inconclusive evidence, and inconclusive
// evidence attached to a readiness claim is worse than none — it looks like
// provenance while proving nothing.
func TestLoadProvenanceRefusesIncompleteScope(t *testing.T) {
	f := readProvenanceFixture(t, "qwen3.6", 1)
	for _, tc := range []struct {
		name  string
		mutfn func(*LoadProvenanceScope)
	}{
		{"no model digest", func(s *LoadProvenanceScope) { s.ModelDigest = "" }},
		{"no model bytes", func(s *LoadProvenanceScope) { s.ModelBytes = 0 }},
		{"no loader rev", func(s *LoadProvenanceScope) { s.LoaderRev = "" }},
		{"no quant", func(s *LoadProvenanceScope) { s.Quant = "" }},
		{"no forward path", func(s *LoadProvenanceScope) { s.ForwardPath = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scope := loadProvenanceScopeForTest()
			tc.mutfn(&scope)
			if _, err := f.LoadProvenance(scope); err == nil {
				t.Fatalf("LoadProvenance accepted a scope with %s", tc.name)
			}
		})
	}
}

// TestLoadProvenanceNoTransformsForIdentityArch proves an empty transform list
// means "this architecture is consumed exactly as stored" rather than "nobody
// looked". An arch whose GGUF already stores the HF rotary layout declares no
// contracts, so its provenance is legitimately transform-free.
func TestLoadProvenanceNoTransformsForIdentityArch(t *testing.T) {
	f := readProvenanceFixture(t, "qwen3", 2)
	scope := loadProvenanceScopeForTest()
	scope.ForwardPath = model.ForwardAttnSeqGQA
	p, err := f.LoadProvenance(scope)
	if err != nil {
		t.Fatalf("LoadProvenance: %v", err)
	}
	if len(p.Transforms) != 0 {
		t.Errorf("arch qwen3 stores the canonical layout; want no transforms, got %+v", p.Transforms)
	}
	if _, _, ok := p.TransformTensors(TransformInvertNegExpDecay); ok {
		t.Error("arch qwen3 must not report the ssm_a decay inversion")
	}
}

// TestLoadProvenanceDiffRoutesTransformDeltaToLoader is the operator-comparison
// acceptance: two runs that differ in loader SEMANTICS must arrive already
// triaged to the loader, not as a JSON diff to eyeball — and the delta must name
// the transform without disclosing a weight or a prompt.
func TestLoadProvenanceDiffRoutesTransformDeltaToLoader(t *testing.T) {
	scope := loadProvenanceScopeForTest()
	good, err := readProvenanceFixture(t, "qwen3.6", 3).LoadProvenance(scope)
	if err != nil {
		t.Fatalf("LoadProvenance (qwen3.6): %v", err)
	}
	// The #4273 shape: the SAME checkpoint read by a loader that no longer
	// applies the decay inversion. Model bytes and manifest are held equal so
	// the only difference is loader semantics.
	broken := good
	broken.Transforms = nil

	deltas := model.DiffLoadProvenance(good, broken)
	if len(deltas) == 0 {
		t.Fatal("a vanished transform produced no delta")
	}
	var sawLoader bool
	for _, d := range deltas {
		if d.Area == model.AreaLoader {
			sawLoader = true
		}
		if d.Area == model.AreaModelBytes {
			t.Errorf("same bytes must not route to model-bytes: %s", d)
		}
	}
	if !sawLoader {
		t.Errorf("a vanished transform must route to %q, got %+v", model.AreaLoader, deltas)
	}
}

// TestSSMADomainRefusalNamesAllFourFacts binds the acceptance criterion
// "invalid source-domain values fail with tensor name, transform ID, index, and
// expected domain" to the LIVE loader path, not to a helper called in isolation:
// normalizeCanonicalTensorData is what a real load runs.
func TestSSMADomainRefusalNamesAllFourFacts(t *testing.T) {
	cfg := transformContractProbeConfig()
	const name = "model.layers.0.linear_attn.A_log"

	for _, tc := range []struct {
		name string
		bad  float32
	}{
		// A value only plausible in the CANONICAL domain — the exact fixture /
		// exporter mistake behind #4273.
		{"positive A_log", 0.5},
		{"zero", 0},
		{"NaN", float32(math.NaN())},
		// -Inf reads as "negative" to a naive `>= 0` test, but log(-(-Inf)) is
		// +Inf: it used to canonicalize into an infinite A_log instead of being
		// refused. The declared domain says FINITE and strictly negative.
		{"negative infinity", float32(math.Inf(-1))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := []float32{-1.1, -1.2, -1.3, -1.4}
			src[1] = tc.bad

			_, err := normalizeCanonicalTensorData(name, src, cfg)
			if err == nil {
				t.Fatalf("loader accepted %v in the ssm_a source domain", tc.bad)
			}
			var domErr *model.SourceDomainError
			if !errors.As(err, &domErr) {
				t.Fatalf("refusal is %T (%v), want a typed *model.SourceDomainError so a "+
					"provenance artifact can record its evidence", err, err)
			}
			// Fact 1: the tensor.
			if domErr.Tensor != name {
				t.Errorf("Tensor = %q, want %q", domErr.Tensor, name)
			}
			// Fact 2: the transform id — the fact the hand-rolled message omitted,
			// leaving the operator to guess which mapping to audit.
			if domErr.Transform != TransformInvertNegExpDecay {
				t.Errorf("Transform = %q, want %q", domErr.Transform, TransformInvertNegExpDecay)
			}
			// Fact 3: the element index.
			if domErr.Index < 0 || domErr.Index >= len(src) {
				t.Errorf("Index = %d, out of range for a %d-element tensor", domErr.Index, len(src))
			}
			// Fact 4: the expected domain, quoted from the one shared constant.
			if domErr.ExpectedDomain != model.NegativeDecayDomain {
				t.Errorf("ExpectedDomain = %q, want %q", domErr.ExpectedDomain, model.NegativeDecayDomain)
			}

			// The operator message shows the offending value; the publish-safe
			// evidence withholds it, so an artifact attached to a public bundle
			// names the failure without leaking a weight element.
			if !strings.Contains(domErr.Error(), "violates transform") {
				t.Errorf("operator error %q does not name the violated transform", domErr.Error())
			}
			ev := domErr.Evidence()
			for _, fact := range []string{name, TransformInvertNegExpDecay, model.NegativeDecayDomain} {
				if !strings.Contains(ev, fact) {
					t.Errorf("evidence %q omits required fact %q", ev, fact)
				}
			}
		})
	}
}

// TestSSMAValidSourceDomainStillLoads guards the other side of the stricter
// predicate: tightening the guard must not start refusing legitimate exports.
func TestSSMAValidSourceDomainStillLoads(t *testing.T) {
	cfg := transformContractProbeConfig()
	src := make([]float32, 4)
	for i := range src {
		src[i] = -float32(math.Exp(float64(i+1) * 0.1))
	}
	out, err := normalizeCanonicalTensorData("model.layers.0.linear_attn.A_log", src, cfg)
	if err != nil {
		t.Fatalf("loader refused a valid finite-negative ssm_a tensor: %v", err)
	}
	if len(out) != len(src) {
		t.Fatalf("canonical tensor has %d values, want %d", len(out), len(src))
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("canonical A_log[%d] = %v, want a finite real", i, v)
		}
	}
}
