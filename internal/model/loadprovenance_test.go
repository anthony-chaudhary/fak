package model

// loadprovenance_test.go — the acceptance gate for the model-load provenance
// artifact (#4746, root incident #4273).
//
// WHAT THIS PINS. The artifact's whole value is that an operator can trust it
// without re-deriving it, so each property the issue names is bound here as a
// behavioral test rather than left to the doc comment:
//
//   - a qwen35/Qwen3.6-shaped load REPORTS the ssm_a decay inversion and how
//     many tensors and LAYERS it ran on (the #4273 question, asked first);
//   - the digest is a function of the recorded FACTS only — recording order
//     cannot move it, and a genuine semantic change always does;
//   - a source-domain violation names tensor, transform, index, and expected
//     domain, and the publish-safe rendering withholds the offending VALUE;
//   - two runs diff into a CLOSED investigation area, so a delta arrives
//     pre-triaged instead of as raw JSON;
//   - no interior weight value survives into the artifact.
//
// The transform-id literals below deliberately duplicate internal/ggufload's
// constants instead of importing them: ggufload imports model, so the reverse
// import is a cycle. Duplicating the STRINGS is correct — they are the shared
// wire vocabulary, and the ggufload contract test owns their declaration side.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
)

const (
	// The composite transform a qwen35 ssm_a tensor carries: the GDN value-head
	// deinterleave, then the negative-decay inversion.
	testSSMATransform = "value-head-deinterleave+invert-neg-exp-decay"
	testInvertDecay   = "invert-neg-exp-decay"
	// The second non-identity qwen35 mapping: the NormGain1p residual shift.
	testGainMinusOne = "gain-minus-one"
)

// qwen36SSMAObservations builds the per-tensor observations a qwen35-family
// load emits for blk.<n>.ssm_a across layers, mirroring the ggufload contract.
func qwen36SSMAObservations(layers []int) []TransformObservation {
	obs := make([]TransformObservation, 0, len(layers))
	for _, n := range layers {
		obs = append(obs, TransformObservation{
			Tensor:          fmt.Sprintf("blk.%d.ssm_a", n),
			Transform:       testSSMATransform,
			External:        "ssm_a",
			Canonical:       "linear_attn.A_log",
			SourceDomain:    NegativeDecayDomain,
			CanonicalDomain: "raw gated-delta-net decay parameter A_log (finite real)",
			Lossless:        false,
			Invertible:      true,
			DomainValidated: true,
		})
	}
	return obs
}

// normGainObservations builds the observations for the second non-identity
// mapping a qwen35 load carries, so the artifact under test holds MORE than one
// transform, domain check, and tensor summary. A single-element artifact cannot
// exercise ordering at all — see TestProvenanceDigestIsOrderIndependent...
func normGainObservations(layers []int) []TransformObservation {
	obs := make([]TransformObservation, 0, len(layers))
	for _, n := range layers {
		obs = append(obs, TransformObservation{
			Tensor:          fmt.Sprintf("blk.%d.attn_norm.weight", n),
			Transform:       testGainMinusOne,
			External:        "attn_norm.weight",
			Canonical:       "input_layernorm.weight",
			SourceDomain:    "full RMSNorm gain g (typically near 1)",
			CanonicalDomain: "residual RMSNorm gain g-1 consumed by the NormGain1p forward",
			Lossless:        false,
			Invertible:      true,
		})
	}
	return obs
}

// completeProvenance is a fully-populated artifact that Validate accepts, used
// as the base for the determinism and diff tests. It deliberately carries TWO
// entries in every list so ordering is observable.
func completeProvenance(t *testing.T) LoadProvenance {
	t.Helper()
	p := LoadProvenance{
		ModelDigest:    "sha256:" + strings.Repeat("ab", 32),
		ModelBytes:     4_294_967_296,
		GGUFArch:       "qwen35",
		GGUFVersion:    3,
		ManifestDigest: "sha256:" + strings.Repeat("cd", 32),
		LoaderRev:      "4f302a441",
		Quant:          "Q4_K_M",
		ForwardPath:    ForwardQwen35GDN,
		Transforms: FoldTransformRecords(append(
			qwen36SSMAObservations([]int{0, 1, 2, 3}),
			normGainObservations([]int{0, 1, 2, 3})...)),
		DomainChecks: []DomainCheck{
			{
				Transform:      testSSMATransform,
				ExpectedDomain: NegativeDecayDomain,
				Tensors:        4,
				Rejected:       0,
			},
			{
				Transform:      testGainMinusOne,
				ExpectedDomain: "finite RMSNorm gain",
				Tensors:        4,
				Rejected:       0,
			},
		},
		TensorSummaries: []TensorSummary{
			SummarizeTransformedTensor("model.layers.0.linear_attn.A_log", testSSMATransform,
				[]int{2, 2}, []float32{0.5, -0.25, 1.5, -2}),
			SummarizeTransformedTensor("model.layers.0.input_layernorm.weight", testGainMinusOne,
				[]int{4}, []float32{0.5, 0.25, 0.125, 0.0625}),
		},
	}
	if err := p.Validate(); err != nil {
		t.Fatalf("base artifact must be valid: %v", err)
	}
	return p
}

// reversed returns a reversed copy, used to feed Normalize a list in the WRONG
// order without routing through FoldTransformRecords (which sorts on its own and
// would mask an ordering bug in Normalize).
func reversed[T any](in []T) []T {
	out := make([]T, len(in))
	for i, v := range in {
		out[len(in)-1-i] = v
	}
	return out
}

// TestProvenanceReportsSSMADecayInversionWithLayerCounts is acceptance box 1: a
// Qwen3.6-shaped load reports the ssm_a negative-decay -> A_log transform and
// how many layers/tensors used it. The layer count is the load-bearing half —
// a bare "the transform is declared" tells an operator nothing about whether it
// actually ran on the whole model, which is exactly what #4273 needed to know.
func TestProvenanceReportsSSMADecayInversionWithLayerCounts(t *testing.T) {
	// 12 hybrid layers out of a deeper model: linear-attention layers are
	// interleaved, so the layer set is sparse and non-contiguous.
	layers := []int{3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}
	p := LoadProvenance{Transforms: FoldTransformRecords(qwen36SSMAObservations(layers))}

	tensors, layerCount, ok := p.TransformTensors(testInvertDecay)
	if !ok {
		t.Fatalf("TransformTensors(%q) reported the decay inversion did not run; the artifact must "+
			"surface it by component id inside the composite %q", testInvertDecay, testSSMATransform)
	}
	if tensors != len(layers) {
		t.Errorf("tensors = %d, want %d", tensors, len(layers))
	}
	if layerCount != len(layers) {
		t.Errorf("layers = %d, want %d", layerCount, len(layers))
	}

	// The composite id must also resolve, and the record must name both domains
	// so the operator can read the mapping without opening the loader.
	if _, _, ok := p.TransformTensors(testSSMATransform); !ok {
		t.Errorf("TransformTensors(%q) must match the full composite id too", testSSMATransform)
	}
	recs := p.Normalize().Transforms
	if len(recs) != 1 {
		t.Fatalf("one tensor mapping must fold to one record, got %d: %+v", len(recs), recs)
	}
	if recs[0].External != "ssm_a" || recs[0].Canonical != "linear_attn.A_log" {
		t.Errorf("record must name the external->canonical mapping, got %s -> %s",
			recs[0].External, recs[0].Canonical)
	}
	if recs[0].SourceDomain != NegativeDecayDomain {
		t.Errorf("source domain = %q, want %q", recs[0].SourceDomain, NegativeDecayDomain)
	}
	if !recs[0].DomainValidated {
		t.Error("ssm_a carries a source-domain guard; the record must say so")
	}

	// A transform that never ran must report ok=false rather than a zero count,
	// so "absent" is distinguishable from "ran on nothing".
	if _, _, ok := p.TransformTensors("rotary-unpermute"); ok {
		t.Error("a transform that did not run must report ok=false")
	}
}

// TestFoldTransformRecordsIsIdempotentPerTensor pins that a loader which
// re-observes a tensor cannot inflate the counts and perturb the digest.
func TestFoldTransformRecordsIsIdempotentPerTensor(t *testing.T) {
	once := qwen36SSMAObservations([]int{0, 1})
	twice := append(append([]TransformObservation(nil), once...), once...)

	a := LoadProvenance{Transforms: FoldTransformRecords(once)}
	b := LoadProvenance{Transforms: FoldTransformRecords(twice)}

	tensorsA, layersA, _ := a.TransformTensors(testInvertDecay)
	tensorsB, layersB, _ := b.TransformTensors(testInvertDecay)
	if tensorsA != tensorsB || layersA != layersB {
		t.Errorf("re-observing a tensor changed the counts: (%d,%d) -> (%d,%d)",
			tensorsA, layersA, tensorsB, layersB)
	}
	if a.Digest() != b.Digest() {
		t.Error("re-observing a tensor moved the digest; folding must be idempotent per tensor")
	}
}

// TestProvenanceDigestIsOrderIndependentAndChangeSensitive is acceptance box 2:
// the artifact is deterministic for identical model bytes + loader commit. Both
// halves matter — an order-sensitive digest produces false "the loader changed"
// alarms, and a change-insensitive one lets a real semantic change publish under
// a stale digest.
func TestProvenanceDigestIsOrderIndependentAndChangeSensitive(t *testing.T) {
	base := completeProvenance(t)

	// Same facts, recorded in a different order, with incidental whitespace.
	//
	// The lists are reversed DIRECTLY rather than re-folded: FoldTransformRecords
	// sorts its own output, so routing through it would hide an ordering bug in
	// Normalize. A loader that appends records as it walks the tensor list — the
	// obvious way to write the producer — hits exactly this path.
	shuffled := base
	shuffled.LoaderRev = "  " + base.LoaderRev + "  "
	shuffled.Transforms = reversed(base.Transforms)
	shuffled.DomainChecks = reversed(base.DomainChecks)
	shuffled.TensorSummaries = reversed(base.TensorSummaries)
	if len(shuffled.Transforms) < 2 || len(shuffled.DomainChecks) < 2 || len(shuffled.TensorSummaries) < 2 {
		t.Fatalf("this test is vacuous unless every list holds >1 entry: %d/%d/%d",
			len(shuffled.Transforms), len(shuffled.DomainChecks), len(shuffled.TensorSummaries))
	}

	if base.Digest() != shuffled.Digest() {
		t.Errorf("recording order/whitespace moved the digest:\n base = %s\n shuf = %s",
			base.Digest(), shuffled.Digest())
	}
	// Repeated evaluation is stable (no map iteration order leaking in).
	for i := 0; i < 8; i++ {
		if got := base.Digest(); got != base.Digest() {
			t.Fatalf("digest is unstable across evaluations at i=%d: %s", i, got)
		}
	}

	// Every semantic change must move the digest. Each mutation below is a fact
	// a publication claim would be wrong to carry under the old address.
	for _, tc := range []struct {
		name  string
		mutot func(*LoadProvenance)
	}{
		{"model bytes differ", func(p *LoadProvenance) { p.ModelDigest = "sha256:" + strings.Repeat("ef", 32) }},
		{"loader revision differs", func(p *LoadProvenance) { p.LoaderRev = "deadbeef" }},
		{"quant differs", func(p *LoadProvenance) { p.Quant = "Q8_0" }},
		{"forward path differs", func(p *LoadProvenance) { p.ForwardPath = ForwardAttnSeqGQA }},
		{"transform ran on fewer layers", func(p *LoadProvenance) {
			p.Transforms = FoldTransformRecords(qwen36SSMAObservations([]int{0, 1}))
		}},
		{"transform vanished", func(p *LoadProvenance) { p.Transforms = nil }},
		{"a domain check rejected values", func(p *LoadProvenance) {
			p.DomainChecks[0].Rejected = 1
			p.DomainChecks[0].FirstFailure = "tensor blk.0.ssm_a element 7 violates ..."
		}},
		{"a transformed tensor hash moved", func(p *LoadProvenance) {
			p.TensorSummaries = []TensorSummary{SummarizeTransformedTensor(
				"model.layers.0.linear_attn.A_log", testSSMATransform,
				[]int{2, 2}, []float32{0.5, -0.25, 1.5, -2.5})}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mutated := base
			mutated.Transforms = append([]TransformRecord(nil), base.Transforms...)
			mutated.DomainChecks = append([]DomainCheck(nil), base.DomainChecks...)
			mutated.TensorSummaries = append([]TensorSummary(nil), base.TensorSummaries...)
			tc.mutot(&mutated)
			if mutated.Digest() == base.Digest() {
				t.Errorf("%s did not move the digest (%s); the artifact would publish under a stale address",
					tc.name, base.Digest())
			}
		})
	}
}

// TestSummarizeTransformedTensorDetectsPermutationAndNonFinite pins that the
// tensor hash is sensitive to a transform that preserves the value SET but
// permutes it — precisely what the value-head deinterleaves do, so a hash that
// missed it would be blind to the most likely loader regression.
func TestSummarizeTransformedTensorDetectsPermutationAndNonFinite(t *testing.T) {
	vals := []float32{-1, -2, -3, -4}
	permuted := []float32{-3, -4, -1, -2}

	a := SummarizeTransformedTensor("linear_attn.A_log", testSSMATransform, []int{2, 2}, vals)
	b := SummarizeTransformedTensor("linear_attn.A_log", testSSMATransform, []int{2, 2}, permuted)
	if a.Hash == b.Hash {
		t.Error("a permutation of the same values must change the hash (the deinterleave class)")
	}
	if a.Min != "-4" || a.Max != "-1" {
		t.Errorf("min/max = %q/%q, want -4/-1", a.Min, a.Max)
	}
	if !a.Finite || a.FirstNonFinite != -1 {
		t.Errorf("clean tensor reported non-finite: finite=%t first=%d", a.Finite, a.FirstNonFinite)
	}

	// A blown-up transform must be visible without reading the values.
	bad := SummarizeTransformedTensor("linear_attn.A_log", testSSMATransform, []int{4},
		[]float32{-1, float32(math.NaN()), -3, float32(math.Inf(1))})
	if bad.Finite {
		t.Error("a tensor containing NaN must report Finite=false")
	}
	if bad.FirstNonFinite != 1 {
		t.Errorf("FirstNonFinite = %d, want 1", bad.FirstNonFinite)
	}
	// The range must still describe the FINITE elements only. A NaN at index 0
	// must not leave the min/max seeded at an implicit zero: reporting max=0 for
	// a tensor whose every finite element is negative invents a value that is not
	// in the tensor, and the artifact's whole claim is that its numbers are facts.
	if bad.Min != "-3" || bad.Max != "-1" {
		t.Errorf("min/max over the finite elements = %q/%q, want -3/-1", bad.Min, bad.Max)
	}

	// A leading NaN in an all-POSITIVE tensor is the mirror case: the min must
	// not collapse to zero either.
	lead := SummarizeTransformedTensor("linear_attn.A_log", testSSMATransform, []int{3},
		[]float32{float32(math.NaN()), 1, 2})
	if lead.Min != "1" || lead.Max != "2" {
		t.Errorf("leading-NaN min/max = %q/%q, want 1/2", lead.Min, lead.Max)
	}

	// With NO finite element there is no range to report, so both must be empty
	// rather than a fabricated zero.
	none := SummarizeTransformedTensor("linear_attn.A_log", testSSMATransform, []int{2},
		[]float32{float32(math.NaN()), float32(math.Inf(-1))})
	if none.Min != "" || none.Max != "" {
		t.Errorf("all-non-finite min/max = %q/%q, want empty", none.Min, none.Max)
	}
	if none.Finite || none.FirstNonFinite != 0 {
		t.Errorf("all-non-finite summary = finite:%t first:%d, want false/0", none.Finite, none.FirstNonFinite)
	}

	// An empty tensor reports no range and stays finite: there is nothing to be
	// non-finite about, and an empty Min/Max is the honest reading.
	empty := SummarizeTransformedTensor("linear_attn.A_log", testSSMATransform, []int{0}, nil)
	if empty.Min != "" || empty.Max != "" || !empty.Finite || empty.FirstNonFinite != -1 || empty.Values != 0 {
		t.Errorf("empty tensor summary = %+v, want empty range, finite, first=-1, values=0", empty)
	}
}

// TestSourceDomainRefusalNamesAllFourFacts is acceptance box 4: an invalid
// source-domain value fails with tensor name, transform ID, index, and expected
// domain. The split between Error() and Evidence() is the privacy seam — the
// operator sees the offending value, the publishable artifact does not.
func TestSourceDomainRefusalNamesAllFourFacts(t *testing.T) {
	// Element 5 is non-negative: a value only plausible in the CANONICAL (A_log)
	// domain, i.e. a fixture or exporter that skipped the negation. This is the
	// #4273 shape exactly.
	vals := []float32{-1, -2, -3, -4, -5, 0.75, -7}
	err := CheckSourceDomain("blk.13.ssm_a", testSSMATransform, NegativeDecayDomain, vals, IsNegativeDecay)
	if err == nil {
		t.Fatal("a non-negative decay value must be refused")
	}
	var sde *SourceDomainError
	if !errors.As(err, &sde) {
		t.Fatalf("refusal must be a *SourceDomainError, got %T", err)
	}
	if sde.Tensor != "blk.13.ssm_a" {
		t.Errorf("tensor = %q, want blk.13.ssm_a", sde.Tensor)
	}
	if sde.Transform != testSSMATransform {
		t.Errorf("transform = %q, want %q", sde.Transform, testSSMATransform)
	}
	if sde.Index != 5 {
		t.Errorf("index = %d, want 5 (the FIRST violation)", sde.Index)
	}
	if sde.ExpectedDomain != NegativeDecayDomain {
		t.Errorf("expected domain = %q, want %q", sde.ExpectedDomain, NegativeDecayDomain)
	}

	// All four facts must be present in BOTH renderings; only the value differs.
	for _, rendering := range []struct {
		name string
		text string
	}{{"Error", sde.Error()}, {"Evidence", sde.Evidence()}} {
		for _, want := range []string{"blk.13.ssm_a", testSSMATransform, "5", NegativeDecayDomain} {
			if !strings.Contains(rendering.text, want) {
				t.Errorf("%s() = %q, missing required fact %q", rendering.name, rendering.text, want)
			}
		}
	}
	if !strings.Contains(sde.Error(), "0.75") {
		t.Errorf("Error() must show the operator the offending value, got %q", sde.Error())
	}
	if strings.Contains(sde.Evidence(), "0.75") {
		t.Errorf("Evidence() is publish-safe and must NOT carry the weight value, got %q", sde.Evidence())
	}

	// A clean tensor passes, and NaN is refused as non-finite.
	if err := CheckSourceDomain("blk.0.ssm_a", testSSMATransform, NegativeDecayDomain,
		[]float32{-1, -2, -3}, IsNegativeDecay); err != nil {
		t.Errorf("an in-domain tensor must pass, got %v", err)
	}
	if err := CheckSourceDomain("blk.0.ssm_a", testSSMATransform, NegativeDecayDomain,
		[]float32{-1, float32(math.NaN())}, IsNegativeDecay); err == nil {
		t.Error("a NaN decay value must be refused")
	}
}

// TestDiffLoadProvenanceRoutesEachDeltaToItsInvestigation is acceptance box 5:
// two runs compare into transform/path differences an operator can act on. The
// test asserts the ROUTING, not just that a diff is non-empty — the artifact's
// claim is that a delta arrives already triaged into loader vs quant vs forward.
func TestDiffLoadProvenanceRoutesEachDeltaToItsInvestigation(t *testing.T) {
	base := completeProvenance(t)

	if d := DiffLoadProvenance(base, base); len(d) != 0 {
		t.Errorf("identical artifacts must diff empty, got %v", d)
	}

	for _, tc := range []struct {
		name  string
		field string
		area  InvestigationArea
		mutot func(*LoadProvenance)
	}{
		{"different model file", "model_digest", AreaModelBytes,
			func(p *LoadProvenance) { p.ModelDigest = "sha256:" + strings.Repeat("ef", 32) }},
		{"different container version", "gguf_version", AreaModelBytes,
			func(p *LoadProvenance) { p.GGUFVersion = 2 }},
		{"different loader revision", "loader_rev", AreaLoader,
			func(p *LoadProvenance) { p.LoaderRev = "deadbeef" }},
		{"transform ran on fewer layers", "transforms." + testSSMATransform + "/ssm_a", AreaLoader,
			func(p *LoadProvenance) { p.Transforms = FoldTransformRecords(qwen36SSMAObservations([]int{0, 1})) }},
		{"different quantization", "quant", AreaQuant,
			func(p *LoadProvenance) { p.Quant = "Q8_0" }},
		{"different forward path", "forward_path", AreaForward,
			func(p *LoadProvenance) { p.ForwardPath = ForwardAttnSeqGQA }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			other := base
			other.Transforms = append([]TransformRecord(nil), base.Transforms...)
			other.DomainChecks = append([]DomainCheck(nil), base.DomainChecks...)
			other.TensorSummaries = append([]TensorSummary(nil), base.TensorSummaries...)
			tc.mutot(&other)

			deltas := DiffLoadProvenance(base, other)
			if len(deltas) == 0 {
				t.Fatalf("%s produced no delta", tc.name)
			}
			var found *ProvenanceDelta
			for i := range deltas {
				if deltas[i].Field == tc.field {
					found = &deltas[i]
					break
				}
			}
			if found == nil {
				t.Fatalf("no delta for field %q; got %v", tc.field, deltas)
			}
			if found.Area != tc.area {
				t.Errorf("%s routed to %q, want %q — an operator would investigate the wrong subsystem",
					tc.field, found.Area, tc.area)
			}
			if !strings.Contains(found.String(), string(tc.area)) {
				t.Errorf("delta line %q must name the investigation area", found.String())
			}
		})
	}

	// A transform that VANISHED is the most important difference there is: it
	// must surface as a delta against "", not silently drop out of the union.
	gone := base
	gone.Transforms = nil
	deltas := DiffLoadProvenance(base, gone)
	var sawVanished bool
	for _, d := range deltas {
		if strings.HasPrefix(d.Field, "transforms.") && d.B == "" && d.A != "" {
			sawVanished, _ = true, d
			if d.Area != AreaLoader {
				t.Errorf("a vanished transform must route to %q, got %q", AreaLoader, d.Area)
			}
		}
	}
	if !sawVanished {
		t.Errorf("a transform disappearing between runs must produce a delta, got %v", deltas)
	}

	// Bytes-level deltas are reported BEFORE loader/quant/forward ones, so the
	// first line an operator reads is the one that invalidates the rest.
	mixed := base
	mixed.ModelDigest = "sha256:" + strings.Repeat("ef", 32)
	mixed.Quant = "Q8_0"
	mixed.ForwardPath = ForwardAttnSeqGQA
	got := DiffLoadProvenance(base, mixed)
	if len(got) == 0 || got[0].Area != AreaModelBytes {
		t.Errorf("model-bytes deltas must be reported first, got %v", got)
	}
}

// TestProvenanceArtifactLeaksNoWeightsOrPaths is the privacy half of acceptance
// box 5: operators compare runs WITHOUT exposing private prompts or weights.
// The artifact is designed so no field can hold one; this pins that the encoded
// bytes actually behave that way.
func TestProvenanceArtifactLeaksNoWeightsOrPaths(t *testing.T) {
	// Interior values are distinctive and are neither the min nor the max, so
	// they have no legitimate reason to appear anywhere in the artifact.
	const interiorA, interiorB = -0.31415927, -0.27182817
	vals := []float32{-9.75, interiorA, interiorB, -0.015625}

	p := completeProvenance(t)
	p.TensorSummaries = []TensorSummary{
		SummarizeTransformedTensor("model.layers.0.linear_attn.A_log", testSSMATransform, []int{4}, vals),
	}

	encoded := string(p.JSON())
	for _, leak := range []string{
		formatF32(interiorA), formatF32(interiorB),
		"0.31415927", "0.27182817",
	} {
		if strings.Contains(encoded, leak) {
			t.Errorf("artifact leaked an interior weight value %q:\n%s", leak, encoded)
		}
	}

	// The summary must still be USEFUL: shape, count, finiteness and a hash.
	s := p.Normalize().TensorSummaries[0]
	if s.Values != len(vals) || len(s.Shape) != 1 || s.Shape[0] != 4 {
		t.Errorf("summary lost the shape/count: %+v", s)
	}
	if !strings.HasPrefix(s.Hash, "sha256:") {
		t.Errorf("summary hash = %q, want a sha256: one-way digest", s.Hash)
	}

	// The artifact must round-trip as JSON and carry its schema id, so an
	// archived digest stays interpretable.
	var back LoadProvenance
	if err := json.Unmarshal(p.JSON(), &back); err != nil {
		t.Fatalf("artifact must round-trip as JSON: %v", err)
	}
	if back.Schema != LoadProvenanceSchema {
		t.Errorf("schema = %q, want %q", back.Schema, LoadProvenanceSchema)
	}
	if back.Digest() != p.Digest() {
		t.Errorf("digest did not survive a JSON round-trip: %s -> %s", p.Digest(), back.Digest())
	}
}

// TestProvenanceValidateIsFailClosed pins that an artifact missing any fact the
// acceptance contract requires is refused, so an inconclusive provenance can
// never silently back a publication claim.
func TestProvenanceValidateIsFailClosed(t *testing.T) {
	for _, tc := range []struct {
		name  string
		mutot func(*LoadProvenance)
	}{
		{"no model digest", func(p *LoadProvenance) { p.ModelDigest = "" }},
		{"no model size", func(p *LoadProvenance) { p.ModelBytes = 0 }},
		{"no gguf arch", func(p *LoadProvenance) { p.GGUFArch = "" }},
		{"no gguf version", func(p *LoadProvenance) { p.GGUFVersion = 0 }},
		{"no manifest digest", func(p *LoadProvenance) { p.ManifestDigest = "" }},
		{"no loader revision", func(p *LoadProvenance) { p.LoaderRev = "" }},
		{"no quant", func(p *LoadProvenance) { p.Quant = "" }},
		{"no forward path", func(p *LoadProvenance) { p.ForwardPath = "" }},
		{"transform with no id", func(p *LoadProvenance) {
			p.Transforms = []TransformRecord{{External: "ssm_a", Canonical: "linear_attn.A_log"}}
		}},
		{"transform recorded on no tensors", func(p *LoadProvenance) {
			p.Transforms = []TransformRecord{{
				ID: testSSMATransform, External: "ssm_a", Canonical: "linear_attn.A_log",
				SourceDomain: NegativeDecayDomain, CanonicalDomain: "A_log", Tensors: 0,
			}}
		}},
		{"rejection with no evidence", func(p *LoadProvenance) { p.DomainChecks[0].Rejected = 3 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := completeProvenance(t)
			p.Transforms = append([]TransformRecord(nil), p.Transforms...)
			p.DomainChecks = append([]DomainCheck(nil), p.DomainChecks...)
			tc.mutot(&p)
			if err := p.Validate(); err == nil {
				t.Errorf("%s must be refused, Validate returned nil", tc.name)
			}
		})
	}
}
