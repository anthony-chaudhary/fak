package model

// loadprovenance.go — the model-load provenance artifact (#4746, root incident #4273).
//
// WHY THIS EXISTS: during #4273 the runtime could report the model id, the quant
// mode, and the tokens/sec, but NOT the semantic loader transforms that produced
// the in-memory tensors. Every one of those reportable facts was identical
// between the broken and the fixed load — the defect lived entirely in the
// loader's SEMANTICS (GGUF blk.*.ssm_a stores the already-transformed negative
// decay -exp(A_log); fak's canonical tensor is the pre-transform A_log, so the
// forward applied exp twice). With no provenance surface, an operator staring at
// degraded output had no way to separate a sampler bug from a prefill bug from a
// decode bug from a quantization bug from a LOADER-DOMAIN bug, and burned the
// investigation on the wrong four. More exporters and architectures will keep
// encoding transformed parameters, so the class recurs.
//
// WHAT THIS IS: the compact, privacy-safe record of what the loader DID to the
// bytes, content-addressed so a publication claim can be bound to the actual
// loader semantics it was produced under. It binds, in one artifact:
//
//   - the model file digest + size, and the GGUF architecture/version;
//   - the canonical (shape-first, #3251) manifest digest;
//   - the loader/canonicalizer revision;
//   - every NON-IDENTITY transform id with how many tensors and how many LAYERS
//     used it (#4744 declares those transforms as arch-keyed semantic contracts;
//     this records which ones actually fired on this load);
//   - the source-domain validation outcomes;
//   - selected transformed-tensor summaries (shape, finite/range check, hash);
//   - the quantization mode and the selected runtime forward path.
//
// THREE PROPERTIES ARE LOAD-BEARING, and each is enforced here rather than
// promised in prose:
//
//  1. DETERMINISM. Digest is a function of the model bytes and the loader
//     revision ONLY. There is deliberately no timestamp, no host, no device, no
//     file PATH, and no wall-clock cost field — those are run scope and belong in
//     internal/provenance.RunManifest, which can carry this Digest as one of its
//     recorded facts. Because the artifact holds no Go maps and every float is
//     stored PRE-FORMATTED as a string, its canonical JSON does not depend on map
//     iteration order or on a float formatter's choices: identical model bytes +
//     identical loader revision serialize byte-identically, on any host.
//
//  2. PRIVACY BY CONSTRUCTION. No field can hold a prompt, a filesystem path, or
//     raw weights, because no field has a type that could: there is no []float32,
//     no []byte, and no open map anywhere in the record. Tensor evidence is
//     reduced to a shape, a finite flag, a formatted min/max, and a one-way hash
//     before it ever enters the artifact (SummarizeTransformedTensor is the only
//     entry point that touches values, and it accepts weights and returns none).
//     That makes the artifact safe to attach to a public failure bundle or a
//     readiness claim unedited.
//
//  3. DIFFERENCES ROUTE THEMSELVES. Comparing two runs is not an exercise in
//     eyeballing JSON: DiffLoadProvenance labels every delta with the closed
//     InvestigationArea it implicates — model bytes, loader, quant, or forward —
//     so "these two runs differ" arrives already triaged into which subsystem to
//     investigate. That routing IS the troubleshooting guide, kept executable so
//     it cannot drift from the fields it explains.
//
// The package is the right home despite the transforms being declared in
// internal/ggufload: ForwardPathKind and ClassifyForwardPath (arch_support.go)
// live here, ggufload imports model (not the reverse), and the artifact is a
// flat evidence record rather than a live contract reference — so the loader
// can populate it from its arch-keyed contracts without model needing to
// import them.
//
// WIRING STATUS, stated plainly so nobody reads a capability into it that is
// not there yet — and, equally, so nobody re-builds a piece that already
// landed. This file is the artifact and its algebra — construction, folding,
// normalization, digest, validation, and diff routing — exercised end to end by
// loadprovenance_test.go.
//
// LIVE. A producer calls it: internal/ggufload's (*File).LoadProvenance builds
// the artifact from a parsed GGUF header, folding TransformObservations copied
// off the arch-keyed transform contracts (#4744), so a real load does yield a
// record. The ssm_a domain guard in normalizeCanonicalTensorData refuses through
// CheckSourceDomain on the live load path. internal/provenance.RunManifest
// carries the Digest as its load_provenance field, refuses anything that is not
// a sha256 content address, and orders it directly after model in the
// fingerprint — so two runs differing only in loader semantics are not
// Equivalent and Compare localizes the divergence to load_provenance.
//
// NOT WIRED. No command renders or diffs the artifact (fak info has no
// provenance mode), and nothing in production constructs a RunManifest, so the
// load_provenance field is enforced but unpopulated: the schema refuses a
// malformed digest, it cannot supply a missing one. DomainChecks and
// TensorSummaries stay empty until a weight-touching pass appends them, because
// a header-only producer cannot honestly synthesize value evidence.
//
// Operator-facing routing for all of the above: docs/model-load-provenance-troubleshooting.md.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// LoadProvenanceSchema is the versioned schema id every artifact carries. It is
// never edited in place: a field whose meaning changes gets a new /N, so an
// archived digest stays interpretable.
const LoadProvenanceSchema = "fak-model-load-provenance/1"

// NegativeDecayDomain is the expected source domain of a GGUF ssm_a tensor: the
// exporter's already-negated exponential decay coefficient. It is named here so
// the loader's refusal, the transform contract, and this artifact all quote ONE
// string instead of three drifting paraphrases.
const NegativeDecayDomain = "finite, strictly negative -exp(A_log)"

// InvestigationArea is the CLOSED set of subsystems a provenance difference can
// implicate. It is closed on purpose: the value of the artifact is that a delta
// arrives pre-triaged, and an open vocabulary would let a caller invent a
// fifth area that no runbook covers.
type InvestigationArea string

const (
	// AreaModelBytes — the two runs did not load the same artifact
	// (different file digest, size, GGUF container version, or canonical
	// manifest). Nothing downstream is comparable until this is reconciled, so
	// it is reported before any loader/quant/forward delta.
	AreaModelBytes InvestigationArea = "model-bytes"
	// AreaLoader — same bytes, different loader SEMANTICS: a transform
	// appeared, disappeared, changed id, changed tensor/layer count, or a
	// transformed-tensor hash moved. This is the #4273 class.
	AreaLoader InvestigationArea = "loader"
	// AreaQuant — same bytes and same loader semantics, different
	// quantization mode: suspect dequant/kernel numerics, not the loader.
	AreaQuant InvestigationArea = "quant"
	// AreaForward — same bytes, loader, and quant, different selected
	// forward path: suspect the arch classification and the token mixer.
	AreaForward InvestigationArea = "forward"
)

// TransformObservation is ONE tensor the loader mapped non-identically. The
// loader emits one per transformed tensor as it canonicalizes; FoldTransformRecords
// folds them into the artifact's per-transform counts. Fields mirror the
// ggufload semantic transform contract (#4744) so the loader can copy them
// across without interpretation.
type TransformObservation struct {
	// Tensor is the transformed tensor's name WITH its layer index, in either
	// external ("blk.17.ssm_a") or canonical ("model.layers.17.linear_attn.A_log")
	// form — the layer count is derived from it.
	Tensor string
	// Transform is the named transform id ("+"-joined if composite).
	Transform string
	// External / Canonical are the contract's layer-stripped tensor names.
	External  string
	Canonical string
	// SourceDomain / CanonicalDomain are the contract's semantic domains.
	SourceDomain    string
	CanonicalDomain string
	// Lossless / Invertible mirror the contract's value-preservation claims.
	Lossless   bool
	Invertible bool
	// DomainValidated reports that the loader VALIDATES the source domain and
	// refuses values only plausible in the canonical domain (the contract's
	// RejectsCanonicalDomain).
	DomainValidated bool
}

// TransformRecord is one non-identity transform folded across every tensor it
// touched on this load. Tensors and Layers are what answer the operator's first
// question — "did the ssm_a decay inversion actually run, and on how much of the
// model?" — which a bare list of declared contracts cannot.
type TransformRecord struct {
	ID              string `json:"id"`
	External        string `json:"external"`
	Canonical       string `json:"canonical"`
	SourceDomain    string `json:"source_domain"`
	CanonicalDomain string `json:"canonical_domain"`
	// Tensors counts DISTINCT tensor names the transform ran on, so a retried
	// or re-observed tensor cannot inflate the count and perturb the digest.
	Tensors int `json:"tensors"`
	// Layers counts distinct blk.<n> / model.layers.<n> indices among them.
	// Model-global tensors (output_norm.weight) contribute no layer.
	Layers          int  `json:"layers"`
	Lossless        bool `json:"lossless"`
	Invertible      bool `json:"invertible"`
	DomainValidated bool `json:"domain_validated"`
}

// TensorSummary is the privacy-safe fingerprint of one transformed tensor: it
// proves WHICH values came out of the transform without disclosing them. Min and
// Max are pre-formatted strings, not floats — the artifact's bytes must not
// depend on a JSON float formatter for the digest to be reproducible.
type TensorSummary struct {
	Canonical string `json:"canonical"`
	Transform string `json:"transform"`
	Shape     []int  `json:"shape"`
	Values    int    `json:"values"`
	// Finite is false when any element is NaN or ±Inf — the range check the
	// issue requires, and the cheapest signal that a transform blew up.
	Finite bool `json:"finite"`
	// FirstNonFinite is the index of the first NaN/Inf element, or -1 when the
	// tensor is clean.
	FirstNonFinite int `json:"first_non_finite"`
	// Min / Max are shortest-round-trip float32 renderings over the FINITE
	// elements. Both are empty when the tensor holds no finite element — an
	// empty tensor, or one that is entirely NaN/Inf — because there is no
	// bound to state and a zero would be a value the tensor does not contain.
	Min string `json:"min"`
	Max string `json:"max"`
	// Hash is a one-way sha256 over the name, shape, and every element's IEEE-754
	// bit pattern: it changes if the transform changes, and cannot be inverted
	// back to weights.
	Hash string `json:"hash"`
}

// DomainCheck is the outcome of one source-domain validation, folded across the
// tensors it guarded. Rejected == 0 on a clean load; a non-zero Rejected with a
// populated FirstFailure is the artifact form of the loader's refusal.
type DomainCheck struct {
	Transform      string `json:"transform"`
	ExpectedDomain string `json:"expected_domain"`
	Tensors        int    `json:"tensors"`
	Rejected       int    `json:"rejected"`
	// FirstFailure is SourceDomainError.Evidence() — tensor name, transform id,
	// index, and expected domain, but NOT the offending VALUE. The value is a
	// raw weight element: it belongs in the operator-facing error, never in an
	// artifact that may be published.
	FirstFailure string `json:"first_failure,omitempty"`
}

// LoadProvenance is the artifact. Field order is the canonical serialization
// order (Go's encoding/json emits struct fields in declaration order), so this
// declaration IS the wire format.
type LoadProvenance struct {
	Schema string `json:"schema"`

	// The artifact identity: what was loaded.
	ModelDigest string `json:"model_digest"` // "sha256:<hex>" over the model file bytes
	ModelBytes  int64  `json:"model_bytes"`
	GGUFArch    string `json:"gguf_arch"`    // header general.architecture, canonicalized
	GGUFVersion uint32 `json:"gguf_version"` // GGUF container version

	// The loader identity: what interpreted it.
	ManifestDigest string `json:"manifest_digest"` // canonical shape-first manifest digest (#3251)
	LoaderRev      string `json:"loader_rev"`      // loader/canonicalizer version or commit

	// The runtime selection the load implies.
	Quant       string          `json:"quant"`
	ForwardPath ForwardPathKind `json:"forward_path"`

	// The loader semantics themselves.
	Transforms      []TransformRecord `json:"transforms"`
	DomainChecks    []DomainCheck     `json:"domain_checks"`
	TensorSummaries []TensorSummary   `json:"tensor_summaries"`
}

// SourceDomainError is the typed refusal for a value that cannot be in its
// declared source domain. It carries all four facts the acceptance contract
// names — tensor NAME, transform ID, element INDEX, and the expected DOMAIN —
// plus the offending value for the operator, so a domain failure is immediately
// actionable instead of a generic "bad tensor".
type SourceDomainError struct {
	Tensor         string
	Transform      string
	Index          int
	Value          float32
	ExpectedDomain string
}

// Error renders the operator-facing message, including the offending value.
func (e *SourceDomainError) Error() string {
	return fmt.Sprintf("model: tensor %s element %d = %g violates transform %s source domain (want %s)",
		e.Tensor, e.Index, e.Value, e.Transform, e.ExpectedDomain)
}

// Evidence renders the publish-safe form: the same four required facts with the
// VALUE withheld. This is what a DomainCheck records, so an artifact attached to
// a public bundle names the failure without leaking a weight element.
func (e *SourceDomainError) Evidence() string {
	return fmt.Sprintf("tensor %s element %d violates transform %s source domain (want %s)",
		e.Tensor, e.Index, e.Transform, e.ExpectedDomain)
}

// CheckSourceDomain validates every element of a transform's SOURCE values
// against inDomain and returns a *SourceDomainError for the FIRST violation, or
// nil when the whole tensor is in domain. It exists so every loader domain guard
// produces the same four-fact refusal instead of hand-rolling a message: the
// #4273 class is exactly a tensor arriving in the wrong domain, and a refusal
// that omits the transform id leaves the operator guessing which mapping to
// audit.
//
// A nil inDomain means the transform DECLARES NO SOURCE DOMAIN — most mappings
// are pure reshapes with nothing to constrain — and every value passes. That is
// a deliberate pass, not a fail-open oversight: the artifact records whether a
// guard ran (TransformRecord.DomainValidated), so a transform that should have
// had one and did not is visible in the provenance rather than hidden here.
func CheckSourceDomain(tensor, transform, expectedDomain string, vals []float32, inDomain func(float32) bool) error {
	if inDomain == nil {
		return nil
	}
	for i, v := range vals {
		if inDomain(v) {
			continue
		}
		return &SourceDomainError{
			Tensor: tensor, Transform: transform,
			Index: i, Value: v, ExpectedDomain: expectedDomain,
		}
	}
	return nil
}

// IsNegativeDecay is the source-domain predicate for a GGUF ssm_a tensor: finite
// and strictly negative. A non-negative or NaN "decay" is a value that is only
// plausible in the CANONICAL (A_log) domain — the exact fixture/exporter mistake
// that produced #4273 — so rejecting it here is what makes that class loud.
func IsNegativeDecay(v float32) bool {
	return v < 0 && !math.IsInf(float64(v), -1)
}

// SummarizeTransformedTensor reduces one transformed tensor to a TensorSummary.
// It is the ONLY function in this file that sees weight values, and it returns
// none of them: values leave as a shape, a finite flag, a formatted min/max, and
// a one-way hash. The hash covers the name, the shape, and every element's
// IEEE-754 bit pattern, so it is sensitive to a transform change (including one
// that preserves the value SET but permutes it, which is precisely what the
// value-head deinterleaves do).
func SummarizeTransformedTensor(canonical, transform string, shape []int, vals []float32) TensorSummary {
	s := TensorSummary{
		Canonical:      strings.TrimSpace(canonical),
		Transform:      strings.TrimSpace(transform),
		Shape:          append([]int(nil), shape...),
		Values:         len(vals),
		Finite:         true,
		FirstNonFinite: -1,
	}
	h := sha256.New()
	h.Write([]byte(s.Canonical))
	var word [8]byte
	for _, d := range s.Shape {
		binary.LittleEndian.PutUint64(word[:], uint64(int64(d)))
		h.Write(word[:])
	}
	// The range covers the FINITE elements only, and is seeded by the first of
	// them rather than by index 0. Seeding at index 0 would leave min/max at an
	// implicit zero whenever the tensor opens with a NaN or an Inf, and the
	// artifact would then report a bound that is not an element of the tensor —
	// exactly the kind of invented number this record exists to rule out. When
	// nothing finite survives there is no range to state, so both stay empty.
	var min, max float32
	var haveRange bool
	for i, v := range vals {
		binary.LittleEndian.PutUint32(word[:4], math.Float32bits(v))
		h.Write(word[:4])
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			if s.Finite {
				s.Finite = false
				s.FirstNonFinite = i
			}
			continue
		}
		if !haveRange {
			min, max, haveRange = v, v, true
			continue
		}
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	if haveRange {
		s.Min = formatF32(min)
		s.Max = formatF32(max)
	}
	s.Hash = "sha256:" + hex.EncodeToString(h.Sum(nil))
	return s
}

// formatF32 renders a float32 with shortest-round-trip precision. Doing the
// formatting HERE, into a string field, is what keeps the artifact's canonical
// bytes independent of any JSON float encoder.
func formatF32(v float32) string {
	return strconv.FormatFloat(float64(v), 'g', -1, 32)
}

// FoldTransformRecords folds per-tensor observations into per-transform records,
// counting DISTINCT tensors and distinct layer indices. Grouping is by the
// (transform id, external, canonical) triple, because the same transform id
// legitimately applies to several different tensor mappings and the operator
// needs them apart. Duplicate observations of the same tensor are idempotent, so
// a loader that re-observes a tensor cannot perturb the digest.
func FoldTransformRecords(obs []TransformObservation) []TransformRecord {
	type key struct{ id, external, canonical string }
	order := make([]key, 0, len(obs))
	rec := make(map[key]*TransformRecord, len(obs))
	tensors := make(map[key]map[string]bool, len(obs))
	layers := make(map[key]map[int]bool, len(obs))
	for _, o := range obs {
		k := key{strings.TrimSpace(o.Transform), strings.TrimSpace(o.External), strings.TrimSpace(o.Canonical)}
		if k.id == "" {
			continue // an identity mapping is not provenance — it is the absence of it
		}
		r, seen := rec[k]
		if !seen {
			r = &TransformRecord{
				ID: k.id, External: k.external, Canonical: k.canonical,
				SourceDomain:    strings.TrimSpace(o.SourceDomain),
				CanonicalDomain: strings.TrimSpace(o.CanonicalDomain),
				Lossless:        o.Lossless,
				Invertible:      o.Invertible,
				DomainValidated: o.DomainValidated,
			}
			rec[k] = r
			tensors[k] = map[string]bool{}
			layers[k] = map[int]bool{}
			order = append(order, k)
		}
		name := strings.TrimSpace(o.Tensor)
		if name == "" || tensors[k][name] {
			continue
		}
		tensors[k][name] = true
		r.Tensors++
		if n, ok := provenanceLayerIndex(name); ok {
			layers[k][n] = true
		}
	}
	out := make([]TransformRecord, 0, len(order))
	for _, k := range order {
		r := *rec[k]
		r.Layers = len(layers[k])
		out = append(out, r)
	}
	sortTransformRecords(out)
	return out
}

// provenanceLayerIndex extracts the layer index from a tensor name in either the
// external GGUF ("blk.17.ssm_a") or canonical ("model.layers.17.linear_attn.A_log")
// form. Model-global tensors carry no index and report ok=false.
func provenanceLayerIndex(tensor string) (int, bool) {
	for _, prefix := range []string{"blk.", "model.layers."} {
		if !strings.HasPrefix(tensor, prefix) {
			continue
		}
		rest := tensor[len(prefix):]
		dot := strings.IndexByte(rest, '.')
		if dot <= 0 {
			continue
		}
		n, err := strconv.Atoi(rest[:dot])
		if err != nil || n < 0 {
			continue
		}
		return n, true
	}
	return 0, false
}

func sortTransformRecords(rs []TransformRecord) {
	sort.SliceStable(rs, func(i, j int) bool {
		if rs[i].ID != rs[j].ID {
			return rs[i].ID < rs[j].ID
		}
		if rs[i].External != rs[j].External {
			return rs[i].External < rs[j].External
		}
		return rs[i].Canonical < rs[j].Canonical
	})
}

// Normalize returns a canonicalized copy: the schema id is stamped, every string
// field is space-trimmed, and all three lists are sorted into a fixed order.
// Two loads that recorded the same facts in a different ORDER normalize to
// byte-identical artifacts and therefore share a Digest, while any genuine
// semantic change survives.
func (p LoadProvenance) Normalize() LoadProvenance {
	out := p
	out.Schema = LoadProvenanceSchema
	out.ModelDigest = strings.TrimSpace(p.ModelDigest)
	out.GGUFArch = strings.TrimSpace(p.GGUFArch)
	out.ManifestDigest = strings.TrimSpace(p.ManifestDigest)
	out.LoaderRev = strings.TrimSpace(p.LoaderRev)
	out.Quant = strings.TrimSpace(p.Quant)
	out.ForwardPath = ForwardPathKind(strings.TrimSpace(string(p.ForwardPath)))

	out.Transforms = append([]TransformRecord(nil), p.Transforms...)
	for i := range out.Transforms {
		r := &out.Transforms[i]
		r.ID = strings.TrimSpace(r.ID)
		r.External = strings.TrimSpace(r.External)
		r.Canonical = strings.TrimSpace(r.Canonical)
		r.SourceDomain = strings.TrimSpace(r.SourceDomain)
		r.CanonicalDomain = strings.TrimSpace(r.CanonicalDomain)
	}
	sortTransformRecords(out.Transforms)

	out.DomainChecks = append([]DomainCheck(nil), p.DomainChecks...)
	for i := range out.DomainChecks {
		c := &out.DomainChecks[i]
		c.Transform = strings.TrimSpace(c.Transform)
		c.ExpectedDomain = strings.TrimSpace(c.ExpectedDomain)
		c.FirstFailure = strings.TrimSpace(c.FirstFailure)
	}
	sort.SliceStable(out.DomainChecks, func(i, j int) bool {
		return out.DomainChecks[i].Transform < out.DomainChecks[j].Transform
	})

	out.TensorSummaries = append([]TensorSummary(nil), p.TensorSummaries...)
	for i := range out.TensorSummaries {
		s := &out.TensorSummaries[i]
		s.Canonical = strings.TrimSpace(s.Canonical)
		s.Transform = strings.TrimSpace(s.Transform)
		s.Hash = strings.TrimSpace(s.Hash)
	}
	sort.SliceStable(out.TensorSummaries, func(i, j int) bool {
		if out.TensorSummaries[i].Canonical != out.TensorSummaries[j].Canonical {
			return out.TensorSummaries[i].Canonical < out.TensorSummaries[j].Canonical
		}
		return out.TensorSummaries[i].Transform < out.TensorSummaries[j].Transform
	})
	return out
}

// canonicalBytes serializes the normalized artifact to deterministic JSON. The
// record holds no Go map and no float, so declaration order plus the sorts in
// Normalize fully determine the bytes.
func (p LoadProvenance) canonicalBytes() []byte {
	b, err := json.Marshal(p.Normalize())
	if err != nil {
		// Unreachable: every field is a string, an int, a bool, or a slice of
		// those. Returning a marker rather than panicking keeps a diagnostic
		// surface from taking down a load.
		return []byte(`{"schema":"` + LoadProvenanceSchema + `","error":"marshal"}`)
	}
	return b
}

// Digest is the content address of the normalized artifact: "sha256:<hex>" over
// canonicalBytes. It is the value run and readiness evidence records, so a
// publication claim is bound to the loader semantics that actually produced the
// weights. Identical model bytes + identical loader revision yield an identical
// digest on any host, because no host-, path-, or time-dependent fact is in
// scope.
func (p LoadProvenance) Digest() string {
	sum := sha256.Sum256(p.canonicalBytes())
	return "sha256:" + hex.EncodeToString(sum[:])
}

// JSON renders the normalized artifact as indented, deterministic JSON: the
// human-readable form, intended for an operator-facing command and for
// attaching to a run bundle once a producer exists (no command emits it yet —
// see WIRING STATUS at the top of this file). It is publish-safe by
// construction: the record cannot hold a prompt, a path, or raw weights.
func (p LoadProvenance) JSON() []byte {
	b, err := json.MarshalIndent(p.Normalize(), "", "  ")
	if err != nil {
		return p.canonicalBytes()
	}
	return b
}

// Validate enforces fail-closed evidence: an artifact missing any fact the
// acceptance contract requires can never back a publication claim. A returned
// error means the provenance is inconclusive, NOT that the load failed.
func (p LoadProvenance) Validate() error {
	n := p.Normalize()
	switch {
	case n.ModelDigest == "":
		return fmt.Errorf("load provenance: missing model_digest")
	case n.ModelBytes <= 0:
		return fmt.Errorf("load provenance: model_bytes must be positive, got %d", n.ModelBytes)
	case n.GGUFArch == "":
		return fmt.Errorf("load provenance: missing gguf_arch")
	case n.GGUFVersion == 0:
		return fmt.Errorf("load provenance: missing gguf_version")
	case n.ManifestDigest == "":
		return fmt.Errorf("load provenance: missing manifest_digest")
	case n.LoaderRev == "":
		return fmt.Errorf("load provenance: missing loader_rev")
	case n.Quant == "":
		return fmt.Errorf("load provenance: missing quant")
	case n.ForwardPath == "":
		return fmt.Errorf("load provenance: missing forward_path")
	}
	for _, r := range n.Transforms {
		switch {
		case r.ID == "":
			return fmt.Errorf("load provenance: transform record with no id")
		case r.External == "" || r.Canonical == "":
			return fmt.Errorf("load provenance: transform %s missing external/canonical names", r.ID)
		case r.SourceDomain == "" || r.CanonicalDomain == "":
			return fmt.Errorf("load provenance: transform %s missing source/canonical domain", r.ID)
		case r.Tensors <= 0:
			return fmt.Errorf("load provenance: transform %s recorded on %d tensors", r.ID, r.Tensors)
		}
	}
	for _, c := range n.DomainChecks {
		if c.Transform == "" {
			return fmt.Errorf("load provenance: domain check with no transform")
		}
		if c.ExpectedDomain == "" {
			return fmt.Errorf("load provenance: domain check %s missing expected domain", c.Transform)
		}
		if c.Rejected > 0 && c.FirstFailure == "" {
			return fmt.Errorf("load provenance: domain check %s rejected %d values with no evidence", c.Transform, c.Rejected)
		}
	}
	return nil
}

// TransformTensors reports how many tensors and how many layers the named
// transform ran on, and whether it ran at all. This is the direct answer to the
// #4273 question an operator asks first — "did the ssm_a decay inversion fire,
// and on how much of the model?" — without parsing the artifact. A composite
// transform matches on any of its "+"-joined components, so asking for
// "invert-neg-exp-decay" finds it inside
// "value-head-deinterleave+invert-neg-exp-decay".
//
// When several records match, tensors SUM but layers take the MAXIMUM: two
// mappings that share a component id generally run on the same layer stack, so
// summing the layer counts would report more layers than the model has.
func (p LoadProvenance) TransformTensors(id string) (tensors, layers int, ok bool) {
	id = strings.TrimSpace(id)
	if id == "" {
		return 0, 0, false
	}
	for _, r := range p.Normalize().Transforms {
		if !transformIDMatches(r.ID, id) {
			continue
		}
		tensors += r.Tensors
		if r.Layers > layers {
			layers = r.Layers
		}
		ok = true
	}
	return tensors, layers, ok
}

func transformIDMatches(recorded, want string) bool {
	if recorded == want {
		return true
	}
	for _, part := range strings.Split(recorded, "+") {
		if part == want {
			return true
		}
	}
	return false
}

// ProvenanceDelta is one difference between two load-provenance artifacts,
// already routed to the subsystem it implicates. Field is the canonical dotted
// path; A and B are the two readings (a side that lacks the entry reads "").
type ProvenanceDelta struct {
	Field string            `json:"field"`
	A     string            `json:"a"`
	B     string            `json:"b"`
	Area  InvestigationArea `json:"area"`
}

// String renders the delta as one operator-readable line: what differs, the two
// readings, and where to look.
func (d ProvenanceDelta) String() string {
	return fmt.Sprintf("%s: %q -> %q [investigate %s]", d.Field, d.A, d.B, d.Area)
}

// DiffLoadProvenance compares two runs' artifacts and returns every difference,
// each labelled with the InvestigationArea it implicates. An empty result means
// the two loads produced identical loader semantics, quantization, and forward
// path — the runs are comparable and any behavioral difference lies elsewhere
// (sampler, prompt, scheduling), which is itself the answer.
//
// THE ROUTING TABLE — this is the troubleshooting guide, executable so it cannot
// drift from the fields it explains:
//
//	model_digest / model_bytes / gguf_version / manifest_digest differ
//	    -> AreaModelBytes. The runs did not load the same artifact.
//	       Reconcile the model first; every downstream delta is uninterpretable
//	       until they agree.
//	gguf_arch / loader_rev / transforms[*] / domain_checks[*] /
//	tensor_summaries[*] differ
//	    -> AreaLoader. Same bytes, different SEMANTICS. A transform that
//	       appeared, vanished, changed id, or changed tensor/layer count is the
//	       #4273 class: audit the ggufload transform contract for that tensor
//	       before touching the forward or the quantizer. A moved tensor hash with
//	       unchanged transform ids means the transform's implementation changed.
//	quant differs
//	    -> AreaQuant. Loader semantics agree, so the mapping is right and
//	       the numerics are suspect: compare dequant kernels and per-tensor
//	       quant assignment, not the loader.
//	forward_path differs
//	    -> AreaForward. Bytes, loader, and quant agree but the runtime
//	       chose a different token mixer: audit the arch classification
//	       (ClassifyForwardPath) and the mixer, not the loader.
//
// Deltas are returned in that order — bytes, loader, quant, forward — so the
// first line an operator reads is the one that invalidates the rest.
func DiffLoadProvenance(a, b LoadProvenance) []ProvenanceDelta {
	na, nb := a.Normalize(), b.Normalize()
	var out []ProvenanceDelta
	add := func(field, x, y string, area InvestigationArea) {
		if x != y {
			out = append(out, ProvenanceDelta{Field: field, A: x, B: y, Area: area})
		}
	}

	// 1. Identity of the loaded artifact.
	add("model_digest", na.ModelDigest, nb.ModelDigest, AreaModelBytes)
	add("model_bytes", strconv.FormatInt(na.ModelBytes, 10), strconv.FormatInt(nb.ModelBytes, 10), AreaModelBytes)
	add("gguf_version", strconv.FormatUint(uint64(na.GGUFVersion), 10), strconv.FormatUint(uint64(nb.GGUFVersion), 10), AreaModelBytes)
	add("manifest_digest", na.ManifestDigest, nb.ManifestDigest, AreaModelBytes)

	// 2. Loader semantics.
	add("gguf_arch", na.GGUFArch, nb.GGUFArch, AreaLoader)
	add("loader_rev", na.LoaderRev, nb.LoaderRev, AreaLoader)
	for _, k := range unionKeys(transformKeys(na.Transforms), transformKeys(nb.Transforms)) {
		add("transforms."+k, transformReading(na.Transforms, k), transformReading(nb.Transforms, k), AreaLoader)
	}
	for _, k := range unionKeys(domainKeys(na.DomainChecks), domainKeys(nb.DomainChecks)) {
		add("domain_checks."+k, domainReading(na.DomainChecks, k), domainReading(nb.DomainChecks, k), AreaLoader)
	}
	for _, k := range unionKeys(summaryKeys(na.TensorSummaries), summaryKeys(nb.TensorSummaries)) {
		add("tensor_summaries."+k, summaryReading(na.TensorSummaries, k), summaryReading(nb.TensorSummaries, k), AreaLoader)
	}

	// 3. Quantization, then 4. forward-path selection.
	add("quant", na.Quant, nb.Quant, AreaQuant)
	add("forward_path", string(na.ForwardPath), string(nb.ForwardPath), AreaForward)
	return out
}

// unionKeys returns the sorted union of two key lists, so an entry present on
// only ONE side still produces a delta (against "") instead of vanishing — a
// transform that DISAPPEARED is the most important difference there is.
func unionKeys(a, b []string) []string {
	seen := make(map[string]bool, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, k := range append(append([]string(nil), a...), b...) {
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func transformKeys(rs []TransformRecord) []string {
	out := make([]string, 0, len(rs))
	for _, r := range rs {
		out = append(out, r.ID+"/"+r.External)
	}
	return out
}

func transformReading(rs []TransformRecord, key string) string {
	for _, r := range rs {
		if r.ID+"/"+r.External != key {
			continue
		}
		return fmt.Sprintf("canonical=%s tensors=%d layers=%d lossless=%t invertible=%t domain_validated=%t",
			r.Canonical, r.Tensors, r.Layers, r.Lossless, r.Invertible, r.DomainValidated)
	}
	return ""
}

func domainKeys(cs []DomainCheck) []string {
	out := make([]string, 0, len(cs))
	for _, c := range cs {
		out = append(out, c.Transform)
	}
	return out
}

func domainReading(cs []DomainCheck, key string) string {
	for _, c := range cs {
		if c.Transform != key {
			continue
		}
		return fmt.Sprintf("tensors=%d rejected=%d expected=%s", c.Tensors, c.Rejected, c.ExpectedDomain)
	}
	return ""
}

func summaryKeys(ss []TensorSummary) []string {
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		out = append(out, s.Canonical)
	}
	return out
}

func summaryReading(ss []TensorSummary, key string) string {
	for _, s := range ss {
		if s.Canonical != key {
			continue
		}
		return fmt.Sprintf("transform=%s shape=%v values=%d finite=%t min=%s max=%s hash=%s",
			s.Transform, s.Shape, s.Values, s.Finite, s.Min, s.Max, s.Hash)
	}
	return ""
}
