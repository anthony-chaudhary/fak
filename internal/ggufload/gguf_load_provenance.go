package ggufload

// gguf_load_provenance.go — the GGUF producer for the model-load provenance
// artifact (#4746, root incident #4273).
//
// internal/model/loadprovenance.go defines the artifact and its algebra, but
// deliberately owns no loader knowledge: model cannot see ggufload's semantic
// transform contracts (#4744) because ggufload imports model and not the
// reverse. This file is the adapter that closes that gap — it reads a PARSED
// GGUF HEADER and emits the artifact's loader-semantics half, so a real load
// finally yields a provenance record instead of the artifact sitting
// available-but-unconsumed.
//
// HEADER-ONLY, ON PURPOSE. Every fact here comes from the tensor directory and
// the metadata KV block: which tensors exist, their shapes and dtypes, and
// which of them the canonicalizer maps non-identically. Read/Open stop at the
// header, so a caller holding only a *File has by construction read no weights,
// and this producer inherits that property: it answers "did the ssm_a decay
// inversion fire, and on how many layers?" for a multi-hundred-GB checkpoint at
// the cost of a header parse, touching no weight byte and recording no
// filesystem path.
//
// WHAT IT DOES NOT FILL, and why that is a refusal rather than an oversight.
// DomainChecks and TensorSummaries are VALUE evidence: a domain check reports
// that the loader ran its guard over real elements and how many it rejected; a
// tensor summary is a finite/range check and hash over transformed values.
// Neither can be honestly synthesized from a header, and emitting
// DomainCheck{Rejected: 0} here would assert a validation that never ran — the
// precise species of unwitnessed claim this artifact exists to kill. They stay
// empty until the weight-touching pass appends them: the ssm_a guard in
// normalizeCanonicalTensorData produces the *model.SourceDomainError a
// DomainCheck records, and model.SummarizeTransformedTensor produces the
// summaries.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// LoadProvenanceScope carries the facts a GGUF header cannot supply. They are
// separated from the header-derived facts because they come from a different
// authority: the header knows what the checkpoint IS, the scope knows how this
// build chose to consume it.
//
// Every field is required — LoadProvenance validates the assembled artifact and
// refuses an incomplete one, so a caller cannot accidentally publish a record
// with a blank loader revision and have it read as "no transform ran".
type LoadProvenanceScope struct {
	// ModelDigest is "sha256:<hex>" over the model file bytes. The caller
	// computes it because only the caller knows whether the logical model is
	// one file or a split shard set; the header cannot see its own container.
	ModelDigest string
	// ModelBytes is that same artifact's size in bytes.
	ModelBytes int64
	// LoaderRev is the loader/canonicalizer version or commit. With ModelDigest
	// it is the whole determinism contract: the artifact digest is a function
	// of these two and nothing else.
	LoaderRev string
	// Quant is the quantization mode the run selected.
	Quant string
	// ForwardPath is the in-kernel forward path the run selected. The caller
	// passes it (rather than this file re-deriving it) because the engine has
	// already resolved it through model.ClassifyForwardPath against the full
	// config; re-classifying here would be a second opinion that could silently
	// disagree with the path actually executing.
	ForwardPath model.ForwardPathKind
}

// TransformObservations returns one observation per tensor in the header that
// the canonicalizer maps NON-IDENTICALLY, copied straight off the declared
// contract (#4744) so the artifact and the contract cannot drift into two
// different accounts of the same transform. Identity mappings are omitted: an
// identity mapping is not provenance, it is the absence of it.
//
// Header-only, like TensorTransformIDs, and for the same reason — the mapping
// is keyed on the tensor NAME and general.architecture, never on a payload.
func (f *File) TransformObservations() []model.TransformObservation {
	arch, _ := f.String("general.architecture")
	contracts := TensorTransformContractsForArch(arch)
	if len(contracts) == 0 {
		return nil
	}
	byExternal := make(map[string]TensorTransformContract, len(contracts))
	for _, c := range contracts {
		byExternal[c.External] = c
	}
	out := make([]model.TransformObservation, 0, len(f.Tensors))
	for _, t := range f.Tensors {
		c, ok := byExternal[ExternalTensorSuffix(t.Name)]
		if !ok {
			continue
		}
		out = append(out, model.TransformObservation{
			Tensor:          t.Name,
			Transform:       c.Transform,
			External:        c.External,
			Canonical:       c.Canonical,
			SourceDomain:    c.SourceDomain,
			CanonicalDomain: c.CanonicalDomain,
			Lossless:        c.Lossless,
			Invertible:      c.Invertible,
			// The contract's RejectsCanonicalDomain is exactly the claim
			// "the loader validates this transform's source domain", which is
			// what the artifact's DomainValidated means.
			DomainValidated: c.RejectsCanonicalDomain,
		})
	}
	return out
}

// CanonicalManifestDigest is the shape-first manifest digest (#3251) over the
// tensor directory: every tensor's name, dtype, and dimensions, and nothing
// else. It changes when the checkpoint's STRUCTURE changes and is stable across
// everything else, which is what makes it the right companion to ModelDigest —
// two files with the same manifest digest but different model digests differ in
// weights alone.
//
// Rows are sorted before hashing so a shard set that enumerates its tensors in a
// different order still digests identically: directory order is a container
// detail, not a semantic one.
func (f *File) CanonicalManifestDigest() string {
	rows := make([]string, 0, len(f.Tensors))
	for _, t := range f.Tensors {
		dims := make([]string, len(t.Dims))
		for i, d := range t.Dims {
			dims[i] = strconv.FormatUint(d, 10)
		}
		rows = append(rows, t.Name+"\t"+t.Type.String()+"\t"+strings.Join(dims, ","))
	}
	sort.Strings(rows)
	h := sha256.New()
	for _, r := range rows {
		h.Write([]byte(r))
		h.Write([]byte{'\n'})
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// LoadProvenance assembles the artifact for this checkpoint under scope, folding
// every non-identity transform the header implies into per-transform tensor and
// layer counts.
//
// It returns the artifact only if it VALIDATES. Fail-closed is the point: an
// artifact missing a required fact is inconclusive evidence, and inconclusive
// evidence that still gets attached to a readiness claim is worse than none —
// it looks like provenance while proving nothing.
func (f *File) LoadProvenance(scope LoadProvenanceScope) (model.LoadProvenance, error) {
	arch, _ := f.String("general.architecture")
	p := model.LoadProvenance{
		ModelDigest:    scope.ModelDigest,
		ModelBytes:     scope.ModelBytes,
		GGUFArch:       canonicalGGUFArch(arch),
		GGUFVersion:    f.Version,
		ManifestDigest: f.CanonicalManifestDigest(),
		LoaderRev:      scope.LoaderRev,
		Quant:          scope.Quant,
		ForwardPath:    scope.ForwardPath,
		Transforms:     model.FoldTransformRecords(f.TransformObservations()),
	}
	p = p.Normalize()
	if err := p.Validate(); err != nil {
		return model.LoadProvenance{}, fmt.Errorf("gguf: load provenance: %w", err)
	}
	return p, nil
}
