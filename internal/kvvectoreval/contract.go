// Package kvvectoreval defines the pinned, evidence-graded interoperability
// contract for the NOVA-KV attention-preserving vector-quantization research
// evaluation tracked by issue #6259.
//
// Contract: guarantees exact-match evaluation against pinned research artifacts
// and evidence without fuzzy aliases or silent fallback.
package kvvectoreval

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

// Public contract identifiers. Exact matching is intentional: an unknown
// version must not inherit support from a nearby paper, recipe, or runtime.
const (
	ContractID        = "fak.kvvectoreval/v1"
	PaperID           = "arxiv:2608.04074v1"
	PaperTitle        = "Spend Bits Where Queries Look: KV Cache Vector Quantization with Attention-Preserving Transforms"
	PaperPDFSHA256    = "7cd51970952e7fd72fd36db00e194dade32cae7c410d45982eb21c148721ab77"
	PaperSourceSHA256 = "72edf6938775c532e63d6703164bbebc8ca7d722aea683cc24c7674601ca17d4"
	RecipeID          = "github.com/Amir-zsh/nova-kv@d81c77b007d7a8e50ed608134fcb0feba0269ef8"
	RuntimeID         = "sglang@v0.5.10+nova-kv.d81c77b"
	ManifestSHA256    = "41691496d9628cb2825bbe1fa87470b2159e931fe869ce74d9f786eb733edd98"
)

// Outcome is the typed disposition of an evaluation request.
type Outcome string

// Typed disposition outcomes for contract requests.
const (
	OutcomeSupported Outcome = "supported"
	OutcomeRefused   Outcome = "unsupported"
	OutcomeDelegate  Outcome = "delegate"
)

// Reason is a stable machine-readable explanation for an outcome.
type Reason string

// Machine-readable reason tokens explaining evaluation outcomes.
const (
	ReasonExactMatch       Reason = "KV_VECTOR_EVAL_EXACT_MATCH"
	ReasonMalformed        Reason = "KV_VECTOR_EVAL_MALFORMED"
	ReasonUnknownContract  Reason = "KV_VECTOR_EVAL_UNKNOWN_CONTRACT"
	ReasonUnknownPaper     Reason = "KV_VECTOR_EVAL_UNKNOWN_PAPER"
	ReasonArtifactMismatch Reason = "KV_VECTOR_EVAL_ARTIFACT_MISMATCH"
	ReasonUnknownRecipe    Reason = "KV_VECTOR_EVAL_UNKNOWN_RECIPE"
	ReasonUnknownRuntime   Reason = "KV_VECTOR_EVAL_UNKNOWN_RUNTIME"
	ReasonExternalRuntime  Reason = "KV_VECTOR_EVAL_EXTERNAL_RUNTIME"
)

// EvidenceKind prevents paper-reported or analytically derived values from
// being silently promoted to locally observed measurements.
type EvidenceKind string

// EvidenceKind classification levels for metric provenance.
const (
	EvidenceObserved EvidenceKind = "observed"
	EvidenceModeled  EvidenceKind = "modeled"
)

// Artifact identifies immutable public research inputs.
type Artifact struct {
	ID     string
	URI    string
	SHA256 string
}

// Metric is one quality, storage, or performance datum with explicit
// provenance. Value is text because units differ and invented conversions are
// worse than preserving the source's exact value.
type Metric struct {
	Name       string
	Value      string
	Kind       EvidenceKind
	Provenance string
	Envelope   string
}

// Request asks whether a precise paper/artifact/recipe/runtime tuple is within
// this leaf's contract. RuntimeAvailable reports availability only; this leaf
// never starts or impersonates the external SGLang/CUDA runtime.
type Request struct {
	ContractID        string
	PaperID           string
	PaperPDFSHA256    string
	PaperSourceSHA256 string
	RecipeID          string
	RuntimeID         string
	RuntimeAvailable  bool
}

// Result is complete enough for a caller to preserve the selected contract,
// external delegation boundary, and evidence ledger.
type Result struct {
	Outcome   Outcome
	Reason    Reason
	Delegate  string
	Artifacts []Artifact
	Recipe    string
	Runtime   string
	Metrics   []Metric
}

// Evaluate returns a typed result without fuzzy aliases or silent fallback.
func Evaluate(req Request) Result {
	base := Result{
		Artifacts: pinnedArtifacts(),
		Recipe:    RecipeID,
		Runtime:   RuntimeID,
		Metrics:   researchLedger(),
	}
	if strings.TrimSpace(req.ContractID) == "" || strings.TrimSpace(req.PaperID) == "" ||
		strings.TrimSpace(req.PaperPDFSHA256) == "" || strings.TrimSpace(req.PaperSourceSHA256) == "" ||
		strings.TrimSpace(req.RecipeID) == "" || strings.TrimSpace(req.RuntimeID) == "" {
		base.Outcome, base.Reason = OutcomeRefused, ReasonMalformed
		return base
	}
	if req.ContractID != ContractID {
		base.Outcome, base.Reason = OutcomeRefused, ReasonUnknownContract
		return base
	}
	if req.PaperID != PaperID {
		base.Outcome, base.Reason = OutcomeRefused, ReasonUnknownPaper
		return base
	}
	if !equalDigest(req.PaperPDFSHA256, PaperPDFSHA256) || !equalDigest(req.PaperSourceSHA256, PaperSourceSHA256) {
		base.Outcome, base.Reason = OutcomeRefused, ReasonArtifactMismatch
		return base
	}
	if req.RecipeID != RecipeID {
		base.Outcome, base.Reason = OutcomeRefused, ReasonUnknownRecipe
		return base
	}
	if req.RuntimeID != RuntimeID {
		base.Outcome, base.Reason = OutcomeRefused, ReasonUnknownRuntime
		return base
	}
	if !req.RuntimeAvailable {
		base.Outcome, base.Reason, base.Delegate = OutcomeDelegate, ReasonExternalRuntime, RuntimeID
		return base
	}
	base.Outcome, base.Reason = OutcomeSupported, ReasonExactMatch
	return base
}

// VerifyArtifact verifies bytes against a pinned artifact and rejects unknown
// artifact IDs rather than accepting a caller-provided digest as authority.
func VerifyArtifact(id string, data []byte) error {
	for _, artifact := range pinnedArtifacts() {
		if artifact.ID != id {
			continue
		}
		sum := sha256.Sum256(data)
		if !equalDigest(hex.EncodeToString(sum[:]), artifact.SHA256) {
			return fmt.Errorf("%s: digest mismatch", ReasonArtifactMismatch)
		}
		return nil
	}
	return fmt.Errorf("%s: unknown artifact %q", ReasonArtifactMismatch, id)
}

func equalDigest(a, b string) bool {
	return strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b))
}

func pinnedArtifacts() []Artifact {
	return []Artifact{
		{ID: PaperID + "/pdf", URI: "https://arxiv.org/pdf/2608.04074v1", SHA256: PaperPDFSHA256},
		{ID: PaperID + "/source", URI: "https://export.arxiv.org/e-print/2608.04074v1", SHA256: PaperSourceSHA256},
		{ID: RecipeID + "/artifacts/MANIFEST.json", URI: "https://raw.githubusercontent.com/Amir-zsh/nova-kv/d81c77b007d7a8e50ed608134fcb0feba0269ef8/artifacts/MANIFEST.json", SHA256: ManifestSHA256},
	}
}

func researchLedger() []Metric {
	const paper = PaperID + " paper source"
	return []Metric{
		{Name: "qwen3-8b.niah.aggregate.nova-kv-2bit", Value: "0.947", Kind: EvidenceObserved, Provenance: paper + " Table 1", Envelope: "Qwen3-8B; 4K/8K/16K/32K NIAH; mean of four cells"},
		{Name: "qwen3-8b.niah.aggregate.oscar-2bit", Value: "0.524", Kind: EvidenceObserved, Provenance: paper + " Table 1", Envelope: "tuned scalar baseline; Qwen3-8B; 4K/8K/16K/32K NIAH; mean of four cells"},
		{Name: "gpt-oss-20b.quality.aggregate.nova-kv-2bit", Value: "0.474", Kind: EvidenceObserved, Provenance: paper + " Table 2", Envelope: "AIME25/GPQA/MATH500/NIAH normalized aggregate; 2-bit setting"},
		{Name: "gpt-oss-20b.quality.aggregate.turboquant-int2", Value: "0.481", Kind: EvidenceObserved, Provenance: paper + " Table 2", Envelope: "tuned scalar baseline; AIME25/GPQA/MATH500/NIAH normalized aggregate"},
		{Name: "gpt-oss-20b.quality.aggregate.oscar-int2", Value: "0.231", Kind: EvidenceObserved, Provenance: paper + " Table 2", Envelope: "scalar baseline; AIME25/GPQA/MATH500/NIAH normalized aggregate"},
		{Name: "qwen3-8b.decode.throughput-vs-bf16", Value: "1.6x-3.1x", Kind: EvidenceObserved, Provenance: paper + " Section 4.2/Figure 5", Envelope: "8x NVIDIA H100 80GB SXM; SGLang; 16K/32K/64K context; batch 1-64"},
		{Name: "metadata.amortized-rate", Value: "<0.01 bits/KV element", Kind: EvidenceModeled, Provenance: paper + " Appendix C", Envelope: "32-layer Qwen3-8B; 128K context; shared transforms/codebooks; served batch sizes"},
		{Name: "cache.nominal-rate", Value: "~2.2 bits/KV element", Kind: EvidenceModeled, Provenance: paper + " Appendix C", Envelope: "2-bit transformed-key VQ plus affine INT2 values and metadata"},
		{Name: "gpt-oss-20b.shipped-codebook", Value: "9,936,810 bytes", Kind: EvidenceObserved, Provenance: RecipeID + " artifacts/MANIFEST.json", Envelope: "repository-shipped codebook.pt; SHA-256 b65796b3a2628d57b038c5ad70fbd27dc1c7628b6b44fbdf90f4836b6144fdb9"},
	}
}
