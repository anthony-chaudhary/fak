// Package residualquant defines the neutral capability contract for recurrent
// residual multi-precision quantization (RRQ). It describes and adjudicates
// pinned artifacts and recipes; it does not implement a model kernel.
package residualquant

import (
	"fmt"
	"slices"
	"strings"
)

const (
	ContractID     = "fak.residualquant/v1"
	MethodID       = "rrq"
	PaperID        = "arXiv:2608.04048v1"
	PaperURL       = "https://arxiv.org/abs/2608.04048v1"
	PaperPDFSHA256 = "08d39e39738289bcd2784364a713ed337dc4bca6467df8e81046dfb2cf6dcd45"
	RecipeID       = "rrq-paper-algorithm-1/v1"
	RuntimePinID   = "external-rrq-runtime"
)

type Verdict string

const (
	CaseSupported   Verdict = "supported"
	CaseUnsupported Verdict = "unsupported"
	CaseDelegate    Verdict = "delegate"
)

type Code string

const (
	CodeMetadata             Code = "RRQ_METADATA_OK"
	CodeExecution            Code = "RRQ_RUNTIME_HANDOFF"
	ReasonUnknownContract    Code = "RESIDUALQUANT_UNKNOWN_CONTRACT"
	ReasonUnknownMethod      Code = "RESIDUALQUANT_UNKNOWN_METHOD"
	ReasonInvalidTiers       Code = "RESIDUALQUANT_INVALID_TIERS"
	ReasonArtifactUnpinned   Code = "RESIDUALQUANT_ARTIFACT_UNPINNED"
	ReasonRecipeUnpinned     Code = "RESIDUALQUANT_RECIPE_UNPINNED"
	ReasonRuntimeUnpinned    Code = "RESIDUALQUANT_RUNTIME_UNPINNED"
	ReasonRuntimeUnavailable Code = "RESIDUALQUANT_RUNTIME_UNAVAILABLE"
)

type EvidenceKind string

const (
	Observed       EvidenceKind = "observed"
	SourceReported EvidenceKind = "source-reported"
	Modeled        EvidenceKind = "modeled"
)

type ArtifactPin struct {
	PaperID       string
	PaperURL      string
	PDFSHA256     string
	WeightsURI    string
	WeightsDigest string
}

type RecipePin struct {
	ID               string
	BaseBits         int
	ResidualBits     int
	ProgressiveTiers []int
	BaseMethod       string
	ResidualMethod   string
	CalibrationFree  bool
}

type RuntimePin struct {
	ID      string
	Version string
	Device  string
}

type Descriptor struct {
	Contract string
	Method   string
	Artifact ArtifactPin
	Recipe   RecipePin
	Runtime  RuntimePin
}

type Request struct {
	Descriptor
	Operation string
	TierBits  int
}

type Result struct {
	Verdict  Verdict
	Reason   Code
	Delegate string
	Detail   string
}

type Finding struct {
	ID       string
	Evidence EvidenceKind
	Claim    string
	Envelope string
	Source   string
}

type ResearchEvaluation struct {
	Name        string
	Disposition string
	Findings    []Finding
}

// PinnedPaperDescriptor identifies the reviewed paper and recipe. The empty
// weights and runtime pins are intentional: v1 promises code only upon
// publication and publishes neither a checkpoint digest nor executable ABI.
func PinnedPaperDescriptor() Descriptor {
	return Descriptor{
		Contract: ContractID,
		Method:   MethodID,
		Artifact: ArtifactPin{PaperID: PaperID, PaperURL: PaperURL, PDFSHA256: PaperPDFSHA256},
		Recipe: RecipePin{
			ID: RecipeID, BaseBits: 2, ResidualBits: 2,
			ProgressiveTiers: []int{2, 4, 6, 8}, BaseMethod: "PTQ-or-RTN",
			ResidualMethod: "RTN", CalibrationFree: true,
		},
		Runtime: RuntimePin{ID: RuntimePinID},
	}
}

// Adjudicate returns an explicit capability outcome. Metadata inspection is
// local; execution is delegated only after artifact, recipe, and runtime pins
// are complete. There is no implicit precision or runtime fallback.
//
// Invariant: residual quantization is fail-closed and bounded. Metadata
// inspection is permitted locally for pinned artifacts, while any execution
// delegation requires complete weights and runtime verification.
func Adjudicate(req Request) Result {
	if req.Contract != ContractID {
		return refuse(ReasonUnknownContract, "contract must equal "+ContractID)
	}
	if req.Method != MethodID {
		return refuse(ReasonUnknownMethod, "method must equal "+MethodID)
	}
	if !validRecipe(req.Recipe) || (req.TierBits != 0 && !slices.Contains(req.Recipe.ProgressiveTiers, req.TierBits)) {
		return refuse(ReasonInvalidTiers, "recipe must be 2-bit base plus 2-bit residual tiers [2,4,6,8]")
	}
	if !artifactPinned(req.Artifact) {
		return refuse(ReasonArtifactUnpinned, "paper PDF must be pinned by version and SHA-256")
	}
	if req.Recipe.ID != RecipeID {
		return refuse(ReasonRecipeUnpinned, "recipe must equal "+RecipeID)
	}

	switch req.Operation {
	case "inspect":
		return Result{Verdict: CaseSupported, Reason: CodeMetadata, Detail: "pinned RRQ metadata is supported; this is not a kernel claim"}
	case "execute":
		if strings.TrimSpace(req.Artifact.WeightsURI) == "" || strings.TrimSpace(req.Artifact.WeightsDigest) == "" {
			return refuse(ReasonArtifactUnpinned, "execution requires a weights URI and digest")
		}
		if strings.TrimSpace(req.Runtime.ID) == "" || strings.TrimSpace(req.Runtime.Version) == "" || strings.TrimSpace(req.Runtime.Device) == "" {
			return refuse(ReasonRuntimeUnpinned, "execution requires runtime ID, version, and device")
		}
		return Result{Verdict: CaseDelegate, Reason: CodeExecution, Delegate: req.Runtime.ID, Detail: "execute with the pinned external runtime; fak supplies no RRQ kernel"}
	default:
		return refuse(ReasonRuntimeUnavailable, fmt.Sprintf("operation %q is unsupported", req.Operation))
	}
}

func refuse(reason Code, detail string) Result {
	return Result{Verdict: CaseUnsupported, Reason: reason, Detail: detail}
}

func artifactPinned(a ArtifactPin) bool {
	return a.PaperID == PaperID && a.PaperURL == PaperURL && a.PDFSHA256 == PaperPDFSHA256
}

func validRecipe(r RecipePin) bool {
	return r.BaseBits == 2 && r.ResidualBits == 2 && slices.Equal(r.ProgressiveTiers, []int{2, 4, 6, 8}) &&
		r.BaseMethod == "PTQ-or-RTN" && r.ResidualMethod == "RTN" && r.CalibrationFree
}

// NamedResearchEvaluation preserves the issue's research decision and keeps
// source measurements separate from local observations and modeled mappings.
func NamedResearchEvaluation() ResearchEvaluation {
	return ResearchEvaluation{
		Name:        "Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs",
		Disposition: "abstain",
		Findings: []Finding{
			{ID: "paper-pin", Evidence: Observed, Claim: "arXiv v1 PDF SHA-256 verified", Envelope: "downloaded public PDF", Source: PaperURL},
			{ID: "tier-map", Evidence: Modeled, Claim: "2/4/6/8-bit prefixes can be represented as artifact tiers", Envelope: "metadata and cache-key modeling only; no executable artifact", Source: RecipeID},
			{ID: "construction-time", Evidence: SourceReported, Claim: "all-RTN Qwen3-8B package construction: 1,293 seconds and 3.3x versus measured MatGPTQ", Envelope: "paper Table 5; authors report 8xA100 80GB for quantization experiments", Source: PaperID},
			{ID: "runtime", Evidence: Observed, Claim: "no released code, checkpoint digest, or runtime ABI is pinned", Envelope: "public arXiv v1 materials reviewed 2026-08-10", Source: PaperID},
		},
	}
}
