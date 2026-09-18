package rotationmeta

import (
	"errors"
	"fmt"
	"strings"
)

// ContractVersion is the additive descriptor contract understood by this leaf.
const ContractVersion = "rotationmeta/v1"

// Outcome is the typed adjudication result for a descriptor.
type Outcome string

const (
	OutcomeSupported   Outcome = "supported"
	OutcomeUnsupported Outcome = "unsupported"
	OutcomeDelegate    Outcome = "delegate"
)

// Reason is a stable machine-readable explanation for an outcome.
type Reason string

const (
	ReasonSupported                   Reason = "supported"
	ReasonUnknownContract             Reason = "unknown_contract_version"
	ReasonUnknownRecipe               Reason = "unknown_recipe"
	ReasonUnknownRecipeVersion        Reason = "unknown_recipe_version"
	ReasonMissingProvenance           Reason = "missing_provenance"
	ReasonMissingTransform            Reason = "missing_transform_metadata"
	ReasonInvalidPlacement            Reason = "invalid_placement"
	ReasonRuntimeRequired             Reason = "runtime_delegation_required"
	ReasonRuntimeTransformUnavailable Reason = "runtime_transform_unavailable"
	// ReasonInvalidSignVector: an explicit-sign recipe declared sign metadata that is
	// internally inconsistent (widths do not sum to the vector length, an element is
	// not -1/+1, or the vector is empty). Fail-closed: never silently drop signs.
	ReasonInvalidSignVector Reason = "invalid_sign_vector"
)

// Placement states when the transform is applied.
type Placement string

const (
	PlacementOffline Placement = "offline"
	PlacementOnline  Placement = "online"
)

// Recipe identifies a public rotation-transform family without selecting a winner.
type Recipe string

const (
	RecipeQuaRot    Recipe = "quarot"
	RecipeSpinQuant Recipe = "spinquant"
	RecipeLightRot  Recipe = "lightrot"
	// RecipePrismHadamard is the prism-ml Ternary-Bonsai-2-27B GGUF rotation: a
	// blockwise (1024) normalized Sylvester Walsh-Hadamard transform folded into the
	// stored ternary weights, with an explicit per-row-width +/-1 sign vector applied
	// to the runtime activation's last dimension. It is ONLINE placement: the runtime
	// must apply the matching transform or refuse the file.
	RecipePrismHadamard Recipe = "prism-hadamard-g128"
)

// Provenance pins the public source used to interpret a recipe. Digest is the
// SHA-256 of the retrieved PDF, not a quality or performance attestation.
type Provenance struct {
	URI     string `json:"uri"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
}

// SignVector is the explicit-sign payload of an explicit-sign recipe such as
// prism-hadamard-g128. Widths names the row-width class of each per-width sign
// block; Values is the concatenation of those blocks. A descriptor carries it so a
// runtime either applies the exact declared signs or refuses, never guessing.
type SignVector struct {
	Widths []int `json:"widths"`
	Values []int `json:"values"`
}

// Transform describes one rotation that must be preserved by an artifact or
// executed by a runtime.
type Transform struct {
	Name      string    `json:"name"`
	Placement Placement `json:"placement"`
	Fusion    string    `json:"fusion,omitempty"`
}

// Descriptor is neutral metadata carried alongside an artifact. ArtifactFormat
// deliberately remains caller-defined: rotationmeta is not an artifact format.
type Descriptor struct {
	ContractVersion string      `json:"contract_version"`
	Recipe          Recipe      `json:"recipe"`
	RecipeVersion   string      `json:"recipe_version"`
	Provenance      Provenance  `json:"provenance"`
	ArtifactFormat  string      `json:"artifact_format"`
	Transforms      []Transform `json:"transforms"`
	// Sign is the explicit per-width +/-1 sign payload. It is required for
	// explicit-sign recipes (prism-hadamard-g128) and ignored for the rest.
	Sign *SignVector `json:"sign,omitempty"`
}

// Capabilities are the runtime facts used for adjudication.
type Capabilities struct {
	Recipes map[Recipe][]string
	Fusions map[string]bool
}

// Decision never silently falls back. Delegate means the descriptor is valid,
// but one or more online transforms must be performed by a capable runtime.
type Decision struct {
	Outcome Outcome `json:"outcome"`
	Reason  Reason  `json:"reason"`
	Detail  string  `json:"detail,omitempty"`
}

var pinned = map[Recipe]map[string]Provenance{
	RecipeQuaRot:    {"arxiv:2404.00456v2": {URI: "https://arxiv.org/abs/2404.00456v2", Version: "arxiv:2404.00456v2", SHA256: "f611888c63ef63a5c0232e2c8416619f0c9ace08d0e05692731da82791202e3e"}},
	RecipeSpinQuant: {"arxiv:2405.16406v4": {URI: "https://arxiv.org/abs/2405.16406v4", Version: "arxiv:2405.16406v4", SHA256: "fe437770d7c981eae9e028eacaa5c772ed0add900ed9dec54c10cfce6dfd86c3"}},
	RecipeLightRot:  {"arxiv:2607.27704v1": {URI: "https://arxiv.org/abs/2607.27704v1", Version: "arxiv:2607.27704v1", SHA256: "e9e6093c0b0025e0fa40b575c416d8e40cb287d97d434373d6878ec6f3762696"}},
	// The prism-hadamard-g128 provenance is the model card's own declared wire
	// contract, pinned to the artifact revision we parsed (GGUF v3 header); SHA256
	// is the digest of the model card README at that revision.
	RecipePrismHadamard: {"prism-bonsai2-gguf-v1": {URI: "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf", Version: "prism-bonsai2-gguf-v1", SHA256: "6ed5e12bf84b7a63069882c91dd9e9218647d17b"}},
}

// PinnedProvenance returns a copy of the reviewed public-source pin.
func PinnedProvenance(recipe Recipe, version string) (Provenance, bool) {
	versions, ok := pinned[recipe]
	if !ok {
		return Provenance{}, false
	}
	p, ok := versions[version]
	return p, ok
}

// Validate checks descriptor completeness and pinned provenance.
func Validate(d Descriptor) Decision {
	if d.ContractVersion != ContractVersion {
		return Decision{OutcomeDelegate, ReasonUnknownContract, d.ContractVersion}
	}
	versions, known := pinned[d.Recipe]
	if !known {
		return Decision{OutcomeDelegate, ReasonUnknownRecipe, string(d.Recipe)}
	}
	expected, known := versions[d.RecipeVersion]
	if !known {
		return Decision{OutcomeDelegate, ReasonUnknownRecipeVersion, d.RecipeVersion}
	}
	if d.Provenance != expected {
		return Decision{OutcomeUnsupported, ReasonMissingProvenance, "provenance must match the reviewed URI, version, and SHA-256"}
	}
	if strings.TrimSpace(d.ArtifactFormat) == "" || len(d.Transforms) == 0 {
		return Decision{OutcomeUnsupported, ReasonMissingTransform, "artifact_format and transforms are required"}
	}
	for i, tr := range d.Transforms {
		if strings.TrimSpace(tr.Name) == "" {
			return Decision{OutcomeUnsupported, ReasonMissingTransform, fmt.Sprintf("transforms[%d].name", i)}
		}
		if tr.Placement != PlacementOffline && tr.Placement != PlacementOnline {
			return Decision{OutcomeUnsupported, ReasonInvalidPlacement, fmt.Sprintf("transforms[%d].placement", i)}
		}
		if tr.Placement == PlacementOnline && strings.TrimSpace(tr.Fusion) == "" {
			return Decision{OutcomeUnsupported, ReasonMissingTransform, fmt.Sprintf("transforms[%d].fusion", i)}
		}
	}
	// Explicit-sign recipes must carry a self-consistent sign payload; a descriptor
	// that declares one but omits or corrupts the signs is refused, never guessed.
	if recipeNeedsSigns(d.Recipe) {
		if err := validateSignVector(d.Sign); err != "" {
			return Decision{OutcomeUnsupported, ReasonInvalidSignVector, err}
		}
	}
	return Decision{OutcomeSupported, ReasonSupported, "metadata is complete"}
}

// Adjudicate combines descriptor validity with explicit runtime support.
func Adjudicate(d Descriptor, c Capabilities) Decision {
	if decision := Validate(d); decision.Outcome != OutcomeSupported {
		return decision
	}
	if !contains(c.Recipes[d.Recipe], d.RecipeVersion) {
		return Decision{OutcomeDelegate, ReasonRuntimeRequired, "runtime has not declared recipe support"}
	}
	for _, tr := range d.Transforms {
		if tr.Placement == PlacementOnline && !c.Fusions[tr.Fusion] {
			return Decision{OutcomeUnsupported, ReasonRuntimeTransformUnavailable, tr.Fusion}
		}
	}
	return Decision{OutcomeSupported, ReasonSupported, "artifact and declared runtime capabilities are compatible"}
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

// recipeNeedsSigns reports whether a recipe's explicit-sign payload is mandatory.
func recipeNeedsSigns(r Recipe) bool { return r == RecipePrismHadamard }

// validateSignVector returns a non-empty diagnostic string when the payload is
// missing or inconsistent, and "" when it is usable. The widths must sum to the
// value count (one sign per weight of each row-width class) and every sign must be
// exactly -1 or +1; anything else is a corrupt declaration, not a fallback case.
func validateSignVector(s *SignVector) string {
	if s == nil {
		return "explicit-sign recipe requires a sign payload"
	}
	if len(s.Widths) == 0 {
		return "sign.widths is empty"
	}
	total := 0
	for i, w := range s.Widths {
		if w <= 0 {
			return fmt.Sprintf("sign.widths[%d] must be positive, got %d", i, w)
		}
		total += w
	}
	if len(s.Values) != total {
		return fmt.Sprintf("sign.values length %d does not match the %d widths total", len(s.Values), total)
	}
	for i, v := range s.Values {
		if v != 1 && v != -1 {
			return fmt.Sprintf("sign.values[%d] must be -1 or +1, got %d", i, v)
		}
	}
	return ""
}

// ValidateOutcome permits callers to reject corrupt serialized decisions.
func ValidateOutcome(o Outcome) error {
	switch o {
	case OutcomeSupported, OutcomeUnsupported, OutcomeDelegate:
		return nil
	}
	return errors.New("rotationmeta: unknown outcome " + string(o))
}
