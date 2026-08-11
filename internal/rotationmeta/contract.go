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
)

// Provenance pins the public source used to interpret a recipe. Digest is the
// SHA-256 of the retrieved PDF, not a quality or performance attestation.
type Provenance struct {
	URI     string `json:"uri"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
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

// ValidateOutcome permits callers to reject corrupt serialized decisions.
func ValidateOutcome(o Outcome) error {
	switch o {
	case OutcomeSupported, OutcomeUnsupported, OutcomeDelegate:
		return nil
	}
	return errors.New("rotationmeta: unknown outcome " + string(o))
}
