// Package bitnetmeta describes BitNet-family model artifacts without conflating
// weight semantics with their storage, conversion recipe, runtime, or benchmark.
package bitnetmeta

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
)

// SchemaV1 is the canonical schema identifier for BitNet metadata v1.
const SchemaV1 = "bitnetmeta/v1"

// Outcome represents the high-level adjudication verdict.
type Outcome string

const (
	// OutcomeAccept indicates the descriptor is valid and supported.
	OutcomeAccept Outcome = "accept"
	// OutcomeAbstain indicates the descriptor cannot be evaluated by this environment.
	OutcomeAbstain Outcome = "abstain"
	// OutcomeRefuse indicates the descriptor is invalid or unsupported.
	OutcomeRefuse Outcome = "refuse"
	// OutcomeDelegate indicates evaluation must be delegated to an external runtime.
	OutcomeDelegate Outcome = "delegate"
)

// ReasonCode represents structured machine-readable adjudication rationale.
type ReasonCode string

const (
	// ReasonSupported indicates all artifact dimensions match capabilities.
	ReasonSupported ReasonCode = "supported"
	// ReasonMalformed indicates JSON parsing or structure failure.
	ReasonMalformed ReasonCode = "malformed_metadata"
	// ReasonUnknownSchema indicates an unhandled metadata schema.
	ReasonUnknownSchema ReasonCode = "unknown_schema"
	// ReasonUnknownWeightSemantic indicates an unrecognized weight semantic.
	ReasonUnknownWeightSemantic ReasonCode = "unknown_weight_semantic"
	// ReasonUnknownArtifactFormat indicates an unrecognized artifact storage format.
	ReasonUnknownArtifactFormat ReasonCode = "unknown_artifact_format"
	// ReasonInconsistentArtifact indicates conflicting metadata declarations.
	ReasonInconsistentArtifact ReasonCode = "inconsistent_artifact"
	// ReasonUnsupportedRecipe indicates an unsupported conversion recipe.
	ReasonUnsupportedRecipe ReasonCode = "unsupported_recipe"
	// ReasonUnsupportedActivation indicates an unsupported activation format or precision.
	ReasonUnsupportedActivation ReasonCode = "unsupported_activation"
	// ReasonUnsupportedPacking indicates an unsupported weight packing scheme.
	ReasonUnsupportedPacking ReasonCode = "unsupported_packing"
	// ReasonRuntimeDelegation indicates an external runtime is required.
	ReasonRuntimeDelegation ReasonCode = "runtime_delegation_required"
	// ReasonHardwareUnverified indicates unverified hardware measurement witness.
	ReasonHardwareUnverified ReasonCode = "hardware_envelope_unverified"
)

// WeightSemantic specifies the mathematical weight quantization semantic.
type WeightSemantic string

const (
	// WeightNativeBinary represents native 1-bit {-1, 1} binary weights.
	WeightNativeBinary WeightSemantic = "native-binary"
	// WeightNativeTernary represents native 1.58-bit {-1, 0, 1} ternary weights.
	WeightNativeTernary WeightSemantic = "native-ternary-1.58bit"
	// WeightPostTernary represents post-training ternary converted weights.
	WeightPostTernary WeightSemantic = "post-training-ternary"
	// WeightInteger2Bit represents 2-bit integer weights.
	WeightInteger2Bit WeightSemantic = "integer-2bit"
)

// ArtifactOrigin specifies whether weights were natively trained or converted.
type ArtifactOrigin string

const (
	// OriginNativeTrained indicates weights were trained natively with low bits.
	OriginNativeTrained ArtifactOrigin = "native-trained"
	// OriginPostTraining indicates weights were converted post-training.
	OriginPostTraining ArtifactOrigin = "post-training-converted"
)

// RecipeKind specifies the training or conversion recipe category.
type RecipeKind string

const (
	// RecipeNativeTraining indicates native low-bit training recipe.
	RecipeNativeTraining RecipeKind = "native-training"
	// RecipeTernarization indicates post-training ternarization.
	RecipeTernarization RecipeKind = "post-training-ternarization"
	// RecipeQuantization indicates post-training quantization.
	RecipeQuantization RecipeKind = "post-training-quantization"
)

// Artifact describes the storage container and origin of the model weights.
type Artifact struct {
	ID      string         `json:"id"`
	Format  string         `json:"format"`
	Version string         `json:"version"`
	Origin  ArtifactOrigin `json:"origin"`
}

// Weights describes the discrete levels and quantization semantics of the weights.
type Weights struct {
	Semantic WeightSemantic `json:"semantic"`
	Label    string         `json:"label"`
	Levels   []int          `json:"levels"`
}

// ActivationPrecision describes the activation numeric format and bit width.
type ActivationPrecision struct {
	Format string `json:"format"`
	Bits   int    `json:"bits"`
}

// Packing describes the bit packing scheme and layout in memory/storage.
type Packing struct {
	Scheme        string `json:"scheme"`
	StorageBits   int    `json:"storage_bits"`
	ValuesPerUnit int    `json:"values_per_unit"`
}

// Recipe identifies the training or conversion procedure used.
type Recipe struct {
	ID      string     `json:"id"`
	Version string     `json:"version"`
	Kind    RecipeKind `json:"kind"`
}

// Runtime identifies an external runtime requirement or delegate.
type Runtime struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// HardwareEnvelope records only the scope of a measured claim. Empty or
// unmeasured envelopes make no performance claim.
type HardwareEnvelope struct {
	Measured     bool    `json:"measured"`
	Vendor       string  `json:"vendor,omitempty"`
	Architecture string  `json:"architecture,omitempty"`
	Metric       string  `json:"metric,omitempty"`
	Value        float64 `json:"value,omitempty"`
	Unit         string  `json:"unit,omitempty"`
	Witness      string  `json:"witness,omitempty"`
}

// Descriptor is the complete specification of a BitNet model artifact.
type Descriptor struct {
	Schema     string              `json:"schema"`
	Artifact   Artifact            `json:"artifact"`
	Weights    Weights             `json:"weights"`
	Activation ActivationPrecision `json:"activation"`
	Packing    Packing             `json:"packing"`
	Recipe     Recipe              `json:"recipe"`
	Runtime    *Runtime            `json:"runtime,omitempty"`
	Hardware   HardwareEnvelope    `json:"hardware"`
}

// Capabilities defines the supported schemas, formats, activations, packings, and environments.
type Capabilities struct {
	Schemas     []string `json:"schemas"`
	Formats     []string `json:"formats"`     // format@version
	Activations []string `json:"activations"` // format/bits
	Packings    []string `json:"packings"`
	Recipes     []string `json:"recipes"`  // id@version
	Runtimes    []string `json:"runtimes"` // id@version
	Hardware    []string `json:"hardware"` // vendor/architecture
}

// Result captures the outcome, machine reason code, and details of an adjudication.
type Result struct {
	Outcome    Outcome     `json:"outcome"`
	Reason     ReasonCode  `json:"reason"`
	Detail     string      `json:"detail"`
	Descriptor *Descriptor `json:"descriptor,omitempty"`
}

// ParseAndAdjudicate is the end-to-end metadata boundary. It never guesses a
// schema, weight semantic, recipe, or runtime from a nearby known value.
func ParseAndAdjudicate(raw []byte, capabilities Capabilities) Result {
	var descriptor Descriptor
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&descriptor); err != nil {
		return result(OutcomeRefuse, ReasonMalformed, err.Error(), nil)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = errors.New("multiple JSON values are not allowed")
		}
		return result(OutcomeRefuse, ReasonMalformed, err.Error(), nil)
	}
	adjudication := Adjudicate(descriptor, capabilities)
	adjudication.Descriptor = &descriptor
	return adjudication
}

// Adjudicate evaluates a parsed Descriptor against declared Capabilities.
func Adjudicate(descriptor Descriptor, capabilities Capabilities) Result {
	if descriptor.Schema == "" || !slices.Contains(capabilities.Schemas, descriptor.Schema) {
		return result(OutcomeAbstain, ReasonUnknownSchema, fmt.Sprintf("schema %q is not recognized", descriptor.Schema), nil)
	}
	if !knownSemantic(descriptor.Weights.Semantic) {
		return result(OutcomeAbstain, ReasonUnknownWeightSemantic, fmt.Sprintf("weight semantic %q is not recognized", descriptor.Weights.Semantic), nil)
	}
	if !slices.Contains(capabilities.Formats, versioned(descriptor.Artifact.Format, descriptor.Artifact.Version)) {
		return result(OutcomeAbstain, ReasonUnknownArtifactFormat, fmt.Sprintf("artifact format %q is not recognized", versioned(descriptor.Artifact.Format, descriptor.Artifact.Version)), nil)
	}
	if err := validate(descriptor); err != nil {
		return result(OutcomeRefuse, ReasonInconsistentArtifact, err.Error(), nil)
	}
	activation := fmt.Sprintf("%s/%d", descriptor.Activation.Format, descriptor.Activation.Bits)
	if !slices.Contains(capabilities.Activations, activation) {
		return result(OutcomeRefuse, ReasonUnsupportedActivation, fmt.Sprintf("activation precision %q is unsupported", activation), nil)
	}
	if !slices.Contains(capabilities.Packings, descriptor.Packing.Scheme) {
		return result(OutcomeRefuse, ReasonUnsupportedPacking, fmt.Sprintf("packing scheme %q is unsupported", descriptor.Packing.Scheme), nil)
	}
	if !slices.Contains(capabilities.Recipes, versioned(descriptor.Recipe.ID, descriptor.Recipe.Version)) {
		return result(OutcomeRefuse, ReasonUnsupportedRecipe, fmt.Sprintf("recipe %q is unsupported", versioned(descriptor.Recipe.ID, descriptor.Recipe.Version)), nil)
	}
	if descriptor.Runtime != nil && !slices.Contains(capabilities.Runtimes, versioned(descriptor.Runtime.ID, descriptor.Runtime.Version)) {
		return result(OutcomeDelegate, ReasonRuntimeDelegation, fmt.Sprintf("runtime %q must be delegated", versioned(descriptor.Runtime.ID, descriptor.Runtime.Version)), nil)
	}
	if descriptor.Hardware.Measured {
		hardware := descriptor.Hardware.Vendor + "/" + descriptor.Hardware.Architecture
		if descriptor.Hardware.Witness == "" || !slices.Contains(capabilities.Hardware, hardware) {
			return result(OutcomeAbstain, ReasonHardwareUnverified, fmt.Sprintf("measured hardware envelope %q is not witnessed here", hardware), nil)
		}
	}
	return result(OutcomeAccept, ReasonSupported, "artifact metadata and declared capability envelope are supported", nil)
}

// MarshalCanonical serializes a Descriptor into canonical indented JSON.
func MarshalCanonical(descriptor Descriptor) ([]byte, error) {
	return json.MarshalIndent(descriptor, "", "  ")
}

// SemanticID returns the composite identifier of weight semantic and artifact origin.
func SemanticID(descriptor Descriptor) string {
	return string(descriptor.Weights.Semantic) + "/" + string(descriptor.Artifact.Origin)
}

func validate(descriptor Descriptor) error {
	if descriptor.Artifact.ID == "" || descriptor.Artifact.Format == "" || descriptor.Artifact.Version == "" {
		return errors.New("artifact id, format, and version are required")
	}
	if descriptor.Recipe.ID == "" || descriptor.Recipe.Version == "" {
		return errors.New("recipe id and version are required")
	}
	if descriptor.Activation.Bits <= 0 || descriptor.Activation.Format == "" {
		return errors.New("activation format and positive precision are required")
	}
	if descriptor.Packing.Scheme == "" || descriptor.Packing.StorageBits <= 0 || descriptor.Packing.ValuesPerUnit <= 0 {
		return errors.New("packing scheme, storage bits, and values per unit are required")
	}
	if descriptor.Runtime != nil && (descriptor.Runtime.ID == "" || descriptor.Runtime.Version == "") {
		return errors.New("runtime id and version are required when runtime is declared")
	}
	if descriptor.Hardware.Measured && (descriptor.Hardware.Vendor == "" || descriptor.Hardware.Architecture == "" || descriptor.Hardware.Metric == "" || descriptor.Hardware.Unit == "" || descriptor.Hardware.Witness == "") {
		return errors.New("measured hardware requires vendor, architecture, metric, unit, and witness")
	}

	wantOrigin := OriginNativeTrained
	wantRecipe := RecipeNativeTraining
	var wantLabel string
	var wantLevels []int
	switch descriptor.Weights.Semantic {
	case WeightNativeBinary:
		wantLabel, wantLevels = "1-bit", []int{-1, 1}
	case WeightNativeTernary:
		wantLabel, wantLevels = "1.58-bit", []int{-1, 0, 1}
	case WeightPostTernary:
		wantOrigin, wantRecipe = OriginPostTraining, RecipeTernarization
		wantLabel, wantLevels = "ternary", []int{-1, 0, 1}
	case WeightInteger2Bit:
		wantLabel, wantLevels = "2-bit", []int{-2, -1, 0, 1}
		if descriptor.Recipe.Kind == RecipeQuantization {
			wantOrigin = OriginPostTraining
			wantRecipe = RecipeQuantization
		}
	}
	if descriptor.Artifact.Origin != wantOrigin {
		return fmt.Errorf("%s requires artifact origin %q", descriptor.Weights.Semantic, wantOrigin)
	}
	if descriptor.Recipe.Kind != wantRecipe {
		return fmt.Errorf("%s requires recipe kind %q", descriptor.Weights.Semantic, wantRecipe)
	}
	if descriptor.Weights.Label != wantLabel || !slices.Equal(descriptor.Weights.Levels, wantLevels) {
		return fmt.Errorf("%s requires label %q and levels %v", descriptor.Weights.Semantic, wantLabel, wantLevels)
	}
	return nil
}

func knownSemantic(semantic WeightSemantic) bool {
	switch semantic {
	case WeightNativeBinary, WeightNativeTernary, WeightPostTernary, WeightInteger2Bit:
		return true
	default:
		return false
	}
}

func versioned(id, version string) string { return id + "@" + version }

func result(outcome Outcome, reason ReasonCode, detail string, descriptor *Descriptor) Result {
	return Result{Outcome: outcome, Reason: reason, Detail: detail, Descriptor: descriptor}
}
