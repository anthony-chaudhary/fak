package model

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// TorchAORecipeID names a public torchao quantization recipe.
type TorchAORecipeID string

const (
	TorchAOInt8WeightOnly                  TorchAORecipeID = "int8-weight-only"
	TorchAOInt8DynamicActivationInt8Weight TorchAORecipeID = "int8-dynamic-activation-int8-weight"
	TorchAOFloat8DynamicActivationWeight   TorchAORecipeID = "float8-dynamic-activation-float8-weight"
)

// TorchAODecision is the typed outcome of capability discovery.
type TorchAODecision string

const (
	TorchAODelegate TorchAODecision = "delegate"
	TorchAOAbstain  TorchAODecision = "abstain"
	TorchAORefuse   TorchAODecision = "refuse"
)

// TorchAOReasonCode explains a discovery outcome without requiring prose parsing.
type TorchAOReasonCode string

const (
	TorchAORuntimeDelegation  TorchAOReasonCode = "runtime_delegation_required"
	TorchAOUnknownVersion     TorchAOReasonCode = "unknown_version"
	TorchAOUnsupportedVersion TorchAOReasonCode = "unsupported_version"
	TorchAOUnsupportedRecipe  TorchAOReasonCode = "unsupported_recipe"
	TorchAOInvalidCapability  TorchAOReasonCode = "invalid_capability"
)

// TorchAOCapability is the versioned request accepted by DiscoverTorchAOCapability.
type TorchAOCapability struct {
	Version  string          `json:"version"`
	Recipe   TorchAORecipeID `json:"recipe"`
	Artifact string          `json:"artifact,omitempty"`
}

// TorchAOCapabilityResult keeps recipe discovery separate from artifact identity,
// runtime execution, and any measured hardware claim.
type TorchAOCapabilityResult struct {
	Decision         TorchAODecision   `json:"decision"`
	Reason           TorchAOReasonCode `json:"reason"`
	Version          string            `json:"version,omitempty"`
	Recipe           TorchAORecipeID   `json:"recipe,omitempty"`
	Artifact         string            `json:"artifact,omitempty"`
	Runtime          string            `json:"runtime,omitempty"`
	MeasuredEnvelope string            `json:"measured_envelope,omitempty"`
}

// DiscoverTorchAOCapability parses and adjudicates a torchao capability request.
// Supported recipes still delegate execution to torchao; discovery does not claim
// an artifact conversion or a measured hardware envelope.
func DiscoverTorchAOCapability(data []byte) TorchAOCapabilityResult {
	var capability TorchAOCapability
	if err := json.Unmarshal(data, &capability); err != nil {
		return TorchAOCapabilityResult{Decision: TorchAORefuse, Reason: TorchAOInvalidCapability}
	}

	result := TorchAOCapabilityResult{
		Version:  capability.Version,
		Recipe:   capability.Recipe,
		Artifact: capability.Artifact,
	}
	major, minor, ok := torchAOVersion(capability.Version)
	if !ok || major != 0 {
		result.Decision, result.Reason = TorchAOAbstain, TorchAOUnknownVersion
		return result
	}
	if minor < 9 {
		result.Decision, result.Reason = TorchAOAbstain, TorchAOUnsupportedVersion
		return result
	}
	if minor > 18 {
		result.Decision, result.Reason = TorchAOAbstain, TorchAOUnknownVersion
		return result
	}

	switch capability.Recipe {
	case TorchAOInt8WeightOnly,
		TorchAOInt8DynamicActivationInt8Weight,
		TorchAOFloat8DynamicActivationWeight:
		result.Decision = TorchAODelegate
		result.Reason = TorchAORuntimeDelegation
		result.Runtime = "torchao@" + capability.Version
		return result
	default:
		result.Decision, result.Reason = TorchAORefuse, TorchAOUnsupportedRecipe
		return result
	}
}

func torchAOVersion(version string) (major, minor int, ok bool) {
	trimmed := strings.TrimPrefix(strings.TrimSpace(version), "v")
	parts := strings.Split(trimmed, ".")
	if len(parts) < 2 || len(parts) > 3 {
		return 0, 0, false
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, false
	}
	minor, err = strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, false
	}
	if len(parts) == 3 {
		if _, err := fmt.Sscanf(parts[2], "%d", new(int)); err != nil {
			return 0, 0, false
		}
	}
	return major, minor, true
}
