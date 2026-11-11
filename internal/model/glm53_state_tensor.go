package model

import (
	"errors"
	"fmt"
	"strings"
)

// -----------------------------------------------------------------------------
// GLM-5.3 extra state tensor (arch-registration contract)
// -----------------------------------------------------------------------------
//
// The GLM-5.3 (glm5-next) family interleaves KDA linear-attention layers with
// DSA sparse-attention layers. Every KDA layer carries two extra per-layer
// state tensors that pre-5.3 GLM-5.x architectures do not register at all:
// the recurrent latent state S[h] (one [headDim x headDim] matrix per head)
// and the short-convolution history holding the q, k and v token windows one
// after another. Upstream llama.cpp only admitted the glm5-next arch to its
// per-tensor ("sm") state-allocation mode once the meta backend registered
// exactly those state tensors for per-layer allocation and graph construction;
// an arch whose descriptor does not register them cannot be allocated under
// that mode.
//
// This component ports that arch-registration contract into fak's model side:
// an arch descriptor declares which extra state tensors the architecture
// registers, per-layer state allocation threads the declared tensors through
// the KDA layer cadence, and the loader refuses a checkpoint/descriptor pair
// whose declarations disagree - in both directions - with sentinel errors.
// It is a declaration and allocation contract only: no kernel, no forward
// wiring, no hardware claim.
//
// Provenance (ADAPT): the contract is adapted from ggml-org/llama.cpp PR
// #30499 ("meta: support GLM-5.3 next sm tensor"), commit
// 259d91709d60945911bedb9bcfb2957c6e153ea7, src/llama-arch.cpp +
// src/llama-model.cpp - see THIRD_PARTY_NOTICES.md. No upstream code is
// copied; the registration facts are re-expressed as Go types with
// fail-closed validation.

// Glm53StateTensorClass identifies one extra per-layer state tensor family
// carried by the GLM-5.x family beyond the ordinary KV cache.
type Glm53StateTensorClass string

const (
	// Glm53StateKDALatent is the KDA recurrent latent state: one
	// [headDim x headDim] matrix per head, flattened to
	// numHeads * headDim * headDim float32 elements per KDA layer.
	Glm53StateKDALatent Glm53StateTensorClass = "kda_latent_state"
	// Glm53StateKDAConv is the KDA short-convolution history: the past
	// (window - 1) tokens of the q, k and v streams held one after another,
	// 3 * (window - 1) * numHeads * headDim float32 elements per KDA layer.
	Glm53StateKDAConv Glm53StateTensorClass = "kda_conv_state"
)

// allGlm53StateTensorClasses lists every extra state tensor class in stable
// order; it is the closed vocabulary the loader check compares against.
var allGlm53StateTensorClasses = []Glm53StateTensorClass{Glm53StateKDALatent, Glm53StateKDAConv}

// Glm53ArchStateDescriptor is the arch-registration half of the contract:
// which extra state tensors an architecture registers for per-layer state
// allocation. Pre-5.3 GLM-5.x descriptors register none; the GLM-5.3
// (glm5-next) family registers the two KDA state tensors on every KDA layer.
type Glm53ArchStateDescriptor struct {
	// Arch names the architecture family, e.g. "glm5-next".
	Arch string
	// StateTensors lists every extra state tensor class the arch registers.
	StateTensors []Glm53StateTensorClass
}

// Declares reports whether the descriptor registers the given state tensor
// class.
func (d Glm53ArchStateDescriptor) Declares(c Glm53StateTensorClass) bool {
	for _, s := range d.StateTensors {
		if s == c {
			return true
		}
	}
	return false
}

// Glm53LegacyStateDescriptor is the descriptor for pre-5.3 GLM-5.x
// architectures: they carry only the ordinary KV cache and register no extra
// state tensor.
var Glm53LegacyStateDescriptor = Glm53ArchStateDescriptor{Arch: "glm5"}

// Glm53NextStateDescriptor is the descriptor for the GLM-5.3 (glm5-next)
// family: it registers both KDA state tensors.
var Glm53NextStateDescriptor = Glm53ArchStateDescriptor{
	Arch:         "glm5-next",
	StateTensors: []Glm53StateTensorClass{Glm53StateKDALatent, Glm53StateKDAConv},
}

// ErrGlm53CheckpointUndeclaredState is the sentinel returned when a
// checkpoint declares an extra state tensor the arch descriptor does not
// register. Callers match it with errors.Is; the wrapped error names the
// offending class.
var ErrGlm53CheckpointUndeclaredState = errors.New("model: glm53 state tensor: checkpoint declares a state tensor the arch descriptor does not register")

// ErrGlm53DescriptorMissingState is the sentinel returned when the arch
// descriptor requires an extra state tensor the checkpoint does not declare.
// Callers match it with errors.Is; the wrapped error names the offending class.
var ErrGlm53DescriptorMissingState = errors.New("model: glm53 state tensor: arch descriptor requires a state tensor the checkpoint does not declare")

// ValidateGlm53StateTensorContract is the loader-side refusal of the
// arch-registration contract. It compares the checkpoint's declared extra
// state tensors against the arch descriptor's registered ones and refuses
// both mismatch directions:
//   - a checkpoint-declared class the descriptor lacks ->
//     ErrGlm53CheckpointUndeclaredState (a newest checkpoint loaded under an
//     older arch descriptor);
//   - a descriptor-required class the checkpoint lacks ->
//     ErrGlm53DescriptorMissingState (an older checkpoint loaded under the
//     newer arch descriptor).
//
// A matching pair validates with nil. When both directions mismatch, the
// checkpoint-declared direction is reported first, in stable class order.
func ValidateGlm53StateTensorContract(desc Glm53ArchStateDescriptor, checkpoint []Glm53StateTensorClass) error {
	for _, c := range checkpoint {
		if !desc.Declares(c) {
			return fmt.Errorf("%w: %s", ErrGlm53CheckpointUndeclaredState, c)
		}
	}
	for _, c := range desc.StateTensors {
		declared := false
		for _, p := range checkpoint {
			if p == c {
				declared = true
				break
			}
		}
		if !declared {
			return fmt.Errorf("%w: %s", ErrGlm53DescriptorMissingState, c)
		}
	}
	return nil
}

// Glm53LayerStateAllocation is the per-layer answer of the state allocation:
// which declared extra state tensors the layer carries and how many bytes each
// occupies.
type Glm53LayerStateAllocation struct {
	// Layer is the layer index the allocation was computed for.
	Layer int
	// IsKDA marks whether the layer is a KDA linear-attention layer under
	// cfg's cadence; only KDA layers carry extra state tensors.
	IsKDA bool
	// TensorBytes maps each declared-and-carried state tensor class to its
	// byte count for this layer. Classes the descriptor does not register are
	// absent, even on KDA layers.
	TensorBytes map[Glm53StateTensorClass]int64
	// Total is the sum over TensorBytes.
	Total int64
}

// AllocateGlm53LayerState threads the descriptor's declared extra state
// tensors through one layer's state allocation using the arch geometry cfg.
// On a KDA layer it includes every class the descriptor registers (the
// recurrent latent state plus the q/k/v convolution history); on a DSA layer,
// or under a descriptor that registers nothing, the allocation carries no
// extra state. Out-of-range layers follow the package convention of
// IsKDALayer: they are not KDA layers and allocate nothing. Geometry defaulting
// mirrors NewGLM5NextKDALayerState so the two seams cannot drift apart.
func AllocateGlm53LayerState(desc Glm53ArchStateDescriptor, cfg GLM5NextConfig, layer int) Glm53LayerStateAllocation {
	alloc := Glm53LayerStateAllocation{
		Layer:       layer,
		IsKDA:       cfg.IsKDALayer(layer),
		TensorBytes: make(map[Glm53StateTensorClass]int64),
	}
	if !alloc.IsKDA {
		return alloc
	}
	numHeads := cfg.KDANumHeads
	if numHeads <= 0 {
		numHeads = 64
	}
	headDim := cfg.KDAHeadDim
	if headDim <= 0 {
		headDim = 128
	}
	convWindow := cfg.ConvWindowSize
	if convWindow <= 0 {
		convWindow = 4
	}
	if desc.Declares(Glm53StateKDALatent) {
		bytes := int64(numHeads*headDim*headDim) * 4
		alloc.TensorBytes[Glm53StateKDALatent] = bytes
		alloc.Total += bytes
	}
	if desc.Declares(Glm53StateKDAConv) {
		bytes := int64(3*(convWindow-1)*numHeads*headDim) * 4
		alloc.TensorBytes[Glm53StateKDAConv] = bytes
		alloc.Total += bytes
	}
	return alloc
}

// Glm53StateTensorClassesFromManifest derives the extra state tensor classes a
// materialized checkpoint declares from its tensor manifest: any per-layer KDA
// linear-attention signature (a ".linear_attn." weight) implies the arch
// carries both the recurrent latent state and the convolution state. A
// manifest without that signature declares no extra state tensor. The rule is
// deliberately fail-closed: the canonical fused naming is the only accepted
// marker, so an ambiguous upstream-style q_proj/k_conv1d split cannot silently
// opt a checkpoint into the new state contract.
func Glm53StateTensorClassesFromManifest(man map[string]tensorMeta) []Glm53StateTensorClass {
	for name := range man {
		if strings.Contains(name, ".linear_attn.") {
			return append([]Glm53StateTensorClass(nil), allGlm53StateTensorClasses...)
		}
	}
	return nil
}
