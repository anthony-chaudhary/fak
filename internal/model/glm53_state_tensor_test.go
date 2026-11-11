package model

import (
	"errors"
	"testing"
)

// TestGlm53StateTensorRefusalDirections pins the loader refusal in both
// directions: a checkpoint declaring the extra state tensor under a legacy
// descriptor is refused with ErrGlm53CheckpointUndeclaredState, a descriptor
// requiring it under a checkpoint lacking it is refused with
// ErrGlm53DescriptorMissingState, and matching pairs validate.
func TestGlm53StateTensorRefusalDirections(t *testing.T) {
	err := ValidateGlm53StateTensorContract(Glm53LegacyStateDescriptor, []Glm53StateTensorClass{Glm53StateKDALatent})
	if !errors.Is(err, ErrGlm53CheckpointUndeclaredState) {
		t.Fatalf("legacy descriptor vs KDA checkpoint: err = %v, want ErrGlm53CheckpointUndeclaredState", err)
	}

	err = ValidateGlm53StateTensorContract(Glm53NextStateDescriptor, nil)
	if !errors.Is(err, ErrGlm53DescriptorMissingState) {
		t.Fatalf("next descriptor vs empty checkpoint: err = %v, want ErrGlm53DescriptorMissingState", err)
	}

	// A partial checkpoint declaration still refuses, naming the missing class.
	err = ValidateGlm53StateTensorContract(Glm53NextStateDescriptor, []Glm53StateTensorClass{Glm53StateKDAConv})
	if !errors.Is(err, ErrGlm53DescriptorMissingState) {
		t.Fatalf("next descriptor vs conv-only checkpoint: err = %v, want ErrGlm53DescriptorMissingState", err)
	}

	if err := ValidateGlm53StateTensorContract(Glm53NextStateDescriptor, []Glm53StateTensorClass{Glm53StateKDALatent, Glm53StateKDAConv}); err != nil {
		t.Fatalf("matching next pair: err = %v, want nil", err)
	}
	if err := ValidateGlm53StateTensorContract(Glm53LegacyStateDescriptor, nil); err != nil {
		t.Fatalf("matching legacy pair: err = %v, want nil", err)
	}
}

// TestGlm53StateTensorAllocationThreading pins the per-layer allocation values
// on the pinned GLM-5.3-Flash geometry: a KDA layer carries exactly the
// recurrent latent state plus the q/k/v convolution history declared by the
// descriptor, a DSA layer carries none, and a legacy descriptor threads
// nothing through even on KDA layers. The threaded total must agree byte for
// byte with the existing KDA layer-state seam.
func TestGlm53StateTensorAllocationThreading(t *testing.T) {
	cfg := DefaultGLM5NextConfig()

	wantLatent := int64(cfg.KDANumHeads*cfg.KDAHeadDim*cfg.KDAHeadDim) * 4
	wantConv := int64(3*(cfg.ConvWindowSize-1)*cfg.KDANumHeads*cfg.KDAHeadDim) * 4
	// Closed contract on the pinned geometry: 64 heads x 128^2 f32 latent,
	// 3 x 3 x 64 x 128 f32 convolution history.
	if wantLatent != 4194304 || wantConv != 294912 {
		t.Fatalf("pinned geometry drifted: latent = %d, conv = %d", wantLatent, wantConv)
	}

	alloc := AllocateGlm53LayerState(Glm53NextStateDescriptor, cfg, 0)
	if !alloc.IsKDA {
		t.Fatalf("layer 0 IsKDA = false, want true")
	}
	if got := alloc.TensorBytes[Glm53StateKDALatent]; got != wantLatent {
		t.Fatalf("latent bytes = %d, want %d", got, wantLatent)
	}
	if got := alloc.TensorBytes[Glm53StateKDAConv]; got != wantConv {
		t.Fatalf("conv bytes = %d, want %d", got, wantConv)
	}
	if got := alloc.Total; got != wantLatent+wantConv {
		t.Fatalf("total = %d, want %d", got, wantLatent+wantConv)
	}

	kda := NewGLM5NextKDALayerState(cfg.KDANumHeads, cfg.KDAHeadDim, cfg.ConvWindowSize)
	if got := kda.ByteSize(); got != alloc.Total {
		t.Fatalf("NewGLM5NextKDALayerState.ByteSize() = %d, allocation total = %d", got, alloc.Total)
	}

	dsa := AllocateGlm53LayerState(Glm53NextStateDescriptor, cfg, 3)
	if dsa.IsKDA || len(dsa.TensorBytes) != 0 || dsa.Total != 0 {
		t.Fatalf("DSA layer 3 allocation = %+v, want no extra state", dsa)
	}

	legacy := AllocateGlm53LayerState(Glm53LegacyStateDescriptor, cfg, 0)
	if legacy.Total != 0 || len(legacy.TensorBytes) != 0 {
		t.Fatalf("legacy allocation = %+v, want no extra state", legacy)
	}
}

// TestGlm53StateTensorManifestDeclaration pins the checkpoint-declaration
// derivation from a materialized manifest: a KDA linear-attention signature
// declares both extra state tensors and validates against the next descriptor
// while refusing the legacy one; a plain attention manifest declares none.
func TestGlm53StateTensorManifestDeclaration(t *testing.T) {
	man := map[string]tensorMeta{
		"model.layers.0.self_attn.linear_attn.in_proj.weight": {},
		"model.layers.0.mlp.router.weight":                    {},
	}
	got := Glm53StateTensorClassesFromManifest(man)
	if len(got) != 2 || got[0] != Glm53StateKDALatent || got[1] != Glm53StateKDAConv {
		t.Fatalf("classes = %v, want [kda_latent_state kda_conv_state]", got)
	}
	if err := ValidateGlm53StateTensorContract(Glm53NextStateDescriptor, got); err != nil {
		t.Fatalf("manifest-derived declaration vs next descriptor: err = %v, want nil", err)
	}
	if err := ValidateGlm53StateTensorContract(Glm53LegacyStateDescriptor, got); !errors.Is(err, ErrGlm53CheckpointUndeclaredState) {
		t.Fatalf("manifest-derived declaration vs legacy descriptor: err = %v, want ErrGlm53CheckpointUndeclaredState", err)
	}

	plain := map[string]tensorMeta{"model.layers.0.self_attn.q_proj.weight": {}}
	if got := Glm53StateTensorClassesFromManifest(plain); len(got) != 0 {
		t.Fatalf("plain manifest classes = %v, want none", got)
	}
}
