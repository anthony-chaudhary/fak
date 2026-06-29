package model

import "strings"

// Architecture-support gate (issue #934).
//
// fak's in-kernel forward implements a fixed set of token-mixer paths: the
// standard separate-projection GQA attention (forward.go attnSeq), the qwen35
// gated full-attention + Gated-DeltaNet linear-attention hybrid (qwen35.go), the
// GLM-DSA MLA path, the MiniMax lightning-indexer path, and the Gemma-4
// heterogeneous-geometry path. A checkpoint whose architecture maps onto none of
// these would, at forward time, look up a projection tensor that does not exist
// and panic deep inside a request — exactly the failure #934 witnessed on the
// real lmstudio-community/Qwen3.6-27B-GGUF: it is a Qwen3-Next-style
// Gated-DeltaNet/SSM hybrid (fused attn_qkv + a per-layer ssm_* core, no
// self_attn.q_proj), and when its GGUF general.architecture is NOT recognized as
// the qwen35 family the loader leaves config.layer_types empty. With empty
// layer_types every layer falls to the standard attnSeq path, which calls
// m.tensor("…self_attn.q_proj.weight") and panics ("model: missing tensor …")
// on the first real /v1/chat/completions turn.
//
// refuseUnsupportedHybridArch turns that would-be mid-request panic into an
// honest, typed, named LOAD-time refusal — the model-support analogue of the
// FitTooBig capacity refusal (internal/compute FitError). It is the interim the
// #934 acceptance asks for ("a typed refusal naming the unsupported arch …
// instead of a nil/missing-tensor panic mid-request") while the full GDN/SSM
// forward + real-artifact parity (epic #931, kernels #65/#67) remains open.

// UnsupportedArchError is the typed refusal newModel returns when a checkpoint
// carries weights for an architecture whose in-kernel forward fak does not yet
// implement. A caller (fak serve) can surface a named refusal before serving a
// single token instead of binding the gateway and panicking on the first turn.
type UnsupportedArchError struct {
	// Arch is the loaded checkpoint's architecture / model_type (cfg.ModelType,
	// i.e. the GGUF general.architecture). Empty if the source did not declare one.
	Arch string
	// Tensor is one witness weight whose presence signals the unsupported arch —
	// here a per-layer Gated-DeltaNet/SSM tensor the standard forward cannot run.
	Tensor string
}

func (e *UnsupportedArchError) Error() string {
	arch := e.Arch
	if arch == "" {
		arch = "<unknown>"
	}
	return "model: unsupported architecture " + arch +
		": checkpoint carries Gated-DeltaNet/SSM hybrid weights (e.g. " + e.Tensor +
		") but config.layer_types is empty, so fak's qwen35-family GDN forward is not enabled and" +
		" the standard attention path would panic on a missing self_attn.q_proj.weight." +
		" Serving this architecture on the in-kernel forward is not yet supported (issue #934);" +
		" use a checkpoint of a supported architecture or a backend that implements it."
}

// refuseUnsupportedHybridArch fails the load with a typed UnsupportedArchError when
// the manifest carries the Gated-DeltaNet/SSM hybrid signature — any per-layer
// linear_attn.* (canonicalized ssm_*) tensor — while the config does NOT enable the
// qwen35-family hybrid forward (IsQwen35Hybrid is false because layer_types is empty).
//
// That combination is unambiguous: linear_attn.* weights exist ONLY in qwen35-family
// (Qwen3.5 / Qwen3.6 / Qwen3-Next) checkpoints, and every config that legitimately
// drives the GDN forward marks its linear-attention layers in layer_types (so
// IsQwen35Hybrid is true). The only way to reach "linear_attn weights present, hybrid
// forward disabled" is a qwen35-family GGUF whose general.architecture the loader did
// not recognize as qwen35 — precisely the #934 state. It must run BEFORE
// splitFusedProjections so the operator gets this named arch refusal rather than a
// misleading "fused tensor has N rows, config implies M" from the GDN in_proj
// (attn_qkv) being mistaken for a standard q/k/v stack.
func refuseUnsupportedHybridArch(cfg Config, man map[string]tensorMeta) error {
	if cfg.IsQwen35Hybrid() {
		return nil // layer_types marks the linear-attention layers — GDN forward is wired
	}
	for name := range man {
		if strings.Contains(name, ".linear_attn.") {
			return &UnsupportedArchError{Arch: cfg.ModelType, Tensor: name}
		}
	}
	return nil
}
