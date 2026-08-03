package model

// arch_support_gemma4.go — the fail-closed refusal for a resident-quant CPU forward run
// over a checkpoint whose per-layer attention geometry the generic uniform-geometry block
// path cannot express (issue #4274), and the predicate that decides it.
//
// Gemma-4 interleaves two attention regimes with DIFFERENT shapes per layer: local/sliding
// layers use a small head_dim, global/full layers a large one (gemma4.go headDimForLayer).
// That is the entire reason gemma4.go exists as a dedicated forward. The generic block path
// (tokenHiddenQ -> blockStep -> composeBlock) runs the shared block code, whose qk-norm band
// passes the SCALAR cfg.HeadDim to applyQKNormCfg. A per-layer q_norm/k_norm weight length
// differs from that scalar, so applyQKNormCfg's `len(w) != hd` guard panics deep in the band
// on the first Prefill — a cryptic crash with no architecture named. This file replaces that
// with a typed, inspectable refusal at the entry.
//
// Since #5495 the refusal's REACHABLE domain is smaller: Session.Prefill / PrefillNoLogits /
// Step now dispatch a gemma4 checkpoint to the dedicated forward (gemma4_session.go) before
// any generic resident-quant lane, and that path feeds applyQKNormCfg the per-layer head_dim.
// The refusal is KEPT, not deleted, because two populations still land on it: a
// heterogeneous-head_dim architecture with NO dedicated forward at all, and a gemma4 session
// whose execution mode the bridge does not wire (device / Metal / dynamic precision). Both
// are genuinely unsupported, and a silent wrong answer there would be worse than a refusal.

// ResidentQuantUnsupportedError is the typed refusal raised when a Session cannot run a
// checkpoint whose per-layer attention head_dim the generic uniform-geometry block path
// cannot serve (issues #4274, #5495). It carries the architecture family key, the
// resident-quant format in play, and — when the blocker is an unwired execution mode rather
// than the geometry itself — the name of that mode, so the message names the exact remedy.
type ResidentQuantUnsupportedError struct {
	// Arch is the checkpoint's architecture family key (cfg.archFamilyKey(), e.g. "gemma4").
	Arch string
	// Format is the resident-quant format in play ("Q4_K", "Q4", "Q8_0", "GPTQ"), or
	// "resident-quant" when the exact flag is not known at the raise site.
	Format string
	// Mode names the unwired execution mode (e.g. "Metal prefill") when the architecture HAS
	// a dedicated forward but this session mode has no twin of it. Empty when the blocker is
	// the geometry itself — no dedicated forward exists for it.
	Mode string
}

func (e *ResidentQuantUnsupportedError) Error() string {
	arch := e.Arch
	if arch == "" {
		arch = "<unknown>"
	}
	format := e.Format
	if format == "" {
		format = "resident-quant"
	}
	msg := "model: unsupported " + format + " forward for architecture " + arch +
		": this checkpoint uses a per-layer attention head_dim (e.g. gemma4's local/global " +
		"layers) that the generic uniform-geometry qk-norm band assumes is the scalar " +
		"cfg.HeadDim (issue #4274)."
	if e.Mode != "" {
		return msg + " The dedicated gemma4 forward IS wired on the host resident path " +
			"(issue #5495), but not for this session's " + e.Mode + " mode; drop that mode and " +
			"serve on a CPU resident session."
	}
	return msg + " No dedicated forward is wired for this geometry on the resident-quant " +
		"path (gemma4 has one since issue #5495; this architecture does not). Serve this " +
		"architecture on the f32 (non-quant) forward."
}

// heterogeneousHeadDim reports whether the model uses a per-layer attention head_dim that
// differs from the scalar cfg.HeadDim (gemma4 local vs global — headDimForLayer). This is the
// exact condition that makes the generic qk-norm band — which passes the scalar HeadDim to
// applyQKNormCfg — panic on a q_norm/k_norm weight whose length is a DIFFERENT per-layer
// head_dim. Uniform-geometry arches (empty HeadDimPerLayer, or every entry == HeadDim) return
// false and keep the shared path bit-for-bit.
func (c Config) heterogeneousHeadDim() bool {
	for _, hd := range c.HeadDimPerLayer {
		if hd > 0 && hd != c.HeadDim {
			return true
		}
	}
	return false
}

// residentQuantForwardFormat names the resident-quant CPU format this session runs, for the
// typed refusal message. Empty when the session is not on a resident-quant CPU forward.
func (s *Session) residentQuantForwardFormat() string {
	switch {
	case s.Q4K:
		return "Q4_K"
	case s.Q4:
		return "Q4"
	case s.GPTQ:
		return "GPTQ"
	case s.Quant:
		return "Q8_0"
	default:
		return ""
	}
}

// residentQuantForwardUnsupported returns a typed *ResidentQuantUnsupportedError when this
// session runs a resident-quant CPU forward over a checkpoint whose per-layer head_dim the
// generic block path cannot express (gemma4), and nil otherwise. It is the fail-closed
// pre-check seam for issue #4274: a serve / bench / preflight caller can refuse BY NAME before
// decoding a token, instead of the mid-Prefill band panic. It reads only cfg + the session's
// resident-quant flags — no weights, no forward run — so it is safe to call anywhere.
func (s *Session) residentQuantForwardUnsupported() error {
	format := s.residentQuantForwardFormat()
	if format == "" {
		return nil // not a resident-quant CPU forward — nothing to refuse here
	}
	if s.M.Cfg.heterogeneousHeadDim() {
		return &ResidentQuantUnsupportedError{Arch: s.M.Cfg.archFamilyKey(), Format: format}
	}
	return nil
}

// requireResidentQuantForwardSupported fails the resident-quant forward closed with the typed
// ResidentQuantUnsupportedError the moment it is entered for an unsupported per-layer geometry,
// so the operator gets a NAMED refusal at the forward entry instead of the cryptic "qk-norm
// weight length does not match head_dim" panic deep in the band (issue #4274). It mirrors
// requireMiniMaxSession's fail-closed idiom for an unwired resident-quant session mode.
func (s *Session) requireResidentQuantForwardSupported() {
	if err := s.residentQuantForwardUnsupported(); err != nil {
		panic(err)
	}
}
