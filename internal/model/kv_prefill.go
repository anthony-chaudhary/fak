package model

import "context"

// Prefill ingests a prompt and returns the logits of its LAST token (the
// distribution over the first generated token). Each token is placed at the next
// absolute position (Cache.Len()), so a prior Evict() compaction shifts these down.
func (s *Session) Prefill(ids []int) []float32 {
	s.requireGLM5NextSessionUnsupported("Session.Prefill")
	scope := s.beginBackendExecution()
	complete := false
	defer scope.finishCompleted(&complete)
	logits := s.prefillOwned(ids)
	complete = true
	return logits
}

// PrefillContext preserves Prefill's dispatch and ownership, but allows the
// partial-offload scalar HAL fallback to stop between completed tokens. Other
// selected operations finish at their existing boundary before observing ctx.
// A cancelled call returns no logits; its caller must retire the partial session.
func (s *Session) PrefillContext(ctx context.Context, ids []int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.requireGLM5NextSessionUnsupported("Session.PrefillContext")
	scope := s.beginBackendExecution()
	complete := false
	// A returned cancellation is primary over a secondary cleanup panic. Keep
	// completion-aware unwind so panic(nil) and Goexit are never made successful.
	defer scope.finishCompleted(&complete)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	logits := s.prefillOwnedContext(ctx, ids)
	if err := ctx.Err(); err != nil {
		s.failGemma4CachedContinuation(err)
		return nil, err
	}
	complete = true
	return logits, nil
}

func (s *Session) prefillOwned(ids []int) []float32 {
	return s.prefillOwnedContext(nil, ids)
}

func (s *Session) prefillOwnedContext(ctx context.Context, ids []int) []float32 {
	if len(ids) == 0 {
		return nil
	}
	if err := s.refuseUnimplementedV4FlashAttention(); err != nil {
		panic(err)
	}
	s.validateDeviceOnlyExecution("prefill")
	if s.M.Cfg.IsDeepSeekV41() {
		completed := false
		defer s.v41EPGuard(&completed)
		logits := func() []float32 {
			if rel := s.epAnnounce(epOpPrefill, ids); rel != nil {
				defer rel()
			}
			return s.prefillV41(ids)
		}()
		completed = true
		return logits
	}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	s.validateDenseGPULayers()
	if result, used, err := s.tryQwen35SequencePrefill(ids, true); used {
		if err != nil {
			s.failBackendForward(-1, "sequence prefill", err)
		}
		logits := s.Backend.Read(result.Logits)
		s.retireRequestResources()
		return logits
	}
	// Coordinated expert-parallel serve (#4835): announce this forward to the follower ranks
	// and hold the group until it completes, so they replay it and reach the same per-layer
	// AllReduces. Inert (one nil field read) unless a coordinator is installed on rank 0.
	if rel := s.epAnnounce(epOpPrefill, ids); rel != nil {
		defer rel()
	}
	if s.M.Cfg.usesMLAMoELayout() {
		s.requireGLMDsaSession()
		return s.glmDsaHead(s.tokenLoopHidden(s.tokenHiddenGLMDsa, ids))
	}
	if s.M.Cfg.isMiniMaxSparseAttn() {
		// MiniMax-M3 MSA: the incremental cache path runs the lightning-indexer block
		// selection per position over the cached K/V (minimax_m3_session.go), so cached
		// decode/prefix-reuse agree with the cacheless Forward. It must precede the generic
		// MoE token-loop below, which would otherwise run dense GQA on the sparse layers.
		s.requireMiniMaxSession()
		return s.head(s.tokenLoopHidden(s.tokenHiddenMiniMax, ids))
	}
	if s.M.Cfg.isGemma4() {
		// #5495: the DEDICATED gemma4 forward, on whatever resident store the model loaded as
		// (residentMatRows dispatches f32/Q8_0/Q4_K/int4/k-quant/GPTQ by name). It must precede
		// every generic lane below — each of them assumes the scalar cfg.HeadDim, which is the
		// wrong shape on this arch's local layers.
		return s.prefillGemma4Context(ctx, ids)
	}
	if s.Backend != nil {
		// A device session executes the WHOLE forward through the HAL (matWeightHAL now
		// dispatches q8w/q4kw/f32), so the device path must take precedence over the CPU
		// resident-quant lanes below — else a Q4_K device session prefilled on the CPU cache
		// (s.Cache) while decoding from the empty device cache (s.halKV) → garbage (#949).
		s.requirePreNorm("HAL prefill")
		return s.prefillHALContext(ctx, ids, true)
	}
	if s.Q4 {
		// Resident int4 prefill: the batched Q8 GEMM has no int4 twin yet, so prefill runs
		// the shared per-token blockStep with the int4 kernel. Slower than batched but uses
		// only the resident int4 weights (the lean q4-only mode freed the Q8_0 copy).
		return s.headQ4(s.tokenLoopHidden(s.tokenHiddenQ, ids))
	}
	if s.Q4K {
		// Resident Q4_K prefill (plan P1/P3). For a PreNorm standard-attention model (the
		// q4_k_m regime the plan targets), run the BATCHED q4 GEMM: q4_k_m majority via
		// q4kGemm, Q6_K / normalize-sensitive minority via the proven qGemm8, KV filled in
		// one pass — each weight super-block dequantized once and reused across all P prompt
		// tokens. Architectures the batched q4 lane does not yet cover (MoE / DenseMLP /
		// Alibi / Qwen35-hybrid / non-PreNorm / layer-specific RoPE theta) fall back to the
		// per-token blockStep, exactly as the Q8 token-loop fallback does; the
		// prefillBatchedMoEOK MoE subset is batched too, its routed FFN run per row on the
		// token loop's kernel. The LM head is whichever resident format it loaded as
		// (headResident).
		if !s.prefillNeedsTokenLoop() {
			return s.headResident(s.prefillBatchedQ4K(ids))
		}
		// Qwen3.5/3.6 hybrid (the q8PrefillNeedsTokenLoop case the generic batched-Q4K lane
		// refuses): batch each layer's projection/MLP GEMMs over the prompt panel while keeping
		// the GDN recurrence, the resident-Q4K twin of the q8Qwen35HybridPrefillOK gate. Closes
		// QWEN36-NATIVE-PERF-PLAN P3's per-token-fallback prefill wall.
		if logits, used := s.tryPrefillQwen35HybridQ4K(ids, true); used {
			return logits
		}
		s.observeQ4KPrefillGEMM(Q4KPrefillGEMMTokenLoop)
		return s.headResident(s.tokenLoopHidden(s.tokenHiddenQ, ids))
	}
	if s.GPTQ {
		return s.headResident(s.tokenLoopHidden(s.tokenHiddenGPTQ, ids))
	}
	if s.Metal {
		// The Qwen3.6 hybrid (Gated-DeltaNet) cannot run through the generic full-attention
		// Metal prefill, so route it to the hybrid twin — projection/MLP GEMMs batched on the
		// GPU, GDN recurrence on the CPU — instead of tripping requirePreNorm (#71).
		if metalQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			return s.headQ(s.prefillBatchedMetalQwen35Hybrid(ids))
		}
		s.requirePreNorm("Metal prefill")
		// Prefill projections on the GPU; the head stays the cheap CPU single-token GEMV.
		return s.headQ(s.prefillBatchedMetal(ids))
	}
	if s.PrecisionPolicy != nil {
		return s.prefillDynamic(ids)
	}
	if s.Quant {
		// The hybrid panel appends at Cache.Len() and carries the existing
		// convolution/recurrent state, including a restored prefix. Keep suffix
		// projections batched instead of re-streaming the weights per token.
		if s.q8Qwen35HybridPrefillSessionOK(len(ids)) {
			return s.prefillQwen35HybridQ(ids)
		}
		if s.prefillNeedsTokenLoop() {
			return s.headQ(s.tokenLoopHidden(s.tokenHiddenQ, ids))
		}
		return s.headQ(s.prefillBatchedQ(ids))
	}
	if s.prefillNeedsTokenLoop() {
		// The batched f32 GEMM is one-weight-many-rows and still hardcodes the PreNorm
		// block copy and one shared RoPE table. DenseMLP removes the up-projection, ALiBi
		// replaces RoPE with score bias, non-PreNorm topology changes the residual/norm
		// graph, and Gemma3-style per-layer RoPE theta changes the rotation by layer; these
		// axes run through blockStep here, where the FFN/topology/RoPE dispatch lives. The
		// prefillBatchedMoEOK subset is the exception: its attention is batched and its
		// routed FFN runs per row through blockStep's exact call.
		return s.prefillTokenLoop(ids)
	}
	// PreNorm (default): batched + parallel, one GEMM over all P tokens instead of
	// GEMV-per-token. Bit-identical to the per-token tokenHidden loop
	// (TestPrefillBatchedMatchesSerial), so the cache it builds is exactly the proven
	// one and R2/R3/R14 stay exact.
	return s.head(s.prefillBatched(ids))
}
