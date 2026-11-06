package modelengine

import "errors"

// Source: sgl-project/sglang, deepseek_v4_backend.py:3862-3927, revision
// 3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba (Apache-2.0):
// https://github.com/sgl-project/sglang/blob/3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba/python/sglang/srt/layers/attention/deepseek_v4_backend.py
// Copyright (c) 2026 sgl-project
//
// Adapted for Fak as idiomatic Go (no upstream code blocks copied): only the
// host-side safety envelope of DeepSeek-V4.1 SWA bounded replay is ported, as
// a pure admission gate. Upstream bounds replay to a 128-row window
// (SWA_WINDOW = 128) and rebuilds each hit's page table from that window
// (BuildCausalSwaPageIndices); any configuration that forces prompt rows to
// be read defeats the trim, so it is refused at admission instead of handled
// per step. The refusal set carried here is the closed contract named by the
// borrow ticket: prompt logprobs, prefill CUDA graphs, and DP attention. The
// replay kernel itself is out of scope for this leaf.

// SwaBoundedReplayConfig captures the host-side flags relevant to admitting
// SWA bounded-replay mode for a DeepSeek-V4.1-class workload.
type SwaBoundedReplayConfig struct {
	// PromptLogprobs is true when the workload requests log probabilities
	// over prompt tokens. Those requests read prompt rows, which a bounded
	// replay never materializes, so they cannot run in replay mode.
	PromptLogprobs bool
	// PrefillCudaGraph is true when full prefill CUDA-graph capture is
	// enabled. Bounded replay floors a hit's extend to whole pages and runs
	// up to one window of replay rows, which a captured prefill graph does
	// not honor; only disabled or breakable capture coexists with replay.
	PrefillCudaGraph bool
	// DPAttention is true when data-parallel attention is enabled. Replay
	// hits are folded into the hitting rank's forward, which a DP-attention
	// split across ranks does not support.
	DPAttention bool
}

// Closed contracts for the three refusal branches of SwaBoundedReplayGate.
// Callers match them with errors.Is; the prose is not part of the API.
var (
	// ErrSwaReplayPromptLogprobs refuses bounded replay when prompt logprobs
	// are requested: the request would have to read prompt rows that replay
	// does not keep.
	ErrSwaReplayPromptLogprobs = errors.New("modelengine: SWA bounded replay refused: prompt logprobs requested")
	// ErrSwaReplayPrefillCudaGraph refuses bounded replay when full prefill
	// CUDA-graph capture is enabled: replay floors extends to whole pages
	// and runs extra replay rows that a captured graph does not honor.
	ErrSwaReplayPrefillCudaGraph = errors.New("modelengine: SWA bounded replay refused: prefill CUDA graph enabled")
	// ErrSwaReplayDPAttention refuses bounded replay when DP attention is
	// enabled: replay hits fold into the hitting rank's forward, which DP
	// attention does not support.
	ErrSwaReplayDPAttention = errors.New("modelengine: SWA bounded replay refused: DP attention enabled")
)

// SwaBoundedReplayGate is the pure host-side admission gate for SWA
// bounded-replay mode. It returns nil when every flag is clear and an error
// naming every violated branch otherwise; each violated branch matches its
// sentinel with errors.Is. No numerics change and no state is read or
// written: the gate is safe to call on any path that resolves these flags.
func SwaBoundedReplayGate(cfg SwaBoundedReplayConfig) error {
	var errs []error
	if cfg.PromptLogprobs {
		errs = append(errs, ErrSwaReplayPromptLogprobs)
	}
	if cfg.PrefillCudaGraph {
		errs = append(errs, ErrSwaReplayPrefillCudaGraph)
	}
	if cfg.DPAttention {
		errs = append(errs, ErrSwaReplayDPAttention)
	}
	return errors.Join(errs...)
}
