package model

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// qwen35_prefill_q4k.go is the resident-Q4_K twin of qwen35_prefill.go (the Q8 hybrid
// Gated-DeltaNet fresh-prefill path). It exists for the same reason prefillBatchedQ4K
// (prefill_q4k.go) is the twin of prefillBatchedQ: when Qwen3.6-27B is loaded through the
// resident-Q4_K path (FAK_Q4K=1), the generic batched-Q4_K prefill refuses the hybrid arch
// (q8PrefillNeedsTokenLoop → IsQwen35Hybrid), so prompt processing fell back to the
// per-token blockStep loop — re-streaming every weight P times. On the M3 Pro that pins
// Qwen3.6 prefill at ~0.5 tok/s (22 tok in ~46 s) regardless of how compact the decode
// stream is. This path keeps the GDN recurrence but batches each layer's projection/MLP
// GEMMs over the prompt panel, exactly as the Q8 hybrid path does, closing
// QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P3's open "qwen35-hybrid falls back to the per-token
// loop" item for the resident-Q4_K lane.
//
// It differs from prefillQwen35HybridQHidden in exactly ONE way — each projection GEMM
// dispatches by resident format via `proj`, the same per-weight dispatch prefillBatchedQ4K
// uses:
//
//   - q4kw-resident (the identity-normalized q4_k_m majority: self_attn v_proj/o_proj,
//     mlp gate/up/down) → q4kGemm on the f32 activation, each super-block dequantized ONCE
//     and reused across all P prompt tokens.
//   - everything else (self_attn q/k, and EVERY linear_attn.* projection — these are
//     reordered/unpermuted for qwen35, so ResidentQ4KEligible keeps them out of q4kw; plus
//     any Q6_K weight) → q8GemmDispatch on a Q8-quantized activation panel: CPU qGemm8 by
//     default, Metal Q8 GEMM when MetalQ4K is enabled (#1087).
//
// Everything else — the conv1d causal scan, the per-head Gated-DeltaNet recurrence, the
// L2/RMS norms, RoPE, the causal GQA over the f32 KV cache, SwiGLU, the residuals — is the
// identical f32 math prefillQwen35HybridQ runs, copied verbatim so the recurrence has a
// single proven reference. The cache it builds is the same f32 object (Kraw pre-RoPE, K
// post-RoPE, V, pos, plus the linearAttnCache conv/recurrent state).
//
// Correctness contract vs the per-token Q4K decode path (tokenHiddenQ via sessionQ4KKernel):
// sessionQ4KKernel.mul resolves a projection by name with the IDENTICAL order proj uses
// (q4kw first → q4kMatRows, else → qGemm8/qMatRows on m.q8). So per weight, both paths take
// the same kernel: the q4_k_m majority is bit-identical on CPU (q4kGemm == q4kMatRows per
// (o,t), TestQ4KGemmMatchesMatRows) and approximate on Metal; the Q8 minority differs only by
// the documented Q8 deferred-reduction/FMA-rounding drift the Q8 hybrid path's own gate already
// covers. The recurrence is the same f32 math fed by those projections. Pinned by
// TestPrefillQwen35HybridQ4KMatchesTokenLoop and the Metal Q4K/Q8 gates.

// q4kQwen35HybridPrefillOK gates the batched resident-Q4_K hybrid prefill. It is the Q8
// hybrid path's architecture gate (qwen35HybridPrefillArchOK), including the qwen3_5_moe
// hybrid: an MoE layer keeps its attention/GDN sublayer batched and runs the routed FFN
// per row on the token loop's kernel (prefillQwen35RowFFNQ). Only the projection kernel
// differs from the Q8 path. tryPrefillQwen35HybridQ4K applies the MoE session limits.
func q4kQwen35HybridPrefillOK(cfg Config, promptLen int) bool {
	return qwen35HybridPrefillArchOK(cfg, promptLen)
}

// q4kQwen35HybridPrefillAtPositionOK preserves the established fresh-prefill
// amortization threshold while admitting a smaller continuation chunk. Once a
// session already owns prefix state, forcing a sub-threshold tail through decode
// would abandon the resident panel path exactly where bounded chunking needs it.
// Reusing the original gate at its threshold preserves every architecture and
// diagnostic escape-hatch check before any cache or recurrent state is touched.
func q4kQwen35HybridPrefillAtPositionOK(cfg Config, promptLen, base int) bool {
	return q4kQwen35HybridPrefillAtPositionOKFor(cfg, promptLen, base, false)
}

// qwen35HybridMetalResidentMinPrompt is the fresh-prompt batched-prefill threshold on
// the backend-nil resident-Q4_K Metal session (#13694). qwen35HybridQBatchMinPrompt (16)
// is an amortization heuristic from the CPU qGemm8 path, not a correctness bound: the
// batched body runs the conv1d scan against the (possibly empty) history row by row and
// the GDN recurrence token by token, and the continuation path already admits every
// P >= 1 (TestPrefillQwen35HybridQ4KAppendMatchesMonolithic). On Metal, P in 2..15 paid
// one full forward per token — P weight streams and P command-buffer round-trips per
// projection — where one batched dispatch per projection group streams each weight once,
// so the Metal resident session batches from two tokens. P=1 keeps the token step.
const qwen35HybridMetalResidentMinPrompt = 2

// q4kQwen35HybridPrefillAtPositionOKFor is q4kQwen35HybridPrefillAtPositionOK with the
// session's Metal residency threaded in: a fresh prompt on the Metal resident session
// needs only qwen35HybridMetalResidentMinPrompt tokens; every other session keeps the
// historical threshold. The architecture and diagnostic gate is evaluated at the
// threshold so the length qualification above is the only thing that changes.
func q4kQwen35HybridPrefillAtPositionOKFor(cfg Config, promptLen, base int, metalResident bool) bool {
	if promptLen <= 0 || base < 0 {
		return false
	}
	if base == 0 {
		minFresh := qwen35HybridQBatchMinPrompt
		if metalResident {
			minFresh = qwen35HybridMetalResidentMinPrompt
		}
		if promptLen < minFresh {
			return false
		}
	}
	return q4kQwen35HybridPrefillOK(cfg, max(promptLen, qwen35HybridQBatchMinPrompt))
}

// qwen35ResidentDecodeAutoDisable is the config-surface opt-out consulted by
// qwen35ResidentDecodeAutoEligible. It is a seam rather than an environment read:
// an admission posture is behavior, not a credential, so it lives on the config
// surface (internal/envconfiglint's CONFIG_NOT_ENV rule; the former
// FAK_QWEN35_RESIDENT_DECODE_AUTO env read was relocated here). Default (false)
// keeps the route ON; SetQwen35ResidentDecodeAuto(false) forces the historical path.
var qwen35ResidentDecodeAutoDisable bool

// SetQwen35ResidentDecodeAuto declares whether the resident GDN sequence-owner
// auto-admission route is enabled. It is the config-surface replacement for the
// retired FAK_QWEN35_RESIDENT_DECODE_AUTO env read (whose only opt-out token was "0").
func SetQwen35ResidentDecodeAuto(on bool) { qwen35ResidentDecodeAutoDisable = !on }

// qwen35ResidentDecodeAutoDisabled reports whether the resident GDN sequence-owner
// auto-admission route has been explicitly turned off. Default (undeclared) is OFF.
func qwen35ResidentDecodeAutoDisabled() bool { return qwen35ResidentDecodeAutoDisable }

// qwen35ResidentDecodeAutoEligible reports whether this session may auto-admit the
// resident GDN sequence owner to reach the one-command-buffer fused linear-attention
// decode block. It is deliberately exact: only a Qwen3.5-family hybrid whose Q8 runtime band
// resolves (the 64-layer Qwen3.8, or a declared-cadence 128/128-GDN hybrid such as Qwen3.5-4B),
// on the backend-nil resident-Q4_K Metal lane, at
// a fresh or restored prompt boundary. Every other case declines so the historical decode path is
// byte-identical. The route is ON by default (measured 0.4-0.9 -> 4.2-4.6 tok/s
// decode on the M3 Pro, Qwen3.8-27B Q4_K_M, bit-identical pooled parity;
// Qwen3.5-4B 6.6 -> 23.5 tok/s with greedy parity); declare
// SetQwen35ResidentDecodeAuto(false) to force the historical path.
func (s *Session) qwen35ResidentDecodeAutoEligible() bool {
	if s == nil || s.M == nil || s.Cache == nil || s.Backend != nil || !s.Q4K || !s.MetalQ4K {
		return false
	}
	if s.M.prism != nil {
		return false // fused Metal block has no signed Hadamard input stage
	}
	if qwen35ResidentDecodeAutoDisabled() {
		return false
	}
	if newQwen35MetalGDNSequenceBackend == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return false
	}
	if s.qwen35HAL != nil && (s.qwen35HAL.sequenceAccepted || s.qwen35HAL.decodeAccepted) {
		return false
	}
	// A restored (non-fresh) prefix is admissible: Enable seeds the owners from
	// the complete host linear-attention state, or declines without mutation.
	if _, err := qwen38MetalQ8RuntimeNames(s.M.Cfg); err != nil {
		return false // not a topology the fused block/owner is proven on
	}
	return true
}

// hybridQ4KProj is the per-weight projection dispatch shared by every GEMM in the batched
// resident-Q4_K hybrid prefill: q4kw-resident -> q4kGemm on the raw f32 activation Xf;
// otherwise -> q8GemmDispatch on the pre-quantized Q8 panel Xq (CPU qGemm8 by default,
// Metal Q8 GEMM when MetalQ4K is enabled). Mirrors prefillBatchedQ4K's `proj`.
type hybridQ4KProj func(name string, Xf []float32, Xq *q8Panel) []float32

// hybridQ4KGroup runs a group of projections that share one activation panel (Xf f32, Xq the
// pre-quantized Q8 panel) through the batched one-command-buffer q4_k GEMM group, filling any
// non-grouped member per-weight via the shared proj. Results are returned in `names` order and are
// identical to calling proj per name — just fewer Metal command buffers (the prefill-wall lever).
type hybridQ4KGroup func(names []string, Xf []float32, Xq *q8Panel) [][]float32

// qwen35PromptPanelQuantum is the narrowest panel the whole-forward graph is
// admitted for on a prompt walk; shorter prompts stay on the host forward.
const qwen35PromptPanelQuantum = 32

// qwen35PromptPanelNarrowRows is the panel width for models above the
// qwen35PromptPanelWideLayerWidth class. A whole-forward panel is one command
// buffer, whose GPU time scales with rows x weights: a 256-row Qwen3.5-4B panel
// measured 448-668 ms on an M3 Pro across 0..4k context, which at Qwen3.8-27B's
// ~6.75x weights is seconds and grows with context toward the 10 s limit.
const qwen35PromptPanelNarrowRows = 128

// qwen35PromptPanelWideLayerWidth is the largest NumLayers*HiddenSize (Qwen3.5-4B:
// 32*2560) admitted to metalgemm.PromptPanelMaxTokens-row panels.
const qwen35PromptPanelWideLayerWidth = 32 * 2560

// qwen35PromptPanelMaxRows is the widest prompt panel the walk issues for cfg.
func qwen35PromptPanelMaxRows(cfg Config) int {
	if cfg.NumLayers*cfg.HiddenSize <= qwen35PromptPanelWideLayerWidth {
		return metalgemm.PromptPanelMaxTokens
	}
	return min(qwen35PromptPanelNarrowRows, metalgemm.PromptPanelMaxTokens)
}

// qwen35PromptPanelEnds splits an n-token prompt into the fewest whole-forward
// graph panels of at most maxRows rows, with widths that differ by at most one,
// and returns each panel's exclusive end offset. Balancing keeps every panel at
// least qwen35PromptPanelQuantum wide, so the walk covers the whole prompt with no
// sub-quantum remainder. It returns nil below one quantum.
func qwen35PromptPanelEnds(n, maxRows int) []int {
	if n < qwen35PromptPanelQuantum {
		return nil
	}
	panels := (n + maxRows - 1) / maxRows
	base, wider := n/panels, n%panels
	ends := make([]int, panels)
	end := 0
	for i := range ends {
		end += base
		if i < wider {
			end++
		}
		ends[i] = end
	}
	return ends
}

// tryPrefillQwen35HybridQ4K is the production selection seam. A declined
// geometry returns before the resident implementation can mutate KV, convolution,
// or recurrent state; the caller retains the historical token-loop behavior.
func (s *Session) tryPrefillQwen35HybridQ4K(ids []int, wantLogits bool) ([]float32, bool) {
	if !q4kQwen35HybridPrefillAtPositionOKFor(s.M.Cfg, len(ids), s.Cache.Len(), s.qwen35HybridMetalResidentPrefill()) {
		return nil, false
	}
	if s.M.Cfg.IsMoE() {
		return s.tryPrefillQwen35MoEHybridQ4K(ids, wantLogits)
	}
	// Keystone decode lever (W1): the exact resident-Q4_K Qwen3.8 hybrid can run its
	// linear-attention layers through the one-command-buffer fused decode block once the
	// whole-sequence owner has been admitted and its recurrent state promoted. That
	// promotion is only reachable through Enable/Finalize, so a caller that did not opt
	// in (e.g. plain `fak serve --metal` without --native-qwen35-metal-gdn-sequence)
	// silently decoded one grouped GEMV command buffer at a time. Admit the owner here,
	// before any prompt state is mutated, and promote it after the prefill completes.
	// Fail-open: any decline leaves the historical host-recurrence decode untouched.
	// A promoted decode owner set is consulted only by single-token decode; the
	// batched host walk below reads Cache.linear. Hand the live owner state back to
	// the host first (fail closed if it cannot be read), so this prefill continues
	// from the true recurrent state and the auto route may re-admit from it.
	if err := s.demoteQwen35ResidentGDNDecodeToHost(); err != nil {
		panic(err)
	}
	autoAdmitted := false
	if s.qwen35ResidentDecodeAutoEligible() {
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err == nil {
			autoAdmitted = true
		}
	}
	if autoAdmitted {
		// The error is not surfaced on this fail-open route, but it is never
		// silent state loss: Finalize either records sequenceFailure (later
		// snapshots and host syncs fail closed) or has already written the
		// finalized owner state into the host cache before failing.
		defer func() { _, _ = s.FinalizeQwen35MetalGDNPreprojectedSequence() }()
	}
	var hidden []float32
	if s.qwen35HAL != nil && s.qwen35HAL.sequenceAccepted {
		if runner, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalForwardSequenceRunner); ok {
			// The whole-forward graph now witnesses prompt panels up to
			// metalgemm.PromptPanelMaxTokens (256) tokens in ONE command buffer (128 above
			// the Qwen3.5-4B class, see qwen35PromptPanelMaxRows), so the
			// prompt is walked in the widest admitted panels instead of the historical
			// exact-32 panel. The panel count — and therefore the per-panel commit +
			// terminal GPU wait + host KV readback — drops from O(len/32) to
			// O(len/maxRows) (#13041). A prompt of at least one quantum is covered entirely
			// by panels of balanced width (see qwen35PromptPanelEnds), so no panel is
			// narrower than the quantum and no remainder falls to the host forward,
			// which costs a fixed ~1.25 s on Qwen3.5-4B whatever its 1..31 rows.
			//
			// The panel attention has no 4096 cap: mg_qwen35_graph_attention uses
			// qg_attn_online (O(head_dim) ordered online softmax) above 4096 context, so
			// a long prompt stays on the batched panel rather than being forced through
			// the CPU per-token loop. See TestProjectionGraphQwenOrderedLongContextAttention.
			//
			// A prompt under 32 tokens deliberately stays on the host batched path:
			// walking it on the whole-sequence device graph was measured on Qwen3.8-27B
			// to stall past the 10 s command-buffer limit for a 12-token warmup prompt
			// under memory pressure, failing closed with no host retry so the server
			// never became ready (#13694). Panels narrower than the 32-token quantum
			// need their own measured admission.
			panelEnds := qwen35PromptPanelEnds(len(ids), qwen35PromptPanelMaxRows(s.M.Cfg))
			panelCover := 0
			if len(panelEnds) > 0 {
				panelCover = panelEnds[len(panelEnds)-1]
			}
			if panelCover > 0 {
				nPanels := len(panelEnds)
				// #13087 device-resident KV: when the native backend admits it, one
				// device KV triple is allocated for the whole prompt and attached for
				// the panel walk. Every full-attention layer then reads its prefix on
				// the device and appends its new rows there, so the walk pays no
				// per-panel host KV readback or prefix re-upload; the triple is
				// downloaded ONCE after the last panel to restore the host cache. Any
				// decline (pair unavailable, backend without the admitter) falls back
				// to the historical host-append walk with KV/state unmutated.
				var deviceKV *metalgemm.DeviceKV
				var deviceAdmitter qwen35MetalDeviceKVAdmitter
				deviceRetained := false
				deviceBase := s.Cache.Len()
				if admitter, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalDeviceKVAdmitter); ok {
					if kv := admitter.AdmitDeviceKV(s, panelCover); kv != nil {
						deviceAdmitter = admitter
						deviceKV = kv
						s.q4kHybridPrefillDevicePanels = 0
						s.q4kHybridPrefillDeviceRows = 0
					}
				}
				if deviceKV != nil {
					defer func() {
						if deviceRetained {
							return
						}
						if attached := deviceAdmitter.DetachDeviceKV(); attached != nil {
							attached.Close()
						}
					}()
				}
				var agg Qwen35MetalForwardSequenceReceipt
				var aggValid bool
				selectedPanels := nPanels
				executedPanels := 0
				executedRows := 0
				start := 0
				for _, end := range panelEnds {
					panelIDs := ids[start:end]
					start = end
					h, receipt, accepted, err := runner.Qwen35MetalForwardSequence(s, panelIDs)
					if !accepted {
						break
					}
					if err != nil {
						// Distinguish a PRE-commit failure from a POST-commit one by the
						// receipt the backend fills only after its single FinishRead commit
						// (Accepted==true with Available==false is the pre-commit shape; the
						// backend returns its terminal receipt with Available==true once the
						// command buffer has committed). A pre-commit encode/graph failure
						// leaves host KV/Kraw/V and Cache positions unmutated, so it fails
						// OPEN: stop the panel walk and replay the unconsumed rows through
						// the host path. #13473 was exactly this shape reported as
						// accepted==true, so the old unconditional panic killed the serving
						// process on every harness-sized prompt.
						if !receipt.Available {
							break
						}
						// Post-commit: the device GDN state has advanced, so the route
						// cannot be replayed on the host. Fail closed and retire the owner.
						panic(s.failQwen35MetalForwardSequence(err))
					}
					hidden = h
					executedPanels++
					executedRows += len(panelIDs)
					if !aggValid {
						agg = receipt
						agg.SelectedPanels = selectedPanels
						agg.ExecutedPanels = 1
						agg.FallbackCount = 0
						agg.Device = metalgemm.DeviceName()
						agg.SourceRevision = qwen35MetalReceiptSourceRevision
						agg.ArtifactSHA256 = qwen35MetalReceiptArtifactSHA256
						aggValid = true
					} else {
						agg.ExecutedPanels++
						agg.Tokens += receipt.Tokens
						agg.CommandBuffers += receipt.CommandBuffers
						agg.Encoders += receipt.Encoders
						agg.IntermediateWaits += receipt.IntermediateWaits
						agg.IntermediateReadbacks += receipt.IntermediateReadbacks
						agg.TerminalWaits += receipt.TerminalWaits
						agg.TerminalReadbacks += receipt.TerminalReadbacks
						agg.HostUploadBytes += receipt.HostUploadBytes
						agg.HostReadbackBytes += receipt.HostReadbackBytes
						agg.AllocatedBuffers = max(agg.AllocatedBuffers, receipt.AllocatedBuffers)
						agg.RetainedBufferBytes = max(agg.RetainedBufferBytes, receipt.RetainedBufferBytes)
						agg.Committed = agg.Committed && receipt.Committed
						agg.CompletedWait = agg.CompletedWait && receipt.CompletedWait
						agg.TimingAvailable = agg.TimingAvailable && receipt.TimingAvailable
						agg.GPUMilliseconds += receipt.GPUMilliseconds
						agg.WaitMilliseconds += receipt.WaitMilliseconds
						if receipt.StateIdentity != nil {
							agg.StateIdentity = receipt.StateIdentity
						}
					}
				}
				if executedPanels > 0 {
					// The executed panels may cover fewer rows than the planned
					// panelCover when a pre-commit panel failure declines mid-walk
					// (fail-open). Reconcile and replay from the ACTUAL executed
					// rows so the host cache order stays [device rows][host rows]
					// and no token is silently skipped.
					covered := executedRows
					// Reconcile the device-resident KV BEFORE the trailing remainder
					// runs: the triple holds the panel-covered rows for every
					// full-attention layer, and this single download restores the host
					// cache contract the decode path depends on. Doing it first keeps
					// the host cache row order [covered device rows][rem host rows]
					// matching the appended positions. A reconcile failure is fatal
					// (accepted), never a silent partial cache.
					if deviceAdmitter != nil && deviceKV != nil {
						if err := deviceAdmitter.ReconcileDeviceKV(s, deviceBase, covered); err != nil {
							panic(s.failQwen35MetalForwardSequence(err))
						}
						if keeper, ok := deviceAdmitter.(qwen35MetalDeviceKVKeeper); ok {
							deviceRetained = keeper.KeepDeviceKV(s)
						}
						if !deviceRetained {
							if attached := deviceAdmitter.DetachDeviceKV(); attached != nil {
								attached.Close()
							}
						}
					}
					rem := ids[covered:]
					if len(rem) > 0 {
						hidden = s.prefillQwen35HybridQ4KHidden(rem)
						agg.Tokens += len(rem)
					}
					s.qwen35HAL.setMetalForwardReceipt(agg)
					if !wantLogits {
						return nil, true
					}
					return s.headResident(hidden), true
				}
			}
		}
	}
	hidden = s.prefillQwen35HybridQ4KHidden(ids)
	if !wantLogits {
		return nil, true
	}
	return s.headResident(hidden), true
}

// tryPrefillQwen35MoEHybridQ4K is the qwen3_5_moe arm of tryPrefillQwen35HybridQ4K: the
// host batched hybrid panel with the routed FFN per row. It declines (token loop) for any
// session whose observers run only inside blockStep, and for the Metal / backend / HAL
// resident lanes, whose whole-sequence graph and bulk weight upload model only a dense MLP.
func (s *Session) tryPrefillQwen35MoEHybridQ4K(ids []int, wantLogits bool) ([]float32, bool) {
	if s.MetalQ4K || s.Metal || s.Backend != nil || s.qwen35HAL != nil || !s.qwen35HybridPrefillObserversOK() {
		return nil, false
	}
	hidden := s.prefillQwen35HybridQ4KHidden(ids)
	if !wantLogits {
		return nil, true
	}
	return s.headResident(hidden), true
}

func (s *Session) prefillQwen35HybridQ4KHidden(ids []int) []float32 {
	return s.prefillQwen35HybridQ4KHiddenWithNormWorkspace(ids, s.newQwen35PrefillNormWorkspace(len(ids)))
}

func (s *Session) prefillQwen35HybridQ4KHiddenWithNormWorkspace(ids []int, normWorkspace *qwen35PrefillNormWorkspace) []float32 {
	return s.prefillQwen35HybridQ4KHiddenWithGDNMode(ids, normWorkspace, qwen35GDNHostBuildSelected)
}

func (s *Session) prefillQwen35HybridQ4KHiddenWithGDNMode(ids []int, normWorkspace *qwen35PrefillNormWorkspace, gdnMode qwen35GDNHostMode) []float32 {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	P := len(ids)
	if P == 0 {
		return nil
	}
	profile := os.Getenv("FAK_QPROFILE") != ""
	var start time.Time
	if profile {
		start = time.Now()
	}
	var gemmTime time.Duration
	// Sub-buckets of gemmTime, split by which resident kernel actually served each projection,
	// so the next prefill-optimization decision is evidence-based rather than assuming the q4_k
	// Metal GEMM dominates. q4kTime is the q4_k-majority GEMM (Metal q4_k dequant-GEMM under
	// -tags fakmetal + MetalQ4K, else CPU q4kGemm) INCLUDING the grouped one-command-buffer
	// GEMMGroup roundtrip; q8Time is the Q8 minority (full-attn q/k + every linear_attn.*),
	// which on the 36 GiB Mac is OOM-gated onto the CPU qGemm8 path (metalQ8UploadAllowed=false)
	// and is the suspected real prefill wall; q6kTime is the resident Q6_K/Q5_K matmul weights
	// (the q4_k_m dense down_proj / lm_head). q4kTime+q8Time+q6kTime == gemmTime by construction
	// (every timed projection lands in exactly one bucket), so the split is exhaustive and honest.
	var q4kTime, q8Time, q6kTime time.Duration
	// q4kGPUCompute is the on-GPU execute window (cb.GPUEndTime-cb.GPUStartTime, via
	// metalgemm.LastGEMMGPUMs) summed over every q4_k Metal dispatch that lands in q4kTime — the
	// grouped GEMMGroup path (q4kGemmGroupDispatch) and the per-weight GEMM path (q4kGemmDispatch).
	// LastGEMMGPUMs reflects only the MOST RECENT dispatch, so it is read immediately after EACH
	// dispatch and accumulated (never once at the end). q4kRoundtrip = q4kTime - q4kGPUCompute is
	// the wall-time remainder: CPU-side encode/commit/sync + the H2D activation upload. On a
	// non-fakmetal build (or when the q4_k upload declined and the dispatch fell back to CPU
	// q4kGemm) LastGEMMGPUMs returns 0, so q4kGPUCompute stays 0 and q4kRoundtrip == q4kTime — the
	// honest degenerate: no GPU window to attribute, all of q4kTime is wall time.
	var q4kGPUCompute time.Duration
	base := s.Cache.Len()
	eps := float32(cfg.RMSNormEps)

	// One reused Q8 activation-panel scratch for the qGemm8 minority projections (q/k always,
	// every linear_attn.* projection, any Q6_K weight). Each panel is fully consumed before
	// the next is built, so a single buffer is safe — same discipline as prefillBatchedQ4K.
	scratch := &q8Panel{}
	qz := func(X []float32, rows, width int, names ...string) *q8Panel {
		// Only unobserved CPU groups whose every consumer ignores the original
		// Q8 panel may skip it. Needed panels retain the same reused scratch.
		if !profile && s.qwen35PrefillQ8PanelUnused(X, rows, width, names) {
			return nil
		}
		t := s.phaseStart()
		quantizeBatchPanelInto(scratch, X, rows, width)
		s.phaseEnd("q8_panel_quantize", t)
		return scratch
	}
	// Every consumer below reads Xrot without retaining or mutating it. A
	// validated unobserved CPU group may supply one group-local transform.
	projectRotated := func(name string, Xrot []float32, Xq *q8Panel) []float32 {
		if qt := m.q4kw[name]; qt != nil {
			// q4kGemmDispatch is the CPU q4kGemm by default; under -tags fakmetal with
			// s.MetalQ4K set it routes the q4_k-majority GEMM to the Metal q4_k dequant-GEMM.
			return s.q4kGemmDispatch(name, qt, Xrot, P)
		}
		if qt := m.kqw[name]; qt != nil {
			// Resident Q5_K/Q6_K matmul weight (the q4_k_m dense down_proj / lm_head now load
			// Q6_K into kqw, not the Q8 store). Without this branch m.q8(name) below would panic
			// ("q8 tensor not built") — the prefill twin of the decode-path kqw consultation in
			// sessionQ4KKernel.mul. kQuantMatRowsIntoBatch dequantizes each weight super-block ONCE
			// and dots it against all P token columns (a GEMM), amortizing the expensive Q6_K/Q5_K
			// super-block dequant P-fold instead of the old per-token GEMV loop that re-dequantized
			// every block P times — the #2378 prefill-wall lever (~78% of prefill was this dequant).
			// The dispatch uses the Metal Q6_K GEMM when available; otherwise it is the same CPU
			// kQuantMatRowsIntoBatch loop. Same Y layout (P×qt.out row-major); still lands in the
			// q6kTime profile bucket.
			return s.kQuantGemmDispatch(name, qt, Xrot, P)
		}
		if qt := m.q2w[name]; qt != nil {
			return q2MatRowsBatch(qt, Xrot, P)
		}
		if m.prism != nil && m.prism.weightWidth[name] != 0 {
			panel := &q8Panel{}
			quantizeBatchPanelInto(panel, Xrot, P, m.prism.weightWidth[name])
			Xq = panel
		}
		return s.q8GemmDispatch(name, m.q8(name), Xq)
	}
	proj := func(name string, Xf []float32, Xq *q8Panel) []float32 {
		return projectRotated(name, m.prismProjectPanel(name, Xf, P), Xq)
	}
	if profile {
		rawProj := proj
		proj = func(name string, Xf []float32, Xq *q8Panel) []float32 {
			t0 := time.Now()
			Y := rawProj(name, Xf, Xq)
			dt := time.Since(t0)
			gemmTime += dt
			// Attribute this projection to a sub-bucket by the SAME resident-map dispatch order
			// rawProj (the proj closure above) uses: q4kw first → q4k, else kqw → q6k, else Q8.
			switch {
			case m.q4kw[name] != nil:
				q4kTime += dt
				// The q4kw branch just ran q4kGemmDispatch → one Q4KWeight.GEMM = one Metal
				// command buffer, which freshly stored its on-GPU execute window. Read it right
				// here (most-recent-dispatch semantics) and add to the GPU-compute sub-total. On
				// CPU fallback / non-fakmetal it is 0, so roundtrip absorbs all of this dt.
				q4kGPUCompute += time.Duration(metalgemm.LastGEMMGPUMs() * float64(time.Millisecond))
			case m.kqw[name] != nil, m.q2w[name] != nil:
				q6kTime += dt
			default:
				q8Time += dt
			}
			return Y
		}
	}
	// pgroup runs projections that SHARE one activation panel Xf/Xq (a layer's q/k/v, gate/up, or
	// the GDN in_proj quad) through grouped Metal paths: q4_k-resident members via
	// q4kGemmGroupDispatch and Q8-minority members via q8GemmGroupDispatch. Any nil member (Q6_K,
	// singleton, declined upload, or non-Metal build) is filled by the same per-weight `proj`, so
	// Unobserved builtin CPU groups may instead share one exact Prism transform; all
	// other results are identical to calling proj per name with the same fallback.
	pgroup := func(names []string, Xf []float32, Xq *q8Panel) [][]float32 {
		out := make([][]float32, len(names))
		t0 := time.Now()
		q4out := s.q4kGemmGroupDispatch(names, Xf, P)
		if profile {
			dt := time.Since(t0)
			if q4out != nil {
				gemmTime += dt
				q4kTime += dt
				q4kGPUCompute += time.Duration(metalgemm.LastGEMMGPUMs() * float64(time.Millisecond))
			}
		}
		if q4out != nil {
			for i := range q4out {
				out[i] = q4out[i]
			}
		}
		t0 = time.Now()
		q8out := s.q8GemmGroupDispatch(names, Xq, P)
		if profile {
			dt := time.Since(t0)
			if q8out != nil {
				gemmTime += dt
				q8Time += dt
			}
		}
		if q8out != nil {
			for i := range q8out {
				if out[i] == nil {
					out[i] = q8out[i]
				}
			}
		}
		// The real Bonsai loader shares an immutable Prism state/sign vector
		// across declared same-transform weights. Reuse only this group's Xf,
		// with no persistent cache and no transformed panel retained after return.
		allPending := true
		for _, result := range out {
			if result != nil {
				allPending = false
				break
			}
		}
		if !profile && allPending {
			if rotated := s.prismPrefillGroupPanel(names, Xf, P); rotated != nil {
				for i, name := range names {
					out[i] = projectRotated(name, rotated, Xq)
				}
				return out
			}
		}
		for i, name := range names {
			if out[i] == nil {
				out[i] = proj(name, Xf, Xq) // per-weight fallback (already time-accounted by proj)
			}
		}
		return out
	}

	t := s.phaseStart()
	X := make([]float32, P*H)
	if m.Q2KEmbedding != nil {
		scale := cfg.embedScale()
		for t, id := range ids {
			if err := m.Q2KEmbedding.GatherRow(id, X[t*H:(t+1)*H], scale); err != nil {
				panic(err)
			}
		}
	} else {
		embed := m.embedRows()
		for t, id := range ids {
			copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
			m.prismInverseEmbeddingRow("model.embed_tokens.weight", X[t*H:(t+1)*H])
			scaleEmbedInPlace(X[t*H:(t+1)*H], cfg)
		}
	}
	s.phaseEnd("embed", t)

	s.prefillQwen35HybridQ4KMetalUpload()
	ropeRows := s.newQwen35PrefillRopeRows(base, P)

	for l := 0; l < cfg.NumLayers; l++ {
		// These local builtin producers return fresh, separately owned outputs.
		// The generic callback entry below never grants a disposable qf loan.
		s.prefillQwen35HybridQ4KLayerWithGDNMode(l, X, P, base, eps, proj, pgroup, qz, !s.MetalQ4K && s.Backend == nil, ropeRows, normWorkspace, gdnMode)
	}

	t = s.phaseStart()
	for t := 0; t < P; t++ {
		s.Cache.appendPosition(base+t, ids[t])
	}
	s.phaseEnd("cache_positions", t)
	t = s.phaseStart()
	xf := rmsnormCfg(X[(P-1)*H:P*H], m.tensor("model.norm.weight"), eps, cfg)
	s.phaseEnd("final_norm", t)
	if profile {
		s.profileQwen35HybridQ4KPrefill(P, start, gemmTime, q4kTime, q8Time, q6kTime, q4kGPUCompute)
	}
	s.q4kHybridPrefillChunks++
	s.q4kHybridPrefillLastBase = base
	return xf
}

// prefillQwen35HybridQ4KLayer runs one transformer layer of the batched resident-Q4_K hybrid
// prefill: input RMSNorm, the attention sub-layer (linear-attn or full-attn), the attention
// residual, the post-attention RMSNorm, the SwiGLU MLP, and the MLP residual. It mutates the
// running hidden panel X in place (element writes persist through the shared backing array),
// so it returns nothing — extracted verbatim from prefillQwen35HybridQ4KHidden's layer loop so
// the hot method stays under its line ceiling. The projection dispatch (proj/pgroup/qz) and the
// per-layer f32 math are byte-for-byte identical to the inlined loop body.
func (s *Session) prefillQwen35HybridQ4KLayer(l int, X []float32, P, base int, eps float32, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize) {
	s.prefillQwen35HybridQ4KLayerOwnedQF(l, X, P, base, eps, proj, pgroup, qz, false)
}

// ownsQF is private producer provenance, not an inference about nonnil callbacks.
// Only the local builtin proj/pgroup closures in prefillQwen35HybridQ4KHidden
// grant it. Arbitrary callback callers must use the non-owning wrapper above.
func (s *Session) prefillQwen35HybridQ4KLayerOwnedQF(l int, X []float32, P, base int, eps float32, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize, ownsQF bool) {
	s.prefillQwen35HybridQ4KLayerWithRopeRows(l, X, P, base, eps, proj, pgroup, qz, ownsQF, nil)
}

func (s *Session) prefillQwen35HybridQ4KLayerWithRopeRows(l int, X []float32, P, base int, eps float32, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize, ownsQF bool, ropeRows *qwen35PrefillRopeRows) {
	// Callback entry points never grant reuse of their activation inputs.
	s.prefillQwen35HybridQ4KLayerWithScratch(l, X, P, base, eps, proj, pgroup, qz, ownsQF, ropeRows, nil)
}

func (s *Session) prefillQwen35HybridQ4KLayerWithScratch(l int, X []float32, P, base int, eps float32, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize, ownsQF bool, ropeRows *qwen35PrefillRopeRows, normWorkspace *qwen35PrefillNormWorkspace) {
	s.prefillQwen35HybridQ4KLayerWithGDNMode(l, X, P, base, eps, proj, pgroup, qz, ownsQF, ropeRows, normWorkspace, qwen35GDNHostBuildSelected)
}

func (s *Session) prefillQwen35HybridQ4KLayerWithGDNMode(l int, X []float32, P, base int, eps float32, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize, ownsQF bool, ropeRows *qwen35PrefillRopeRows, normWorkspace *qwen35PrefillNormWorkspace, gdnMode qwen35GDNHostMode) {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	lp := func(str string) string { return layerName(l, str) }
	Xn := normWorkspace.panel(P, H)
	wIn := m.tensor(lp("input_layernorm.weight"))
	t := s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				qwen35PrefillNormIntoDisjoint(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps, cfg)
			} else {
				rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
			}
		}
	})
	s.phaseEnd("input_norm", t)

	var o []float32
	if cfg.isLinearAttnLayer(l) {
		o = s.prefillQwen35LinearLayerQ4KWithHostChunks(l, Xn, P, proj, pgroup, qz, ownsQF && qwen35GDNHostModeAllows(gdnMode))
	} else {
		o = s.prefillQwen35FullAttnLayerQ4KWithRopeRows(l, Xn, P, base, proj, pgroup, qz, ownsQF, ropeRows)
	}
	t = s.phaseStart()
	parFor(len(X), dispatchWorkers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			X[i] += o[i]
		}
	})
	s.phaseEnd("attn_residual", t)

	if cfg.IsMoE() {
		// Routed FFN: each row picks its own top-k experts, so it runs per row through
		// the token loop's exact PreNorm sublayer on the token loop's kernel.
		t = s.phaseStart()
		s.prefillQwen35RowFFNQ(l, X, P)
		s.phaseEnd("qwen35_moe_row_ffn", t)
		return
	}

	Xn2 := normWorkspace.panel(P, H)
	wPost := m.tensor(lp("post_attention_layernorm.weight"))
	t = s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				qwen35PrefillNormIntoDisjoint(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps, cfg)
			} else {
				rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
			}
		}
	})
	s.phaseEnd("post_attn_norm", t)
	I := cfg.IntermediateSize
	gateUpNames := []string{lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight")}
	Xn2q := qz(Xn2, P, H, gateUpNames...)
	t = s.phaseStart()
	gu := pgroup(gateUpNames, Xn2, Xn2q)
	G, U := gu[0], gu[1]
	s.phaseEnd("mlp_gate_up_proj", t)
	for t := 0; t < P; t++ {
		m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
		m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
	}
	t = s.phaseStart()
	parFor(len(G), dispatchWorkers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			G[i] = act(G[i], cfg) * U[i]
		}
	})
	s.phaseEnd("mlp_activation", t)
	t = s.phaseStart()
	downName := lp("mlp.down_proj.weight")
	Down := proj(downName, G, qz(G, P, I, downName))
	s.phaseEnd("mlp_down_proj", t)
	for t := 0; t < P; t++ {
		m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
	}
	t = s.phaseStart()
	parFor(len(X), dispatchWorkers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			X[i] += Down[i]
		}
	})
	s.phaseEnd("mlp_residual", t)
}

// prefillQwen35HybridQ4KMetalUpload bulk-uploads the resident projection weights to the GPU
// before the layer loop when Metal Q4_K is enabled. It is pure non-numeric setup — it moves no
// computation and touches no float value on the prefill's return path — extracted verbatim so
// the hot method stays under its line ceiling. No-op unless s.MetalQ4K is set.
func (s *Session) prefillQwen35HybridQ4KMetalUpload() {
	if !s.MetalQ4K {
		return
	}
	m := s.M
	// Bulk-upload every Q4_K projection to the GPU before the layer loop, exactly as the
	// full-attention batched path does (prefill_q4k.go). Without it the lazy per-weight
	// upload in metalQ4KWeight interleaves an H2D round-trip with the first use of each
	// projection, which caps warm hybrid prefill at ~7x under llama.cpp-Metal (#1113);
	// amortizing all the copies up front restores full prefill speed on the Metal hybrid
	// path the 27B Qwen3.6 takes (#71). No-op on the pure-Go build (stub returns nil).
	m.metalQ4KWeights()
	// Upload the Q8-minority projections too (full-attn q/k and linear_attn.*). Otherwise
	// #1087's Metal Q8 GEMM path would pay one upload inside the first timed projection call.
	m.metalQ8Weights()
}

// profileQwen35HybridQ4KPrefill emits the FAK_QPROFILE prefill telemetry (the three
// [metalprof-*] stderr lines) for prefillQwen35HybridQ4KHidden. It is a pure reporting
// helper: it reads the already-measured per-bucket durations and writes to stderr, touching
// no model state and no value on the prefill's return path — extracted verbatim so the hot
// method stays under its ceiling without moving any computation.
func (s *Session) profileQwen35HybridQ4KPrefill(P int, start time.Time, gemmTime, q4kTime, q8Time, q6kTime, q4kGPUCompute time.Duration) {
	total := time.Since(start)
	rest := total - gemmTime
	if rest < 0 {
		rest = 0
	}
	ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
	fmt.Fprintf(os.Stderr, "[metalprof-hybrid P=%d] total=%.1f  gemm+roundtrip=%.1f  rest(recurrence/attn/norm)=%.1f ms path=q4k\n",
		P, ms(total), ms(gemmTime), ms(rest))
	// Split the gemm+roundtrip bucket by which resident kernel served each projection. The
	// three buckets sum to gemm+roundtrip; the point is to tell the next session whether the
	// durable prefill lever is the Q8 CPU path (q8_cpu dominates → the OOM-gated qGemm8
	// minority is the wall) or the q4_k Metal kernel (q4k_metal dominates → kernel cleverness
	// like FAK_Q4K_MM is the lever). The mix on the 27B Mac is the whole question (#71, #977).
	fmt.Fprintf(os.Stderr, "[metalprof-split P=%d] q4k_metal=%.1f  q8_cpu=%.1f  q6k=%.1f ms  (sum=gemm+roundtrip=%.1f) path=q4k\n",
		P, ms(q4kTime), ms(q8Time), ms(q6kTime), ms(gemmTime))
	// Split q4k_metal itself into GPU-compute vs roundtrip. q4kGPUCompute is the summed
	// cb.GPUEndTime-cb.GPUStartTime of every q4_k Metal dispatch (grouped + per-weight);
	// q4kRoundtrip is the wall-time remainder (CPU encode/commit/sync + H2D upload). This is
	// the lever question this session answers: if roundtrip dominates, the next lever is
	// upload-caching / command-buffer batching (fewer submit/sync); if gpu_compute dominates,
	// it is fp16-staging / a GPU counter trace / kernel cleverness. Sum-check: gpu_compute +
	// roundtrip == q4k_metal by construction (roundtrip is defined as the remainder), though
	// gpu_compute may carry small slop vs the wall window since the GPU-execute window and the
	// wall window are not perfectly nested (the wall clock also brackets the cgo call boundary).
	q4kRoundtrip := q4kTime - q4kGPUCompute
	if q4kRoundtrip < 0 {
		q4kRoundtrip = 0
	}
	fmt.Fprintf(os.Stderr, "[metalprof-q4ksplit P=%d] q4k_gpu_compute=%.1f  q4k_roundtrip=%.1f ms  (sum=q4k_metal=%.1f) path=q4k\n",
		P, ms(q4kGPUCompute), ms(q4kRoundtrip), ms(q4kTime))
}

func (s *Session) prefillQwen35LinearLayerQ4K(l int, Xn []float32, P int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize) []float32 {
	return s.prefillQwen35LinearLayerQ4KWithHostChunks(l, Xn, P, proj, pgroup, qz, false)
}

// ordinaryHost is only granted by the builtin producer; custom callbacks keep
// the original recurrence even in a chunked-host qualification build.
func (s *Session) prefillQwen35LinearLayerQ4KWithHostChunks(l int, Xn []float32, P int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize, ordinaryHost bool) []float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	nK, nV, kHd, vHd, keyDim, valDim, convDim := cfg.linearAttnDims()
	K := cfg.LinearConvKernelDim
	eps := float32(cfg.RMSNormEps)
	p := func(str string) string { return layerName(l, str) }
	if s.Cache.linear == nil {
		s.Cache.linear = newLinearAttnCache(cfg)
	}
	lst := s.Cache.linear.layer(cfg, l)

	inputNames := []string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
	}
	Xnq := qz(Xn, P, H, inputNames...)
	t := s.phaseStart()
	// The in_proj quad all reads the same post-norm panel Xn → one command buffer for whichever
	// members are q4_k-resident (in a q4_k_m Qwen3.6 the linear_attn.* projections are unpermuted
	// and resolve to Q8, so pgroup falls back to proj for them — harmless, no regression).
	ip := pgroup(inputNames, Xn, Xnq)
	mixed, zAll, bvec, avec := ip[0], ip[1], ip[2], ip[3]
	s.phaseEnd("qwen35_linear_in_proj", t)

	conv := m.tensor(p("linear_attn.conv1d.weight"))
	aLog := m.tensor(p("linear_attn.A_log"))
	dtBias := m.tensor(p("linear_attn.dt_bias"))
	normW := m.tensor(p("linear_attn.norm.weight"))
	if result, accepted, err := s.tryQwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequenceRequest{
		Layer: l, Tokens: P, Mixed: mixed, Z: zAll, B: bvec, A: avec,
		Conv1D: conv, ALog: aLog, DTBias: dtBias, Norm: normW, RMSNormEpsilon: eps,
	}); accepted {
		if err != nil {
			panic(err)
		}
		t = s.phaseStart()
		outName := p("linear_attn.out_proj.weight")
		out := proj(outName, result.Core, qz(result.Core, P, valDim, outName))
		s.phaseEnd("qwen35_linear_out_proj", t)
		return out
	}
	convOut := make([]float32, P*convDim)
	hist := lst.conv
	t = s.phaseStart()
	if K != 4 || !qwen35PrefillConv4(convOut, mixed, conv, hist, P, convDim, dispatchWorkers) {
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				outRow := convOut[t*convDim : (t+1)*convDim]
				for c := 0; c < convDim; c++ {
					var acc float32
					cb := c * K
					for j := 0; j < K; j++ {
						src := t + j - (K - 1)
						var row []float32
						switch {
						case src >= 0:
							row = mixed[src*convDim : (src+1)*convDim]
						default:
							idx := len(hist) + src
							if idx < 0 {
								continue
							}
							row = hist[idx]
						}
						acc += conv[cb+j] * row[c]
					}
					outRow[c] = silu(acc)
				}
			}
		})
	}
	s.phaseEnd("qwen35_linear_conv", t)
	if !lst.pushConvPrefillTail(mixed, P, convDim, K-1) {
		for t := 0; t < P; t++ {
			lst.pushConvRow(mixed[t*convDim:(t+1)*convDim], K-1)
		}
	}

	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK
	aExp := make([]float32, nV)
	for h := 0; h < nV; h++ {
		aExp[h] = float32(math.Exp(float64(aLog[h])))
	}
	core := make([]float32, P*valDim)
	qNormAll := make([]float32, P*keyDim)
	kNormAll := make([]float32, P*keyDim)
	t = s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := convOut[t*convDim : (t+1)*convDim]
			q := row[0:keyDim]
			k := row[keyDim : 2*keyDim]
			qNorm := qNormAll[t*keyDim : (t+1)*keyDim]
			kNorm := kNormAll[t*keyDim : (t+1)*keyDim]
			for h := 0; h < nK; h++ {
				l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
				l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
				for i := h * kHd; i < (h+1)*kHd; i++ {
					qNorm[i] *= scale
				}
			}
		}
	})
	s.phaseEnd("qwen35_linear_qk_norm", t)
	t = s.phaseStart()
	// As in Q8 prefill, reuse the existing decode transition only under its
	// explicit operator opt-in. Scalar/hardware numeric qualification is still
	// required even for finite operands; this does not promote a default.
	vectorized := os.Getenv("FAK_VECTORIZED_DELTANET") == "1" && HasVectorizedDeltaNetFor(kHd, vHd)
	recurrentPhase := "qwen35_linear_recurrent"
	chunked := false
	if s.qwen35GDNHostSelected(ordinaryHost, vectorized, P) {
		result, ok := qwen35GDNHostCompute(qwen35GDNHostInput{
			tokens: P, keyHeads: nK, valueHeads: nV, keyWidth: kHd, valueWidth: vHd,
			q: qNormAll, k: kNormAll, conv: convOut, a: avec, b: bvec,
			aExp: aExp, dtBias: dtBias, state: lst.recurrent,
		}, dispatchWorkers)
		if ok {
			core = result.core
			for h := range lst.recurrent {
				copy(lst.recurrent[h], result.state[h])
			}
			chunked = true
			recurrentPhase = "qwen35_linear_chunked_host"
		}
	}
	if !chunked {
		parFor(nV, dispatchWorkers, func(lo, hi int) {
			for h := lo; h < hi; h++ {
				kh := h / repeat
				st := lst.recurrent[h]
				a := aExp[h]
				dtB := dtBias[h]
				kvmem := make([]float32, vHd)
				delta := make([]float32, vHd)
				for t := 0; t < P; t++ {
					row := convOut[t*convDim : (t+1)*convDim]
					qn := qNormAll[t*keyDim+kh*kHd : t*keyDim+(kh+1)*kHd]
					kn := kNormAll[t*keyDim+kh*kHd : t*keyDim+(kh+1)*kHd]
					vh := row[2*keyDim+h*vHd : 2*keyDim+(h+1)*vHd]
					bt := sigmoidf(bvec[t*nV+h])
					dt := softplus(avec[t*nV+h] + dtB)
					g := float32(math.Exp(float64(-a * dt)))
					if vectorized && qwen35PrefillVectorizedStep(st, qn, kn, vh, core, kvmem, delta, bt, g, t, valDim, h) {
						continue
					}
					// Private per-head state/scratch and the joined normalized key
					// panel satisfy the same decay/read loan as the Q8 consumer.
					// Refused geometry retains the entire original mutation/panic order.
					if !qwen35PrefillDecayRead(st, kn, kvmem, g, kHd, vHd) {
						for i := range st {
							st[i] *= g
						}
						for d := range kvmem {
							kvmem[d] = 0
						}
						for i := 0; i < kHd; i++ {
							ki := kn[i]
							base := i * vHd
							for d := 0; d < vHd; d++ {
								kvmem[d] = gdnFMA32(st[base+d], ki, kvmem[d])
							}
						}
					}
					for d := 0; d < vHd; d++ {
						delta[d] = (vh[d] - kvmem[d]) * bt
					}
					od := core[t*valDim+h*vHd : t*valDim+(h+1)*vHd]
					for i := 0; i < kHd; i++ {
						ki := kn[i]
						qi := qn[i]
						base := i * vHd
						for d := 0; d < vHd; d++ {
							st[base+d] = gdnFMA32(ki, delta[d], st[base+d])
							od[d] = gdnFMA32(st[base+d], qi, od[d])
						}
					}
				}
			}
		})
	}
	s.phaseEnd(recurrentPhase, t)
	t = s.phaseStart()
	parFor(P*nV, dispatchWorkers, func(lo, hi int) {
		for idx := lo; idx < hi; idx++ {
			t := idx / nV
			h := idx - t*nV
			rmsNormGatedInPlace(
				core[t*valDim+h*vHd:t*valDim+(h+1)*vHd],
				normW,
				zAll[t*valDim+h*vHd:t*valDim+(h+1)*vHd],
				eps,
			)
		}
	})
	s.phaseEnd("qwen35_linear_gated_norm", t)
	t = s.phaseStart()
	outName := p("linear_attn.out_proj.weight")
	O := proj(outName, core, qz(core, P, valDim, outName))
	s.phaseEnd("qwen35_linear_out_proj", t)
	return O
}

func (s *Session) prefillQwen35FullAttnLayerQ4K(l int, Xn []float32, P, base int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize) []float32 {
	return s.prefillQwen35FullAttnLayerQ4KOwnedQF(l, Xn, P, base, proj, pgroup, qz, false)
}

// See prefillQwen35HybridQ4KLayerOwnedQF for the only production loan authority.
func (s *Session) prefillQwen35FullAttnLayerQ4KOwnedQF(l int, Xn []float32, P, base int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize, ownsQF bool) []float32 {
	return s.prefillQwen35FullAttnLayerQ4KWithRopeRows(l, Xn, P, base, proj, pgroup, qz, ownsQF, nil)
}

func (s *Session) prefillQwen35FullAttnLayerQ4KWithRopeRows(l int, Xn []float32, P, base int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz hybridQ4KQuantize, ownsQF bool, ropeRows *qwen35PrefillRopeRows) []float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	qWidth := nH * hd
	w := nKV * hd
	grp := cfg.GroupSize()
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	p := func(str string) string { return layerName(l, str) }
	qkvNames := []string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")}
	Xnq := qz(Xn, P, H, qkvNames...)
	t := s.phaseStart()
	// q/k/v all read the same post-norm panel Xn → one command buffer for the group.
	qkv := pgroup(qkvNames, Xn, Xnq)
	qf, Kp, V := qkv[0], qkv[1], qkv[2]
	s.phaseEnd("qwen35_full_qkv_proj", t)
	Q := make([]float32, P*qWidth)
	gate := make([]float32, P*qWidth)
	t = s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			src := qf[t*2*qWidth : (t+1)*2*qWidth]
			for h := 0; h < nH; h++ {
				copy(Q[t*qWidth+h*hd:t*qWidth+(h+1)*hd], src[h*2*hd:h*2*hd+hd])
				copy(gate[t*qWidth+h*hd:t*qWidth+(h+1)*hd], src[h*2*hd+hd:h*2*hd+2*hd])
			}
		}
	})
	s.phaseEnd("qwen35_full_split_gate", t)
	t = s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			m.applyProjBias(l, Q[t*qWidth:(t+1)*qWidth], Kp[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*qWidth:(t+1)*qWidth], Kp[t*w:(t+1)*w])
		}
	})
	s.Cache.Kraw[l] = append(s.Cache.Kraw[l], Kp...)
	reuseRope := ropeRows.prepare(cfg, l, base, P)
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			var cos, sin []float32
			if reuseRope {
				cos, sin = ropeRows.row(t)
			} else {
				cos, sin = ropeRowForLayer(cfg, l, base+t)
			}
			ropeRowQKInto(Q[t*qWidth:(t+1)*qWidth], Kp[t*w:(t+1)*w], cos, sin, hd, nH, nKV)
		}
	})
	s.Cache.K[l] = append(s.Cache.K[l], Kp...)
	s.Cache.V[l] = append(s.Cache.V[l], V...)
	s.phaseEnd("qwen35_full_qk_norm_rope", t)

	attnOut := make([]float32, P*qWidth)
	t = s.phaseStart()
	// The joined split copied Q and gate out of qf. The builtin CPU producer
	// owns a fresh qf allocation, disjoint from K/V and no longer consumed.
	// Loan only its existing length; the shared helper keeps the old worker
	// count and declines before writes when the loan/grouped work is too small.
	// Custom callback and Metal paths keep the original dispatch unchanged.
	W := cfg.windowForLayer(l)
	tiled := qwen35PrefillTiledAttentionQualification && ownsQF && !s.Metal && !s.MetalQ4K && s.Backend == nil && m.prism == nil &&
		!cfg.Alibi && !m.hasWeight(p("self_attn.sinks")) &&
		qwen35PrefillTiledAttentionInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], qf, P, base, nH, hd, w, grp, W, scale, attnCap, s.M.attnObs)
	if !tiled && (!ownsQF || s.MetalQ4K || s.Backend != nil || !attnPrefillGQA3FDotInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], qf, P, base, nH, hd, w, grp, W, scale, attnCap, s.M.attnObs)) {
		s.attnPrefillDispatch(s.MetalQ4K, attnOut, Q, s.Cache.K[l], s.Cache.V[l], P, base, nH, hd, w, grp, W, l, scale, attnCap, fdot, s.M.attnObs)
	}
	s.phaseEnd("qwen35_full_attn", t)
	t = s.phaseStart()
	qwen35PrefillGateDisjoint(attnOut, gate, P, qWidth)
	s.phaseEnd("qwen35_full_gate", t)
	t = s.phaseStart()
	outName := p("self_attn.o_proj.weight")
	O := proj(outName, attnOut, qz(attnOut, P, qWidth, outName))
	s.phaseEnd("qwen35_full_o_proj", t)
	for t := 0; t < P; t++ {
		m.addBiasIfPresent(O[t*H:(t+1)*H], p("self_attn.o_proj.bias"))
	}
	return O
}
