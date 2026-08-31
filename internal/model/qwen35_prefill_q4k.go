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

// q4kQwen35HybridPrefillOK gates the batched resident-Q4_K hybrid prefill. It is the same
// architecture gate the Q8 hybrid path uses (q8Qwen35HybridPrefillOK) — the resident-Q4_K
// path covers the identical Qwen3.5/3.6 hybrid family; only the projection kernel differs.
func q4kQwen35HybridPrefillOK(cfg Config, promptLen int) bool {
	return q8Qwen35HybridPrefillOK(cfg, promptLen)
}

// q4kQwen35HybridPrefillAtPositionOK preserves the established fresh-prefill
// amortization threshold while admitting a smaller continuation chunk. Once a
// session already owns prefix state, forcing a sub-threshold tail through decode
// would abandon the resident panel path exactly where bounded chunking needs it.
// Reusing the original gate at its threshold preserves every architecture and
// diagnostic escape-hatch check before any cache or recurrent state is touched.
func q4kQwen35HybridPrefillAtPositionOK(cfg Config, promptLen, base int) bool {
	if promptLen <= 0 || base < 0 {
		return false
	}
	if base == 0 {
		return q4kQwen35HybridPrefillOK(cfg, promptLen)
	}
	return q4kQwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt)
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

// Qwen35MetalGDNSequenceForwardPath names the resident-state candidate. It is
// reported only after the native operation ran and its final state was handed
// to the historical decode path.
const Qwen35MetalGDNSequenceForwardPath = "metal/qwen35-gdn-preprojected-sequence-v1"

// The Darwin implementation installs this factory at package initialization.
// The model-only type keeps pure-Go builds free of Darwin/cgo GDNState symbols.
var newQwen35MetalGDNSequenceBackend func() Qwen35GDNPreprojectedSequenceBackend

// Qwen35MetalGDNPreprojectedSequenceAvailable reports whether this build owns
// the native Metal sequence backend. Callers use this capability readback to
// distinguish an unselected supported route from a platform-unavailable route;
// it does not admit a fallback implementation.
func Qwen35MetalGDNPreprojectedSequenceAvailable() bool {
	return newQwen35MetalGDNSequenceBackend != nil
}

// Qwen35MetalSequenceSelectorState is model-authored selection provenance. It
// reflects whether this Session admitted the native sequence owner; callers do
// not supply or override it when the public receipt is assembled.
type Qwen35MetalSequenceSelectorState string

const (
	Qwen35MetalSequenceSelectorOff Qwen35MetalSequenceSelectorState = "off"
	Qwen35MetalSequenceSelectorOn  Qwen35MetalSequenceSelectorState = "on"
)

// Qwen35MetalSequenceEvidenceState distinguishes a truthful zero from a route
// that did not run or cannot run in the current execution envelope.
type Qwen35MetalSequenceEvidenceState string

const (
	Qwen35MetalSequenceEvidenceNotSelected Qwen35MetalSequenceEvidenceState = "not_selected"
	Qwen35MetalSequenceEvidenceUnsupported Qwen35MetalSequenceEvidenceState = "unsupported"
	Qwen35MetalSequenceEvidenceUnavailable Qwen35MetalSequenceEvidenceState = "unavailable"
	Qwen35MetalSequenceEvidenceExecuted    Qwen35MetalSequenceEvidenceState = "executed"
)

type qwen35GDNSequenceSnapshotter interface {
	SnapshotQwen35GDNAuxState(Qwen35GDNAuxState) (conv, recurrent []float32, err error)
}

// Qwen35MetalForwardSequenceReceipt is an immutable value snapshot of the
// model-owned whole-sequence Metal graph. Available is false when that route
// has not produced a terminal receipt; a committed post-submit failure remains
// available so callers can distinguish it from the default path.
type Qwen35MetalForwardSequenceReceipt struct {
	Path                  string                           `json:"path"`
	Available             bool                             `json:"available"`
	SelectorState         Qwen35MetalSequenceSelectorState `json:"selector_state"`
	EvidenceState         Qwen35MetalSequenceEvidenceState `json:"evidence_state"`
	Tokens                int                              `json:"tokens"`
	CommandBuffers        int                              `json:"command_buffers"`
	Encoders              int                              `json:"encoders"`
	IntermediateWaits     int                              `json:"intermediate_waits"`
	IntermediateReadbacks int                              `json:"intermediate_readbacks"`
	TerminalWaits         int                              `json:"terminal_waits"`
	TerminalReadbacks     int                              `json:"terminal_readbacks"`
	HostUploadBytes       uint64                           `json:"host_upload_bytes"`
	HostReadbackBytes     uint64                           `json:"host_readback_bytes"`
	Committed             bool                             `json:"committed"`
	CompletedWait         bool                             `json:"completed_wait"`
	TimingAvailable       bool                             `json:"timing_available"`
	GPUMilliseconds       float64                          `json:"gpu_milliseconds"`
	WaitMilliseconds      float64                          `json:"wait_milliseconds"`
	StateIdentity         *Qwen35MetalStateIdentityReceipt `json:"state_identity,omitempty"`
}

type qwen35MetalForwardSequenceRunner interface {
	Qwen35MetalForwardSequence(*Session, []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error)
	Qwen35MetalForwardSequenceReceipt() (Qwen35MetalForwardSequenceReceipt, bool)
}

type qwen35MetalStateIdentityBinder interface {
	bindQwen35MetalStateIdentity(Qwen35MetalStateIdentityReceipt)
}

// EnableQwen35MetalGDNPreprojectedSequence admits the opt-in owner before any
// prompt state is mutated. Unsupported builds and non-fresh sessions refuse
// explicitly instead of changing the requested path to host recurrence.
func (s *Session) EnableQwen35MetalGDNPreprojectedSequence() error {
	path := Qwen35MetalGDNSequenceForwardPath
	if s == nil || s.M == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "session is not a Qwen hybrid"}
	}
	if s.Backend != nil || !s.Q4K || !s.MetalQ4K {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "requires backend-nil resident-Q4_K Metal session"}
	}
	if s.Cache == nil || s.Cache.Len() != 0 {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "requires a fresh session before prompt-state mutation"}
	}
	if newQwen35MetalGDNSequenceBackend == nil {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "native Metal GDN sequence is unavailable in this build"}
	}
	accepted, err := s.initQwen35GDNPreprojectedSequence(newQwen35MetalGDNSequenceBackend())
	if !accepted && err == nil {
		err = &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "native capability did not advertise the canonical sequence contract"}
	}
	return err
}

// FinalizeQwen35MetalGDNPreprojectedSequence performs the one explicit state
// transfer required by the still-host-owned decode path. All layers are
// snapshotted and validated before any historical cache row is changed.
func (s *Session) FinalizeQwen35MetalGDNPreprojectedSequence() (bool, error) {
	if s == nil || s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted {
		return false, nil
	}
	q := s.qwen35HAL
	if q.sequenceFailure != nil {
		return true, q.sequenceFailure
	}
	snapshotter, ok := q.sequenceBackend.(qwen35GDNSequenceSnapshotter)
	if !ok {
		return true, s.failQwen35GDNSequence(-1, "final state synchronization", fmt.Errorf("admitted backend cannot snapshot auxiliary state"))
	}
	cfg := s.M.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	convElems := (cfg.LinearConvKernelDim - 1) * convDim
	recurrentElems := nV * kHd * vHd
	snapshots := make([]qwen35GDNLayerSnapshot, 0, len(q.sequenceLayers))
	for layer, state := range q.sequenceLayers {
		if !state.valid() {
			continue
		}
		conv, recurrent, err := snapshotter.SnapshotQwen35GDNAuxState(state)
		if err != nil {
			return true, s.failQwen35GDNSequence(layer, "final state synchronization", err)
		}
		if len(conv) != convElems || len(recurrent) != recurrentElems {
			return true, s.failQwen35GDNSequence(layer, "final state shape", fmt.Errorf("conv/recurrent elements=%d/%d, want %d/%d", len(conv), len(recurrent), convElems, recurrentElems))
		}
		snapshots = append(snapshots, qwen35GDNLayerSnapshot{layer: layer, conv: conv, recurrent: recurrent})
	}
	var stateIdentity *Qwen35MetalStateIdentityReceipt
	if s.qwen35MetalStateIdentity != nil {
		if _, ok := q.sequenceBackend.(qwen35GDNStateSeeder); !ok {
			return len(snapshots) > 0, s.failQwen35GDNSequence(-1, "state identity accounting", fmt.Errorf("receipt-opted sequence backend cannot seed the finalized GDN owners"))
		}
		stateBytes := uint64(0)
		for _, snapshot := range snapshots {
			stateBytes += uint64(len(snapshot.conv)+len(snapshot.recurrent)) * 4
		}
		receipt, err := s.buildQwen35MetalStateIdentityReceipt(Qwen35MetalStateAuthoritySequence, snapshots, qwen35MetalStateIdentityAccounting{
			GDNSnapshotOps: len(snapshots), GDNSeedOps: len(snapshots),
			GDNStateD2HBytes: stateBytes, GDNStateH2DBytes: stateBytes,
		})
		if err != nil {
			return len(snapshots) > 0, s.failQwen35GDNSequence(-1, "state identity", err)
		}
		stateIdentity = &receipt
	}
	selected, err := s.promoteQwen35MetalGDNDecode(snapshots)
	if err != nil || selected {
		if err == nil && selected && stateIdentity != nil {
			s.installQwen35MetalStateIdentityReceipt(*stateIdentity)
		}
		return len(snapshots) > 0, err
	}
	// A backend that cannot seed decode owners retains the historical one-time
	// snapshot path. Seed failures return above before this host cache mutates.
	if s.Cache.linear == nil {
		s.Cache.linear = newLinearAttnCache(cfg)
	}
	for _, snapshot := range snapshots {
		state := s.Cache.linear.layer(cfg, snapshot.layer)
		state.conv = make([][]float32, cfg.LinearConvKernelDim-1)
		for row := range state.conv {
			start := row * convDim
			state.conv[row] = append([]float32(nil), snapshot.conv[start:start+convDim]...)
		}
		for head := range state.recurrent {
			start := head * kHd * vHd
			copy(state.recurrent[head], snapshot.recurrent[start:start+kHd*vHd])
		}
	}
	return len(snapshots) > 0, nil
}

// tryPrefillQwen35HybridQ4K is the production selection seam. A declined
// geometry returns before the resident implementation can mutate KV, convolution,
// or recurrent state; the caller retains the historical token-loop behavior.
func (s *Session) tryPrefillQwen35HybridQ4K(ids []int, wantLogits bool) ([]float32, bool) {
	if !q4kQwen35HybridPrefillAtPositionOK(s.M.Cfg, len(ids), s.Cache.Len()) {
		return nil, false
	}
	var hidden []float32
	if s.qwen35HAL != nil && s.qwen35HAL.sequenceAccepted {
		if runner, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalForwardSequenceRunner); ok && len(ids) == 32 {
			var receipt Qwen35MetalForwardSequenceReceipt
			var accepted bool
			var err error
			hidden, receipt, accepted, err = runner.Qwen35MetalForwardSequence(s, ids)
			_ = receipt
			if accepted {
				if err != nil {
					panic(s.failQwen35MetalForwardSequence(err))
				}
				if !wantLogits {
					return nil, true
				}
				return s.headResident(hidden), true
			}
		}
	}
	hidden = s.prefillQwen35HybridQ4KHidden(ids)
	if !wantLogits {
		return nil, true
	}
	return s.headResident(hidden), true
}

func (s *Session) failQwen35MetalForwardSequence(cause error) error {
	err := &Qwen35GDNSequenceOperationError{Layer: -1, Stage: "whole-sequence forward", Cause: cause}
	if s == nil || s.qwen35HAL == nil {
		return err
	}
	q := s.qwen35HAL
	backend := q.sequenceBackend
	states := q.sequenceLayers
	q.sequenceLayers = nil
	for _, state := range states {
		if state.valid() && backend != nil {
			_ = backend.FreeQwen35GDNAuxState(state)
		}
	}
	// Retain only the receipt-bearing backend object so the accepted failure can
	// still be audited. It owns no live state after the loop above.
	q.sequenceAccepted = true
	q.decodeAccepted = false
	q.sequenceFailure = err
	return err
}

// Qwen35MetalForwardSequenceStatus returns model-authored selection and support
// state even when no terminal execution receipt exists. The status is separate
// from Qwen35MetalForwardSequenceReceipt so historical execution-only callers
// retain the zero-value absence contract.
func (s *Session) Qwen35MetalForwardSequenceStatus() Qwen35MetalForwardSequenceReceipt {
	base := Qwen35MetalForwardSequenceReceipt{
		Path:          Qwen35MetalGDNSequenceForwardPath,
		SelectorState: Qwen35MetalSequenceSelectorOff,
		EvidenceState: Qwen35MetalSequenceEvidenceUnsupported,
	}
	if s == nil || s.M == nil || !s.M.Cfg.IsQwen35Hybrid() || s.Backend != nil || !s.Q4K || !s.MetalQ4K {
		return base
	}
	if newQwen35MetalGDNSequenceBackend == nil {
		base.EvidenceState = Qwen35MetalSequenceEvidenceUnavailable
		return base
	}
	base.EvidenceState = Qwen35MetalSequenceEvidenceNotSelected
	if s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted && !s.qwen35HAL.decodeAccepted {
		return base
	}
	base.SelectorState = Qwen35MetalSequenceSelectorOn
	base.EvidenceState = Qwen35MetalSequenceEvidenceUnavailable
	runner, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalForwardSequenceRunner)
	if !ok {
		return base
	}
	receipt, ok := runner.Qwen35MetalForwardSequenceReceipt()
	if !ok {
		return base
	}
	receipt.SelectorState = Qwen35MetalSequenceSelectorOn
	receipt.EvidenceState = Qwen35MetalSequenceEvidenceExecuted
	return receipt
}

// Qwen35MetalForwardSequenceReceipt returns the last terminal whole-sequence
// receipt as a scalar-only value snapshot. The zero value has Available=false
// and means this session has no such execution evidence.
func (s *Session) Qwen35MetalForwardSequenceReceipt() Qwen35MetalForwardSequenceReceipt {
	receipt := s.Qwen35MetalForwardSequenceStatus()
	if receipt.EvidenceState != Qwen35MetalSequenceEvidenceExecuted {
		return Qwen35MetalForwardSequenceReceipt{}
	}
	return receipt
}

func (s *Session) prefillQwen35HybridQ4KHidden(ids []int) []float32 {
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
	qz := func(X []float32, rows, width int) *q8Panel {
		t := s.phaseStart()
		quantizeBatchPanelInto(scratch, X, rows, width)
		s.phaseEnd("q8_panel_quantize", t)
		return scratch
	}
	proj := func(name string, Xf []float32, Xq *q8Panel) []float32 {
		if qt := m.q4kw[name]; qt != nil {
			// q4kGemmDispatch is the CPU q4kGemm by default; under -tags fakmetal with
			// s.MetalQ4K set it routes the q4_k-majority GEMM to the Metal q4_k dequant-GEMM.
			return s.q4kGemmDispatch(name, qt, Xf, P)
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
			return s.kQuantGemmDispatch(name, qt, Xf, P)
		}
		return s.q8GemmDispatch(name, m.q8(name), Xq)
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
			case m.kqw[name] != nil:
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
	// the result is identical to calling proj per name — just fewer command buffers.
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
		for i, name := range names {
			if out[i] == nil {
				out[i] = proj(name, Xf, Xq) // per-weight fallback (already time-accounted by proj)
			}
		}
		return out
	}

	t := s.phaseStart()
	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg)
	}
	s.phaseEnd("embed", t)

	s.prefillQwen35HybridQ4KMetalUpload()

	for l := 0; l < cfg.NumLayers; l++ {
		s.prefillQwen35HybridQ4KLayer(l, X, P, base, eps, proj, pgroup, qz)
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
func (s *Session) prefillQwen35HybridQ4KLayer(l int, X []float32, P, base int, eps float32, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz func([]float32, int, int) *q8Panel) {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	lp := func(str string) string { return layerName(l, str) }
	Xn := make([]float32, P*H)
	wIn := m.tensor(lp("input_layernorm.weight"))
	t := s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				copy(Xn[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wIn, eps, cfg))
			} else {
				rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
			}
		}
	})
	s.phaseEnd("input_norm", t)

	var o []float32
	if cfg.isLinearAttnLayer(l) {
		o = s.prefillQwen35LinearLayerQ4K(l, Xn, P, proj, pgroup, qz)
	} else {
		o = s.prefillQwen35FullAttnLayerQ4K(l, Xn, P, base, proj, pgroup, qz)
	}
	t = s.phaseStart()
	parFor(len(X), dispatchWorkers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			X[i] += o[i]
		}
	})
	s.phaseEnd("attn_residual", t)

	Xn2 := make([]float32, P*H)
	wPost := m.tensor(lp("post_attention_layernorm.weight"))
	t = s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				copy(Xn2[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wPost, eps, cfg))
			} else {
				rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
			}
		}
	})
	s.phaseEnd("post_attn_norm", t)
	I := cfg.IntermediateSize
	Xn2q := qz(Xn2, P, H)
	t = s.phaseStart()
	gu := pgroup([]string{lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight")}, Xn2, Xn2q)
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
	Down := proj(lp("mlp.down_proj.weight"), G, qz(G, P, I))
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

func (s *Session) prefillQwen35LinearLayerQ4K(l int, Xn []float32, P int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz func([]float32, int, int) *q8Panel) []float32 {
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

	Xnq := qz(Xn, P, H)
	t := s.phaseStart()
	// The in_proj quad all reads the same post-norm panel Xn → one command buffer for whichever
	// members are q4_k-resident (in a q4_k_m Qwen3.6 the linear_attn.* projections are unpermuted
	// and resolve to Q8, so pgroup falls back to proj for them — harmless, no regression).
	ip := pgroup([]string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
	}, Xn, Xnq)
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
		out := proj(p("linear_attn.out_proj.weight"), result.Core, qz(result.Core, P, valDim))
		s.phaseEnd("qwen35_linear_out_proj", t)
		return out
	}
	convOut := make([]float32, P*convDim)
	hist := lst.conv
	t = s.phaseStart()
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
	s.phaseEnd("qwen35_linear_conv", t)
	for t := 0; t < P; t++ {
		lst.pushConvRow(mixed[t*convDim:(t+1)*convDim], K-1)
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
						kvmem[d] += st[base+d] * ki
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
						st[base+d] += ki * delta[d]
						od[d] += st[base+d] * qi
					}
				}
			}
		}
	})
	s.phaseEnd("qwen35_linear_recurrent", t)
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
	O := proj(p("linear_attn.out_proj.weight"), core, qz(core, P, valDim))
	s.phaseEnd("qwen35_linear_out_proj", t)
	return O
}

func (s *Session) prefillQwen35FullAttnLayerQ4K(l int, Xn []float32, P, base int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz func([]float32, int, int) *q8Panel) []float32 {
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
	Xnq := qz(Xn, P, H)
	t := s.phaseStart()
	// q/k/v all read the same post-norm panel Xn → one command buffer for the group.
	qkv := pgroup([]string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")}, Xn, Xnq)
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
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			cos, sin := ropeRowForLayer(cfg, l, base+t)
			ropeRowQKInto(Q[t*qWidth:(t+1)*qWidth], Kp[t*w:(t+1)*w], cos, sin, hd, nH, nKV)
		}
	})
	s.Cache.K[l] = append(s.Cache.K[l], Kp...)
	s.Cache.V[l] = append(s.Cache.V[l], V...)
	s.phaseEnd("qwen35_full_qk_norm_rope", t)

	attnOut := make([]float32, P*qWidth)
	t = s.phaseStart()
	attnPrefillInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, fdot, s.M.attnObs)
	s.phaseEnd("qwen35_full_attn", t)
	t = s.phaseStart()
	for i := range attnOut {
		attnOut[i] *= sigmoidf(gate[i])
	}
	s.phaseEnd("qwen35_full_gate", t)
	t = s.phaseStart()
	O := proj(p("self_attn.o_proj.weight"), attnOut, qz(attnOut, P, qWidth))
	s.phaseEnd("qwen35_full_o_proj", t)
	for t := 0; t < P; t++ {
		m.addBiasIfPresent(O[t*H:(t+1)*H], p("self_attn.o_proj.bias"))
	}
	return O
}
