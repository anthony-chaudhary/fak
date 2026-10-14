package model

// v41_forward.go assembles the DeepSeek V4.1 text forward behind new entry
// points (issue #12901, leaf of parent #12640). It composes the already-landed
// V4.1 leaves — the projected-input shared attention state (v41_attention_state.go),
// the mHC coefficient split/mix (v41_mhc.go), the routed+shared router
// (v41_router.go), the sparse sink attention contraction + grouped output
// projection (v41_sparse_attention.go) — into one ordered pass: embedding ->
// per-layer [mHC + attention + MoE] -> final norm -> head logits.
//
// Scope (gold-plating boundary from #12901): TEXT-forward assembly only. No GPU
// qualification, no GGUF/checkpoint loading, no streaming, no vision, no DSpark.
// Engram packed-row retrieval is wired as of #13007 (v41_forward_engram.go):
// a declared in-range Engram layer is admitted only when the model carries a
// packed-row stage and the three mixing tensors, and the stage is injected at
// the START of the layer per the reference schedule. That is fail-closed: a
// config declaring an in-range Engram layer WITHOUT a wired row source is
// refused at admission by v41EngramForwardAdmitted with an error wrapping
// ErrV41NativeUnsupported, so the assembly never emits non-Engram logits for a
// model it did not fully run. The same holds
// for a declared shared-KV source layer (v41KVSourceForwardAdmitted) and a
// declared compressor/indexer layer (v41CompressIndexForwardAdmitted): the
// reduced assembly projects a per-layer attn.wkv.weight, executes neither
// compression stage, and never reuses a shared KV/index source, so an in-range
// declaration is refused rather than silently run against a per-layer cache. The
// blocked-candidate selection the lightning indexer reads is not executed; it is
// the identity below its pool bound, so a forward past that bound is refused
// (v41CandidatePoolCovers) rather than silently run as unblocked attention. The
// mHC hyperconnection multiplicity is likewise fixed at the published four-stream
// layout (v41HCMultForwardAdmitted), so a declared hc_mult other than 4 is refused
// rather than silently run as four streams. The assembled
// pass is exercised by a REDUCED model
// against an independent scalar oracle (v41_forward_test.go); it does not itself
// qualify the official checkpoint for generation.
//
// Fail-closed design. Every stage is gated by v41ForwardAdmitted before any math
// runs, and forwardV41 re-checks each stage as it goes, returning a typed
// *V41ForwardError. A weightless V4.1 *Model (the probe in
// v41_failclosed_test.go) has no manifest, so admission fails on the embedding
// stage with an error that WRAPS ErrV41NativeUnsupported — keeping
// errors.Is(err, ErrV41NativeUnsupported) true for that fence test — while a
// fully-populated reduced model proceeds. A loaded model missing exactly one
// stage fails with an error that wraps ErrV41ForwardStage. forwardV41 NEVER
// returns logits when a stage is missing.

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrV41ForwardStage reports that a required V4.1 forward stage could not be
// assembled (absent weight, inconsistent shape, or missing stream/state). It is
// the closed class every stage-specific V41ForwardError wraps.
var ErrV41ForwardStage = errors.New("model: DeepSeek V4.1 forward stage is not implemented")

// v41ForwardStage names the ordered assembly stage a failure occurred at.
type v41ForwardStage string

const (
	v41StageEmbedding v41ForwardStage = "embedding"
	v41StageLayer     v41ForwardStage = "layer"
	v41StageMHC       v41ForwardStage = "mhc"
	v41StageAttention v41ForwardStage = "attention"
	v41StageMoE       v41ForwardStage = "moe"
	v41StageEngram    v41ForwardStage = "engram"
	v41StageFinalNorm v41ForwardStage = "final_norm"
	v41StageHead      v41ForwardStage = "head"
	v41StageCompress  v41ForwardStage = "compress"
	v41StageIndexer   v41ForwardStage = "indexer"
)

// v41StageCompress and v41StageIndexer are the stage names for the CED/CSA2
// compressor and the lightning indexer. As of #13006 the reduced text forward
// EXECUTES both stages: the compressor pools a compressed layer's projected KV
// rows (v41CompressedRows), and the lightning indexer scores a declared index
// source against those compressed keys and selects rows (v41IndexRows). A config
// declaring an in-range compressor or indexer layer without its weights is still
// refused at admission by v41CompressIndexForwardAdmitted; a malformed schedule
// fails closed there too.

// V41ForwardError is the typed fail-closed error for one V4.1 assembly stage.
// Layer is the zero-based decoder layer the stage belongs to, or -1 for a
// model-global stage (embedding, final norm, head).
type V41ForwardError struct {
	Stage v41ForwardStage
	Layer int
	Err   error
}

func (e *V41ForwardError) Error() string {
	if e == nil {
		return "model: nil V4.1 forward error"
	}
	where := ""
	if e.Layer >= 0 {
		where = fmt.Sprintf(" layer=%d", e.Layer)
	}
	if e.Err != nil {
		return fmt.Sprintf("model: V4.1 forward stage %s%s: %v", e.Stage, where, e.Err)
	}
	return fmt.Sprintf("model: V4.1 forward stage %s%s failed", e.Stage, where)
}

// Unwrap exposes the wrapped cause so errors.Is reaches ErrV41ForwardStage or,
// for the weightless probe, ErrV41NativeUnsupported.
func (e *V41ForwardError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func v41StageErr(stage v41ForwardStage, layer int, err error) error {
	if err == nil {
		err = ErrV41ForwardStage
	}
	return &V41ForwardError{Stage: stage, Layer: layer, Err: err}
}

// v41ForwardState is the session-owned continuation state for the reduced V4.1
// text forward. It carries committed token history and one bounded temporal
// attention state per decoder layer. attn is the per-forward registry through
// which source layers publish completed rows and selections to later readers.
// Session.Step still recomputes full history; these states are seeded for later
// incremental leaves and do not activate incremental arithmetic.
type v41ForwardState struct {
	history []int
	attn    *V41AttentionState
	layers  []*V41AttentionState
	// forwardCoverage is replaced only after a successful logits-producing
	// traversal and is intentionally absent from prefix snapshots.
	forwardCoverage *NativeForwardCoverage

	// indexKeyMemo is the session's per-index-source-layer memo of the last
	// committed fold; indexKeyPrior is the read-only copy a re-fold run reuses.
	indexKeyMemo  map[int][]v41IndexKeyMemoRow
	indexKeyPrior map[int][]v41IndexKeyMemoRow

	// expertGateUp is the OPTIONAL device gate/up callback the session installs
	// when its backend can run a routed expert's gate/up projections plus SwiGLU
	// (the shared q4kExpertInputHALWithLimit operation, bound in
	// Session.v41State). When non-nil the MoE loop offers each pick to it first:
	// on a handled result the I-wide intermediate uses device projections and
	// the configured clamp (bounded host projection rows when needed), and
	// only the existing host down contraction runs over it, so the gate/up f32
	// weights are never materialized on the host. A nil callback (Model.Forward,
	// a non-device session, or a backend the shared operation declines) preserves
	// the historical host triple byte-for-byte.
	expertGateUp v41ExpertGateUpFunc

	// expertDown is the OPTIONAL device down callback and the symmetric partner of
	// expertGateUp. The session installs it only when the backend can run a routed
	// expert's down projection on device for the resolved encoding (the shared
	// q4kExpertDownDeviceWeight operation, bound in Session.v41State). When both
	// callbacks are non-nil and handled, all three expert projections execute on the
	// backend and no host expert GEMM remains — the requirement for the pinned Q2_K
	// streamed route to satisfy the Halo GPU-only guard. A nil callback (or any
	// decline) preserves the historical host f32 down contraction byte-for-byte.
	expertDown       v41ExpertDownFunc
	projectionRoute  *v41ProjectionRouteBinding
	groupedOutput    v41GroupedOutputFunc
	engramProjection v41EngramProjectionFunc
	mhcProjection    v41MHCProjectionFunc
	mhcFFNProjection v41MHCProjectionFunc
	finalNorm        v41FinalNormFunc
	bf16Boundary     v41BF16BoundaryFunc
	logitScale       v41LogitScaleFunc
	attentionNorm    v41AttentionInputNormFunc
	attentionBF16    v41BF16BoundaryFunc
	queryNorm        v41QueryNormFunc
	queryPrefix      v41QueryPrefixFunc
	kvNorm           v41KVNormFunc
	ffnNorm          v41FFNNormFunc
	compressorNorm   v41CompressorNormFunc
	indexKeyNorm     v41IndexKeyNormFunc
	indexerScore     v41IndexerScoreFunc
	indexScoreHealth v41IndexerScoreHealthFunc
	sharedActivation v41SharedActivationFunc
	tailRoPE         v41TailRoPEFunc
	sharedAttention  v41SharedAttentionFunc
	routedSum        v41RoutedSumFunc
	sharedAdd        v41SharedAddFunc
	gatherOps        *v41DeviceGatherOps
	// routedExpert is the OPTIONAL fused device routed expert (gate/up, BF16
	// clamp-SwiGLU route weight, down). The full profile offers each pick to it
	// before expertGateUp/expertDown; nil keeps those seams unchanged.
	routedExpert v41RoutedExpertFunc
	// routedLayer is the OPTIONAL batched decode-layer routed seam (#13820);
	// nil keeps the per-pick loop.
	routedLayer v41RoutedLayerFunc
	// routedPrefill binds full-forward EP work separately from incremental Step.
	// It consumes original absolute positions and is never snapshot data.
	routedPrefill v41RoutedPrefillFunc
	// mhcDevice is the session's mHC stream seam (initial streams, split,
	// pre-collapse, post-mix). nil keeps the host reference; a non-nil seam
	// dispatches every full-geometry mHC op on the backend or refuses.
	mhcDevice     *v41MHCDevice
	callbackOwner *Session
}

// v41ExpertGateUpOutcome is the closed result vocabulary of one v41ExpertGateUpFunc
// call. handled-success returns the I-wide fused intermediate for the existing host
// down contraction; declined leaves the pick to the historical host gate/up/down
// triple; handled-error is a SELECTED execution failure and must surface, never be
// swallowed as a decline.
type v41ExpertGateUpOutcome uint8

const (
	// v41GateUpDeclined: the shared device operation did not admit this expert, so
	// the caller falls through to the host triple byte-for-byte.
	v41GateUpDeclined v41ExpertGateUpOutcome = iota
	// v41GateUpHandled: gate/up MatMuls ran on the backend, and the returned
	// slice is the I-wide intermediate after the configured SwiGLU activation.
	v41GateUpHandled
	// v41GateUpError: a selected device execution failed; it must remain visible.
	v41GateUpError
)

// v41ExpertGateUpFunc is the optional per-pick device gate/up operation. It
// receives the routed expert's layer stem and the normalized input, and reports
// one of the closed v41ExpertGateUpOutcome values. On v41GateUpHandled it returns
// the I-wide fused intermediate from device gate/up projections and the
// configured SwiGLU activation, sized to the expert intermediate width. Full
// V4.1 bindings publish BF16 gate/up operands before F32 host SwiGLU; the
// returned intermediate remains F32 until its caller applies the route weight.
type v41ExpertGateUpFunc func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error)

// v41DeviceOnlyGateUp is the device-only gate/up wrapper (see
// v41BindDeviceOnlyDeclineRefusals): an unbound or declining seam refuses typed.
func v41DeviceOnlyGateUp(s *Session, f v41ExpertGateUpFunc) v41ExpertGateUpFunc {
	return func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, stem, xn)
			if outcome != v41GateUpDeclined {
				return y, outcome, err
			}
		}
		return nil, v41GateUpError, s.v41DeviceOnlyDeclined(layer, "routed-expert-gate-up")
	}
}

// v41ExpertDownOutcome is the closed result vocabulary of one v41ExpertDownFunc
// call. handled-success returns the H-wide expert output with the down projection
// computed on the backend; declined leaves the caller on the historical host
// f32 down contraction (hostExpertDown + matRows); handled-error is a SELECTED
// execution failure and must surface, never be swallowed as a decline.
type v41ExpertDownOutcome uint8

const (
	// v41DownDeclined: no device kernel resolved for this expert's down encoding,
	// so the caller falls through to the host down contraction byte-for-byte.
	v41DownDeclined v41ExpertDownOutcome = iota
	// v41DownHandled: the down MatMul ran on the backend and the returned slice is
	// the H-wide expert output.
	v41DownHandled
	// v41DownError: a selected device execution failed; it must remain visible.
	v41DownError
)

// v41ExpertDownFunc is the optional device down-projection operation, the
// symmetric partner of v41ExpertGateUpFunc (#13358/#13511 moved gate/up to the
// device; without this the down projection — the last expert GEMM — still ran on
// the host, so a streamed serve that fits only via host expert placement could
// never satisfy the GPU-only guard, #13668/#13128). It receives the routed
// expert's layer stem and the I-wide fused intermediate the gate/up seam
// produced, and reports one of the closed v41ExpertDownOutcome values. On
// v41DownHandled it returns the H-wide expert output computed on the backend.
// Full V4.1 supplies the already-weighted BF16 operand and owns a BF16 result
// before the device tensor is freed; reduced bindings retain their F32 contract.
type v41ExpertDownFunc func(layer int, stem string, fused []float32) ([]float32, v41ExpertDownOutcome, error)

func (st *v41ForwardState) attentionState(headDim, ratioCap int, indexHeadDim ...int) (*V41AttentionState, error) {
	if st.attn != nil {
		return st.attn, nil
	}
	state, err := NewV41AttentionState(headDim, ratioCap)
	if err != nil {
		return nil, err
	}
	if len(indexHeadDim) > 0 && indexHeadDim[0] > 0 {
		state.indexHeadDim = indexHeadDim[0]
	}
	st.attn = state
	return state, nil
}

func (st *v41ForwardState) layerState(layer int) *V41AttentionState {
	if st == nil || layer < 0 || layer >= len(st.layers) {
		return nil
	}
	return st.layers[layer]
}

func (st *v41ForwardState) setLayerState(layer, count int, state *V41AttentionState) {
	if len(st.layers) != count {
		st.layers = make([]*V41AttentionState, count)
	}
	st.layers[layer] = state
}

func (st *v41ForwardState) retainedCopyCount() int {
	if st == nil {
		return 0
	}
	total := 0
	for _, layer := range st.layers {
		if layer != nil {
			total += layer.retainedCopies
		}
	}
	return total
}

// ---- reduced V4.1 geometry helpers -----------------------------------------

// ---- admission -------------------------------------------------------------

// ---- the ordered assembly --------------------------------------------------

// forwardV41 runs the reduced V4.1 text forward over ids (or the state's full
// token history when st is non-nil) and returns per-position hidden states and
// logits. Every stage is checked; a missing stage returns a typed error and no
// logits. It is package-private: the public entry points are Model.Forward and
// Session.Prefill/Step.
func (m *Model) forwardV41(ids []int, st *v41ForwardState) (act *Activations, err error) {
	return m.forwardV41WithHeadSelection(ids, st, false)
}

func (m *Model) forwardV41WithHeadSelection(ids []int, st *v41ForwardState, lastHeadOnly bool) (act *Activations, err error) {
	if m.v41EPRequested() && (st == nil || st.callbackOwner == nil || st.routedLayer == nil || st.routedPrefill == nil) {
		return nil, fmt.Errorf("%w: requires the bound Session forward", ErrV41ExpertParallel)
	}
	if st != nil {
		st.forwardCoverage = nil
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	// Model.Forward (st == nil) is a pure full-prefill over ids. A session folds
	// ids into its history and recomputes the whole history so Step is consistent
	// with a single longer Forward.
	seq := ids
	runState := st
	committed := false
	if st != nil {
		historyLen := len(st.history)
		st.history = append(st.history, ids...)
		seq = st.history
		// The step-local run state carries the session-owned device gate/up callback
		// (#13358) so the MoE loop offers each pick to the device seam. It is not
		// step-local continuation data and is never written back below.
		runState = &v41ForwardState{history: seq, expertGateUp: st.expertGateUp, expertDown: st.expertDown, projectionRoute: st.projectionRoute, groupedOutput: st.groupedOutput, engramProjection: st.engramProjection, mhcProjection: st.mhcProjection, mhcFFNProjection: st.mhcFFNProjection, finalNorm: st.finalNorm, bf16Boundary: st.bf16Boundary, logitScale: st.logitScale, attentionNorm: st.attentionNorm, attentionBF16: st.attentionBF16, queryNorm: st.queryNorm, queryPrefix: st.queryPrefix, kvNorm: st.kvNorm, ffnNorm: st.ffnNorm, compressorNorm: st.compressorNorm, indexKeyNorm: st.indexKeyNorm, indexerScore: st.indexerScore, indexScoreHealth: st.indexScoreHealth, sharedActivation: st.sharedActivation, tailRoPE: st.tailRoPE, sharedAttention: st.sharedAttention, gatherOps: st.gatherOps, routedSum: st.routedSum, sharedAdd: st.sharedAdd, routedExpert: st.routedExpert, routedLayer: st.routedLayer, routedPrefill: st.routedPrefill, mhcDevice: st.mhcDevice, callbackOwner: st.callbackOwner}
		runState.indexKeyPrior = st.indexKeyMemo
		runState.indexKeyMemo = map[int][]v41IndexKeyMemoRow{}
		defer func() {
			if !committed {
				st.history = st.history[:historyLen]
			}
		}()
	}
	// A stateless cold forward still needs a transient cross-layer source registry.
	if runState == nil && m.v41RoleSchedule() {
		runState = &v41ForwardState{}
	}
	if len(seq) == 0 {
		if st != nil {
			committed = true
		}
		return &Activations{}, nil
	}
	if err := v41CandidatePoolCovers(cfg, len(seq)); err != nil {
		return nil, err
	}
	for _, id := range seq {
		if id < 0 || id >= cfg.VocabSize {
			return nil, v41StageErr(v41StageEmbedding, -1,
				fmt.Errorf("%w: token id %d out of range [0,%d)", ErrV41ForwardStage, id, cfg.VocabSize))
		}
	}

	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)

	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}
	var mhcDevice *v41MHCDevice
	if runState != nil {
		mhcDevice = runState.mhcDevice
	}
	if !full {
		if err := mhcDevice.refuseReduced(); err != nil {
			return nil, err
		}
	}

	// ---- embedding ----
	embedded, err := m.v41EmbeddingPanelOn(seq, runState.deviceGatherOps())
	if err != nil {
		return nil, err
	}
	x := make([][]float32, len(seq))
	for t := range seq {
		x[t] = embedded[t*H : (t+1)*H : (t+1)*H]
	}

	// The official full graph repeats the BF16 embedding into four distinct
	// allocations; the reduced graph retains its legacy initialization.
	streams := make([][][]float32, len(seq))
	var resident *v41MHCResident
	if full && mhcDevice.resident() {
		if resident, err = mhcDevice.newResident(x); err != nil {
			return nil, err
		}
		defer resident.free()
		zero, err := resident.streamZero()
		if err != nil {
			return nil, err
		}
		for t := range x {
			copy(x[t], zero[t])
		}
	}
	for t := range streams {
		if resident != nil {
			continue
		}
		if full {
			streams[t], err = v41FullInitialStreamsOn(mhcDevice, x[t])
			if err != nil {
				return nil, err
			}
			copy(x[t], streams[t][0])
		} else {
			streams[t] = [][]float32{x[t], make([]float32, H), make([]float32, H), make([]float32, H)}
		}
	}

	hcIters := 1
	hcEps := float32(1e-6)
	if cfg.DeepSeekV41 != nil {
		if cfg.DeepSeekV41.HCSinkhornIters > 0 {
			hcIters = cfg.DeepSeekV41.HCSinkhornIters
		}
		if cfg.DeepSeekV41.HCEps > 0 {
			hcEps = float32(cfg.DeepSeekV41.HCEps)
		}
	}
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return nil, err
	}

	act = &Activations{Seq: len(seq), Hidden: [][]float32{flatten(x)}}
	// One scratch for the whole forward: the grouped output projections are read
	// as whole f32 blocks but the layer loop is sequential, so a single reused
	// buffer bounds the wo_a/wo_b term to one layer's worth instead of the
	// 40-layer accumulated churn that OOM-killed the warmup (#13288).
	scratch := &v41ProjScratch{}
	if full {
		scratch.mhcCarry = newV41MHCCarry(len(seq))
		scratch.mhcCarry.device = mhcDevice
		scratch.mhcCarry.resident = resident
	}
	if finish := m.v41ActivationTraceFromEnv(seq); finish != nil {
		defer func() { err = finish(err) }()
	}
	// #13325: resolve the optional activation-checkpoint producer ONCE per
	// forward. Disabled (the default) it is nil and every emit is skipped, so the
	// forward is byte-for-byte unchanged and adds no allocations.
	trace := m.v41ActivationTracer(seq)
	executedLayers := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41Layer(l, seq, x, streams, full, hd, nH, H, eps, hcIters, hcEps, routeCfg, runState, scratch, trace); err != nil {
			return nil, err
		}
		executedLayers++
		if err := trace.recordLayerStreams(l, streams, resident); err != nil {
			return nil, err
		}
		hidden := flatten(x)
		if resident != nil {
			rows, err := resident.hiddenRows(l)
			if err != nil {
				return nil, err
			}
			hidden = make([]float32, 0, len(seq)*H)
			for _, row := range rows {
				hidden = append(hidden, row...)
			}
		} else if full {
			hidden = make([]float32, 0, len(seq)*H)
			for t := range streams {
				row, err := v41MHCPreBF16On(scratch.mhcCarry.device, l, streams[t], scratch.mhcCarry.pre[t])
				if err != nil {
					return nil, err
				}
				hidden = append(hidden, row...)
			}
		}
		act.Hidden = append(act.Hidden, hidden)
	}

	var headProjection v41DenseProjectionFunc
	var finalNorm v41FinalNormFunc
	var bf16Boundary v41BF16BoundaryFunc
	var logitScale v41LogitScaleFunc
	var logitRefusal v41LogitHostRefusal
	if runState != nil {
		headProjection = runState.denseProjection()
		finalNorm = runState.finalNorm
		bf16Boundary = runState.bf16Boundary
		logitScale = runState.logitScale
		logitRefusal = runState.logitHostRefusal()
	}
	act.Logits = make([][]float32, len(seq))
	for t := 0; t < len(seq); t++ {
		headInput := x[t]
		if full {
			headInput = act.Hidden[len(act.Hidden)-1][t*H : (t+1)*H]
		}
		if lastHeadOnly && t != len(seq)-1 {
			if _, err := m.v41PrepareHeadInput(headInput, finalNorm, bf16Boundary); err != nil {
				return nil, err
			}
			continue
		}
		logits, err := m.v41HeadWithLogitScale(headInput, headProjection, finalNorm, bf16Boundary, logitScale, logitRefusal)
		if err != nil {
			return nil, err
		}
		act.Logits[t] = logits
	}
	if st != nil {
		st.layers = runState.layers
		st.indexKeyMemo = runState.indexKeyMemo
		// Completed cross-layer publications are step-local and can be released
		// -- EXCEPT when the schedule has a compressed / shared-source / reader
		// role (#13480): a later incremental Step's reader resolves the source's
		// published compressed rows from this registry, so it must survive the
		// prefill. A plain-only session keeps the historical release byte-for-byte.
		if m.v41RoleSchedule() {
			st.attn = runState.attn
		} else {
			st.attn = nil
		}
		st.forwardCoverage = newNativeV41ForwardCoverage(cfg, executedLayers, NativeForwardLogitsSourceV41Full)
		committed = true
	}
	return act, nil
}

func newNativeV41ForwardCoverage(cfg Config, executedLayers int, source NativeForwardLogitsSource) *NativeForwardCoverage {
	return &NativeForwardCoverage{
		Schema:           NativeForwardCoverageSchemaV1,
		ModelType:        cfg.ModelType,
		ConfiguredLayers: cfg.NumLayers,
		HeadDim:          cfg.HeadDim,
		ExecutedLayers:   executedLayers,
		FullDepth:        executedLayers == cfg.NumLayers,
		Complete:         true,
		LogitsSource:     source,
	}
}

// v41Head retains the host final norm and resident LM head for callers without
// a session-owned projection callback.
func (m *Model) v41Head(x []float32) ([]float32, error) {
	return m.v41HeadWithProjection(x, nil)
}

func (m *Model) v41HeadWithProjection(x []float32, project v41DenseProjectionFunc) ([]float32, error) {
	return m.v41HeadWithFinalNorm(x, project, nil)
}

func (m *Model) v41HeadWithFinalNorm(x []float32, project v41DenseProjectionFunc, normalize v41FinalNormFunc) ([]float32, error) {
	return m.v41HeadWithBF16Boundary(x, project, normalize, nil)
}

func (m *Model) v41HeadWithBF16Boundary(x []float32, project v41DenseProjectionFunc, normalize v41FinalNormFunc, round v41BF16BoundaryFunc) ([]float32, error) {
	return m.v41HeadWithLogitScale(x, project, normalize, round, nil, nil)
}

func (m *Model) v41HeadWithLogitScale(x []float32, project v41DenseProjectionFunc, normalize v41FinalNormFunc, round v41BF16BoundaryFunc, scale v41LogitScaleFunc, refuse v41LogitHostRefusal) ([]float32, error) {
	xf, err := m.v41PrepareHeadInput(x, normalize, round)
	if err != nil {
		return nil, err
	}
	logits, err := m.v41ProjectHeadInput(xf, project, 1)
	if err != nil {
		return nil, err
	}
	return v41ScaleHeadLogits(logits, m.Cfg, scale, refuse)
}

// v41PrepareHeadInput retains the serial final-normalization and BF16 boundaries.
// A cohort prepares each row separately; normalization never crosses sessions.
func (m *Model) v41PrepareHeadInput(x []float32, normalize v41FinalNormFunc, round v41BF16BoundaryFunc) ([]float32, error) {
	if !m.has("model.norm.weight") {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: missing model.norm.weight", ErrV41ForwardStage))
	}
	if len(x) != m.Cfg.HiddenSize {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: final norm width %d, want %d", ErrV41ForwardStage, len(x), m.Cfg.HiddenSize))
	}
	if !m.hasWeight("lm_head.weight") && !m.hasWeight("model.embed_tokens.weight") {
		return nil, v41StageErr(v41StageHead, -1,
			fmt.Errorf("%w: no lm_head.weight and no tied embedding", ErrV41ForwardStage))
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil {
		return nil, err
	}
	var xf []float32
	if full {
		xf, err = m.v41FullFinalNormWithBF16Boundary(x, normalize, round)
		if err != nil {
			return nil, err
		}
	} else if normalize == nil {
		xf = m.finalNorm(x)
	} else {
		var err error
		xf, err = normalize(x)
		if err == nil && len(xf) != m.Cfg.HiddenSize {
			err = errV41ProjectionResult
		}
		if err == nil {
			for _, v := range xf {
				if !finite32(v) {
					err = errV41ProjectionResult
					break
				}
			}
		}
		if err != nil {
			return nil, &V41ProjectionOperationError{Layer: -1, Leaf: "model.norm.weight", Stage: string(v41StageFinalNorm), Cause: v41StageErr(v41StageFinalNorm, -1, err)}
		}
	}
	return xf, nil
}

func (m *Model) v41ProjectHeadInput(xf []float32, project v41DenseProjectionFunc, rows int) ([]float32, error) {
	if project == nil {
		for _, v := range xf {
			if !finite32(v) {
				return nil, v41StageErr(v41StageHead, -1, errV41ProjectionResult)
			}
		}
	}
	return m.v41ProjectionRows(-1, m.headName(), xf, m.Cfg.VocabSize, m.Cfg.HiddenSize, rows, project)
}

// v41FullFinalNorm owns the reference's BF16 -> learned F32 RMSNorm -> BF16
// boundary before ParallelHead widens to F32. The callback itself remains F32;
// generic norm flags belong to the reduced compatibility path, not this graph.
func (m *Model) v41FullFinalNorm(input []float32, normalize v41FinalNormFunc) ([]float32, error) {
	return m.v41FullFinalNormWithBF16Boundary(input, normalize, nil)
}

func (m *Model) v41FullFinalNormWithBF16Boundary(input []float32, normalize v41FinalNormFunc, round v41BF16BoundaryFunc) ([]float32, error) {
	fail := func(cause error) error {
		return &V41ProjectionOperationError{Layer: -1, Leaf: "model.norm.weight", Stage: string(v41StageFinalNorm), Cause: v41StageErr(v41StageFinalNorm, -1, cause)}
	}
	gain := m.tensor("model.norm.weight")
	eps := float32(m.Cfg.RMSNormEps)
	if len(input) == 0 || len(input) != m.Cfg.HiddenSize || len(gain) != len(input) || !finite32(eps) || eps <= 0 {
		return nil, fail(errV41ProjectionResult)
	}
	for _, value := range gain {
		if !finite32(value) {
			return nil, fail(errV41ProjectionResult)
		}
	}
	staged, err := v41BF16BoundaryCopy(-1, "model.norm.weight input", input, round)
	if err != nil {
		if round == nil {
			return nil, fail(errV41ProjectionResult)
		}
		return nil, fail(err)
	}
	var values []float32
	if normalize == nil {
		values = rmsnorm(staged, gain, eps)
	} else {
		var err error
		values, err = normalize(staged)
		if err != nil {
			return nil, fail(err)
		}
	}
	if len(values) != len(input) {
		return nil, fail(errV41ProjectionResult)
	}
	out, err := v41BF16BoundaryCopy(-1, "model.norm.weight output", values, round)
	if err != nil {
		if round == nil {
			return nil, fail(errV41ProjectionResult)
		}
		return nil, fail(err)
	}
	return out, nil
}

// ---- session entry points --------------------------------------------------

// prefillV41 is the DeepSeek V4.1 branch of Session.Prefill. It fails closed
// (panics) with a typed error before any math when a stage is missing, matching
// the requirePreNorm convention but preserving the #12967 weightless-probe
// errors.Is(err, ErrV41NativeUnsupported) contract.
//
// Phase attribution (#13294 DoD item 1): the whole pass runs under
// V41PhasePrefill and one token is noted PER ID — the prompt delta this call
// was ASKED to ingest — only after the forward succeeded, so a fail-closed
// refusal never inflates the phase's denominator. The defer restores the phase
// to the inert default, so anything the next forward path runs notes nothing.
//
// Prefill first-token feasibility (#13294 follow-on): when the model declares a
// routed-expert fault bandwidth AND a prior pass has measured a per-token fault
// volume, the pass projects its first-token latency BEFORE entering the forward
// and fails closed with the typed ErrV41PrefillLatency refusal when it cannot
// clear the session's watchdog window. On the default (no declared bandwidth, or
// no measurement yet) this is inert and the pass runs byte-for-byte as before —
// the physical `fed6a6a37` rung's 0.1 tok/s prefill is exactly the case this
// turns from a 492 s wedge into a named, pre-emptive "no".
func (s *Session) prefillV41(ids []int) []float32 {
	s.ensureOpenBackendSession()
	defer s.v41IndexerScoreGuard()
	if err := s.M.v41ForwardAdmitted(); err != nil {
		panic(err)
	}
	if err := s.M.v41PrefillFirstTokenAdmitted(len(ids)); err != nil {
		panic(err)
	}
	s.M.v41SetExpertFaultPhase(V41PhasePrefill)
	defer s.M.v41SetExpertFaultPhase(V41PhaseUnknown)
	// Suffix-prefill cache reuse (#13346): when the session already holds a
	// seeded, plain-layer continuation state, append the incoming tokens through
	// the same transactional incremental step Session.Step uses (#13313) instead
	// of re-folding the entire committed history. Each suffix token advances one
	// position; only the final token's logits are returned (Prefill's contract).
	// A cold session (no state, or a non-plain/compressed plan) keeps the
	// historical full-prefill path byte-for-byte, so the first request still
	// seeds the cache and a failed step never claims a cache hit.
	if s.v41IncrementalEligible() && len(s.v41Forward.history) > 0 {
		return s.prefillV41Suffix(ids)
	}
	var act *Activations
	var err error
	func() {
		endBatch := s.v41BeginStepBatch()
		defer func() {
			v41FinishStepBatch(endBatch, &err)
			if err != nil {
				act = nil
			}
		}()
		act, err = s.M.forwardV41WithHeadSelection(ids, s.v41State(), true)
	}()
	if err != nil {
		panic(err)
	}
	s.M.v41NoteExpertFaultToken(len(ids))
	return lastLogits(act)
}

// v41PrefillStepProbe is an optional test-only observer invoked once per suffix
// token advanced on the INCREMENTAL prefill route (#13346). It is nil in
// production (a single nil check on the success path) and exists so a witness can
// prove the production Prefill entry point reused prepared cache state rather
// than re-folding the whole history.
var v41PrefillStepProbe func()

// prefillV41Suffix advances each incoming token through the shadow incremental
// seam (forwardV41Step, #13311) over the session's already-seeded state, so the
// committed prefix P is never reprocessed. It is the cached continuation of a
// nonempty session. Each step commits exactly one token and returns the last
// token's logits, matching Prefill's single-last-row contract.
//
// Commit discipline mirrors Session.Step: a structural ErrV41ForwardStage
// refusal (e.g. a retained cursor mismatch) retries the WHOLE suffix through
// full history. A selected device failure is fatal even if it wraps that stage
// sentinel, so it never authorizes host replay. The failing token is rolled
// back; earlier successful suffix tokens retain their per-token commits.
func (s *Session) prefillV41Suffix(ids []int) []float32 {
	st := s.v41State()
	startLen := len(st.history)
	var logits []float32
	for i, id := range ids {
		var got []float32
		var err error
		func() {
			endBatch := s.v41BeginStepBatch()
			defer func() {
				v41FinishStepBatch(endBatch, &err)
				if err != nil {
					got = nil
				}
			}()
			got, _, err = s.M.forwardV41Step(id, st, &v41ProjScratch{})
		}()
		if err != nil {
			var batchErr *v41StepBatchError
			var selectedRoPE *V41TailRoPEOperationError
			var selectedAttention *V41SharedAttentionOperationError
			var selectedScore *V41IndexerScoreOperationError
			var selectedMHC *V41MHCOperationError
			if errors.As(err, &batchErr) || errors.As(err, &selectedRoPE) || errors.As(err, &selectedAttention) || errors.As(err, &selectedScore) || errors.As(err, &selectedMHC) || errors.Is(err, ErrV41BF16BoundaryOperation) || errors.Is(err, ErrV41LogitScaleOperation) || errors.Is(err, ErrV41IndexWeightScaleOperation) || errors.Is(err, ErrV41QueryPrefixOperation) || s.M.v41EPRequested() || errors.Is(err, ErrV41ExpertParallel) || !errors.Is(err, ErrV41ForwardStage) {
				panic(err)
			}
			// Roll the suffix back to the pre-call boundary and re-fold the
			// whole suffix on the reference path so the session state and the
			// returned logits are exactly what a cold prefill would produce.
			st.history = st.history[:startLen]
			var act *Activations
			var ferr error
			func() {
				endBatch := s.v41BeginStepBatch()
				defer func() {
					v41FinishStepBatch(endBatch, &ferr)
					if ferr != nil {
						act = nil
					}
				}()
				act, ferr = s.M.forwardV41(ids, st)
			}()
			if ferr != nil {
				panic(ferr)
			}
			s.M.v41NoteExpertFaultToken(len(ids))
			return lastLogits(act)
		}
		if v41PrefillStepProbe != nil {
			v41PrefillStepProbe()
		}
		if i == len(ids)-1 {
			logits = got
		}
	}
	s.M.v41NoteExpertFaultToken(len(ids))
	return logits
}

// v41IncrementalStepProbe is an optional test-only observer invoked after
// Session.Step advances on the INCREMENTAL route (#13313). It is nil in
// production (a single nil check on the step's success path) and exists so a
// witness can prove the production entry point actually takes the incremental
// seam rather than the full-history fallback.
var v41IncrementalStepProbe func()

// v41IncrementalEligible reports whether the session can advance its next Step
// with the shadow one-token seam (forwardV41Step, #13311) instead of a
// full-history recompute. It is the EXPLICIT eligibility status the #13313 leaf
// requires: every configured layer must hold a seeded retained state and carry a
// resolved plan the one-position seam can execute -- a plain per-layer window
// (v41LayerStep) OR a compressed / shared-source / reader role, which the
// role-aware composition (v41LayerStepRole, #13480) advances over the retained
// compressed stream. Anything else, or a session that has not prefilled yet,
// keeps the historical cold/full route.
//
// The role gate is now an ADMISSION check, not a blanket refusal: a role layer is
// admissible only when a preceding shared source is resolvable for a reader (so a
// reader can never be stepped before its source has published a row). An
// unresolvable reader keeps the full-history fallback rather than letting the
// seam refuse mid-step.
func (s *Session) v41IncrementalEligible() bool {
	st := s.v41Forward
	if st == nil {
		return false
	}
	cfg := s.M.Cfg
	roles := s.M.v41AttentionRolesCached()
	for l := 0; l < cfg.NumLayers; l++ {
		if st.layerState(l) == nil {
			return false
		}
		plan, err := v41AttentionPlanFor(cfg, l, roles)
		if err != nil {
			return false
		}
		if plan.Role != V41AttentionRolePerLayer && plan.Ratio <= 1 && plan.KVSourceLayer < 0 {
			// A reader role with no resolvable shared source cannot be stepped;
			// keep the full-history fallback.
			return false
		}
	}
	return true
}

// stepV41 is the DeepSeek V4.1 branch of Session.Step. When the session holds a
// seeded, plain-layer decode state it advances ONE token through the shadow
// incremental seam (forwardV41Step, #13311): the token is embedded, carried
// through every configured layer and the head, and committed to history only on
// success. The logits are byte-identical to the full-history recompute's
// last-position logits (the #12901 prefill/step consistency property), without
// replaying the prepared prefix. A session that is not incrementally eligible --
// no state yet, a compressed / shared-source / reader layer, or a malformed
// plan -- keeps the cacheless full-history route, so cold requests remain
// usable with an explicit unsupported status and a failed step never claims a
// cache hit.
//
// Phase attribution (#13294): the step runs under V41PhaseDecode and notes ONE
// token — the Step call, i.e. the served decode token the tok/s gate counts —
// on both routes, so the decode ledger's fault cost honestly attributes the
// per-step work the assembly actually pays (the incremental seam on a prepared
// state, the whole-history recompute on the fallback).
func (s *Session) stepV41(id int) []float32 {
	s.ensureOpenBackendSession()
	defer s.v41IndexerScoreGuard()
	if err := s.M.v41ForwardAdmitted(); err != nil {
		panic(err)
	}
	s.M.v41SetExpertFaultPhase(V41PhaseDecode)
	defer s.M.v41SetExpertFaultPhase(V41PhaseUnknown)
	if s.v41IncrementalEligible() {
		var logits []float32
		var err error
		func() {
			endBatch := s.v41BeginStepBatch()
			defer func() {
				v41FinishStepBatch(endBatch, &err)
				if err != nil {
					logits = nil
				}
			}()
			logits, _, err = s.M.forwardV41StepAdmitted(id, s.v41State(), &v41ProjScratch{})
		}()
		if err == nil {
			s.M.v41NoteExpertFaultToken(1)
			if v41IncrementalStepProbe != nil {
				v41IncrementalStepProbe()
			}
			return logits
		}
		var batchErr *v41StepBatchError
		var selectedRoPE *V41TailRoPEOperationError
		var selectedAttention *V41SharedAttentionOperationError
		var selectedScore *V41IndexerScoreOperationError
		var selectedMHC *V41MHCOperationError
		if errors.As(err, &batchErr) || errors.As(err, &selectedRoPE) || errors.As(err, &selectedAttention) || errors.As(err, &selectedScore) || errors.As(err, &selectedMHC) || errors.Is(err, ErrV41BF16BoundaryOperation) || errors.Is(err, ErrV41LogitScaleOperation) || errors.Is(err, ErrV41IndexWeightScaleOperation) || errors.Is(err, ErrV41QueryPrefixOperation) || s.M.v41EPRequested() || errors.Is(err, ErrV41ExpertParallel) || !errors.Is(err, ErrV41ForwardStage) {
			panic(err)
		}
		// A typed stage refusal the eligibility check could not foresee (e.g. the
		// step position does not match the retained cursor). The seam rolled the
		// step back, so history is untouched; fall through to the reference path,
		// which re-folds the token into the committed history.
	}
	act, err := s.M.forwardV41([]int{id}, s.v41State())
	if err != nil {
		panic(err)
	}
	s.M.v41NoteExpertFaultToken(1)
	return lastLogits(act)
}

// v41State lazily installs the session's V4.1 continuation state. When the
// session's backend can run a routed expert's gate/up projections on device
// (the shared q4kExpertInputHALWithLimit operation), the state also
// binds the optional device gate/up callback so the MoE loop can keep the gate/up
// f32 weights off the host and feed the existing host down contraction. A
// non-device session or a backend the shared operation declines leaves the
// callback nil, preserving the historical host triple byte-for-byte.
func (s *Session) v41State() *v41ForwardState {
	if s.v41Forward == nil {
		s.v41Forward = &v41ForwardState{}
	}
	if s.v41Forward.callbackOwner != s {
		s.v41Forward.expertGateUp = s.v41ExpertGateUpFunc()
		s.v41Forward.expertDown = s.v41ExpertDownFunc()
		s.v41Forward.routedExpert = s.v41RoutedExpertFunc()
		s.v41Forward.routedLayer = s.v41RoutedLayerFunc()
		s.v41Forward.routedPrefill = nil
		s.v41Forward.bindCanonicalDense(s.v41BindDenseProjection())
		s.v41Forward.groupedOutput = s.v41GroupedOutputFunc()
		s.v41Forward.engramProjection = s.v41EngramProjectionFunc()
		s.v41Forward.mhcProjection = s.v41MHCProjectionFunc()
		s.v41Forward.mhcFFNProjection = s.v41MHCProjectionFuncNamed("mhc.ffn_mixes.weight")
		s.v41Forward.finalNorm = s.v41FinalNormFunc()
		s.v41Forward.bf16Boundary = s.v41FinalNormBF16CheckedFunc()
		s.v41Forward.logitScale = s.v41LogitScaleFunc()
		s.v41Forward.attentionNorm = s.v41AttentionInputNormFunc()
		s.v41Forward.attentionBF16 = s.v41AttentionBF16Func()
		s.v41Forward.queryNorm = s.v41QueryNormFunc()
		s.v41Forward.queryPrefix = s.v41QueryPrefixFunc()
		s.v41Forward.kvNorm = s.v41KVNormFunc()
		s.v41Forward.ffnNorm = s.v41FFNNormFunc()
		s.v41Forward.compressorNorm = s.v41CompressorNormFunc()
		s.v41Forward.indexKeyNorm = s.v41IndexKeyNormFunc()
		s.v41Forward.indexerScore = s.v41IndexerScoreFunc()
		s.v41Forward.indexScoreHealth = s.v41IndexerScoreHealthFunc()
		s.v41Forward.sharedActivation = s.v41SharedActivationFunc()
		s.v41Forward.tailRoPE = s.v41TailRoPEFunc()
		s.v41Forward.sharedAttention = s.v41SharedAttentionFunc()
		s.v41Forward.bindCanonicalRouter(s.v41BindRouterTopK())
		s.v41Forward.routedSum = s.v41RoutedSumFunc()
		s.v41Forward.sharedAdd = s.v41SharedAddFunc()
		s.v41Forward.gatherOps = s.v41DeviceGatherOpsFunc()
		if s.v41Forward.bf16Boundary == nil {
			s.v41Forward.bf16Boundary = s.v41Forward.gatherOps.bf16PublishFunc()
		}
		s.v41Forward.mhcDevice = s.v41MHCDeviceSeam()
		if s.executionPolicy == ExecutionPolicyDeviceOnly {
			s.v41BindDeviceOnlyDeclineRefusals(s.v41Forward)
		}
		s.v41Forward.callbackOwner = s
		s.bindV41ExpertParallel(s.v41Forward)
	}
	return s.v41Forward
}

// v41AttentionInputNorm implements Block.forward's hc_pre -> attn_norm boundary.
// hc_pre first copies its F32 sum to the BF16 residual dtype. RMSNorm then
// widens to F32, normalizes, applies the learned gain, and copies back to BF16.
// This owns its result: the raw collapse and persistent residual streams remain
// untouched on success or failure. Only full-profile callers use this boundary.
// It retains the host execution policy; no selected device operation is replayed.
//
// Adapted from DeepSeek-V4.1-Flash inference/model.py RMSNorm and Block.forward:
// https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py
// Modification: explicit widened-BF16 boundaries and fail-closed validation in Go.
// This preserves the reference operations, not GPU reduction-order bit identity.
//
// # MIT License
//
// # Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
func (m *Model) v41AttentionInputNorm(layer int, collapsed []float32, eps float32) ([]float32, error) {
	return m.v41AttentionInputNormWithLeaf(layer, "attn_norm.weight", collapsed, eps)
}

func (m *Model) v41AttentionInputNormWithLeaf(layer int, leaf string, collapsed []float32, eps float32) ([]float32, error) {
	gain := m.tensor(layerName(layer, leaf))
	if len(collapsed) != m.Cfg.HiddenSize || len(collapsed) == 0 || len(gain) != len(collapsed) || !finite32(eps) || eps <= 0 {
		return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
	}
	input := make([]float32, len(collapsed))
	for i, value := range collapsed {
		if !finite32(value) || !finite32(gain[i]) {
			return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
		}
		input[i] = v41RoundBF16(value)
		if !finite32(input[i]) {
			return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
		}
	}
	// RMSNorm's learned weight multiplies the already normalized F32 value.
	values := rmsnorm(input, gain, eps)
	for i, value := range values {
		if !finite32(value) {
			return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
		}
		values[i] = v41RoundBF16(value)
		if !finite32(values[i]) {
			return nil, v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
		}
	}
	return values, nil
}

// A nil final-norm callback preserves the portable path. Once selected, an
// operation failure is fatal; it must never retry normalization on the host.
type v41FinalNormFunc func([]float32) ([]float32, error)

func (s *Session) v41FinalNormFunc() v41FinalNormFunc {
	normalize := s.v41RMSNormFunc()
	if normalize == nil {
		return nil
	}
	return func(input []float32) ([]float32, error) {
		return normalize("model.norm.weight", input, s.M.Cfg.HiddenSize, "v41-final-norm", -1)
	}
}

// Query normalization stays between wq_a and wq_b on full-profile models.
// The owner stages BF16 projection input and publishes BF16 normalized output;
// the F32 host/callback arithmetic is unchanged. A selected callback has no
// decline outcome and cannot retry a failed device operation on the host.
type v41QueryNormFunc func(layer int, input []float32) ([]float32, error)

func (s *Session) v41QueryNormFunc() v41QueryNormFunc {
	normalize := s.v41RMSNormFunc()
	if normalize == nil {
		return nil
	}
	return func(layer int, input []float32) ([]float32, error) {
		return normalize(layerName(layer, "attn.wq_a_norm.weight"), input, s.M.Cfg.QLoraRank, "v41-query-norm", layer)
	}
}

func (m *Model) v41QueryNormInPlace(layer int, input []float32, eps float32, normalize v41QueryNormFunc) error {
	return m.v41QueryNormInPlaceWithPublish(layer, input, eps, normalize, nil)
}

func (m *Model) v41QueryNormInPlaceWithPublish(layer int, input []float32, eps float32, normalize v41QueryNormFunc, publish v41BF16BoundaryFunc) error {
	const leaf = "attn.wq_a_norm.weight"
	if m.Cfg.QLoraRank <= 0 || len(input) != m.Cfg.QLoraRank {
		return v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
	}
	staged, err := v41LatentNormBF16Publish(layer, leaf, "projection", input, publish)
	if err != nil {
		return err
	}
	var values []float32
	if normalize == nil {
		gain := m.tensor(layerName(layer, leaf))
		if len(gain) != m.Cfg.QLoraRank {
			return v41StageErr(v41StageAttention, layer,
				fmt.Errorf("%w: q-lora norm has %d values, want %d", ErrV41ForwardStage, len(gain), m.Cfg.QLoraRank))
		}
		values = rmsnormCfg(staged, gain, eps, m.Cfg)
	} else {
		values, err = normalize(layer, staged)
	}
	if err == nil && len(values) != len(input) {
		err = errV41ProjectionResult
	}
	if err != nil {
		return v41ProjectionOperationErr(layer, leaf, err)
	}
	rounded, err := v41LatentNormBF16Publish(layer, leaf, "normalization", values, publish)
	if err != nil {
		return err
	}
	copy(input, rounded)
	return nil
}

// Full-profile KV rows have the published latent width (512) and normalize
// before tail RoPE, with BF16 projection-input and normalized-output boundaries.
// A selected device operation has no host-retry outcome.
type v41KVNormFunc func(layer int, input []float32) ([]float32, error)

func (s *Session) v41KVNormFunc() v41KVNormFunc {
	normalize := s.v41RMSNormFunc()
	if normalize == nil {
		return nil
	}
	return func(layer int, input []float32) ([]float32, error) {
		return normalize(layerName(layer, "attn.kv_norm.weight"), input, v41KVLoraRank, "v41-kv-norm", layer)
	}
}

func (m *Model) v41KVNormInPlace(layer int, input []float32, eps float32, normalize v41KVNormFunc) error {
	return m.v41KVNormInPlaceWithPublish(layer, input, eps, normalize, nil)
}

func (m *Model) v41KVNormInPlaceWithPublish(layer int, input []float32, eps float32, normalize v41KVNormFunc, publish v41BF16BoundaryFunc) error {
	const leaf = "attn.kv_norm.weight"
	if len(input) != v41KVLoraRank {
		return v41ProjectionOperationErr(layer, leaf, errV41ProjectionResult)
	}
	staged, err := v41LatentNormBF16Publish(layer, leaf, "projection", input, publish)
	if err != nil {
		return err
	}
	var values []float32
	if normalize == nil {
		gain := m.tensor(layerName(layer, leaf))
		if len(gain) != v41KVLoraRank {
			return v41StageErr(v41StageAttention, layer,
				fmt.Errorf("%w: kv norm has %d values, want %d", ErrV41ForwardStage, len(gain), v41KVLoraRank))
		}
		values = rmsnormCfg(staged, gain, eps, m.Cfg)
	} else {
		values, err = normalize(layer, staged)
	}
	if err == nil && len(values) != len(input) {
		err = errV41ProjectionResult
	}
	if err != nil {
		return v41ProjectionOperationErr(layer, leaf, err)
	}
	rounded, err := v41LatentNormBF16Publish(layer, leaf, "normalization", values, publish)
	if err != nil {
		return err
	}
	copy(input, rounded)
	return nil
}

// v41LatentNormBF16Copy implements the pinned model.py RMSNorm input/output
// dtype boundaries around wq_a/wkv, whose reference GEMMs return BF16. The
// reference source and MIT notice are retained above v41AttentionInputNorm.
// The F32 RMSNorm callback ABI remains unchanged: neither its input nor a shared
// result is rounded in place, and the caller only publishes after both copies
// succeed. This does not emulate FP8 activation quantization or GPU reductions.
func v41LatentNormBF16Copy(layer int, leaf, phase string, values []float32) ([]float32, error) {
	out := make([]float32, len(values))
	for i, value := range values {
		if finite32(value) {
			out[i] = v41RoundBF16(value)
			if finite32(out[i]) {
				continue
			}
		}
		return nil, v41ProjectionOperationErr(layer, leaf,
			fmt.Errorf("%w: BF16 %s value[%d] is non-finite or overflows", ErrV41ForwardStage, phase, i))
	}
	return out, nil
}

type v41RMSNormFunc func(name string, input []float32, width int, path string, layer int) ([]float32, error)

// Each row uploads and reads back once. Only immutable learned gains are cached;
// this does not establish device-resident forward execution or a speedup.
func (s *Session) v41RMSNormFunc() v41RMSNormFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		s.M.Cfg.LayerNorm || s.M.Cfg.NormGain1p || !compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	if eps := float32(s.M.Cfg.RMSNormEps); !finite32(eps) || eps <= 0 {
		return nil
	}
	return func(name string, input []float32, width int, path string, layer int) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeSelected := func(primary error) {
			if s.halFailure == nil {
				s.halFailure = primary
			}
			if s.halClosed {
				return
			}
			// A typed incoming fault does not prove this callback's owner is
			// retired. Keep its primary even if this owner's Close also panics.
			defer func() { s.halClosed = true; _ = recover() }()
			s.closeBackendOwned()
		}
		closeFailure := func(err error) error {
			if s.halFailure != nil {
				closeSelected(s.halFailure)
				return s.halFailure
			}
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: path, Layer: layer, Stage: stage, Cause: err}
			closeSelected(closed)
			return closed
		}
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						closeSelected(err)
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, cause = nil, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, cause = nil, closeFailure(err)
					return
				}
				// Some backends report plain errors. Preserve an unclassified
				// panic's identity, but never leave its selected session reusable.
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		eps := float32(s.M.Cfg.RMSNormEps)
		meta, present := s.M.manifest[name]
		if width <= 0 || len(input) != width || !finite32(eps) || eps <= 0 ||
			!present || len(meta.Shape) != 1 || meta.Shape[0] != width {
			return nil, closeFailure(errV41ProjectionResult)
		}
		gain := s.M.tensor(name)
		if len(gain) != width {
			return nil, closeFailure(errV41ProjectionResult)
		}
		for i, v := range input {
			if !finite32(v) || !finite32(gain[i]) {
				return nil, closeFailure(errV41ProjectionResult)
			}
		}
		run := func() ([]float32, error) {
			stage = "weight upload"
			weight := s.weightHAL(name)
			stage = "activation upload"
			x := s.uploadHostF32([]int{width}, input, compute.MemoryActivation, "V4.1 RMSNorm activation "+name)
			defer s.Backend.Free(x)
			stage = "rmsnorm"
			y := s.Backend.RMSNorm(x, weight, eps)
			defer s.Backend.Free(y)
			if y.Buf() == nil || y.Dtype != compute.F32 || len(y.Shape) != 1 || y.Shape[0] != width {
				return nil, errV41ProjectionResult
			}
			stage = "readback"
			values := s.Backend.Read(y)
			if len(values) != width {
				return nil, errV41ProjectionResult
			}
			for _, v := range values {
				if !finite32(v) {
					return nil, errV41ProjectionResult
				}
			}
			// Read may expose backend-owned host memory; retain the row before
			// releasing the temporary output tensor.
			return append([]float32(nil), values...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}

// v41ExpertGateUpFunc binds the shared device gate/up operation (#13357) onto the
// V4.1 session backend, or returns nil when the session has no device backend
// that could execute it. It resolves the two gate/up projection names from the
// routed expert stem exactly as the host triple does, runs the shared
// q4kExpertInputHALWithLimit (which itself admits only bias-free SiLU experts whose gate
// and up weights have a device representation the ACTUAL backend can serve), and
// maps its (out, ok) result onto the closed outcome vocabulary. The helper's own
// admission is the gate: a non-device backend, a GELU expert, a biased
// projection, or an unservable dtype returns ok=false here as v41GateUpDeclined,
// never a panic and never a silent host fallback.
func (s *Session) v41ExpertGateUpFunc() v41ExpertGateUpFunc {
	if s == nil || s.Backend == nil || s.M == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	cfg := s.M.Cfg
	full, geometryErr := v41ForwardGeometry(cfg)
	return func(layer int, stem string, xn []float32) (result []float32, outcome v41ExpertGateUpOutcome, cause error) {
		s.ensureOpenBackendSession()
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-expert", Layer: layer, Stage: "gate/up", Cause: err}
			s.halFailure = closed
			s.closeBackendOwned()
			return closed
		}
		// Admission resolves and stages native expert weights, so its failures
		// belong to this selected callback just like GEMM and readback failures.
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, outcome, cause = nil, v41GateUpError, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, outcome, cause = nil, v41GateUpError, closeFailure(err)
					return
				}
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		if geometryErr != nil {
			return nil, v41GateUpError, geometryErr
		}
		gateName := layerName(layer, stem+".w1.weight")
		upName := layerName(layer, stem+".w3.weight")
		out, ok := q4kExpertInputHALWithLimitAndBF16(s, gateName, upName, xn, cfg.MoEIntermediateSize, cfg.HiddenSize, float32(cfg.SwigluLimit), full, layer)
		if !ok {
			return nil, v41GateUpDeclined, nil
		}
		return out, v41GateUpHandled, nil
	}
}

// v41ExpertDownFunc binds the shared device down operation onto the V4.1 session
// backend, or returns nil when the session has no device backend that could
// execute it. It resolves the down projection name from the routed expert stem
// exactly as the host triple does, runs the shared q4kExpertDownHAL (which itself
// admits only bias-free experts whose down weight has a device representation the
// ACTUAL backend can serve), and maps its (out, ok) result onto the closed
// outcome vocabulary. The helper's own admission is the gate: a non-device
// backend, a biased projection, or an unservable dtype returns ok=false here as
// v41DownDeclined, never a panic and never a silent host fallback.
func (s *Session) v41ExpertDownFunc() v41ExpertDownFunc {
	if s == nil || s.Backend == nil || s.M == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	cfg := s.M.Cfg
	full, geometryErr := v41ForwardGeometry(cfg)
	return func(layer int, stem string, fused []float32) (result []float32, outcome v41ExpertDownOutcome, cause error) {
		s.ensureOpenBackendSession()
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-expert", Layer: layer, Stage: "down", Cause: err}
			s.halFailure = closed
			s.closeBackendOwned()
			return closed
		}
		// Admission resolves and stages native expert weights, so its failures
		// belong to this selected callback just like GEMM and readback failures.
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, outcome, cause = nil, v41DownError, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, outcome, cause = nil, v41DownError, closeFailure(err)
					return
				}
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		if geometryErr != nil {
			return nil, v41DownError, geometryErr
		}
		downName := layerName(layer, stem+".w2.weight")
		out, ok := q4kExpertDownHALWithBF16(s, downName, fused, cfg.MoEIntermediateSize, cfg.HiddenSize, full, layer)
		if !ok {
			return nil, v41DownDeclined, nil
		}
		return out, v41DownHandled, nil
	}
}

func lastLogits(act *Activations) []float32 {
	if act == nil || len(act.Logits) == 0 {
		return nil
	}
	return act.Logits[len(act.Logits)-1]
}
