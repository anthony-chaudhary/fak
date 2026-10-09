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
// blocked-candidate selection the lightning indexer reads is likewise not executed
// (v41CandidateSourceForwardAdmitted), so an in-range candidate source is refused
// rather than silently run as unblocked attention. The
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
	"github.com/anthony-chaudhary/fak/internal/model/ffn"
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
	denseProjection  v41DenseProjectionFunc
	groupedOutput    v41GroupedOutputFunc
	engramProjection v41EngramProjectionFunc
	mhcProjection    v41MHCProjectionFunc
	finalNorm        v41FinalNormFunc
	queryNorm        v41QueryNormFunc
	kvNorm           v41KVNormFunc
	ffnNorm          v41FFNNormFunc
	sharedActivation v41SharedActivationFunc
	tailRoPE         v41TailRoPEFunc
	sharedAttention  v41SharedAttentionFunc
	callbackOwner    *Session
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
// configured SwiGLU activation, sized to the expert intermediate width.
type v41ExpertGateUpFunc func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error)

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
		runState = &v41ForwardState{history: seq, expertGateUp: st.expertGateUp, expertDown: st.expertDown, denseProjection: st.denseProjection, groupedOutput: st.groupedOutput, engramProjection: st.engramProjection, mhcProjection: st.mhcProjection, finalNorm: st.finalNorm, queryNorm: st.queryNorm, kvNorm: st.kvNorm, ffnNorm: st.ffnNorm, sharedActivation: st.sharedActivation, tailRoPE: st.tailRoPE, sharedAttention: st.sharedAttention, callbackOwner: st.callbackOwner}
		defer func() {
			if !committed {
				st.history = st.history[:historyLen]
			}
		}()
	}
	if len(seq) == 0 {
		if st != nil {
			committed = true
		}
		return &Activations{}, nil
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

	// ---- embedding ----
	embedded, err := m.v41EmbeddingPanel(seq)
	if err != nil {
		return nil, err
	}
	x := make([][]float32, len(seq))
	for t := range seq {
		x[t] = embedded[t*H : (t+1)*H : (t+1)*H]
	}

	// Persistent mHC streams, one four-stream set per position. On the full path
	// stream 0 carries the live hidden state and streams 1..3 are the reference's
	// persistent residual streams initialized to zero -- DISTINCT from stream 0,
	// not the reduced stand-in's four identical copies. Both paths carry the same
	// [][][]float32 shape; only the initialization and mixing differ, so the
	// reduced arithmetic is byte-identical to the pre-#13009 assembly.
	streams := make([][][]float32, len(seq))
	for t := range streams {
		set := make([][]float32, 4)
		set[0] = x[t]
		for h := 1; h < 4; h++ {
			set[h] = make([]float32, H)
		}
		streams[t] = set
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
	// #13325: resolve the optional activation-checkpoint producer ONCE per
	// forward. Disabled (the default) it is nil and every emit is skipped, so the
	// forward is byte-for-byte unchanged and adds no allocations.
	trace := m.v41ActivationTracer(seq)
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41Layer(l, seq, x, streams, full, hd, nH, H, eps, hcIters, hcEps, routeCfg, runState, scratch, trace); err != nil {
			return nil, err
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	var headProjection v41DenseProjectionFunc
	var finalNorm v41FinalNormFunc
	if runState != nil {
		headProjection = runState.denseProjection
		finalNorm = runState.finalNorm
	}
	act.Logits = make([][]float32, len(seq))
	for t := 0; t < len(seq); t++ {
		logits, err := m.v41HeadWithFinalNorm(x[t], headProjection, finalNorm)
		if err != nil {
			return nil, err
		}
		act.Logits[t] = logits
	}
	if st != nil {
		st.layers = runState.layers
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
		committed = true
	}
	return act, nil
}

// v41RoleSchedule reports whether any configured layer resolves to a non-plain
// attention role. It is the release gate for the shared source registry: a role
// schedule must retain the source publications across a prefill so a later
// incremental Step can resolve them (#13480).
func (m *Model) v41RoleSchedule() bool {
	cfg := m.Cfg
	roles := m.v41AttentionRolesCached()
	for l := 0; l < cfg.NumLayers; l++ {
		if v41CompressRatioAt(cfg, l) > 1 {
			return true
		}
		if role, ok := roles[l]; ok && role != V41AttentionRolePerLayer {
			return true
		}
	}
	return false
}

// v41Layer applies one V4.1 decoder layer to x in place, updating the persistent
// mHC streams[t] for each position. tokens carries the ids for the positions in
// x so a declared Engram layer can hash them. The reduced path reproduces the
// pre-#13009 arithmetic exactly (four identical stand-in streams derived from the
// normalized input, and stream 0 of the post-mix written back); the full path
// reads and writes all four DISTINCT persistent streams.
func (m *Model) v41Layer(l int, tokens []int, x [][]float32, streams [][][]float32, full bool, hd, nH, H int, eps float32, hcIters int, hcEps float32, routeCfg v41RouterConfig, st *v41ForwardState, scratch *v41ProjScratch, trace *v41ActivationTraceState) error {
	cfg := m.Cfg
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	scratch.denseProjection = nil
	scratch.groupedOutput = nil
	scratch.mhcProjection = nil
	scratch.queryNorm = nil
	scratch.kvNorm = nil
	scratch.ffnNorm = nil
	scratch.sharedActivation = nil
	scratch.tailRoPE = nil
	scratch.sharedAttention = nil
	if st != nil {
		scratch.denseProjection = st.denseProjection
		scratch.groupedOutput = st.groupedOutput
		scratch.mhcProjection = st.mhcProjection
		scratch.queryNorm = st.queryNorm
		scratch.kvNorm = st.kvNorm
		scratch.ffnNorm = st.ffnNorm
		scratch.sharedActivation = st.sharedActivation
		scratch.tailRoPE = st.tailRoPE
		scratch.sharedAttention = st.sharedAttention
	}
	defer func() {
		scratch.denseProjection = nil
		scratch.groupedOutput = nil
		scratch.mhcProjection = nil
		scratch.queryNorm = nil
		scratch.kvNorm = nil
		scratch.ffnNorm = nil
		scratch.sharedActivation = nil
		scratch.tailRoPE = nil
		scratch.sharedAttention = nil
	}()

	// Engram injection happens at the START of the layer, into the residual,
	// before attention and before attn_norm (ds41_graph_before_attention).
	if cfg.DeepSeekV41 != nil {
		for _, eng := range cfg.DeepSeekV41.EngramLayerIDs {
			if eng == l {
				var project v41EngramProjectionFunc
				if st != nil {
					project = st.engramProjection
				}
				if err := m.v41EngramInjectWithProjection(l, x, streams, full, tokens, eps, project); err != nil {
					return err
				}
				break
			}
		}
	}

	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)
	// The remaining per-layer projections are applied only through matRows, so
	// they read through the streaming-safe v41ProjMatRows at their use site
	// instead of materializing a whole f32 block here (#2150). Fail closed NOW,
	// with the same typed #13276 refusal, if any is absent from every store, so
	// the refuse happens before any layer arithmetic rather than mid-forward.
	for _, leaf := range []string{
		"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
		"ffn.gate.weight",
		"ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight",
		"ffn.shared_experts.w2.weight",
	} {
		if !m.hasResidentWeight(layerName(l, leaf)) {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, layerName(l, leaf)))
		}
	}
	// attn.sink is a 1-D per-head vector (not an isQuantWeight matmul leaf), so it
	// stays on the f32 manifest.
	sink := m.tensor(layerName(l, "attn.sink"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))

	seq := len(x)
	// ---- mHC coefficient split (one split per layer) ----
	//
	// The mHC mix projection geometry is per-layer (it is a per-layer weight), so
	// resolve it once. The published artifact's hc_attn_fn is the flattened
	// four-stream projection (logical 24 x 4H, stored [4H, 24]); the reference
	// (inference/model.py mHC; v4_flash_oracle_test.go:168) computes it over the
	// four width-H streams laid end to end with a single shared flatten-RMS. The
	// reduced fixture uses the legacy 24 x H single-stream matmul. Both are
	// executed here; only the reduced path's arithmetic is held byte-identical to
	// pre-#13009.
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	projectMHC := m.v41MHCProjector(l, H, eps, mhcFlat, mhcTransposed, scratch)
	hcByPos := make([]v41MHCMix, seq)
	preByPos := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// The reduced path projects the normalized input through the legacy
		// single-stream [24, H] block; the full path projects the four DISTINCT
		// persistent streams through the flattened 4H residual with one shared RMS.
		// xn is retained for the reduced pre-collapse stand-in below either way.
		xn := rmsnormCfg(x[t], attnNorm, eps, cfg)
		input := xn
		if mhcFlat {
			width, ok := checkedMulInt(4, H)
			if !ok || len(streams[t]) != 4 {
				return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
			input = make([]float32, 0, width)
			for _, stream := range streams[t] {
				if len(stream) != H {
					return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
				}
				input = append(input, stream...)
			}
		}
		mixes, err := projectMHC(input)
		if err != nil {
			return err
		}
		mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcIters, hcEps)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		hcByPos[t] = mix
		// The mHC pre-collapse always reads a four-stream set collapsed by the
		// learned pre coefficients. The reduced path reconstructs the reference's
		// stand-in (four identical copies of the normalized input, byte-identical
		// to pre-#13009) rather than reading the persistent set, so its numerics
		// are unchanged. The full path reads the four DISTINCT persistent streams
		// carried into this layer (stream 0 = live hidden, 1..3 = residual).
		streams4 := streams[t]
		if !full {
			streams4 = [][]float32{xn, xn, xn, xn}
		}
		collapsed, err := v41MHCPre(streams4, mix.pre)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		preByPos[t] = collapsed
	}

	// ---- attention: projected q/kv per position, then sparse sink per position ----
	qHeads := make([][]float32, seq) // [t][nH*hd], rotated
	kvRows := make([][]float32, seq) // [t][hd], rotated (single KV head)
	qLatRows := make([][]float32, seq)
	// RoPE geometry, fail-closed. The reference rotates only the LAST
	// qk_rope_head_dim components of each head (the leading qk_nope_head_dim pass
	// through byte-identical) using the interleaved adjacent-pair convention. A
	// head width that cannot carry that slice (a non-positive, odd, or
	// head-wider rope dim) is refused with a typed error rather than silently
	// rotating a wrong sub-vector.
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	// Batched query projections across the prefill panel: residentMatMulBatch
	// reads each attn.wq_a / attn.wq_b weight row ONCE and reuses it across all
	// seq prompt tokens, instead of the per-token GEMV loop re-streaming every
	// weight seq times (#13301). The per-token q_norm and RoPE glue below stays
	// scalar, so this is byte-for-byte the per-token v41ProjMatRows path (the
	// adapter's contract); seq==1 (decode) stays entirely on that scalar path.
	preFlat := make([]float32, seq*H)
	for t := 0; t < seq; t++ {
		copy(preFlat[t*H:(t+1)*H], preByPos[t])
	}
	qLatPanel, err := m.v41ProjPanelWithProjection(l, "attn.wq_a.weight", preFlat, cfg.QLoraRank, H, seq, scratch.denseProjection)
	if err != nil {
		return err
	}
	for t := 0; t < seq; t++ {
		qLat := qLatPanel[t*cfg.QLoraRank : (t+1)*cfg.QLoraRank]
		if full {
			// Reference: qr = self.q_norm(self.wq_a(x)). The RMSNorm over the
			// q-lora latent keeps its magnitude O(1) before the wq_b projection;
			// omitting it lets ~1e18 activations reach the 512-term attention
			// accumulation and overflow it to +Inf at HeadDim=512 (#13290).
			// Artifact-only: the reduced fixture carries no q_norm leaf and its
			// pre-#13009 arithmetic is unchanged. Written back into the panel row
			// so the batched wq_b panel reads the normed latents.
			if err := m.v41QueryNormInPlace(l, qLat, eps, scratch.queryNorm); err != nil {
				return err
			}
		}
		// #13325 q_latent: the q-lora latent as fed to wq_b, after the optional q
		// RMSNorm. Captured BEFORE the wq_b panel consumes it.
		trace.record(l, t, v41TraceStageQLatent, qLat)
	}
	qPanel, err := m.v41ProjPanelWithProjection(l, "attn.wq_b.weight", qLatPanel, nH*hd, cfg.QLoraRank, seq, scratch.denseProjection)
	if err != nil {
		return err
	}
	for t := 0; t < seq; t++ {
		c := preByPos[t]
		q := qPanel[t*nH*hd : (t+1)*nH*hd]
		qLat := qLatPanel[t*cfg.QLoraRank : (t+1)*cfg.QLoraRank]
		// KV latent seam. The full path admits and projects attn.wkv at the
		// published latent rank (v41KVLoraRank = 512); the attention contraction
		// below consumes a per-position row of width hd (head_dim), so the full
		// projection is taken at the published latent rank and sliced to the first
		// hd columns for the sink contraction. On a valid full config
		// hd == v41KVLoraRank == 512, so the slice is a no-op; an hd wider than the
		// latent rank is refused rather than silently mis-read. The reduced fixture
		// keeps its HeadDim-wide projection byte-for-byte.
		var kv []float32
		if full {
			if hd > v41KVLoraRank {
				return v41StageErr(v41StageAttention, l,
					fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
			}
			kvFull, err := m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", c, v41KVLoraRank, H, scratch.denseProjection)
			if err != nil {
				return err
			}
			kv = kvFull[:hd]
		} else {
			kv, err = m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", c, hd, H, scratch.denseProjection)
			if err != nil {
				return err
			}
		}
		// Reference: kv = self.kv_norm(self.wkv(x)) then RoPE on the rope tail
		// (_window_kv). The RMSNorm over the KV vector keeps its magnitude O(1);
		// omitting it lets an unbounded projection reach the 512-term attention
		// accumulation and overflow it to +Inf at HeadDim=512 (#13290). Applied to
		// the full head_dim before the rope tail; artifact-only, so the reduced
		// fixture's pre-#13009 arithmetic is unchanged.
		if full {
			if err := m.v41KVNormInPlace(l, kv, eps, scratch.kvNorm); err != nil {
				return err
			}
		}
		// #13325 kv_latent: the KV row after the optional kv RMSNorm and BEFORE
		// RoPE, so a rotary or geometry fault is distinguishable from a projection
		// fault. Captured here because the RoPE just below rotates `kv` in place.
		trace.record(l, t, v41TraceStageKVLatent, kv)
		cos, sin := v41RopeTableForLayer(cfg, l, t)
		if err := v41TailRoPEInPlace(l, q, kv, cos, sin, nH, hd, ropeDim, scratch.tailRoPE); err != nil {
			return err
		}
		qHeads[t] = q
		kvRows[t] = kv
		qLatRows[t] = qLat
	}

	// ---- CED/CSA2 compressor + lightning indexer stages (#13006, #12896) ----
	//
	// A layer declaring CompressRatios[l] > 1 pools its per-position projected KV
	// rows through the CED/CSA2 compressor (v41CompressedRows), and a layer
	// declaring an in-range index source scores its projected index query against
	// those compressed keys and selects rows (v41IndexRows). Both stages execute
	// here and fail closed with a typed *V41ForwardError on malformed geometry; a
	// config that declares an in-range stage without its weights is refused at
	// admission, not here.
	//
	// #12896 maps the sink contraction onto the reference's compressed KV cache:
	// a compressed layer contracts the COMPRESSED stream directly (block-causal
	// visibility, one pooled key/value row per group) instead of expanding the
	// pooled latent back onto causal positions. A declared shared-KV source layer
	// publishes its compressed rows into the session state, and a later reader
	// layer resolves its KV stream from that published source.
	plan, err := v41AttentionPlanFor(cfg, l, m.v41AttentionRolesCached())
	if err != nil {
		return err
	}
	var compressedKV [][]float32
	if plan.Ratio > 1 {
		compressed, err := m.v41CompressedRowsWithProjection(l, plan.Ratio, kvRows, preByPos, scratch.denseProjection)
		if err != nil {
			return err
		}
		compressedKV = compressed
	}
	// A reader layer whose KV source precedes it consumes the source's published
	// compressed stream; a source layer publishes its own for later readers.
	sharedKV := compressedKV
	if plan.Role == V41AttentionRoleReader && plan.KVSourceLayer >= 0 && st != nil {
		attn, err := st.attentionState(hd, 8, cfg.IndexHeadDim)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if rows, ok := attn.KVSourceRows(plan.KVSourceLayer); ok && len(rows) > 0 {
			sharedKV = rows
		}
	}
	var indexKeys [][]float32
	if indexSourceAt(cfg.DeepSeekV41, l) {
		indexKeys, err = m.v41IndexKeys(l, compressedKV, scratch.denseProjection)
		if err != nil {
			return err
		}
	}
	var indexList []int32
	if plan.Ratio > 1 && indexSourceAt(cfg.DeepSeekV41, l) {
		for t := 0; t < seq; t++ {
			groups := min((t+1)/plan.Ratio, len(compressedKV))
			localIdx, err := m.v41IndexRowsProjected(l, qLatRows[t], preByPos[t], indexKeys[:groups], scratch.denseProjection)
			if err != nil {
				return err
			}
			row, err := m.v41AttentionIndexList(plan, st, localIdx, hd, groups, 1)
			if err != nil {
				return err
			}
			indexList = append(indexList, row...)
		}
	} else {
		localIdx, err := m.v41IndexRowsProjected(l, qLatRows[seq-1], preByPos[seq-1], indexKeys, scratch.denseProjection)
		if err != nil {
			return err
		}
		indexList, err = m.v41AttentionIndexList(plan, st, localIdx, hd, len(sharedKV), seq)
		if err != nil {
			return err
		}
	}

	// V41AttentionState is the session-owned validation anchor for the projected
	// KV window. It runs no projection and never changes the arithmetic; it fails
	// closed on malformed geometry before the sink contraction reads the rows.
	//
	// Each layer gets independent temporal ownership. Completed source rows stay
	// in the per-forward registry; retained session state contains only the
	// configured window tail and incomplete compressor group.
	if st != nil {
		layerState, err := NewV41AttentionState(hd, 8)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if cfg.IndexHeadDim > 0 {
			layerState.indexHeadDim = cfg.IndexHeadDim
		}
		// Seed the retained temporal state. For a compressed source layer the
		// incomplete trailing group's PRE-ATTENTION carriers are passed through so
		// the incremental source append (#13480) can continue a mid-group prefix
		// without replaying the projected latents; a non-compressed layer passes
		// none and keeps the historical seed byte-for-byte.
		var seedInputs [][]float32
		if plan.Ratio > 1 {
			if partial := len(preByPos) % plan.Ratio; partial > 0 {
				seedInputs = preByPos[len(preByPos)-partial:]
			}
		}
		if err := layerState.seedTemporalWithInputs(kvRows, seedInputs, plan.Ratio, cfg.windowForLayer(l)); err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if plan.Ratio > 1 {
			ownPlan := plan
			ownPlan.KVSourceLayer = l
			updates := m.v41AttentionSourceUpdates(ownPlan, compressedKV, qLatRows, indexKeys)
			if err := layerState.publishUpdates(updates); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		}
		st.setLayerState(l, cfg.NumLayers, layerState)

		registry, err := st.attentionState(hd, 8, cfg.IndexHeadDim)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		// A declared source layer publishes its compressed rows and index keys so
		// later readers can resolve them within this same forward pass (the
		// V41AttentionState source-then-consumer ordering).
		updates := m.v41AttentionSourceUpdates(plan, compressedKV, qLatRows, indexKeys)
		if len(updates) > 0 {
			if err := registry.publishUpdates(updates); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		}
		// An index source publishes its own per-position top-k selection so a
		// later reader layer can reuse it without recomputing the scoring path.
		// The selection is the source's local index list, one row per published
		// query position.
		if indexSourceAt(cfg.DeepSeekV41, l) && indexList != nil && plan.TopKWidth > 0 {
			rows := make([][]int32, seq)
			for t := range rows {
				rows[t] = append([]int32(nil), indexList[t*plan.TopKWidth:(t+1)*plan.TopKWidth]...)
			}
			if err := registry.PublishTopK(plan.Ratio, rows); err != nil {
				return v41StageErr(v41StageIndexer, l, err)
			}
		}
	}

	scale := cfg.attnScale()
	attnOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// A compressed/shared layer contracts the COMPRESSED KV stream directly:
		// block-causal visibility over pooled group rows, with the lightning
		// indexer's row selection when the layer published one. A per-layer layer
		// keeps the exact per-position causal sink contraction.
		if plan.Ratio > 1 || (plan.Role == V41AttentionRoleReader && len(sharedKV) > 0 && len(sharedKV) < seq) {
			opt := V41AttentionSharedKVOptions{
				Layer: l, Ratio: plan.kvGroupSize(cfg), QueryOffset: t, Groups: len(sharedKV),
				HeadDim: hd, Heads: nH, Softmax: scale, Sink: sink,
			}
			if indexList != nil && len(indexList) >= (t+1)*plan.topKWidth() {
				// The index source publishes one selection row per query position.
				opt.Idx = indexList[t*plan.topKWidth() : (t+1)*plan.topKWidth()]
				opt.IndexTopK = plan.topKWidth()
				opt.TopK = plan.topKWidth()
			}
			attentionOpened := m.v41NowNanos()
			o, err := v41AttentionCompressedForwardWithDevice(qHeads[t], sharedKV, opt, scratch.sharedAttention)
			m.v41NoteAttentionContraction(attentionOpened)
			if err != nil {
				return err
			}
			projected, err := projectOutput(o)
			if err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
			attnOut[t] = projected
			trace.record(l, t, v41TraceStageAttnOut, projected)
			continue
		}
		// #13303: the plain layer's visible keys are its CONFIGURED causal
		// window, not the unconditioned full prefix. A positive
		// Config.Window[l] restricts the row set to the trailing w causal keys
		// max(0,t-w+1)..t; the -1 sentinel (and a nil/short Window, which
		// windowForLayer defaults to -1) keeps the historical full-causal
		// prefix 0..t byte-for-byte. The window changes only WHICH ordered rows
		// are contracted; the score/softmax/value math is untouched.
		window := cfg.windowForLayer(l)
		keys := v41PlainWindowKeys(t, window)
		idx := v41PlainWindowIndexList(keys)
		rows := len(keys)
		flatKV := make([]float32, 0, rows*hd)
		for _, k := range keys {
			flatKV = append(flatKV, kvRows[k]...)
		}
		attentionOpened := m.v41NowNanos()
		o, err := v41SparseAttentionSinkWithDevice(l, qHeads[t], flatKV, sink, idx, V41SparseAttentionSinkOptions{
			B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: rows + 1, N: rows, Softmax: scale,
		}, scratch.sharedAttention)
		m.v41NoteAttentionContraction(attentionOpened)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		projected, err := projectOutput(o)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		attnOut[t] = projected
		trace.record(l, t, v41TraceStageAttnOut, projected)
	}

	// ---- MoE: router + shared expert + routed experts ----
	//
	// The router runs once per position and its picks are retained for the
	// contraction loop below (byte-identical to the per-token router it replaces:
	// the same v41ProjMatRows over the same rmsnormCfg(x[t]) and the same
	// v41Route). The layer-scoped expert cache is reset here so its working set is
	// exactly this layer's routed experts (#13296).
	scratch.v41LayerCacheReset()
	if scratch.expertLayerCacheBytes == 0 {
		if v41TestLayerCacheBudgetOverride > 0 {
			scratch.expertLayerCacheBytes = v41TestLayerCacheBudgetOverride
		} else {
			scratch.expertLayerCacheBytes = m.v41LayerExpertCacheBudget()
		}
	}
	perTokenPicks := make([][]routePick, seq)
	// Reuse one normalized host row for the router, every routed expert, and
	// the shared expert. Device normalization uploads and reads back only once
	// per layer/token; grouped contraction consumes this same panel.
	ffnInputs := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xn, err := m.v41FFNNorm(l, x[t], eps, scratch.ffnNorm)
		if err != nil {
			return err
		}
		ffnInputs[t] = xn
		routerLogits, err := m.v41ProjMatRowsWithProjection(l, "ffn.gate.weight", xn, cfg.NumExperts, H, scratch.denseProjection)
		if err != nil {
			return err
		}
		picks, err := v41Route(routerLogits, gateBias, routeCfg)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}
		perTokenPicks[t] = picks
	}

	// ---- routed-expert contraction ----
	//
	// #13304: a multi-token prefill panel is contracted EXPERT-MAJOR when its
	// routed union exceeds the layer cache. The grouped path materializes one
	// expert triple, contracts every (token, slot) row assigned to it, releases
	// it, then replays each token's weighted sum in original slot order -- so a
	// panel whose alternating routes re-read the same experts faults each distinct
	// projection once, and peak retained expert materialization stays one triple.
	// When the union fits the layer cache the two paths are equivalent (the cache
	// served the repeats either way); the single-token path keeps the historical
	// token-major stream byte-for-byte.
	routedByToken := make([][]float32, seq)
	if seq > 1 && !v41ForceTokenMajor {
		// #13511: thread the session's optional device gate/up callback into the
		// grouped (expert-major) contraction too, so a multi-token prefill offers
		// each routed row to the same device seam the token-major arm uses
		// (#13358) instead of unconditionally running the host SwiGLU. A nil
		// callback (Model.Forward, or no device backend) keeps the grouped path
		// byte-for-byte.
		if err := m.v41ContractRoutedGrouped(l, ffnInputs, perTokenPicks, scratch, cfg, routedByToken, st); err != nil {
			return err
		}
	} else {
		// The multi-token token-major arm (test-forced, or a panel that the
		// grouped path did not take) contracts each pick through the row-parallel
		// twin; the seq == 1 decode fallback through this same arm keeps the
		// package-level serial contraction, because three tiny expert GEMVs per
		// pick would pay the parFor dispatch barrier with no bandwidth to reclaim.
		contract := v41SwiGLU
		if seq > 1 {
			contract = v41SwiGLUParallel
		}
		for t := 0; t < seq; t++ {
			xn := ffnInputs[t]
			routed := make([]float32, H)
			for _, pick := range perTokenPicks[t] {
				stem := "ffn.experts." + itoa(pick.expert)
				// #13358: offer the pick to the session's optional device gate/up
				// seam first. A handled result returns the I-wide fused intermediate
				// from the backend, so the existing host down contraction below runs
				// over it WITHOUT ever materializing the gate/up f32 weights. A
				// decline (or no callback at all) keeps the historical host triple
				// byte-for-byte. A selected device failure must surface.
				if st != nil && st.expertGateUp != nil {
					h, outcome, gerr := st.expertGateUp(l, stem, xn)
					switch outcome {
					case v41GateUpError:
						return v41StageErr(v41StageMoE, l, gerr)
					case v41GateUpHandled:
						// #13704: offer the I-wide intermediate to the device down seam.
						// When it handles, all three projections ran on the backend and no
						// host expert GEMM remains; a decline keeps the historical host f32
						// down contraction byte-for-byte.
						if st.expertDown != nil {
							yd, dOutcome, derr := st.expertDown(l, stem, h)
							switch dOutcome {
							case v41DownError:
								return v41StageErr(v41StageMoE, l, derr)
							case v41DownHandled:
								contractOpen := m.v41NowNanos()
								if contractOpen != 0 {
									m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
								} else {
									m.v41NoteExpertContraction()
								}
								_ = ffn.AddScaled(routed, yd, pick.weight)
								continue
							}
						}
						w2, err := m.hostExpertDown(l, stem, scratch)
						if err != nil {
							return err
						}
						contractOpen := m.v41NowNanos()
						y := matRows(w2, h, H, cfg.MoEIntermediateSize)
						if contractOpen != 0 {
							m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
						} else {
							m.v41NoteExpertContraction()
						}
						_ = ffn.AddScaled(routed, y, pick.weight)
						continue
					}
				}
				w1, w3, w2, err := m.v41ExpertTripleInto(l, stem, scratch, true)
				if err != nil {
					return err
				}
				// #13299: time the routed-expert contraction (the SwiGLU the pick
				// applies) separately from the fault/dequant that produced its
				// weights, so the ledger can attribute the 492 s first token
				// (fak#13294) to scalar contraction vs tier IO. Inert with no clock.
				contractOpen := m.v41NowNanos()
				y := contract(w1, w3, w2, xn, cfg.MoEIntermediateSize, H, cfg)
				if contractOpen != 0 {
					m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
				} else {
					m.v41NoteExpertContraction()
				}
				_ = ffn.AddScaled(routed, y, pick.weight)
			}
			routedByToken[t] = routed
		}
	}

	for t := 0; t < seq; t++ {
		xn := ffnInputs[t]
		routed := routedByToken[t]
		shared, err := m.v41SharedExpertSwiGLUWithActivation(l, xn, cfg, scratch.denseProjection, scratch.sharedActivation)
		if err != nil {
			return err
		}
		moe, err := v41SharedExpertAdd(routed, shared, routeCfg)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}
		// #13325 moe_sum: the summed routed + shared expert output, captured BEFORE
		// the delta/residual mixing so an expert-contraction fault is distinguishable
		// from an mHC post-mix fault.
		trace.record(l, t, v41TraceStageMoESum, moe)

		// delta = attention + MoE, then the mHC post-mix back into four streams.
		delta := make([]float32, H)
		for i := 0; i < H; i++ {
			delta[i] = attnOut[t][i] + moe[i]
		}
		// The post-mix always reads a four-stream residual set. The reduced path
		// reconstructs the stand-in (four identical copies of the collapsed pre
		// vector) so its arithmetic is unchanged; the full path mixes the four
		// distinct persistent streams.
		residual := [][]float32{preByPos[t], preByPos[t], preByPos[t], preByPos[t]}
		if full {
			residual = streams[t]
		}
		next, err := v41MHCPost(delta, residual, hcByPos[t].post, hcByPos[t].comb)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		if full {
			// Write ALL FOUR post-mix streams back into the persistent set so the
			// next layer reads the updated state; stream 0 is the live hidden.
			for h := 0; h < 4; h++ {
				copy(streams[t][h], next[h])
			}
			copy(x[t], next[0])
		} else {
			// Reduced path: only stream 0 is propagated, exactly as before.
			copy(x[t], next[0])
		}
	}
	return nil
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
	var xf []float32
	if normalize == nil {
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
	if project == nil {
		for _, v := range xf {
			if !finite32(v) {
				return nil, v41StageErr(v41StageHead, -1, errV41ProjectionResult)
			}
		}
	}
	logits, err := m.v41ProjectionRows(-1, m.headName(), xf, m.Cfg.VocabSize, m.Cfg.HiddenSize, 1, project)
	if err != nil {
		return nil, err
	}
	logitScaleInPlace(logits, m.Cfg)
	return logits, nil
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
	act, err := s.M.forwardV41(ids, s.v41State())
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
		got, _, err := s.M.forwardV41Step(id, st, &v41ProjScratch{})
		if err != nil {
			var selectedRoPE *V41TailRoPEOperationError
			var selectedAttention *V41SharedAttentionOperationError
			if errors.As(err, &selectedRoPE) || errors.As(err, &selectedAttention) || !errors.Is(err, ErrV41ForwardStage) {
				panic(err)
			}
			// Roll the suffix back to the pre-call boundary and re-fold the
			// whole suffix on the reference path so the session state and the
			// returned logits are exactly what a cold prefill would produce.
			st.history = st.history[:startLen]
			act, ferr := s.M.forwardV41(ids, st)
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
	if err := s.M.v41ForwardAdmitted(); err != nil {
		panic(err)
	}
	s.M.v41SetExpertFaultPhase(V41PhaseDecode)
	defer s.M.v41SetExpertFaultPhase(V41PhaseUnknown)
	if s.v41IncrementalEligible() {
		logits, _, err := s.M.forwardV41Step(id, s.v41State(), &v41ProjScratch{})
		if err == nil {
			s.M.v41NoteExpertFaultToken(1)
			if v41IncrementalStepProbe != nil {
				v41IncrementalStepProbe()
			}
			return logits
		}
		var selectedRoPE *V41TailRoPEOperationError
		var selectedAttention *V41SharedAttentionOperationError
		if errors.As(err, &selectedRoPE) || errors.As(err, &selectedAttention) || !errors.Is(err, ErrV41ForwardStage) {
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
		s.v41Forward.denseProjection = s.v41DenseProjectionFunc()
		s.v41Forward.groupedOutput = s.v41GroupedOutputFunc()
		s.v41Forward.engramProjection = s.v41EngramProjectionFunc()
		s.v41Forward.mhcProjection = s.v41MHCProjectionFunc()
		s.v41Forward.finalNorm = s.v41FinalNormFunc()
		s.v41Forward.queryNorm = s.v41QueryNormFunc()
		s.v41Forward.kvNorm = s.v41KVNormFunc()
		s.v41Forward.ffnNorm = s.v41FFNNormFunc()
		s.v41Forward.sharedActivation = s.v41SharedActivationFunc()
		s.v41Forward.tailRoPE = s.v41TailRoPEFunc()
		s.v41Forward.sharedAttention = s.v41SharedAttentionFunc()
		s.v41Forward.callbackOwner = s
	}
	return s.v41Forward
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
// Portable sessions retain the original host arithmetic; a selected callback
// has no decline outcome and cannot retry a failed device operation on the host.
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
	const leaf = "attn.wq_a_norm.weight"
	if normalize == nil {
		gain := m.tensor(layerName(layer, leaf))
		if len(gain) != m.Cfg.QLoraRank {
			return v41StageErr(v41StageAttention, layer,
				fmt.Errorf("%w: q-lora norm has %d values, want %d", ErrV41ForwardStage, len(gain), m.Cfg.QLoraRank))
		}
		copy(input, rmsnormCfg(input, gain, eps, m.Cfg))
		return nil
	}
	values, err := normalize(layer, input)
	if err == nil && len(values) != len(input) {
		err = errV41ProjectionResult
	}
	if err == nil {
		for _, v := range values {
			if !finite32(v) {
				err = errV41ProjectionResult
				break
			}
		}
	}
	if err != nil {
		return v41ProjectionOperationErr(layer, leaf, err)
	}
	copy(input, values)
	return nil
}

// Full-profile KV rows have the published latent width (512) and normalize
// before tail RoPE. A selected device operation has no host-retry outcome.
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
	const leaf = "attn.kv_norm.weight"
	if normalize == nil {
		gain := m.tensor(layerName(layer, leaf))
		if len(gain) != v41KVLoraRank {
			return v41StageErr(v41StageAttention, layer,
				fmt.Errorf("%w: kv norm has %d values, want %d", ErrV41ForwardStage, len(gain), v41KVLoraRank))
		}
		copy(input, rmsnormCfg(input, gain, eps, m.Cfg))
		return nil
	}
	values, err := normalize(layer, input)
	if err == nil && len(values) != len(input) {
		err = errV41ProjectionResult
	}
	if err == nil {
		for _, v := range values {
			if !finite32(v) {
				err = errV41ProjectionResult
				break
			}
		}
	}
	if err != nil {
		return v41ProjectionOperationErr(layer, leaf, err)
	}
	copy(input, values)
	return nil
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
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: path, Layer: layer, Stage: stage, Cause: err}
			s.halFailure = closed
			s.Close()
			return closed
		}
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
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
	return func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		gateName := layerName(layer, stem+".w1.weight")
		upName := layerName(layer, stem+".w3.weight")
		out, ok := q4kExpertInputHALWithLimit(s, gateName, upName, xn, cfg.MoEIntermediateSize, cfg.HiddenSize, float32(cfg.SwigluLimit))
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
	return func(layer int, stem string, fused []float32) ([]float32, v41ExpertDownOutcome, error) {
		downName := layerName(layer, stem+".w2.weight")
		out, ok := q4kExpertDownHAL(s, downName, fused, cfg.MoEIntermediateSize, cfg.HiddenSize)
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

func (m *Model) v41EmbeddingPanel(ids []int) ([]float32, error) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	count, ok := checkedMulInt(len(ids), H)
	if !ok || H <= 0 {
		return nil, v41StageErr(v41StageEmbedding, -1, errV41ProjectionResult)
	}
	var panel []float32
	if m.has("model.embed_tokens.weight") {
		embed := m.tensor("model.embed_tokens.weight")
		expected, valid := checkedMulInt(cfg.VocabSize, H)
		if !valid || len(embed) < expected {
			return nil, v41StageErr(v41StageEmbedding, -1, fmt.Errorf("%w: invalid embedding table length", ErrV41ForwardStage))
		}
		panel = make([]float32, count)
		for t, id := range ids {
			copy(panel[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		}
	} else if m.Q2KEmbedding != nil {
		var err error
		panel, err = m.Q2KEmbedding.GatherRows(ids, 1)
		if err != nil {
			return nil, v41StageErr(v41StageEmbedding, -1, err)
		}
		if len(panel) != count {
			return nil, v41StageErr(v41StageEmbedding, -1, errV41ProjectionResult)
		}
	} else {
		return nil, v41StageErr(v41StageEmbedding, -1, fmt.Errorf("%w: missing model.embed_tokens.weight", ErrV41ForwardStage))
	}
	for t := range ids {
		scaleEmbedInPlace(panel[t*H:(t+1)*H], cfg)
	}
	return panel, nil
}
