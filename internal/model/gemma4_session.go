package model

import (
	"context"
	"errors"
	"fmt"
)

// gemma4_session.go — the Session bridge that puts the DEDICATED gemma4 forward on the
// resident-quant path (issue #5495).
//
// The problem it closes. gemma4 interleaves two attention regimes with DIFFERENT per-layer
// head_dim (gemma4.go headDimForLayer), which is the whole reason a dedicated forward
// exists. Session.Prefill / PrefillNoLogits / Step used to fall through to the generic
// uniform-geometry lanes (tokenHidden / tokenHiddenQ -> blockStep), whose every shape — the
// q/k/v widths, the KV stride, the RoPE table and, fatally, the qk-norm band's
// applyQKNormCfg argument — is the SCALAR cfg.HeadDim. So a gemma4 checkpoint could only be
// served through the cacheless Model.Forward, and a resident-quant Session refused by name
// (*ResidentQuantUnsupportedError, issue #4274). f32-only is not a real serving option: a
// 26B-class checkpoint at f32 is ~100 GB of weights, so the dedicated forward existed and no
// real checkpoint could reach it on a single card in the <=48 GB class.
//
// What the bridge does. It routes a gemma4 Session to layerGemma4 — the dedicated forward —
// and that path already reads every projection through residentKernel -> residentMatRows,
// which dispatches by NAME over the resident stores (f32, Q8_0, int4, Q4_K, raw k-quant,
// Q2_0, GPTQ). So "wiring gemma4 onto the resident-quant path" needs no new decode kernel and
// no new quant store: it is exactly this dispatch. Every per-layer shape inside it comes from
// headDimForLayer / numKVHeadsForLayer / ropeDimForLayer, so applyQKNormCfg receives the
// PER-LAYER head_dim (gemma4.go:161,164) — the band panic that motivated the refusal is not
// on this path at all.
//
// The DEFAULT bridge deliberately does not build KV state. The explicit bounded
// ConfigureGemma4CachedContinuation qualification route below owns separate full/ring
// rows and complete live snapshots; it does not enable bare-KV or planner/disk reuse.
// Without that typed selection this bridge remains a RECOMPUTE
// session — it carries the token history and re-runs the cacheless forward over the whole
// prefix on each ingest, which is O(n^2) over a generation and is exactly what
// `diagtok --cacheless` does today, promoted from a diagnostic flag into the Session API so a
// serve path can reach it. Because it holds no K/V rows it leaves s.Cache untouched:
// Cache.Len() stays 0.
//
// Eviction is therefore inert — there are no rows to drop. Prefix reuse is NOT, and saying so
// was the #5548 defect: a reuser that CLONES s.Cache clones nothing, so the clone silently
// omits gemma4Hist, which is the entire prefix. A planner that admitted that empty cache and
// then prefilled only the divergent suffix against it served wrong logits with no error and no
// panic. Reuse is fail-CLOSED here instead — KVPrefixReuseSupported below is the one predicate
// every reuser must ask before treating a *KVCache as a complete session prefix — and becomes
// available, correctly, when #5496 lands the cached per-layer-window path.
//
// Note this is NOT a decode-store change: it touches neither of the two q4 stores (the
// GGUF-native Q4_K super-block with its ggml nibble interleave, nor the re-quantized int4
// q4Tensor with consecutive elements per byte). It only chooses which forward runs; each
// store keeps its own decoder, reached unchanged through residentMatRows.

// KVPrefixReuseSupported reports whether a *KVCache is a COMPLETE prefix for a Session over
// this Config — i.e. whether cloning the cache carries the whole of what the session already
// ingested. It is false for the gemma4 recompute bridge, whose state is the token history
// (gemma4Hist) and whose cache stays empty. The explicitly selected cached route
// also keeps its heterogeneous rows OUTSIDE the uniform KVCache; only its complete
// live PrefixSnapshot carries them. Bare-KV/planner/disk admission remains closed.
// It is also false for DeepSeek V4.1 until its
// Engram and shared-KV continuation state is included in prefix snapshots.
//
// It exists because "the cache is empty" is indistinguishable from "the cache is a valid
// zero-length prefix" at every consumer: radixkv.truncatePrefix returns a non-nil zero-length
// clone, a nil-guard passes, and the radix tree matches on TOKEN IDS rather than on KV depth.
// So a reuser holding only a *KVCache cannot detect the difference and must ask the Config
// (#5548). Splitting the rule out of the reuser keeps it from being re-derived — and re-missed
// — at each of the several places that build a session from a cached prefix.
func (c Config) KVPrefixReuseSupported() bool { return !c.isGemma4() && !c.IsDeepSeekV41() }

// gemma4SessionModeWired reports whether this Session's execution mode is one the gemma4
// bridge actually runs. The bridge is the HOST resident path (residentKernel over whatever
// store the model loaded as). A device session (Backend), the Metal prefill lanes, and
// dynamic whole-token precision are hand-copies of the uniform-geometry PreNorm block and
// have no gemma4 twin, so they are NOT wired and must fail closed instead of silently
// running the wrong geometry.
func (s *Session) gemma4SessionModeWired() bool {
	return s.Backend == nil && !s.Metal && !s.MetalQ4K && s.PrecisionPolicy == nil
}

// requireGemma4Session fails a gemma4 Session closed, by name, for the execution modes the
// dedicated forward is not wired for. It mirrors requireMiniMaxSession's idiom: a loud,
// readable boundary at the entry instead of a cryptic panic deep inside a band that assumed
// a uniform geometry. It carries the same typed *ResidentQuantUnsupportedError the generic
// band raises, so an operator (or a preflight) reads ONE inspectable type for "this gemma4
// session cannot run here", with Format naming the resident-quant store in play.
func (s *Session) requireGemma4Session() {
	if s.gemma4SessionModeWired() {
		return
	}
	panic(&ResidentQuantUnsupportedError{
		Arch:   s.M.Cfg.archFamilyKey(),
		Format: s.residentQuantForwardFormat(),
		Mode:   s.unwiredGemma4ModeName(),
	})
}

// unwiredGemma4ModeName names the FIRST unwired execution mode this session carries, so the
// refusal says which one to drop rather than listing every mode it is not.
func (s *Session) unwiredGemma4ModeName() string {
	switch {
	case s.Backend != nil:
		return "compute.Backend (device HAL)"
	case s.Metal:
		return "Metal prefill"
	case s.MetalQ4K:
		return "Metal Q4_K prefill"
	case s.PrecisionPolicy != nil:
		return "dynamic whole-token precision"
	default:
		return ""
	}
}

// gemma4Ingest appends ids to this session's token history. It is the whole of a
// KV-advancing prefill for the recompute bridge: the dedicated gemma4 forward is cacheless,
// so the ONLY state an ingest can advance is the history itself — there is no per-layer K/V
// to fill. That is why PrefillNoLogits costs nothing here rather than running a forward whose
// every output would be discarded.
func (s *Session) gemma4Ingest(ids []int) {
	s.requireGemma4Session()
	s.gemma4Hist = append(s.gemma4Hist, ids...)
}

// gemma4StackLast ingests ids and returns the LAST position's hidden state after the whole
// dedicated gemma4 decoder stack, recomputed over the entire retained history. It is
// Model.forwardHiddenRows's gemma4 arm with the per-position logit fill dropped: a Session
// consumes only the final position's distribution, and the LM head is the single largest GEMM
// in the model, so filling it at every position would be pure waste. The final norm is NOT
// applied here — logitsFromHidden owns it, and sharing that one tail is what keeps this
// bit-exact with Forward.
func (s *Session) gemma4StackLast(ids []int) []float32 {
	s.gemma4Ingest(ids)
	m := s.M
	x := m.embedBand(s.gemma4Hist)
	ropeFreqs := m.gemma4RopeFreqs()
	for l := 0; l < m.Cfg.NumLayers; l++ {
		m.layerGemma4(l, x, ropeFreqs)
	}
	return x[len(x)-1]
}

// prefillGemma4 ingests a prompt through the dedicated gemma4 forward and returns the logits
// of its LAST token — Session.Prefill's contract, served by the gemma4 path.
func (s *Session) prefillGemma4(ids []int) []float32 {
	return s.prefillGemma4Context(nil, ids)
}

func (s *Session) prefillGemma4Context(ctx context.Context, ids []int) []float32 {
	if s.gemma4Cached != nil {
		return s.gemma4CachedForward(ctx, ids, true)
	}
	return s.M.logitsFromHidden(s.gemma4StackLast(ids))
}

func (s *Session) prefillNoLogitsGemma4Context(ctx context.Context, ids []int) {
	if s.gemma4Cached != nil {
		s.gemma4CachedForward(ctx, ids, false)
		return
	}
	s.gemma4Ingest(ids)
}

// stepGemma4 decodes one already-chosen token. On a recompute session a decode step is an
// ingest of one id followed by the same tail, so Prefill and Step cannot diverge.
func (s *Session) stepGemma4(id int) []float32 {
	return s.prefillGemma4([]int{id})
}

// Gemma4CachedContinuationOptions selects the bounded host continuation for
// qualification. The zero/default Session keeps its original recompute route.
// MaxStateBytes bounds owned K/V and reserved token-history payload, NOT model
// weights, transient forward work, allocator overhead or total process memory.
type Gemma4CachedContinuationOptions struct {
	MaxTokens     int
	MaxStateBytes int64
}

// ErrGemma4CachedContinuation is a fail-closed selection, state or lifetime refusal.
var ErrGemma4CachedContinuation = errors.New("model: Gemma4 cached continuation unavailable")

// Gemma4CachedContinuationObservation names actual selected state and completed
// new work. Restoring a prefix resets ComputedTokens: copied history is not new
// execution. These counters are not a wall-time or whole-model quality result.
// Like the Session's forward/configuration API, observation requires its exclusive caller.
type Gemma4CachedContinuationObservation struct {
	Selected       bool
	Tokens         int
	ComputedTokens uint64
	StateBytes     int64
	Failed, Closed bool
}

type gemma4CachedContinuation struct {
	model                    *Model
	epoch                    uint64
	configDigest             [32]byte
	options                  Gemma4CachedContinuationOptions
	rows                     *attentionStateRows
	skipRopeFreqs, scaleSqrt bool
	computed                 uint64
	failure                  error
	closed                   bool
}

// ConfigureGemma4CachedContinuation selects once, on a fresh host Session. Model
// weights/configuration are immutable for its lifetime, as on the reference path;
// the complete existing config digest also refuses accidental geometry changes.
// No backend fallback, environment toggle or bare-KV admission is introduced.
func (s *Session) ConfigureGemma4CachedContinuation(options Gemma4CachedContinuationOptions) error {
	if s == nil || s.M == nil || !s.M.Cfg.isGemma4() {
		return fmt.Errorf("%w: requires a Gemma4 model", ErrGemma4CachedContinuation)
	}
	s.cacheGeometryMu.Lock()
	defer s.cacheGeometryMu.Unlock()
	if s.gemma4Closed || s.halClosed || s.gemma4Cached != nil ||
		s.Cache == nil || s.Cache.Len() != 0 || len(s.gemma4Hist) != 0 {
		return fmt.Errorf("%w: selection requires a fresh open session", ErrGemma4CachedContinuation)
	}
	if err := s.gemma4CachedModeError(); err != nil {
		return err
	}
	cfg := s.M.Cfg
	if options.MaxTokens <= 0 || options.MaxStateBytes <= 0 ||
		cfg.NumLayers <= 0 || int64(cfg.NumLayers) > options.MaxStateBytes/8 ||
		cfg.HiddenSize <= 0 || cfg.VocabSize <= 0 || cfg.NumHeads <= 0 {
		return fmt.Errorf("%w: invalid bounded geometry", ErrGemma4CachedContinuation)
	}
	digest, err := spanConfigDigest(cfg)
	if err != nil {
		return fmt.Errorf("%w: config identity: %v", ErrGemma4CachedContinuation, err)
	}
	specs := make([]attentionRowSpec, cfg.NumLayers)
	for l := range specs {
		hd, kv, rd := cfg.headDimForLayer(l), cfg.numKVHeadsForLayer(l), cfg.ropeDimForLayer(l)
		// The reference clamps rotary width to head width; retain that exact
		// interpretation, but reject invalid/odd geometry before state allocation.
		if rd > hd {
			rd = hd
		}
		if hd <= 0 || kv <= 0 || cfg.NumHeads%kv != 0 || rd <= 0 || rd%2 != 0 || cfg.windowForLayer(l) == 0 || cfg.ropeThetaForLayer(l) <= 0 {
			return fmt.Errorf("%w: invalid layer %d geometry", ErrGemma4CachedContinuation, l)
		}
		qwidth, qok := checkedMulInt(cfg.NumHeads, hd)
		kvwidth, kvok := checkedMulInt(kv, hd)
		_, qmatOK := checkedMulInt(qwidth, cfg.HiddenSize)
		_, kvmatOK := checkedMulInt(kvwidth, cfg.HiddenSize)
		if !qok || !kvok || !qmatOK || !kvmatOK {
			return fmt.Errorf("%w: layer %d dimensions overflow", ErrGemma4CachedContinuation, l)
		}
		specs[l] = attentionRowSpec{KVHeads: kv, HeadDim: hd, Window: cfg.windowForLayer(l)}
	}
	rows, err := newAttentionStateRows(specs, options.MaxTokens, options.MaxStateBytes)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrGemma4CachedContinuation, err)
	}
	s.gemma4Cached = &gemma4CachedContinuation{model: s.M, epoch: s.cacheGeometryEpoch,
		configDigest: digest, options: options, rows: rows,
		skipRopeFreqs: gemma4SkipRopeFreqs(), scaleSqrt: gemma4ScaleSqrt()}
	return nil
}

func (s *Session) gemma4CachedModeError() error {
	// mlpSeq's optional routing observer numbers a whole sequence from zero.
	// Its incremental absolute-position contract is not wired by this profile.
	if s.M.routeObs != nil {
		return fmt.Errorf("%w: routing observer is not bound to cached positions", ErrGemma4CachedContinuation)
	}
	if !s.gemma4SessionModeWired() || s.DenseGPULayers != 0 || s.GPULayers != 0 ||
		s.executionPolicy == ExecutionPolicyDeviceOnly ||
		(s.KVPrecision != "" && s.KVPrecision != KVPrecisionFP32) {
		return fmt.Errorf("%w: only the host resident, F32-state mode is selected", ErrGemma4CachedContinuation)
	}
	return nil
}

func (st *gemma4CachedContinuation) validate(s *Session) error {
	if st == nil || s == nil || st.closed || s.gemma4Closed || s.halClosed || st.failure != nil || st.rows == nil {
		return fmt.Errorf("%w: closed, failed or absent owner", ErrGemma4CachedContinuation)
	}
	if st.model != s.M || st.epoch != s.cacheGeometryEpoch || s.Cache == nil || s.Cache.Len() != 0 || len(s.gemma4Hist) != 0 {
		return fmt.Errorf("%w: model, epoch or position authority mismatch", ErrGemma4CachedContinuation)
	}
	if err := s.gemma4CachedModeError(); err != nil {
		return err
	}
	digest, err := spanConfigDigest(s.M.Cfg)
	if err != nil || digest != st.configDigest || gemma4SkipRopeFreqs() != st.skipRopeFreqs || gemma4ScaleSqrt() != st.scaleSqrt {
		return fmt.Errorf("%w: model configuration or rotary/scale mode changed", ErrGemma4CachedContinuation)
	}
	return nil
}

func (s *Session) Gemma4CachedContinuationObservation() Gemma4CachedContinuationObservation {
	if s == nil || s.gemma4Cached == nil {
		return Gemma4CachedContinuationObservation{}
	}
	st := s.gemma4Cached
	out := Gemma4CachedContinuationObservation{Selected: true, ComputedTokens: st.computed, Failed: st.failure != nil, Closed: st.closed}
	if st.rows != nil {
		out.Tokens, out.StateBytes = len(st.rows.history), st.rows.payloadBytes
	}
	return out
}

func (s *Session) failGemma4CachedContinuation(err error) {
	if s != nil && s.gemma4Cached != nil && s.gemma4Cached.failure == nil {
		s.gemma4Cached.failure = err
	}
}

func (s *Session) closeGemma4CachedContinuation() {
	s.gemma4Closed = true
	if st := s.gemma4Cached; st != nil {
		st.closed = true
		st.rows.close()
	}
}

// gemma4CachedForward checks the entire call's input/capacity before mutation.
// Each token attends to old rows plus its separate new projections; all layers
// (and the requested final head) finish before one no-allocation publication.
// A panic/Goexit poisons the selected owner without swallowing the primary
// failure. Cancellation is observed between complete tokens, never silently
// retried through recompute; callers must close that partial selected session.
func (s *Session) gemma4CachedForward(ctx context.Context, ids []int, wantLogits bool) []float32 {
	st := s.gemma4Cached
	if err := st.validate(s); err != nil {
		panic(err)
	}
	if len(ids) > st.rows.maxTokens-len(st.rows.history) {
		panic(fmt.Errorf("%w: token reservation exhausted", ErrGemma4CachedContinuation))
	}
	for _, id := range ids {
		if id < 0 || id >= s.M.Cfg.VocabSize {
			panic(fmt.Errorf("%w: token ID %d outside vocabulary", ErrGemma4CachedContinuation, id))
		}
	}
	complete := false
	defer func() {
		if !complete {
			s.failGemma4CachedContinuation(fmt.Errorf("%w: interrupted forward", ErrGemma4CachedContinuation))
		}
	}()
	var logits []float32
	ropeFreqs := s.M.gemma4RopeFreqs()
	for i, id := range ids {
		if ctx != nil && ctx.Err() != nil {
			s.failGemma4CachedContinuation(ctx.Err())
			return nil
		}
		position := len(st.rows.history)
		x := s.M.embedBand([]int{id})
		pending := make([]attentionPendingRow, len(st.rows.layers))
		for l := range st.rows.layers {
			s.M.layerGemma4WithAttention(l, x, func(xn [][]float32) [][]float32 {
				q, k, v := s.M.gemma4ProjectQKV(l, xn, ropeFreqs, position)
				pending[l] = attentionPendingRow{k: k[0], v: v[0]}
				return s.M.gemma4AttentionRows(l, q, k, v, position, st.rows)
			})
		}
		if wantLogits && i == len(ids)-1 {
			logits = s.M.logitsFromHidden(x[0])
		}
		st.rows.publish(id, pending)
		st.computed++
	}
	if ctx != nil && ctx.Err() != nil {
		s.failGemma4CachedContinuation(ctx.Err())
		return nil
	}
	complete = true
	return logits
}

func (st *gemma4CachedContinuation) clone() *gemma4CachedContinuation {
	if st == nil {
		return nil
	}
	out := *st
	out.rows = st.rows.clone()
	out.computed = 0
	return &out
}

// validateRestore does not consume either owner on a refusal. A receiver must
// explicitly select the same bounded profile; restore never enables a route.
func (st *gemma4CachedContinuation) validateRestore(s *Session, tokens int) error {
	if st == nil || s == nil || s.gemma4Cached == nil {
		return fmt.Errorf("%w: restore requires complete selected state", ErrGemma4CachedContinuation)
	}
	if err := s.gemma4Cached.validate(s); err != nil {
		return err
	}
	if st.closed || st.failure != nil || st.rows == nil || st.model != s.M ||
		st.configDigest != s.gemma4Cached.configDigest || st.options != s.gemma4Cached.options ||
		st.skipRopeFreqs != s.gemma4Cached.skipRopeFreqs || st.scaleSqrt != s.gemma4Cached.scaleSqrt ||
		len(st.rows.history) != tokens {
		return fmt.Errorf("%w: snapshot identity, profile or token count mismatch", ErrGemma4CachedContinuation)
	}
	return nil
}
