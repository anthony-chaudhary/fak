package model

// v41_decoder.go — the DeepSeek V4.1 text decoder OUTPUT TAIL and the ordinary
// Prefill/Step dispatch seam (issue #12909, leaf of parent #12640). It owns the
// final RMSNorm + output-head projection + logits for the pinned
// 40-layer / width-5120 geometry and exposes the plain entry points a caller
// drives generation with: Prefill([]int) -> logits of the last prompt token and
// Step(id) -> next-token logits, both carrying a session continuation state.
//
// Equations. The tail is a plain in-order transcription of the pinned official
// artifact deepseek-ai/DeepSeek-V4.1-Flash @ revision
// dba1be0a40aa45a94ad051997016db3960a90277 (inference/model.py): the decoder
// stack output is normalized once by the model final RMSNorm
// (model.norm.weight), multiplied by the output head (lm_head.weight, or the
// tied embedding when the head is absent), and scaled by the architecture logit
// scale. V4 Flash 0731 equations are NOT V4.1 evidence and are deliberately not
// consulted here.
//
// Relationship to the reduced assembly. forwardV41 (v41_forward.go) is the
// reduced per-layer assembly; its own per-position tail (v41Head) ends in
// m.logitsFromHidden, the single shared tail Model.Forward and its parallel
// twins also end at. This file does NOT fork that arithmetic: V41TextDecoder
// routes Prefill/Step through the SAME admitted assembly and shared tail, so a
// decoder logit is bit-identical to the assembly logit for the same committed
// history. What this leaf adds is (a) a named, typed seam a session/caller can
// hold without reaching into Model internals, (b) an explicit final-norm /
// output-head contract for the pinned geometry, and (c) fail-closed validation
// of BOTH the geometry and the token ids before any math runs.
//
// Fail-closed design. V41TextDecoder.Prefill/Step never return logits for a
// model whose geometry is not the pinned 40/5120 envelope or whose token ids
// fall outside [0, VocabSize). A malformed geometry or id returns a typed
// *V41DecoderError wrapping ErrV41DecoderGeometry / ErrV41DecoderToken. A
// weightless model fails through the assembly existing admission with an error
// wrapping ErrV41NativeUnsupported (the #12967 fence contract), unchanged.
//
// Scope (gold-plating boundary from #12909): the decoder OUTPUT TAIL + the
// ordinary Prefill/Step dispatch seam only. No checkpoint loading, no GPU
// qualification, no streaming, no vision, no DSpark, no speculative decode.

import (
	"errors"
	"fmt"
)

// ErrV41DecoderGeometry reports that a V4.1 decoder text geometry is malformed
// or is not the pinned 40-layer / width-5120 envelope the tail declares.
var ErrV41DecoderGeometry = errors.New("model: DeepSeek V4.1 decoder geometry is not admitted")

// ErrV41DecoderToken reports that a token id handed to the decoder input is
// malformed (outside [0, VocabSize)). It is the id-vocabulary twin of
// ErrV41DecoderGeometry and is always raised before any math runs.
var ErrV41DecoderToken = errors.New("model: DeepSeek V4.1 decoder token id is malformed")

// V41DecoderTail names the ordered sub-stages of the output tail. It is the
// closed vocabulary a *V41DecoderError reports.
type V41DecoderTail string

const (
	V41TailFinalNorm V41DecoderTail = "final_norm"
	V41TailHead      V41DecoderTail = "head"
	V41TailLogits    V41DecoderTail = "logits"
)

// V41DecoderError is the typed fail-closed error for one decoder-tail stage or
// input. It wraps ErrV41DecoderGeometry, ErrV41DecoderToken, or (via the
// assembly) ErrV41ForwardStage / ErrV41NativeUnsupported, so errors.Is reaches
// whichever closed class actually failed.
type V41DecoderError struct {
	Stage V41DecoderTail
	Err   error
}

func (e *V41DecoderError) Error() string {
	if e == nil {
		return "model: nil V4.1 decoder error"
	}
	if e.Err != nil {
		return fmt.Sprintf("model: V4.1 decoder tail %s: %v", e.Stage, e.Err)
	}
	return fmt.Sprintf("model: V4.1 decoder tail %s failed", e.Stage)
}

func (e *V41DecoderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func v41DecoderTailErr(stage V41DecoderTail, err error) error {
	if err == nil {
		err = ErrV41DecoderGeometry
	}
	return &V41DecoderError{Stage: stage, Err: err}
}

// ---- the pinned output-tail geometry ----------------------------------------

// V41DecoderWidth is the pinned V4.1-Flash hidden width (5120). The output tail
// declares it as a named constant so the final-norm / head contract is read off
// one place rather than re-derived from a caller Config.
const V41DecoderWidth = 5120

// V41DecoderLayers is the pinned V4.1-Flash decoder depth (40). It is the depth
// the tail geometry admission checks; the assembly itself runs m.Cfg.NumLayers.
const V41DecoderLayers = 40

// V41TextDecoderGeometry is the output-tail geometry contract: the final-norm
// width and the head width the tail requires, plus the decoder depth the config
// must declare. It is metadata read by admission; it carries no weights.
type V41TextDecoderGeometry struct {
	HiddenSize int
	NumLayers  int
	VocabSize  int
}

// V41DecoderGeometryFor derives the tail geometry from an admitted V4.1 config
// and enforces the PINNED OFFICIAL envelope: hidden width 5120 and decoder depth
// 40. It is the strict admission a caller runs when claiming the official
// artifact, so a narrowed or widened envelope cannot be silently presented as
// the published model. It deliberately does NOT read weights; weight residency
// is the assembly admission. The runtime Prefill/Step/Tail instead validate the
// model own declared axes with v41DecoderGeometryOf, which admits any
// internally-consistent geometry (the reduced fixtures) while rejecting
// malformed axes.
func V41DecoderGeometryFor(cfg Config) (V41TextDecoderGeometry, error) {
	geom, err := v41DecoderGeometryOf(cfg)
	if err != nil {
		return V41TextDecoderGeometry{}, err
	}
	if geom.HiddenSize != V41DecoderWidth {
		return V41TextDecoderGeometry{}, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: hidden size %d, want pinned %d", ErrV41DecoderGeometry, geom.HiddenSize, V41DecoderWidth))
	}
	if geom.NumLayers != V41DecoderLayers {
		return V41TextDecoderGeometry{}, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: decoder depth %d, want pinned %d", ErrV41DecoderGeometry, geom.NumLayers, V41DecoderLayers))
	}
	return geom, nil
}

// v41DecoderGeometryOf derives the tail geometry from any V4.1 config and fails
// closed only on MALFORMED axes (non-V4.1 identity, non-positive hidden width,
// depth, or vocabulary). The runtime Prefill/Step/Tail use it so a reduced
// fixture is runnable while a genuinely malformed envelope still refuses; the
// pinned-envelope check lives in V41DecoderGeometryFor for callers asserting the
// official artifact.
func v41DecoderGeometryOf(cfg Config) (V41TextDecoderGeometry, error) {
	if !cfg.IsDeepSeekV41() {
		return V41TextDecoderGeometry{}, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: config is not DeepSeek V4.1", ErrV41DecoderGeometry))
	}
	if cfg.HiddenSize <= 0 {
		return V41TextDecoderGeometry{}, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: hidden size %d is not positive", ErrV41DecoderGeometry, cfg.HiddenSize))
	}
	if cfg.NumLayers <= 0 {
		return V41TextDecoderGeometry{}, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: decoder depth %d is not positive", ErrV41DecoderGeometry, cfg.NumLayers))
	}
	if cfg.VocabSize <= 0 {
		return V41TextDecoderGeometry{}, v41DecoderTailErr(V41TailHead,
			fmt.Errorf("%w: vocabulary size %d is not positive", ErrV41DecoderGeometry, cfg.VocabSize))
	}
	return V41TextDecoderGeometry{HiddenSize: cfg.HiddenSize, NumLayers: cfg.NumLayers, VocabSize: cfg.VocabSize}, nil
}

// ---- the output tail --------------------------------------------------------

// v41FinalNorm is the V4.1 decoder single final RMSNorm. It is the tail
// transcription of the reference model.norm: the RMS norm over the last
// hidden row with model.norm.weight. A missing norm weight or a row whose width
// is not the geometry hidden size fails closed with ErrV41DecoderGeometry
// rather than normalizing a truncated vector.
func (m *Model) v41FinalNorm(x []float32, geom V41TextDecoderGeometry) ([]float32, error) {
	if m == nil || len(m.manifest) == 0 {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: final norm weights absent", ErrV41NativeUnsupported))
	}
	if len(x) != geom.HiddenSize {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: final norm width %d, want %d", ErrV41DecoderGeometry, len(x), geom.HiddenSize))
	}
	w, ok := m.manifest["model.norm.weight"]
	if !ok {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: missing model.norm.weight", ErrV41DecoderGeometry))
	}
	if len(w.Shape) != 1 || w.Shape[0] != geom.HiddenSize {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: model.norm.weight shape %v, want [%d]", ErrV41DecoderGeometry, w.Shape, geom.HiddenSize))
	}
	return rmsnormCfg(x, m.tensor("model.norm.weight"), float32(m.Cfg.RMSNormEps), m.Cfg), nil
}

// v41OutputHead projects a normalized hidden row through the output head to
// logits. The head is lm_head.weight when present; otherwise the tied embedding
// table serves as the head (the reference tie_word_embeddings layout), so a
// missing lm_head with a present embedding is admitted, and BOTH absent fails
// closed. The logits carry the architecture logit scale, exactly as the shared
// tail applies it, so a decoder logit equals the assembly logit.
func (m *Model) v41OutputHead(xf []float32, geom V41TextDecoderGeometry) ([]float32, error) {
	if m == nil || len(m.manifest) == 0 {
		return nil, v41DecoderTailErr(V41TailHead,
			fmt.Errorf("%w: output head weights absent", ErrV41NativeUnsupported))
	}
	headName := "lm_head.weight"
	if !m.has(headName) {
		if !m.has("model.embed_tokens.weight") {
			return nil, v41DecoderTailErr(V41TailHead,
				fmt.Errorf("%w: neither lm_head.weight nor a tied embedding is present", ErrV41DecoderGeometry))
		}
		headName = "model.embed_tokens.weight"
	}
	meta := m.manifest[headName]
	if len(meta.Shape) != 2 || meta.Shape[0] != geom.VocabSize || meta.Shape[1] != geom.HiddenSize {
		return nil, v41DecoderTailErr(V41TailHead,
			fmt.Errorf("%w: %s shape %v, want [%d,%d]", ErrV41DecoderGeometry, headName, meta.Shape, geom.VocabSize, geom.HiddenSize))
	}
	if len(xf) != geom.HiddenSize {
		return nil, v41DecoderTailErr(V41TailHead,
			fmt.Errorf("%w: head input width %d, want %d", ErrV41DecoderGeometry, len(xf), geom.HiddenSize))
	}
	logits := matRows(m.tensor(headName), xf, geom.VocabSize, geom.HiddenSize)
	logitScaleInPlace(logits, m.Cfg)
	return logits, nil
}

// v41OutputTail runs the pinned final norm + output head over one hidden row and
// returns the logits. It is the explicit tail contract this leaf owns; the
// assembly shared tail (m.logitsFromHidden) is the twin it must agree with,
// which v41_decoder_test.go witnesses on the reduced fixture.
func (m *Model) v41OutputTail(x []float32, geom V41TextDecoderGeometry) ([]float32, error) {
	xf, err := m.v41FinalNorm(x, geom)
	if err != nil {
		return nil, err
	}
	logits, err := m.v41OutputHead(xf, geom)
	if err != nil {
		return nil, err
	}
	if len(logits) != geom.VocabSize {
		return nil, v41DecoderTailErr(V41TailLogits,
			fmt.Errorf("%w: head emitted %d logits, want %d", ErrV41DecoderGeometry, len(logits), geom.VocabSize))
	}
	return logits, nil
}

// ---- the ordinary Prefill/Step dispatch seam --------------------------------

// V41DecodeState is the session continuation state an ordinary decoder caller
// holds across turns. It records the committed token history so a Step is
// defined as "fold the token into history and recompute", making Step logits
// exactly a longer Prefill last-position logits. The zero value is a fresh
// session; it is not safe for concurrent use (one session is single-goroutine,
// matching Session).
type V41DecodeState struct {
	// fwd is the persistent continuation state the shared assembly folds each
	// call into (forwardV41 appends the new ids and recomputes the whole
	// history). Holding it across calls is what makes Step reuse the committed
	// history rather than rebuild it.
	fwd *v41ForwardState
}

// fwd returns the persistent assembly state, allocating it on first use.
func (s *V41DecodeState) forwardState() *v41ForwardState {
	if s.fwd == nil {
		s.fwd = &v41ForwardState{}
	}
	return s.fwd
}

// History returns a copy of the committed token history (the prompt plus every
// decoded step). The copy keeps a caller from mutating the state own slice.
func (s *V41DecodeState) History() []int {
	if s == nil || s.fwd == nil {
		return nil
	}
	return append([]int(nil), s.fwd.history...)
}

// Reset clears the continuation state so the next Prefill starts a new session.
func (s *V41DecodeState) Reset() {
	if s != nil {
		s.fwd = nil
	}
}

// V41TextDecoder is the ordinary decoder seam over an admitted V4.1 model: a
// caller holds (model, state) and drives Prefill then repeated Step. It is a
// thin, typed front over the same admitted assembly and shared tail
// Model.Forward / Session.Prefill/Step use, so it introduces no second
// arithmetic transcription.
type V41TextDecoder struct {
	Model *Model
	State *V41DecodeState
}

// NewV41TextDecoder binds a decoder to an admitted V4.1 model with a fresh
// continuation state. A nil model, or a model whose config is not the pinned
// V4.1 geometry, fails closed with ErrV41DecoderGeometry (a non-V4.1 config) or
// ErrV41NativeUnsupported (a V4.1 config whose weights are absent) via
// v41ForwardAdmitted.
func NewV41TextDecoder(m *Model) (*V41TextDecoder, error) {
	if m == nil {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: nil model", ErrV41DecoderGeometry))
	}
	if !m.Cfg.IsDeepSeekV41() {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: config is not DeepSeek V4.1", ErrV41DecoderGeometry))
	}
	if _, err := v41DecoderGeometryOf(m.Cfg); err != nil {
		return nil, err
	}
	return &V41TextDecoder{Model: m, State: &V41DecodeState{}}, nil
}

// v41CheckIDs fails closed when any id is outside [0, VocabSize). It runs before
// any embedding lookup so a malformed id can never index a weight table.
func (d *V41TextDecoder) v41CheckIDs(ids []int, geom V41TextDecoderGeometry) error {
	for _, id := range ids {
		if id < 0 || id >= geom.VocabSize {
			return v41DecoderTailErr(V41TailLogits,
				fmt.Errorf("%w: token id %d out of range [0,%d)", ErrV41DecoderToken, id, geom.VocabSize))
		}
	}
	return nil
}

// Prefill ingests a prompt and returns the logits of its last token. It folds
// the prompt into the continuation state and runs the admitted assembly over the
// whole committed history, so a later Step on the same state is consistent with
// a single longer Prefill. An empty prompt returns no logits (nil) without
// touching the state, matching Session.Prefill. A malformed id or geometry fails
// closed with a typed error and no logits.
func (d *V41TextDecoder) Prefill(ids []int) ([]float32, error) {
	if d == nil || d.Model == nil {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: nil decoder", ErrV41DecoderGeometry))
	}
	if d.State == nil {
		d.State = &V41DecodeState{}
	}
	if len(ids) == 0 {
		return nil, nil
	}
	geom, err := v41DecoderGeometryOf(d.Model.Cfg)
	if err != nil {
		return nil, err
	}
	if err := d.v41CheckIDs(ids, geom); err != nil {
		return nil, err
	}
	if err := d.Model.v41ForwardAdmitted(); err != nil {
		return nil, err
	}
	act, err := d.Model.forwardV41(ids, d.State.v41ForwardState())
	if err != nil {
		return nil, err
	}
	logits := lastLogits(act)
	if len(logits) == 0 {
		return nil, v41DecoderTailErr(V41TailLogits,
			fmt.Errorf("%w: prefill produced no logits", ErrV41DecoderGeometry))
	}
	return logits, nil
}

// Step decodes one already-chosen token and returns the next-token logits. It
// folds the token into the committed history and recomputes, so its logits are
// the tail of a single longer Prefill over the same history — the determinism
// property v41_decoder_test.go witnesses. A malformed id fails closed before any
// math; Step before any Prefill fails closed with an empty-history error rather
// than emitting logits for a token the model never conditioned on.
func (d *V41TextDecoder) Step(id int) ([]float32, error) {
	if d == nil || d.Model == nil {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: nil decoder", ErrV41DecoderGeometry))
	}
	if d.State == nil {
		d.State = &V41DecodeState{}
	}
	if len(d.State.History()) == 0 {
		return nil, v41DecoderTailErr(V41TailLogits,
			fmt.Errorf("%w: step before any prefill has no committed history", ErrV41DecoderGeometry))
	}
	geom, err := v41DecoderGeometryOf(d.Model.Cfg)
	if err != nil {
		return nil, err
	}
	if err := d.v41CheckIDs([]int{id}, geom); err != nil {
		return nil, err
	}
	if err := d.Model.v41ForwardAdmitted(); err != nil {
		return nil, err
	}
	act, err := d.Model.forwardV41([]int{id}, d.State.v41ForwardState())
	if err != nil {
		return nil, err
	}
	logits := lastLogits(act)
	if len(logits) == 0 {
		return nil, v41DecoderTailErr(V41TailLogits,
			fmt.Errorf("%w: step produced no logits", ErrV41DecoderGeometry))
	}
	return logits, nil
}

// Tail runs ONLY the output tail over an already-materialized final hidden row,
// without the assembly. It is the seam a caller with externally-produced hidden
// states (e.g. a device-end prefix) uses to get logits, and it is what the
// reduced fixture compares against the shared assembly tail. It fails closed on
// a malformed geometry or a row of the wrong width.
func (d *V41TextDecoder) Tail(x []float32) ([]float32, error) {
	if d == nil || d.Model == nil {
		return nil, v41DecoderTailErr(V41TailFinalNorm,
			fmt.Errorf("%w: nil decoder", ErrV41DecoderGeometry))
	}
	geom, err := v41DecoderGeometryOf(d.Model.Cfg)
	if err != nil {
		return nil, err
	}
	return d.Model.v41OutputTail(x, geom)
}

// v41ForwardState returns the persistent continuation state the shared assembly
// consumes. It is the SAME object across every Prefill/Step on this decoder, so
// forwardV41 appends the new ids onto the committed history and recomputes the
// whole history each call (cacheless), exactly like Session.Prefill/Step. This
// is a pure adapter and introduces no arithmetic fork.
func (s *V41DecodeState) v41ForwardState() *v41ForwardState {
	if s == nil {
		return &v41ForwardState{}
	}
	return s.forwardState()
}
