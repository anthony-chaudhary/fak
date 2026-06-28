package modelengine

// lifecycle.go — the in-kernel model as a native abi.LifecycleEngine.
//
// This is the migration the engine-seam issue calls for: the one-shot Complete is
// re-expressed AS the admit -> per-step -> stream -> reclaim lifecycle, rather than
// living beside it. Admit runs the SAME real Prefill+Step greedy decode over the
// kernel-owned KV cache that Generate/Complete always did, but exposes it one token
// at a time and — the thing the buffered Complete could not do — checks ctx on
// EVERY step, so a cancelled request stops decode MID-GENERATION (not after the
// fixed genTokens budget) and releases its KV-bearing session (slot reclaim).
// Complete is now a thin shim that drains this stream and assembles the identical
// result, so the existing one-shot path rides the new contract with byte-identical
// output (the modelengine tests pin that).
//
// SHAPE NOTE. This is the per-request native lifecycle (one Session per request).
// The continuous-batching shape — many requests sharing ONE StepBatch loop over a
// model.BatchSession — is the NativeScheduler stub in nativesched.go; both
// implement the SAME unchanged abi.LifecycleEngine, which is the cross-shape
// review the seam exists to pass.

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernelRequest is one in-flight in-kernel decode admitted via Engine.Admit.
type inkernelRequest struct {
	tokens chan abi.EngineToken
	done   chan struct{}
	cancel context.CancelFunc

	// tok is the engine's NL tokenizer snapshotted at Admit (nil = byte-level). It
	// detokenizes the generated ids into the result's "generated_text".
	tok NLTokenizer

	// written once by the decode goroutine before close(done); read only after
	// <-done (the close is the happens-before edge — no mutex needed).
	res       *abi.Result
	err       error
	reclaimed bool
}

func (r *inkernelRequest) Tokens() <-chan abi.EngineToken { return r.tokens }

func (r *inkernelRequest) Result() (*abi.Result, error) {
	<-r.done
	return r.res, r.err
}

func (r *inkernelRequest) Cancel() { r.cancel() }

// Reclaimed reports whether the request released its KV-bearing session (the
// slot-reclaim signal). True once the request is terminal — on EOS/budget AND on
// cancellation. Blocks until terminal, mirroring Result.
func (r *inkernelRequest) Reclaimed() bool {
	<-r.done
	return r.reclaimed
}

// Admit registers one decode request against the in-kernel model and starts a real
// greedy Prefill+Step decode on a goroutine, streaming each token on the returned
// handle. The decode honors ctx (or handle.Cancel) at every step: a cancelled
// request stops mid-generation and drops its session, reclaiming the KV. Admit
// itself never fails (the model builds lazily and falls back to synthetic); decode
// is asynchronous, so the handle is returned immediately.
func (e *Engine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	m := e.model()
	in := refBytes(ctx, c.Args)
	prompt := e.buildPrompt(c.Tool, in, m.Cfg.VocabSize)

	sess := m.NewSession()
	if e.q4k {
		// Resident-Q4_K preload: engage the Q4_K decode kernel, exactly as Complete did.
		sess.Quant = true
		sess.Q4K = true
	}

	cctx, cancel := context.WithCancel(ctx)
	r := &inkernelRequest{
		tokens: make(chan abi.EngineToken),
		done:   make(chan struct{}),
		cancel: cancel,
		tok:    e.tok,
	}

	go func() {
		defer cancel() // release the derived context on any exit path
		r.decode(cctx, c.Tool, sess, prompt, ctx)
	}()
	return r, nil
}

// decode runs the per-step greedy loop. It reproduces Session.Generate's exact
// token sequence (Prefill -> argmax -> EOS-break -> Step) so the assembled result
// is byte-identical to the old buffered Complete, while adding a per-step ctx check
// and a per-send ctx select so cancellation lands mid-decode. putCtx (the original
// admit ctx, not the cancellable derivative) backs the success-path Ref Put.
func (r *inkernelRequest) decode(cctx context.Context, tool string, sess *model.Session, prompt []int, putCtx context.Context) {
	gen := make([]int, 0, genTokens)
	logits := sess.Prefill(prompt)
	for i := 0; i < genTokens; i++ {
		if err := cctx.Err(); err != nil { // cancelled between steps — stop before doing more work
			r.finish(nil, err, true)
			return
		}
		next := argmax(logits)
		gen = append(gen, next)
		select {
		case r.tokens <- abi.EngineToken{ID: next}: // ID only: the byte model has no detokenizer
		case <-cctx.Done(): // cancelled while delivering — stop and reclaim
			r.finish(nil, cctx.Err(), true)
			return
		}
		if sess.M.Cfg.IsEOS(next) {
			break
		}
		logits = sess.Step(next)
	}
	r.finish(assembleResult(putCtx, tool, len(prompt), gen, r.tok), nil, true)
}

// assembleResult builds the SAME payload + Meta the one-shot Complete produced, so
// every lifecycle consumer (the per-request decode here AND the native scheduler)
// yields a result byte-identical to the pre-lifecycle buffered path for the same
// call. When tok is non-nil it ADDITIVELY carries the detokenized "generated_text"
// (and a "detokenized" Meta flag) so the /v1/fak/syscall route returns decoded text,
// not just raw ids (#463); a nil tok (the byte-level default) omits both, leaving the
// payload byte-identical to the pre-tokenizer path.
func assembleResult(ctx context.Context, tool string, promptLen int, gen []int, tok NLTokenizer) *abi.Result {
	text := ""
	if tok != nil {
		if s, err := tok.Decode(gen); err == nil {
			text = s
		}
	}
	body, _ := json.Marshal(struct {
		Tool   string `json:"tool"`
		Engine string `json:"engine"`
		Model  string `json:"model"`
		Tokens []int  `json:"generated_tokens"`
		Text   string `json:"generated_text,omitempty"`
	}{
		Tool:   tool,
		Engine: EngineID,
		Model:  "smollm2-inkernel",
		Tokens: gen,
		Text:   text,
	})
	ref := putBytes(ctx, body)
	meta := map[string]string{
		"engine":        EngineID,
		"input_tokens":  strconv.Itoa(promptLen),
		"output_tokens": strconv.Itoa(len(gen)),
	}
	if tok != nil {
		meta["detokenized"] = "true"
	}
	return &abi.Result{
		Payload: ref,
		Status:  abi.StatusOK,
		Meta:    meta,
	}
}

// finish records the terminal state once, flags the KV reclaim, and closes the
// stream + done edges (in that order so a Result/Reclaimed reader past <-done sees
// the written fields).
func (r *inkernelRequest) finish(res *abi.Result, err error, reclaimed bool) {
	r.res, r.err, r.reclaimed = res, err, reclaimed
	close(r.tokens)
	close(r.done)
}

// argmax returns the index of the strictly-greatest logit, matching model's
// internal argmaxF32 tie-break (first max wins) so the streamed token sequence is
// identical to Session.Generate.
func argmax(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

// Engine is a native LifecycleEngine and each handle satisfies EngineRequest.
var (
	_ abi.LifecycleEngine = (*Engine)(nil)
	_ abi.EngineRequest   = (*inkernelRequest)(nil)
)
