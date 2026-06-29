package modelengine

// lifecycle.go — the registered in-kernel engine as a native abi.LifecycleEngine.
//
// Engine.Admit now delegates to the process-local NativeScheduler in nativesched.go:
// many admitted requests share one scheduler loop, and each decode iteration advances
// the live lanes with one model.BatchSession StepBatch call when the model path supports
// it. Complete remains the same one-shot shim over Admit, so existing kernel.Syscall /
// Reap callers ride the continuous-batching lifecycle without changing the ABI.
//
// The payload builder and token chooser live here because both the registered Engine
// and the standalone NativeScheduler use the same result contract.

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Admit registers one decode request with the engine's native continuous-batching
// scheduler. The scheduler streams tokens one at a time, observes ctx / Cancel between
// steps, and reclaims the lane's KV-bearing Session when the request reaches terminal.
func (e *Engine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	return e.nativeScheduler().Admit(ctx, c)
}

// assembleResult builds the SAME payload + Meta the one-shot Complete produced, so
// every lifecycle consumer (the per-request decode here AND the native scheduler)
// yields a result byte-identical to the pre-lifecycle buffered path for the same
// call. When tok is non-nil it ADDITIVELY carries the detokenized "generated_text"
// (and a "detokenized" Meta flag) so the /v1/fak/syscall route returns decoded text,
// not just raw ids (#463); a nil tok (the byte-level default) omits both, leaving the
// payload byte-identical to the pre-tokenizer path.
func assembleResult(ctx context.Context, tool string, promptLen int, gen []int, tok NLTokenizer) *abi.Result {
	return assembleSyscallResult(ctx, tool, EngineID, "smollm2-inkernel", promptLen, gen, tok)
}

// assembleSyscallResult builds the syscall decode result shared by the buffered lifecycle
// path (assembleResult) and the pipeline path (assemblePipelineResult): the two produce a
// byte-identical payload + Meta and differ ONLY in the engineID/modelName labels they
// stamp, so those are parameters here. A non-nil tok ADDITIVELY carries the detokenized
// "generated_text" + a "detokenized" Meta flag (#463); a nil tok omits both, leaving the
// payload byte-identical to the pre-tokenizer path.
func assembleSyscallResult(ctx context.Context, tool, engineID, modelName string, promptLen int, gen []int, tok NLTokenizer) *abi.Result {
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
		Engine: engineID,
		Model:  modelName,
		Tokens: gen,
		Text:   text,
	})
	meta := map[string]string{
		"engine":        engineID,
		"input_tokens":  strconv.Itoa(promptLen),
		"output_tokens": strconv.Itoa(len(gen)),
	}
	if tok != nil {
		meta["detokenized"] = "true"
	}
	return &abi.Result{
		Payload: putBytes(ctx, body),
		Status:  abi.StatusOK,
		Meta:    meta,
	}
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

// Engine is a native LifecycleEngine.
var _ abi.LifecycleEngine = (*Engine)(nil)
