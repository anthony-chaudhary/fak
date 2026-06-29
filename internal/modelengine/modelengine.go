// Package modelengine wires the in-kernel model (internal/model) into the kernel
// as a registered abi.EngineDriver under the id "inkernel".
//
// Until now internal/model was a proven-correct forward-pass runtime with NO seam
// into the dispatch path: internal/agent and internal/engine never imported it, so
// `fak run`/`fak agent` could only dispatch a tool call to the mock or an HTTP
// upstream — never to the model fused into the kernel. This package closes that
// gap with the SAME mechanism every other backend uses: a driver that implements
// EngineDriver and registers itself from init(), so selecting it is one
// `--engine inkernel` flag (or one blank-import line), never a kernel edit.
//
// What "completing a tool call on the in-kernel model" means here: the driver
// materializes the call's argument bytes, tokenizes them into a bounded prompt, and
// runs a REAL greedy decode over kernel-owned per-request KV caches. Concurrent
// calls are admitted into the native continuous-batching scheduler, which advances
// live lanes with model.BatchSession StepBatch when that path is supported. The
// result payload carries the generated token ids + token accounting.
//
// Weights: by default the driver runs a small DETERMINISTIC synthetic checkpoint
// (model.NewSynthetic) so the engine works on a CI box with no model export and a
// test is reproducible — the same honesty stance the KV-quarantine bridge takes
// (its wiring is proven on a synthetic model; the numerics are proven separately by
// the HF oracle in internal/model). Point FAK_MODEL_DIR at a real export to load
// genuine weights (model.Load); the dispatch path is identical either way.
//
// The model is built LAZILY on the first Complete (guarded by sync.Once) so merely
// blank-importing this package — which every binary does via internal/registrations
// — costs nothing at startup; the synthetic checkpoint is only constructed if a
// call is actually routed to "inkernel".
package modelengine

import (
	"context"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// EngineID is the registered id the kernel selects this backend by.
const EngineID = "inkernel"

// NLTokenizer is the OPTIONAL natural-language tokenizer the in-kernel engine uses
// to turn a tool call's arguments into a real NL prompt (Encode) and to turn the
// generated token ids back into decoded TEXT (Decode), instead of the zero-dependency
// byte-level default. The `fak serve|run|guard --gguf` boot installs the GGUF's
// embedded BPE tokenizer here via SetTokenizer (resolveServeTokenizer); a
// *tokenizer.Tokenizer satisfies it. When it is nil — the CI default with no model
// export — the engine keeps the byte-level path verbatim, so the synthetic-vs-real
// split stays exactly as it is for weights. This closes issue #463's named gap: the
// /v1/fak/syscall route returns decoded text, not raw token ids, once a real
// tokenizer is configured.
type NLTokenizer interface {
	Encode(text string) ([]int, error)
	Decode(ids []int) (string, error)
}

// genTokens is how many tokens a single tool-call completion decodes. Small: the
// engine demonstrates the live in-kernel decode loop, it is not a chat surface.
const genTokens = 16

// maxPromptTokens caps the byte-tokenized prompt length so a large argument blob
// cannot make one adjudicated call run an unbounded prefill.
const maxPromptTokens = 64

// Engine is the in-kernel-model EngineDriver. The model is constructed lazily.
type Engine struct {
	once      sync.Once
	m         *model.Model
	cfg       model.Config
	q4k       bool // resident-Q4_K preload: Complete routes the dispatch decode through Session.Q4K
	schedOnce sync.Once
	sched     *NativeScheduler

	// tok is the OPTIONAL NL tokenizer (nil = byte-level default). Set ONCE at boot via
	// SetTokenizer, before the server accepts requests, then read-only on the dispatch
	// path — the same set-at-boot-then-read contract q4k uses (no lock needed).
	tok NLTokenizer
}

// New returns an Engine backed by the default synthetic config. The model itself
// is not built until the first Complete (or an explicit warmup in a test).
func New() *Engine { return &Engine{cfg: SyntheticConfig()} }

// SyntheticConfig is the small, valid, deterministic checkpoint shape the engine
// runs when no real export is configured. VocabSize 256 makes the byte->token map
// total (every input byte is a valid token id).
func SyntheticConfig() model.Config {
	return model.Config{
		HiddenSize:        64,
		NumLayers:         3,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           16,
		IntermediateSize:  128,
		VocabSize:         256,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1, // never early-stop: a tool completion decodes a fixed length
	}
}

// model builds (once) and returns the backing model. If FAK_MODEL_DIR names a real
// export it is loaded; otherwise the synthetic checkpoint is used. A load failure
// falls back to synthetic rather than wedging the engine (the dispatch path is the
// same; only the weights differ).
func (e *Engine) model() *model.Model {
	e.once.Do(func() {
		if dir := os.Getenv("FAK_MODEL_DIR"); dir != "" {
			if m, err := model.Load(dir); err == nil {
				e.m, e.cfg = m, m.Cfg
				return
			}
		}
		e.m = model.NewSynthetic(e.cfg)
	})
	return e.m
}

// Preload installs an already-constructed model as this engine's backing weights,
// claiming the once-guard so the lazy synthetic/FAK_MODEL_DIR path never runs. The
// host calls it at boot (fak serve --gguf) so the heavy weight load is part of the
// measured startup sequence rather than a lazy cost paid on the first request. The
// FIRST caller wins; a later Preload or lazy model() is a no-op.
func (e *Engine) Preload(m *model.Model) {
	if e == nil || m == nil {
		return
	}
	e.once.Do(func() { e.m, e.cfg = m, m.Cfg })
}

// Preload installs preloaded weights on the registered Default engine.
func Preload(m *model.Model) { Default.Preload(m) }

// PreloadQ4K installs a resident-Q4_K-constructed model and flags the engine so
// Complete routes the dispatch decode through the Q4_K kernel (Session.Q4K=true),
// the path P1/P2 shipped for Qwen3.6-27B (NEON SDOT int8 decode GEMV). It mirrors
// the FAK_Q4K branch in cmd/fakchat and cmd/q4kdiag: the same loader, the same
// session flags. The once-guard means a plain Preload already claimed by an earlier
// caller makes this a no-op — the host picks ONE preload path at boot.
func (e *Engine) PreloadQ4K(m *model.Model) {
	if e == nil || m == nil {
		return
	}
	e.once.Do(func() { e.m, e.cfg = m, m.Cfg; e.q4k = true })
}

// PreloadQ4K installs preloaded resident-Q4_K weights on the registered Default engine.
func PreloadQ4K(m *model.Model) { Default.PreloadQ4K(m) }

// SetTokenizer arms the in-kernel engine with a real NL tokenizer so the dispatch
// path NL-tokenizes a call's arguments (instead of byte-tokenizing them) and
// detokenizes the generated ids back to TEXT in the result payload. Call it at boot,
// before serving — it is a plain field write read lock-free on the request path. A
// nil tokenizer leaves the byte-level default intact; the LAST non-nil caller wins.
func (e *Engine) SetTokenizer(t NLTokenizer) {
	if e == nil || t == nil {
		return
	}
	e.tok = t
}

// SetTokenizer arms the registered Default engine's NL tokenizer.
func SetTokenizer(t NLTokenizer) { Default.SetTokenizer(t) }

// Caps advertises the in-kernel engine capability AND the lifecycle seam
// (EngineLifecycleCap) the Engine implements via Admit, so a consumer can negotiate
// streaming/cancel without a type assertion. A worker that doesn't know either cap
// simply never negotiates it; the engine is still selectable by id.
func (e *Engine) Caps() []abi.Capability {
	return []abi.Capability{"engine.inkernel", "engine.continuous-batching", abi.EngineLifecycleCap}
}

// Complete runs the call's arguments through a real in-kernel-model decode and
// returns the generated tokens as the result. This is the EngineDriver seam: the
// kernel folds adjudication at Submit, then dispatches an ALLOWED call here at Reap.
//
// It is now a thin one-shot shim OVER the native scheduler lifecycle (Admit): it
// drains the per-step token stream and returns the assembled turn. The decode, the
// payload, and the Meta remain byte-identical to the pre-scheduler path for one
// request, while overlapping requests share the scheduler's StepBatch loop.
func (e *Engine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	req, err := e.Admit(ctx, c)
	if err != nil {
		return nil, err
	}
	for range req.Tokens() { // one-shot caller wants only the finished turn; drain the stream
	}
	res, err := req.Result()
	if err != nil {
		return nil, err
	}
	if res != nil && res.Call == nil {
		res.Call = c // preserve the pre-shim Result.Call binding
	}
	return res, nil
}

// nativeScheduler returns the process-local continuous-batching scheduler that backs this
// Engine. The prepare hook reads Engine fields at admit time so the scheduler preserves the
// boot-selected tokenizer and resident-Q4_K mode without widening the public ABI.
func (e *Engine) nativeScheduler() *NativeScheduler {
	e.schedOnce.Do(func() {
		e.sched = newNativeScheduler(e.model(), func(ctx context.Context, c *abi.ToolCall, m *model.Model) schedPrepare {
			args := refBytes(ctx, c.Args)
			return schedPrepare{
				prompt: e.buildPrompt(c.Tool, args, m.Cfg.VocabSize),
				tok:    e.tok,
				q4k:    e.q4k,
			}
		})
	})
	return e.sched
}

// buildPrompt turns a tool name + argument bytes into a bounded prompt of token ids,
// using the NL tokenizer when one is armed (SetTokenizer) and the byte-level map
// otherwise. The NL path encodes "<tool> <args>" so distinct tools still yield distinct
// prompts, the same property the byte path has. It falls back to the byte map if the
// tokenizer errors, yields nothing, or emits an id outside the model's vocab (a
// tokenizer/model mismatch must never crash prefill) — so the NL path is purely
// additive and the byte default is preserved exactly when tok is nil.
func (e *Engine) buildPrompt(tool string, args []byte, vocab int) []int {
	if e.tok != nil {
		if ids, err := e.tok.Encode(tool + " " + string(args)); err == nil && len(ids) > 0 {
			if len(ids) > maxPromptTokens {
				ids = ids[:maxPromptTokens]
			}
			if idsWithinVocab(ids, vocab) {
				return ids
			}
		}
	}
	return tokenize(tool, args, vocab)
}

// idsWithinVocab reports whether every id is a valid index under a vocab of size
// vocab — the mismatch guard buildPrompt uses before handing NL ids to prefill.
func idsWithinVocab(ids []int, vocab int) bool {
	if vocab <= 0 {
		return false
	}
	for _, id := range ids {
		if id < 0 || id >= vocab {
			return false
		}
	}
	return true
}

// tokenize turns a tool name + argument bytes into a bounded prompt of token ids.
// There is no NL tokenizer in-tree (the model is proven at the tensor layer, not
// against a vocab), so this is a deterministic byte-level map: every byte is a
// valid id under a VocabSize>=256 checkpoint. The tool name is folded in first so
// distinct tools yield distinct prompts. An empty call still yields one token, so
// Generate always has a prefix to decode from.
func tokenize(tool string, args []byte, vocab int) []int {
	if vocab <= 0 {
		vocab = 256
	}
	ids := make([]int, 0, maxPromptTokens)
	for i := 0; i < len(tool) && len(ids) < maxPromptTokens; i++ {
		ids = append(ids, int(tool[i])%vocab)
	}
	for i := 0; i < len(args) && len(ids) < maxPromptTokens; i++ {
		ids = append(ids, int(args[i])%vocab)
	}
	if len(ids) == 0 {
		ids = append(ids, 0)
	}
	return ids
}

// refBytes materializes a Ref through the active resolver (mirrors engine.refBytes).
func refBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

// putBytes stores result bytes via the active resolver, returning an inline Ref as
// a last resort so a missing backend never drops the payload.
func putBytes(ctx context.Context, b []byte) abi.Ref {
	if res := abi.ActiveResolver(); res != nil {
		if ref, err := res.Put(ctx, b); err == nil {
			return ref
		}
	}
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}
}

// Default is the registered instance.
var Default = New()

func init() {
	abi.RegisterEngine(EngineID, Default)
	abi.RegisterCapability("engine.inkernel")
	abi.RegisterCapability("engine.continuous-batching")
	// Register the in-process model.Session adapter as the default KV-MMU enforcement
	// backend (the existing model->abi seam this package already owns). The KV-MMU
	// (internal/kvmmu) enforces its quarantine through whatever KVBackend is registered;
	// this is the in-process default (last-wins), so a remote/zero-copy KV backend can
	// override it by blank-import order with no kvmmu edit. Capability "kvbackend.v1"
	// advertises the seam to negotiation.
	abi.RegisterKVBackend(model.KVBackendFor)
	abi.RegisterCapability("kvbackend.v1")
}
