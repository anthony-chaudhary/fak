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
// materializes the call's argument bytes, byte-tokenizes them into a prompt, and
// runs a REAL greedy Prefill+Step decode over a kernel-owned KV cache
// (model.Session.Generate) — the exact cache path the HF-oracle-verified model
// uses. The result payload carries the generated token ids + token accounting.
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
	"encoding/json"
	"os"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// EngineID is the registered id the kernel selects this backend by.
const EngineID = "inkernel"

// genTokens is how many tokens a single tool-call completion decodes. Small: the
// engine demonstrates the live in-kernel decode loop, it is not a chat surface.
const genTokens = 16

// maxPromptTokens caps the byte-tokenized prompt length so a large argument blob
// cannot make one adjudicated call run an unbounded prefill.
const maxPromptTokens = 64

// Engine is the in-kernel-model EngineDriver. The model is constructed lazily.
type Engine struct {
	once sync.Once
	m    *model.Model
	cfg  model.Config
	q4k  bool // resident-Q4_K preload: Complete routes the dispatch decode through Session.Q4K
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

// Caps advertises the in-kernel engine capability. A worker that doesn't know it
// simply never negotiates it; the engine is still selectable by id.
func (e *Engine) Caps() []abi.Capability { return []abi.Capability{"engine.inkernel"} }

// Complete runs the call's arguments through a real in-kernel-model decode and
// returns the generated tokens as the result. This is the EngineDriver seam: the
// kernel folds adjudication at Submit, then dispatches an ALLOWED call here at Reap.
func (e *Engine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	m := e.model()
	in := refBytes(ctx, c.Args)
	prompt := tokenize(c.Tool, in, m.Cfg.VocabSize)

	sess := m.NewSession()
	if e.q4k {
		// Resident-Q4_K preload: engage the Q4_K decode kernel (the q4_k_m majority
		// streams raw, Q6_K minority via Q8). Mirrors cmd/fakchat's s.Q4K = q4kLoad.
		sess.Quant = true
		sess.Q4K = true
	}
	gen := sess.Generate(prompt, genTokens)

	body, _ := json.Marshal(struct {
		Tool   string `json:"tool"`
		Engine string `json:"engine"`
		Model  string `json:"model"`
		Tokens []int  `json:"generated_tokens"`
	}{
		Tool:   c.Tool,
		Engine: EngineID,
		Model:  "smollm2-inkernel",
		Tokens: gen,
	})

	ref := putBytes(ctx, body)
	return &abi.Result{
		Call:    c,
		Payload: ref,
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"engine":        EngineID,
			"input_tokens":  strconv.Itoa(len(prompt)),
			"output_tokens": strconv.Itoa(len(gen)),
		},
	}, nil
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
	// Register the in-process model.Session adapter as the default KV-MMU enforcement
	// backend (the existing model->abi seam this package already owns). The KV-MMU
	// (internal/kvmmu) enforces its quarantine through whatever KVBackend is registered;
	// this is the in-process default (last-wins), so a remote/zero-copy KV backend can
	// override it by blank-import order with no kvmmu edit. Capability "kvbackend.v1"
	// advertises the seam to negotiation.
	abi.RegisterKVBackend(model.KVBackendFor)
	abi.RegisterCapability("kvbackend.v1")
}
