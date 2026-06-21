package modelengine

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // register the Ref resolver (CAS backend)
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// genResult is the JSON shape Complete writes to the result payload.
type genResult struct {
	Tool   string `json:"tool"`
	Engine string `json:"engine"`
	Model  string `json:"model"`
	Tokens []int  `json:"generated_tokens"`
}

func inlineCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

func decodeGen(t *testing.T, ctx context.Context, r *abi.Result) genResult {
	t.Helper()
	b := refBytes(ctx, r.Payload)
	var g genResult
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("result payload is not the generated-token JSON: %v (%s)", err, b)
	}
	return g
}

// TestCompleteRunsRealDecode proves Complete dispatches into the in-kernel model:
// it returns a StatusOK result whose payload carries exactly genTokens generated
// ids, all valid under the checkpoint vocab, with the engine + token accounting in
// Meta. A real forward pass, not a stub echo.
func TestCompleteRunsRealDecode(t *testing.T) {
	ctx := context.Background()
	e := New()
	r, err := e.Complete(ctx, inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if r.Status != abi.StatusOK {
		t.Fatalf("status = %v, want OK", r.Status)
	}
	if r.Meta["engine"] != EngineID {
		t.Fatalf("meta engine = %q, want %q", r.Meta["engine"], EngineID)
	}
	g := decodeGen(t, ctx, r)
	if g.Engine != EngineID {
		t.Fatalf("payload engine = %q, want %q", g.Engine, EngineID)
	}
	if len(g.Tokens) != genTokens {
		t.Fatalf("generated %d tokens, want %d", len(g.Tokens), genTokens)
	}
	vocab := e.model().Cfg.VocabSize
	for i, id := range g.Tokens {
		if id < 0 || id >= vocab {
			t.Fatalf("token %d = %d out of vocab [0,%d)", i, id, vocab)
		}
	}
}

// TestDecodeIsDeterministicAndInputDriven proves the model is genuinely running:
// the same call decodes the same tokens twice (deterministic forward pass), and a
// DIFFERENT call decodes a DIFFERENT sequence (the input drives the model — not a
// constant). A non-vacuous "it really ran" witness.
func TestDecodeIsDeterministicAndInputDriven(t *testing.T) {
	ctx := context.Background()
	e := New()

	a1 := decodeGen(t, ctx, mustComplete(t, ctx, e, inlineCall("get_user_details", `{"id":1}`)))
	a2 := decodeGen(t, ctx, mustComplete(t, ctx, e, inlineCall("get_user_details", `{"id":1}`)))
	if !equalInts(a1.Tokens, a2.Tokens) {
		t.Fatalf("same call must decode deterministically:\n %v\n %v", a1.Tokens, a2.Tokens)
	}

	b := decodeGen(t, ctx, mustComplete(t, ctx, e, inlineCall("list_all_airports", `{"region":"EU"}`)))
	if equalInts(a1.Tokens, b.Tokens) {
		t.Fatalf("distinct prompts must decode distinctly (input is not driving the model): %v", a1.Tokens)
	}
}

// allowAll is a permissive adjudicator so the witness loop's calls reach the engine.
type allowAll struct{}

func (allowAll) Caps() []abi.Capability { return nil }
func (allowAll) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test"}
}

// TestAgentLoopOnInKernelEngine is the issue-#14 witness: a multi-call agent-style
// loop whose kernel is bound to the "inkernel" engine completes every turn on the
// in-kernel model. Each tool call crosses the real syscall boundary (adjudicate ->
// dispatch -> in-kernel decode -> admit) and comes back with a model-produced
// result. This is `fak agent --engine inkernel` exercised at the kernel seam.
func TestAgentLoopOnInKernelEngine(t *testing.T) {
	ctx := context.Background()
	abi.RegisterAdjudicator(0, allowAll{}) // affirmative allow so calls dispatch

	k := kernel.New(EngineID) // bind the kernel to the in-kernel model engine
	k.SetVDSO(false)          // every turn must actually hit the engine, no fast-path

	// A small scripted tool-calling "conversation": the planner's turns, each a tool
	// call the loop dispatches through the kernel.
	turns := []*abi.ToolCall{
		inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`),
		inlineCall("get_reservation_details", `{"id":"ABC123"}`),
		inlineCall("calculate", `{"expr":"2+2"}`),
	}

	served := 0
	for i, tc := range turns {
		r, v := k.Syscall(ctx, tc)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("turn %d: verdict = %v, want Allow", i, v.Kind)
		}
		if r == nil || r.Status != abi.StatusOK {
			t.Fatalf("turn %d: result = %+v, want StatusOK", i, r)
		}
		if r.Meta["engine"] != EngineID {
			t.Fatalf("turn %d: served by %q, want the in-kernel engine %q", i, r.Meta["engine"], EngineID)
		}
		g := decodeGen(t, ctx, r)
		if len(g.Tokens) != genTokens {
			t.Fatalf("turn %d: in-kernel decode produced %d tokens, want %d", i, len(g.Tokens), genTokens)
		}
		served++
	}
	if served != len(turns) {
		t.Fatalf("loop completed %d/%d turns on the in-kernel engine", served, len(turns))
	}

	// The kernel's own tally must confirm the engine was actually driven once per turn.
	if ec := k.Counters().EngineCalls; ec != int64(len(turns)) {
		t.Fatalf("EngineCalls = %d, want %d (the in-kernel engine must serve every turn)", ec, len(turns))
	}
}

func mustComplete(t *testing.T, ctx context.Context, e *Engine, c *abi.ToolCall) *abi.Result {
	t.Helper()
	r, err := e.Complete(ctx, c)
	if err != nil {
		t.Fatalf("Complete(%s): %v", c.Tool, err)
	}
	return r
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestPreloadClaimsTheLazyGuard proves Preload installs given weights and that the
// lazy synthetic/FAK_MODEL_DIR path never overwrites them: the engine dispatches
// against the preloaded model, and a first-caller-wins guard ignores later loads.
func TestPreloadClaimsTheLazyGuard(t *testing.T) {
	// A checkpoint with a distinctive vocab so we can tell it apart from the engine's
	// default SyntheticConfig (VocabSize 256).
	cfg := SyntheticConfig()
	cfg.VocabSize = 257
	pre := model.NewSynthetic(cfg)

	e := New()
	e.Preload(pre)
	if got := e.model(); got != pre {
		t.Fatalf("model() did not return the preloaded checkpoint")
	}
	// once-guard already claimed: a later Preload is a no-op.
	e.Preload(model.NewSynthetic(SyntheticConfig()))
	if got := e.model(); got != pre {
		t.Fatalf("a second Preload overwrote the first; first-caller must win")
	}
	if e.model().Cfg.VocabSize != 257 {
		t.Fatalf("engine cfg not adopted from preloaded model: vocab=%d", e.model().Cfg.VocabSize)
	}
}

// TestPreloadNilIsSafe asserts a nil model leaves the lazy path intact.
func TestPreloadNilIsSafe(t *testing.T) {
	e := New()
	e.Preload(nil)
	if m := e.model(); m == nil { // falls through to synthetic
		t.Fatalf("nil Preload wedged the engine; lazy synthetic should still build")
	}
}
