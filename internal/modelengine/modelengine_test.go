package modelengine

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // register the Ref resolver (CAS backend)
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/model"

	"github.com/anthony-chaudhary/fak/internal/refutil"
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
	b := refutil.Bytes(ctx, r.Payload)
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

func TestModelNameReportsSyntheticVsPreloadedHonestly(t *testing.T) {
	// 1. Synthetic engine reports "smollm2-synthetic"
	eSynth := New()
	if got := eSynth.ModelName(); got != "smollm2-synthetic" {
		t.Fatalf("synthetic ModelName = %q, want \"smollm2-synthetic\"", got)
	}

	// 2. Preloaded real model reports "smollm2-inkernel"
	ePre := New()
	ePre.Preload(model.NewSynthetic(SyntheticConfig()))
	if got := ePre.ModelName(); got != "smollm2-inkernel" {
		t.Fatalf("preloaded ModelName = %q, want \"smollm2-inkernel\"", got)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestEngineResultsIdentifyOwningModel(t *testing.T) {
	const childEnv = "FAK_TEST_ENGINE_RESULT_IDENTITY"
	if os.Getenv(childEnv) != "1" {
		// Default's lazy model must start untouched, independent of suite order or
		// a developer's model export. Do not replace the registered global engine.
		exe, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "-test.run=^TestEngineResultsIdentifyOwningModel$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), childEnv+"=1", "FAK_MODEL_DIR=", "FAK_NATIVE_KV_MAX_BLOCKS=0", "FAK_NATIVE_MAX_RUNNING=", "FAK_NATIVE_IN_BATCH_PREFIX_DEDUP=")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("isolated engine identity witness: %v\n%s", err, out)
		}
		t.Logf("%s", out)
		return
	}

	cfg := SyntheticConfig()
	cfg.HiddenSize, cfg.NumLayers = 8, 1
	cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 2, 1, 4
	cfg.IntermediateSize = 16
	m := model.NewSynthetic(cfg)
	e := New()
	e.Preload(m)
	want := e.ModelName()
	defaultName := Default.ModelName()
	if want == defaultName || defaultName != "smollm2-synthetic" {
		t.Fatalf("precondition: receiver model = %q, default = %q; want distinct synthetic default", want, defaultName)
	}
	standalone := NewNativeScheduler(m)
	t.Cleanup(func() {
		for _, s := range []*NativeScheduler{e.nativeScheduler(), Default.nativeScheduler(), standalone} {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			err := s.CloseAndWait(ctx)
			cancel()
			if err != nil {
				t.Errorf("CloseAndWait: %v", err)
			}
		}
	})

	for _, tc := range []struct {
		name  string
		e     abi.LifecycleEngine
		model string
	}{
		{"preloaded", e, want},
		{"default", Default, defaultName},
		{"standalone", standalone, defaultName},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			call := inlineCall("identity", `{"input":"x"}`)
			check := func(api string, r *abi.Result) genResult {
				t.Helper()
				if r == nil || r.Status != abi.StatusOK {
					t.Fatalf("%s result = %v, want StatusOK", api, r)
				}
				g := decodeGen(t, ctx, r)
				if g.Model != tc.model {
					t.Errorf("%s payload model = %q, want owning model %q", api, g.Model, tc.model)
				}
				if g.Tool != call.Tool || g.Engine != EngineID || len(g.Tokens) != genTokens {
					t.Errorf("%s payload contract changed: %+v", api, g)
				}
				if r.Meta["engine"] != EngineID || r.Meta["output_tokens"] != strconv.Itoa(genTokens) || r.Meta["input_tokens"] != strconv.Itoa(len(call.Tool)+len(call.Args.Inline)) {
					t.Errorf("%s token metadata changed: %v", api, r.Meta)
				}
				return g
			}
			res, err := tc.e.Complete(ctx, call)
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			complete := check("Complete", res)
			req, err := tc.e.Admit(ctx, call)
			if err != nil {
				t.Fatalf("Admit: %v", err)
			}
			var streamed []int
			for tok := range req.Tokens() {
				streamed = append(streamed, tok.ID)
			}
			res, err = req.Result()
			if err != nil {
				t.Fatalf("Result: %v", err)
			}
			admit := check("Admit", res)
			if !equalInts(admit.Tokens, streamed) || !equalInts(complete.Tokens, streamed) {
				t.Errorf("generated tokens differ: Complete=%v Admit=%v stream=%v", complete.Tokens, admit.Tokens, streamed)
			}
		})
	}
}

func TestNativePreemptionPolicyDefaultsMaxBlocksFromProbe(t *testing.T) {
	t.Setenv("FAK_NATIVE_KV_BLOCK_TOKENS", "")
	probe := func() (uint64, bool) { return 10 << 20, true }
	const bytesPerToken = 1024 // 16-token blocks -> 16 KiB per block

	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "")
	if got := nativePreemptionPolicyFromEnvWithProbe(probe, bytesPerToken).MaxBlocks; got != 576 {
		t.Fatalf("env unset + probe: MaxBlocks = %d, want 576", got)
	}
	unavailable := func() (uint64, bool) { return 0, false }
	if got := nativePreemptionPolicyFromEnvWithProbe(unavailable, bytesPerToken).MaxBlocks; got != 0 {
		t.Fatalf("env unset + probe unavailable: MaxBlocks = %d, want 0", got)
	}
	if got := nativePreemptionPolicyFromEnvWithProbe(nil, bytesPerToken).MaxBlocks; got != 0 {
		t.Fatalf("env unset + nil probe: MaxBlocks = %d, want 0", got)
	}

	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "0")
	if got := nativePreemptionPolicyFromEnvWithProbe(probe, bytesPerToken).MaxBlocks; got != 0 {
		t.Fatalf("env=0 + probe: MaxBlocks = %d, want 0", got)
	}
	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "7")
	if got := nativePreemptionPolicyFromEnvWithProbe(probe, bytesPerToken).MaxBlocks; got != 7 {
		t.Fatalf("env=7 + probe: MaxBlocks = %d, want 7", got)
	}
}
