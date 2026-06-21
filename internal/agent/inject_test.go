package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// PLUMBING-ONLY tests.
//
// READ THIS FIRST: the offline MockPlanner is SCRIPTED. It walks a fixed
// sequence keyed off the conversation it has SEEN; it does NOT adaptively
// re-plan after an ARBITRARY injected error (it only retries the one
// convert_currency malform its own script anticipates). So these tests prove the
// DECORATOR PLUMBING — that injection corrupts args, that the kernel/grammar path
// repairs an alias on the fak arm while the baseline tool rejects it — NOT that a
// model fires "+1 retry turn per error". The empirical "+1 turn" claim can only be
// WITNESSED with a LIVE model (see turntax-injection-live.json), because only a
// live model genuinely re-plans on an error the script did not foresee.
// ---------------------------------------------------------------------------

// stubPlanner is a minimal Planner that emits a single, pre-set tool call. It lets
// us test the decorator's corruption directly, independent of the mock's own
// scripted alias logic.
type stubPlanner struct {
	tool string
	args string
}

func (s *stubPlanner) Model() string { return "stub" }
func (s *stubPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	return &Completion{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "c1", Type: "function", Function: Func{Name: s.tool, Arguments: s.args}},
		}},
		FinishReason: "tool_calls",
		Usage:        Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}, nil
}

// TestInjectAliasCorruptsCanonicalArgs confirms the decorator renames the
// canonical convert_currency args to the grammar aliases the kernel repairs.
func TestInjectAliasCorruptsCanonicalArgs(t *testing.T) {
	inner := &stubPlanner{tool: toolConvert, args: `{"from_currency":"USD","to_currency":"EUR","amount":240}`}
	ip := NewInjectingPlanner(inner, 1)

	comp, err := ip.Complete(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	got := comp.Message.ToolCalls[0].Function.Arguments

	var m map[string]any
	if err := json.Unmarshal([]byte(got), &m); err != nil {
		t.Fatalf("corrupted args not valid JSON: %v (%s)", err, got)
	}
	if _, ok := m["from_currency"]; ok {
		t.Errorf("expected from_currency to be renamed away; still present in %s", got)
	}
	if _, ok := m["to_currency"]; ok {
		t.Errorf("expected to_currency to be renamed away; still present in %s", got)
	}
	if m["from"] != "USD" || m["to"] != "EUR" {
		t.Errorf("expected aliases from=USD,to=EUR; got %s", got)
	}
	if *ip.Injected != 1 {
		t.Errorf("expected Injected counter == 1; got %d", *ip.Injected)
	}
}

// TestInjectedAliasIsRejectedByBaselineToolButRepairedByKernel is the heart of the
// plumbing proof: the SAME corrupted args ERROR on the naive baseline tool but are
// REPAIRED in-syscall on the fak (kernel) path.
func TestInjectedAliasIsRejectedByBaselineToolButRepairedByKernel(t *testing.T) {
	// 1. The corrupted (aliased) args, as the decorator would produce them.
	corrupted, ok := corruptArgs(`{"from_currency":"USD","to_currency":"EUR","amount":240}`, InjectAlias)
	if !ok {
		t.Fatal("expected corruptArgs to report a change")
	}

	// 2. BASELINE arm contract: the tool validates its own inputs and rejects the
	//    aliased args (this is execNaive's error path — the +1 retry trigger).
	var args map[string]any
	_ = json.Unmarshal([]byte(corrupted), &args)
	_, isErr := execTool(toolConvert, args)
	if !isErr {
		t.Error("expected the baseline tool to REJECT aliased convert_currency args (the retry trigger)")
	}

	// 3. FAK arm contract: the same corrupted call, run through the kernel, is
	//    repaired in-syscall (grammar Transform) and succeeds with NO tool error.
	Configure()
	var fakLog []traceEvent
	m := ArmMetrics{Arm: "fak"}
	_ = m
	stub := &stubPlanner{tool: toolConvert, args: corrupted}
	ip := &InjectingPlanner{Inner: stub, Prob: 0, Kind: InjectNone} // already corrupted; don't double-inject
	fakM, err := RunArm(context.Background(), ip, "convert please", true, 3, &fakLog)
	if err != nil {
		t.Fatalf("fak arm: %v", err)
	}
	if fakM.Repairs < 1 {
		t.Errorf("expected the kernel to REPAIR the aliased call in-syscall (>=1 repair); got %d", fakM.Repairs)
	}
	if fakM.ToolErrors != 0 {
		t.Errorf("expected NO tool error on the fak arm (repair, not retry); got %d", fakM.ToolErrors)
	}
}

// TestInjectDeterministicUnderSeed confirms two decorators with the same seed make
// the same corruption decision on the same inner response (reproducibility — the
// property the live measurement relies on to compare reruns).
func TestInjectDeterministicUnderSeed(t *testing.T) {
	mk := func(prob float64) *InjectingPlanner {
		n := 0
		return &InjectingPlanner{
			Inner: &stubPlanner{tool: toolConvert, args: `{"from_currency":"USD","to_currency":"EUR","amount":240}`},
			Prob:  prob, Seed: 42, Kind: InjectAlias, Target: toolConvert, Injected: &n,
		}
	}
	// Probabilistic gate (Prob=0.5) must be identical across two same-seed runs.
	a := mk(0.5)
	b := mk(0.5)
	ca, _ := a.Complete(context.Background(), nil, nil)
	cb, _ := b.Complete(context.Background(), nil, nil)
	if ca.Message.ToolCalls[0].Function.Arguments != cb.Message.ToolCalls[0].Function.Arguments {
		t.Errorf("non-deterministic under fixed seed: %s vs %s",
			ca.Message.ToolCalls[0].Function.Arguments, cb.Message.ToolCalls[0].Function.Arguments)
	}
	if *a.Injected != *b.Injected {
		t.Errorf("non-deterministic injection count: %d vs %d", *a.Injected, *b.Injected)
	}
}

// TestInjectDropRemovesRequiredField confirms the drop mode deletes a required
// field, producing a HARD error neither arm can repair (the control case).
func TestInjectDropRemovesRequiredField(t *testing.T) {
	corrupted, ok := corruptArgs(`{"from_currency":"USD","to_currency":"EUR","amount":240}`, InjectDrop)
	if !ok {
		t.Fatal("expected drop to change the args")
	}
	if strings.Contains(corrupted, "from_currency") {
		t.Errorf("expected from_currency dropped; got %s", corrupted)
	}
	var args map[string]any
	_ = json.Unmarshal([]byte(corrupted), &args)
	if _, isErr := execTool(toolConvert, args); !isErr {
		t.Error("expected a dropped required field to error on the tool")
	}
}

// TestRunInjectionOfflinePlumbing drives the FULL A/B with the decorator over the
// scripted mock and asserts the harness computes a coherent measurement struct.
//
// CRITICAL: this is PLUMBING ONLY. The mock is scripted and emits an aliased
// convert call on its FIRST attempt by design, so the decorator (targeting
// convert_currency) finds no canonical key to rename on that first call and does
// not double-corrupt it. The mock retries with CANONICAL args after seeing the
// error — and the decorator corrupts THAT retry back to an alias. The point of
// this test is NOT to show a model adaptively retrying (the mock's retry is
// scripted, not learned); it is to show the runner produces a well-formed
// InjectionResult with the live flag false. The empirical "+1 turn/error" is
// asserted to be UNWITNESSED offline.
func TestRunInjectionOfflinePlumbing(t *testing.T) {
	inner := NewMockPlanner("plumbing")
	res, _, err := RunInjection(context.Background(), inner, DefaultTask, 14, 7)
	if err != nil {
		t.Fatalf("run injection: %v", err)
	}
	if res.AB == nil {
		t.Fatal("expected an embedded A/B result")
	}
	if res.Live {
		t.Error("offline mock must NOT be flagged live")
	}
	// The measurement struct must be internally coherent: fields mirror the arms.
	if res.FakRepairs != res.AB.Fak.Repairs {
		t.Errorf("FakRepairs %d != arm %d", res.FakRepairs, res.AB.Fak.Repairs)
	}
	if res.BaselineToolErrors != res.AB.Baseline.ToolErrors {
		t.Errorf("BaselineToolErrors %d != arm %d", res.BaselineToolErrors, res.AB.Baseline.ToolErrors)
	}
	// Honest assertion: the offline scripted mock does NOT witness adaptive
	// retry. RetrySupported being true here would only reflect the mock's scripted
	// retry, not a real model — we record the result but do not treat the offline
	// number as evidence for the benchmark. We only require the field to be
	// computed without panicking and the note to flag comparability.
	if res.AB.BothCompleted && res.BaselineToolErrors > 0 && res.RetryTurnsPerError <= 0 {
		t.Errorf("with both completed and baseline errors > 0, expected a positive retry-turns-per-error; got %g",
			res.RetryTurnsPerError)
	}
	t.Logf("OFFLINE PLUMBING: injected=%d fak_repairs=%d base_tool_errors=%d retry/err=%.2f supported=%v note=%q",
		res.Injected, res.FakRepairs, res.BaselineToolErrors, res.RetryTurnsPerError, res.RetrySupported, res.Note)
}
