package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/trajhook"
)

// guard_context_health_test.go — issue #3099 witness: the live guard-status
// context-health line reflects the deterministic #3098 verdict and the #3096
// shed-span use-after-free count, moving as the counters move (HEALTHY -> STALL/DRIFT,
// and a shed-then-reference bumping the use-after-free count), and stays silent when
// there is no live corpus.

// TestGuardContextHealthSilentWithoutCorpus — no recorder / an empty corpus is a
// no-op, exactly like the sibling exit formatters stay quiet when their signal is
// absent (FAK_TRAJECTORY off is the default).
func TestGuardContextHealthSilentWithoutCorpus(t *testing.T) {
	if got := contextHealthLineFrom(nil); got != "" {
		t.Fatalf("nil recorder = %q, want empty", got)
	}
	if got := renderContextHealthLine(nil); got != "" {
		t.Fatalf("empty corpus = %q, want empty", got)
	}
}

// TestGuardContextHealthHealthy — a short corpus of distinct allowed calls reads
// HEALTHY, and the verdict is shown (the line is a meter, not a fire-only warning).
func TestGuardContextHealthHealthy(t *testing.T) {
	corpus := []trajectory.Turn{
		{TraceID: "sess-ok", Seq: 1, Tool: "Read", Query: "read a", Verdict: "ALLOW"},
		{TraceID: "sess-ok", Seq: 2, Tool: "Write", Query: "write b", Verdict: "ALLOW"},
	}
	got := renderContextHealthLine(corpus)
	if !strings.Contains(got, string(trajctl.SignalHealthy)) {
		t.Fatalf("healthy corpus line = %q, want it to name %s", got, trajctl.SignalHealthy)
	}
	if strings.Contains(got, "use-after-free") {
		t.Fatalf("clean corpus should not mention use-after-free: %q", got)
	}
}

// TestGuardContextHealthStallOnLoop — a repeat-failure loop (identical allowed calls
// at the loop threshold) moves the meter to STALL.
func TestGuardContextHealthStallOnLoop(t *testing.T) {
	var corpus []trajectory.Turn
	for i := 1; i <= 4; i++ { // defaultContextHealthLoopMin == 4
		corpus = append(corpus, trajectory.Turn{
			TraceID: "sess-loop", Seq: i, Tool: "Read",
			Query: "read plan", ArgsDigest: "d41d8cd9", Verdict: "ALLOW",
		})
	}
	got := renderContextHealthLine(corpus)
	if !strings.Contains(got, string(trajctl.SignalStall)) {
		t.Fatalf("looping corpus line = %q, want it to name %s", got, trajctl.SignalStall)
	}
}

// TestGuardContextHealthDriftOnDenyLoop — a trace the kernel kept refusing reads DRIFT
// via the reused high_deny_rate fold.
func TestGuardContextHealthDriftOnDenyLoop(t *testing.T) {
	corpus := []trajectory.Turn{
		{TraceID: "sess-drift", Seq: 1, Tool: "sql_exec", Query: "drop users", Verdict: "DENY"},
		{TraceID: "sess-drift", Seq: 2, Tool: "sql_exec", Query: "drop users again", Verdict: "DENY"},
	}
	got := renderContextHealthLine(corpus)
	if !strings.Contains(got, string(trajctl.SignalDrift)) {
		t.Fatalf("deny-loop corpus line = %q, want it to name %s", got, trajctl.SignalDrift)
	}
}

// TestGuardContextHealthCountsUseAfterFree — a span shed then referenced by a LATER
// turn is a context use-after-free; the count appears on the line (the shed-then-
// reference the acceptance names visibly moving the meter). This axis is independent
// of the behavioral verdict, so the line may read HEALTHY yet still carry the count.
func TestGuardContextHealthCountsUseAfterFree(t *testing.T) {
	corpus := []trajectory.Turn{
		{TraceID: "sess-uaf", Seq: 1, Tool: "compact", Verdict: "ALLOW",
			Labels: map[string]string{trajhook.LabelShedSpan: "span-42"}},
		{TraceID: "sess-uaf", Seq: 2, Tool: "Read", Query: "re-read plan", Verdict: "ALLOW",
			Labels: map[string]string{trajhook.LabelRefSpan: "span-42", trajhook.LabelRefKind: "reread"}},
	}
	got := renderContextHealthLine(corpus)
	if !strings.Contains(got, "1 context use-after-free") {
		t.Fatalf("shed-then-reference corpus line = %q, want a use-after-free count", got)
	}
}

// TestGuardContextHealthDetourOverrunWins — DETOUR_OVERRUN outranks a co-occurring
// DRIFT, pinning the trajctl classify precedence the worst-first pick uses.
func TestGuardContextHealthDetourOverrunWins(t *testing.T) {
	overBudget := []trajhook.Finding{
		{TraceID: "a", Label: string(trajctl.SignalDrift), Score: 1.0},
		{TraceID: "b", Label: string(trajctl.SignalDetourOverrun), Score: 0.1},
	}
	worst, ok := worstContextHealthVerdict(overBudget)
	if !ok || worst.Label != string(trajctl.SignalDetourOverrun) {
		t.Fatalf("worst = %+v (ok=%v), want DETOUR_OVERRUN despite lower score", worst, ok)
	}
}

// TestGuardContextHealthReadsLiveRecorder — the end-to-end wiring: a fresh recorder
// fed real ABI decision events (the shape the kernel emits during a guard session)
// folds through Turns() into the same STALL verdict. This proves the line reads the
// LIVE corpus, not a stale offline file.
func TestGuardContextHealthReadsLiveRecorder(t *testing.T) {
	rec := trajectory.New()
	for i := 0; i < 4; i++ {
		call := &abi.ToolCall{
			TraceID: "live-loop", Tool: "Read",
			Meta: map[string]string{"query": "read plan"},
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte("read plan")},
		}
		rec.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &abi.Verdict{Kind: abi.VerdictAllow}})
	}
	got := contextHealthLineFrom(rec)
	if !strings.Contains(got, string(trajctl.SignalStall)) {
		t.Fatalf("live-recorder line = %q, want it to name %s", got, trajctl.SignalStall)
	}
}
