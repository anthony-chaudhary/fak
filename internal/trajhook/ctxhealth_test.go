package trajhook

import (
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// contextHealthVocabulary is the closed trajctl signal set the verdict labels
// must stay inside — the #3098 acceptance cross-check against
// internal/trajctl/curve.go. Any other label is a new signal string, which the
// issue forbids.
var contextHealthVocabulary = map[string]bool{
	string(trajctl.SignalHealthy):       true,
	string(trajctl.SignalStall):         true,
	string(trajctl.SignalDrift):         true,
	string(trajctl.SignalDetourOverrun): true,
}

// runContextHealth registers the scorer the way an application would and runs
// it over the corpus, returning the per-trace verdicts.
func runContextHealth(t *testing.T, cfg ContextHealthConfig, corpus []trajectory.Turn) map[string]Finding {
	t.Helper()
	reg := NewRegistry()
	reg.RegisterCorpus("context_health", ContextHealth(cfg))
	verdicts := map[string]Finding{}
	for _, f := range reg.Run(corpus) {
		if _, dup := verdicts[f.TraceID]; dup {
			t.Fatalf("trace %s got more than one verdict", f.TraceID)
		}
		if !contextHealthVocabulary[f.Label] {
			t.Fatalf("trace %s emitted %q — a signal string outside the closed trajctl vocabulary", f.TraceID, f.Label)
		}
		verdicts[f.TraceID] = f
	}
	return verdicts
}

// TestContextHealth_RealCorpusVerdicts is the #3098 acceptance gate: running
// the scorer against the shipped real corpus yields exactly one verdict per
// trace, each in the existing vocabulary, deterministically. sess-c (all three
// destructive SQL turns refused) reads as DRIFT via the reused high_deny_rate
// fold; sess-a and sess-b carry no stall/drift/overrun evidence.
func TestContextHealth_RealCorpusVerdicts(t *testing.T) {
	f, err := os.Open(sampleCorpus)
	if err != nil {
		t.Skipf("sample corpus not found (run from repo root): %v", err)
	}
	defer f.Close()
	rec, n, err := trajectory.ImportFrom(f)
	if err != nil || n == 0 {
		t.Fatalf("import sample corpus: n=%d err=%v", n, err)
	}
	corpus := rec.Turns()

	verdicts := runContextHealth(t, ContextHealthConfig{}, corpus)
	want := map[string]string{
		"sess-a": string(trajctl.SignalHealthy),
		"sess-b": string(trajctl.SignalHealthy),
		"sess-c": string(trajctl.SignalDrift),
	}
	if len(verdicts) != len(want) {
		t.Fatalf("got %d verdicts, want one per trace (%d)", len(verdicts), len(want))
	}
	for trace, sig := range want {
		if got := verdicts[trace].Label; got != sig {
			t.Errorf("trace %s = %q, want %q (reason: %s)", trace, got, sig, verdicts[trace].Reason)
		}
	}

	// Deterministic: the same corpus yields the same verdicts on a re-run.
	again := runContextHealth(t, ContextHealthConfig{}, corpus)
	if !reflect.DeepEqual(verdicts, again) {
		t.Fatalf("verdicts changed across runs:\n first %+v\nsecond %+v", verdicts, again)
	}
}

// TestContextHealth_StallShape — a trace of identical allowed calls at the
// loop threshold reads as STALL (busy but not moving); one call short of the
// threshold stays HEALTHY.
func TestContextHealth_StallShape(t *testing.T) {
	loop := func(n int) []trajectory.Turn {
		corpus := make([]trajectory.Turn, 0, n)
		for i := 1; i <= n; i++ {
			corpus = append(corpus, trajectory.Turn{
				TraceID: "sess-loop", Seq: i, Tool: "Read",
				Query: "read docs/plan.md", ArgsDigest: "d41d8cd9", Verdict: "ALLOW",
			})
		}
		return corpus
	}

	verdicts := runContextHealth(t, ContextHealthConfig{}, loop(defaultContextHealthLoopMin))
	if got := verdicts["sess-loop"].Label; got != string(trajctl.SignalStall) {
		t.Fatalf("looping trace = %q, want %q", got, trajctl.SignalStall)
	}
	verdicts = runContextHealth(t, ContextHealthConfig{}, loop(defaultContextHealthLoopMin-1))
	if got := verdicts["sess-loop"].Label; got != string(trajctl.SignalHealthy) {
		t.Fatalf("sub-threshold trace = %q, want %q", got, trajctl.SignalHealthy)
	}
}

// TestContextHealth_DetourOverrunPrecedence — a trace past its declared turn
// budget reads as DETOUR_OVERRUN even when deny evidence also fired, pinning
// the trajctl classify precedence (DETOUR_OVERRUN > DRIFT > STALL); without a
// declared budget the same trace reads as DRIFT.
func TestContextHealth_DetourOverrunPrecedence(t *testing.T) {
	corpus := []trajectory.Turn{
		{TraceID: "sess-over", Seq: 1, Tool: "sql_exec", Query: "drop users", Verdict: "DENY"},
		{TraceID: "sess-over", Seq: 2, Tool: "sql_exec", Query: "drop users again", Verdict: "DENY"},
		{TraceID: "sess-over", Seq: 3, Tool: "sql_exec", Query: "truncate users", Verdict: "QUARANTINE"},
	}

	verdicts := runContextHealth(t, ContextHealthConfig{TurnBudget: 2}, corpus)
	if got := verdicts["sess-over"].Label; got != string(trajctl.SignalDetourOverrun) {
		t.Fatalf("over-budget trace = %q, want %q", got, trajctl.SignalDetourOverrun)
	}
	verdicts = runContextHealth(t, ContextHealthConfig{}, corpus)
	if got := verdicts["sess-over"].Label; got != string(trajctl.SignalDrift) {
		t.Fatalf("refused trace = %q, want %q", got, trajctl.SignalDrift)
	}
}
