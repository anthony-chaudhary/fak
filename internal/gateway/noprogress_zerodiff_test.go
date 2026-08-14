package gateway

import (
	"encoding/json"
	"testing"
)

func TestWorktreeNoProgressStopsVaryingOutputZeroEffectLoop(t *testing.T) {
	d := NewWorktreeNoProgress(3)
	outputs := []string{"clock=10:00:01", "clock=10:00:02", "clock=10:00:03", "clock=10:00:04"}
	var got NoProgressVerdict
	for _, output := range outputs {
		got = d.Observe(NoProgressSample{Tool: "shell_command", OutputDigest: output, WorktreeDigest: "tree-A"})
	}
	if got.Kind != NoProgressWorktreeUnchanged || got.Retryable || got.ConsecutiveTurns != 3 {
		t.Fatalf("varying-output zero-effect loop not stopped: %+v", got)
	}
}

func TestWorktreeNoProgressResetsAfterObservedEffect(t *testing.T) {
	d := NewWorktreeNoProgress(2)
	d.Observe(NoProgressSample{OutputDigest: "one", WorktreeDigest: "tree-A"})
	stalled := d.Observe(NoProgressSample{OutputDigest: "two", WorktreeDigest: "tree-A"})
	if stalled.Kind != NoProgressWorktreeUnchanged || stalled.ConsecutiveTurns != 1 {
		t.Fatalf("precondition: %+v", stalled)
	}
	got := d.Observe(NoProgressSample{OutputDigest: "three", WorktreeDigest: "tree-B"})
	if got.Kind != NoProgressContinue || !got.Retryable || got.ConsecutiveTurns != 0 {
		t.Fatalf("witnessed worktree effect must reset the loop: %+v", got)
	}
}

func TestWorktreeNoProgressPreservesOutputRepeatReason(t *testing.T) {
	d := NewWorktreeNoProgress(1)
	d.Observe(NoProgressSample{OutputDigest: "same", WorktreeDigest: "tree-A"})
	got := d.Observe(NoProgressSample{OutputDigest: "same", WorktreeDigest: "tree-A"})
	if got.Kind != NoProgressOutputRepeat || got.Retryable {
		t.Fatalf("repeated output should retain the specific verdict: %+v", got)
	}
}

func TestWorktreeNoProgressVerdictJSONIsMachineReadable(t *testing.T) {
	d := NewWorktreeNoProgress(1)
	d.Observe(NoProgressSample{Tool: " edit ", OutputDigest: "one", WorktreeDigest: "tree-A"})
	got := d.Observe(NoProgressSample{Tool: " edit ", OutputDigest: "two", WorktreeDigest: "tree-A"})
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"kind":"worktree_unchanged","tool":"edit","consecutive_turns":1,"threshold":1,"retryable":false}`
	if string(b) != want {
		t.Fatalf("verdict JSON = %s, want %s", b, want)
	}
}
