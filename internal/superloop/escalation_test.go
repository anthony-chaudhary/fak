package superloop

import "testing"

func TestEscalateNoProgressIsBoundedAndResettable(t *testing.T) {
	want := []string{"dispatch", "retry", "replan", "unblock", "unstick", "operator-decision", "operator-decision"}
	for streak, name := range want {
		got := EscalateNoProgress(streak)
		if got.Name != name || got.Command == "" {
			t.Fatalf("streak %d = %+v, want %s with command", streak, got, name)
		}
	}
	if got := EscalateNoProgress(0); got.Name != "dispatch" {
		t.Fatalf("progress reset = %+v", got)
	}
}
