package agent

import (
	"strings"
	"testing"
)

func TestProgressResultSummaryBoundsAndRedactsRead(t *testing.T) {
	if got := progressResultSummary("Read", `{"content":"secret"}`); got != "" {
		t.Fatalf("Read summary=%q", got)
	}
	got := progressResultSummary("Bash", `{"stdout":"ok\n","stderr":"","exit_code":0}`)
	if got != "exit 0: ok" {
		t.Fatalf("Bash summary=%q", got)
	}
	got = progressResultSummary("Edit", strings.Repeat("x", maxProgressSummaryBytes+20))
	if len(got) <= maxProgressSummaryBytes || !strings.HasSuffix(got, "[truncated]") {
		t.Fatalf("bounded summary len=%d suffix=%q", len(got), got[len(got)-20:])
	}
}
