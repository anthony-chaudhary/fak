package taskmgr

import (
	"bytes"
	"strings"
	"testing"
)

func TestRenderQueueCapturedStatesAndAttemptDrilldown(t *testing.T) {
	issues := []QueueIssue{
		{Number: 1, Title: "Ready leaf", State: "OPEN", Body: "## Outcome\nShip readable output.\n## Witness\nCaptured render.", Labels: []IssueLabel{{"priority/P1"}, {"gen/now"}, {"lane/taskmgr"}}},
		{Number: 2, Title: "Held leaf", State: "OPEN", Body: "## Parent\n#9\n## Outcome\nWait safely.\n## Dependencies\nRequires #1.\n## Witness\nDependency read-back.", Labels: []IssueLabel{{"priority/P2"}, {"gen/next"}, {"lane/taskmgr"}}},
		{Number: 3, Title: "Active leaf", State: "OPEN", Body: "## Outcome\nRun now.\n## Witness\nLive read-back.", Labels: []IssueLabel{{"priority/P1"}, {"gen/now"}, {"lane/taskmgr"}}},
		{Number: 4, Title: "Done leaf", State: "CLOSED", Body: "## Outcome\nFinished.\n## Witness\nCommit abc.", Labels: []IssueLabel{{"priority/P1"}, {"gen/now"}, {"lane/taskmgr"}}},
	}
	attempts := []Attempt{{Holder: "issue-3", PID: 1234, Account: "seat-a", Token: "secret", HeartbeatAt: "soon", AcquiredAt: "earlier", Lane: "taskmgr"}}
	q := BuildQueue(issues, attempts)
	// Durable issue state remains ready; activity is represented separately.
	if q.Leaves[2].State != "active" || q.Leaves[2].DurableState != "ready" || len(q.Leaves[2].Attempts) != 1 {
		t.Fatalf("active leaf = %+v", q.Leaves[2])
	}
	var compact bytes.Buffer
	RenderQueue(&compact, q, false)
	got := compact.String()
	for _, want := range []string{"#1 state=ready", "#2 state=held", "#3 state=active", "#4 state=done", "priority=P1 generation=now lane=taskmgr", "outcome=\"Ship readable output.\"", "requires=#1", "witness=\"Captured render.\"", "parent=#9"} {
		if !strings.Contains(got, want) {
			t.Fatalf("captured default render missing %q:\n%s", want, got)
		}
	}
	for _, hidden := range []string{"pid=", "account=", "token=", "heartbeat=", "acquired="} {
		if strings.Contains(got, hidden) {
			t.Fatalf("default render leaked %q:\n%s", hidden, got)
		}
	}
	var detail bytes.Buffer
	RenderQueue(&detail, q, true)
	for _, want := range []string{"attempt holder=issue-3", "pid=1234", "account=seat-a", "token=secret", "heartbeat=soon", "acquired=earlier"} {
		if !strings.Contains(detail.String(), want) {
			t.Fatalf("drilldown missing %q:\n%s", want, detail.String())
		}
	}
}
