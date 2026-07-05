package dispatchstatus

import (
	"strings"
	"testing"
)

func baseCardInputs() CardInputs {
	return CardInputs{
		Workspace: "C:\\work\\fak",
		Preflight: map[string]any{
			"verdict":     "SPAWN_OK",
			"cap":         2,
			"live":        0,
			"max_workers": 2,
			"host":        map[string]any{"safe": true},
			"account":     map[string]any{"tag": "worker-a", "tier": 1, "model": "claude", "available": true},
		},
		Supervisor: map[string]any{
			"verdict":   "READY_TO_CANARY",
			"supervise": map[string]any{"target": 3, "alive": 1},
			"plans":     map[string]any{"total_plans": 2, "total_units": 17},
		},
		Watchdog: map[string]any{"installed": true, "status": "Ready"},
		Backlog: map[string]any{
			"lanes":  map[string]any{"docs": map[string]any{"issues": []any{1, 2, 3}}},
			"counts": map[string]any{"open": 3, "routed": 3, "unrouted": 0},
		},
		Closure: map[string]any{
			"closure_rate": 0.8,
			"counts":       map[string]any{"TRUE_RESOLVED": 8, "CLAIMED_CLOSED": 10, "OPEN_WITNESSED": 2},
		},
	}
}

func TestBuildCardBlocksOnSeatRefusal(t *testing.T) {
	in := baseCardInputs()
	in.Preflight["verdict"] = "REFUSE_NO_SEAT"

	payload := BuildCard(in)
	if got := payload["verdict"]; got != "BLOCKED_ON_SEAT" {
		t.Fatalf("verdict = %v, want BLOCKED_ON_SEAT", got)
	}
	for _, reason := range payload["reasons"].([]string) {
		if strings.Contains(reason, "safe to spawn") {
			t.Fatalf("seat refusal reason advertised growth: %q", reason)
		}
	}
}

func TestBuildCardGuardReasonIncludesChildCrashes(t *testing.T) {
	in := baseCardInputs()
	in.Guard = map[string]any{
		"sessions":    2,
		"rows":        5,
		"denied":      1,
		"quarantined": 1,
		"by_kind":     map[string]any{"CHILD_CRASH": 2},
	}

	payload := BuildCard(in)
	for _, reason := range payload["reasons"].([]string) {
		if strings.Contains(reason, "2 child crashes") {
			return
		}
	}
	t.Fatalf("guard child crash count missing from reasons: %#v", payload["reasons"])
}
