package loopfleet

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestBuildBackgroundStatusIncludesManagedAndUnregisteredSuperLoops(t *testing.T) {
	now := time.Date(2026, 7, 17, 12, 0, 0, 0, time.UTC)
	status := loopmgr.Status{Loops: []loopmgr.LoopSnapshot{{
		LoopID: "issue-resolve-dispatch/claude", State: string(loopmgr.StateRunning),
		LastKind: loopmgr.EventHeartbeat, LastEventUnixNano: now.Add(-30 * time.Second).UnixNano(),
	}}}
	got := BuildBackgroundStatus(status, []Process{{PID: 42, StartedAt: now.Add(-time.Hour), Command: `python tools/meta_superloop_night.py`}}, now)
	if got.Total != 2 || got.Live != 2 || got.Managed != 1 || got.ProcessOnly != 1 {
		t.Fatalf("rollup = %+v", got)
	}
	if got.Loops[1].Kind != "super-loop" || got.Loops[1].PID != 42 {
		t.Fatalf("process row = %+v", got.Loops[1])
	}
}

func TestClassifyBackgroundCommandDoesNotTreatOrdinarySlackWorkAsLoop(t *testing.T) {
	for _, command := range []string{`fak slack outbox status`, `python tools/fleet_slack_status.py --slack`} {
		if _, _, ok := ClassifyBackgroundCommand(command); ok {
			t.Fatalf("ordinary Slack command classified as loop: %q", command)
		}
	}
	for _, command := range []string{`codex /super-loop --watch`, `python dispatch_loop.py`, `fak loop run --interval 5m`, `powershell fleet_resume_watchdog.ps1`} {
		if _, _, ok := ClassifyBackgroundCommand(command); !ok {
			t.Fatalf("did not classify %q", command)
		}
	}
}

func TestBuildBackgroundStatusEncodesEmptyLoopsAsArray(t *testing.T) {
	got := BuildBackgroundStatus(loopmgr.Status{}, nil, time.Date(2026, 7, 17, 12, 0, 0, 0, time.UTC))
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) == "" || !containsJSONFragment(encoded, `"loops":[]`) {
		t.Fatalf("empty status JSON = %s", encoded)
	}
}

func containsJSONFragment(got []byte, want string) bool {
	return strings.Contains(string(got), want)
}
