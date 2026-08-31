package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/looptrigger"
)

func TestSuperloopTriggerJSON(t *testing.T) {
	var out, errout bytes.Buffer
	code := runSuperloop(&out, &errout, []string{"trigger", "--loop", "quality", "--observed-at", "2026-08-31T12:00:00Z", "--eligible", "3", "--oldest-age", "5h", "--source-age", "10s", "--max-source-age", "5m", "--capacity", "2", "--required-capacity", "1", "--since-last-run", "2h", "--cooldown", "1h", "--service-window", "4h", "--expected-value", "8", "--value-floor", "5", "--evidence", "issue:10352,lease:loops", "--json"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	var r looptrigger.Receipt
	if e := json.Unmarshal(out.Bytes(), &r); e != nil {
		t.Fatal(e)
	}
	if r.Schema != looptrigger.Schema || r.Decision != looptrigger.Run || r.Reason != looptrigger.Deadline || r.Timing.State != "OVERDUE" {
		t.Fatalf("receipt=%+v", r)
	}
}
func TestSuperloopTriggerHumanIsReadOnlyDecision(t *testing.T) {
	var out, errout bytes.Buffer
	code := runSuperloop(&out, &errout, []string{"trigger", "--observed-at", "2026-08-31T12:00:00Z", "--eligible", "2", "--source-age", "1m", "--capacity", "1", "--since-last-run", "10m", "--cooldown", "1h", "--expected-value", "9", "--value-floor", "5"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	for _, want := range []string{"super-loop DEFER", "COOLDOWN", "timing=EARLY"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in %s", want, out.String())
		}
	}
}
