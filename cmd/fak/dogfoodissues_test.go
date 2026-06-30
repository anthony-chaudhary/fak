package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDogfoodIssuesDryRunSkipsUnscopedAggregateRows(t *testing.T) {
	report := filepath.Join(t.TempDir(), "report.json")
	const raw = `{
		"schema": "fak.recent-feature-dogfood.v1",
		"ok": true,
		"probes": [{
			"key": "code-slop-scorecard",
			"ok": true,
			"payload": {
				"schema": "fleet-code-slop-scorecard/1",
				"verdict": "ACTION",
				"finding": "code_slop",
				"corpus": {"score": 71.5, "grade": "C", "slop_debt": 12},
				"next_action": "retire slop-debt worst-first; re-run to prove the drop"
			}
		}]
	}`
	if err := os.WriteFile(report, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--json", report})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		Planned []struct {
			Key string `json:"key"`
		} `json:"planned"`
		Skipped []struct {
			Key    string `json:"key"`
			Reason string `json:"reason"`
		} `json:"skipped"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if len(got.Planned) != 0 {
		t.Fatalf("planned = %+v, want no dispatchable aggregate rows", got.Planned)
	}
	if len(got.Skipped) != 1 || got.Skipped[0].Key != "recent-feature-dogfood/code-slop-scorecard/code_slop" {
		t.Fatalf("skipped = %+v, want code-slop aggregate row", got.Skipped)
	}
	if got.Skipped[0].Reason != "ISSUE_SCOPE_INCOMPLETE,ISSUE_UNROUTED" {
		t.Fatalf("skip reason = %q", got.Skipped[0].Reason)
	}
}
