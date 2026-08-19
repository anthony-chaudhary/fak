package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

func TestMergeSessiondiagCandidatesPreservesLegacyRowsAndAddsCodex(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "fak-dev")
	if os.PathSeparator == '\\' {
		fake += ".bat"
	}
	body := "#!/bin/sh\nprintf '%s' '{\"schema\":\"fak.sessiondiag.watchdog-candidates.v1\",\"candidates\":[{\"session\":\"legacy\",\"harness\":\"codex\",\"reason\":\"ABRUPT_INTERACTIVE_NO_CURRENT_EVIDENCE\"},{\"session\":\"codex-thread\",\"harness\":\"codex\",\"cwd\":\"/repo\",\"reason\":\"ABRUPT_INTERACTIVE_NO_CURRENT_EVIDENCE\"}],\"exclusions\":[{\"session\":\"active\",\"reason\":\"CURRENT_OR_AMBIGUOUS_HEALTH\"}],\"counts\":{}}'\n"
	if os.PathSeparator == '\\' {
		body = "@echo off\r\necho {\"schema\":\"fak.sessiondiag.watchdog-candidates.v1\",\"candidates\":[{\"session\":\"legacy\",\"harness\":\"codex\",\"reason\":\"ABRUPT_INTERACTIVE_NO_CURRENT_EVIDENCE\"},{\"session\":\"codex-thread\",\"harness\":\"codex\",\"cwd\":\"C:/repo\",\"reason\":\"ABRUPT_INTERACTIVE_NO_CURRENT_EVIDENCE\"}],\"exclusions\":[{\"session\":\"active\",\"reason\":\"CURRENT_OR_AMBIGUOUS_HEALTH\"}],\"counts\":{}}\r\n"
	}
	if err := os.WriteFile(fake, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_DEV_EXE", fake)
	legacy := []resume.WatchdogPlanRow{{Session: "legacy", Account: ".claude-a"}}
	got, report := rwMergeSessiondiagCandidates(legacy, 8)
	if len(got) != 2 || len(report.Candidates) != 2 || len(report.Exclusions) != 1 {
		t.Fatalf("plan=%+v report=%+v", got, report)
	}
	if !reflect.DeepEqual(got[0], legacy[0]) {
		t.Fatalf("legacy row changed: %+v", got[0])
	}
	if got[1].Session != "codex-thread" || got[1].Harness != "codex" || got[1].Disp == "" {
		t.Fatalf("candidate row=%+v", got[1])
	}
}

func TestMergeSessiondiagTwentyCandidateParity(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "fak-dev")
	if os.PathSeparator == '\\' {
		fake += ".bat"
	}
	candidates := make([]map[string]string, 20)
	for i := range candidates {
		candidates[i] = map[string]string{"session": fmt.Sprintf("session-%02d", i), "harness": "codex", "reason": "ABRUPT_INTERACTIVE_NO_CURRENT_EVIDENCE"}
	}
	raw, err := json.Marshal(map[string]any{"schema": "fak.sessiondiag.watchdog-candidates.v1", "candidates": candidates, "exclusions": []any{}, "counts": map[string]int{"ABRUPT_INTERACTIVE_NO_CURRENT_EVIDENCE": 20}})
	if err != nil {
		t.Fatal(err)
	}
	body := "#!/bin/sh\nprintf '%s' '" + string(raw) + "'\n"
	if os.PathSeparator == '\\' {
		body = "@echo off\r\necho " + string(raw) + "\r\n"
	}
	if err := os.WriteFile(fake, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_DEV_EXE", fake)
	got, report := rwMergeSessiondiagCandidates(nil, 8)
	if len(got) != 20 || len(report.Candidates) != 20 {
		t.Fatalf("plan=%d candidates=%d", len(got), len(report.Candidates))
	}
	for i := range got {
		if got[i].Session != report.Candidates[i].Session {
			t.Fatalf("parity[%d]: plan=%s report=%s", i, got[i].Session, report.Candidates[i].Session)
		}
	}
}
