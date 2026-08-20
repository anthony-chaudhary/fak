package main

import (
	"bytes"
	"encoding/json"
	"runtime"
	"testing"
)

func TestWindowgateSelfcheckReportJSON(t *testing.T) {
	want := desktopConsoleSelfcheckReport{
		Schema: desktopConsoleSelfcheckSchema, OK: true, Applicable: true, Platform: "windows",
		Backend: "codex", SharedHiddenConsoles: 1, VisibleWindows: 0,
		Processes: []desktopConsoleSelfcheckProcess{{Label: "pwsh", PID: 42, ConsolePIDs: []uint32{7, 42}}},
		Reason:    "hidden",
	}
	var out bytes.Buffer
	if err := writeDesktopConsoleSelfcheck(&out, want, true); err != nil {
		t.Fatal(err)
	}
	var got desktopConsoleSelfcheckReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode selfcheck JSON: %v\n%s", err, out.String())
	}
	if !got.OK || got.SharedHiddenConsoles != 1 || len(got.Processes) != 1 || got.Processes[0].Label != "pwsh" {
		t.Fatalf("selfcheck JSON lost evidence: %+v", got)
	}
}

func TestRunWindowgateSelfcheckIsExplicitlyNotApplicableOffWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the live Windows command is captured by the normal-executable selfcheck witness")
	}
	var stdout, stderr bytes.Buffer
	if code := runWindowgate(&stdout, &stderr, []string{"--selfcheck", "--json"}); code != 0 {
		t.Fatalf("selfcheck code=%d stderr=%s", code, stderr.String())
	}
	var got desktopConsoleSelfcheckReport
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode selfcheck JSON: %v\n%s", err, stdout.String())
	}
	if !got.OK || got.Applicable || got.Platform != runtime.GOOS {
		t.Fatalf("off-Windows selfcheck = %+v, want explicit non-applicable success", got)
	}
}
