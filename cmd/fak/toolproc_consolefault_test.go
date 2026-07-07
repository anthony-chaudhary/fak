package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// TestToolprocConsoleFaultsReportSearchable is the #2170 row-4 operator-surface
// witness (split to #3139): the pwsh HostException / Win32 0xE9 console-host
// FailFast class — previously visible only in Windows Event Viewer — is
// searchable from `fak toolproc console-faults`, the LeakEvent precedent
// applied to the console-fault journal.
func TestToolprocConsoleFaultsReportSearchable(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "console-faults.jsonl")
	rows := []toolprocgate.ConsoleFaultEvent{
		{
			Schema:  toolprocgate.ConsoleFaultEventSchema,
			Class:   toolprocgate.ConsoleHostFailFast,
			AtMS:    1_700_000_000_000,
			CallID:  "pwsh-child",
			Tool:    "PowerShell",
			Session: "s-crash",
			Surface: string(toolprocgate.ConsoleSurfaceStderr),
			Detail:  "System.Management.Automation.Host.HostException: No process is on the other end of the pipe.",
		},
		{
			Schema:  toolprocgate.ConsoleFaultEventSchema,
			Class:   toolprocgate.ConsoleRendererExit,
			AtMS:    1_700_000_000_100,
			CallID:  "render-pane",
			Tool:    "tui[pane]",
			Session: "s-operator",
			Surface: string(toolprocgate.ConsoleSurfaceRenderer),
			Detail:  "renderer panic: pane paint failed",
		},
	}
	var raw strings.Builder
	for _, row := range rows {
		b, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		raw.Write(b)
		raw.WriteByte('\n')
	}
	if err := os.WriteFile(journal, []byte(raw.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr strings.Builder
	if rc := runToolproc(&stdout, &stderr, []string{"console-faults", "--events", journal}); rc != 0 {
		t.Fatalf("toolproc console-faults rc=%d stderr=%s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"console faults: 2 row(s)",
		string(toolprocgate.ConsoleHostFailFast),
		string(toolprocgate.ConsoleRendererExit),
		"pwsh-child",
		"render-pane",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("human report missing %q:\n%s", want, out)
		}
	}

	stdout.Reset()
	stderr.Reset()
	if rc := runToolproc(&stdout, &stderr, []string{"console-faults", "--events", journal, "--json"}); rc != 0 {
		t.Fatalf("toolproc console-faults --json rc=%d stderr=%s", rc, stderr.String())
	}
	encoded := stdout.String()
	for _, want := range []string{
		`"CONSOLE_HOST_FAILFAST": 1`,
		`"CONSOLE_RENDERER_EXIT": 1`,
		`"call_id": "pwsh-child"`,
		"HostException",
	} {
		if !strings.Contains(encoded, want) {
			t.Fatalf("JSON report missing %q:\n%s", want, encoded)
		}
	}
}

// TestToolprocConsoleFaultsFailClosed: a drifted or fabricated row (unknown
// class) refuses the WHOLE report — it can never enter an operator report as
// a legitimate crash record.
func TestToolprocConsoleFaultsFailClosed(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "console-faults.jsonl")
	bad := `{"schema":"` + toolprocgate.ConsoleFaultEventSchema + `","class":"NOT_A_REAL_CLASS","at_unix_ms":1700000000000,"call_id":"x"}` + "\n"
	if err := os.WriteFile(journal, []byte(bad), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr strings.Builder
	if rc := runToolproc(&stdout, &stderr, []string{"console-faults", "--events", journal}); rc != 1 {
		t.Fatalf("toolproc console-faults rc=%d, want 1 (fail-closed refusal); stdout=%s", rc, stdout.String())
	}
	if !strings.Contains(stderr.String(), "unknown class") {
		t.Fatalf("refusal did not name the unknown class: %s", stderr.String())
	}
}
