package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardcomplaint"
)

// runComplainCapture runs the CLI core and returns (exitCode, stdout, stderr).
func runComplainCapture(argv []string) (int, string, string) {
	var out, errb bytes.Buffer
	code := runComplain(&out, &errb, argv)
	return code, out.String(), errb.String()
}

func TestComplainRequiresSummary(t *testing.T) {
	code, _, errs := runComplainCapture([]string{"--reason", "FILE_ADMISSION"})
	if code != 2 {
		t.Fatalf("missing --summary exit = %d, want 2", code)
	}
	if !strings.Contains(errs, "--summary is required") {
		t.Fatalf("stderr should explain the missing flag: %q", errs)
	}
}

func TestComplainRejectsUnknownKind(t *testing.T) {
	code, _, errs := runComplainCapture([]string{"--summary", "x", "--kind", "nonsense"})
	if code != 2 {
		t.Fatalf("unknown kind exit = %d, want 2", code)
	}
	if !strings.Contains(errs, "unknown complaint kind") {
		t.Fatalf("stderr should name the closed kind set: %q", errs)
	}
}

func TestComplainDryRunPlansCreate(t *testing.T) {
	code, out, _ := runComplainCapture([]string{
		"--summary", "floor blocked a legit docs/notes commit",
		"--reason", "FILE_ADMISSION", "--tool", "Bash", "--json",
	})
	if code != 0 {
		t.Fatalf("dry-run exit = %d, want 0", code)
	}
	var res guardcomplaint.Result
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("json output did not parse: %v\n%s", err, out)
	}
	if res.Schema != guardcomplaint.Schema || res.Mode != "dry-run" {
		t.Fatalf("schema/mode = %q/%q", res.Schema, res.Mode)
	}
	if len(res.Planned) != 1 || res.Planned[0].Action != "create" || res.Planned[0].Occurrences != 1 {
		t.Fatalf("planned = %+v, want one create at occurrences 1", res.Planned)
	}
	if len(res.Synced) != 0 {
		t.Fatalf("dry-run must not sync: %+v", res.Synced)
	}
}

func TestComplainExistingJSONEscalatesToUpdate(t *testing.T) {
	// Build a fixture existing issue whose body carries the SAME dedup marker key the
	// CLI will compute for this complaint, so the plan must fold onto it as an update.
	c := guardcomplaint.Complaint{
		Kind: "false-positive", Reason: "FILE_ADMISSION", Tool: "Bash",
		Summary: "blocked a legit note",
	}
	fixture := filepath.Join(t.TempDir(), "existing.json")
	issues := []dogfoodissues.Issue{{Number: 314, State: "OPEN", Title: c.Title(), Body: c.Body(1)}}
	b, _ := json.Marshal(issues)
	if err := os.WriteFile(fixture, b, 0o600); err != nil {
		t.Fatal(err)
	}

	code, out, _ := runComplainCapture([]string{
		"--summary", "blocked a legit note", "--reason", "FILE_ADMISSION", "--tool", "Bash",
		"--existing-json", fixture, "--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	var res guardcomplaint.Result
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("json parse: %v", err)
	}
	row := res.Planned[0]
	if row.Action != "update" || row.Number == nil || *row.Number != 314 {
		t.Fatalf("re-file should update #314: %+v", row)
	}
	if row.Occurrences != 2 {
		t.Fatalf("occurrences = %d, want 2 (escalated)", row.Occurrences)
	}
}

func TestComplainFromJournalAttachesWitnessOrDisclosesMiss(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "guard-audit.jsonl")
	// A real DENY row matching the complaint's reason+tool.
	row := `{"seq":7,"ts_unix_nano":700,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"FILE_ADMISSION","by":"ifc-sink","trace_id":"t-live","args_digest":"sha256:abc"}`
	if err := os.WriteFile(jpath, []byte(row+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	// Matching witness present -> no "no matching" note, and the body carries the verdict.
	code, _, errs := runComplainCapture([]string{
		"--summary", "blocked a legit note", "--reason", "FILE_ADMISSION", "--tool", "Bash",
		"--from-journal", "--journal", jpath, "--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if strings.Contains(errs, "no matching DENY") {
		t.Fatalf("a matching witness should NOT print the miss note: %q", errs)
	}

	// No matching row -> honest miss note on stderr, still exit 0 (files on rationale alone).
	code, _, errs = runComplainCapture([]string{
		"--summary", "blocked a legit note", "--reason", "NO_SUCH_REASON",
		"--from-journal", "--journal", jpath, "--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(errs, "no matching DENY") {
		t.Fatalf("a missing witness must be disclosed on stderr: %q", errs)
	}
}
