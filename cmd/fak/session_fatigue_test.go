package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// fatigueLedger writes a fak.guard-stop.v1 JSONL fixture and returns its path. The
// schema string comes from the same const the reader filters on, so a rename cannot
// leave this test silently matching nothing.
func fatigueLedger(t *testing.T, rows ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "guard-stops.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(rows, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("write ledger: %v", err)
	}
	return path
}

// fatigueRow builds one stop event. lastTool == "" means no transcript was read at
// all, which the fold must treat as UNKNOWN rather than as absence of inspection.
func fatigueRow(session, stage, kind, disposition, lastTool string) string {
	if lastTool == "" {
		return fmt.Sprintf(`{"schema":%q,"session":%q,"stage":%q,"kind":%q,"disposition":%q}`,
			sessionctl.FatigueEventSchema, session, stage, kind, disposition)
	}
	return fmt.Sprintf(`{"schema":%q,"session":%q,"stage":%q,"kind":%q,"disposition":%q,"transcript":{"read":true,"last_tool_use":%q}}`,
		sessionctl.FatigueEventSchema, session, stage, kind, disposition, lastTool)
}

func TestRunFatigueFlagsARubberStampedGate(t *testing.T) {
	// DefaultFatigueMinFires firings of one gate, every one approved with a
	// non-inspect tool last — rate 1.0, which clears the default threshold.
	var rows []string
	for i := 0; i < sessionctl.DefaultFatigueMinFires+2; i++ {
		rows = append(rows, fatigueRow("s1", "pretool", "confirm", "approved", "bash"))
	}
	var stdout, stderr bytes.Buffer
	if rc := runFatigue(&stdout, &stderr, []string{"--ledger", fatigueLedger(t, rows...)}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, sessionctl.RubberStampedFlag) {
		t.Errorf("want the %s flag in the render, got:\n%s", sessionctl.RubberStampedFlag, out)
	}
	if !strings.Contains(out, "pretool/confirm/approved") {
		t.Errorf("want the gate identity in the render, got:\n%s", out)
	}
}

func TestRunFatigueDoesNotFlagAnInspectedGate(t *testing.T) {
	// Same volume, but each approval followed an inspect-class tool. Inspection is
	// the whole point of the signal: volume alone must not flag.
	var rows []string
	for i := 0; i < sessionctl.DefaultFatigueMinFires+2; i++ {
		rows = append(rows, fatigueRow("s1", "pretool", "confirm", "approved", "read"))
	}
	var stdout, stderr bytes.Buffer
	if rc := runFatigue(&stdout, &stderr, []string{"--ledger", fatigueLedger(t, rows...)}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, stderr.String())
	}
	if out := stdout.String(); strings.Contains(out, sessionctl.RubberStampedFlag) {
		t.Errorf("inspected gate must not be flagged, got:\n%s", out)
	}
}

func TestRunFatigueUnknownTranscriptIsNotHabituation(t *testing.T) {
	// No transcript => we cannot show there was no inspection. The fold must keep
	// these out of the numerator, so a ledger of only-unknown rows flags nothing.
	var rows []string
	for i := 0; i < sessionctl.DefaultFatigueMinFires+2; i++ {
		rows = append(rows, fatigueRow("s1", "pretool", "confirm", "approved", ""))
	}
	var stdout, stderr bytes.Buffer
	if rc := runFatigue(&stdout, &stderr, []string{"--ledger", fatigueLedger(t, rows...)}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, stderr.String())
	}
	if out := stdout.String(); strings.Contains(out, sessionctl.RubberStampedFlag) {
		t.Errorf("unknown-inspection rows must not be counted as habituation, got:\n%s", out)
	}
}

func TestRunFatigueTraceSelectsOneSession(t *testing.T) {
	rows := []string{
		fatigueRow("keep", "pretool", "confirm", "approved", "bash"),
		fatigueRow("drop", "pretool", "confirm", "approved", "bash"),
		fatigueRow("drop", "pretool", "confirm", "approved", "bash"),
	}
	var stdout, stderr bytes.Buffer
	rc := runFatigue(&stdout, &stderr, []string{"--ledger", fatigueLedger(t, rows...), "--trace", "keep", "--json"})
	if rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, stderr.String())
	}
	var rep sessionctl.FatigueReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode --json: %v; out=%s", err, stdout.String())
	}
	if rep.Events != 1 {
		t.Errorf("events = %d, want 1 (only the selected session)", rep.Events)
	}
	if rep.Session != "keep" {
		t.Errorf("session = %q, want %q", rep.Session, "keep")
	}
}

func TestRunFatigueJSONCarriesTheReportSchema(t *testing.T) {
	rows := []string{fatigueRow("s1", "stop", "guard", "approved", "bash")}
	var stdout, stderr bytes.Buffer
	if rc := runFatigue(&stdout, &stderr, []string{"--ledger", fatigueLedger(t, rows...), "--json"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, stderr.String())
	}
	var rep sessionctl.FatigueReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode --json: %v; out=%s", err, stdout.String())
	}
	if rep.Schema != sessionctl.FatigueReportSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, sessionctl.FatigueReportSchema)
	}
	if rep.Threshold != sessionctl.DefaultFatigueThreshold || rep.MinFires != sessionctl.DefaultFatigueMinFires {
		t.Errorf("defaults not echoed: threshold=%v min_fires=%d", rep.Threshold, rep.MinFires)
	}
}

func TestRunFatigueEmptyLedgerSaysSoInsteadOfPrintingAnEmptyTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.jsonl")
	var stdout, stderr bytes.Buffer
	if rc := runFatigue(&stdout, &stderr, []string{"--ledger", path}); rc != 0 {
		t.Fatalf("rc = %d, want 0 for a missing ledger; stderr=%s", rc, stderr.String())
	}
	if out := stdout.String(); !strings.Contains(out, "no guard-stop events recorded yet") {
		t.Errorf("want the empty-ledger explanation, got:\n%s", out)
	}
}

func TestRunFatigueRejectsAPositionalArgument(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runFatigue(&stdout, &stderr, []string{"--ledger", fatigueLedger(t, fatigueRow("s1", "stop", "guard", "approved", "bash")), "oops"})
	if rc != 2 {
		t.Fatalf("rc = %d, want 2 for a positional arg; stdout=%s stderr=%s", rc, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "unexpected argument") {
		t.Errorf("want the usage error on stderr, got: %s", stderr.String())
	}
}
