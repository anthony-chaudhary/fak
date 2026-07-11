package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// runSJ drives the verb against a temp journal, always disabling the guard_sessions.jsonl
// fold so the test is hermetic (no dependency on the host registry).
func runSJ(t *testing.T, journal string, argv ...string) (string, string, int) {
	t.Helper()
	var out, errb bytes.Buffer
	full := append([]string{argv[0], "--path", journal}, argv[1:]...)
	code := runSessionJournal(&out, &errb, full)
	return out.String(), errb.String(), code
}

func TestSessionJournalOpenThenReportLive(t *testing.T) {
	j := filepath.Join(t.TempDir(), "j.jsonl")
	if _, e, code := runSJ(t, j, "open", "--id", "S1", "--cwd", "C:/work/fak"); code != 0 {
		t.Fatalf("open exit %d, stderr=%s", code, e)
	}
	out, e, code := runSJ(t, j, "report", "--no-guard-sessions")
	if code != 0 {
		t.Fatalf("report exit %d, stderr=%s", code, e)
	}
	if !strings.Contains(out, "S1") || !strings.Contains(out, "LIVE") {
		t.Fatalf("report should show S1 as LIVE, got:\n%s", out)
	}
}

func TestSessionJournalReportCrashedAfterReboot(t *testing.T) {
	j := filepath.Join(t.TempDir(), "j.jsonl")
	if _, e, code := runSJ(t, j, "open", "--id", "S2", "--cwd", "C:/work/fak"); code != 0 {
		t.Fatalf("open exit %d, stderr=%s", code, e)
	}
	// A boot instant in the future means S2 started before "the current boot" — the
	// machine-wide-reboot signal that survives with no live process to probe.
	future := time.Now().UTC().Add(time.Hour).Format(time.RFC3339)
	out, e, code := runSJ(t, j, "report", "--no-guard-sessions", "--boot-time", future)
	if code != 0 {
		t.Fatalf("report exit %d, stderr=%s", code, e)
	}
	if !strings.Contains(out, "CRASHED") || !strings.Contains(out, "MACHINE_REBOOT") {
		t.Fatalf("S2 should classify CRASHED/MACHINE_REBOOT after a later boot, got:\n%s", out)
	}
	// The crashed row must carry the cwd to relaunch from.
	if !strings.Contains(out, "C:/work/fak") {
		t.Fatalf("crashed row must surface the recorded cwd, got:\n%s", out)
	}
}

func TestSessionJournalCloseThenClosed(t *testing.T) {
	j := filepath.Join(t.TempDir(), "j.jsonl")
	runSJ(t, j, "open", "--id", "S3")
	if _, e, code := runSJ(t, j, "close", "--id", "S3", "--reason", "graceful"); code != 0 {
		t.Fatalf("close exit %d, stderr=%s", code, e)
	}
	// Default report hides CLOSED; --all reveals it.
	out, _, _ := runSJ(t, j, "report", "--no-guard-sessions")
	if strings.Contains(out, "S3") {
		t.Fatalf("closed session should be hidden by default, got:\n%s", out)
	}
	outAll, _, _ := runSJ(t, j, "report", "--no-guard-sessions", "--all")
	if !strings.Contains(outAll, "S3") || !strings.Contains(outAll, "CLOSED") {
		t.Fatalf("--all should show S3 as CLOSED, got:\n%s", outAll)
	}
}

func TestSessionJournalRequiresID(t *testing.T) {
	j := filepath.Join(t.TempDir(), "j.jsonl")
	if _, _, code := runSJ(t, j, "open"); code != 2 {
		t.Fatalf("open without --id should exit 2, got %d", code)
	}
}

// TestSessionJournalReportCrashedDrive: a pre-boot open carrying a drive block classifies
// CRASHED and the report surfaces the carried budget — a populated DRIVE column in the human
// table and the `drive` block + resume_with_drive:true in the JSON envelope.
func TestSessionJournalReportCrashedDrive(t *testing.T) {
	j := filepath.Join(t.TempDir(), "j.jsonl")
	// A raw journal line with a drive block, started before a future boot -> MACHINE_REBOOT.
	line := `{"schema":"fak.sessionjournal.v1","kind":"open","id":"SD1","ts":"2026-07-08T10:00:00Z","cwd":"C:/work/fak","drive":{"turns_left":3,"tokens_left":12000,"spend_micro_cents_left":450000000,"generation":2}}` + "\n"
	if err := os.WriteFile(j, []byte(line), 0o644); err != nil {
		t.Fatal(err)
	}
	future := time.Now().UTC().Add(time.Hour).Format(time.RFC3339)

	// Human table: CRASHED with a populated DRIVE cell.
	out, e, code := runSJ(t, j, "report", "--no-guard-sessions", "--boot-time", future)
	if code != 0 {
		t.Fatalf("report exit %d, stderr=%s", code, e)
	}
	if !strings.Contains(out, "CRASHED") {
		t.Fatalf("SD1 should classify CRASHED, got:\n%s", out)
	}
	for _, want := range []string{"DRIVE", "t=3", "tok=12k", "$4.50", "gen=2"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human table should surface drive summary %q, got:\n%s", want, out)
		}
	}

	// JSON: the drive block round-trips and resume_with_drive is true for the crashed row.
	outJSON, e, code := runSJ(t, j, "report", "--no-guard-sessions", "--boot-time", future, "--json")
	if code != 0 {
		t.Fatalf("json report exit %d, stderr=%s", code, e)
	}
	row := firstReportSession(t, outJSON)
	if row.Drive == nil {
		t.Fatalf("json envelope should carry a drive block, got:\n%s", outJSON)
	}
	if row.Drive.TurnsLeft != 3 || row.Drive.Generation != 2 {
		t.Fatalf("drive block should carry turns_left=3 generation=2, got %+v in:\n%s", row.Drive, outJSON)
	}
	if !row.ResumeWithDrive {
		t.Fatalf("a crashed row carrying a drive must be resume_with_drive:true, got:\n%s", outJSON)
	}
}

// TestSessionJournalReportNoDriveDash: a crashed row with no carried drive renders DRIVE=-
// and resume_with_drive:false (a legacy row the pipeline must not treat as budget-carrying).
func TestSessionJournalReportNoDriveDash(t *testing.T) {
	j := filepath.Join(t.TempDir(), "j.jsonl")
	if _, e, code := runSJ(t, j, "open", "--id", "SD2", "--cwd", "C:/work/fak"); code != 0 {
		t.Fatalf("open exit %d, stderr=%s", code, e)
	}
	future := time.Now().UTC().Add(time.Hour).Format(time.RFC3339)
	outJSON, _, code := runSJ(t, j, "report", "--no-guard-sessions", "--boot-time", future, "--json")
	if code != 0 {
		t.Fatalf("json report exit %d", code)
	}
	row := firstReportSession(t, outJSON)
	if row.ResumeWithDrive {
		t.Fatalf("a crashed row with no drive must be resume_with_drive:false, got:\n%s", outJSON)
	}
	if row.Drive != nil {
		t.Fatalf("a crashed row with no drive must omit the drive block, got:\n%s", outJSON)
	}
}

// reportEnvelopeSession is the subset of a report row a drive-carry test asserts on: the carried
// drive block (nil when the row has none) and the resume_with_drive flag the C4 relaunch pipeline
// (#3788) filters on. Decoding the envelope keeps these assertions independent of the shared JSON
// writer's two-space indentation (writeIndentedJSON) rather than substring-matching whitespace.
type reportEnvelopeSession struct {
	ID    string `json:"id"`
	Drive *struct {
		TurnsLeft  int64 `json:"turns_left"`
		Generation int   `json:"generation"`
	} `json:"drive"`
	ResumeWithDrive bool `json:"resume_with_drive"`
}

// firstReportSession decodes the report envelope and returns its first session row, failing the
// test if the JSON is malformed or carries no sessions.
func firstReportSession(t *testing.T, envelope string) reportEnvelopeSession {
	t.Helper()
	var env struct {
		Sessions []reportEnvelopeSession `json:"sessions"`
	}
	if err := json.Unmarshal([]byte(envelope), &env); err != nil {
		t.Fatalf("report envelope is not valid JSON: %v\n%s", err, envelope)
	}
	if len(env.Sessions) == 0 {
		t.Fatalf("report envelope carried no sessions:\n%s", envelope)
	}
	return env.Sessions[0]
}

func TestSessionJournalReportJSON(t *testing.T) {
	j := filepath.Join(t.TempDir(), "j.jsonl")
	runSJ(t, j, "open", "--id", "S4")
	out, _, code := runSJ(t, j, "report", "--no-guard-sessions", "--json")
	if code != 0 {
		t.Fatalf("json report exit %d", code)
	}
	if !strings.Contains(out, "fak.sessionjournal.report.v1") || !strings.Contains(out, "\"counts\"") {
		t.Fatalf("json report missing schema/counts, got:\n%s", out)
	}
}
