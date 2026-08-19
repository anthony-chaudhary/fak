package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"

	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
)

func TestRunSessionDiagCapturedPressure(t *testing.T) {
	var out, er bytes.Buffer
	q := func(string, time.Duration) (sessiondiag.Evidence, error) {
		return sessiondiag.Evidence{DBBasename: "logs_2.sqlite", DBBytes: 1755549696, WALBytes: 400748312, PageSize: 4096, PageCount: 428601, FreelistPages: 359146, QueueDrops: 996, SlowWrites: 4, Integrity: "ok"}, nil
	}
	code := RunSessionDiag(&out, &er, []string{"--json"}, q)
	if code != 1 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	var r sessiondiag.Report
	if json.Unmarshal(out.Bytes(), &r) != nil || r.Verdict != "CORRELATED_RUNTIME_PRESSURE" || r.Causality != "not_established" {
		t.Fatalf("%s", out.String())
	}
	if strings.Contains(out.String(), "C:\\") || strings.Contains(out.String(), "secret") {
		t.Fatal("leak")
	}
}
func TestRunSessionDiagRedactsReaderFailure(t *testing.T) {
	var o, e bytes.Buffer
	code := RunSessionDiag(&o, &e, nil, func(string, time.Duration) (sessiondiag.Evidence, error) {
		return sessiondiag.Evidence{}, errors.New(`C:\private\token-secret.db: authorization bearer-secret`)
	})
	if code != 2 || strings.Contains(e.String(), `C:\private`) || strings.Contains(e.String(), "bearer-secret") {
		t.Fatalf("code=%d err=%q", code, e.String())
	}
}
func TestRunSessionDiagMissing(t *testing.T) {
	var o, e bytes.Buffer
	code := RunSessionDiag(&o, &e, nil, func(string, time.Duration) (sessiondiag.Evidence, error) {
		return sessiondiag.Evidence{}, errors.New("missing store")
	})
	if code != 2 || !strings.Contains(e.String(), "reader error") {
		t.Fatalf("%d %s", code, e.String())
	}
}

func TestRunSessionDiagMalformedStore(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "broken.sqlite")
	if err := os.WriteFile(p, []byte("not sqlite"), 0o600); err != nil {
		t.Fatal(err)
	}
	var o, e bytes.Buffer
	code := RunSessionDiag(&o, &e, []string{"--db", p}, queryCodexLogReadOnly)
	if code != 2 || !strings.Contains(e.String(), "reader error") || strings.Contains(e.String(), dir) {
		t.Fatalf("code=%d err=%q", code, e.String())
	}
}

func TestCountCodexProcessesIsNonNegative(t *testing.T) {
	if got := countCodexProcesses(); got < 0 {
		t.Fatalf("count=%d", got)
	}
}

func TestRunSessionDiagWritesRedactedIncident(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "incident.json")
	var out, er bytes.Buffer
	q := func(string, time.Duration) (sessiondiag.Evidence, error) {
		return sessiondiag.Evidence{QueueDrops: 3, SlowWrites: 2, WALBytes: 4096, ProcessCount: 4}, nil
	}
	code := RunSessionDiag(&out, &er, []string{"--incident-out", dst, "--process-id", "42", "--process-uuid", "proc-42", "--thread-id", "thread-7", "--exit-kind", "failure", "--exit-code", "23", "--os-failure-event"}, q)
	if code != 1 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	b, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	var got sessiondiag.Incident
	if err = json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "DIRECT_PROCESS_FAILURE" || got.ProcessUUID != "proc-42" || got.QueueDropDelta != 3 {
		t.Fatalf("%s", b)
	}
	if strings.Contains(string(b), dir) {
		t.Fatalf("path leaked: %s", b)
	}
	if st, err := os.Stat(dst); err != nil || runtime.GOOS != "windows" && st.Mode().Perm()&0077 != 0 {
		t.Fatalf("incident permissions=%v err=%v", st.Mode(), err)
	}
}

func TestRunSessionDiagInventoryJSONAndHumanRender(t *testing.T) {
	now := time.Date(2026, 8, 11, 18, 30, 0, 0, time.UTC)
	threadID := "91000001-0000-4000-8000-000000000001"
	inventoryQuery := func(string, time.Duration, time.Time) (sessiondiag.InventoryInput, error) {
		return sessiondiag.InventoryInput{
			Threads: []sessiondiag.ThreadEvidence{{
				ThreadID: threadID, Source: "cli", ThreadSource: "user",
				CreatedAt: now.Add(-time.Minute).Add(100 * time.Millisecond), UpdatedAt: now.Add(-time.Second),
			}},
			Turns:         []sessiondiag.TurnEvidence{{ThreadID: threadID, TurnID: "92000001-0000-4000-8000-000000000001", Status: "inProgress", StartedAt: now.Add(-time.Minute), RolloutOrdinal: 1}},
			WriterLocks:   []sessiondiag.WriterLockEvidence{{ThreadID: threadID, ModifiedAt: now.Add(-time.Second)}},
			GuardReceipts: []sessiondiag.GuardReceiptEvidence{{ThreadID: threadID, RecordedAt: now.Add(-50 * time.Second)}},
			Processes: []sessiondiag.ProcessEvidence{
				{PID: 1, Name: "fak.exe", CommandLine: "fak guard codex", StartedAt: now.Add(-62 * time.Second)},
				{PID: 2, ParentPID: 1, Name: "node.exe", CommandLine: "node codex.js", StartedAt: now.Add(-61 * time.Second)},
				{PID: 3, ParentPID: 2, Name: "codex.exe", CommandLine: "codex -c model_provider=fak", StartedAt: now.Add(-time.Minute)},
			},
		}, nil
	}
	pressureQuery := func(string, time.Duration) (sessiondiag.Evidence, error) {
		return sessiondiag.Evidence{}, errors.New("pressure query must not run in inventory mode")
	}

	var jsonOut, jsonErr bytes.Buffer
	code := runSessionDiagWith(&jsonOut, &jsonErr, []string{"--inventory", "--json"}, pressureQuery, inventoryQuery, func() time.Time { return now })
	if code != 0 || jsonErr.Len() != 0 {
		t.Fatalf("json code=%d err=%s", code, jsonErr.String())
	}
	var report sessiondiag.InventoryReport
	if err := json.Unmarshal(jsonOut.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != sessiondiag.InventorySchema || report.Counts.Active != 1 ||
		report.Sessions[0].Kind != sessiondiag.KindGuardedTUI {
		t.Fatalf("inventory=%s", jsonOut.String())
	}

	var humanOut, humanErr bytes.Buffer
	code = runSessionDiagWith(&humanOut, &humanErr, []string{"--inventory"}, pressureQuery, inventoryQuery, func() time.Time { return now })
	if code != 0 || humanErr.Len() != 0 {
		t.Fatalf("human code=%d err=%s", code, humanErr.String())
	}
	for _, want := range []string{"CODEX SESSION INVENTORY", "active=1", "fak_guarded_tui", "launch receipts", "none is treated as liveness"} {
		if !strings.Contains(humanOut.String(), want) {
			t.Fatalf("human render missing %q:\n%s", want, humanOut.String())
		}
	}
}

func TestRunSessionDiagInventoryLoadsRegistrationStore(t *testing.T) {
	now := time.Date(2026, 8, 11, 20, 0, 0, 0, time.UTC)
	store := sessionregistry.Store{Path: filepath.Join(t.TempDir(), "registry.jsonl")}
	r, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "root", RootIssue: "6468", TaskID: "issue-6468", AttemptID: "a", LaunchKind: "guarded_tui", Runtime: "codex", Now: now})
	if err := store.Register(r); err != nil {
		t.Fatal(err)
	}
	q := func(string, time.Duration, time.Time) (sessiondiag.InventoryInput, error) {
		return sessiondiag.InventoryInput{}, nil
	}
	var out, errb bytes.Buffer
	code := runSessionDiagWith(&out, &errb, []string{"--inventory", "--registry", store.Path, "--json"}, nil, q, func() time.Time { return now })
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var got sessiondiag.InventoryReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Sources.Registrations != 1 || len(got.Registrations) != 1 || got.Registrations[0].RootIssue != "6468" {
		t.Fatalf("report=%+v", got)
	}
}

func TestSessionDiagWatchdogCandidatesUsesInventoryClassifier(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	id := "10000000-0000-4000-8000-000000000001"
	inventory := func(string, time.Duration, time.Time) (sessiondiag.InventoryInput, error) {
		return sessiondiag.InventoryInput{
			Threads: []sessiondiag.ThreadEvidence{{ThreadID: id, Source: "cli", CWD: "/repo", UpdatedAt: now.Add(-time.Hour)}},
			Turns:   []sessiondiag.TurnEvidence{{ThreadID: id, Status: "inProgress", StartedAt: now.Add(-time.Hour)}},
		}, nil
	}
	var out, errOut bytes.Buffer
	code := runSessionDiagWith(&out, &errOut, []string{"--inventory", "--watchdog-candidates", "--json"}, nil, inventory, func() time.Time { return now })
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report sessiondiag.WatchdogCandidateReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Candidates) != 1 || report.Candidates[0].Session != id {
		t.Fatalf("report=%+v", report)
	}
}
