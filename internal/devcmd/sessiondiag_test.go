package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
	if st, err := os.Stat(dst); err != nil || st.Mode().Perm()&0077 != 0 {
		t.Fatalf("incident permissions=%v err=%v", st.Mode(), err)
	}
}
