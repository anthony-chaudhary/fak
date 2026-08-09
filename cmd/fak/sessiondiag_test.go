package main

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
	code := runSessionDiag(&out, &er, []string{"--json"}, q)
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
	code := runSessionDiag(&o, &e, nil, func(string, time.Duration) (sessiondiag.Evidence, error) {
		return sessiondiag.Evidence{}, errors.New(`C:\private\token-secret.db: authorization bearer-secret`)
	})
	if code != 2 || strings.Contains(e.String(), `C:\private`) || strings.Contains(e.String(), "bearer-secret") {
		t.Fatalf("code=%d err=%q", code, e.String())
	}
}
func TestRunSessionDiagMissing(t *testing.T) {
	var o, e bytes.Buffer
	code := runSessionDiag(&o, &e, nil, func(string, time.Duration) (sessiondiag.Evidence, error) {
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
	code := runSessionDiag(&o, &e, []string{"--db", p}, queryCodexLogReadOnly)
	if code != 2 || !strings.Contains(e.String(), "reader error") || strings.Contains(e.String(), dir) {
		t.Fatalf("code=%d err=%q", code, e.String())
	}
}

func TestCountCodexProcessesIsNonNegative(t *testing.T) {
	if got := countCodexProcesses(); got < 0 {
		t.Fatalf("count=%d", got)
	}
}
