package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

func TestRunGuardStopHookResidualEnqueuesSessionThreadReply(t *testing.T) {

	regDir := t.TempDir()
	outboxDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv(outboxDirEnv, outboxDir)
	t.Setenv(guardSessionsTokenEnv, "xoxb-test")
	t.Setenv(guardSessionsChannelEnv, "C-guard")

	traceID := "trace-stop-residual"
	sessionID := "session-stop-residual"
	rootNonce := "root-stop-residual"
	if err := guardsessions.Record(regDir, guardsessions.NewRow(traceID, "claude", 1, t.TempDir(), "audit", rootNonce, time.Now())); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(resume.IdentityLedgerPath(regDir), []byte(`{"uuid":"`+sessionID+`","trace":"`+traceID+`"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	transcriptPath := filepath.Join(t.TempDir(), "residual.jsonl")
	writeGuardEscalationTranscript(t, transcriptPath,
		`{"type":"user","message":{"role":"user","content":"prepare the production cutover"}}`,
		`{"type":"assistant","message":{"role":"assistant","content":"The irreversible production cutover is ready. Please approve it before I proceed."}}`,
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"session_id":"`+sessionID+`","transcript_path":"`+filepath.ToSlash(transcriptPath)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--operator-directed", guardPreCompactModeEnforce,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want clean HUMAN_RESIDUAL stop; stderr=%s", code, stderr.String())
	}
	ob, err := slackoutbox.Open(outboxDir)
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	rows := 0
	for _, row := range snap.Rows {
		if row.Source != guardOperatorEscalationSlackSource {
			continue
		}
		rows++
		if row.ParentNonce != rootNonce || !strings.Contains(row.Text, "HUMAN_RESIDUAL") {
			t.Fatalf("bad escalation row: %+v", row)
		}
	}
	if rows != 1 {
		t.Fatalf("escalation rows = %d, want 1; stderr=%s", rows, stderr.String())
	}
}

func writeGuardEscalationTranscript(t *testing.T, path string, lines ...string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}
