package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

func TestGuardOperatorEscalationEnqueuesExactlyOneSessionThreadReply(t *testing.T) {
	t.Setenv(guardSessionsTokenEnv, "")

	regDir := t.TempDir()
	outboxDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv(outboxDirEnv, outboxDir)
	t.Setenv(guardSessionsTokenEnv, "xoxb-test")
	t.Setenv(guardSessionsChannelEnv, "C-guard")

	traceID := "trace-escalation"
	sessionID := "transcript-escalation"
	rootNonce := "guard-root-nonce"
	if err := guardsessions.Record(regDir, guardsessions.NewRow(traceID, "claude", 123, t.TempDir(), "audit.jsonl", rootNonce, time.Now())); err != nil {
		t.Fatal(err)
	}
	identity := `{"uuid":"` + sessionID + `","trace":"` + traceID + `"}` + "\n"
	if err := os.WriteFile(resume.IdentityLedgerPath(regDir), []byte(identity), 0o600); err != nil {
		t.Fatal(err)
	}

	tr := &guardStopTranscript{
		OperatorDirected:            true,
		OperatorDirectedClass:       "AUTHORITY_BOUNDARY",
		OperatorDirectedDisposition: "HUMAN_RESIDUAL",
		OperatorDirectedResolve:     "approve the irreversible production cutover",
	}
	routeGuardOperatorEscalationFailOpen(sessionID, stopDispOperatorDirectedContinue, tr)
	routeGuardOperatorEscalationFailOpen(sessionID, stopDispOperatorDirectedWarn, tr)
	routeGuardOperatorEscalationFailOpen(sessionID, stopDispOperatorDirectedEscalate, tr)
	routeGuardOperatorEscalationFailOpen(sessionID, stopDispOperatorDirectedEscalate, tr) // repeated Stop must not duplicate

	ob, err := slackoutbox.Open(outboxDir)
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	var rows []slackoutbox.Row
	for _, row := range snap.Rows {
		if row.Source == guardOperatorEscalationSlackSource {
			rows = append(rows, row)
		}
	}
	if len(rows) != 1 {
		t.Fatalf("escalation rows = %d, want exactly 1: %+v", len(rows), rows)
	}
	row := rows[0]
	if row.Channel != "C-guard" || row.ParentNonce != rootNonce || row.ThreadTS != "" {
		t.Fatalf("row not deferred under session root: %+v", row)
	}
	if !strings.Contains(row.Text, "HUMAN_RESIDUAL") || !strings.Contains(row.Text, tr.OperatorDirectedResolve) {
		t.Fatalf("row lacks typed reason/remediation: %q", row.Text)
	}
}

func TestGuardOperatorEscalationNoopsWithoutTokenOrSessionThread(t *testing.T) {
	for _, tc := range []struct {
		name  string
		token string
		seed  bool
	}{
		{name: "no token", token: "", seed: true},
		{name: "no session thread", token: "xoxb-test", seed: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			regDir := t.TempDir()
			outboxDir := t.TempDir()
			t.Setenv("FLEET_REG_DIR", regDir)
			t.Setenv(outboxDirEnv, outboxDir)
			t.Setenv(guardSessionsTokenEnv, tc.token)
			if tc.seed {
				traceID := "trace"
				if err := guardsessions.Record(regDir, guardsessions.NewRow(traceID, "claude", 1, t.TempDir(), "audit", "root", time.Now())); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(resume.IdentityLedgerPath(regDir), []byte(`{"uuid":"session","trace":"trace"}`+"\n"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			enqueueGuardOperatorEscalationFailOpen("session", &guardStopTranscript{OperatorDirected: true, OperatorDirectedClass: "AUTHORITY_BOUNDARY", OperatorDirectedDisposition: "HUMAN_RESIDUAL", OperatorDirectedResolve: "ask the operator"})
			entries, err := os.ReadDir(outboxDir)
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				if strings.Contains(entry.Name(), "spool") {
					b, _ := os.ReadFile(filepath.Join(outboxDir, entry.Name()))
					if len(strings.TrimSpace(string(b))) != 0 {
						t.Fatalf("unexpected outbox row: %s", b)
					}
				}
			}
		})
	}
}
