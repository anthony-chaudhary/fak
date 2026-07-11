package main

// Tests for the `fak slack outbox limits` operator verb — the read-only view of the
// retention/compaction envelope and where the live spool sits against it.

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

func TestRunSlackOutboxLimitsJSONReportsWindows(t *testing.T) {
	outboxTestDir(t)

	ob, err := openOutbox()
	if err != nil {
		t.Fatalf("open outbox: %v", err)
	}
	// One pending row: terminal-count zero, nothing droppable, no pass due.
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C_X", Text: "owed", Source: "test"}); err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"limits", "--json"}); rc != 0 {
		t.Fatalf("limits --json rc=%d stderr=%s", rc, errb.String())
	}
	var lim slackoutbox.Limits
	if err := json.Unmarshal(out.Bytes(), &lim); err != nil {
		t.Fatalf("decode limits: %v\n%s", err, out.String())
	}
	if lim.RetainDeadS != int64(slackoutbox.DefaultRetainDead.Seconds()) {
		t.Fatalf("retain_dead_s = %d, want %d", lim.RetainDeadS, int64(slackoutbox.DefaultRetainDead.Seconds()))
	}
	if lim.TerminalRows != 0 || lim.DroppableRows != 0 || lim.CompactionDue {
		t.Fatalf("a lone pending row should leave the outbox idle vs its limits: %+v", lim)
	}
}

func TestRunSlackOutboxLimitsHumanLine(t *testing.T) {
	outboxTestDir(t)
	if _, err := openOutbox(); err != nil {
		t.Fatalf("open outbox: %v", err)
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"limits"}); rc != 0 {
		t.Fatalf("limits rc=%d stderr=%s", rc, errb.String())
	}
	// The human line names each window and the live occupancy summary.
	for _, want := range []string{"retain settled", "dead 336h", "pass due no"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("limits human line missing %q:\n%s", want, out.String())
		}
	}
}
