package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// prPlanFakeLog's gateway unit (aaa1111 feat + bbb2222 fix) with a THIRD
// gateway commit landed after the operator's look — the member set an old ack
// must not cover.
const steerLateJoinerLog = "\x1eeee5555555555555555555555555555555555555\x1ffeat(gateway): late joiner after the look (fak gateway)\x1f\x1f\ninternal/gateway/late.go\n" + prPlanFakeLog

// withSteerRoot pins the steer verbs to a throwaway repo root so a test never
// reads or writes the real overlay ack ledger.
func withSteerRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	orig := steerRoot
	steerRoot = func() string { return root }
	t.Cleanup(func() { steerRoot = orig })
	return root
}

// `fak steer ack <unit>` appends one attributable ledger row bound to the
// unit's exact member SHA set, and the prs view then renders the unit as
// "RESIDUAL (acked by X)" — the band itself untouched, never CLEARED.
func TestSteerAckRecordsLedgerRowAndRendersBesideTheBand(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)

	var stdout, stderr bytes.Buffer
	if code := runSteer(&stdout, &stderr, []string{"ack", "gateway", "--by", "op-jane", "--note", "reviewed the counter fix", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var row steerpr.Ack
	dec := json.NewDecoder(strings.NewReader(stdout.String()))
	if err := dec.Decode(&row); err != nil {
		t.Fatalf("decode echoed row: %v\n%s", err, stdout.String())
	}
	if row.Schema != steerpr.AckSchema || row.Leaf != "gateway" || row.By != "op-jane" || row.At == "" {
		t.Fatalf("echoed row = %#v, want an attributable fak.steerpr.ack.v1 row", row)
	}
	if len(row.SHAs) != 2 || row.SHAs[0] != steerFeatSHA || row.SHAs[1] != steerFixSHA {
		t.Fatalf("row SHAs = %v, want the unit's exact member set [%s %s]", row.SHAs, steerFeatSHA, steerFixSHA)
	}
	if _, err := os.Stat(steerpr.AckLedgerPath(root)); err != nil {
		t.Fatalf("ledger row not on disk: %v", err)
	}

	// The prs view now shows the ack BESIDE the machine band — and the machine
	// numbers are exactly what they were: still RESIDUAL, still counted.
	stdout.Reset()
	stderr.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "## [RESIDUAL (acked by op-jane)] gateway") {
		t.Fatalf("render missing the acked-beside-the-band header:\n%s", out)
	}
	if !strings.Contains(out, "1 RESIDUAL") {
		t.Fatalf("acking must not deflate the residual count:\n%s", out)
	}
	if strings.Contains(out, "CLEARED (") || strings.Contains(out, "## [CLEARED]") {
		t.Fatalf("an acked residual rendered as CLEARED — an ack laundered into a witness:\n%s", out)
	}

	// The JSON payload carries the acked state as a separate field from band.
	stdout.Reset()
	stderr.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--json", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs --json exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var payload struct {
		ResidualCount int `json:"residual_count"`
		Units         []struct {
			Band string `json:"band"`
		} `json:"units"`
		Acks map[string]steerpr.Ack `json:"acks"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	if payload.ResidualCount != 1 || len(payload.Units) != 1 || payload.Units[0].Band != string(steerpr.BandResidual) {
		t.Fatalf("machine band/count moved under an ack: %#v", payload)
	}
	if a, ok := payload.Acks["gateway"]; !ok || a.By != "op-jane" {
		t.Fatalf("acks payload = %#v, want a gateway row acked by op-jane", payload.Acks)
	}
}

// Done condition: a new member commit invalidates the ack — the ledger row was
// bound to the OLD SHA set, so the grown unit reads RESIDUAL/unacked again.
func TestSteerAckInvalidatedByNewMemberCommit(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	withSteerRoot(t)

	var stdout, stderr bytes.Buffer
	if code := runSteerAck(&stdout, &stderr, []string{"gateway", "--by", "op-jane", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("ack exit = %d, want 0; stderr=%s", code, stderr.String())
	}

	// A third gateway commit lands after the look.
	withSteerFakes(t, steerLateJoinerLog, steerpr.VerdictUnwitnessed)
	stdout.Reset()
	stderr.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "## [RESIDUAL] gateway") {
		t.Fatalf("grown unit should read bare RESIDUAL again:\n%s", out)
	}
	if strings.Contains(out, "acked by") {
		t.Fatalf("the prior ack covered a different SHA set and must not decorate the grown unit:\n%s", out)
	}
}

// An unattributable ack is refused, and acking a unit that is not forming
// fails rather than inventing a ledger row.
func TestSteerAckRefusals(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)

	// No --by, and the faked git yields no config user.name.
	var stdout, stderr bytes.Buffer
	if code := runSteerAck(&stdout, &stderr, []string{"gateway", "--base", "baseref", "--head", "headref"}); code != 2 {
		t.Fatalf("unattributable ack exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "attributable") {
		t.Fatalf("refusal should say why attribution is required: %s", stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := runSteerAck(&stdout, &stderr, []string{"no-such-leaf", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("unknown unit exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "no forming unit") {
		t.Fatalf("refusal should name the missing unit: %s", stderr.String())
	}

	// No unit at all: usage.
	stdout.Reset()
	stderr.Reset()
	if code := runSteerAck(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("missing unit exit = %d, want 2; stderr=%s", code, stderr.String())
	}

	// None of the refusals may have written a ledger row.
	if rows := steerpr.LoadAcks(steerpr.AckLedgerPath(root)); len(rows) != 0 {
		t.Fatalf("a refused ack wrote %d ledger row(s): %#v", len(rows), rows)
	}
}
