package main

// Tests for the `fak slack outbox compact` operator verb — the manual complement to the
// automatic post-drain compaction (o.maybeCompact). They pin the two invariants an
// operator relies on: --dry-run touches nothing, and a real pass keeps a within-retention
// posted row so a deferred producer's PostedTS read-back still resolves (#2262).

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

// dirSnapshot returns a stable name->bytes map of every regular file in dir, so a test can
// assert a command left the spool byte-for-byte untouched.
func dirSnapshot(t *testing.T, dir string) map[string]string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read spool dir: %v", err)
	}
	snap := map[string]string{}
	names := make([]string, 0, len(ents))
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	for _, n := range names {
		b, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			t.Fatalf("read %s: %v", n, err)
		}
		snap[n] = string(b)
	}
	return snap
}

func TestRunSlackOutboxCompactDryRunInertThenKeepsPostedRow(t *testing.T) {
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	// Enqueue + drain one row to a posted terminal state via the fake server.
	ob, err := openOutbox()
	if err != nil {
		t.Fatalf("open outbox: %v", err)
	}
	nonce, err := ob.Enqueue(slackoutbox.Row{Channel: "C_X", Text: "keep me across a compaction", Source: "test"})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	wire, err := outboxWire("xoxb-test", srv.URL+"/")
	if err != nil {
		t.Fatalf("wire: %v", err)
	}
	if _, err := ob.Drain(ctx(), wire, slackoutbox.DrainOpts{Root: "."}); err != nil {
		t.Fatalf("drain: %v", err)
	}
	if outboxPostedTS(ob, nonce) == "" {
		t.Fatal("setup: row was not posted after drain")
	}
	dir := ob.Dir()

	// --dry-run --json: report is flagged DryRun and the spool is byte-for-byte unchanged.
	before := dirSnapshot(t, dir)
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"compact", "--dry-run", "--json"}); rc != 0 {
		t.Fatalf("compact --dry-run rc=%d stderr=%s", rc, errb.String())
	}
	var dry slackoutbox.CompactReport
	if err := json.Unmarshal(out.Bytes(), &dry); err != nil {
		t.Fatalf("decode dry-run report: %v\n%s", err, out.String())
	}
	if !dry.DryRun {
		t.Fatalf("dry-run report not flagged DryRun: %+v", dry)
	}
	if got := dirSnapshot(t, dir); !mapsEqual(before, got) {
		t.Fatal("compact --dry-run mutated the spool")
	}

	// Real pass: succeeds, is not flagged DryRun, and keeps the posted row (within the 48h
	// posted-retention window) so PostedTS still resolves for a deferred producer.
	out.Reset()
	errb.Reset()
	if rc := runSlackOutbox(&out, &errb, []string{"compact", "--json"}); rc != 0 {
		t.Fatalf("compact rc=%d stderr=%s", rc, errb.String())
	}
	var rep slackoutbox.CompactReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode report: %v\n%s", err, out.String())
	}
	if rep.DryRun {
		t.Fatalf("real compact wrongly flagged DryRun: %+v", rep)
	}
	if outboxPostedTS(ob, nonce) == "" {
		t.Fatal("compaction dropped a within-retention posted row — PostedTS read-back lost")
	}
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func TestRunSlackOutboxCompactPendingAgeRequiresExplicitOptIn(t *testing.T) {
	outboxTestDir(t)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C_STALE", Text: "undelivered", Source: "guard-session", EnqueuedAt: "2020-01-01T00:00:00Z"}); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"compact", "--dry-run", "--max-pending-age", "24h", "--json"}); rc != 0 {
		t.Fatalf("dry-run rc=%d stderr=%s", rc, errb.String())
	}
	var preview slackoutbox.CompactReport
	if err := json.Unmarshal(out.Bytes(), &preview); err != nil {
		t.Fatal(err)
	}
	if preview.DroppedPending != 1 || !preview.DryRun {
		t.Fatalf("preview=%+v, want one aged pending row without mutation", preview)
	}
	out.Reset()
	if rc := runSlackOutbox(&out, &errb, []string{"compact", "--max-pending-age", "24h", "--json"}); rc != 0 {
		t.Fatalf("apply rc=%d stderr=%s", rc, errb.String())
	}
	st, err := ob.Status(time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if st.Pending != 0 {
		t.Fatalf("pending=%d after explicit aged-row retirement", st.Pending)
	}
}
