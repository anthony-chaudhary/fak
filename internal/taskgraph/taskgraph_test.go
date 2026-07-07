package taskgraph

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// findTask returns the folded row for id, failing the test if it is absent.
func findTask(t *testing.T, tab Table, id string) Task {
	t.Helper()
	for _, tk := range tab.Tasks {
		if tk.ID == id {
			return tk
		}
	}
	t.Fatalf("task %q not in table", id)
	return Task{}
}

func hasFinding(tk Task, reason string) bool {
	for _, f := range tk.Findings {
		if f.Reason == reason {
			return true
		}
	}
	return false
}

// TestTaskFold_ByteIdenticalReplay is the pure-fold witness: the same journal +
// same now + same config folds to a byte-identical table on replay. The Sample
// journal exercises every status and both lease refusals, so this also pins the
// determinism of the finding order and the counts.
func TestTaskFold_ByteIdenticalReplay(t *testing.T) {
	events, now, cfg := Sample()

	first, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	second, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatalf("replay fold: %v", err)
	}

	a, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal first: %v", err)
	}
	b, err := json.Marshal(second)
	if err != nil {
		t.Fatalf("marshal second: %v", err)
	}
	if string(a) != string(b) {
		t.Fatalf("fold is not byte-identical on replay:\n first=%s\nsecond=%s", a, b)
	}

	// A folded-again table from a freshly parsed copy of the same bytes must also
	// match — the journal round-trips through JSONL without changing the fold.
	var lines strings.Builder
	for _, ev := range events {
		row, err := json.Marshal(ev)
		if err != nil {
			t.Fatalf("marshal event: %v", err)
		}
		lines.Write(row)
		lines.WriteByte('\n')
	}
	reparsed, err := ParseEvents(strings.NewReader(lines.String()))
	if err != nil {
		t.Fatalf("reparse: %v", err)
	}
	third, err := Fold(reparsed, now, cfg)
	if err != nil {
		t.Fatalf("fold reparsed: %v", err)
	}
	c, err := json.Marshal(third)
	if err != nil {
		t.Fatalf("marshal third: %v", err)
	}
	if string(a) != string(c) {
		t.Fatalf("fold over reparsed journal differs:\n direct=%s\nreparsed=%s", a, c)
	}
}

// TestClaim_WithoutLiveLeaseRefused: a claim carrying an already-dead lease
// (expiry at or before the claim instant) is refused — the task stays unclaimed
// and reads OPEN, with a closed TASK_CLAIM_NO_LIVE_LEASE reason a reader can
// witness from the journal alone.
func TestClaim_WithoutLiveLeaseRefused(t *testing.T) {
	const base int64 = 1_000_000
	events := []Event{
		{Kind: EvCreated, Task: "t1", Tree: []string{"internal/foo/**"}, AtMS: base},
		// dead-on-arrival: the lease expired before the moment it was cited.
		{Kind: EvClaimed, Task: "t1", Owner: "w1", Lease: "L-dead", LeaseExpiresMS: base + 100, AtMS: base + 500},
	}
	tab, err := Fold(events, base+1_000, Config{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	tk := findTask(t, tab, "t1")
	if tk.Status != StatusOpen {
		t.Fatalf("refused claim must leave the task OPEN, got %s", tk.Status)
	}
	if tk.Owner != "" || tk.Lease != "" {
		t.Fatalf("refused claim must not set an owner/lease, got owner=%q lease=%q", tk.Owner, tk.Lease)
	}
	if !hasFinding(tk, ReasonTaskClaimNoLiveLeaseName) {
		t.Fatalf("expected %s finding, got %+v", ReasonTaskClaimNoLiveLeaseName, tk.Findings)
	}
	if !tab.AttentionNeeded || tab.Counts.Refused != 1 {
		t.Fatalf("a refusal must raise attention: attention=%t refused=%d", tab.AttentionNeeded, tab.Counts.Refused)
	}

	// An empty lease id is equally refused — a claim with no admitted lease.
	events[1].Lease = ""
	events[1].LeaseExpiresMS = base + 100_000
	tab, err = Fold(events, base+1_000, Config{})
	if err != nil {
		t.Fatalf("fold empty-lease: %v", err)
	}
	tk = findTask(t, tab, "t1")
	if tk.Status != StatusOpen || !hasFinding(tk, ReasonTaskClaimNoLiveLeaseName) {
		t.Fatalf("a claim with no lease id must be refused, got status=%s findings=%+v", tk.Status, tk.Findings)
	}
}

// TestComplete_OpenBlockersRefused: completing a task while a declared blocker
// is not yet completed is refused — the task stays BLOCKED with a closed
// TASK_COMPLETE_OPEN_BLOCKERS reason, never silently marked done.
func TestComplete_OpenBlockersRefused(t *testing.T) {
	const base int64 = 2_000_000
	events := []Event{
		{Kind: EvCreated, Task: "dep", Title: "the blocker", AtMS: base},
		{Kind: EvCreated, Task: "leaf", Title: "waits on dep", BlockedBy: []string{"dep"}, AtMS: base + 100},
		{Kind: EvClaimed, Task: "leaf", Owner: "w1", Lease: "L1", LeaseExpiresMS: base + 500_000, AtMS: base + 200},
		// dep is still open here — the completion must be refused.
		{Kind: EvCompleted, Task: "leaf", AtMS: base + 300},
	}
	tab, err := Fold(events, base+1_000, Config{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	tk := findTask(t, tab, "leaf")
	if tk.Status == StatusCompleted {
		t.Fatalf("completing over an open blocker must be refused, got %s", tk.Status)
	}
	if tk.Status != StatusBlocked {
		t.Fatalf("a task with an open blocker reads BLOCKED, got %s", tk.Status)
	}
	if !hasFinding(tk, ReasonTaskCompleteOpenBlockersName) {
		t.Fatalf("expected %s finding, got %+v", ReasonTaskCompleteOpenBlockersName, tk.Findings)
	}

	// Once the blocker completes, the same completion event succeeds — the fold
	// gates on evidence, not on the honor rule.
	unblocked := []Event{
		{Kind: EvCreated, Task: "dep", AtMS: base},
		{Kind: EvCreated, Task: "leaf", BlockedBy: []string{"dep"}, AtMS: base + 100},
		{Kind: EvClaimed, Task: "leaf", Owner: "w1", Lease: "L1", LeaseExpiresMS: base + 500_000, AtMS: base + 200},
		{Kind: EvCompleted, Task: "dep", AtMS: base + 250},
		{Kind: EvCompleted, Task: "leaf", AtMS: base + 300},
	}
	tab, err = Fold(unblocked, base+1_000, Config{})
	if err != nil {
		t.Fatalf("fold unblocked: %v", err)
	}
	if got := findTask(t, tab, "leaf"); got.Status != StatusCompleted {
		t.Fatalf("a fully-unblocked completion should stand, got %s findings=%+v", got.Status, got.Findings)
	}
}

// TestClaim_TreeCollisionRefused: two workers cannot hold live claims over
// intersecting trees. The second colliding claim is refused so the fold enforces
// the same disjointness the arbiter admits leases under.
func TestClaim_TreeCollisionRefused(t *testing.T) {
	const base int64 = 3_000_000
	events := []Event{
		{Kind: EvCreated, Task: "wide", Tree: []string{"internal/**"}, AtMS: base},
		{Kind: EvCreated, Task: "narrow", Tree: []string{"internal/taskgraph/**"}, AtMS: base + 10},
		{Kind: EvClaimed, Task: "wide", Owner: "w1", Lease: "L1", LeaseExpiresMS: base + 500_000, AtMS: base + 100},
		// narrow's tree is under wide's live claim — refused.
		{Kind: EvClaimed, Task: "narrow", Owner: "w2", Lease: "L2", LeaseExpiresMS: base + 500_000, AtMS: base + 200},
	}
	tab, err := Fold(events, base+1_000, Config{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	if got := findTask(t, tab, "wide"); got.Status != StatusClaimed {
		t.Fatalf("first claim should stand, got %s", got.Status)
	}
	narrow := findTask(t, tab, "narrow")
	if narrow.Status == StatusClaimed {
		t.Fatalf("colliding claim must be refused, got %s", narrow.Status)
	}
	if !hasFinding(narrow, ReasonTaskClaimTreeCollisionName) {
		t.Fatalf("expected %s finding, got %+v", ReasonTaskClaimTreeCollisionName, narrow.Findings)
	}

	// Disjoint trees claim cleanly — the check is not blanket-deny.
	disjoint := []Event{
		{Kind: EvCreated, Task: "a", Tree: []string{"internal/aaa/**"}, AtMS: base},
		{Kind: EvCreated, Task: "b", Tree: []string{"internal/bbb/**"}, AtMS: base + 10},
		{Kind: EvClaimed, Task: "a", Owner: "w1", Lease: "L1", LeaseExpiresMS: base + 500_000, AtMS: base + 100},
		{Kind: EvClaimed, Task: "b", Owner: "w2", Lease: "L2", LeaseExpiresMS: base + 500_000, AtMS: base + 200},
	}
	tab, err = Fold(disjoint, base+1_000, Config{})
	if err != nil {
		t.Fatalf("fold disjoint: %v", err)
	}
	if got := findTask(t, tab, "b"); got.Status != StatusClaimed {
		t.Fatalf("a disjoint claim should stand, got %s findings=%+v", got.Status, got.Findings)
	}
}

// TestClaim_ExpiredAtNowReclaimable: a claim admitted live but whose lease has
// since aged out at nowMS is not a refusal — the task returns to OPEN with the
// owner cleared, so a stale hold never blocks the pool.
func TestClaim_ExpiredAtNowReclaimable(t *testing.T) {
	const base int64 = 4_000_000
	events := []Event{
		{Kind: EvCreated, Task: "t", Tree: []string{"internal/foo/**"}, AtMS: base},
		{Kind: EvClaimed, Task: "t", Owner: "w1", Lease: "L1", LeaseExpiresMS: base + 1_000, AtMS: base + 100},
	}
	// nowMS is past the lease expiry.
	tab, err := Fold(events, base+5_000, Config{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	tk := findTask(t, tab, "t")
	if tk.Status != StatusOpen {
		t.Fatalf("an expired claim reads OPEN, got %s", tk.Status)
	}
	if tk.Owner != "" || tk.LeaseLive {
		t.Fatalf("an expired claim clears the owner: owner=%q live=%t", tk.Owner, tk.LeaseLive)
	}
	if len(tk.Findings) != 0 {
		t.Fatalf("an aged-out claim is not a refusal, got %+v", tk.Findings)
	}
	if len(Dispatchable(tab)) != 1 || Dispatchable(tab)[0] != "t" {
		t.Fatalf("a reclaimable task is dispatchable, got %v", Dispatchable(tab))
	}
}

// TestParseEvents_FailsClosed: an unknown enum token or a missing identity field
// refuses the whole journal at the boundary, naming the offending line.
func TestParseEvents_FailsClosed(t *testing.T) {
	cases := []string{
		`{"kind":"teleported","task":"t","at_unix_ms":1}`,
		`{"kind":"created","at_unix_ms":1}`,            // no task id
		`{"kind":"claimed","task":"t","at_unix_ms":1}`, // no owner
		`{"kind":"created","task":"t","at_unix_ms":0}`, // non-positive clock
		`{"kind":"created","task":"t"`,                 // malformed JSON
	}
	for _, raw := range cases {
		if _, err := ParseEvents(strings.NewReader(raw)); err == nil {
			t.Fatalf("expected a fail-closed refusal for %q", raw)
		}
	}

	// A comment line and a blank line are skipped, not refused.
	ok := "# a comment\n\n" + `{"kind":"created","task":"t","at_unix_ms":1}` + "\n"
	evs, err := ParseEvents(strings.NewReader(ok))
	if err != nil {
		t.Fatalf("valid journal refused: %v", err)
	}
	if len(evs) != 1 {
		t.Fatalf("expected 1 event past the comment/blank, got %d", len(evs))
	}
}

// TestSample_Statuses pins the demo journal to one row per status class plus both
// refusals, so `fak tasks sample` stays a real proof of every column.
func TestSample_Statuses(t *testing.T) {
	events, now, cfg := Sample()
	tab, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatalf("fold sample: %v", err)
	}
	want := map[string]Status{
		"spine": StatusCompleted,
		"cli":   StatusOpen, // its only blocker (spine) is done
		"wire":  StatusBlocked,
		"docs":  StatusClaimed,
		"flaky": StatusOpen, // its bad claim was refused
	}
	for id, status := range want {
		if got := findTask(t, tab, id); got.Status != status {
			t.Errorf("sample task %q: want %s, got %s", id, status, got.Status)
		}
	}
	if got := findTask(t, tab, "flaky"); !hasFinding(got, ReasonTaskClaimNoLiveLeaseName) {
		t.Errorf("sample must carry the lease refusal, got %+v", got.Findings)
	}
	if !tab.AttentionNeeded {
		t.Errorf("sample carries a refusal, so attention must be raised")
	}
}

// TestRenderText_Columns is the render witness for `fak tasks table`: the folded
// sample journal renders the owner-lease, status, and blockedBy columns from the
// journal alone. It captures the exact bytes the CLI emits (cmd/fak delegates to
// RenderText), so the acceptance is provable off the cmd build.
func TestRenderText_Columns(t *testing.T) {
	events, now, cfg := Sample()
	tab, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	var buf bytes.Buffer
	RenderText(&buf, tab)
	out := buf.String()

	for _, col := range []string{"STATUS", "OWNER-LEASE", "BLOCKED-BY"} {
		if !strings.Contains(out, col) {
			t.Errorf("render is missing the %q column header:\n%s", col, out)
		}
	}
	// A live-claimed task shows its owner-lease; a blocked task shows its open
	// blocker; a refused claim shows its closed reason — all from the journal.
	for _, want := range []string{
		"docs", "CLAIMED", "w2@L-docs", // owner-lease column
		"wire", "BLOCKED", "cli", // blockedBy column
		"!! " + ReasonTaskClaimNoLiveLeaseName, // refusal surfaced
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render is missing %q:\n%s", want, out)
		}
	}
}
