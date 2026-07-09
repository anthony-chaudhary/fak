package slackoutbox

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// noSleepOutbox returns a test outbox whose compaction never really sleeps and whose clock
// is pinned to base, so age-based retention is exercised deterministically.
func noSleepOutbox(t *testing.T, base time.Time) *Outbox {
	t.Helper()
	o := testOutbox(t)
	o.now = func() time.Time { return base }
	o.compactSleep = func(time.Duration) {}
	return o
}

// setState drives a row to a terminal/settled state stamped at a specific time, so
// retention windows can be measured against a known age.
func setState(t *testing.T, o *Outbox, nonce, state, ts string, at time.Time) {
	t.Helper()
	if err := o.appendState(transition{Nonce: nonce, State: state, TS: ts, At: at.UTC().Format(time.RFC3339)}); err != nil {
		t.Fatal(err)
	}
}

func mustLoad(t *testing.T, o *Outbox) *Snapshot {
	t.Helper()
	snap, err := o.Load()
	if err != nil {
		t.Fatal(err)
	}
	return snap
}

func hasNonce(snap *Snapshot, nonce string) bool {
	for _, r := range snap.Rows {
		if r.Nonce == nonce {
			return true
		}
	}
	return false
}

func TestCompactDropsSettledKeepsOwedPostedAndDead(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	nPost, _ := o.Enqueue(Row{Channel: "C1", Text: "posted"})
	nSup, _ := o.Enqueue(Row{Channel: "C1", Text: "superseded"})
	nRef, _ := o.Enqueue(Row{Channel: "C1", Text: "refused"})
	nDead, _ := o.Enqueue(Row{Channel: "C1", Text: "dead"})
	nPend, _ := o.Enqueue(Row{Channel: "C1", Text: "pending"})

	setState(t, o, nPost, statePosted, "1.1", base)
	setState(t, o, nSup, stateSuperseded, "", base)
	setState(t, o, nRef, stateRefused, "", base)
	setState(t, o, nDead, stateDead, "", base)

	// Two hours later: past the 1h settled window, well within the 48h posted window.
	rep, err := o.Compact(CompactOpts{Now: base.Add(2 * time.Hour)})
	if err != nil {
		t.Fatal(err)
	}
	if rep.DroppedSuperseded != 1 || rep.DroppedRefused != 1 || rep.DroppedPosted != 0 {
		t.Fatalf("drop tallies wrong: %+v", rep)
	}

	snap := mustLoad(t, o)
	if hasNonce(snap, nSup) || hasNonce(snap, nRef) {
		t.Fatal("settled rows past their window must be dropped")
	}
	for _, keep := range []string{nPost, nDead, nPend} {
		if !hasNonce(snap, keep) {
			t.Fatalf("row %s must survive compaction", keep)
		}
	}
	// Guard-card safety: a surviving posted row's ts is still resolvable by nonce.
	if snap.PostedTS(nPost) != "1.1" {
		t.Fatalf("posted ts not resolvable after compaction: %q", snap.PostedTS(nPost))
	}
	if snap.state(nDead).State != stateDead || snap.state(nPend).State != statePending {
		t.Fatalf("survivor states not preserved: dead=%q pend=%q", snap.state(nDead).State, snap.state(nPend).State)
	}
}

func TestCompactDropsPostedOnlyPastSafeWindow(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)

	// Within 48h: kept (a guard session could still probe its ts).
	o := noSleepOutbox(t, base)
	n, _ := o.Enqueue(Row{Channel: "C1", Text: "posted"})
	setState(t, o, n, statePosted, "9.9", base)
	if rep, err := o.Compact(CompactOpts{Now: base.Add(47 * time.Hour)}); err != nil || rep.DroppedPosted != 0 {
		t.Fatalf("posted within window dropped: rep=%+v err=%v", rep, err)
	}
	if !hasNonce(mustLoad(t, o), n) {
		t.Fatal("posted row within window must survive")
	}

	// Past 48h: dropped.
	o2 := noSleepOutbox(t, base)
	n2, _ := o2.Enqueue(Row{Channel: "C1", Text: "posted"})
	setState(t, o2, n2, statePosted, "9.9", base)
	if rep, err := o2.Compact(CompactOpts{Now: base.Add(49 * time.Hour)}); err != nil || rep.DroppedPosted != 1 {
		t.Fatalf("posted past window not dropped: rep=%+v err=%v", rep, err)
	}
	if hasNonce(mustLoad(t, o2), n2) {
		t.Fatal("posted row past 48h must be dropped")
	}
}

func TestCompactDropsDeadOnlyPastLongWindow(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)

	// A dead row well past the 48h posted window is STILL kept — an operator may retry it.
	o := noSleepOutbox(t, base)
	n, _ := o.Enqueue(Row{Channel: "C1", Text: "dead"})
	setState(t, o, n, stateDead, "", base)
	if rep, err := o.Compact(CompactOpts{Now: base.Add(13 * 24 * time.Hour)}); err != nil || rep.DroppedDead != 0 {
		t.Fatalf("dead within window dropped: rep=%+v err=%v", rep, err)
	}
	if !hasNonce(mustLoad(t, o), n) {
		t.Fatal("dead row within the 14d window must survive so it stays retryable")
	}

	// Past the 14d window a dead row nobody re-armed is finally dropped — the retention
	// exception that keeps a steadily dead-lettering fleet's spool bounded.
	o2 := noSleepOutbox(t, base)
	n2, _ := o2.Enqueue(Row{Channel: "C1", Text: "dead"})
	setState(t, o2, n2, stateDead, "", base)
	if rep, err := o2.Compact(CompactOpts{Now: base.Add(15 * 24 * time.Hour)}); err != nil || rep.DroppedDead != 1 {
		t.Fatalf("dead past window not dropped: rep=%+v err=%v", rep, err)
	}
	if hasNonce(mustLoad(t, o2), n2) {
		t.Fatal("dead row past the 14d window must be dropped")
	}

	// A custom --retain-dead window is honored.
	o3 := noSleepOutbox(t, base)
	n3, _ := o3.Enqueue(Row{Channel: "C1", Text: "dead"})
	setState(t, o3, n3, stateDead, "", base)
	if rep, err := o3.Compact(CompactOpts{Now: base.Add(2 * time.Hour), RetainDead: time.Hour}); err != nil || rep.DroppedDead != 1 {
		t.Fatalf("custom dead window not honored: rep=%+v err=%v", rep, err)
	}
}

func TestCompactCollapsesDrainPassHeartbeats(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	// One owed row plus a storm of drain_pass heartbeats.
	nPend, _ := o.Enqueue(Row{Channel: "C1", Text: "owed"})
	last := base
	for i := 0; i < 20; i++ {
		last = base.Add(time.Duration(i) * time.Minute)
		if err := o.appendState(transition{State: stateDrainPass, At: last.UTC().Format(time.RFC3339)}); err != nil {
			t.Fatal(err)
		}
	}
	before := mustLoad(t, o)
	if before.drainPasses != 20 || !before.LastDrainAt.Equal(last) {
		t.Fatalf("pre-compact heartbeats=%d last=%v", before.drainPasses, before.LastDrainAt)
	}

	rep, err := o.Compact(CompactOpts{Now: base.Add(time.Hour)})
	if err != nil {
		t.Fatal(err)
	}
	if rep.CollapsedDrainPass != 20 {
		t.Fatalf("report collapsed=%d, want 20", rep.CollapsedDrainPass)
	}

	after := mustLoad(t, o)
	if after.drainPasses != 1 {
		t.Fatalf("heartbeats not collapsed: %d", after.drainPasses)
	}
	if !after.LastDrainAt.Equal(last) {
		t.Fatalf("LastDrainAt not preserved: %v want %v", after.LastDrainAt, last)
	}
	if !hasNonce(after, nPend) {
		t.Fatal("owed row lost during heartbeat collapse")
	}
	if after.LastCompactAt.IsZero() {
		t.Fatal("compact_pass heartbeat not written")
	}
}

func TestCompactDoesNotLoseAnInFlightSealAppend(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := testOutbox(t)
	o.now = func() time.Time { return base }

	// A row already on the head; it will be sealed when compaction starts.
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "already-head"}); err != nil {
		t.Fatal(err)
	}

	// Simulate a producer whose O_APPEND opened before the seal rename and whose write
	// lands in the sealed inode DURING quiesce: append a fresh row to the seal segment on
	// the first quiesce wait. The size grows, quiesce re-loops, and the fold must see it.
	injected := false
	o.compactSleep = func(time.Duration) {
		if injected {
			return
		}
		injected = true
		f, err := os.OpenFile(filepath.Join(o.Dir(), spoolSealFile), os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return // seal not present yet on this wait — fine, nothing to inject
		}
		_, _ = f.WriteString(`{"nonce":"inflight","channel":"C1","text":"landed-in-seal"}` + "\n")
		_ = f.Close()
	}

	if _, err := o.Compact(CompactOpts{Now: base}); err != nil {
		t.Fatal(err)
	}
	if !injected {
		t.Fatal("quiesce never waited — the in-flight-append window was not exercised")
	}
	snap := mustLoad(t, o)
	if !hasNonce(snap, "inflight") {
		t.Fatal("an in-flight append that landed in the sealed segment was LOST by compaction")
	}
	if !hasNonce(snap, o.mustNonce(t, "already-head")) {
		t.Fatal("the pre-seal head row was lost")
	}
}

// mustNonce finds the nonce of the (single) row with the given text — the enqueue minted it.
func (o *Outbox) mustNonce(t *testing.T, text string) string {
	t.Helper()
	for _, r := range mustLoad(t, o).Rows {
		if r.Text == text {
			return r.Nonce
		}
	}
	t.Fatalf("no row with text %q", text)
	return ""
}

func TestCompactRecoversLeftoverSeal(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	// A row on the head, then simulate a crash mid-compaction: the head was sealed but the
	// archive rewrite never ran, so a *.seal.jsonl is left behind.
	n, _ := o.Enqueue(Row{Channel: "C1", Text: "survivor"})
	if err := os.Rename(filepath.Join(o.Dir(), spoolFile), filepath.Join(o.Dir(), spoolSealFile)); err != nil {
		t.Fatal(err)
	}

	// Load still sees the row (a leftover seal is just an older segment).
	if !hasNonce(mustLoad(t, o), n) {
		t.Fatal("leftover seal not folded by Load")
	}

	if _, err := o.Compact(CompactOpts{Now: base.Add(time.Minute)}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(o.Dir(), spoolSealFile)); !os.IsNotExist(err) {
		t.Fatalf("seal not consumed by compaction: stat err=%v", err)
	}
	if !hasNonce(mustLoad(t, o), n) {
		t.Fatal("survivor lost during seal recovery")
	}
}

func TestCompactGCsFinalCardsOnly(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	cardsDir := filepath.Join(o.Dir(), cardsSubdir)
	if err := os.MkdirAll(cardsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(name, body string, mtime time.Time) string {
		p := filepath.Join(cardsDir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, mtime, mtime); err != nil {
			t.Fatal(err)
		}
		return p
	}
	oldFinal := write("final-old.json", `{"channel":"C1","final":true}`, base.Add(-72*time.Hour))
	oldLive := write("live-old.json", `{"channel":"C1"}`, base.Add(-72*time.Hour))
	freshFinal := write("final-fresh.json", `{"channel":"C1","final":true}`, base)

	rep, err := o.Compact(CompactOpts{Now: base})
	if err != nil {
		t.Fatal(err)
	}
	if rep.DroppedCards != 1 {
		t.Fatalf("dropped cards = %d, want 1", rep.DroppedCards)
	}
	if _, err := os.Stat(oldFinal); !os.IsNotExist(err) {
		t.Fatal("finalized old card must be GC'd")
	}
	if _, err := os.Stat(oldLive); err != nil {
		t.Fatal("non-final (live) card must be kept regardless of age")
	}
	if _, err := os.Stat(freshFinal); err != nil {
		t.Fatal("finalized but fresh card must be kept")
	}
}

func TestCompactDryRunMutatesNothing(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	n, _ := o.Enqueue(Row{Channel: "C1", Text: "superseded"})
	setState(t, o, n, stateSuperseded, "", base)

	spool := filepath.Join(o.Dir(), spoolFile)
	sizeBefore := fileSize(spool)

	rep, err := o.Compact(CompactOpts{Now: base.Add(2 * time.Hour), DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.DryRun || rep.DroppedSuperseded != 1 {
		t.Fatalf("dry run report wrong: %+v", rep)
	}
	if fileSize(spool) != sizeBefore {
		t.Fatal("dry run mutated the live spool")
	}
	if _, err := os.Stat(filepath.Join(o.Dir(), spoolArchFile)); !os.IsNotExist(err) {
		t.Fatal("dry run created an archive")
	}
	if !hasNonce(mustLoad(t, o), n) {
		t.Fatal("dry run dropped a row")
	}
}

func TestCompactionDueGate(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	// A fresh posted row (nothing droppable, tiny backlog): not due.
	n, _ := o.Enqueue(Row{Channel: "C1", Text: "posted"})
	setState(t, o, n, statePosted, "1.1", base)
	fresh := CompactOpts{Now: base.Add(time.Minute)}.norm(o)
	if o.compactionDue(mustLoad(t, o), fresh) {
		t.Fatal("compaction should not be due when nothing is past its window")
	}

	// A superseded row past the settled window: due.
	nSup, _ := o.Enqueue(Row{Channel: "C1", Text: "superseded"})
	setState(t, o, nSup, stateSuperseded, "", base)
	due := CompactOpts{Now: base.Add(2 * time.Hour)}.norm(o)
	if !o.compactionDue(mustLoad(t, o), due) {
		t.Fatal("compaction should be due once a settled row is past its window")
	}

	// Recently compacted (within the interval): not due even with a droppable row.
	snap := mustLoad(t, o)
	snap.LastCompactAt = base.Add(2*time.Hour - time.Minute)
	if o.compactionDue(snap, due) {
		t.Fatal("compaction should back off within CompactMinInterval of the last pass")
	}
}

func TestDrainAutoCompactsWhenDue(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := testOutbox(t)
	o.now = func() time.Time { return base.Add(2 * time.Hour) } // 2h after the settled row
	o.compactSleep = func(time.Duration) {}

	// A stale superseded row (droppable) plus a fresh row for the drain to actually post.
	nSup, _ := o.Enqueue(Row{Channel: "C1", Text: "superseded"})
	setState(t, o, nSup, stateSuperseded, "", base)
	nFresh, _ := o.Enqueue(Row{Channel: "C1", Text: "fresh"})

	rep, err := o.Drain(context.Background(), newFakeWire(), drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Posted != 1 {
		t.Fatalf("drain should have posted the fresh row: %+v", rep)
	}

	snap := mustLoad(t, o)
	if hasNonce(snap, nSup) {
		t.Fatal("drain did not auto-compact the stale superseded row")
	}
	if snap.PostedTS(nFresh) == "" {
		t.Fatal("freshly posted row lost across the auto-compaction")
	}
	if snap.LastCompactAt.IsZero() {
		t.Fatal("auto-compaction left no compact_pass heartbeat")
	}
}
