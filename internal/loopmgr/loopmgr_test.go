package loopmgr

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

func TestAppendLoadValidatesHashChainAndSummary(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	now := fixedClock()
	events := []Event{
		{LoopID: "issue-dispatch/default", Kind: EventFire, Source: "schedule", Principal: "timer"},
		{LoopID: "issue-dispatch/default", Kind: EventAdmit, RunID: "run-1", Status: StatusAdmitted},
		{LoopID: "issue-dispatch/default", Kind: EventStart, RunID: "run-1"},
		{LoopID: "issue-dispatch/default", Kind: EventEnd, RunID: "run-1", Status: StatusClaimedDone},
		{LoopID: "issue-dispatch/default", Kind: EventWitness, RunID: "run-1", Status: StatusWitnessedDone, EvidenceRefs: []EvidenceRef{{Kind: "commit", Ref: "8469c56"}}},
		{LoopID: "issue-dispatch/default", Kind: EventNotify, RunID: "run-1", Reason: "DONE_WITNESSED"},
	}
	for _, ev := range events {
		if _, err := Append(path, ev, WithClock(now)); err != nil {
			t.Fatalf("Append(%s): %v", ev.Kind, err)
		}
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded) != len(events) {
		t.Fatalf("loaded %d events, want %d", len(loaded), len(events))
	}
	for i, ev := range loaded {
		if ev.Seq != uint64(i+1) {
			t.Fatalf("event %d seq = %d", i, ev.Seq)
		}
		if ev.Hash == "" {
			t.Fatalf("event %d missing hash", i)
		}
		if i > 0 && ev.PrevHash != loaded[i-1].Hash {
			t.Fatalf("event %d prev hash = %q, want prior %q", i, ev.PrevHash, loaded[i-1].Hash)
		}
	}

	st := Summarize(loaded, time.Unix(0, 7000).UTC())
	if st.Schema != SchemaStatus {
		t.Fatalf("status schema = %q, want %q", st.Schema, SchemaStatus)
	}
	if len(st.Loops) != 1 {
		t.Fatalf("loops = %d, want 1", len(st.Loops))
	}
	loop := st.Loops[0]
	if loop.LoopID != "issue-dispatch/default" || loop.Fires != 1 || loop.Admitted != 1 || loop.Started != 1 || loop.Ended != 1 || loop.Witnessed != 1 || loop.Notifications != 1 {
		t.Fatalf("summary = %+v", loop)
	}
	if loop.LastRun == nil || loop.LastRun.Status != StatusWitnessedDone || len(loop.LastRun.EvidenceRefs) != 1 {
		t.Fatalf("last run = %+v", loop.LastRun)
	}
}

func TestLoadRejectsTamperedLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	if _, err := Append(path, Event{LoopID: "loop-a", Kind: EventFire, Source: "schedule"}, WithClock(fixedClock())); err != nil {
		t.Fatalf("Append: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	tampered := strings.Replace(string(b), `"schedule"`, `"slack"`, 1)
	if err := os.WriteFile(path, []byte(tampered), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "hash") {
		t.Fatalf("Load tampered err = %v, want hash error", err)
	}
}

func TestLoadMissingLedgerIsEmpty(t *testing.T) {
	events, err := Load(filepath.Join(t.TempDir(), "missing.jsonl"))
	if err != nil {
		t.Fatalf("Load missing: %v", err)
	}
	if len(events) != 0 {
		t.Fatalf("missing ledger events = %d, want 0", len(events))
	}
}

func TestSummarizeSortsLoops(t *testing.T) {
	st := Summarize([]Event{
		{Schema: SchemaEvent, Seq: 1, LoopID: "z-loop", Kind: EventFire},
		{Schema: SchemaEvent, Seq: 2, LoopID: "a-loop", Kind: EventFire},
	}, time.Unix(0, 1))
	if len(st.Loops) != 2 || st.Loops[0].LoopID != "a-loop" || st.Loops[1].LoopID != "z-loop" {
		t.Fatalf("loops sorted = %+v", st.Loops)
	}
}

func fixedClock() func() time.Time {
	return func() time.Time { return time.Unix(0, 1000).UTC() }
}

// TestAppendConcurrentDoesNotForkChain is the regression for the lock-free read-
// compute-write race that forked the live .fak/loops.jsonl (two events stamped the
// same seq + prev_hash). Many goroutines append distinct events at one ledger at once;
// with the cross-process append lock the result must Load clean under the STRICT reader
// with strictly contiguous seqs 1..N. Pre-fix this fails: Load aborts on a forked seq.
func TestAppendConcurrentDoesNotForkChain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	const n = 40
	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ev := Event{
				LoopID:    "race/loop",
				Kind:      EventFire,
				Source:    "schedule",
				Principal: "timer",
				RunID:     "run-" + strconv.Itoa(i), // distinct so each event hashes uniquely
			}
			_, errs[i] = Append(path, ev)
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("Append #%d: %v", i, err)
		}
	}

	loaded, err := Load(path) // STRICT reader: any fork (dup/missing seq, bad chain) errors here.
	if err != nil {
		t.Fatalf("strict Load of concurrently-appended ledger: %v (chain forked)", err)
	}
	if len(loaded) != n {
		t.Fatalf("loaded %d events, want %d", len(loaded), n)
	}
	for i, ev := range loaded {
		if ev.Seq != uint64(i+1) {
			t.Fatalf("event %d seq = %d, want %d (non-contiguous => fork)", i, ev.Seq, i+1)
		}
	}
}

// TestLoadPrefixRecoversBeforeFork covers the tolerant console reader: a ledger whose
// tail is forked (a duplicate seq, exactly the live failure) must yield the valid
// prefix plus a Broken Integrity pointing at the first bad line — not an error and not
// an empty result. The strict Load still aborts; tolerance lives only in LoadPrefix.
func TestLoadPrefixRecoversBeforeFork(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 3; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", Principal: "p", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed Append #%d: %v", i, err)
		}
	}
	// Forge a forked tail: duplicate the last line verbatim (same seq, same prev_hash).
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
	forked := string(body) + lines[len(lines)-1] + "\n"
	if err := os.WriteFile(path, []byte(forked), 0o644); err != nil {
		t.Fatalf("write forked: %v", err)
	}

	if _, err := Load(path); err == nil {
		t.Fatalf("strict Load accepted a forked ledger; integrity gate must reject it")
	}

	events, integ, err := LoadPrefix(path)
	if err != nil {
		t.Fatalf("LoadPrefix returned error for a chain break (want tolerant): %v", err)
	}
	if !integ.Broken {
		t.Fatalf("Integrity.Broken = false, want true for a forked ledger")
	}
	if len(events) != 3 || integ.Recovered != 3 {
		t.Fatalf("recovered %d events (Integrity.Recovered=%d), want 3 before the fork", len(events), integ.Recovered)
	}
	if integ.AtLine != 4 {
		t.Fatalf("Integrity.AtLine = %d, want 4 (the duplicate tail line)", integ.AtLine)
	}
}

func TestAppendRepairsDuplicateSeqTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 3; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", Principal: "p", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed Append #%d: %v", i, err)
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
	if err := os.WriteFile(path, []byte(string(body)+lines[len(lines)-1]+"\n"), 0o644); err != nil {
		t.Fatalf("write duplicate tail: %v", err)
	}

	appended, err := Append(path, Event{LoopID: "l", Kind: EventAdmit, Source: "s", Principal: "p", RunID: "after-repair"})
	if err != nil {
		t.Fatalf("Append after duplicate tail: %v", err)
	}
	if appended.Seq != 4 {
		t.Fatalf("appended seq = %d, want 4", appended.Seq)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load repaired ledger: %v", err)
	}
	if len(loaded) != 4 {
		t.Fatalf("loaded %d events, want repaired prefix plus append (4)", len(loaded))
	}
	for i, ev := range loaded {
		if ev.Seq != uint64(i+1) {
			t.Fatalf("event %d seq = %d, want %d", i, ev.Seq, i+1)
		}
	}
	if loaded[3].RunID != "after-repair" {
		t.Fatalf("last RunID = %q, want after-repair", loaded[3].RunID)
	}
}

func TestAppendDoesNotRepairTamperedHash(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", Principal: "p", RunID: "before"}); err != nil {
		t.Fatalf("seed Append: %v", err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	tampered := strings.Replace(string(body), `"before"`, `"after"`, 1)
	if err := os.WriteFile(path, []byte(tampered), 0o644); err != nil {
		t.Fatalf("write tampered: %v", err)
	}

	if _, err := Append(path, Event{LoopID: "l", Kind: EventAdmit, Source: "s", Principal: "p", RunID: "should-not-append"}); err == nil || !strings.Contains(err.Error(), "hash") {
		t.Fatalf("Append tampered err = %v, want hash error", err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "hash") {
		t.Fatalf("Load tampered err = %v, want hash error", err)
	}
}

// TestAppendBusyFailsClosed proves the contended-out path never forks: when the append
// lock cannot be taken in time, Append returns ErrLedgerBusy rather than writing an
// unserialized line. Holding the sidecar <path>.lock simulates a stuck peer.
func TestAppendBusyFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", Principal: "p"}); err != nil {
		t.Fatalf("seed Append: %v", err)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open lock: %v", err)
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		t.Fatalf("hold lock: %v", err)
	}
	defer func() { _ = flock.Unlock(lock) }()

	// withLedgerLock polls for appendLockWait (2s) then ErrLedgerBusy; assert the
	// fail-closed contract without waiting the full budget by checking the error type.
	done := make(chan error, 1)
	go func() {
		_, e := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", Principal: "p", RunID: "x"})
		done <- e
	}()
	select {
	case e := <-done:
		if !errors.Is(e, ErrLedgerBusy) {
			t.Fatalf("contended Append err = %v, want ErrLedgerBusy", e)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("contended Append did not return within 10s")
	}

	// The ledger still has exactly the one seeded event — no fork was written.
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load after busy: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("ledger has %d events, want 1 (busy Append must not write)", len(loaded))
	}
}
