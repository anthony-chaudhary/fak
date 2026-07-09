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

// TestAppendFastPathWritesAndUsesTailSidecar proves the O(1) tail cache: a normal
// append writes a <path>.tail sidecar recording the last seq/hash and the exact file
// size, and the next append derives its seq/prev_hash from that sidecar (the chain
// stays contiguous under the strict reader) rather than re-scanning the whole file.
func TestAppendFastPathWritesAndUsesTailSidecar(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	first, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: "0"})
	if err != nil {
		t.Fatalf("Append #1: %v", err)
	}

	tc, ok := readTailCache(path)
	if !ok {
		t.Fatalf("tail sidecar missing after append")
	}
	if tc.Seq != 1 || tc.Hash != first.Hash {
		t.Fatalf("sidecar = %+v, want seq 1 hash %s", tc, first.Hash)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if tc.Size != fi.Size() {
		t.Fatalf("sidecar size %d != file size %d", tc.Size, fi.Size())
	}

	second, err := Append(path, Event{LoopID: "l", Kind: EventAdmit, Source: "s", RunID: "1"})
	if err != nil {
		t.Fatalf("Append #2: %v", err)
	}
	if second.Seq != 2 || second.PrevHash != first.Hash {
		t.Fatalf("second = seq %d prev %q, want seq 2 prev %q", second.Seq, second.PrevHash, first.Hash)
	}
	if _, err := Load(path); err != nil {
		t.Fatalf("Load after fast-path appends: %v", err)
	}
}

// TestAppendTailPathSkipsFullChainVerify pins the deliberate trade the fast path
// makes for O(1): Append verifies the tip it extends (size + last-line re-hash) but
// does NOT re-verify the whole chain, so a same-size corruption of an EARLIER line
// (which leaves the file size and the intact final line — and thus both fast-path
// guards — untouched) does not block the append. The corruption is still caught on
// the read side by the strict Load reader, which is where full-chain verification
// now lives (issue #3462). Its tail-line sibling is caught at append time by
// TestAppendTailCorruptionRoutesToSlowPath.
func TestAppendTailPathSkipsFullChainVerify(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 3; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed #%d: %v", i, err)
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	// Same-length edit of line 1's payload => file size (and the size guard) unchanged.
	corrupt := strings.Replace(string(body), `"run_id":"0"`, `"run_id":"Z"`, 1)
	if len(corrupt) != len(body) {
		t.Fatalf("corruption changed length %d->%d; test needs a same-size edit", len(body), len(corrupt))
	}
	if corrupt == string(body) {
		t.Fatalf("corruption was a no-op; run_id token not found")
	}
	if err := os.WriteFile(path, []byte(corrupt), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	appended, err := Append(path, Event{LoopID: "l", Kind: EventEnd, Source: "s", RunID: "after"})
	if err != nil {
		t.Fatalf("fast-path Append over same-size early corruption: %v", err)
	}
	if appended.Seq != 4 {
		t.Fatalf("appended seq = %d, want 4", appended.Seq)
	}

	// The corruption is still real: the strict reader must reject the chain.
	if _, lerr := Load(path); lerr == nil {
		t.Fatalf("Load accepted a corrupted chain; read-side verify must reject it")
	}
}

// TestAppendTailCorruptionRoutesToSlowPath is the regression guard for the O(1) fast
// path's silent-fork hole (issue #3462 review follow-up): a SAME-SIZE corruption of the
// FINAL line leaves the file size — and thus the sidecar's size guard — intact, so only
// fastTail's re-hash of the last line can notice it. This models the crash-durability
// case where the file size and sidecar persist but the tail data bytes are lost/forged
// (NTFS ValidDataLength / ext4 writeback) as well as plain tail bit-rot. Trusting size
// alone, Append would chain the next event onto a tip that no longer hashes to the
// recorded hash — a permanent fork. It must instead route to the slow path and fail
// closed on the tail hash break, never silently extend the fork.
func TestAppendTailCorruptionRoutesToSlowPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < 3; i++ {
		if _, err := Append(path, Event{LoopID: "l", Kind: EventFire, Source: "s", RunID: strconv.Itoa(i)}); err != nil {
			t.Fatalf("seed #%d: %v", i, err)
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	// Same-length edit of the LAST line's payload (run_id "2" -> "Z", unique to line 3):
	// file size (and the size guard) unchanged, but the tail no longer hashes to the
	// sidecar. The stored hash field is left untouched, so only recomputing hashEvent
	// over the line catches it — the exact check the fast path must perform.
	corrupt := strings.Replace(string(body), `"run_id":"2"`, `"run_id":"Z"`, 1)
	if len(corrupt) != len(body) {
		t.Fatalf("corruption changed length %d->%d; test needs a same-size edit", len(body), len(corrupt))
	}
	if corrupt == string(body) {
		t.Fatalf("corruption was a no-op; tail run_id token not found")
	}
	if err := os.WriteFile(path, []byte(corrupt), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	// The sidecar still records the pre-corruption size, so the size guard passes; only
	// the last-line re-hash can catch the corrupted tip. Append must fail closed here.
	if _, err := Append(path, Event{LoopID: "l", Kind: EventEnd, Source: "s", RunID: "after"}); err == nil {
		t.Fatal("Append trusted a same-size-corrupted tail (silent fork); want a fail-closed integrity error")
	} else if !strings.Contains(err.Error(), "hash") {
		t.Fatalf("Append over corrupted tail err = %v, want a tail hash break", err)
	}

	// And no forked 4th line was written: the ledger still holds exactly the 3 seeded
	// (now-corrupt) lines, so the strict reader rejects it rather than a longer fork.
	if _, lerr := Load(path); lerr == nil {
		t.Fatal("Load accepted the corrupted tail; the break must remain visible")
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

// BenchmarkAppendTailLatency is the O(1)-amortized proof for issue #3462: it seeds
// ledgers of increasing size (untimed) and benchmarks single appends onto each. With
// the tail sidecar every append derives seq/prev_hash from an O(1) size-checked read
// instead of re-parsing + hash-verifying the whole file, so ns/op is FLAT across the
// seed sizes. Pre-fix (LoadPrefix per append) ns/op grew ~linearly with the seed —
// run this against the two implementations to see the old cost climb and the new
// cost hold. (Benchmarks are not run by the default `go test`/CI, so the untimed
// seeding cost never lands on the CI clock.)
func BenchmarkAppendTailLatency(b *testing.B) {
	for _, seed := range []int{100, 2000, 20000} {
		b.Run("seed="+strconv.Itoa(seed), func(b *testing.B) {
			path := filepath.Join(b.TempDir(), "loops.jsonl")
			for i := 0; i < seed; i++ {
				if _, err := Append(path, Event{LoopID: "bench/loop", Kind: EventHeartbeat, Source: "s", RunID: strconv.Itoa(i)}); err != nil {
					b.Fatalf("seed append #%d: %v", i, err)
				}
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := Append(path, Event{LoopID: "bench/loop", Kind: EventHeartbeat, Source: "s", RunID: "b" + strconv.Itoa(i)}); err != nil {
					b.Fatalf("bench append: %v", err)
				}
			}
		})
	}
}
