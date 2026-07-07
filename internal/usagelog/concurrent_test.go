package usagelog

// concurrent_test.go — the #2608 repro + guard. The July 4 overnight fleet run
// surfaced `fak audit usage` reporting CHAIN_BROKEN with
// `usagelog: sequence gap: seq=738 want 739`: two concurrent fak invocations both
// recovered the chain head at Open time and both stamped the same next seq off
// it, forking the chain. TestConcurrentInvocationsShareOneChain reproduces that
// exact interleaving (and FAILS on the pre-#2608 writer); the tamper tests pin
// that the fix did NOT weaken Verify — a genuinely gapped or forked journal
// still surfaces as broken instead of being silently dropped.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// TestConcurrentInvocationsShareOneChain is the #2608 repro: two fak processes
// race — both Open the journal (recovering the SAME head) before either appends.
// The pre-fix writer stamped seq off the head cached at Open time, so both rows
// landed as seq=1 and Verify reported `sequence gap: seq=1 want 2`. The fixed
// writer re-recovers the head inside Append's cross-process critical section, so
// the second append chains onto the first.
func TestConcurrentInvocationsShareOneChain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")

	// Interleaving: A opens, B opens (same recovered head), A appends, B appends.
	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open a: %v", err)
	}
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open b: %v", err)
	}
	rowA, err := a.Append(Row{Verb: "guard"})
	if err != nil {
		t.Fatalf("Append a: %v", err)
	}
	rowB, err := b.Append(Row{Verb: "run"})
	if err != nil {
		t.Fatalf("Append b: %v", err)
	}
	_ = a.Close()
	_ = b.Close()

	n, err := Verify(path)
	if err != nil {
		t.Fatalf("Verify after two concurrent invocations: %v (the pre-#2608 writer forked the chain here)", err)
	}
	if n != 2 {
		t.Fatalf("Verify counted %d rows, want 2", n)
	}
	if rowA.Seq != 1 || rowB.Seq != 2 {
		t.Errorf("seqs = %d,%d, want 1,2 (one chain, no fork)", rowA.Seq, rowB.Seq)
	}
	if rowB.PrevHash != rowA.Hash {
		t.Errorf("second invocation not chained onto the first: prev=%q want %q", rowB.PrevHash, rowA.Hash)
	}
}

// TestParallelAppendsNeverGap hammers the cross-process critical section: several
// loggers (each standing in for a concurrent fak process, all opened before any
// append so every cached head starts stale) append in parallel goroutines. The
// resulting journal must be ONE contiguous chain: Verify passes and seq runs
// 1..N with no gap and no duplicate.
func TestParallelAppendsNeverGap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	const loggers, perLogger = 4, 3

	lgs := make([]*Logger, loggers)
	for i := range lgs {
		lg, err := Open(path)
		if err != nil {
			t.Fatalf("Open %d: %v", i, err)
		}
		lgs[i] = lg
	}
	var wg sync.WaitGroup
	errs := make(chan error, loggers*perLogger)
	for i, lg := range lgs {
		wg.Add(1)
		go func(i int, lg *Logger) {
			defer wg.Done()
			for j := 0; j < perLogger; j++ {
				if _, err := lg.Append(Row{Verb: fmt.Sprintf("verb-%d", i), Argc: j}); err != nil {
					errs <- fmt.Errorf("logger %d append %d: %w", i, j, err)
				}
			}
		}(i, lg)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	for _, lg := range lgs {
		_ = lg.Close()
	}

	n, err := Verify(path)
	if err != nil {
		t.Fatalf("Verify after parallel appends: %v", err)
	}
	if n != loggers*perLogger {
		t.Fatalf("Verify counted %d rows, want %d", n, loggers*perLogger)
	}
	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	for i, r := range rows {
		if r.Seq != uint64(i+1) {
			t.Fatalf("row %d: seq=%d, want %d (contiguous, no gap/duplicate)", i, r.Seq, i+1)
		}
	}
}

// TestVerifyStillCatchesDeletedRow pins the don't-mask-corruption acceptance: a
// journal with a row genuinely REMOVED (real tampering, not a concurrency
// artifact) must still fail Verify with the sequence-gap finding.
func TestVerifyStillCatchesDeletedRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := lg.Append(Row{Verb: "run"}); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	_ = lg.Close()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(lines))
	}
	// Delete the middle row.
	tampered := lines[0] + "\n" + lines[2] + "\n"
	if err := os.WriteFile(path, []byte(tampered), 0o600); err != nil {
		t.Fatalf("write tampered: %v", err)
	}

	_, err = Verify(path)
	if err == nil {
		t.Fatal("Verify accepted a journal with a deleted row; want a sequence-gap error")
	}
	if !strings.Contains(err.Error(), "sequence gap") {
		t.Fatalf("Verify error = %v, want a sequence-gap finding", err)
	}
}

// TestVerifyStillReportsLegacyFork is the durable fixture of the witnessed #2608
// failure shape: a journal written by the PRE-fix racing writer, where two rows
// carry the same seq chained to the same parent. Such a journal is genuinely
// non-linear on disk and must KEEP surfacing as broken (`sequence gap: seq=N
// want N+1`, the exact seq=738-want-739 shape) — the fix prevents new forks, it
// does not silently launder historic ones.
func TestVerifyStillReportsLegacyFork(t *testing.T) {
	r1 := Row{Schema: SchemaV1, Seq: 1, TSUnixNano: 1, Verb: "guard", PID: 100}
	r1.Hash = chainHash("", r1)
	// Two pre-fix writers both recovered head (seq=1, r1.Hash) and appended seq=2.
	r2a := Row{Schema: SchemaV1, Seq: 2, TSUnixNano: 2, Verb: "run", PID: 101, PrevHash: r1.Hash}
	r2a.Hash = chainHash(r2a.PrevHash, r2a)
	r2b := Row{Schema: SchemaV1, Seq: 2, TSUnixNano: 3, Verb: "usage", PID: 102, PrevHash: r1.Hash}
	r2b.Hash = chainHash(r2b.PrevHash, r2b)

	path := filepath.Join(t.TempDir(), "usage.jsonl")
	var sb strings.Builder
	for _, r := range []Row{r1, r2a, r2b} {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		sb.Write(b)
		sb.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	_, err := Verify(path)
	if err == nil {
		t.Fatal("Verify accepted a forked journal; want the sequence-gap finding")
	}
	if !strings.Contains(err.Error(), "sequence gap: seq=2 want 3") {
		t.Fatalf("Verify error = %v, want `sequence gap: seq=2 want 3`", err)
	}
}

// TestStaleLoggerRefreshesAcrossAppends pins the head refresh beyond the first
// append: a logger that appended once must STILL pick up rows a peer landed
// afterwards, not just at Open time.
func TestStaleLoggerRefreshesAcrossAppends(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open a: %v", err)
	}
	if _, err := a.Append(Row{Verb: "guard"}); err != nil {
		t.Fatalf("Append a1: %v", err)
	}

	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open b: %v", err)
	}
	rowB, err := b.Append(Row{Verb: "run"})
	if err != nil {
		t.Fatalf("Append b: %v", err)
	}
	_ = b.Close()

	rowA2, err := a.Append(Row{Verb: "guard"})
	if err != nil {
		t.Fatalf("Append a2: %v", err)
	}
	_ = a.Close()

	if rowA2.Seq != 3 || rowA2.PrevHash != rowB.Hash {
		t.Errorf("stale logger did not refresh: seq=%d prev=%q, want seq=3 prev=%q", rowA2.Seq, rowA2.PrevHash, rowB.Hash)
	}
	if n, err := Verify(path); err != nil || n != 3 {
		t.Errorf("Verify: n=%d err=%v, want n=3 err=nil", n, err)
	}
}
