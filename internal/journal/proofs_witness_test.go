package journal

// Witness test for the OPEN proof obligation:
//
//	[per-write-durable-flush] For a file-backed journal, each Emit that produces
//	an audit Row flushes that row's bytes to the OS file before returning. So a
//	process crash (not power loss) after Emit returns loses no row already
//	committed: Verify(path) called WITHOUT any intervening Close()/Flush()
//	recovers every emitted row.
//
// Mechanism under test: Emit -> append (journal.go:131-168) -> writeRow
// (journal.go:296-308), whose final statement is bw.Flush() — buffered bytes are
// pushed to the OS file on every row, not held until Close/Flush.
//
// Proof discipline: fak/docs/proofs/00-METHOD.md. This is a metamorphic/observ-
// ability test that simulates a process crash by reading the on-disk bytes of a
// STILL-OPEN journal from a SEPARATE file handle (Verify opens the path afresh),
// without ever calling the journal's own Close()/Flush(). It is NON-VACUOUS: if
// append did not flush per row (e.g. the bw.Flush() in writeRow were removed),
// the buffered rows would not yet be on disk and the post-Emit Verify count would
// be strictly less than the number emitted, failing the exact-equality assertion.
//
// Determinism: a fixed injected clock and a fixed sequence of emitted events; no
// randomness, no timing, no goroutines. Verify reads bytes the OS already holds.

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// emitN emits k deterministic DENY rows on j (no Flush/Close between them).
func emitN(j *Journal, k int) {
	for i := 0; i < k; i++ {
		j.Emit(testDenyEvent("send_email", "trace", `{"to":"x@y.com"}`))
	}
}

// TestPerWriteDurableFlush_VerifyWithoutCloseRecoversEveryEmittedRow is the core
// witness: after EACH Emit returns, Verify(path) — which opens the file fresh,
// exactly as a crash-recovery auditor would — already sees that row, with NO
// intervening Close()/Flush() on the producing journal. The relation asserted is
// the exact equality  Verify(path).n == (number of Emits so far)  at every step.
func TestPerWriteDurableFlush_VerifyWithoutCloseRecoversEveryEmittedRow(t *testing.T) {
	path := t.TempDir() + "/audit.jsonl"
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Fixed clock => deterministic timestamps => deterministic chain hashes.
	j.clock = func() time.Time { return time.Unix(7, 11) }
	// IMPORTANT: we deliberately never call j.Flush() or j.Close() before Verify.
	// The whole point is that Emit alone makes the bytes durable on the OS file.

	const total = 64
	for emitted := 1; emitted <= total; emitted++ {
		j.Emit(testDenyEvent("send_email", "trace", `{"to":"x@y.com"}`))

		// Simulate "process crashes here, after Emit returned" by opening the
		// path from scratch and verifying the chain — no Close/Flush happened.
		n, verr := Verify(path)
		if verr != nil {
			t.Fatalf("after %d Emit(s) WITHOUT Close/Flush, Verify errored: %v "+
				"(a per-row flush is missing — rows are stranded in the buffer)", emitted, verr)
		}
		if n != emitted {
			t.Fatalf("after %d Emit(s) WITHOUT Close/Flush, Verify recovered %d rows; "+
				"want %d — each Emit must flush its row's bytes to the OS file before returning",
				emitted, n, emitted)
		}
	}

	// Sanity: a final Close must not change the count (everything was already
	// durable). This also guards against the test accidentally being vacuous by
	// having written nothing.
	if err := j.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if n, verr := Verify(path); verr != nil || n != total {
		t.Fatalf("post-Close Verify = n=%d err=%v, want %d nil", n, verr, total)
	}
}

// TestPerWriteDurableFlush_StatsMatchDurableRows pins the companion invariant:
// the journal's live head Seq counter equals the number of rows an independent
// crash-recovery Verify finds on disk — i.e. nothing the producer counts as
// committed is missing from the durable file (no buffered-but-lost rows), again
// with NO Close()/Flush() between producing and reading.
func TestPerWriteDurableFlush_StatsMatchDurableRows(t *testing.T) {
	path := t.TempDir() + "/audit.jsonl"
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Release the append handle at function exit (after the no-Close assertions
	// below) so t.TempDir cleanup can unlink audit.jsonl on Windows, where an open
	// handle blocks RemoveAll. The durability-without-Close property is still proven:
	// Stats + Verify run before this deferred Close.
	defer j.Close()
	j.clock = func() time.Time { return time.Unix(99, 0) }

	const k = 17
	emitN(j, k)
	// No Flush/Close.

	headSeq, _, _ := j.Stats()
	if headSeq != k {
		t.Fatalf("Stats head seq = %d, want %d", headSeq, k)
	}
	n, verr := Verify(path)
	if verr != nil {
		t.Fatalf("Verify WITHOUT Close/Flush errored: %v", verr)
	}
	if uint64(n) != headSeq {
		t.Fatalf("durable rows on disk = %d but producer head seq = %d; "+
			"a committed row is not yet flushed (per-write durability violated)", n, headSeq)
	}
}

// guard: keep the abi import load-bearing even if testDenyEvent's signature
// changes; this also documents the kinds that produce an audit Row.
var _ = abi.EvDeny
