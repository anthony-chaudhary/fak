package preflight

// leak_test.go — regression proof for the hard-negative-ledger memory leak.
//
// Before the fix, Ladder.negatives was an unbounded [][]byte: every malformed /
// schema-failing tool call (the caughtAt path) appended one JSONL row with no cap,
// ring, or drain, and Negatives() returned a copy while leaving the backing slice
// intact. Because the registered rank-10 preflight.Default ladder serves EVERY tool
// call, a long-lived serving process under sustained malformed/adversarial traffic
// — exactly what preflight exists to catch — grew negatives for the life of the
// process. These tests pin the bound (via leakcheck.BoundedSize): the resident
// ledger plateaus at maxNeg, the oldest rows are the ones dropped (FIFO), the
// overflow is fully accounted as evictions, and Negatives() keeps its defensive-copy
// semantics over the resident set.

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/leakcheck"
)

// malformedCall builds a DISTINCT tool call whose inline args are not valid JSON,
// so every Adjudicate hits the rung-0 caughtAt path and mints one hard-negative
// row. The per-i Digest makes each call genuinely distinct (distinct call_hash →
// distinct row bytes), mirroring sustained adversarial traffic rather than one
// replayed call.
func malformedCall(i int) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: "tool",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte("{bad"), Digest: fmt.Sprintf("d%010d", i)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// TestNegativesLedgerIsBounded drives far more distinct malformed calls than the
// ledger cap through Adjudicate and asserts — at EVERY step, via the reusable
// leakcheck.BoundedSize guard — that the resident negative count never exceeds the
// bound. This is the core leak proof: pre-fix the ledger grew to N (the guard would
// fire the moment the resident count first passed cap); post-fix it plateaus at cap
// and the overflow is accounted as evictions.
func TestNegativesLedgerIsBounded(t *testing.T) {
	const cap = 16
	const N = 10000
	l := NewWithLimit(cap)
	ctx := context.Background()

	leakcheck.BoundedSize(t, N, cap,
		func(i int) {
			v := l.Adjudicate(ctx, malformedCall(i))
			if v.Kind != abi.VerdictDeny {
				t.Fatalf("call %d: want VerdictDeny (rung-0 catch), got kind %d", i, v.Kind)
			}
		},
		l.NegativesLen, // size() — must stay ≤ cap across all N ops
	)

	// Steady-state shape: the resident ledger is exactly full, not merely ≤ cap.
	if got := l.NegativesLen(); got != cap {
		t.Fatalf("final resident negatives=%d, want exactly cap %d", got, cap)
	}
	// Negatives() keeps its defensive-copy semantics over the resident set.
	rows := l.Negatives()
	if len(rows) != cap {
		t.Fatalf("Negatives() returned %d rows, want %d (defensive copy of resident set)", len(rows), cap)
	}
	// Everything beyond the cap must have been accounted as an eviction.
	if got, want := l.Evicted(), int64(N-cap); got != want {
		t.Fatalf("Evicted=%d, want %d (N-cap)", got, want)
	}
	// The OLDEST rows are the ones dropped (FIFO): the first resident row is the
	// (N-cap)-th call, not the very first. callHash == "tool:" + the per-i Digest.
	first := decodeRow(t, rows[0])
	if want := fmt.Sprintf("tool:d%010d", N-cap); first.CallHash != want {
		t.Fatalf("oldest resident row call_hash=%q, want %q (oldest cap rows should be evicted, freshest retained)", first.CallHash, want)
	}
	last := decodeRow(t, rows[len(rows)-1])
	if want := fmt.Sprintf("tool:d%010d", N-1); last.CallHash != want {
		t.Fatalf("newest resident row call_hash=%q, want %q (the most recent call must be retained)", last.CallHash, want)
	}

	// caught/total are lifetime catch-rate counters and are intentionally NOT
	// bounded — only the resident ledger is. Every call was caught.
	caught, total, _ := l.CatchRate()
	if caught != int64(N) || total != int64(N) {
		t.Fatalf("CatchRate caught/total=%d/%d, want %d/%d (every malformed call caught and counted)", caught, total, N, N)
	}
}

// TestNegativesBackingArrayCompacts proves the FIFO bound does not silently leak via
// the backing array: after streaming many multiples of the cap, the negatives slice's
// length stays ≈ the resident set (bounded), not ≈ N. The eviction path nils dropped
// slots and compacts the consumed prefix, so cap(negatives) stays O(maxNeg).
func TestNegativesBackingArrayCompacts(t *testing.T) {
	const cap = 8
	const N = 5000
	l := NewWithLimit(cap)
	ctx := context.Background()

	for i := 0; i < N; i++ {
		l.Adjudicate(ctx, malformedCall(i))
	}

	// White-box: the backing slice's len/cap must stay bounded by ≈2·maxNeg (the
	// compaction threshold), never grow toward N — that array IS the leak surface.
	l.mu.RLock()
	sliceLen, sliceCap := len(l.negatives), cap2(l.negatives)
	l.mu.RUnlock()
	if sliceLen > 2*cap {
		t.Fatalf("negatives slice len=%d exceeds 2*cap=%d after %d ops (backing array leak)", sliceLen, 2*cap, N)
	}
	if sliceCap > 4*cap {
		t.Fatalf("negatives backing cap=%d exceeds 4*cap=%d after %d ops (backing array leak)", sliceCap, 4*cap, N)
	}
}

// cap2 is a tiny wrapper so the builtin cap() is reachable in a scope that shadows
// it with a `cap` constant above (the reference ctxmmu/normgate tests use the same
// `const cap` idiom).
func cap2(s [][]byte) int { return cap(s) }
