package blob

// leak_test.go — regression proof for the content-addressed blob store byte leak.
//
// Found by the agent-orchestrated leak-sweep workflow (NOT the manual pass): the
// process-wide Default CAS grew unbounded — every distinct >InlineMax payload Put / paged
// out minted a permanent map entry with no eviction. ctxmmu/normgate cap their HANDLE
// ledgers, but the BYTES those handles point at live here; the claimed "blob store's own
// policy" did not exist. The fix is a resident-byte budget with FIFO eviction; this test
// pins it via the reusable leakcheck.BoundedSize primitive.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/leakcheck"
)

// distinctPayload returns an n-byte payload unique per i (so content-addressing does NOT
// dedup it) and larger than InlineMax (so it lands in the CAS, not inline on the Ref).
func distinctPayload(i, n int) []byte {
	b := make([]byte, n)
	for j := range b {
		b[j] = byte(i*7 + j*31)
	}
	for k := 0; k < 8 && k < n; k++ {
		b[k] = byte(uint(i) >> (8 * uint(k)))
	}
	return b
}

func TestBlobStoreResidentBytesAreBounded(t *testing.T) {
	ctx := context.Background()
	const payload = 1024               // > InlineMax (256) → stored in the CAS
	const budget = int64(64) * payload // holds ~64 distinct blobs
	const N = 5000
	s := NewWithBudget(budget)

	// Resident bytes must never exceed the budget, no matter how many distinct payloads
	// pass through — pre-fix this grew to N*payload (~5 MB); post-fix it plateaus at budget.
	leakcheck.BoundedSize(t, N, int(budget),
		func(i int) { s.Put(ctx, distinctPayload(i, payload)) },
		func() int { _, b, _ := s.Resident(); return int(b) })

	blobs, bytes, evicted := s.Resident()
	if bytes > budget {
		t.Fatalf("resident bytes %d exceed budget %d (LEAK)", bytes, budget)
	}
	if blobs > int(budget/payload)+1 {
		t.Fatalf("resident blob count %d exceeds budget/payload %d", blobs, budget/payload)
	}
	if evicted < int64(N)-budget/payload-1 {
		t.Fatalf("evicted=%d, expected ≈%d (N - budget capacity)", evicted, int64(N)-budget/payload)
	}

	// Fail-closed degradation: an early (evicted) digest no longer resolves, while a fresh
	// one within budget does — eviction refuses, it never leaks bytes into context.
	agedRef := abi.Ref{Kind: abi.RefBlob, Digest: Digest(distinctPayload(1, payload))}
	if _, err := s.Resolve(ctx, agedRef); err == nil {
		t.Fatalf("an aged-out digest must fail to resolve, but it returned bytes")
	}
	freshRef, _ := s.Put(ctx, distinctPayload(N+1, payload))
	if _, err := s.Resolve(ctx, freshRef); err != nil {
		t.Fatalf("a fresh in-budget blob must resolve: %v", err)
	}
}
