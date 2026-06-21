package ctxmmu_test

// leak_test.go — regression proof for the quarantine-ledger memory leak.
//
// Before the fix, MMU.held and MMU.cleared were unbounded maps: every quarantined
// result minted a permanent held["q<n>"] entry (monotonic id, no delete/TTL/eviction
// anywhere), so a long-lived gate (ctxmmu.Default is the registered rank-10
// ResultAdmitter on the live serving path) grew without bound under a poison-heavy
// stream. These tests pin the bound: held plateaus at maxHeld, cleared stays ⊆ held,
// the freshest quarantines remain page-in-able, and the eviction counter accounts
// for everything dropped.

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// poison returns a body that trips ScreenBytes (a secret), with a unique tail so
// every result is a distinct payload (distinct CAS digest, distinct quarantine id).
func poison(i int) []byte {
	return []byte(fmt.Sprintf("api_key=sk-abcdef0123456789abcdef%010d leaked", i))
}

// TestQuarantineLedgerIsBounded streams far more poison results than the ledger cap
// through Admit and asserts the held/cleared maps never exceed the bound — the core
// leak proof. Pre-fix this loop grew held to N (10000); post-fix it plateaus at cap.
func TestQuarantineLedgerIsBounded(t *testing.T) {
	ctx := context.Background()
	const cap = 16
	const N = 10000
	m := ctxmmu.NewWithLimit(cap)

	for i := 0; i < N; i++ {
		c := call("read_file")
		r := result(c, poison(i))
		v := m.Admit(ctx, c, r)
		if v.Kind != abi.VerdictQuarantine {
			t.Fatalf("result %d: want VerdictQuarantine, got %v", i, v.Kind)
		}
		// Clear a fraction of them to also exercise the cleared map's growth.
		if i%3 == 0 {
			m.Clear(r.Meta["quarantine_id"])
		}
		if hl := m.HeldLen(); hl > cap {
			t.Fatalf("after %d admits, HeldLen=%d exceeds cap %d (LEAK)", i+1, hl, cap)
		}
	}

	if hl := m.HeldLen(); hl != cap {
		t.Fatalf("final HeldLen=%d, want exactly cap %d", hl, cap)
	}
	// cleared must be a subset of held, hence also bounded by cap.
	cleared := m.Cleared()
	if len(cleared) > cap {
		t.Fatalf("cleared map len=%d exceeds cap %d (LEAK)", len(cleared), cap)
	}
	held := m.Held()
	for id := range cleared {
		if _, ok := held[id]; !ok {
			t.Fatalf("cleared id %q is not in held — cleared is not ⊆ held", id)
		}
	}
	// Everything beyond the cap must have been accounted as an eviction.
	if got, want := m.Evicted(), int64(N-cap); got != want {
		t.Fatalf("Evicted=%d, want %d (N-cap)", got, want)
	}
}

// TestEvictedQuarantineRefusesPageIn proves the eviction degrades fail-closed: an
// aged-out id can no longer be paged in (it refuses, never leaks bytes), while a
// fresh, cleared id within the cap still pages its bytes back.
func TestEvictedQuarantineRefusesPageIn(t *testing.T) {
	ctx := context.Background()
	const cap = 8
	m := ctxmmu.NewWithLimit(cap)

	// First quarantine: capture its id, clear it (so it WOULD page in if retained).
	c0 := call("read_file")
	r0 := result(c0, poison(0))
	if v := m.Admit(ctx, c0, r0); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("first admit: want Quarantine, got %v", v.Kind)
	}
	oldID := r0.Meta["quarantine_id"]
	m.Clear(oldID)
	if _, err := m.PageIn(ctx, oldID); err != nil {
		t.Fatalf("fresh cleared id should page in: %v", err)
	}

	// Push cap+ more distinct quarantines to age the first one out.
	var freshID string
	var freshOrig []byte
	for i := 1; i <= cap+4; i++ {
		c := call("read_file")
		body := poison(i)
		r := result(c, append([]byte(nil), body...))
		if v := m.Admit(ctx, c, r); v.Kind != abi.VerdictQuarantine {
			t.Fatalf("admit %d: want Quarantine, got %v", i, v.Kind)
		}
		freshID = r.Meta["quarantine_id"]
		freshOrig = body
	}

	// The aged-out id must now refuse (its handle was dropped) — even though it was
	// cleared. This is the fail-closed degradation, not a bytes-into-context leak.
	if _, err := m.PageIn(ctx, oldID); err == nil {
		t.Fatalf("evicted id %q must refuse PageIn, but it returned bytes", oldID)
	}
	// A re-Clear of the aged-out id is a no-op and must not resurrect it.
	m.Clear(oldID)
	if _, err := m.PageIn(ctx, oldID); err == nil {
		t.Fatalf("re-cleared evicted id %q must still refuse PageIn", oldID)
	}

	// The freshest id is still within the cap; after a witness clear it pages in.
	m.Clear(freshID)
	got, err := m.PageIn(ctx, freshID)
	if err != nil {
		t.Fatalf("fresh in-cap id should page in after clear: %v", err)
	}
	if string(got) != string(freshOrig) {
		t.Fatalf("fresh page-in bytes mismatch:\n want %q\n got  %q", freshOrig, got)
	}
}
