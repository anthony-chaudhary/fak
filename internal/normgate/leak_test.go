package normgate_test

// leak_test.go — regression proof for the normgate quarantine-ledger memory leak.
//
// This leak was found NOT by the manual sweep but by the repeatable static scanner
// (tools/leak_scan.py), which flagged `Gate.held` as a runtime-written map with no
// delete and no cap — the same unbounded-ledger shape as ctxmmu's twin, in a gate that
// registers EARLIER on the live path (rank 5, before ctxmmu's rank 10). Pre-fix, every
// caught threat minted a permanent held["ng-q<n>"] entry. The fix bounds it FIFO; this
// test pins the bound: held plateaus at maxHeld and Evicted accounts for the overflow.

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/normgate"
)

// secretBody returns a body canon.Scan flags as a secret (distinct per i → distinct id).
func secretBody(i int) string {
	return fmt.Sprintf(`token := "sk-abcdef0123456789abcdef%010d"`, i)
}

func TestNormgateHeldLedgerIsBounded(t *testing.T) {
	ctx := context.Background()
	const cap = 16
	const N = 5000
	g := normgate.NewWithLimit(cap)

	for i := 0; i < N; i++ {
		c := untrusted("read_file")
		r := result(secretBody(i))
		g.Admit(ctx, c, r)
		if hl := g.HeldLen(); hl > cap {
			t.Fatalf("after %d admits, HeldLen=%d exceeds cap %d (LEAK)", i+1, hl, cap)
		}
	}

	if hl := g.HeldLen(); hl != cap {
		t.Fatalf("final HeldLen=%d, want exactly cap %d", hl, cap)
	}
	if got, want := g.Evicted(), int64(N-cap); got != want {
		t.Fatalf("Evicted=%d, want %d (N-cap)", got, want)
	}
	if _, q, _ := g.Stats(); q != int64(N) {
		t.Fatalf("quarantine tally=%d, want %d (every secret quarantined)", q, N)
	}
}
