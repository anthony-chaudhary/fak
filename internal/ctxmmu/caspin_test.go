package ctxmmu_test

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestQuarantinePageInSurvivesCASEviction proves the held-quarantine CAS pin
// end-to-end: a sealed result paged out to the BOUNDED global CAS still pages back
// in (after a witness Clear) even after CAS churn that evicts unpinned blobs. The
// gated page-in is not broken by the byte bound — the soundness property the pin
// exists to protect.
func TestQuarantinePageInSurvivesCASEviction(t *testing.T) {
	ctx := context.Background()
	old := blob.Default.MaxBytes()
	blob.Default.SetMaxBytes(8192) // tight enough that churn forces eviction
	defer blob.Default.SetMaxBytes(old)

	m := ctxmmu.New()

	// A >256B secret-shaped body: quarantines AND pages out to the CAS (not inline).
	secret := []byte("sk-" + strings.Repeat("a", 400))
	c := call("read_secret")
	r := result(c, secret)
	if v := m.Admit(ctx, c, r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("want Quarantine, got %v", v.Kind)
	}
	id := r.Meta["quarantine_id"]
	if id == "" {
		t.Fatalf("no quarantine_id stamped")
	}

	// Control: an UNPINNED CAS blob put now should be evicted by the churn below.
	ctrl, err := blob.Default.Put(ctx, bytes.Repeat([]byte{0xAB}, 1000))
	if err != nil {
		t.Fatalf("put control: %v", err)
	}
	for i := 0; i < 64; i++ { // distinct >256B blobs drive the bound past capacity
		if _, err := blob.Default.Put(ctx, bytes.Repeat([]byte{byte(i) + 1}, 1024)); err != nil {
			t.Fatalf("churn put: %v", err)
		}
	}
	if _, err := blob.Default.Resolve(ctx, ctrl); err == nil {
		t.Fatalf("control (unpinned) blob should have been evicted by the bound")
	}

	// The quarantine bytes are PINNED, so the gated page-in still resolves them.
	m.Clear(id)
	got, err := m.PageIn(ctx, id)
	if err != nil {
		t.Fatalf("pinned quarantine page-in failed under CAS eviction (soundness break): %v", err)
	}
	if !bytes.Equal(got, secret) {
		t.Fatalf("page-in returned wrong bytes")
	}
}
