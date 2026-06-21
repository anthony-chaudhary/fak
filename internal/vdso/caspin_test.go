package vdso

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// TestTier2HitSurvivesCASEviction is the end-to-end soundness proof for the vDSO
// CAS pin: a tier-2 entry whose payload lives in the BOUNDED global CAS stays
// resolvable on a later HIT even after heavy CAS churn that would otherwise have
// evicted it. Without the pin, the hit would return a Ref the store can no longer
// resolve — violating "a cache hit equals a fresh call". With the pin, it holds.
func TestTier2HitSurvivesCASEviction(t *testing.T) {
	ctx := context.Background()
	old := blob.Default.MaxBytes()
	blob.Default.SetMaxBytes(8192)
	defer blob.Default.SetMaxBytes(old)

	v := New(DefaultCacheSize)

	// A >256B payload so the result Ref is CAS-backed (RefBlob), not inline.
	payload := []byte("RESULT:" + strings.Repeat("x", 500))
	pref, err := blob.Default.Put(ctx, payload)
	if err != nil {
		t.Fatalf("put payload: %v", err)
	}
	if pref.Kind != abi.RefBlob {
		t.Fatalf("precondition: payload should be CAS-backed, got kind %d", pref.Kind)
	}

	c := roCall("get_thing", `{"q":"thing"}`)
	v.Emit(abi.Event{Kind: abi.EvComplete, Call: c, Result: &abi.Result{Call: c, Status: abi.StatusOK, Payload: pref}})

	// Control: an unpinned CAS blob put now should be evicted by the churn.
	ctrl, err := blob.Default.Put(ctx, []byte("CTRL:"+strings.Repeat("c", 1000)))
	if err != nil {
		t.Fatalf("put control: %v", err)
	}
	for i := 0; i < 64; i++ { // distinct >256B blobs drive the bound past capacity
		if _, err := blob.Default.Put(ctx, []byte("churn:"+strconv.Itoa(i)+":"+strings.Repeat("z", 1000))); err != nil {
			t.Fatalf("churn put: %v", err)
		}
	}
	if _, err := blob.Default.Resolve(ctx, ctrl); err == nil {
		t.Fatalf("control (unpinned) blob should have been evicted")
	}

	// The tier-2 HIT returns the stored Ref; resolving it MUST still work (pinned).
	res, hit := v.Lookup(ctx, c)
	if !hit {
		t.Fatalf("expected a tier-2 cache hit")
	}
	got := resolveBytes(t, res.Payload)
	if string(got) != string(payload) {
		t.Fatalf("cache hit resolved wrong/empty bytes under eviction (soundness break): got %d bytes", len(got))
	}
}
