package ctxplan

import (
	"context"
	"testing"
)

// TestStoreAuditPartitionsAndFaithful is the store-level faithfulness witness (issue #565):
// over a noisy store where the index prunes most spans, resident ∪ elided ∪ pruned partitions
// the WHOLE store, every pruned span is recoverable, and the view is store-faithful — so
// index pruning provably left the lossless store intact.
func TestStoreAuditPartitionsAndFaithful(t *testing.T) {
	ctx := context.Background()
	st := goodPlusNoiseStore(300)
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)
	f := Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0", "span:1"}}
	plan := ix.PlanCells(f, Budget{Tokens: 48}, nil, ProbeOptions{})

	w := StoreAudit(spans, plan)
	if !w.Partition {
		t.Fatalf("resident+elided+pruned did not partition the store: %+v", w)
	}
	if w.Resident+w.Elided+w.Pruned != w.StoreSpans {
		t.Fatalf("partition counts do not sum to |S|: %d+%d+%d != %d", w.Resident, w.Elided, w.Pruned, w.StoreSpans)
	}
	if w.Pruned == 0 {
		t.Fatal("the index should have PRUNED the noise; a zero-pruned store-audit is vacuous here")
	}
	if w.Recoverable != w.Pruned || len(w.Unrecoverable) != 0 {
		t.Errorf("every pruned span must be recoverable: recoverable=%d pruned=%d unrecoverable=%v", w.Recoverable, w.Pruned, w.Unrecoverable)
	}
	if len(w.Foreign) != 0 {
		t.Errorf("no plan id should be foreign to the store, got %v", w.Foreign)
	}
	if !w.Faithful {
		t.Errorf("the index-bounded view must be store-faithful: %+v", w)
	}
}

// TestStoreAuditDetectsCompaction proves the contrast Audit draws is lifted to store scope: a
// faithful index-bounded plan is store-faithful, but its CompactionView twin (elided spans
// stripped of their page-back-in handles — lossy compaction) is store-UNFAITHFUL at identical
// residency. Faithful vs lossy is checkable, not a slogan.
func TestStoreAuditDetectsCompaction(t *testing.T) {
	ctx := context.Background()
	st := goodPlusNoiseStore(120)
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)
	f := Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0", "span:1"}}
	plan := ix.PlanCells(f, Budget{Tokens: 40}, nil, ProbeOptions{})

	if w := StoreAudit(spans, plan); !w.Faithful {
		t.Fatalf("the planned view should be store-faithful: %+v", w)
	}
	if w := StoreAudit(spans, CompactionView(plan)); w.Faithful {
		t.Errorf("a compaction view (elided handles stripped) must be store-UNFAITHFUL: %+v", w)
	}
}

// TestStoreAuditDetectsForeignSpan proves the witness refuses to certify a plan built over a
// different/stale store: a plan whose Selected span names no store span is reported Foreign
// and fails the partition — the view cannot be reconciled with the store audited.
func TestStoreAuditDetectsForeignSpan(t *testing.T) {
	spans := []Span{
		{ID: "real0", Step: 0, Role: "user", Descriptor: "a", Digest: "d0", Bytes: 4},
		{ID: "real1", Step: 1, Role: "user", Descriptor: "b", Digest: "d1", Bytes: 4},
	}
	// A plan that selects a span absent from the store (a stale/foreign plan).
	plan := Plan{
		Objective:  ObjGreedy,
		Candidates: 1,
		Selected:   []Selection{{ID: "ghost", Step: 9}},
	}
	w := StoreAudit(spans, plan)
	if len(w.Foreign) == 0 || w.Foreign[0] != "ghost" {
		t.Fatalf("a plan id absent from the store must be reported Foreign, got %v", w.Foreign)
	}
	if w.Partition || w.Faithful {
		t.Errorf("a foreign plan id must fail the partition and faithfulness: %+v", w)
	}
}

// TestStoreAuditAllResidentNoPruned is the conservation control: a plan that selects every
// store span (small store, ample budget, no index pruning) has zero pruned and zero elided —
// resident == the whole store — and is faithful. Nothing pruned, nothing lost.
func TestStoreAuditAllResidentNoPruned(t *testing.T) {
	ctx := context.Background()
	st := NewMemStore()
	st.Add("user", DurabilityDurable, []byte("rotate the auth token"), false)
	st.Add("tool", DurabilitySession, []byte("auth token runbook"), false)
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)
	// A budget large enough that the whole (tiny) probe is resident.
	plan := ix.PlanCells(Forecast{Intents: []string{"auth token"}}, Budget{Tokens: 9999}, nil, ProbeOptions{})

	w := StoreAudit(spans, plan)
	if w.Pruned != 0 {
		t.Errorf("a small store under a big budget should prune nothing, got %d pruned", w.Pruned)
	}
	if w.Resident != w.StoreSpans {
		t.Errorf("every store span should be resident: resident=%d store=%d", w.Resident, w.StoreSpans)
	}
	if !w.Faithful || !w.Partition {
		t.Errorf("an all-resident plan must be store-faithful: %+v", w)
	}
}

// TestStoreAuditRefusesDuplicateStoreIDs is the fail-closed guard the adversarial review
// demanded: two DISTINCT physical spans sharing one id make the recovery handle (the id the
// store pages back in by) AMBIGUOUS, so the store cannot honestly be certified recoverable.
// StoreAudit must detect the duplicate id and refuse to certify — never silently absorb the
// second physical span into a "faithful" verdict. (No shipped store produces duplicate ids;
// this is the malformed-store guard, the store-scope counterpart of the index's unique-id
// addressing contract.)
func TestStoreAuditRefusesDuplicateStoreIDs(t *testing.T) {
	spans := []Span{
		{ID: "dup", Step: 0, Role: "tool", Descriptor: "auth token one", Digest: "d0", Bytes: 20, Durability: DurabilitySession},
		{ID: "dup", Step: 1, Role: "tool", Descriptor: "auth token two", Digest: "d1", Bytes: 20, Durability: DurabilitySession},
	}
	ix := BuildIndex(spans)
	plan := ix.PlanCells(Forecast{Intents: []string{"auth token"}}, Budget{Tokens: 5}, nil, ProbeOptions{})

	w := StoreAudit(spans, plan)
	if len(w.DuplicateStoreIDs) == 0 || w.DuplicateStoreIDs[0] != "dup" {
		t.Fatalf("a duplicate store id must be reported, got %v", w.DuplicateStoreIDs)
	}
	if w.Partition || w.Faithful {
		t.Errorf("a duplicate-id store must NOT be certified (ambiguous recovery handle): %+v", w)
	}
}
