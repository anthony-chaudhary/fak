package recall

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// sessionForCtxStore builds the airline core image and reloads it into a fresh Session,
// then wraps it as a ctxplan.Store — the first real-store backing (#545). The image has
// two benign pages (0 = account, 2 = flights) and two quarantined pages (1 = injected
// policy, 3 = secret leak), so every adapter code path is exercised.
func sessionForCtxStore(t *testing.T) *CtxStore {
	t.Helper()
	return NewCtxStore(persistAndReload(t, recordAirline(t)))
}

// TestCtxStoreSpansLowersPagesWithSealedFlag witnesses the SAFE-metadata lowering: one
// ctxplan.Span per recall page, with the quarantined pages marked Sealed, a reversible
// "page:<step>" id, and the content-address Digest that keeps an elided span recoverable.
func TestCtxStoreSpansLowersPagesWithSealedFlag(t *testing.T) {
	ctx := context.Background()
	store := sessionForCtxStore(t)

	spans, err := store.Spans(ctx)
	if err != nil {
		t.Fatalf("Spans: %v", err)
	}
	if len(spans) != 4 {
		t.Fatalf("want 4 spans, got %d", len(spans))
	}
	wantSealed := map[string]bool{"page:0": false, "page:1": true, "page:2": false, "page:3": true}
	got := map[string]bool{}
	for _, s := range spans {
		if s.Digest == "" {
			t.Errorf("span %s has no Digest — an elided span would be unrecoverable", s.ID)
		}
		if s.ID != pageSpanID(s.Step) {
			t.Errorf("span id/step mismatch: id=%q step=%d", s.ID, s.Step)
		}
		if s.Role == "" {
			t.Errorf("span %s has no Role", s.ID)
		}
		got[s.ID] = s.Sealed
	}
	for id, sealed := range wantSealed {
		if got[id] != sealed {
			t.Errorf("span %s: want Sealed=%v, got %v", id, sealed, got[id])
		}
	}
	// The benign account page's Span.Bytes must equal the recorded body length so the
	// planner's cost contract holds (Materialize hands back that many bytes).
	if spans[0].Bytes != int64(len(benignAccount)) {
		t.Errorf("page:0 Bytes=%d, want %d (len body)", spans[0].Bytes, len(benignAccount))
	}
}

// TestCtxStoreMaterializeRoutesThroughTrustGate is the load-bearing property: page-in is
// NOT a raw CAS read, it goes through Session.Resolve. Benign pages come back
// byte-identical; quarantined pages surface as ctxplan.ErrSealed so the planner refuses
// them; an unknown id is a plain (non-sealed) error.
func TestCtxStoreMaterializeRoutesThroughTrustGate(t *testing.T) {
	ctx := context.Background()
	store := sessionForCtxStore(t)

	for _, tc := range []struct {
		id     string
		want   string // byte-identical body for a benign page; "" for an error case
		sealed bool
	}{
		{"page:0", benignAccount, false},
		{"page:2", benignFlights, false},
		{"page:1", "", true}, // injected policy — quarantined
		{"page:3", "", true}, // secret leak — quarantined
	} {
		body, err := store.Materialize(ctx, tc.id)
		if tc.want != "" {
			if err != nil {
				t.Errorf("%s: unexpected error %v", tc.id, err)
				continue
			}
			if string(body) != tc.want {
				t.Errorf("%s: not byte-identical\n got %q\nwant %q", tc.id, body, tc.want)
			}
			continue
		}
		if err == nil {
			t.Errorf("%s: expected a trust-gate refusal, got bytes", tc.id)
			continue
		}
		if sealed := errors.Is(err, ctxplan.ErrSealed); sealed != tc.sealed {
			t.Errorf("%s: errors.Is(err, ctxplan.ErrSealed)=%v, want %v (err=%v)", tc.id, sealed, tc.sealed, err)
		}
	}

	if _, err := store.Materialize(ctx, "span:0"); err == nil {
		t.Error(`an unknown id "span:0" must error, not return bytes`)
	} else if errors.Is(err, ctxplan.ErrSealed) || errors.Is(err, ctxplan.ErrTombstoned) {
		t.Errorf("an unknown id must be a generic error, not sealed/tombstoned: %v", err)
	}
}

// TestCtxStoreBacksCtxPlanMaterialize is the end-to-end witness: the REAL recall store
// drives the planner's full Materialize pass. Pinning the two benign spans forces them
// resident deterministically; the quarantined spans are Sealed (scored 0) and elided.
// The view's Witness must be Faithful (no candidate destroyed), Reconciled (every
// selected span rendered or refused), and hold its CostContract (rendered bytes == the
// Span.Bytes the planner charged) — proving a real store is an honest backing.
func TestCtxStoreBacksCtxPlanMaterialize(t *testing.T) {
	ctx := context.Background()
	store := sessionForCtxStore(t)

	// Pins force the benign spans resident; the sealed spans can never be pinned in.
	f := ctxplan.Forecast{Pins: []string{"page:0", "page:2"}}
	v, err := ctxplan.Materialize(ctx, store, f, ctxplan.Budget{Tokens: 1000}, nil)
	if err != nil {
		t.Fatalf("ctxplan.Materialize over recall store: %v", err)
	}

	rendered := map[string]bool{}
	for _, r := range v.Rendered {
		rendered[r.ID] = true
	}
	if !rendered["page:0"] || !rendered["page:2"] {
		t.Errorf("benign pages must render, got rendered=%v", rendered)
	}
	if rendered["page:1"] || rendered["page:3"] {
		t.Errorf("quarantined pages must never render, got rendered=%v", rendered)
	}
	if len(v.Refused) != 0 {
		t.Errorf("no selected span should be refused (only benign pages are pinned), got %+v", v.Refused)
	}

	w := v.Witness
	if !w.Faithful {
		t.Errorf("Witness must be Faithful over a real store: %+v", w)
	}
	if !w.Reconciled {
		t.Errorf("Witness must be Reconciled: rendered=%d refused=%d selected=%d", w.Rendered, w.Refused, len(v.Plan.Selected))
	}
	if !w.CostContract {
		t.Errorf("CostContract must hold (Page.Len == len(Resolve body)): diverged=%v", w.CostDiverged)
	}

	// The planner charges Span.Bytes; the store hands back exactly that many bytes, so
	// re-materializing a rendered span returns the real body — the content check behind
	// the cost contract.
	body, err := store.Materialize(ctx, "page:0")
	if err != nil {
		t.Fatalf("re-materialize page:0: %v", err)
	}
	if string(body) != benignAccount {
		t.Errorf("page:0 content drifted:\n got %q\nwant %q", body, benignAccount)
	}
}

// TestCtxStoreTombstoneRefusesAsCtxPlanTombstoned witnesses the second gate: a page
// suppressed by context control surfaces through the adapter as ctxplan.ErrTombstoned
// (a SEPARATE vocabulary from sealed), and Spans marks it Tombstoned live — so a
// context-control change between plans is honored without rebuilding the store.
func TestCtxStoreTombstoneRefusesAsCtxPlanTombstoned(t *testing.T) {
	ctx := context.Background()
	store := sessionForCtxStore(t)

	// page:0 is benign and resolves cleanly before the tombstone.
	if _, err := store.Materialize(ctx, "page:0"); err != nil {
		t.Fatalf("page:0 should resolve before tombstoning: %v", err)
	}

	if _, err := store.Session().RequestContextChange(ContextChangeRequest{
		Action: ContextActionTombstone,
		Step:   0,
		Reason: "stale account — suppressed from future context",
	}); err != nil {
		t.Fatalf("RequestContextChange: %v", err)
	}

	// Spans reads tombstone state live.
	spans, err := store.Spans(ctx)
	if err != nil {
		t.Fatalf("Spans after tombstone: %v", err)
	}
	if !spans[0].Tombstoned {
		t.Error("page:0 must be Tombstoned in Spans after the context-control change")
	}

	// Page-in is refused with the ctxplan tombstone vocabulary the planner branches on.
	if _, err := store.Materialize(ctx, "page:0"); err == nil {
		t.Fatal("a tombstoned page must be refused on page-in")
	} else if !errors.Is(err, ctxplan.ErrTombstoned) {
		t.Errorf("want errors.Is(err, ctxplan.ErrTombstoned), got %v", err)
	} else if errors.Is(err, ctxplan.ErrSealed) {
		t.Error("a tombstone must NOT also report as sealed — the two gates are distinct")
	}
}

// TestCtxStoreDemandPageRefusesSealedSpan witnesses the recovery path: a mid-turn
// demand-page of a quarantined span is REFUSED (the gate holds, the lossless store keeps
// the bytes for audit), and an id naming no page is FaultAbsent, not an error.
func TestCtxStoreDemandPageRefusesSealedSpan(t *testing.T) {
	ctx := context.Background()
	store := sessionForCtxStore(t)

	_, fault, err := ctxplan.DemandPage(ctx, store, ctxplan.View{}, "page:1")
	if err != nil {
		t.Fatalf("DemandPage on a sealed span must not return a go error: %v", err)
	}
	if fault.Status != ctxplan.FaultRefused {
		t.Fatalf("want FaultRefused, got %q", fault.Status)
	}
	if fault.Reason != "sealed_by_trust_gate" {
		t.Errorf("want reason sealed_by_trust_gate, got %q", fault.Reason)
	}

	_, fault, err = ctxplan.DemandPage(ctx, store, ctxplan.View{}, "page:999")
	if err != nil {
		t.Fatalf("DemandPage on an absent id must not return a go error: %v", err)
	}
	if fault.Status != ctxplan.FaultAbsent {
		t.Errorf("want FaultAbsent for an unknown id, got %q", fault.Status)
	}
}
