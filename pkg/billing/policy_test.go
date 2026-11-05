package billing_test

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// i64 boxes a token count for the record pointer fields.
func i64(v int64) *int64 { return &v }

// day1/day2 are two distinct UTC days used by the windowing tests.
var (
	day1 = time.Date(2026, 11, 3, 9, 0, 0, 0, time.UTC)
	day2 = time.Date(2026, 11, 4, 9, 0, 0, 0, time.UTC)
)

// recordedRec builds a valid UsageRecord with the requested classification
// inferred from which token fields are present: at least one reported field
// means recorded, none means missing.
func recordedRec(event, tenant, key, model string, requested time.Time, prompt, completion, cached *int64) billing.UsageRecord {
	class := billing.UsageMissing
	if prompt != nil || completion != nil || cached != nil {
		class = billing.UsageRecorded
	}
	return billing.UsageRecord{
		SchemaVersion:    billing.SchemaVersionV1,
		EventID:          event,
		TenantID:         tenant,
		KeyID:            key,
		Model:            model,
		RequestedAt:      requested,
		PromptTokens:     prompt,
		CompletionTokens: completion,
		CachedTokens:     cached,
		Classification:   class,
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestVerdictVocabularyClosed(t *testing.T) {
	want := map[billing.Verdict]bool{
		billing.VerdictAdmit:        true,
		billing.VerdictRejectBudget: true,
		billing.VerdictRejectRate:   true,
	}
	if len(billing.Verdicts) != len(want) {
		t.Fatalf("verdict vocabulary has %d members, want %d", len(billing.Verdicts), len(want))
	}
	for _, v := range billing.Verdicts {
		if !want[v] {
			t.Fatalf("unexpected verdict member %q", v)
		}
		if !v.Valid() {
			t.Fatalf("member %q reports invalid", v)
		}
	}
	for _, bad := range []billing.Verdict{"", "ADMIT", "reject", "reject-budget", "admit ", "reject_budget_extra"} {
		if bad.Valid() {
			t.Fatalf("non-member %q reports valid", bad)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPricingModeVocabularyClosed(t *testing.T) {
	want := map[billing.PricingMode]bool{
		billing.PricingNoCharge: true,
		billing.PricingCharged:  true,
		billing.PricingUnpriced: true,
	}
	if len(billing.PricingModes) != len(want) {
		t.Fatalf("pricing-mode vocabulary has %d members, want %d", len(billing.PricingModes), len(want))
	}
	for _, m := range billing.PricingModes {
		if !want[m] {
			t.Fatalf("unexpected pricing-mode member %q", m)
		}
		if !m.Valid() {
			t.Fatalf("member %q reports invalid", m)
		}
	}
	for _, bad := range []billing.PricingMode{"", "free", "no-charge", "charged ", "UNPRICED"} {
		if bad.Valid() {
			t.Fatalf("non-member %q reports valid", bad)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPriceClosedContract(t *testing.T) {
	cases := []struct {
		name    string
		p       billing.Price
		wantErr error
	}{
		{"no charge with zero amount", billing.Price{Mode: billing.PricingNoCharge}, nil},
		{"charged positive amount", billing.Price{Mode: billing.PricingCharged, AmountMicroUSD: 123, PriceBookVersion: "stamp-1"}, nil},
		{"charged zero amount", billing.Price{Mode: billing.PricingCharged, AmountMicroUSD: 0}, nil},
		{"unpriced with zero amount", billing.Price{Mode: billing.PricingUnpriced}, nil},
		{"no charge carrying an amount", billing.Price{Mode: billing.PricingNoCharge, AmountMicroUSD: 5}, billing.ErrInconsistentPrice},
		{"unpriced carrying an amount", billing.Price{Mode: billing.PricingUnpriced, AmountMicroUSD: 5}, billing.ErrInconsistentPrice},
		{"charged negative amount", billing.Price{Mode: billing.PricingCharged, AmountMicroUSD: -1}, billing.ErrInconsistentPrice},
		{"unknown mode", billing.Price{Mode: "free"}, billing.ErrUnknownPricingMode},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.p.Validate()
			if tc.wantErr == nil {
				if err != nil {
					t.Fatalf("valid price rejected: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("want errors.Is(%v), got %v", tc.wantErr, err)
			}
			if !errors.Is(err, billing.ErrInvalidPrice) {
				t.Fatalf("specific sentinel does not wrap ErrInvalidPrice: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNeutralDefaultPricesNoChargeExplicitly(t *testing.T) {
	p := billing.NewNeutral(billing.CapConfig{})

	// Reported usage prices no-charge with a known-zero amount.
	rec := recordedRec("e1", "t1", "k1", "model-a", day1, i64(100), i64(50), i64(10))
	price := p.Price(rec)
	if price.Mode != billing.PricingNoCharge {
		t.Fatalf("neutral default priced %q, want no_charge", price.Mode)
	}
	if price.AmountMicroUSD != 0 {
		t.Fatalf("no-charge price carries amount %d", price.AmountMicroUSD)
	}
	if err := price.Validate(); err != nil {
		t.Fatalf("no-charge price fails its own closed contract: %v", err)
	}

	// Unknown usage (nothing reported) is STILL no-charge, explicitly
	// typed — never unpriced, never a coerced zero masquerading as known.
	missing := recordedRec("e2", "t1", "k1", "model-a", day1, nil, nil, nil)
	pm := p.Price(missing)
	if pm.Mode != billing.PricingNoCharge {
		t.Fatalf("unknown usage priced %q, want no_charge", pm.Mode)
	}
	if pm.Mode == billing.PricingUnpriced {
		t.Fatal("unknown usage must not read as unpriced under the neutral default")
	}

	// The record's price-book version pointer passes through unchanged.
	stamped := rec
	stamped.PriceBookVersion = "stamp-9"
	if got := p.Price(stamped).PriceBookVersion; got != "stamp-9" {
		t.Fatalf("price-book version pointer %q, want stamp-9", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNeutralDailyCapAdmitsThenRefuses(t *testing.T) {
	p := billing.NewNeutral(billing.CapConfig{DailyTokenCap: 1000})
	principal := billing.Principal{TenantID: "t1"}
	ctx := billing.AdmissionContext{Principal: principal, RequestedAt: day1}

	if v := p.Admit(ctx); v != billing.VerdictAdmit {
		t.Fatalf("fresh principal refused before any spend: %q", v)
	}

	// 600 of 1000 tokens: still under.
	r1 := recordedRec("e1", "t1", "k1", "model-a", day1, i64(600), nil, nil)
	if _, err := p.Settle(r1, p.Price(r1)); err != nil {
		t.Fatalf("settle: %v", err)
	}
	if v := p.Admit(ctx); v != billing.VerdictAdmit {
		t.Fatalf("principal under cap refused: %q", v)
	}

	// Another 500 crosses the ceiling: 1100 >= 1000.
	r2 := recordedRec("e2", "t1", "k1", "model-a", day1, i64(500), nil, nil)
	if _, err := p.Settle(r2, p.Price(r2)); err != nil {
		t.Fatalf("settle: %v", err)
	}
	if v := p.Admit(ctx); v != billing.VerdictRejectBudget {
		t.Fatalf("breached principal admitted with verdict %q, want reject_budget", v)
	}

	// A different tenant is unaffected.
	other := billing.AdmissionContext{Principal: billing.Principal{TenantID: "t2"}, RequestedAt: day1}
	if v := p.Admit(other); v != billing.VerdictAdmit {
		t.Fatalf("unrelated tenant refused: %q", v)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNeutralSessionCapIsCumulativeNotDaily(t *testing.T) {
	p := billing.NewNeutral(billing.CapConfig{SessionTokenCap: 100})

	// Session k1 spends its whole ceiling on day 1.
	r1 := recordedRec("e1", "t1", "k1", "model-a", day1, i64(100), nil, nil)
	if _, err := p.Settle(r1, p.Price(r1)); err != nil {
		t.Fatalf("settle: %v", err)
	}

	same := billing.AdmissionContext{
		Principal:   billing.Principal{TenantID: "t1", SessionID: "k1"},
		RequestedAt: day2, // a later day: the session window does not reset
	}
	if v := p.Admit(same); v != billing.VerdictRejectBudget {
		t.Fatalf("session over cumulative cap admitted on a later day: %q", v)
	}

	// A fresh session under the same tenant is unaffected.
	fresh := billing.AdmissionContext{
		Principal:   billing.Principal{TenantID: "t1", SessionID: "k2"},
		RequestedAt: day2,
	}
	if v := p.Admit(fresh); v != billing.VerdictAdmit {
		t.Fatalf("fresh session refused: %q", v)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNeutralDailyWindowResetsAcrossDays(t *testing.T) {
	p := billing.NewNeutral(billing.CapConfig{DailyTokenCap: 1000})
	principal := billing.Principal{TenantID: "t1"}

	r1 := recordedRec("e1", "t1", "k1", "model-a", day1, i64(1000), nil, nil)
	if _, err := p.Settle(r1, p.Price(r1)); err != nil {
		t.Fatalf("settle: %v", err)
	}

	if v := p.Admit(billing.AdmissionContext{Principal: principal, RequestedAt: day1}); v != billing.VerdictRejectBudget {
		t.Fatalf("day-1 principal admitted after spending its daily cap: %q", v)
	}
	// The next UTC day starts a fresh window: the same principal admits.
	if v := p.Admit(billing.AdmissionContext{Principal: principal, RequestedAt: day2}); v != billing.VerdictAdmit {
		t.Fatalf("day-2 principal refused although the daily window moved: %q", v)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNeutralSettleIsIdempotentPerEventID(t *testing.T) {
	// Ceiling 1200: single-charged totals (600 + 500 = 1100) stay under;
	// a double charge of the first record (1700) would cross it.
	p := billing.NewNeutral(billing.CapConfig{SessionTokenCap: 1200})

	r1 := recordedRec("e1", "t1", "k1", "model-a", day1, i64(600), nil, nil)
	s1, err := p.Settle(r1, p.Price(r1))
	if err != nil {
		t.Fatalf("first settle: %v", err)
	}
	s1again, err := p.Settle(r1, p.Price(r1))
	if err != nil {
		t.Fatalf("replay settle: %v", err)
	}
	if s1again.EventID != s1.EventID || s1again.Model != s1.Model || s1again.Classification != s1.Classification {
		t.Fatalf("replay returned a different settlement: %+v vs %+v", s1again, s1)
	}

	r2 := recordedRec("e2", "t1", "k1", "model-a", day1, i64(500), nil, nil)
	if _, err := p.Settle(r2, p.Price(r2)); err != nil {
		t.Fatalf("second settle: %v", err)
	}

	ctx := billing.AdmissionContext{
		Principal:   billing.Principal{TenantID: "t1", SessionID: "k1"},
		RequestedAt: day1,
	}
	if v := p.Admit(ctx); v != billing.VerdictAdmit {
		t.Fatalf("caps charged more than once per event id (total crossed 1200): %q", v)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNeutralSettleRejectsInvalidRecordAndPrice(t *testing.T) {
	p := billing.NewNeutral(billing.CapConfig{})

	bad := recordedRec("e1", "t1", "k1", "model-a", day1, i64(10), nil, nil)
	bad.Model = ""
	if _, err := p.Settle(bad, p.Price(bad)); !errors.Is(err, billing.ErrInvalidRecord) {
		t.Fatalf("want ErrInvalidRecord, got %v", err)
	}

	good := recordedRec("e2", "t1", "k1", "model-a", day1, i64(10), nil, nil)
	badPrice := billing.Price{Mode: billing.PricingNoCharge, AmountMicroUSD: 7}
	if _, err := p.Settle(good, badPrice); !errors.Is(err, billing.ErrInconsistentPrice) {
		t.Fatalf("want ErrInconsistentPrice, got %v", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNeutralReconcileReportsNothingOutstanding(t *testing.T) {
	p := billing.NewNeutral(billing.CapConfig{})
	r, err := p.Reconcile(billing.ReconcileScope{Since: day1, Until: day2, TenantID: "t1"})
	if err != nil {
		t.Fatalf("neutral reconcile errored: %v", err)
	}
	if r != (billing.Reconciliation{}) {
		t.Fatalf("neutral reconcile reported %+v, want the zero report", r)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsConcurrentAddAndRead(t *testing.T) {
	caps := billing.NewSpendCaps(billing.CapConfig{DailyTokenCap: 1_000_000, SessionTokenCap: 1_000_000})
	p := billing.Principal{TenantID: "t", SessionID: "s"}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				caps.Add(p, 1, day1)
				_ = caps.Breached(p, day1)
				_, _ = caps.Spent(p, day1)
			}
		}()
	}
	wg.Wait()

	daily, session := caps.Spent(p, day1)
	if daily != 4000 || session != 4000 {
		t.Fatalf("concurrent totals daily=%d session=%d, want 4000/4000", daily, session)
	}
	if caps.Breached(p, day1) {
		t.Fatal("totals far below both ceilings report breached")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsActionDefaultsToPause(t *testing.T) {
	if got := billing.NewSpendCaps(billing.CapConfig{}).Action(); got != billing.CapActionPause {
		t.Fatalf("default breach action %q, want pause", got)
	}
	if got := billing.NewSpendCaps(billing.CapConfig{Action: billing.CapActionKill}).Action(); got != billing.CapActionKill {
		t.Fatalf("configured breach action %q, want kill", got)
	}
	for _, bad := range []billing.CapAction{"", "PAUSE", "halt"} {
		if bad.Valid() {
			t.Fatalf("non-member cap action %q reports valid", bad)
		}
	}
}

// fakePolicy is a hand-rolled Policy that records what it was asked.
type fakePolicy struct {
	admit       billing.Verdict
	price       billing.Price
	settleCalls int
	lastRec     billing.UsageRecord
}

func (f *fakePolicy) Admit(ctx billing.AdmissionContext) billing.Verdict { return f.admit }

func (f *fakePolicy) Price(rec billing.UsageRecord) billing.Price {
	f.lastRec = rec
	return f.price
}

func (f *fakePolicy) Settle(rec billing.UsageRecord, p billing.Price) (billing.Settlement, error) {
	f.settleCalls++
	return billing.Settlement{EventID: rec.EventID, Model: rec.Model, Price: p}, nil
}

func (f *fakePolicy) Reconcile(scope billing.ReconcileScope) (billing.Reconciliation, error) {
	return billing.Reconciliation{}, nil
}

// fakeSink records submitted event IDs and can fail on demand.
type fakeSink struct {
	mu   sync.Mutex
	got  []string
	fail bool
}

func (s *fakeSink) Submit(rec billing.UsageRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.fail {
		return errors.New("sink down")
	}
	s.got = append(s.got, rec.EventID)
	return nil
}

func (s *fakeSink) events() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, len(s.got))
	copy(out, s.got)
	return out
}

// fak-test:runtime fast est=1ms lane=default
func TestEngineWiresPolicyAndSink(t *testing.T) {
	fp := &fakePolicy{admit: billing.VerdictAdmit, price: billing.Price{Mode: billing.PricingNoCharge}}
	sink := &fakeSink{}
	eng, err := billing.New(fp, sink)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if eng.Policy() != billing.Policy(fp) {
		t.Fatal("engine does not expose the injected policy")
	}

	rec := recordedRec("e1", "t1", "k1", "model-a", day1, i64(10), nil, nil)
	if v := eng.Admit(billing.AdmissionContext{RequestedAt: day1}); v != billing.VerdictAdmit {
		t.Fatalf("admit not delegated: %q", v)
	}

	s, err := eng.Complete(rec)
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if s.EventID != "e1" || s.Price.Mode != billing.PricingNoCharge {
		t.Fatalf("unexpected settlement: %+v", s)
	}
	if fp.settleCalls != 1 || fp.lastRec.EventID != "e1" {
		t.Fatalf("policy not driven exactly once: calls=%d last=%q", fp.settleCalls, fp.lastRec.EventID)
	}
	if got := sink.events(); len(got) != 1 || got[0] != "e1" {
		t.Fatalf("sink did not receive the settled record: %v", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestEngineCompleteReportsForwardFailureAfterSettle(t *testing.T) {
	fp := &fakePolicy{price: billing.Price{Mode: billing.PricingNoCharge}}
	sink := &fakeSink{fail: true}
	eng, err := billing.New(fp, sink)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	rec := recordedRec("e2", "t1", "k1", "model-a", day1, i64(10), nil, nil)
	s, err := eng.Complete(rec)
	if !errors.Is(err, billing.ErrSinkForward) {
		t.Fatalf("want ErrSinkForward, got %v", err)
	}
	if s.EventID != "e2" {
		t.Fatalf("settled record lost on forward failure: %+v", s)
	}
	if fp.settleCalls != 1 {
		t.Fatalf("settle ran %d times, want 1", fp.settleCalls)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestEngineFailsClosedWithoutPolicy(t *testing.T) {
	if _, err := billing.New(nil, nil); !errors.Is(err, billing.ErrNoPolicy) {
		t.Fatalf("want ErrNoPolicy, got %v", err)
	}

	var zero billing.Engine
	if v := zero.Admit(billing.AdmissionContext{}); v != billing.VerdictRejectBudget {
		t.Fatalf("zero-value engine admitted: %q", v)
	}
	rec := recordedRec("e1", "t1", "k1", "model-a", day1, i64(10), nil, nil)
	if p := zero.Price(rec); p.Mode != billing.PricingUnpriced {
		t.Fatalf("zero-value engine priced %q, want unpriced", p.Mode)
	}
	if _, err := zero.Settle(rec); !errors.Is(err, billing.ErrNoPolicy) {
		t.Fatalf("want ErrNoPolicy from settle, got %v", err)
	}
	if _, err := zero.Complete(rec); !errors.Is(err, billing.ErrNoPolicy) {
		t.Fatalf("want ErrNoPolicy from complete, got %v", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestEngineNilSinkDisablesForwarding(t *testing.T) {
	fp := &fakePolicy{price: billing.Price{Mode: billing.PricingNoCharge}}
	eng, err := billing.New(fp, nil)
	if err != nil {
		t.Fatalf("New with nil sink must be accepted: %v", err)
	}
	rec := recordedRec("e1", "t1", "k1", "model-a", day1, i64(10), nil, nil)
	if _, err := eng.Complete(rec); err != nil {
		t.Fatalf("complete without a sink: %v", err)
	}
	if err := eng.Submit(rec); err != nil {
		t.Fatalf("submit without a sink: %v", err)
	}
	if fp.settleCalls != 1 {
		t.Fatalf("settle ran %d times, want 1", fp.settleCalls)
	}
}
