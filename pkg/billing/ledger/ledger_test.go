package ledger_test

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
	"github.com/anthony-chaudhary/fak/pkg/billing/ledger"
)

func openLedger(t *testing.T, dir string) *ledger.Ledger {
	t.Helper()
	l, err := ledger.Open(filepath.Join(dir, "log.jsonl"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return l
}

// snapshot captures the full externally visible state: every account touched
// by an entry, every reservation named by an entry, and the entry log.
type snapshot struct {
	Accounts     map[string]ledger.Account
	Reservations map[string]ledger.Reservation
	Entries      []ledger.Entry
}

func snap(l *ledger.Ledger) snapshot {
	s := snapshot{
		Accounts:     map[string]ledger.Account{},
		Reservations: map[string]ledger.Reservation{},
	}
	for _, e := range l.Entries() {
		if _, ok := s.Accounts[e.TenantID]; !ok {
			a, err := l.GetAccount(e.TenantID)
			if err != nil {
				panic(err)
			}
			s.Accounts[e.TenantID] = *a
		}
		if e.ReservationID != "" {
			r, err := l.GetReservation(e.ReservationID)
			if err != nil {
				panic(err)
			}
			s.Reservations[e.ReservationID] = *r
		}
	}
	s.Entries = l.Entries()
	return s
}

func i64p(v int64) *int64 { return &v }

// fak-test:runtime fast est=5ms lane=default
func TestReplayYieldsIdenticalState(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "log.jsonl")
	l, err := ledger.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	// A mixed workload across two tenants: deposits, partial / full /
	// known-zero settles, a release, and a reconcile.
	if err := l.Deposit("ev-d1", "tenant-a", 1000); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
	if err := l.Deposit("ev-d2", "tenant-a", 500); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
	if err := l.Deposit("ev-d3", "tenant-b", 400); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
	resA, err := l.Reserve("ev-r1", "tenant-a", "req-a", 400)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	resB, err := l.Reserve("ev-r2", "tenant-a", "req-b", 200)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	resC, err := l.Reserve("ev-r3", "tenant-b", "req-c", 300)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	resD, err := l.Reserve("ev-r4", "tenant-a", "req-d", 100)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if _, err = l.Reserve("ev-r5", "tenant-b", "req-e", 100); err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if err := l.Settle("ev-s1", resA.ID, 150); err != nil { // partial
		t.Fatalf("Settle: %v", err)
	}
	if err := l.Settle("ev-s2", resB.ID, 200); err != nil { // full
		t.Fatalf("Settle: %v", err)
	}
	if err := l.Settle("ev-s3", resD.ID, 0); err != nil { // known zero
		t.Fatalf("Settle: %v", err)
	}
	if err := l.Release("ev-x1", resC.ID); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if err := l.Reconcile("ev-q1", "tenant-b", "req-e", 75); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	if err := l.CheckInvariants(); err != nil {
		t.Fatalf("CheckInvariants before replay: %v", err)
	}
	before := snap(l)

	// Reopen the same path: a fresh ledger rebuilt from the log alone.
	l2, err := ledger.Open(path)
	if err != nil {
		t.Fatalf("re-Open: %v", err)
	}
	after := snap(l2)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("replayed state differs from live state:\nbefore=%#v\nafter=%#v", before, after)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDuplicateEventIDAppliesOnce(t *testing.T) {
	l := ledger.New()
	if err := l.Deposit("e1", "t", 100); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
	res, err := l.Reserve("e2", "t", "req", 100)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if err := l.Settle("e3", res.ID, 40); err != nil {
		t.Fatalf("Settle: %v", err)
	}

	// The same event id arriving again — same operation, or any other
	// operation — is the dedup verdict and changes nothing.
	if err := l.Settle("e3", res.ID, 40); !errors.Is(err, ledger.ErrAlreadyApplied) {
		t.Errorf("repeat Settle: got %v, want ErrAlreadyApplied", err)
	}
	if err := l.Deposit("e3", "t", 50); !errors.Is(err, ledger.ErrAlreadyApplied) {
		t.Errorf("repeat Deposit: got %v, want ErrAlreadyApplied", err)
	}
	if err := l.Release("e3", res.ID); !errors.Is(err, ledger.ErrAlreadyApplied) {
		t.Errorf("repeat Release: got %v, want ErrAlreadyApplied", err)
	}

	acc, err := l.GetAccount("t")
	if err != nil {
		t.Fatalf("GetAccount: %v", err)
	}
	if acc.Granted != 100 || acc.Available != 60 || acc.Reserved != 0 || acc.Settled != 40 {
		t.Errorf("account after duplicate events = %+v, want granted=100 available=60 reserved=0 settled=40", acc)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestReconcileOnceAcrossReplays(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "log.jsonl")
	l, err := ledger.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := l.Deposit("d1", "t", 1000); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
	if _, err := l.Reserve("r1", "t", "req-x", 500); err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if err := l.Reconcile("q1", "t", "req-x", 300); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	acc, _ := l.GetAccount("t")
	if acc.Settled != 300 {
		t.Fatalf("settled after reconcile = %d, want 300", acc.Settled)
	}

	// Same event, different event id: the pair is already settled.
	if err := l.Reconcile("q2", "t", "req-x", 300); !errors.Is(err, ledger.ErrAlreadySettled) {
		t.Errorf("second Reconcile: got %v, want ErrAlreadySettled", err)
	}
	// Same event id: the dedup verdict.
	if err := l.Reconcile("q1", "t", "req-x", 300); !errors.Is(err, ledger.ErrAlreadyApplied) {
		t.Errorf("repeat Reconcile: got %v, want ErrAlreadyApplied", err)
	}

	// Replay rebuilds the settled-key index from the log: still refused.
	l2, err := ledger.Open(path)
	if err != nil {
		t.Fatalf("re-Open: %v", err)
	}
	if err := l2.Reconcile("q3", "t", "req-x", 300); !errors.Is(err, ledger.ErrAlreadySettled) {
		t.Errorf("post-replay Reconcile: got %v, want ErrAlreadySettled", err)
	}
	acc2, _ := l2.GetAccount("t")
	if acc2.Settled != 300 {
		t.Errorf("settled after replay = %d, want 300", acc2.Settled)
	}
	if err := l2.CheckInvariants(); err != nil {
		t.Errorf("CheckInvariants after replay: %v", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTypedErrorContract(t *testing.T) {
	l := ledger.New()
	if err := l.Deposit("d1", "t", 200); err != nil {
		t.Fatalf("seed Deposit: %v", err)
	}
	resA, err := l.Reserve("r1", "t", "req-a", 60)
	if err != nil {
		t.Fatalf("seed Reserve: %v", err)
	}
	resB, err := l.Reserve("r2", "t", "req-b", 40)
	if err != nil {
		t.Fatalf("seed Reserve: %v", err)
	}
	resC, err := l.Reserve("r3", "t", "req-c", 40)
	if err != nil {
		t.Fatalf("seed Reserve: %v", err)
	}
	if err := l.Settle("s1", resA.ID, 60); err != nil {
		t.Fatalf("seed Settle: %v", err)
	}
	if err := l.Release("x1", resB.ID); err != nil {
		t.Fatalf("seed Release: %v", err)
	}

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"reserve unknown tenant", func() error { _, e := l.Reserve("e", "ghost", "req", 10); return e }, ledger.ErrInsufficientBalance},
		{"reserve exceeds available", func() error { _, e := l.Reserve("e", "t", "req-big", 1000); return e }, ledger.ErrInsufficientBalance},
		{"duplicate pending reservation", func() error { _, e := l.Reserve("e", "t", "req-c", 10); return e }, ledger.ErrDuplicateReservation},
		{"settle unknown reservation", func() error { return l.Settle("e", "res-none", 1) }, ledger.ErrReservationNotFound},
		{"settle already settled", func() error { return l.Settle("e", resA.ID, 1) }, ledger.ErrAlreadySettled},
		{"settle already released", func() error { return l.Settle("e", resB.ID, 1) }, ledger.ErrAlreadyReleased},
		{"settle exceeds reserved", func() error { return l.Settle("e", resC.ID, 41) }, ledger.ErrExceedsReserved},
		{"settle negative quantity", func() error { return l.Settle("e", resC.ID, -1) }, ledger.ErrInvalidQuantity},
		{"release already settled", func() error { return l.Release("e", resA.ID) }, ledger.ErrAlreadySettled},
		{"release already released", func() error { return l.Release("e", resB.ID) }, ledger.ErrAlreadyReleased},
		{"empty event id", func() error { return l.Deposit("", "t", 1) }, ledger.ErrMissingEventID},
		{"empty tenant id", func() error { return l.Deposit("e", "", 1) }, ledger.ErrEmptyTenantID},
		{"zero deposit", func() error { return l.Deposit("e", "t", 0) }, ledger.ErrInvalidQuantity},
		{"unknown account", func() error { _, e := l.GetAccount("ghost"); return e }, ledger.ErrAccountNotFound},
		{"reconcile without pending reservation", func() error { return l.Reconcile("e", "t", "req-z", 5) }, ledger.ErrReservationNotFound},
		{"reconcile already-settled pair", func() error { return l.Reconcile("e", "t", "req-a", 5) }, ledger.ErrAlreadySettled},
		{"nil receiver fails closed", func() error { return (*ledger.Ledger)(nil).Deposit("e", "t", 1) }, ledger.ErrInvalidArgument},
	}
	for _, tc := range cases {
		err := tc.fn()
		if !errors.Is(err, tc.want) {
			t.Errorf("%s: got %v, want errors.Is match on %v", tc.name, err, tc.want)
		}
	}

	// Argument-shape faults share their root sentinel.
	for _, err := range []error{
		ledger.ErrMissingEventID,
		ledger.ErrEmptyTenantID,
		ledger.ErrEmptyRequestID,
		ledger.ErrEmptyReservationID,
		ledger.ErrInvalidQuantity,
	} {
		if !errors.Is(err, ledger.ErrInvalidArgument) {
			t.Errorf("%v does not wrap ErrInvalidArgument", err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestOpenRefusesCorruptLog(t *testing.T) {
	validDeposit := `{"seq":1,"event_id":"e1","time":"2026-11-04T00:00:00Z","kind":"deposit","tenant_id":"t","units":5}`
	validSecond := `{"seq":2,"event_id":"e2","time":"2026-11-04T00:00:01Z","kind":"deposit","tenant_id":"t","units":5}`

	cases := []struct {
		name    string
		content string
	}{
		{"not json", "garbage"},
		{"unknown field", `{"seq":1,"event_id":"e1","time":"2026-11-04T00:00:00Z","kind":"deposit","tenant_id":"t","units":5,"extra":1}`},
		{"unknown kind", `{"seq":1,"event_id":"e1","time":"2026-11-04T00:00:00Z","kind":"teleport","tenant_id":"t","units":5}`},
		{"missing event id", `{"seq":1,"time":"2026-11-04T00:00:00Z","kind":"deposit","tenant_id":"t","units":5}`},
		{"sequence starts high", `{"seq":7,"event_id":"e1","time":"2026-11-04T00:00:00Z","kind":"deposit","tenant_id":"t","units":5}`},
		{"sequence gap", validDeposit + "\n" + `{"seq":3,"event_id":"e2","time":"2026-11-04T00:00:01Z","kind":"deposit","tenant_id":"t","units":5}`},
		{"negative units", `{"seq":1,"event_id":"e1","time":"2026-11-04T00:00:00Z","kind":"deposit","tenant_id":"t","units":-1}`},
		{"zero-unit deposit", `{"seq":1,"event_id":"e1","time":"2026-11-04T00:00:00Z","kind":"deposit","tenant_id":"t","units":0}`},
		{"settle without reserve", `{"seq":1,"event_id":"e1","time":"2026-11-04T00:00:00Z","kind":"settle","tenant_id":"t","request_id":"req","reservation_id":"res-x","units":5}`},
		{"trailing data", validDeposit + `{"seq":2}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "log.jsonl")
			if err := os.WriteFile(path, []byte(tc.content), 0o600); err != nil {
				t.Fatalf("write fixture: %v", err)
			}
			_, err := ledger.Open(path)
			if !errors.Is(err, ledger.ErrCorruptLog) {
				t.Fatalf("Open: got %v, want errors.Is match on ErrCorruptLog", err)
			}
		})
	}

	// A well-formed log opens cleanly.
	dir := t.TempDir()
	path := filepath.Join(dir, "log.jsonl")
	if err := os.WriteFile(path, []byte(validDeposit+"\n"+validSecond+"\n"), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	l, err := ledger.Open(path)
	if err != nil {
		t.Fatalf("Open of well-formed log: %v", err)
	}
	if got := len(l.Entries()); got != 2 {
		t.Fatalf("entries = %d, want 2", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestZeroUsageSafety(t *testing.T) {
	l := ledger.New()
	if err := l.Deposit("d", "t", 100); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
	resKnown, err := l.Reserve("r1", "t", "req-known", 60)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	resUnknown, err := l.Reserve("r2", "t", "req-unknown", 40)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}

	// Known zero: an explicit settle with zero records a KNOWN settlement.
	if err := l.Settle("s1", resKnown.ID, 0); err != nil {
		t.Fatalf("Settle(0): %v", err)
	}
	got, err := l.GetReservation(resKnown.ID)
	if err != nil {
		t.Fatalf("GetReservation: %v", err)
	}
	if got.Status != ledger.StatusSettled || got.ActualUnits != 0 || got.RefundedUnits != 60 {
		t.Errorf("known-zero settlement = status %s actual %d refunded %d, want settled/0/60",
			got.Status, got.ActualUnits, got.RefundedUnits)
	}

	// Unknown usage: no settle call at all. The reservation reports PENDING —
	// absence, never a fabricated zero settlement.
	gotU, err := l.GetReservation(resUnknown.ID)
	if err != nil {
		t.Fatalf("GetReservation: %v", err)
	}
	if gotU.Status != ledger.StatusPending {
		t.Errorf("unsettled reservation status = %s, want pending", gotU.Status)
	}
	// The known-zero settlement refunded the full 60: available = 0 + 60.
	acc, _ := l.GetAccount("t")
	if acc.Settled != 0 || acc.Available != 60 || acc.Reserved != 40 {
		t.Errorf("account = %+v, want settled=0 available=60 reserved=40", acc)
	}
	if err := l.CheckInvariants(); err != nil {
		t.Errorf("CheckInvariants: %v", err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestConcurrentOperations(t *testing.T) {
	const total = 1_000_000
	l := ledger.New()
	if err := l.Deposit("seed", "t", total); err != nil {
		t.Fatalf("Deposit: %v", err)
	}

	const workers = 8
	const opsPerWorker = 25
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < opsPerWorker; i++ {
				n := strconv.Itoa(w*opsPerWorker + i)
				ev, req := "ev-"+n, "req-"+n
				res, err := l.Reserve(ev+"-r", "t", req, 10)
				if err != nil {
					t.Errorf("worker %d op %d Reserve: %v", w, i, err)
					return
				}
				if err := l.Settle(ev+"-s", res.ID, 5); err != nil {
					t.Errorf("worker %d op %d Settle: %v", w, i, err)
					return
				}
			}
		}(w)
	}
	wg.Wait()

	if err := l.CheckInvariants(); err != nil {
		t.Fatalf("CheckInvariants: %v", err)
	}
	acc, _ := l.GetAccount("t")
	if acc.Available+acc.Reserved+acc.Settled != total {
		t.Fatalf("balance sum = %d, want %d", acc.Available+acc.Reserved+acc.Settled, total)
	}
	if acc.Settled != workers*opsPerWorker*5 {
		t.Fatalf("settled = %d, want %d", acc.Settled, workers*opsPerWorker*5)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUsageRecordIntegration(t *testing.T) {
	// The landed UsageRecord drives the ledger through its identity fields:
	// EventID is the dedup key, TenantID/RequestID the reservation join.
	rec := billing.UsageRecord{
		SchemaVersion:    billing.SchemaVersionV1,
		EventID:          "evt-77",
		RequestID:        "req-77",
		TenantID:         "tenant-77",
		Model:            "model-a",
		RequestedAt:      time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC),
		PromptTokens:     i64p(1200),
		CompletionTokens: i64p(300),
		Classification:   billing.UsageRecorded,
	}
	if err := rec.Validate(); err != nil {
		t.Fatalf("record must validate: %v", err)
	}

	l := ledger.New()
	if err := l.Deposit(rec.EventID+"-fund", rec.TenantID, 1000); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
	res, err := l.Reserve(rec.EventID+"-reserve", rec.TenantID, rec.RequestID, 500)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	// The consumer prices the record itself (the ledger holds no rates) and
	// settles under the record's event id.
	if err := l.Settle(rec.EventID, res.ID, 120); err != nil {
		t.Fatalf("Settle: %v", err)
	}
	// The same record arriving again: its event id is the dedup key.
	if err := l.Settle(rec.EventID, res.ID, 120); !errors.Is(err, ledger.ErrAlreadyApplied) {
		t.Errorf("replayed record: got %v, want ErrAlreadyApplied", err)
	}

	// A zero_known record settles as a known zero; a missing record produces
	// no settlement at all.
	zero := rec
	zero.EventID, zero.RequestID = "evt-78", "req-78"
	zero.Classification = billing.UsageZeroKnown
	zero.PromptTokens, zero.CompletionTokens = i64p(0), i64p(0)
	if err := zero.Validate(); err != nil {
		t.Fatalf("zero record must validate: %v", err)
	}
	resZ, err := l.Reserve(zero.EventID+"-reserve", zero.TenantID, zero.RequestID, 100)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if err := l.Settle(zero.EventID, resZ.ID, 0); err != nil {
		t.Fatalf("Settle(0): %v", err)
	}
	gotZ, _ := l.GetReservation(resZ.ID)
	if gotZ.Status != ledger.StatusSettled || gotZ.ActualUnits != 0 {
		t.Errorf("zero_known settlement = status %s actual %d, want settled/0", gotZ.Status, gotZ.ActualUnits)
	}

	missing := rec
	missing.EventID, missing.RequestID = "evt-79", "req-79"
	missing.Classification = billing.UsageMissing
	missing.PromptTokens, missing.CompletionTokens = nil, nil
	if err := missing.Validate(); err != nil {
		t.Fatalf("missing record must validate: %v", err)
	}
	resM, err := l.Reserve(missing.EventID+"-reserve", missing.TenantID, missing.RequestID, 100)
	if err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	// No Settle for missing usage: the reservation stays PENDING.
	gotM, _ := l.GetReservation(resM.ID)
	if gotM.Status != ledger.StatusPending {
		t.Errorf("missing-usage reservation status = %s, want pending", gotM.Status)
	}
	if err := l.CheckInvariants(); err != nil {
		t.Errorf("CheckInvariants: %v", err)
	}
}
