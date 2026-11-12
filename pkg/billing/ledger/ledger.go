// Package ledger provides fak's neutral ledger mechanics for multi-tenant
// usage accounting: an append-only, replayable log of balance events, and
// the reserve / settle / release state machine built on top of it.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/billing/ledger)
// on purpose. Placement follows Plan C ("off-the-shelf public spine"),
// billing-modularization lane, Phase 1 item 4 (companion private repository,
// docs/tickets/billing-modularization/PLAN-C-off-the-shelf-public-spine.md):
// ledger mechanics — append-only store, replay, dedup-by-event-id,
// reserve/settle/release operations, the reconcile-once invariant, and
// typed errors.
//
// This package is pure MECHANISM. It contains NO pricing and NO policy, and
// no default may encode a billing choice:
//
//   - Units are opaque integers. The package names no currency, holds no
//     rates, and computes no amounts: a consumer decides what one unit is
//     and how many units a usage record costs, and passes the result in.
//   - No funding-source taxonomy and no external-event plumbing: a deposit
//     is a deposit; nothing here asks who funded it or how the units were
//     obtained.
//   - No admission or entitlement decisions: the ledger tracks balances and
//     reservation lifecycles; whether an attempt may proceed is the caller's
//     verdict.
//
// Three invariants shape the surface:
//
//   - Append-only, replay-derived state. Every state change is one entry in
//     the log; in-memory state is derived by applying entries in sequence
//     order. Replaying a written log into a fresh ledger yields identical
//     state; the log is the source of truth.
//   - Dedup by event id. Every operation carries an event id; an id that
//     was already applied is recognized and not re-applied
//     (ErrAlreadyApplied), so retried or re-sent operations are safe. The
//     index is durable: it is rebuilt from the log on every replay, unlike
//     the process-local billing.Seen index in the parent package.
//   - Reconcile-once. A settle that never reached the log can be made up
//     with Reconcile, and the settled-key index (also rebuilt on replay)
//     guarantees a given event is reconciled at most once, across replays.
//
// Unknown is never treated as known. There is no path from "no data" to a
// recorded amount: an outcome with no usage data simply leaves the
// reservation PENDING, and settling with an explicit zero records a KNOWN
// zero (the entry's very existence is the knowledge marker), consistent
// with the parent package's closed usage classifications (recorded /
// missing / zero_known / not_applicable). No default, floor, or estimate
// ever fabricates an amount.
//
// The package has no external dependencies. A Ledger is constructed by its
// owner: New for a volatile ledger, Open for a durable file-backed one.
package ledger

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"
)

// EntryKind is the closed vocabulary of durable log entry kinds.
type EntryKind string

const (
	// KindDeposit adds units to a tenant's available balance.
	KindDeposit EntryKind = "deposit"

	// KindReserve moves units from available to reserved for a request.
	KindReserve EntryKind = "reserve"

	// KindSettle charges actual units against a reservation and refunds the
	// unused remainder. A settle with zero units is a known-zero settlement.
	KindSettle EntryKind = "settle"

	// KindRelease returns a reservation's full reserved units to available.
	KindRelease EntryKind = "release"
)

// Kinds is the closed, ordered kind vocabulary.
var Kinds = [...]EntryKind{
	KindDeposit,
	KindReserve,
	KindSettle,
	KindRelease,
}

// Valid reports whether k is a member of the closed kind vocabulary.
func (k EntryKind) Valid() bool {
	for _, known := range Kinds {
		if k == known {
			return true
		}
	}
	return false
}

// Status is the closed lifecycle vocabulary of a Reservation.
type Status string

const (
	// StatusPending means units are held and the outcome is still open.
	StatusPending Status = "pending"

	// StatusSettled means a known actual quantity was charged and the
	// unused remainder refunded. ActualUnits of zero is a known-zero
	// settlement, not an absence of data.
	StatusSettled Status = "settled"

	// StatusReleased means the held units were returned in full.
	StatusReleased Status = "released"
)

// Statuses is the closed, ordered status vocabulary.
var Statuses = [...]Status{
	StatusPending,
	StatusSettled,
	StatusReleased,
}

// Valid reports whether s is a member of the closed status vocabulary.
func (s Status) Valid() bool {
	for _, known := range Statuses {
		if s == known {
			return true
		}
	}
	return false
}

// Sentinel errors. Argument-shape faults wrap ErrInvalidArgument; durable-log
// faults wrap ErrCorruptLog; the rest name one specific fault. Callers match
// with errors.Is against the specific sentinel (or a root).
var (
	// ErrInvalidArgument is the root sentinel for rejected call arguments.
	ErrInvalidArgument = errors.New("ledger: invalid argument")

	// ErrMissingEventID marks an operation without its dedup key.
	ErrMissingEventID = fmt.Errorf("%w: event id is required", ErrInvalidArgument)

	// ErrEmptyTenantID marks an operation without a tenant.
	ErrEmptyTenantID = fmt.Errorf("%w: tenant id is required", ErrInvalidArgument)

	// ErrEmptyRequestID marks an operation without a request id.
	ErrEmptyRequestID = fmt.Errorf("%w: request id is required", ErrInvalidArgument)

	// ErrEmptyReservationID marks an operation without a reservation id.
	ErrEmptyReservationID = fmt.Errorf("%w: reservation id is required", ErrInvalidArgument)

	// ErrInvalidQuantity marks a quantity that is negative, is zero where a
	// positive value is required, or would overflow int64.
	ErrInvalidQuantity = fmt.Errorf("%w: quantity out of range", ErrInvalidArgument)

	// ErrInsufficientBalance is returned when a tenant's available balance
	// cannot cover a requested reservation.
	ErrInsufficientBalance = errors.New("ledger: insufficient available balance")

	// ErrAlreadySettled is returned when a reservation has already been
	// settled, including by a prior Reconcile.
	ErrAlreadySettled = errors.New("ledger: reservation already settled")

	// ErrAlreadyReleased is returned when a reservation has already been
	// released.
	ErrAlreadyReleased = errors.New("ledger: reservation already released")

	// ErrReservationNotFound is returned when a requested reservation does
	// not exist.
	ErrReservationNotFound = errors.New("ledger: reservation not found")

	// ErrAccountNotFound is returned when an operation names an unknown
	// tenant.
	ErrAccountNotFound = errors.New("ledger: account not found")

	// ErrExceedsReserved is returned when a settled quantity exceeds the
	// reserved maximum.
	ErrExceedsReserved = errors.New("ledger: settled quantity exceeds reserved quantity")

	// ErrDuplicateReservation is returned when a PENDING reservation for the
	// same tenant and request already exists.
	ErrDuplicateReservation = errors.New("ledger: duplicate pending reservation for tenant and request")

	// ErrAlreadyApplied is the dedup verdict: the event id was already
	// applied, so the operation is a no-op. It is not an argument defect, so
	// it deliberately does not wrap ErrInvalidArgument.
	ErrAlreadyApplied = errors.New("ledger: event already applied")

	// ErrInvariantViolated is returned when the accounting invariant fails:
	// Granted != Available + Reserved + Settled, a component is negative, or
	// Reserved disagrees with the sum of pending reservations.
	ErrInvariantViolated = errors.New("ledger: accounting invariant violated")

	// ErrCorruptLog marks a durable log that is not a well-formed log of this
	// format: undecodable lines, fields outside the entry schema, failed
	// entry validation, broken sequence continuity, or entries the state
	// machine refuses.
	ErrCorruptLog = errors.New("ledger: corrupt log")

	// ErrInvalidEntry marks a single entry that violates the entry contract.
	ErrInvalidEntry = errors.New("ledger: invalid entry")
)

// Entry is one line of the durable log: a single applied balance event.
//
// The log is the source of truth. Entries are immutable once written;
// sequence numbers start at 1 and increase by one, so a truncated, reordered,
// or hand-edited log is detectable on replay.
type Entry struct {
	// Seq is the 1-based position of this entry in the log.
	Seq uint64 `json:"seq"`

	// EventID is the dedup key: the identity of the external event this
	// entry was applied for. Required, unique within the log.
	EventID string `json:"event_id"`

	// Time is when the event was applied.
	Time time.Time `json:"time"`

	// Kind selects the entry's meaning. Closed set: see EntryKind.
	Kind EntryKind `json:"kind"`

	// TenantID is the opaque registry identifier of the affected tenant.
	// Opaque: this package neither mints nor interprets it. Required for
	// every kind.
	TenantID string `json:"tenant_id"`

	// RequestID joins the entry to the wire-level request. Required for
	// reserve, settle, and release; optional for deposit.
	RequestID string `json:"request_id,omitempty"`

	// ReservationID names the reservation a reserve creates or a
	// settle/release closes. Required for reserve, settle, and release.
	ReservationID string `json:"reservation_id,omitempty"`

	// Units is the integer quantity the event moves. Positive for deposit
	// and reserve; the known actual charge for settle (zero allowed: a
	// known-zero settlement); always zero for release.
	Units int64 `json:"units"`
}

// Validate checks the closed entry contract: a known kind, the required
// identity fields for that kind, and a quantity consistent with it. On the
// first violated rule it returns a wrapped ErrInvalidEntry.
func (e *Entry) Validate() error {
	if e == nil {
		return ErrInvalidEntry
	}
	if e.EventID == "" {
		return fmt.Errorf("%w: event_id is required", ErrInvalidEntry)
	}
	if e.Time.IsZero() {
		return fmt.Errorf("%w: time is required", ErrInvalidEntry)
	}
	if !e.Kind.Valid() {
		return fmt.Errorf("%w: unknown kind %q", ErrInvalidEntry, e.Kind)
	}
	if e.TenantID == "" {
		return fmt.Errorf("%w: tenant_id is required", ErrInvalidEntry)
	}
	if e.Units < 0 {
		return fmt.Errorf("%w: negative units %d", ErrInvalidEntry, e.Units)
	}
	switch e.Kind {
	case KindDeposit:
		if e.Units == 0 {
			return fmt.Errorf("%w: deposit requires positive units", ErrInvalidEntry)
		}
	case KindReserve:
		if e.RequestID == "" || e.ReservationID == "" {
			return fmt.Errorf("%w: reserve requires request_id and reservation_id", ErrInvalidEntry)
		}
		if e.Units == 0 {
			return fmt.Errorf("%w: reserve requires positive units", ErrInvalidEntry)
		}
	case KindSettle:
		if e.RequestID == "" || e.ReservationID == "" {
			return fmt.Errorf("%w: settle requires request_id and reservation_id", ErrInvalidEntry)
		}
	case KindRelease:
		if e.RequestID == "" || e.ReservationID == "" {
			return fmt.Errorf("%w: release requires request_id and reservation_id", ErrInvalidEntry)
		}
		if e.Units != 0 {
			return fmt.Errorf("%w: release carries no units", ErrInvalidEntry)
		}
	}
	return nil
}

// Account is one tenant's balance state, in integer units. All components
// are non-negative and Granted == Available + Reserved + Settled.
type Account struct {
	// TenantID is the opaque registry identifier of the tenant.
	TenantID string `json:"tenant_id"`

	// Granted is the cumulative units deposited.
	Granted int64 `json:"granted"`

	// Available is the unreserved portion of the balance.
	Available int64 `json:"available"`

	// Reserved is the portion currently held by PENDING reservations.
	Reserved int64 `json:"reserved"`

	// Settled is the cumulative charged units.
	Settled int64 `json:"settled"`
}

// InvariantHolds checks the fundamental accounting balance:
// Granted == Available + Reserved + Settled with all components
// non-negative. The subtraction form keeps an overflowing sum from
// masquerading as a valid balance.
func (a *Account) InvariantHolds() bool {
	if a == nil || a.Granted < 0 || a.Available < 0 ||
		a.Reserved < 0 || a.Settled < 0 {
		return false
	}
	remaining := a.Granted - a.Available
	return remaining >= a.Reserved && remaining-a.Reserved == a.Settled
}

// Snapshot returns a copy of the Account.
func (a *Account) Snapshot() *Account {
	if a == nil {
		return nil
	}
	cp := *a
	return &cp
}

// Reservation records an atomic hold of units for one request.
type Reservation struct {
	// ID is the ledger-minted reservation identifier.
	ID string `json:"id"`

	// TenantID is the opaque registry identifier of the owning tenant.
	TenantID string `json:"tenant_id"`

	// RequestID joins the reservation to the wire-level request.
	RequestID string `json:"request_id"`

	// MaxUnits is the held maximum.
	MaxUnits int64 `json:"max_units"`

	// ActualUnits is the known charge, once settled.
	ActualUnits int64 `json:"actual_units"`

	// RefundedUnits is the portion returned to available: reserved minus
	// actual on settle, the full reservation on release.
	RefundedUnits int64 `json:"refunded_units"`

	// Status is the lifecycle state. Closed set: see Status.
	Status Status `json:"status"`

	// CreatedAt is when the reservation was made.
	CreatedAt time.Time `json:"created_at"`

	// SettledAt is when the reservation settled, when it did.
	SettledAt *time.Time `json:"settled_at,omitempty"`

	// ReleasedAt is when the reservation was released, when it did.
	ReleasedAt *time.Time `json:"released_at,omitempty"`
}

// Snapshot returns a thread-safe point-in-time copy of the Reservation.
func (r *Reservation) Snapshot() *Reservation {
	if r == nil {
		return nil
	}
	cp := *r
	if r.SettledAt != nil {
		t := *r.SettledAt
		cp.SettledAt = &t
	}
	if r.ReleasedAt != nil {
		t := *r.ReleasedAt
		cp.ReleasedAt = &t
	}
	return &cp
}

// Ledger is an append-only, replayable unit ledger with a
// reserve/settle/release state machine on top.
//
// All exported methods are safe for concurrent use. A nil *Ledger fails
// closed with ErrInvalidArgument rather than panicking.
type Ledger struct {
	mu sync.RWMutex

	// path is the durable log location; empty for a volatile ledger.
	path string

	seq          uint64
	accounts     map[string]*Account
	reservations map[string]*Reservation
	byRequestKey map[string]*Reservation // tenantID+"\x00"+requestID -> latest reservation
	entries      []Entry                 // applied entries, in sequence order
	seenEvents   map[string]struct{}     // dedup index: applied event ids
	settledKeys  map[string]struct{}     // reconcile-once index: settled tenant+\x00+request pairs
}

// New returns an empty volatile ledger: operations mutate in-memory state
// only and nothing is persisted. Owners who need durability use Open.
func New() *Ledger {
	return newLedger("")
}

func newLedger(path string) *Ledger {
	return &Ledger{
		path:         path,
		accounts:     make(map[string]*Account),
		reservations: make(map[string]*Reservation),
		byRequestKey: make(map[string]*Reservation),
		seenEvents:   make(map[string]struct{}),
		settledKeys:  make(map[string]struct{}),
	}
}

// Deposit adds units to a tenant's available balance, creating the account
// on first use. It is the only way units enter the ledger.
//
// eventID is the dedup key: calling Deposit again with an id that was
// already applied returns ErrAlreadyApplied without changing state.
func (l *Ledger) Deposit(eventID, tenantID string, units int64) error {
	if l == nil {
		return ErrInvalidArgument
	}
	if strings.TrimSpace(eventID) == "" {
		return ErrMissingEventID
	}
	if strings.TrimSpace(tenantID) == "" {
		return ErrEmptyTenantID
	}
	if units <= 0 {
		return ErrInvalidQuantity
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	e := Entry{
		Seq:      l.seq + 1,
		EventID:  eventID,
		Time:     time.Now().UTC(),
		Kind:     KindDeposit,
		TenantID: tenantID,
		Units:    units,
	}
	return l.commitLocked(&e)
}

// Reserve holds maxUnits for a request, moving them from the tenant's
// available balance to reserved. It returns a PENDING reservation whose ID
// identifies it for Settle and Release.
//
// One PENDING reservation may exist per (tenant, request) pair; a second
// Reserve for the same pair while the first is pending returns
// ErrDuplicateReservation. Reserving the same pair again after the first
// was settled or released is allowed, but request ids must stay unique per
// settled event for a given tenant: the reconcile-once index keeps settled
// pairs marked for the lifetime of the ledger.
func (l *Ledger) Reserve(eventID, tenantID, requestID string, maxUnits int64) (*Reservation, error) {
	if l == nil {
		return nil, ErrInvalidArgument
	}
	if strings.TrimSpace(eventID) == "" {
		return nil, ErrMissingEventID
	}
	if strings.TrimSpace(tenantID) == "" {
		return nil, ErrEmptyTenantID
	}
	if strings.TrimSpace(requestID) == "" {
		return nil, ErrEmptyRequestID
	}
	if maxUnits <= 0 {
		return nil, ErrInvalidQuantity
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	seq := l.seq + 1
	e := Entry{
		Seq:           seq,
		EventID:       eventID,
		Time:          time.Now().UTC(),
		Kind:          KindReserve,
		TenantID:      tenantID,
		RequestID:     requestID,
		ReservationID: reserveID(requestID, seq),
		Units:         maxUnits,
	}
	if err := l.commitLocked(&e); err != nil {
		return nil, err
	}
	return l.reservations[e.ReservationID].Snapshot(), nil
}

// Settle settles a PENDING reservation exactly once: actualUnits are charged
// to the tenant's settled balance and the unused remainder (max minus
// actual) is refunded to available.
//
// Idempotent by event id: a second Settle carrying the same event id is
// recognized as the same event and changes nothing (ErrAlreadyApplied). A
// Settle for a reservation that was settled under a different event id is
// refused with ErrAlreadySettled.
//
// actualUnits must be a known quantity. Settling with zero records a
// known-zero settlement; unknown usage is expressed by NOT calling Settle,
// which leaves the reservation PENDING.
func (l *Ledger) Settle(eventID, reservationID string, actualUnits int64) error {
	if l == nil {
		return ErrInvalidArgument
	}
	if strings.TrimSpace(eventID) == "" {
		return ErrMissingEventID
	}
	if strings.TrimSpace(reservationID) == "" {
		return ErrEmptyReservationID
	}
	if actualUnits < 0 {
		return ErrInvalidQuantity
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	res, ok := l.reservations[reservationID]
	if !ok {
		return ErrReservationNotFound
	}
	e := Entry{
		Seq:           l.seq + 1,
		EventID:       eventID,
		Time:          time.Now().UTC(),
		Kind:          KindSettle,
		TenantID:      res.TenantID,
		RequestID:     res.RequestID,
		ReservationID: res.ID,
		Units:         actualUnits,
	}
	return l.commitLocked(&e)
}

// Release returns a PENDING reservation's full reserved units to the
// tenant's available balance (aborted or cancelled work). The dedup and
// exactly-once rules of Settle apply: a repeated event id is a no-op
// (ErrAlreadyApplied), a terminal reservation is refused
// (ErrAlreadySettled / ErrAlreadyReleased).
func (l *Ledger) Release(eventID, reservationID string) error {
	if l == nil {
		return ErrInvalidArgument
	}
	if strings.TrimSpace(eventID) == "" {
		return ErrMissingEventID
	}
	if strings.TrimSpace(reservationID) == "" {
		return ErrEmptyReservationID
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	res, ok := l.reservations[reservationID]
	if !ok {
		return ErrReservationNotFound
	}
	e := Entry{
		Seq:           l.seq + 1,
		EventID:       eventID,
		Time:          time.Now().UTC(),
		Kind:          KindRelease,
		TenantID:      res.TenantID,
		RequestID:     res.RequestID,
		ReservationID: res.ID,
		Units:         0,
	}
	return l.commitLocked(&e)
}

// Reconcile settles a PENDING reservation identified by (tenantID,
// requestID) with a caller-supplied known quantity, for events whose settle
// never reached the log (a crash or a failed append between the in-memory
// settle and the durable one).
//
// Reconcile-once: the settled-key index is rebuilt from the log on every
// replay, so a pair that was settled — by Settle or by an earlier
// Reconcile — is refused with ErrAlreadySettled forever, across replays. A
// repeated event id is a no-op (ErrAlreadyApplied).
//
// units must be a known quantity: zero settles a known-zero outcome, and a
// missing outcome is expressed by NOT calling Reconcile. A pair with no
// PENDING reservation is refused with ErrReservationNotFound: this package
// never fabricates a charge against a balance.
func (l *Ledger) Reconcile(eventID, tenantID, requestID string, units int64) error {
	if l == nil {
		return ErrInvalidArgument
	}
	if strings.TrimSpace(eventID) == "" {
		return ErrMissingEventID
	}
	if strings.TrimSpace(tenantID) == "" {
		return ErrEmptyTenantID
	}
	if strings.TrimSpace(requestID) == "" {
		return ErrEmptyRequestID
	}
	if units < 0 {
		return ErrInvalidQuantity
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if _, ok := l.seenEvents[eventID]; ok {
		return ErrAlreadyApplied
	}
	key := requestKey(tenantID, requestID)
	if _, done := l.settledKeys[key]; done {
		return ErrAlreadySettled
	}
	res, ok := l.byRequestKey[key]
	if !ok || res.Status != StatusPending {
		return ErrReservationNotFound
	}
	e := Entry{
		Seq:           l.seq + 1,
		EventID:       eventID,
		Time:          time.Now().UTC(),
		Kind:          KindSettle,
		TenantID:      res.TenantID,
		RequestID:     res.RequestID,
		ReservationID: res.ID,
		Units:         units,
	}
	return l.commitLocked(&e)
}

// GetAccount returns a snapshot of the tenant's account.
func (l *Ledger) GetAccount(tenantID string) (*Account, error) {
	if l == nil {
		return nil, ErrInvalidArgument
	}
	if strings.TrimSpace(tenantID) == "" {
		return nil, ErrEmptyTenantID
	}
	l.mu.RLock()
	defer l.mu.RUnlock()

	acct, ok := l.accounts[tenantID]
	if !ok {
		return nil, ErrAccountNotFound
	}
	return acct.Snapshot(), nil
}

// GetReservation returns a snapshot of the reservation by its ID.
func (l *Ledger) GetReservation(reservationID string) (*Reservation, error) {
	if l == nil {
		return nil, ErrInvalidArgument
	}
	if strings.TrimSpace(reservationID) == "" {
		return nil, ErrEmptyReservationID
	}
	l.mu.RLock()
	defer l.mu.RUnlock()

	res, ok := l.reservations[reservationID]
	if !ok {
		return nil, ErrReservationNotFound
	}
	return res.Snapshot(), nil
}

// GetReservationByRequest returns the latest reservation for the (tenant,
// request) pair, in whatever state it is in.
func (l *Ledger) GetReservationByRequest(tenantID, requestID string) (*Reservation, error) {
	if l == nil {
		return nil, ErrInvalidArgument
	}
	if strings.TrimSpace(tenantID) == "" {
		return nil, ErrEmptyTenantID
	}
	if strings.TrimSpace(requestID) == "" {
		return nil, ErrEmptyRequestID
	}
	l.mu.RLock()
	defer l.mu.RUnlock()

	res, ok := l.byRequestKey[requestKey(tenantID, requestID)]
	if !ok {
		return nil, ErrReservationNotFound
	}
	return res.Snapshot(), nil
}

// Entries returns a copy of the applied entries in sequence order.
func (l *Ledger) Entries() []Entry {
	if l == nil {
		return nil
	}
	l.mu.RLock()
	defer l.mu.RUnlock()

	out := make([]Entry, len(l.entries))
	copy(out, l.entries)
	return out
}

// CheckInvariants verifies the accounting invariants across every account:
// non-negative components, Granted == Available + Reserved + Settled, and
// Reserved equal to the sum of that tenant's PENDING reservations.
func (l *Ledger) CheckInvariants() error {
	if l == nil {
		return ErrInvalidArgument
	}
	l.mu.RLock()
	defer l.mu.RUnlock()

	return l.invariantsLocked()
}

// commitLocked runs the live-operation pipeline for one fully populated
// entry (including its Seq): dedup check, semantic check, durable append,
// state mutation. It is the only live mutation path; replay shares its two
// halves (checkLocked, mutateLocked), which is what makes replay of a
// written log yield identical state. Callers hold mu.
func (l *Ledger) commitLocked(e *Entry) error {
	if _, ok := l.seenEvents[e.EventID]; ok {
		return ErrAlreadyApplied
	}
	if err := l.checkLocked(*e); err != nil {
		return err
	}
	if err := l.appendLine(*e); err != nil {
		return err
	}
	l.mutateLocked(*e)
	return nil
}

// checkLocked verifies that e can be applied to the current state. It
// mutates nothing. Callers hold mu.
func (l *Ledger) checkLocked(e Entry) error {
	switch e.Kind {
	case KindDeposit:
		if acct, ok := l.accounts[e.TenantID]; ok &&
			e.Units > math.MaxInt64-acct.Granted {
			return fmt.Errorf("%w: cumulative deposit would overflow", ErrInvalidQuantity)
		}
	case KindReserve:
		acct, ok := l.accounts[e.TenantID]
		if !ok || acct.Available < e.Units {
			return ErrInsufficientBalance
		}
		if res, exists := l.byRequestKey[requestKey(e.TenantID, e.RequestID)]; exists &&
			res.Status == StatusPending {
			return ErrDuplicateReservation
		}
	case KindSettle, KindRelease:
		res, ok := l.reservations[e.ReservationID]
		if !ok {
			return ErrReservationNotFound
		}
		switch res.Status {
		case StatusSettled:
			return ErrAlreadySettled
		case StatusReleased:
			return ErrAlreadyReleased
		}
		if res.TenantID != e.TenantID || res.RequestID != e.RequestID {
			return fmt.Errorf("%w: entry does not match reservation %s",
				ErrInvalidEntry, e.ReservationID)
		}
		if e.Kind == KindSettle && e.Units > res.MaxUnits {
			return ErrExceedsReserved
		}
	}
	return nil
}

// mutateLocked applies e to in-memory state. It assumes checkLocked passed
// and therefore cannot fail: every move preserves
// Granted == Available + Reserved + Settled with non-negative components.
// Callers hold mu.
func (l *Ledger) mutateLocked(e Entry) {
	switch e.Kind {
	case KindDeposit:
		acct := l.accountLocked(e.TenantID)
		acct.Granted += e.Units
		acct.Available += e.Units
	case KindReserve:
		acct := l.accountLocked(e.TenantID)
		acct.Available -= e.Units
		acct.Reserved += e.Units
		res := &Reservation{
			ID:        e.ReservationID,
			TenantID:  e.TenantID,
			RequestID: e.RequestID,
			MaxUnits:  e.Units,
			Status:    StatusPending,
			CreatedAt: e.Time,
		}
		l.reservations[res.ID] = res
		l.byRequestKey[requestKey(e.TenantID, e.RequestID)] = res
	case KindSettle:
		res := l.reservations[e.ReservationID]
		acct := l.accountLocked(res.TenantID)
		acct.Reserved -= res.MaxUnits
		acct.Settled += e.Units
		acct.Available += res.MaxUnits - e.Units
		res.ActualUnits = e.Units
		res.RefundedUnits = res.MaxUnits - e.Units
		res.Status = StatusSettled
		now := e.Time
		res.SettledAt = &now
		l.settledKeys[requestKey(res.TenantID, res.RequestID)] = struct{}{}
	case KindRelease:
		res := l.reservations[e.ReservationID]
		acct := l.accountLocked(res.TenantID)
		acct.Reserved -= res.MaxUnits
		acct.Available += res.MaxUnits
		res.RefundedUnits = res.MaxUnits
		res.Status = StatusReleased
		now := e.Time
		res.ReleasedAt = &now
	}

	l.entries = append(l.entries, e)
	l.seenEvents[e.EventID] = struct{}{}
	l.seq = e.Seq
}

// invariantsLocked recomputes the accounting invariants from scratch.
// Callers hold mu (in any mode).
func (l *Ledger) invariantsLocked() error {
	for _, acct := range l.accounts {
		if !acct.InvariantHolds() {
			return fmt.Errorf("%w: tenant %s: granted=%d != available=%d + reserved=%d + settled=%d",
				ErrInvariantViolated, acct.TenantID, acct.Granted, acct.Available, acct.Reserved, acct.Settled)
		}
	}
	for tid, acct := range l.accounts {
		var pending int64
		for _, res := range l.reservations {
			if res.TenantID == tid && res.Status == StatusPending {
				pending += res.MaxUnits
			}
		}
		if pending != acct.Reserved {
			return fmt.Errorf("%w: tenant %s: reserved=%d != pending reservations=%d",
				ErrInvariantViolated, tid, acct.Reserved, pending)
		}
	}
	return nil
}

// accountLocked returns the tenant's account, creating an empty one on
// first use. Callers hold mu.
func (l *Ledger) accountLocked(tenantID string) *Account {
	acct, ok := l.accounts[tenantID]
	if !ok {
		acct = &Account{TenantID: tenantID}
		l.accounts[tenantID] = acct
	}
	return acct
}

// requestKey composes the (tenant, request) pair into one map key.
func requestKey(tenantID, requestID string) string {
	return tenantID + "\x00" + requestID
}

// reserveID mints a deterministic reservation identifier. The id is persisted
// in the reserve entry, so replay restores it exactly regardless of scheme.
func reserveID(requestID string, seq uint64) string {
	return "res-" + requestID + "-" + strconv.FormatUint(seq, 10)
}
