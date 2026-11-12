package billing

// policy.go — the billing authority seam (Plan C §3 "Alpha seam" row): the
// closed shape of every decision a serving boundary needs from whoever
// bills, plus the off-the-shelf neutral default that implements it.
//
// Design source: Plan C ("off-the-shelf public spine"), companion private
// repository, docs/tickets/billing-modularization/PLAN-C-off-the-shelf-
// public-spine.md — §3 split table ("Admission" and "Alpha seam" rows),
// Phase 1 items 6–7, Phase 2 item 8. Placement follows BOUNDARY.md: the
// mechanism half of the billing port is public; the policy half stays in
// each consuming authority.
//
// WHAT LIVES HERE, AND WHY:
//
//   - The Policy interface and its closed vocabularies (Verdict,
//     PricingMode) are MECHANISM: they fix the shape of the four decisions
//     (admit / price / settle / reconcile) so any authority — this
//     package's neutral default, an external billing authority, or an out-of-process
//     sink adapter — plugs into the same seam.
//   - Every VALUE stays out of this file. Which principals are admitted,
//     what anything costs, which ceilings exist, and how reconciliation
//     joins are all operator or authority choices supplied at
//     construction. Nothing here defaults to a rate, a threshold, or a
//     reservation formula: a zero ceiling means unlimited, and the neutral
//     default prices everything no-charge BY POLICY (explicitly typed),
//     never as an unknown coerced to zero.
//   - SpendCaps ships as a neutral in-memory cap implementation rather
//     than a bare injection slot because this package cannot import the
//     internal gateway's control-plane governor, and the shelf product
//     owes a fresh clone a working admission story out of the box:
//     no-charge pricing plus operator-configured local spend caps. The cap
//     state is pure bookkeeping over reported tokens; the ceilings
//     themselves arrive as CapConfig arguments, so shipping the mechanism
//     discloses no value.
//
// The balance-based refusal TYPES (VerdictRejectBudget -> HTTP 402,
// VerdictRejectRate -> HTTP 429) are defined here as a closed enum; the
// balance and rate VALUES that trigger them live in each authority.

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// Sentinel errors for the policy seam and the engine.
var (
	// ErrNoPolicy is returned when an Engine is built without a Policy, or
	// called through a zero-value Engine: an engine with no authority
	// cannot decide anything and fails closed.
	ErrNoPolicy = errors.New("billing: no policy injected")

	// ErrSinkForward wraps a record-sink submission failure during
	// Engine.Complete: the record was settled, but forwarding did not
	// complete.
	ErrSinkForward = errors.New("billing: record sink forward failed")

	// ErrInvalidPrice is the root sentinel for every Price that violates
	// the closed amount/mode contract.
	ErrInvalidPrice = errors.New("billing: invalid price")

	// ErrUnknownPricingMode marks a Price whose mode is not a member of
	// the closed pricing vocabulary.
	ErrUnknownPricingMode = fmt.Errorf("%w: unknown pricing mode", ErrInvalidPrice)

	// ErrInconsistentPrice marks a Price whose amount contradicts its
	// mode (for example a no-charge price carrying a nonzero amount).
	ErrInconsistentPrice = fmt.Errorf("%w: amount inconsistent with pricing mode", ErrInvalidPrice)
)

// Verdict is the closed set of admission outcomes a Policy can return. It
// is a TYPE only: which verdict applies, and at what balance or rate, is a
// decision made by each consuming authority. This package defines the
// shape, not the values.
type Verdict string

const (
	// VerdictAdmit means the attempt may proceed.
	VerdictAdmit Verdict = "admit"

	// VerdictRejectBudget means the attempt is refused because the
	// principal's budget or balance is exhausted. Authorities map this to
	// HTTP 402 (Payment Required).
	VerdictRejectBudget Verdict = "reject_budget"

	// VerdictRejectRate means the attempt is refused because a rate cap
	// (requests-per-window or tokens-per-window) is saturated. Authorities
	// map this to HTTP 429 (Too Many Requests).
	VerdictRejectRate Verdict = "reject_rate"
)

// Verdicts is the closed, ordered verdict vocabulary.
var Verdicts = [...]Verdict{
	VerdictAdmit,
	VerdictRejectBudget,
	VerdictRejectRate,
}

// Valid reports whether v is a member of the closed verdict vocabulary.
func (v Verdict) Valid() bool {
	for _, known := range Verdicts {
		if v == known {
			return true
		}
	}
	return false
}

// PricingMode is the closed set of ways a Price establishes its amount. It
// keeps "zero by policy" distinct from "no price could be established":
// the two must never be confused, because collapsing unpriced into zero
// makes unknown usage indistinguishable from settled usage downstream.
type PricingMode string

const (
	// PricingNoCharge means the amount is zero BY POLICY: the authority
	// charges nothing. The amount is known, and it is zero.
	PricingNoCharge PricingMode = "no_charge"

	// PricingCharged means the amount was computed from a price table.
	// AmountMicroUSD carries the charge.
	PricingCharged PricingMode = "charged"

	// PricingUnpriced means no price could be established (for example the
	// model has no entry in the table in force). The amount is UNKNOWN,
	// not zero: a consumer must carry it forward as unknown rather than
	// substitute a computed value.
	PricingUnpriced PricingMode = "unpriced"
)

// PricingModes is the closed, ordered pricing-mode vocabulary.
var PricingModes = [...]PricingMode{
	PricingNoCharge,
	PricingCharged,
	PricingUnpriced,
}

// Valid reports whether m is a member of the closed pricing-mode
// vocabulary.
func (m PricingMode) Valid() bool {
	for _, known := range PricingModes {
		if m == known {
			return true
		}
	}
	return false
}

// Price is the outcome of pricing one usage record. Its Mode fixes what
// the amount means; Validate enforces the closed contract between the two.
type Price struct {
	// Mode is a member of the closed PricingMode vocabulary.
	Mode PricingMode

	// AmountMicroUSD is the charge in micro-USD (1e-6 USD). It is
	// meaningful only when Mode is PricingCharged; it must be zero for
	// every other mode.
	AmountMicroUSD int64

	// PriceBookVersion stamps the price table in force when the price was
	// established (a version stamp only; this package never stores or
	// computes rates). Empty when no table was consulted.
	PriceBookVersion string
}

// Validate checks the closed amount/mode contract. On the first violated
// rule it returns a wrapped sentinel error, matchable with errors.Is
// against the specific sentinel or against ErrInvalidPrice.
func (p Price) Validate() error {
	if !p.Mode.Valid() {
		return fmt.Errorf("%w: %q", ErrUnknownPricingMode, p.Mode)
	}
	switch p.Mode {
	case PricingCharged:
		if p.AmountMicroUSD < 0 {
			return fmt.Errorf("%w: charged amount %d is negative", ErrInconsistentPrice, p.AmountMicroUSD)
		}
	case PricingNoCharge, PricingUnpriced:
		if p.AmountMicroUSD != 0 {
			return fmt.Errorf("%w: mode %q carries amount %d", ErrInconsistentPrice, p.Mode, p.AmountMicroUSD)
		}
	}
	return nil
}

// PriceTable resolves model names to prices. It is the minimal seam a
// non-neutral Authority uses to bring its own rates onto the spine: the
// spine defines the SHAPE of the lookup, and every rate value stays in the
// implementing table. When the price-book subpackage lands (Plan C Phase 1
// item 5) its built table satisfies this interface directly or through a
// thin adapter; until then authorities implement it privately.
type PriceTable interface {
	// Lookup returns the per-request amount in micro-USD for model, the
	// version stamp of the table in force, and whether the model is
	// priced. ok=false marks UNPRICED: the caller must treat the amount
	// as unknown, never as zero.
	Lookup(model string) (amountMicroUSD int64, version string, ok bool)
}

// Principal names who an admission decision and a spend cap bound. Both
// fields are opaque registry identifiers: this package neither mints nor
// interprets them. An empty field means "not attributed to that dimension";
// a cap whose dimension is absent is simply not evaluated there.
type Principal struct {
	TenantID  string
	SessionID string
}

// AdmissionContext carries everything Admit needs to decide: who is asking
// and when. RequestedAt places the attempt in time so windowed caps (daily)
// attribute it to the right window; a zero time contributes to no window.
type AdmissionContext struct {
	Principal   Principal
	RequestedAt time.Time
}

// CapAction is the closed set of actions a crossed ceiling takes. Pause is
// reversible (an operator may reset the principal back under the ceiling);
// kill is terminal.
type CapAction string

const (
	// CapActionPause drains the principal and leaves it resumable by an
	// operator.
	CapActionPause CapAction = "pause"

	// CapActionKill is terminal — the principal is stopped for good.
	CapActionKill CapAction = "kill"
)

// CapActions is the closed, ordered cap-action vocabulary.
var CapActions = [...]CapAction{
	CapActionPause,
	CapActionKill,
}

// Valid reports whether a is a member of the closed cap-action vocabulary.
func (a CapAction) Valid() bool {
	for _, known := range CapActions {
		if a == known {
			return true
		}
	}
	return false
}

// CapConfig declares the ceilings a SpendCaps enforces. A zero ceiling on
// an axis means that axis is unlimited; a config with both ceilings zero
// never breaches. Ceilings are OPERATOR CHOICES supplied at construction:
// this package ships none.
type CapConfig struct {
	// DailyTokenCap is the cumulative token ceiling per tenant per UTC
	// calendar day. 0 = unlimited.
	DailyTokenCap int64

	// SessionTokenCap is the cumulative token ceiling per (tenant,
	// session), for the life of the session identifier. 0 = unlimited.
	SessionTokenCap int64

	// Action is the closed action taken on breach. Empty defaults to
	// CapActionPause: pause is the safe direction (reversible), while a
	// kill is terminal and must be chosen explicitly.
	Action CapAction
}

func (c CapConfig) action() CapAction {
	if c.Action == "" {
		return CapActionPause
	}
	return c.Action
}

// SpendCaps is the neutral in-memory spend-cap mechanism: it accumulates
// REPORTED token usage per principal and reports when an operator-
// configured ceiling is met or crossed. It is pure bookkeeping — no money,
// no rates, no policy beyond the ceilings it is constructed with — and it
// is the enforcement arm of the neutral default policy.
//
// Two windows are tracked, each only when its ceiling is configured:
//
//   - daily: total tokens per tenant, reset on each UTC calendar-day
//     change. Spend is attributed by the timestamp the caller supplies, so
//     replaying history lands in the day it happened, not in wall clock.
//   - session: total tokens per (tenant, session), cumulative for the life
//     of the identifier.
//
// Safe for concurrent use. A nil receiver is inert (never breaches).
type SpendCaps struct {
	mu      sync.Mutex
	cfg     CapConfig
	daily   map[string]*capDayCounter
	session map[string]int64
}

// capDayCounter is one tenant's daily accumulator stamped with the UTC day
// it belongs to.
type capDayCounter struct {
	day    string
	tokens int64
}

// NewSpendCaps builds cap state enforcing the given ceilings.
func NewSpendCaps(cfg CapConfig) *SpendCaps {
	return &SpendCaps{
		cfg:     cfg,
		daily:   make(map[string]*capDayCounter),
		session: make(map[string]int64),
	}
}

// Add folds reported tokens into the principal's windows at the given time.
// A nil receiver, a non-positive token count, or an unconfigured ceiling is
// a no-op: the caps track exactly what they enforce.
func (c *SpendCaps) Add(p Principal, tokens int64, at time.Time) {
	if c == nil || tokens <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if p.TenantID != "" && c.cfg.DailyTokenCap > 0 {
		day := utcDay(at)
		d := c.daily[p.TenantID]
		if d == nil || d.day != day {
			d = &capDayCounter{day: day}
			c.daily[p.TenantID] = d
		}
		d.tokens += tokens
	}
	if p.TenantID != "" && p.SessionID != "" && c.cfg.SessionTokenCap > 0 {
		c.session[sessionKey(p)] += tokens
	}
}

// Breached reports whether the principal meets or crosses any configured
// ceiling as of the given time. It is a pure read: it neither charges nor
// mutates state, so a served boundary can call it repeatedly to keep
// refusing a breached principal idempotently.
func (c *SpendCaps) Breached(p Principal, at time.Time) bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cfg.DailyTokenCap > 0 && c.dailyTokensLocked(p, at) >= c.cfg.DailyTokenCap {
		return true
	}
	if c.cfg.SessionTokenCap > 0 && p.TenantID != "" && p.SessionID != "" &&
		c.session[sessionKey(p)] >= c.cfg.SessionTokenCap {
		return true
	}
	return false
}

// Spent returns the principal's current daily and session token totals as
// of the given time. A zero receiver yields zeros.
func (c *SpendCaps) Spent(p Principal, at time.Time) (daily, session int64) {
	if c == nil {
		return 0, 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	daily = c.dailyTokensLocked(p, at)
	if p.TenantID != "" && p.SessionID != "" {
		session = c.session[sessionKey(p)]
	}
	return daily, session
}

// Action reports the resolved breach action: the configured action, or
// CapActionPause when none was set.
func (c *SpendCaps) Action() CapAction {
	if c == nil {
		return CapActionPause
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cfg.action()
}

// dailyTokensLocked reads the principal's daily total for the day of at,
// ignoring stale counters from earlier days. Callers hold c.mu.
func (c *SpendCaps) dailyTokensLocked(p Principal, at time.Time) int64 {
	if p.TenantID == "" {
		return 0
	}
	d := c.daily[p.TenantID]
	if d == nil || d.day != utcDay(at) {
		return 0
	}
	return d.tokens
}

// sessionKey renders the (tenant, session) pair as one map key. The NUL
// separator cannot appear in a registry identifier, so the encoding is
// injective.
func sessionKey(p Principal) string {
	return p.TenantID + "\x00" + p.SessionID
}

// utcDay renders the UTC calendar day of t as "2006-01-02". A zero time
// renders as "0001-01-01": no real usage lands there, so a zero timestamp
// simply contributes to no daily window.
func utcDay(t time.Time) string {
	return t.UTC().Format("2006-01-02")
}

// Settlement is the durable outcome of settling one usage record: the
// identity of the settled event, the model it billed, the classification
// lineage carried by the record, the price that was applied, and when the
// settlement happened. It is append-only data, not a balance: balances and
// ledgers live in the consuming authority.
type Settlement struct {
	EventID        string
	Model          string
	Classification UsageClassification
	Price          Price
	SettledAt      time.Time
}

// ReconcileScope bounds one reconciliation pass: which window of settled
// usage to check, optionally narrowed to one tenant. Zero times select the
// whole available history.
type ReconcileScope struct {
	Since    time.Time
	Until    time.Time
	TenantID string
}

// Reconciliation is the closed report of a reconciliation pass: how much
// settled usage agreed with the independent source, how much did not, and
// how much the source carries that never settled. All counts are
// non-negative; a zero value reports "nothing examined, nothing
// outstanding".
type Reconciliation struct {
	Matched  int64
	Disputed int64
	Missing  int64
}

// Policy is the billing authority seam: the four decisions a serving
// boundary needs from whoever bills. The spine defines the SHAPE of each
// decision; every VALUE — which attempts are admitted, what they cost, how
// settlements are recorded, what reconciliation joins — belongs to the
// implementing authority. NeutralPolicy implements the interface as
// no-charge plus local spend caps; external authorities implement the
// same interface (in-process) or attach out-of-process against the record
// schema.
type Policy interface {
	// Admit decides, before an attempt is served, whether it may proceed.
	// It returns a member of the closed Verdict vocabulary.
	Admit(ctx AdmissionContext) Verdict

	// Price establishes the cost of a recorded attempt. It returns a Price
	// whose Mode is a member of the closed PricingMode vocabulary; an
	// amount that cannot be established must come back unpriced, never as
	// a zero.
	Price(rec UsageRecord) Price

	// Settle records the final outcome of an attempt: it folds the record
	// into the authority's durable state (caps, ledgers) and returns the
	// settlement line. Settle is idempotent per event ID: settling the
	// same record twice returns the original settlement without charging
	// twice.
	Settle(rec UsageRecord, p Price) (Settlement, error)

	// Reconcile compares settled usage against an independent source of
	// truth within the given scope and reports the closed discrepancy
	// counts.
	Reconcile(scope ReconcileScope) (Reconciliation, error)
}

// NeutralPolicy is the off-the-shelf default Policy: no-charge pricing
// plus local spend caps. Every record is priced as zero-cost-BY-POLICY
// (PricingNoCharge, explicitly typed — never an unknown coerced to zero),
// and per-principal token ceilings configured by the operator are enforced
// locally. It encodes no rate, threshold, or billing choice: ceilings
// arrive as CapConfig arguments, and an unset ceiling is unlimited.
//
// A nil *NeutralPolicy is not usable.
type NeutralPolicy struct {
	mu      sync.Mutex
	caps    *SpendCaps
	settled map[string]Settlement
}

// NewNeutral builds the neutral default policy around fresh cap state with
// the given operator ceilings. A zero cfg means no ceilings: pure
// no-charge, admit-all.
func NewNeutral(cfg CapConfig) *NeutralPolicy {
	return NewNeutralWithCaps(NewSpendCaps(cfg))
}

// NewNeutralWithCaps builds the neutral default policy around existing cap
// state, so several authorities (or an external observer) can share one
// SpendCaps instance. A nil caps is treated as uncapped.
func NewNeutralWithCaps(caps *SpendCaps) *NeutralPolicy {
	if caps == nil {
		caps = NewSpendCaps(CapConfig{})
	}
	return &NeutralPolicy{caps: caps, settled: make(map[string]Settlement)}
}

// Caps exposes the cap state this policy enforces, for observation.
func (n *NeutralPolicy) Caps() *SpendCaps {
	return n.caps
}

// Admit refuses with VerdictRejectBudget when the principal has met or
// crossed a configured ceiling as of the request time; otherwise it
// admits. The neutral default performs no rate limiting, so
// VerdictRejectRate is never produced by it.
func (n *NeutralPolicy) Admit(ctx AdmissionContext) Verdict {
	if n.caps.Breached(ctx.Principal, ctx.RequestedAt) {
		return VerdictRejectBudget
	}
	return VerdictAdmit
}

// Price returns the no-charge price for every record, carrying the record's
// price-book version pointer through unchanged. Unknown usage (a missing
// classification) is still no-charge: the authority charges nothing, full
// stop — it does not guess.
func (n *NeutralPolicy) Price(rec UsageRecord) Price {
	return Price{
		Mode:             PricingNoCharge,
		AmountMicroUSD:   0,
		PriceBookVersion: rec.PriceBookVersion,
	}
}

// Settle validates the record and the price, folds the record's REPORTED
// tokens into the caps once per event ID, and returns the settlement line.
// Settling the same event ID again returns the original settlement without
// charging again.
func (n *NeutralPolicy) Settle(rec UsageRecord, p Price) (Settlement, error) {
	if err := rec.Validate(); err != nil {
		return Settlement{}, err
	}
	if err := p.Validate(); err != nil {
		return Settlement{}, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if s, ok := n.settled[rec.EventID]; ok {
		return s, nil
	}
	n.caps.Add(PrincipalFromRecord(rec), reportedTokens(rec), rec.RequestedAt)
	s := Settlement{
		EventID:        rec.EventID,
		Model:          rec.Model,
		Classification: rec.Classification,
		Price:          p,
		SettledAt:      time.Now().UTC(),
	}
	n.settled[rec.EventID] = s
	return s, nil
}

// Reconcile performs no provider join: the neutral default has no
// independent source of truth to reconcile against, so it reports nothing
// outstanding. Authorities that bill against provider exports override this
// with a real join.
func (n *NeutralPolicy) Reconcile(scope ReconcileScope) (Reconciliation, error) {
	return Reconciliation{}, nil
}

// PrincipalFromRecord derives the neutral cap identity from a record.
// UsageRecord v1 carries no dedicated session field, so the neutral default
// keys the session window by the record's KeyID and the daily window by its
// TenantID. An authority that tracks true sessions should build its own
// Principal and drive its own caps.
func PrincipalFromRecord(rec UsageRecord) Principal {
	return Principal{TenantID: rec.TenantID, SessionID: rec.KeyID}
}

// reportedTokens sums the token fields a record REPORTED. Fields left nil
// (unknown) contribute nothing: the caps account for measured usage only,
// and estimating unknown usage is a reservation decision this package does
// not make.
func reportedTokens(rec UsageRecord) int64 {
	var t int64
	for _, f := range []*int64{rec.PromptTokens, rec.CompletionTokens, rec.CachedTokens} {
		if f != nil && *f > 0 {
			t += *f
		}
	}
	return t
}

// RecordSink is the destination for usage records after they are settled.
// Implementations may append to a file, forward to a broker, or buffer in
// memory; the spine only requires that Submit accept one record and report
// a failure. Out-of-process sinks (file, Kafka, socket) attach here so
// non-Go authorities join the spine without a compile dependency.
type RecordSink interface {
	Submit(rec UsageRecord) error
}

// Engine wires one Policy to one RecordSink and drives the per-attempt
// lifecycle around it. It holds no policy of its own and no state: every
// decision is delegated to the injected Policy, and every record is
// forwarded to the injected sink. A nil sink disables forwarding (settle
// still happens), which keeps the engine usable where records stay local.
//
// A zero-value Engine is not usable: it fails closed (refuses admission,
// prices unpriced, errors on settle).
type Engine struct {
	policy Policy
	sink   RecordSink
}

// New wires a Policy to a RecordSink. It refuses a nil policy (an engine
// with no authority fails closed); a nil sink is accepted and disables
// forwarding.
func New(policy Policy, sink RecordSink) (*Engine, error) {
	if policy == nil {
		return nil, ErrNoPolicy
	}
	return &Engine{policy: policy, sink: sink}, nil
}

// Policy exposes the injected policy.
func (e *Engine) Policy() Policy {
	if e == nil {
		return nil
	}
	return e.policy
}

// Admit asks the policy whether an attempt may proceed. A zero-value engine
// refuses.
func (e *Engine) Admit(ctx AdmissionContext) Verdict {
	if e == nil || e.policy == nil {
		return VerdictRejectBudget
	}
	return e.policy.Admit(ctx)
}

// Price asks the policy for the cost of a recorded attempt. A zero-value
// engine prices unpriced (unknown, never zero).
func (e *Engine) Price(rec UsageRecord) Price {
	if e == nil || e.policy == nil {
		return Price{Mode: PricingUnpriced}
	}
	return e.policy.Price(rec)
}

// Settle prices the record through the policy and settles it, returning
// the settlement line. A zero-value engine errors with ErrNoPolicy.
func (e *Engine) Settle(rec UsageRecord) (Settlement, error) {
	if e == nil || e.policy == nil {
		return Settlement{}, ErrNoPolicy
	}
	p := e.policy.Price(rec)
	return e.policy.Settle(rec, p)
}

// Submit forwards one record to the sink. With no sink attached it is a
// no-op success: forwarding is disabled, not failed.
func (e *Engine) Submit(rec UsageRecord) error {
	if e == nil || e.sink == nil {
		return nil
	}
	return e.sink.Submit(rec)
}

// Complete runs the post-attempt pipeline for one record: price, settle,
// then forward to the sink. It returns the settlement even when forwarding
// fails, because the record is already durably settled; the error (wrapping
// ErrSinkForward) tells the caller forwarding did not complete. A settle
// failure short-circuits: the sink is not called.
func (e *Engine) Complete(rec UsageRecord) (Settlement, error) {
	s, err := e.Settle(rec)
	if err != nil {
		return Settlement{}, err
	}
	if e.sink != nil {
		if err := e.sink.Submit(rec); err != nil {
			return s, fmt.Errorf("%w: %v", ErrSinkForward, err)
		}
	}
	return s, nil
}
