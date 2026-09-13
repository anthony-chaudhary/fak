package localadmission

// envelope.go — the complete enforced resident session/cache envelope for a
// native serve. This lives in the local-admission package rather than
// internal/compute so the reservation arithmetic stays with the reservation
// store that owns it, and (operationally) so a change here is not classified as
// GPU-kernel work requiring the Linux-only Strix hardware authority.

// MemClass names a resident memory class. The string values match the class
// labels persisted on reservations (see Reservation.Classes), so the ledger
// stays stable across this package boundary.
type MemClass string

const (
	MemClassWeights    MemClass = "weights"
	MemClassKVCache    MemClass = "kv_cache"
	MemClassActivation MemClass = "activation"
)

// EnvelopeDemand is one classed byte demand in an envelope plan.
type EnvelopeDemand struct {
	Class  MemClass `json:"class"`
	Bytes  int64    `json:"bytes"`
	Detail string   `json:"detail,omitempty"`
}

// EnvelopePlan is a classed memory plan: the reservation-side counterpart of the
// model load plan, sized from a token/sequence/cache envelope rather than a
// single weights figure.
type EnvelopePlan []EnvelopeDemand

// Total is the sum of all demands.
func (p EnvelopePlan) Total() int64 {
	var total int64
	for _, d := range p {
		total = saturatingAddInt64(total, d.Bytes)
	}
	return total
}

// ByClass folds the plan into per-class totals. Zero and negative demands are
// ignored so a class appears only when it actually consumes bytes.
func (p EnvelopePlan) ByClass() map[MemClass]int64 {
	out := map[MemClass]int64{}
	for _, d := range p {
		if d.Bytes <= 0 {
			continue
		}
		out[d.Class] = saturatingAddInt64(out[d.Class], d.Bytes)
	}
	return out
}

// SessionResidencyPlan is the complete enforced resident footprint for a native
// serve: weights + aggregate per-session KV/recurrent state for the enforced
// simultaneous-session bound + per-cohort scratch. It preserves the class
// breakdown so admission can refuse before a loader when weights fit but the
// full enforced state envelope does not.
//
// A weight-only reservation under-counts the real footprint: once the enforced
// MaxSessions bound is reached, every session's KV/recurrent state is resident
// simultaneously, so a plan that only accounted for weights would admit a serve
// that then OOMs on the first concurrent decode. This envelope makes the
// aggregate state demand explicit and classed.
type SessionResidencyPlan struct {
	Weights       EnvelopePlan // weight-class demands (from the model load plan)
	PerSession    EnvelopePlan // KV + session state + scratch for ONE session at the enforced token budget
	MaxSessions   int          // enforced simultaneous-session bound (>=1)
	CohortScratch EnvelopePlan // extra scratch when a decode cohort of size>1 runs (>=0)
}

// effectiveMaxSessions floors the enforced concurrency bound at 1. A zero or
// negative MaxSessions is treated as 1 rather than 0 so the envelope can never
// silently drop the per-session state a single running session still needs; it
// is a fail-safe floor, not a validation error.
func (p SessionResidencyPlan) effectiveMaxSessions() int {
	if p.MaxSessions < 1 {
		return 1
	}
	return p.MaxSessions
}

// StartupPeakBytes is the transient peak during load: weights total + one
// session's startup state + cohort scratch, floored at SteadyBytes.
//
// The transient load window holds exactly one session (the serve that is
// loading) plus any in-flight cohort scratch, so the *base* peak adds a single
// PerSession rather than the full MaxSessions aggregate. But the enforced
// MaxSessions resident state is part of the serve's footprint from the moment
// the session bound is reached, and reservation.go rejects any plan where
// steady > peak. The startup peak is therefore the larger of the one-session
// load window and the full steady envelope, so the >= Steady invariant holds
// for every MaxSessions (base startup and steady coincide once MaxSessions >= 1
// makes the aggregate dominate).
func (p SessionResidencyPlan) StartupPeakBytes() int64 {
	base := saturatingAddInt64(
		saturatingAddInt64(p.Weights.Total(), p.PerSession.Total()),
		p.CohortScratch.Total(),
	)
	if steady := p.SteadyBytes(); steady > base {
		return steady
	}
	return base
}

// SteadyBytes is the resident footprint after load: weights + per-session state
// for the full enforced MaxSessions bound + cohort scratch. The aggregate term
// multiplies the single-session state by MaxSessions; saturation keeps the
// result monotone on overflow instead of wrapping negative.
func (p SessionResidencyPlan) SteadyBytes() int64 {
	perSession := saturatingMulInt64(p.PerSession.Total(), int64(p.effectiveMaxSessions()))
	return saturatingAddInt64(
		saturatingAddInt64(p.Weights.Total(), perSession),
		p.CohortScratch.Total(),
	)
}

// Classes returns the total bytes per memory class across the whole envelope.
// The per-session plan is counted at the full enforced MaxSessions bound (the
// same aggregate SteadyBytes uses) and cohort scratch is added once.
func (p SessionResidencyPlan) Classes() map[MemClass]int64 {
	out := map[MemClass]int64{}
	addEnvelopeClasses(out, p.Weights, 1)
	addEnvelopeClasses(out, p.PerSession, int64(p.effectiveMaxSessions()))
	addEnvelopeClasses(out, p.CohortScratch, 1)
	return out
}

// NewSessionResidencyPlan builds the envelope from a model load plan and a
// per-context state plan. maxSessions<=0 -> 1. cohort>1 adds cohortScratch.
//
// The cohort scratch is a copy of perContext: a decode cohort that advances more
// than one sequence at once holds an additional in-flight state buffer per
// grouped step, so one copy of the per-context state is the conservative bound
// when more than one sequence is in flight. cohort<=1 adds no scratch.
//
// PerSession and CohortScratch share the perContext slice by design:
// EnvelopePlan holds value-type EnvelopeDemand and nothing in this package
// mutates either slice, so the alias is read-only and avoids a copy on the hot
// admission path.
func NewSessionResidencyPlan(weights EnvelopePlan, perContext EnvelopePlan, maxSessions, cohort int) SessionResidencyPlan {
	p := SessionResidencyPlan{
		Weights:     weights,
		PerSession:  perContext,
		MaxSessions: maxSessions,
	}
	if p.MaxSessions < 1 {
		p.MaxSessions = 1
	}
	if cohort > 1 {
		p.CohortScratch = perContext
	}
	return p
}

// addEnvelopeClasses folds one plan's class totals into out, scaled by copies
// (the number of times that plan is resident). Copies <= 0 contribute nothing.
func addEnvelopeClasses(out map[MemClass]int64, plan EnvelopePlan, copies int64) {
	if copies <= 0 {
		return
	}
	for class, bytes := range plan.ByClass() {
		out[class] = saturatingAddInt64(out[class], saturatingMulInt64(bytes, copies))
	}
}

// saturatingAddInt64 adds two envelope quantities, saturating at the int64
// bounds instead of wrapping.
func saturatingAddInt64(a, b int64) int64 {
	sum := a + b
	if b > 0 && sum < a {
		return int64(^uint64(0) >> 1) // MaxInt64
	}
	if b < 0 && sum > a {
		return -int64(^uint64(0)>>1) - 1 // MinInt64
	}
	return sum
}

// saturatingMulInt64 multiplies two envelope quantities, saturating at the int64
// bounds instead of wrapping. A wrapped product could shrink a demand and
// under-reserve a load, so overflow must fail toward "too big".
func saturatingMulInt64(a, b int64) int64 {
	if a == 0 || b == 0 {
		return 0
	}
	if a == -1 && b == -int64(^uint64(0)>>1)-1 {
		return int64(^uint64(0) >> 1)
	}
	if b == -1 && a == -int64(^uint64(0)>>1)-1 {
		return int64(^uint64(0) >> 1)
	}
	product := a * b
	if product/b != a {
		if (a > 0) == (b > 0) {
			return int64(^uint64(0) >> 1)
		}
		return -int64(^uint64(0)>>1) - 1
	}
	return product
}
