package modelroute

import "fmt"

// ---------------------------------------------------------------------------
// CAPACITY-AWARE REROUTE (#3520) — the missing control-path arrow.
// ---------------------------------------------------------------------------
//
// The routing spine (Route) picks a model by aspect/complexity/labels; it is
// blind to whether the chosen model FITS the local faithful path. The hardware
// docs name this exact gap: the policy plane and the physical plane "meet only
// at the meter, never at the control"
// (docs/explainers/hardware-limits-and-capacity.md §2). This file adds that
// missing control arrow as a PURE kernel: when a task exceeds a local capacity
// ceiling, it emits a typed CAPACITY_REROUTE directive so a wiring layer can
// dispatch THAT task to the fleet-GPU / larger-window lane (Platform 4, the
// 8-GPU server where "single-box ceilings stop binding", docs/HARDWARE-MATRIX.md)
// instead of failing or stalling. Per the doctrine note §1.2, "no local capacity"
// is a routing INPUT, not a dead end.
//
// Three orthogonal reroute triggers (two numeric ceilings plus one pool signal),
// each fail-OPEN when its signal is absent (an unknown capacity NEVER reroutes —
// docs Planks 1-5):
//
//   - FAITHFUL PARAM CEILING. The local box serves a model faithfully only up to
//     a parameter size ("fak faithful <= 7B on the 36 GB Mac",
//     hardware-limits-and-capacity.md:72). A model larger than the configured
//     ceiling over-runs the local faithful path.
//   - CONTEXT-WINDOW OVER-SUBSCRIPTION. Prompt + planned output exceeding the
//     model's headroom-adjusted context window (the served capacity-precheck,
//     with a fixed 15% device-headroom reserve, hardware-limits-and-capacity.md:303)
//     means the request cannot be served whole on the local device.
//   - SEAT-POOL EXHAUSTION (#3577). The provider seat pool is blocked fleet-wide
//     (REFUSE_NO_ACCOUNT holds everywhere), so a task that could run on self-hosted
//     fleet compute has no provider seat to wait for and should spill to idle fleet
//     GPU capacity instead of stalling (CONCEPT-IDEAL §1.2, memory
//     local-machine-not-a-constraint). Unlike the two numeric ceilings this is a
//     boolean pool signal rather than a threshold, but it obeys the SAME fail-OPEN
//     rule: an absent (false) signal is not evidence of exhaustion and never
//     reroutes. The live wiring of this signal from the deficit fold is separate
//     (issue #2); the kernel only decides the verdict from the boolean a caller sets.
//
// This stays a pre-Submit MODEL-ID decision, NOT a dispatch-time override: it
// hands a wiring layer a directive to route elsewhere, honoring the
// ROUTE-BEFORE-ADJUDICATE contract at the top of this package. Locality remains
// the roster/Target's job — the kernel never reasons about Kind or hardware; it
// compares declared numbers a caller measured. The caller supplies the intended
// model's size already parsed to billions (e.g. via turnbench's paramsBillions or
// a roster field), so the routing spine takes on no size-string parsing and no
// dependency on the benchmark leaves.

// CapacityReason is the closed vocabulary for a capacity-driven routing verdict —
// emittable, verifiable, and refusable per the doctrine note §2.6 closed-reason
// contract. It is an ADDITIVE set: a new value is an added constant plus a
// knownCapacityReason arm, never a free-text field. The subject-then-disposition
// shape (CAPACITY names the cause, the token names the disposition) mirrors the
// fleet's other closed reasons (REFUSE_NO_ACCOUNT, ISSUE_SCOPE_INCOMPLETE).
type CapacityReason string

const (
	// CapacityOK — a capacity signal was present and the task fits every
	// configured local ceiling; keep the route local.
	CapacityOK CapacityReason = "CAPACITY_OK"
	// CapacityReroute — the task exceeds a configured local capacity ceiling
	// (param size or context window); route it to the fleet-GPU / larger-window
	// lane instead of failing or stalling.
	CapacityReroute CapacityReason = "CAPACITY_REROUTE"
	// CapacityUnknown — no actionable capacity signal (no configured ceiling, or
	// no measured demand); conservative-degrade to today's routing. This value
	// NEVER reroutes — an absent signal is not evidence of overflow.
	CapacityUnknown CapacityReason = "CAPACITY_UNKNOWN"
	// SeatExhausted — the provider seat pool is blocked fleet-wide, so the task has
	// no provider seat to wait for; route it to self-hosted fleet GPU compute
	// instead of stalling (#3577). This is a REROUTE disposition like
	// CapacityReroute (Reroute() reports true for it); it differs only in naming the
	// seat pool — not model shape — as the cause. Like every other reroute it fires
	// only on a present signal: SeatPoolExhausted must be true, never merely unset.
	SeatExhausted CapacityReason = "SEAT_EXHAUSTED"
)

// knownCapacityReason reports whether r is one of the closed CapacityReason set.
func knownCapacityReason(r CapacityReason) bool {
	switch r {
	case CapacityOK, CapacityReroute, CapacityUnknown, SeatExhausted:
		return true
	}
	return false
}

// Valid reports whether r is a member of the closed CapacityReason vocabulary.
func (r CapacityReason) Valid() bool { return knownCapacityReason(r) }

// Sourced capacity constants — documented, not invented. See
// docs/explainers/hardware-limits-and-capacity.md and docs/HARDWARE-MATRIX.md.
const (
	// LocalFaithfulCeilingBillion is fak's single authoritative local faithful
	// ceiling: "fak faithful <= 7B on the 36 GB Mac"
	// (hardware-limits-and-capacity.md:72). It is a DEFAULT an operator may adopt
	// for CapacityCeiling.FaithfulParamsB, not a hard-coded gate — other platforms
	// carry their own ceiling (sourced from BENCHMARK-AUTHORITY.md), so the kernel
	// takes the ceiling as configuration rather than compiling in one number.
	LocalFaithfulCeilingBillion = 7.0
	// DefaultDeviceHeadroom is the reserved device headroom — allocator
	// fragmentation, backend pools, unmodeled runtime uploads: 15%
	// (hardware-limits-and-capacity.md:303). A request may therefore use at most
	// (1 - DefaultDeviceHeadroom) of the context window before it over-subscribes.
	DefaultDeviceHeadroom = 0.15
)

// CapacityCeiling is the LOCAL faithful path's capacity policy. It is
// operator-configurable so the ceiling lives in a reviewable value, never a
// compiled-in gate. A zero-valued field means "this ceiling is not configured",
// which conservative-degrades (that axis never reroutes).
type CapacityCeiling struct {
	// FaithfulParamsB is the largest model size (billions of params) the local box
	// serves faithfully. 0 => no param ceiling configured (the param axis is
	// inert). A common value is LocalFaithfulCeilingBillion.
	FaithfulParamsB float64
	// HeadroomFraction is the reserved device headroom in [0,1). 0 (or an
	// out-of-range value) => DefaultDeviceHeadroom. A request may use at most
	// (1 - HeadroomFraction) of the context window.
	HeadroomFraction float64
}

// CapacityDemand is the per-task capacity facts a caller measures BEFORE Submit:
// the intended local model's size and window, plus the request's token demand.
// Every field zero => no signal => the kernel conservative-degrades.
type CapacityDemand struct {
	// ModelParamsB is the intended local model's size in billions of params.
	// 0 => unknown (the param axis is inert).
	ModelParamsB float64
	// ContextWindow is the intended model's context window in tokens.
	// 0 => unknown / unbounded (the window axis is inert), NOT zero-capacity.
	ContextWindow int
	// PromptTokens is the request's prompt length in tokens (== Subject.PromptTokens).
	PromptTokens int
	// ExpectedOutputTokens is the planned max output tokens to reserve. 0 => none.
	ExpectedOutputTokens int
	// SeatPoolExhausted reports that the provider seat pool is blocked fleet-wide
	// (REFUSE_NO_ACCOUNT everywhere), fed later by the deficit fold (#2). true =>
	// reroute to self-hosted fleet compute with reason SEAT_EXHAUSTED. false (the
	// zero value) is an ABSENT signal, not evidence of a healthy pool: it is inert
	// and never reroutes on this axis, exactly like an unconfigured ceiling.
	SeatPoolExhausted bool
}

// CapacityVerdict is the closed decision: the typed Reason, which ceiling(s)
// tripped, and a free-text Detail for the trace. It is returned ALONGSIDE a Route
// Decision (rather than embedded in it) so the capacity verdict is a legible,
// separately-typed value that a wiring layer surfaces in the tick payload, and so
// the widely-constructed Decision struct is left unchanged.
type CapacityVerdict struct {
	Reason     CapacityReason
	OverParam  bool
	OverWindow bool
	Detail     string
}

// Reroute reports whether the verdict directs the task elsewhere. Both reroute
// dispositions count: CapacityReroute (model-shape overflow) and SeatExhausted
// (provider seat pool blocked) — each spills the task to the fleet-GPU lane.
func (v CapacityVerdict) Reroute() bool {
	return v.Reason == CapacityReroute || v.Reason == SeatExhausted
}

// AssessCapacity is the pure capacity kernel: given a task's measured demand and
// the local faithful ceiling, decide keep-local (CapacityOK) vs route-elsewhere
// (CapacityReroute), or conservative-degrade (CapacityUnknown) when no actionable
// signal is present. Pure and deterministic — same inputs, same verdict, no I/O.
func AssessCapacity(d CapacityDemand, c CapacityCeiling) CapacityVerdict {
	paramConfigured := c.FaithfulParamsB > 0 && d.ModelParamsB > 0
	windowConfigured := d.ContextWindow > 0 && (d.PromptTokens > 0 || d.ExpectedOutputTokens > 0)

	if !paramConfigured && !windowConfigured && !d.SeatPoolExhausted {
		return CapacityVerdict{
			Reason: CapacityUnknown,
			Detail: "no configured capacity ceiling, measured demand, or seat-pool signal; keeping today's route",
		}
	}

	overParam := paramConfigured && d.ModelParamsB > c.FaithfulParamsB

	headroom := c.HeadroomFraction
	if headroom <= 0 || headroom >= 1 {
		headroom = DefaultDeviceHeadroom
	}
	usable := int(float64(d.ContextWindow) * (1 - headroom))
	demand := d.PromptTokens + d.ExpectedOutputTokens
	overWindow := windowConfigured && demand > usable

	switch {
	case overParam && overWindow:
		return CapacityVerdict{
			Reason: CapacityReroute, OverParam: true, OverWindow: true,
			Detail: fmt.Sprintf("model %.1fB > %.1fB faithful ceiling AND demand %d tok > %d usable window (%d ctx - %.0f%% headroom)",
				d.ModelParamsB, c.FaithfulParamsB, demand, usable, d.ContextWindow, headroom*100),
		}
	case overParam:
		return CapacityVerdict{
			Reason: CapacityReroute, OverParam: true,
			Detail: fmt.Sprintf("model %.1fB exceeds %.1fB local faithful ceiling", d.ModelParamsB, c.FaithfulParamsB),
		}
	case overWindow:
		return CapacityVerdict{
			Reason: CapacityReroute, OverWindow: true,
			Detail: fmt.Sprintf("demand %d tok over-subscribes %d usable window (%d ctx - %.0f%% headroom)",
				demand, usable, d.ContextWindow, headroom*100),
		}
	case d.SeatPoolExhausted:
		// Model shape fits (or is unconfigured) but the provider seat pool is blocked
		// fleet-wide: spill to self-hosted fleet compute rather than stall on a seat.
		return CapacityVerdict{
			Reason: SeatExhausted,
			Detail: "provider seat pool exhausted (REFUSE_NO_ACCOUNT fleet-wide); routing to self-hosted fleet compute",
		}
	default:
		return CapacityVerdict{Reason: CapacityOK, Detail: "fits local faithful path"}
	}
}

// CapacityRerouteLabel is the Subject label a capacity reroute stamps so an
// operator's top-of-list manifest rule can select the fleet-GPU / larger-window
// lane, e.g. match:{labels:{capacity:"reroute"}} -> plan:{members:[{model:"fleet-large"}]}.
// Reusing the OPEN Labels channel keeps Route's hot path unchanged: a Subject
// without this label routes exactly as it does today.
const CapacityRerouteLabel = "capacity"

// CapacityRerouteValue is the value stamped under CapacityRerouteLabel on reroute.
const CapacityRerouteValue = "reroute"

// RouteWithCapacity folds the capacity kernel over Route: it assesses the task's
// capacity and, on CapacityReroute, stamps the reroute label on a COPY of the
// Subject before routing (so a capacity rule selects the fleet lane); otherwise it
// routes the Subject unchanged. It returns both the Decision and the typed
// CapacityVerdict so the reroute is legible (the §2.6 typed reason) rather than a
// silent relabel.
//
// Conservative-degrade is structural: CapacityOK and CapacityUnknown leave the
// Subject byte-identical, so RouteWithCapacity(s, ...) == Route(s) whenever the
// task fits or carries no capacity signal.
func (m Manifest) RouteWithCapacity(s Subject, d CapacityDemand, c CapacityCeiling) (Decision, CapacityVerdict) {
	v := AssessCapacity(d, c)
	if v.Reroute() {
		s = withCapacityLabel(s)
	}
	return m.Route(s), v
}

// withCapacityLabel returns a copy of the Subject carrying the reroute label,
// never mutating the caller's Labels map (Route callers may share it).
func withCapacityLabel(s Subject) Subject {
	labels := make(map[string]string, len(s.Labels)+1)
	for k, v := range s.Labels {
		labels[k] = v
	}
	labels[CapacityRerouteLabel] = CapacityRerouteValue
	s.Labels = labels
	return s
}
