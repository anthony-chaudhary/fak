package compute

// prewarm_admit.go — TOOL-LATENCY PREWARM ADMISSION: the scheduler-side decision that
// warms the next request's byte-known KV prefix during a tool's I/O latency window so the
// real request lands hot instead of cold-starting prefill (issue #810; parent epic #809).
//
// WHY THIS EXISTS — the concrete tension it resolves. On tool dispatch the kernel already
// KNOWS the next `/v1/messages` request's prefix up to the tool-result slot: it is the
// current context plus the result about to slot in, byte-determined NOW. The inference
// server never sees that — it gets the request at submit time and cold-starts prefill. The
// idle tool-latency window between dispatch and the result returning is free real estate:
// warm the known prefix into the serving path (the RadixAttention tree / kernel KVCache for
// self-hosted, or the provider's `cache_control` prefix cache for passthrough) so the real
// turn re-prefills nothing it could have had resident. This is the dual of #808's
// discard-aware admission: where that DECLINES to compute a forward pass known to be thrown
// away, this ADMITS a forward pass known to be needed — the cheapest prefill is the one
// already resident when the request lands.
//
// WHAT THIS FILE IS. The pure, deterministic, host-tractable DECISION the scheduler runs at
// the tool-call boundary — NOT the KV-moving wiring. It consumes the throughput-relevant
// projection of the boundary as plain data (is the continuation prefix byte-known? how long
// until the tool returns? how long does the warm take? how long does a lowest-priority warm
// survive before a demand session reclaims it?) so the tier-1 compute layer decides WITHOUT
// importing the session/serve layers that produce the signal — the same shape as
// discard_admit.go taking a bare `WillDiscard bool` and batchsched taking bare `lengths`.
// Given a candidate it returns one of three verdicts:
//
//   WarmSkip  — fail-safe: do not warm. The prefix is not byte-known (we will not warm a
//               continuation we cannot prove), or the warm pool has no free lowest-priority
//               capacity (pollution gate), or there is no latency window to exploit, or the
//               window is too short for the warm to land hot. The zero value, so a
//               default-constructed decision warms nothing.
//   WarmNow   — warm the byte-known prefix now: it completes before the request AND survives
//               (within the residency budget) until the request arrives — it lands hot.
//   WarmDefer — warming now lands too early: a lowest-priority prefix is reclaimed within the
//               residency budget, so it would be cold again by arrival. Defer the start by
//               DeferMillis so the warm lands inside the residency window — the prefetch-
//               distance knob, in closed form.
//
// THE FENCES (carried verbatim from the issue's prefetch prior art, load-bearing here).
//   - Zero mis-speculation risk. The prefix BEFORE the tool-result slot is byte-determined,
//     so this is a pure prefetch, not a speculation: it cannot be wrong, and so it needs no
//     effect-witnessed invalidation (the closed-loop machinery the riskier siblings #809(b)/
//     (c) need). That is exactly why prewarm is the P1 first cut. The decision FAILS CLOSED
//     on the trigger: a prefix that is not proven byte-known is never warmed (PrefixByteKnown
//     false -> WarmSkip), never speculated.
//   - Lowest-priority eviction class. A warmed prefix is an opportunistic bet, never a
//     dependency, always re-derivable: it may be reclaimed by any demand-driven session at
//     any moment. So the warm is gated by pollution — WarmPoolFree false -> WarmSkip — and a
//     WarmNow directive ALWAYS implies insertion at the lowest eviction priority with an
//     eviction horizon of ResidencyMillis. Correctness never depends on a warm landing or
//     surviving; a skipped or evicted warm only ever costs a cold prefill (the status quo),
//     never a wrong answer.
//   - Timeliness (the prefetch-distance knob). Warm too early and a lowest-priority prefix is
//     evicted before the request (warmed for nothing); warm too late and it has not finished
//     when the request lands (still cold). DeferMillis ties the warm's completion to the
//     tool's predicted completion time so it lands hot, not evicted-stale.
//
// HONEST SCOPE — what this is NOT. This is the DECISION, not its live wiring. It moves no KV
// and re-issues no provider request: a caller drives the actual warm via the existing paths
// (internal/radixkv Insert / the vcachewarm passthrough primitives) when this returns
// WarmNow. The defensible refinement over the closest published prior — KVFlow (arXiv
// 2507.07400), which predicts the next-needed prefix probabilistically via an Agent Step
// Graph — is DETERMINISM: the tool-call boundary gives the EXACT continuation prefix, so the
// trigger here is byte-known rather than predicted. The prewarm idea itself is not claimed as
// novel. The realized wall-clock win (RadixAttention hit-rate / `cache_read_input_tokens`
// readback) is measurable only on a live serving host with a real model and is therefore
// host-gated; this file closes the decidable, host-free core the same way #807 shipped
// TurnIntent and #808 shipped discard_admit ahead of their first live readers. It depends
// only on integer inputs, so it is byte-deterministic across machines — the repo's house
// form for a scheduling policy that must not drift with hardware.

// PrewarmVerdict is the typed outcome of the tool-latency prewarm admission decision.
type PrewarmVerdict uint8

const (
	// WarmSkip is the fail-safe verdict: warm nothing. It is the zero value, so a
	// default-constructed decision warms nothing.
	WarmSkip PrewarmVerdict = iota
	// WarmNow: warm the byte-known prefix now — it lands hot for the coming request.
	WarmNow
	// WarmDefer: warming now lands too early; defer the start by DeferMillis (the
	// prefetch-distance knob) so the warm lands inside the residency window.
	WarmDefer
)

// String renders the verdict for logs and the observability/witness surface.
func (v PrewarmVerdict) String() string {
	switch v {
	case WarmNow:
		return "warm_now"
	case WarmDefer:
		return "warm_defer"
	default:
		return "warm_skip"
	}
}

// PrewarmReason explains why a verdict was chosen — for the audit log and the witness. It is
// a closed vocabulary; policy code never parses a free-text reason.
type PrewarmReason string

const (
	// ReasonPrefixNotKnown: the continuation prefix is not proven byte-known, so warming it
	// would be speculation, which this pure-prefetch primitive forbids (fence 1).
	ReasonPrefixNotKnown PrewarmReason = "prefix_not_byte_known"
	// ReasonPoolPressure: the lowest-priority warm pool has no free capacity, so warming
	// would evict demand-driven residency — refused by the pollution gate (fence 2).
	ReasonPoolPressure PrewarmReason = "warm_pool_pressure"
	// ReasonNoLatencyWindow: there is no tool-latency window to exploit (the request is
	// effectively already here), so a warm would only race the live request.
	ReasonNoLatencyWindow PrewarmReason = "no_latency_window"
	// ReasonWindowTooShort: the warm cannot complete before the request arrives, so it would
	// not land hot; the demand prefill handles it (fence 3).
	ReasonWindowTooShort PrewarmReason = "window_too_short"
	// ReasonLandsHot: the warm completes in time AND survives until the request — WarmNow.
	ReasonLandsHot PrewarmReason = "lands_hot"
	// ReasonDeferTooEarly: warming now lands too early to survive eviction; defer to land it
	// inside the residency window — WarmDefer (fence 3, the prefetch-distance knob).
	ReasonDeferTooEarly PrewarmReason = "defer_lands_too_early"
)

// PrewarmCandidate is one tool-call boundary the scheduler is deciding about. It is the
// throughput-relevant projection of the boundary plus the deterministic trigger — pure data
// so a scheduler builds it from what it already holds, without this tier-1 package importing
// the session/serve layers above it. All millisecond fields are predicted/budgeted values
// the caller supplies; the decision is pure integer arithmetic over them.
type PrewarmCandidate struct {
	// PrefixByteKnown is the deterministic trigger: true iff the next request's prefix up to
	// the tool-result slot is byte-determined NOW (the tool-call boundary projection of a
	// byte-stable prefix fingerprint). False (the zero value) -> fail-closed: never warmed.
	PrefixByteKnown bool
	// WarmPoolFree is the pollution gate: true iff the lowest-priority warm pool has free
	// capacity to hold this prefix without evicting demand-driven residency. False -> skip.
	WarmPoolFree bool
	// ToolLatencyMillis is the predicted time until the real request arrives — the tool's
	// I/O completion time. Non-positive means no window to exploit (-> WarmSkip).
	ToolLatencyMillis int
	// WarmMillis is the predicted time the warm itself takes to make the prefix resident
	// (prefill the prefix / re-issue the passthrough request). Negative is clamped to 0.
	WarmMillis int
	// ResidencyMillis is how long a lowest-priority warmed prefix is expected to survive
	// before a demand-driven session reclaims it — the eviction horizon. Negative -> 0.
	ResidencyMillis int
	// PrefixTokens is the size of the prefix that would be warmed, for the realized-benefit
	// stat (the prefill rows that land hot). Negative is treated as 0. Never gates policy.
	PrefixTokens int
	// Reason is an optional human label for the audit log ("tool-dispatch:bash"). It is
	// never parsed by policy code.
	Reason string
}

// PrewarmDecision is the typed outcome for one candidate: the verdict, the defer delay (only
// meaningful for WarmDefer), and the closed-vocabulary reason for the audit/witness surface.
type PrewarmDecision struct {
	Verdict PrewarmVerdict
	// DeferMillis is how long to defer the warm START, in milliseconds. It is > 0 only for
	// WarmDefer; 0 for WarmNow and WarmSkip. Re-deciding the candidate after this delay
	// yields WarmNow (see DecidePrewarmAdmission's closed-form derivation).
	DeferMillis int
	Reason      PrewarmReason
}

// DecidePrewarmAdmission returns the verdict for a single tool-call boundary candidate. It
// FAILS CLOSED on the trigger (an unproven prefix is never warmed) and FAILS SAFE on
// everything else (a skipped or deferred warm only ever costs a cold prefill, never a wrong
// answer). The positive verdict WarmNow means exactly "this warm lands hot": it completes
// before the request AND survives until the request within the residency budget.
func DecidePrewarmAdmission(c PrewarmCandidate) PrewarmDecision {
	// Fence 1 — zero mis-speculation: only ever warm a byte-determined continuation. A prefix
	// we cannot prove byte-known would be speculation, which this pure-prefetch primitive
	// forbids; never warm it.
	if !c.PrefixByteKnown {
		return PrewarmDecision{Verdict: WarmSkip, Reason: ReasonPrefixNotKnown}
	}
	// Fence 2 — pollution gate: a warm is an opportunistic bet and must never evict
	// demand-driven residency to place it.
	if !c.WarmPoolFree {
		return PrewarmDecision{Verdict: WarmSkip, Reason: ReasonPoolPressure}
	}
	// Fence 3 — timeliness. Need a latency window to exploit at all.
	t := c.ToolLatencyMillis
	if t <= 0 {
		return PrewarmDecision{Verdict: WarmSkip, Reason: ReasonNoLatencyWindow}
	}
	w := c.WarmMillis
	if w < 0 {
		w = 0
	}
	r := c.ResidencyMillis
	if r < 0 {
		r = 0
	}
	if w > t {
		// The warm cannot complete before the request arrives — it would not land hot. Leave
		// it to the demand prefill rather than spend pool on a doomed warm.
		return PrewarmDecision{Verdict: WarmSkip, Reason: ReasonWindowTooShort}
	}
	// slack is the ms a completed warm would sit resident before the request arrives.
	slack := t - w
	if slack > r {
		// Warming now lands too early: a lowest-priority prefix is reclaimed within r, so it
		// would be cold again by arrival. Defer the START by (slack - r) so the warm completes
		// exactly r ms before the request — inside the residency window. Re-deciding after the
		// defer gives t' = r + w, slack' = r, slack' <= r -> WarmNow.
		return PrewarmDecision{Verdict: WarmDefer, DeferMillis: slack - r, Reason: ReasonDeferTooEarly}
	}
	// w <= t and slack <= r: the warm completes in time and survives until the request.
	return PrewarmDecision{Verdict: WarmNow, Reason: ReasonLandsHot}
}

// PrewarmPlanItem is one candidate's decision plus its original index, so a caller can map
// the plan back onto its own boundary slice without re-deriving order.
type PrewarmPlanItem struct {
	Index    int
	Decision PrewarmDecision
}

// PrewarmStats summarises a plan for observability and for the witness — the concrete
// prefill the decision warms ahead of demand.
type PrewarmStats struct {
	Candidates   int // total candidates decided
	Warmed       int // WarmNow: byte-known prefixes warmed to land hot
	Deferred     int // WarmDefer: warms scheduled closer to completion (prefetch-distance)
	Skipped      int // WarmSkip: not warmed (unknown prefix, pool pressure, or no/short window)
	TokensWarmed int // Σ PrefixTokens over WarmNow candidates — the prefill rows that land hot
}

// PlanPrewarmAdmission decides every candidate and rolls the verdicts into per-candidate
// decisions plus aggregate stats. The decisions are index-aligned and 1:1 with the input
// (every candidate is decided exactly once). It is a pure fold over DecidePrewarmAdmission,
// so it is deterministic for a given input and carries the same fail-closed/fail-safe
// contract. A nil/empty input yields a nil item slice and a zero stats.
func PlanPrewarmAdmission(cands []PrewarmCandidate) ([]PrewarmPlanItem, PrewarmStats) {
	if len(cands) == 0 {
		return nil, PrewarmStats{}
	}
	items := make([]PrewarmPlanItem, len(cands))
	stats := PrewarmStats{Candidates: len(cands)}
	for i, c := range cands {
		d := DecidePrewarmAdmission(c)
		items[i] = PrewarmPlanItem{Index: i, Decision: d}
		switch d.Verdict {
		case WarmNow:
			stats.Warmed++
			if c.PrefixTokens > 0 {
				stats.TokensWarmed += c.PrefixTokens
			}
		case WarmDefer:
			stats.Deferred++
		default:
			stats.Skipped++
		}
	}
	return items, stats
}
