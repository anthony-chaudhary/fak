package dispatchtick

import "fmt"

// Cross-provider seat failover (#3575, gen/next). When the primary worker product's
// seats are all walled -- preflight refuses REFUSE_NO_ACCOUNT because every account in
// the target pool is busy, throttled, or login-blocked -- re-attempting on the very next
// tick just re-checks a wall only a peer FINISHING can move (the same transient
// internal/seatpark parks). But a fleet that also holds a DIFFERENT provider's pool
// (e.g. an ambient ~/.codex login while the Claude roster is capped) has real capacity
// sitting idle. This fold is the pure decision for routing THIS tick's launch onto that
// fallback pool instead of parking: given the primary verdict, a debounce count, and
// whether the fallback pool is servable, it returns which product the tick should launch
// on and a legible readout of which pool refused and which took the work.
//
// Pure: same Input in, same Choice out; zero I/O, zero clock/env reads. The impure
// shell (cmd/fak dispatchPreflightTimed) gathers the facts -- the FLEET_DISPATCH_FALLBACK_
// PRODUCT knob, the trailing no-seat refusal count from the loop ledger, and the fallback
// pool's own account/seat probe -- and hands them here as data, the same discipline
// internal/seatpark uses for its clock-supplied backoff.
//
// Gated and additive: with the knob unset the shell never builds an Enabled input, so a
// disabled Choice launches on the primary and the shell attaches nothing -- the common
// preflight payload stays byte-identical to before this fold existed. The fallback only
// LOWERS the risk of a wasted tick; it never overrides a servable primary (a non-
// REFUSE_NO_ACCOUNT verdict short-circuits to "hold on primary") and never fails over
// before the debounce, so a single transient no-seat blip does not thrash providers.

// FallbackProductEnv names the operator knob that arms cross-provider failover. Its value
// is the fallback worker product to route to when the primary is seat-walled (e.g.
// "codex"). Empty/unset => the feature is off and the preflight payload is byte-identical.
const FallbackProductEnv = "FLEET_DISPATCH_FALLBACK_PRODUCT"

// DefaultFallbackDebounceTicks is how many CONSECUTIVE REFUSE_NO_ACCOUNT ticks (including
// the current one) the primary pool must show before a launch fails over. A single no-seat
// blip is a transient a peer finishing clears; only a sustained wall means the primary pool
// is genuinely out of capacity for this window, so the debounce keeps one refused tick from
// thrashing the fleet across providers.
const DefaultFallbackDebounceTicks = 3

// FallbackSchema tags the readout attached to the tick payload.
const FallbackSchema = "fleet-dispatch-fallback/1"

// FallbackProductInput is the set of facts the impure shell gathers for one tick.
type FallbackProductInput struct {
	// Enabled is true iff FLEET_DISPATCH_FALLBACK_PRODUCT is set to a usable value. A
	// false Enabled always yields a disabled Decision that launches on the primary.
	Enabled bool
	// PrimaryProduct is the product the tick was invoked for (the ProductForBackend of
	// the tick's --backend).
	PrimaryProduct string
	// FallbackProduct is the pool to route to when the primary is seat-walled.
	FallbackProduct string
	// PrimaryVerdict is the primary preflight verdict for THIS tick. Only
	// PreflightRefuseNoAccount (the seat-wall transient) can arm a failover; any other
	// verdict -- SPAWN_OK, a cap/host refusal -- holds on the primary.
	PrimaryVerdict string
	// ConsecutiveRefusals is how many consecutive REFUSE_NO_ACCOUNT ticks the primary
	// pool has shown, INCLUDING this one. 1 => this is the first refusal.
	ConsecutiveRefusals int
	// DebounceThreshold is N; <=0 => DefaultFallbackDebounceTicks.
	DebounceThreshold int
	// FallbackServable is whether the fallback pool's own account/seat probe reports it
	// can serve a fresh worker right now. Gathered by the shell ONLY once the debounce is
	// satisfied, so a below-threshold tick spends no extra probe.
	FallbackServable bool
	// FallbackReason is a free-text note from the fallback probe (why it is / is not
	// servable), carried into the readout for the trace.
	FallbackReason string
}

// FallbackProductChoice is the verdict: which product to launch on, plus a legible
// account of why.
type FallbackProductChoice struct {
	// Enabled echoes whether the feature was armed (a misconfigured knob -- fallback ==
	// primary -- reports disabled).
	Enabled bool
	// Engaged is true iff this tick's launch is routed to the fallback product.
	Engaged bool
	// Armed is true iff the debounce threshold was reached (the primary is genuinely
	// walled), whether or not the fallback turned out to be servable. An armed-but-not-
	// engaged Choice means BOTH pools are walled.
	Armed               bool
	PrimaryProduct      string
	FallbackProduct     string
	FallbackServable    bool
	FallbackReason      string
	RefusedProduct      string
	LaunchProduct       string
	ConsecutiveRefusals int
	Threshold           int
	Reason              string
}

// ShouldFailover reports whether the caller should launch this tick on the fallback
// product instead of the primary.
func (d FallbackProductChoice) ShouldFailover() bool { return d.Engaged }

// DecideFallbackProduct folds one tick's facts into the launch-target decision. Order:
// disabled/misconfigured -> launch primary; primary not seat-walled -> launch primary;
// walled but below the debounce -> hold on primary (armed=false); walled past the debounce
// but the fallback is also unservable -> hold on primary (armed=true, both walls); walled
// past the debounce with a servable fallback -> fail over (engaged).
func DecideFallbackProduct(in FallbackProductInput) FallbackProductChoice {
	threshold := in.DebounceThreshold
	if threshold <= 0 {
		threshold = DefaultFallbackDebounceTicks
	}
	d := FallbackProductChoice{
		Enabled:             in.Enabled,
		PrimaryProduct:      in.PrimaryProduct,
		FallbackProduct:     in.FallbackProduct,
		FallbackServable:    in.FallbackServable,
		FallbackReason:      in.FallbackReason,
		LaunchProduct:       in.PrimaryProduct,
		ConsecutiveRefusals: in.ConsecutiveRefusals,
		Threshold:           threshold,
	}
	if !in.Enabled || in.FallbackProduct == "" || in.FallbackProduct == in.PrimaryProduct {
		d.Enabled = false
		d.Reason = "cross-provider failover disabled (FLEET_DISPATCH_FALLBACK_PRODUCT unset, blank, or equal to the primary product)"
		return d
	}
	if in.PrimaryVerdict != PreflightRefuseNoAccount {
		d.Reason = fmt.Sprintf("primary %s pool is servable (verdict %s); no failover", in.PrimaryProduct, fallbackVerdictLabel(in.PrimaryVerdict))
		return d
	}
	d.RefusedProduct = in.PrimaryProduct
	if in.ConsecutiveRefusals < threshold {
		d.Reason = fmt.Sprintf("primary %s seat-walled %d/%d consecutive ticks; below the failover debounce -- holding on primary (a single no-seat blip is a peer-finishing transient)", in.PrimaryProduct, in.ConsecutiveRefusals, threshold)
		return d
	}
	d.Armed = true
	if !in.FallbackServable {
		d.Reason = fmt.Sprintf("primary %s seat-walled %d/%d consecutive ticks and the %s fallback pool is also unservable (%s); holding on primary", in.PrimaryProduct, in.ConsecutiveRefusals, threshold, in.FallbackProduct, fallbackReasonLabel(in.FallbackReason))
		return d
	}
	d.Engaged = true
	d.LaunchProduct = in.FallbackProduct
	d.Reason = fmt.Sprintf("primary %s seat-walled %d/%d consecutive ticks; failing this launch over to the servable %s pool", in.PrimaryProduct, in.ConsecutiveRefusals, threshold, in.FallbackProduct)
	return d
}

// Map renders the decision as the legible payload block the tick attaches under
// out["fallback"], so an operator (and the loop ledger) can see which pool refused and
// which product took the work.
func (d FallbackProductChoice) Map() map[string]any {
	return map[string]any{
		"schema":               FallbackSchema,
		"enabled":              d.Enabled,
		"engaged":              d.Engaged,
		"armed":                d.Armed,
		"primary_product":      d.PrimaryProduct,
		"refused_product":      d.RefusedProduct,
		"fallback_product":     d.FallbackProduct,
		"fallback_servable":    d.FallbackServable,
		"launch_product":       d.LaunchProduct,
		"launch_backend":       d.LaunchProduct,
		"consecutive_refusals": d.ConsecutiveRefusals,
		"debounce_threshold":   d.Threshold,
		"reason":               d.Reason,
	}
}

// CountTrailingNoAccountRefusals counts the leading run of REFUSE_NO_ACCOUNT verdicts in a
// NEWEST-FIRST slice of a product's recent tick verdicts -- the consecutive no-seat wall
// the debounce keys on. The count stops at the first non-REFUSE_NO_ACCOUNT verdict (a tick
// that admitted or refused for another reason ended the wall), so a fresh wall always
// starts from zero. The caller adds 1 for the current in-flight tick, which is not yet in
// the ledger.
func CountTrailingNoAccountRefusals(reasonsNewestFirst []string) int {
	n := 0
	for _, r := range reasonsNewestFirst {
		if r != PreflightRefuseNoAccount {
			break
		}
		n++
	}
	return n
}

// fallbackVerdictLabel keeps the readout honest when the shell passes an empty verdict.
func fallbackVerdictLabel(v string) string {
	if v == "" {
		return PreflightOKVerdict
	}
	return v
}

func fallbackReasonLabel(r string) string {
	if r == "" {
		return "no free seat"
	}
	return r
}
