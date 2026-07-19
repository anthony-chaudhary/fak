package cacheprice

// route.go closes the loop on this package's pricing: cacheobs OBSERVES where a turn's prefix was
// served from (its provenance axis, #3896); this DECIDES where to serve the next one from, at
// least token cost. It is the prospective mirror of that retrospective axis, and it synthesizes
// the two per-turn leaves already landed — AdmissionTokens (the residency-discounted suffix) and
// the DisaggregationDividend inequality (overhead vs recompute) — into the single choice a
// throughput-parity scheduler makes per admission, without introducing any new cost model.

// AdmissionRoute is the cheapest source a scheduler picks to admit a turn's prefix. It is the
// decision counterpart to cacheobs's observed ReuseSource: that records provenance after the fact,
// this chooses it before.
type AdmissionRoute int

const (
	// RouteRecompute prefills the whole prompt locally — no usable residency, or every cached
	// source costs more than just recomputing.
	RouteRecompute AdmissionRoute = iota
	// RouteLocal serves the prefix from this box's own resident KV — a full discount with no
	// fabric toll, always the cheapest option when it is available.
	RouteLocal
	// RouteRemote fetches the prefix across the fabric from the disaggregated KV tier, paying the
	// transfer toll — chosen only when it is cheaper than recomputing and no local copy exists.
	RouteRemote
)

// String renders the route for logs and metrics labels.
func (r AdmissionRoute) String() string {
	switch r {
	case RouteRecompute:
		return "recompute"
	case RouteLocal:
		return "local"
	case RouteRemote:
		return "remote"
	default:
		return "unknown"
	}
}

// CheapestRoute returns the least-token-cost way to admit a turn of promptTokens whose prefix of
// prefixTokens is available locally resident (localAvailable) and/or across the fabric from the
// disaggregated tier at transferOverheadTokens toll (remoteAvailable), together with that route's
// billable admission cost in tokens. The three candidate costs are all expressed through the
// landed leaves so they read on one axis:
//
//	recompute = AdmissionTokens(promptTokens, 0)            // prefill everything, no residency
//	local     = AdmissionTokens(promptTokens, prefixTokens) // resident suffix only, no toll
//	remote    = local + transferOverheadTokens              // suffix plus the fabric toll
//
// Local, when present, is never worse than remote (same suffix, zero toll), so it wins outright.
// Remote is taken only when it is STRICTLY cheaper than recomputing — exactly the
// DisaggregationWorthwhile condition (toll < resident prefix ⟺ dividend > 0) — so a break-even or
// losing fabric fetch falls back to the local recompute rather than taking on a fabric dependency
// for no gain. A prefix that yields no discount (prefixTokens ≤ 0) leaves recompute as the choice.
// Inputs clamp defensively via AdmissionTokens; a negative toll books as 0.
func CheapestRoute(promptTokens, prefixTokens, transferOverheadTokens int, localAvailable, remoteAvailable bool) (AdmissionRoute, int) {
	if transferOverheadTokens < 0 {
		transferOverheadTokens = 0
	}
	suffix := AdmissionTokens(promptTokens, prefixTokens)

	route := RouteRecompute
	cost := AdmissionTokens(promptTokens, 0) // = clamped promptTokens

	// Local: a real residency discount (strictly below recompute) always wins — cheapest, no toll.
	if localAvailable && suffix < cost {
		route, cost = RouteLocal, suffix
	}
	// Remote: only when the tolled fetch still beats the current best (which, if local was taken,
	// it cannot, since toll ≥ 0). This reduces to dividend > 0 when local is absent.
	if remoteAvailable {
		if remote := suffix + transferOverheadTokens; remote < cost {
			route, cost = RouteRemote, remote
		}
	}
	return route, cost
}
