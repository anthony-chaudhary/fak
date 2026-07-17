package cachemeta

// Provider constants are MEASURED-or-HYPOTHESIS records, not bare literals.
//
// A provider cache constant — the advisory TTL a retention hint ages out at, the
// minimum prefix a provider will cache, the read-cost rebate a cache hit earns —
// is only as trustworthy as its provenance. A hard-coded literal (`5 * 60 * 1000`)
// silently blends a provider-published window, an in-repo measurement, and a
// plausible guess into the same anonymous number, so a reader cannot tell which
// constants rest on evidence and which are placeholders due for a re-measure.
//
// This file expresses those constants as ProviderConstant records that each carry
// a Freshness status (MEASURED vs HYPOTHESIS), an absolute date, and a source, so
// the provenance travels with the value. It is the cache-frontier
// DEFAULT-ENABLEMENT-NEXT-50 item 24 default (#1542): "TTL/min-prefix/read-discount
// are `MEASURED` or `HYPOTHESIS` with date and source." The numeric behavior of
// providerTTLMillis is unchanged — the resolver reads Value off these records — so
// this adds a provenance surface without moving any number.

// Freshness is the closed provenance status of a ProviderConstant: whether its
// value rests on a repo-witnessed measurement or on an unverified hypothesis. It is
// a fixed vocabulary, not free text, so a report can render provider provenance the
// same way the cache-frontier planes keep provider/kernel/context/forecast separate.
type Freshness string

const (
	// FreshnessMeasured marks a constant whose value is backed by a repo witness or
	// a provider-published/observed measurement, with the source recorded.
	FreshnessMeasured Freshness = "MEASURED"
	// FreshnessHypothesis marks a constant whose value is a plausible placeholder
	// that has not yet been independently measured in this repo; its Source points
	// at the doc or issue that will retire the guess.
	FreshnessHypothesis Freshness = "HYPOTHESIS"
)

// ProviderConstant is a single provider cache constant carrying its own provenance.
// Value is the numeric constant in the record's Unit; Status is its Freshness; Date
// is the absolute (YYYY-MM-DD) date the status was last affirmed; Source is the doc
// path, issue, or provider reference that backs it. A record is never a bare number:
// a consumer that reads Value can always ask how fresh it is and where it came from.
type ProviderConstant struct {
	Value  float64   // the constant, expressed in Unit
	Unit   string    // "ms" | "tokens" | "ratio" (unit of Value)
	Status Freshness // MEASURED or HYPOTHESIS
	Date   string    // absolute date the status was affirmed, YYYY-MM-DD
	Source string    // doc path / issue / provider reference backing the value
}

// ProviderTTLConstants are the advisory prompt-cache retention windows a provider
// keeps a prefix resident for, keyed by canonical retention label. Both windows are
// MEASURED: the 5-minute and 1-hour ephemeral windows are the ones the vcache
// observation plane records per turn (internal/vcacheobserve/vcacheobserve.go folds
// distinct `ephemeral_1h` / `5m` provider-telemetry fields), and the 1h window is
// the target of the gateway's live cache_control TTL upgrade (#1850). They are the
// exact values providerTTLMillis returned as literals; only the provenance is new.
var ProviderTTLConstants = map[string]ProviderConstant{
	"5m": {
		Value:  5 * 60 * 1000,
		Unit:   "ms",
		Status: FreshnessMeasured,
		Date:   "2026-07-17",
		Source: "Anthropic 5-minute ephemeral prompt-cache window; observed per-turn as the `5m` provider-telemetry field in internal/vcacheobserve/vcacheobserve.go",
	},
	"1h": {
		Value:  60 * 60 * 1000,
		Unit:   "ms",
		Status: FreshnessMeasured,
		Date:   "2026-07-17",
		Source: "Anthropic 1-hour extended prompt-cache window; observed as the `ephemeral_1h` provider-telemetry field (internal/vcacheobserve/vcacheobserve.go) and driven by the gateway cache_control TTL upgrade (#1850)",
	},
}

// ProviderMinPrefixConstant is the minimum prefix length a provider will admit to
// its prompt cache. It is HYPOTHESIS: the ~1024-token floor is a provider-published
// figure, not a number this repo has independently measured, so it carries a
// forward reference to the item that will retire the guess rather than a repo
// witness. It is a provenance record only — no cachemeta code path branches on it
// today — so recording it does not change any behavior.
var ProviderMinPrefixConstant = ProviderConstant{
	Value:  1024,
	Unit:   "tokens",
	Status: FreshnessHypothesis,
	Date:   "2026-07-17",
	Source: "Anthropic-published ~1024-token minimum cacheable prefix, not yet repo-measured; see docs/cache-frontier/external-engine-cache-capability-inventory.md and #1542",
}

// ProviderReadDiscountConstant is the fraction of the base input cost a provider
// cache-read hit is charged at (a 0.1 ratio ⇒ a ~90% read rebate). It is HYPOTHESIS
// for the same reason as the min-prefix floor: it is a provider-published economics
// figure this repo prices projections from but has not independently measured, so
// its provenance points at the retiring item. Provenance record only; no behavior
// depends on it here.
var ProviderReadDiscountConstant = ProviderConstant{
	Value:  0.1,
	Unit:   "ratio",
	Status: FreshnessHypothesis,
	Date:   "2026-07-17",
	Source: "Anthropic-published ~0.1x cache-read cost multiplier (~90% read rebate), not yet repo-measured; see docs/cache-frontier/external-engine-cache-capability-inventory.md and #1542",
}

// lookupProviderTTL resolves a provider retention hint to its ProviderConstant,
// preserving the exact alias handling providerTTLMillis had as a literal switch:
// "5m"/"5min" ⇒ the 5-minute window, "1h"/"60m" ⇒ the 1-hour window, everything
// else ⇒ no record (ok=false). Routing the resolver through the record is what
// gives the returned TTL its provenance without moving the number.
func lookupProviderTTL(retention string) (ProviderConstant, bool) {
	switch retention {
	case "5m", "5min":
		return ProviderTTLConstants["5m"], true
	case "1h", "60m":
		return ProviderTTLConstants["1h"], true
	default:
		return ProviderConstant{}, false
	}
}
