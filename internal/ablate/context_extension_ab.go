package ablate

// Context-extension fire classification (#2808). Sibling to compaction_ab.go (#2805) and
// anchor_ab.go (#2809): where those two split an A/B ARM (compaction ON/OFF, anchor FirstBP/Head),
// this one splits the fire POPULATION. A compaction "fire" is one firing of the context-extension
// path that sheds resident tokens. Not every fire is worth the same, and today they are LUMPED
// into one shed total so the report cannot attribute the real value:
//
//   - A LIMIT-AVOIDING fire kept the session UNDER the hard context window — absent it the
//     resident prefix would have met or crossed the cap and the session would have stalled. This
//     is the genuinely valuable population.
//   - A CACHE-SHAVING fire fired while the session was still comfortably under the window and only
//     shaved tokens the provider was already serving WARM from cache. Dropping an already-cached
//     token saves only the 0.1x read marginal, so this population is near-zero net.
//
// CLASSIFICATION is the WINDOW test: a fire is limit-avoiding iff the resident token count
// immediately BEFORE it met or exceeded the hard context window (ResidentTokensBeforeFire >=
// ContextWindowTokens) — without the fire the session would have been at/over the cap, so the fire
// is what kept it under. Any fire below the window only shaved (mostly already-cached) tokens.
//
// VALUE is priced with the SAME warm/cold shed blend the live gateway split
// (gateway.MechanismSavings.FakTokenEquiv, backed by cacheprice.ShedTokenEquiv) and the Track-2
// report value a shed token at (#2794/#2798): the warm portion min(shed, warm) is worth only the
// 0.1x read marginal, the cold remainder full input. Reusing that one function by construction is
// what makes a cache-shaving fire (fully-warm shed) score near-zero and a limit-avoiding fire
// (cold/over-window shed) carry the real value — the two surfaces cannot disagree.
//
// Generation posture (gen/now, #2783): a deterministic ($0, no model, no GPU) valuation over
// caller-supplied fire axes — it does not itself run the compaction path. Promotion evidence = a
// live guard capture feeding real recorded fires whose limit-avoiding population carries the bulk
// of the priced value. Invalidating assumption (named on the report Caveat): the ledger must
// record resident-before-fire AND the hard window; if it does not, the classifier needs a NEW
// counterfactual signal from the gateway compaction path (would this session have crossed the cap
// absent the fire?), which widens scope past this deterministic valuation.

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// The two fire-population labels the report splits fires into.
const (
	FireLimitAvoiding = "limit_avoiding"
	FireCacheShaving  = "cache_shaving"
)

// ContextExtensionFire is ONE recorded compaction firing event, on the axes needed to classify it
// and price its shed:
//
//   - ContextWindowTokens is the hard context window the session must stay under (the cap a fire
//     avoids). A fire with no window cannot be classified.
//   - ResidentTokensBeforeFire is the model-visible resident token count immediately before the
//     fire — the counterfactual "where would the session be without this fire" the window test reads.
//   - ShedTokens is how many resident tokens the fire removed.
//   - WarmShedTokens is how many of ShedTokens the provider was already serving WARM as cache_read
//     (the OBSERVED warm witness). It only DISCOUNTS the shed's value; it never reclassifies the
//     fire (classification is the window test, not the warm mix).
type ContextExtensionFire struct {
	SessionID                string
	ContextWindowTokens      int64
	ResidentTokensBeforeFire int64
	ShedTokens               int64
	WarmShedTokens           int64
}

// LimitAvoiding reports whether this fire kept the session under the hard context window: true iff
// the pre-fire resident count met or exceeded the window, so absent the fire the session would have
// been at/over the cap. A fire below the window is cache-shaving, not limit-avoiding.
func (f ContextExtensionFire) LimitAvoiding() bool {
	return f.ResidentTokensBeforeFire >= f.ContextWindowTokens
}

// ShedValueUSD is the deterministic net dollar value of what this fire shed, priced with the SAME
// warm/cold blend the live gateway split and the Track-2 report use (gateway.MechanismSavings.
// FakTokenEquiv wraps cacheprice.ShedTokenEquiv): the warm portion min(shed, warm) prices at the
// 0.1x read marginal, the cold remainder at full input, times the model's base input price. A
// fully-warm shed therefore lands near-zero — the near-zero-net the issue names — and a cold shed
// carries the full value. Reusing the one canonical function keeps this from disagreeing with the
// gateway/report numbers on the same tokens.
func (f ContextExtensionFire) ShedValueUSD(p gateway.CachePricing) float64 {
	if f.ShedTokens <= 0 {
		return 0
	}
	warm := f.WarmShedTokens
	if warm < 0 {
		warm = 0
	}
	tokEquiv := gateway.MechanismSavings{
		FakCompactionShedTokens:      uint64(f.ShedTokens),
		FakCompactionCacheReadTokens: uint64(warm),
	}.FakTokenEquiv()
	return tokEquiv * (p.InputPerMTokUSD / 1_000_000)
}

// FirePopulation is one classified population's aggregate: how many fires fell in it, the total
// tokens they shed, the priced dollar value attributed to them, and the sessions that fired.
type FirePopulation struct {
	Label      string   `json:"label"`
	N          int      `json:"n"`
	ShedTokens int64    `json:"shed_tokens"`
	ValueUSD   float64  `json:"value_usd"`
	Sessions   []string `json:"sessions,omitempty"`
}

// ContextExtensionReport is the two-population split the issue asks for: the limit-avoiding fires
// (the genuinely valuable population) reported SEPARATELY from the cache-shaving fires (near-zero
// net), each with its own count, shed total, and attributed dollar value.
type ContextExtensionReport struct {
	LimitAvoiding FirePopulation `json:"limit_avoiding"`
	CacheShaving  FirePopulation `json:"cache_shaving"`
	PricingSource string         `json:"pricing_source,omitempty"`
	Caveat        string         `json:"caveat,omitempty"`
}

// ClassifyContextExtensionFires folds recorded fires into the two-population report: each fire is
// classified by the window test and its shed priced by the warm/cold blend, then accumulated into
// its population. It fails closed on an empty fire set (no report to render, and a fabricated
// zero-value split would read as a measured no-op), on a fire with no context window (nothing to
// classify against), and on a fire that shed nothing (not a fire).
func ClassifyContextExtensionFires(fires []ContextExtensionFire, p gateway.CachePricing, pricingSource string) (ContextExtensionReport, error) {
	if len(fires) == 0 {
		return ContextExtensionReport{}, errors.New("ablate: context-extension classification needs at least one recorded fire")
	}
	rep := ContextExtensionReport{
		LimitAvoiding: FirePopulation{Label: FireLimitAvoiding},
		CacheShaving:  FirePopulation{Label: FireCacheShaving},
		PricingSource: pricingSource,
		Caveat:        contextExtensionCaveat(),
	}
	for i, f := range fires {
		if f.ContextWindowTokens <= 0 {
			return ContextExtensionReport{}, fmt.Errorf("ablate: fire %d (%q) has no context window to classify against", i, f.SessionID)
		}
		if f.ShedTokens <= 0 {
			return ContextExtensionReport{}, fmt.Errorf("ablate: fire %d (%q) shed no tokens — not a fire", i, f.SessionID)
		}
		pop := &rep.CacheShaving
		if f.LimitAvoiding() {
			pop = &rep.LimitAvoiding
		}
		pop.N++
		pop.ShedTokens += f.ShedTokens
		pop.ValueUSD += f.ShedValueUSD(p)
		pop.Sessions = append(pop.Sessions, f.SessionID)
	}
	return rep, nil
}

// RealValueUSD is the dollar value attributed to the LIMIT-AVOIDING fires — the genuinely valuable
// population, and the figure the report headline is about. This is the "real value" the acceptance
// says must land on the limit-avoiding fires, kept separate from the near-zero-net cache-shaving.
func (r ContextExtensionReport) RealValueUSD() float64 { return r.LimitAvoiding.ValueUSD }

// NearZeroNetUSD is the dollar value the CACHE-SHAVING fires contributed — reported separately, and
// near-zero by construction because their shed was tokens the provider was already serving warm.
func (r ContextExtensionReport) NearZeroNetUSD() float64 { return r.CacheShaving.ValueUSD }

// TotalFires is the number of fires the report classified across both populations.
func (r ContextExtensionReport) TotalFires() int { return r.LimitAvoiding.N + r.CacheShaving.N }

// AttributesRealValueToLimitAvoiding reports whether the priced value concentrates in the
// limit-avoiding population — true iff its value is at least the cache-shaving population's. This
// is the mechanical form of the acceptance: shaving cached tokens is near-zero net, so any real
// value the report shows sits with the fires that kept a session under the window.
func (r ContextExtensionReport) AttributesRealValueToLimitAvoiding() bool {
	return r.LimitAvoiding.ValueUSD >= r.CacheShaving.ValueUSD
}

// SweepRow renders the human one-liner the acceptance calls for: the two populations reported
// separately, each with its count and attributed dollar value, so a reader sees the real value on
// the limit-avoiding fires and the near-zero net on the cache-shaving ones.
func (r ContextExtensionReport) SweepRow() string {
	return fmt.Sprintf("context-extension fires: %d limit-avoiding ($%+.4f real value) vs %d cache-shaving ($%+.4f near-zero net) over N=%d fires",
		r.LimitAvoiding.N, r.LimitAvoiding.ValueUSD, r.CacheShaving.N, r.CacheShaving.ValueUSD, r.TotalFires())
}

// JSON renders the report as canonical indented JSON terminated by a newline — the rendered
// artifact witness showing the two fire populations and their attributed value, separately.
func (r ContextExtensionReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

func contextExtensionCaveat() string {
	return "classification is the window test (resident-before-fire ≥ hard window ⇒ limit-avoiding); it assumes the ledger records resident-before-fire and the hard window. Absent that counterfactual signal a fire cannot be classified and the split needs a new gateway signal. Per-fire value is the deterministic warm/cold shed blend (cacheprice.ShedTokenEquiv), not a live-traffic witness."
}
