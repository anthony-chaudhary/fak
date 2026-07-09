package cachevaluereport

// Per-model breakdown of the Track 2 OBSERVED provider-$ cache economics.
//
// The package-level Track 2 fold prices every observed session against one
// blended rate, so an expensive apex model's larger dollar savings are hidden
// under a cheaper model's rate. FableMoment re-groups the same four billable
// token axes BY MODEL and prices each group with that model's own base $/MTok,
// making the per-model net cache saving visible. Like the rest of Track 2 this
// is a cost PROJECTION priced from caller-supplied list prices, never a
// fak-WITNESSED claim.
//
// The cached-token multipliers are READ from the canonical cacheprice leaf
// (ReadMultiplier / Write5mMultiplier) rather than re-declared as bare 0.1 /
// 1.25 literals, so a cached token is valued identically here and on every
// other pricing surface by CONSTRUCTION (#2798).

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// ModelSession is one observed session's four billable token axes plus the
// model it ran on.
type ModelSession struct {
	Model                    string
	InputTokens              int64 // uncached input, billed 1.0x
	CacheReadInputTokens     int64 // cached-prefix read, billed ReadMultiplier x
	CacheCreationInputTokens int64 // cache write, billed Write5mMultiplier x
	OutputTokens             int64
}

// ModelPrice is the base $/MTok list price for a model.
type ModelPrice struct {
	InputPerMTokUSD  float64
	OutputPerMTokUSD float64
}

// ModelMoment is the per-model rollup FableMoment emits.
type ModelMoment struct {
	Model             string
	Sessions          int
	BilledUSD         float64 // full billed cost under fak's cache pricing
	NetCacheSavingUSD float64 // read rebate minus write premium
}

// FableMoment groups sessions by Model and prices each group with
// prices[model]. A session whose Model is absent from prices is SKIPPED and
// its model name is returned in the sorted unpriced slice, so nothing is
// silently dropped. Groups are sorted by Model for determinism.
func FableMoment(sessions []ModelSession, prices map[string]ModelPrice) (groups []ModelMoment, unpriced []string) {
	byModel := make(map[string]*ModelMoment)
	unpricedSeen := make(map[string]bool)
	for _, s := range sessions {
		price, ok := prices[s.Model]
		if !ok {
			unpricedSeen[s.Model] = true
			continue
		}
		m := byModel[s.Model]
		if m == nil {
			m = &ModelMoment{Model: s.Model}
			byModel[s.Model] = m
		}
		ipt := price.InputPerMTokUSD / 1e6
		opt := price.OutputPerMTokUSD / 1e6
		m.Sessions++
		m.BilledUSD += float64(s.InputTokens)*ipt +
			float64(s.CacheReadInputTokens)*ipt*cacheprice.ReadMultiplier +
			float64(s.CacheCreationInputTokens)*ipt*cacheprice.Write5mMultiplier +
			float64(s.OutputTokens)*opt
		m.NetCacheSavingUSD += float64(s.CacheReadInputTokens)*ipt*(1.0-cacheprice.ReadMultiplier) -
			float64(s.CacheCreationInputTokens)*ipt*(cacheprice.Write5mMultiplier-1.0)
	}
	groups = make([]ModelMoment, 0, len(byModel))
	for _, m := range byModel {
		groups = append(groups, *m)
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i].Model < groups[j].Model })
	unpriced = make([]string, 0, len(unpricedSeen))
	for model := range unpricedSeen {
		unpriced = append(unpriced, model)
	}
	sort.Strings(unpriced)
	return groups, unpriced
}
