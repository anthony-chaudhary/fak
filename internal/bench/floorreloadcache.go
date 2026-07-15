package bench

import "math"

// FloorReloadCacheScenario is a hermetic prompt-cache cost model for one
// reachability-changing capability-floor reload in an N-turn session.
type FloorReloadCacheScenario struct {
	Turns              int `json:"turns"`
	ReloadTurn         int `json:"reload_turn"`
	StablePrefixTokens int `json:"stable_prefix_tokens"`
	FreshTokensPerTurn int `json:"fresh_tokens_per_turn"`
}

// FloorReloadCacheResult compares the no-reload baseline with one mid-session
// tools[] prefix bust. TokenCost uses cache-read=0.1x and fresh/write=1.0x.
type FloorReloadCacheResult struct {
	Scenario              FloorReloadCacheScenario `json:"scenario"`
	BaselineCostTokens    float64                  `json:"baseline_cost_tokens"`
	ReloadCostTokens      float64                  `json:"reload_cost_tokens"`
	ForfeitedCachedTokens int                      `json:"forfeited_cached_tokens"`
	SpendFraction         float64                  `json:"spend_fraction"`
	TurnsToAmortize       int                      `json:"turns_to_amortize"`
}

const floorReloadCacheReadRate = 0.1

// ModelFloorReloadCache prices one reload. The changed tools[] block invalidates
// the stable prefix once; subsequent turns reuse the new prefix normally.
func ModelFloorReloadCache(s FloorReloadCacheScenario) FloorReloadCacheResult {
	if s.Turns < 1 {
		s.Turns = 1
	}
	if s.ReloadTurn < 1 {
		s.ReloadTurn = 1
	}
	if s.ReloadTurn > s.Turns {
		s.ReloadTurn = s.Turns
	}
	if s.StablePrefixTokens < 0 {
		s.StablePrefixTokens = 0
	}
	if s.FreshTokensPerTurn < 0 {
		s.FreshTokensPerTurn = 0
	}
	baseline := float64(s.StablePrefixTokens) + float64(s.FreshTokensPerTurn)
	if s.Turns > 1 {
		baseline += float64(s.Turns-1) * float64(s.StablePrefixTokens) * floorReloadCacheReadRate
		baseline += float64(s.Turns-1) * float64(s.FreshTokensPerTurn)
	}
	forfeited := 0
	if s.ReloadTurn > 1 {
		forfeited = s.StablePrefixTokens
	}
	extra := float64(forfeited) * (1 - floorReloadCacheReadRate)
	reload := baseline + extra
	fraction := 0.0
	if reload > 0 {
		fraction = extra / reload
	}
	amortize := 0
	if s.StablePrefixTokens > 0 && s.FreshTokensPerTurn > 0 {
		amortize = int(math.Ceil(extra / float64(s.FreshTokensPerTurn)))
	}
	return FloorReloadCacheResult{Scenario: s, BaselineCostTokens: baseline, ReloadCostTokens: reload, ForfeitedCachedTokens: forfeited, SpendFraction: fraction, TurnsToAmortize: amortize}
}

type FloorReloadCacheReport struct {
	Schema      string                   `json:"schema"`
	Assumptions map[string]string        `json:"assumptions"`
	Results     []FloorReloadCacheResult `json:"results"`
}

func DefaultFloorReloadCacheReport() FloorReloadCacheReport {
	report := FloorReloadCacheReport{
		Schema: "fak.floor-reload-cache/1",
		Assumptions: map[string]string{
			"cache_read_rate": "0.1x token-equivalent",
			"reload_effect":   "one reachability flip changes tools[] and busts the stable prefix once",
			"promotion_path":  "replace modeled bust counts with live #2916 metrics before a performance claim",
		},
	}
	for _, turns := range []int{8, 32, 128} {
		report.Results = append(report.Results, ModelFloorReloadCache(FloorReloadCacheScenario{Turns: turns, ReloadTurn: turns / 2, StablePrefixTokens: 35_800, FreshTokensPerTurn: 2_000}))
	}
	return report
}
