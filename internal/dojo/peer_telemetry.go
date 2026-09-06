package dojo

import (
	"fmt"
)

// peer_telemetry.go is the progressive peer search telemetry and calibration cell (#11668):
// measuring conversion_rate_0_to_1 (fraction of Level 0 roster scans that advance to Level 1 queries),
// conversion_rate_1_to_2 (fraction of Level 1 queries that advance to Level 2 excerpts),
// tokens_saved_ratio (net tokens saved via avoided tool calls relative to peer query overhead),
// and defending the zero-taint floor (taint_leak_rate / taint_leaks).

// Registered claims for progressive peer context search.
// Estimates are genuine seed theories the RSI loop may recalibrate toward measured reality.
// The zero-taint floor is an intentional floor defending against cross-agent data leakage.
var (
	_ = RegisterClaim("peer-search", "conversion_rate_0_to_1", claim(0.35,
		"seed theory (#11668): ~35% of Level 0 peer roster scans convert to Level 1 queries; a genuine estimate the RSI loop recalibrates toward measured reality"))

	_ = RegisterClaim("peer-search", "conversion_rate_1_to_2", claim(0.25,
		"seed theory (#11668): ~25% of Level 1 queries convert to Level 2 excerpts; a genuine estimate the RSI loop recalibrates toward measured reality"))

	_ = RegisterClaim("peer-search", "tokens_saved_ratio", claim(0.40,
		"seed theory (#11668): ~40% net tokens saved ratio via avoided tool calls minus peer query overhead; a genuine estimate the RSI loop recalibrates toward measured reality"))

	_ = RegisterClaim("peer-search", "taint_leak_rate", floor(0.0, true,
		"zero-tolerance safety floor (#11668): peer context search must never leak tainted or unauthorized context across agents; any taint leak breaches this floor"))

	_ = RegisterClaim("peer-search", "taint_leaks", floor(0.0, true,
		"zero-tolerance safety floor (#11668): count of taint leaks during peer search must remain zero"))
)

// PeerSearchTelemetryLedger captures execution counts, token economies, and safety boundaries
// for multi-agent progressive peer context search funnels.
type PeerSearchTelemetryLedger struct {
	Level0Queries     int  `json:"level0_queries"`
	Level1Queries     int  `json:"level1_queries"`
	Level2Queries     int  `json:"level2_queries"`
	AvoidedToolTokens int  `json:"avoided_tool_tokens"`
	PeerQueryTokens   int  `json:"peer_query_tokens"`
	TaintLeaks        int  `json:"taint_leaks"`
	Recorded          bool `json:"recorded"`
}

// NetTokensSaved computes the net token savings: AvoidedToolTokens - PeerQueryTokens.
func (l PeerSearchTelemetryLedger) NetTokensSaved() int {
	return l.AvoidedToolTokens - l.PeerQueryTokens
}

// ConversionRate0to1 computes the conversion rate from Level 0 to Level 1 queries, clamped to [0, 1].
func (l PeerSearchTelemetryLedger) ConversionRate0to1() float64 {
	if l.Level0Queries <= 0 {
		return 0.0
	}
	cr := float64(l.Level1Queries) / float64(l.Level0Queries)
	if cr < 0.0 {
		return 0.0
	}
	if cr > 1.0 {
		return 1.0
	}
	return cr
}

// ConversionRate1to2 computes the conversion rate from Level 1 to Level 2 queries, clamped to [0, 1].
func (l PeerSearchTelemetryLedger) ConversionRate1to2() float64 {
	if l.Level1Queries <= 0 {
		return 0.0
	}
	cr := float64(l.Level2Queries) / float64(l.Level1Queries)
	if cr < 0.0 {
		return 0.0
	}
	if cr > 1.0 {
		return 1.0
	}
	return cr
}

// TokensSavedRatio computes net tokens saved divided by AvoidedToolTokens.
// Returns 0.0 if AvoidedToolTokens <= 0 or if TaintLeaks > 0.
func (l PeerSearchTelemetryLedger) TokensSavedRatio() float64 {
	if l.AvoidedToolTokens <= 0 || l.TaintLeaks > 0 {
		return 0.0
	}
	return float64(l.NetTokensSaved()) / float64(l.AvoidedToolTokens)
}

// PeerSearchEpisodes folds a PeerSearchTelemetryLedger into dojo ScoredInputs:
// - conversion_rate_0_to_1: Level1Queries / Level0Queries (clamped [0, 1])
// - conversion_rate_1_to_2: Level2Queries / Level1Queries (clamped [0, 1])
// - tokens_saved_ratio: (AvoidedToolTokens - PeerQueryTokens) / AvoidedToolTokens (penalized to failure if TaintLeaks > 0)
// - taint_leak_rate: zero-tolerance intentional floor (breached if TaintLeaks > 0)
// - taint_leaks: zero-tolerance intentional count floor (breached if TaintLeaks > 0)
func PeerSearchEpisodes(ledger PeerSearchTelemetryLedger) []ScoredInput {
	var episodes []ScoredInput

	// 1. conversion_rate_0_to_1
	cr01Pred := Registry.MustPredict("peer-search", "conversion_rate_0_to_1", "fraction")
	if !ledger.Recorded || ledger.Level0Queries <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: cr01Pred,
			Outcome: Outcome{
				Measured: false,
				Sample:   ledger.Level0Queries,
				Source:   "no Level 0 queries recorded in telemetry — conversion_rate_0_to_1 is UNMEASURED",
			},
		})
	} else {
		cr01 := ledger.ConversionRate0to1()
		episodes = append(episodes, ScoredInput{
			Prediction: cr01Pred,
			Outcome: Outcome{
				Realized:   cr01,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     ledger.Level0Queries,
				Source: fmt.Sprintf("%d Level 1 queries from %d Level 0 queries (cr=%.2f) (WITNESSED)",
					ledger.Level1Queries, ledger.Level0Queries, cr01),
			},
		})
	}

	// 2. conversion_rate_1_to_2
	cr12Pred := Registry.MustPredict("peer-search", "conversion_rate_1_to_2", "fraction")
	if !ledger.Recorded || ledger.Level1Queries <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: cr12Pred,
			Outcome: Outcome{
				Measured: false,
				Sample:   ledger.Level1Queries,
				Source:   "no Level 1 queries recorded in telemetry — conversion_rate_1_to_2 is UNMEASURED",
			},
		})
	} else {
		cr12 := ledger.ConversionRate1to2()
		episodes = append(episodes, ScoredInput{
			Prediction: cr12Pred,
			Outcome: Outcome{
				Realized:   cr12,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     ledger.Level1Queries,
				Source: fmt.Sprintf("%d Level 2 queries from %d Level 1 queries (cr=%.2f) (WITNESSED)",
					ledger.Level2Queries, ledger.Level1Queries, cr12),
			},
		})
	}

	// 3. tokens_saved_ratio
	tokensPred := Registry.MustPredict("peer-search", "tokens_saved_ratio", "fraction")
	if !ledger.Recorded || ledger.AvoidedToolTokens <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: tokensPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   ledger.AvoidedToolTokens,
				Source:   "no avoided tool tokens recorded in telemetry — tokens_saved_ratio is UNMEASURED",
			},
		})
	} else if ledger.TaintLeaks > 0 {
		// Zero-taint floor breached: penalizes calibration and sets failure outcome (0.0 savings credited).
		episodes = append(episodes, ScoredInput{
			Prediction: tokensPred,
			Outcome: Outcome{
				Realized:   0.0,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     ledger.AvoidedToolTokens,
				Source: fmt.Sprintf("taint leaks detected (%d): zero-taint floor breached, token savings penalized to 0.0 (WITNESSED)",
					ledger.TaintLeaks),
			},
		})
	} else {
		netSaved := ledger.NetTokensSaved()
		ratio := float64(netSaved) / float64(ledger.AvoidedToolTokens)
		episodes = append(episodes, ScoredInput{
			Prediction: tokensPred,
			Outcome: Outcome{
				Realized:   ratio,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     ledger.AvoidedToolTokens,
				Source: fmt.Sprintf("net %d tokens saved (avoided %d - query %d) over %d avoided tool tokens (ratio=%.2f) (WITNESSED)",
					netSaved, ledger.AvoidedToolTokens, ledger.PeerQueryTokens, ledger.AvoidedToolTokens, ratio),
			},
		})
	}

	// 4. taint_leak_rate (zero-taint floor)
	taintPred := Registry.MustPredict("peer-search", "taint_leak_rate", "rate")
	if !ledger.Recorded {
		episodes = append(episodes, ScoredInput{
			Prediction: taintPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   0,
				Source:   "peer search telemetry not recorded — taint_leak_rate is UNMEASURED",
			},
		})
	} else if ledger.TaintLeaks > 0 {
		sample := ledger.Level0Queries
		if sample < 0 {
			sample = 0
		}
		episodes = append(episodes, ScoredInput{
			Prediction: taintPred,
			Outcome: Outcome{
				Realized:   float64(ledger.TaintLeaks),
				Provenance: Witnessed,
				Measured:   true,
				Sample:     sample,
				Source: fmt.Sprintf("%d taint leak(s) detected during peer search — zero-taint floor BREACHED (WITNESSED)",
					ledger.TaintLeaks),
			},
		})
	} else {
		sample := ledger.Level0Queries
		if sample < 0 {
			sample = 0
		}
		episodes = append(episodes, ScoredInput{
			Prediction: taintPred,
			Outcome: Outcome{
				Realized:   0.0,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     sample,
				Source:     "zero taint leaks detected during peer context search (zero-taint floor holds) (WITNESSED)",
			},
		})
	}

	// 5. taint_leaks (zero-taint floor count)
	taintLeaksPred := Registry.MustPredict("peer-search", "taint_leaks", "count")
	if !ledger.Recorded {
		episodes = append(episodes, ScoredInput{
			Prediction: taintLeaksPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   0,
				Source:   "peer search telemetry not recorded — taint_leaks is UNMEASURED",
			},
		})
	} else {
		sample := ledger.Level0Queries
		if sample < 0 {
			sample = 0
		}
		var source string
		if ledger.TaintLeaks > 0 {
			source = fmt.Sprintf("%d taint leak(s) detected during peer search — zero-taint floor BREACHED (WITNESSED)",
				ledger.TaintLeaks)
		} else {
			source = "zero taint leaks detected during peer context search (zero-taint floor holds) (WITNESSED)"
		}
		episodes = append(episodes, ScoredInput{
			Prediction: taintLeaksPred,
			Outcome: Outcome{
				Realized:   float64(ledger.TaintLeaks),
				Provenance: Witnessed,
				Measured:   true,
				Sample:     sample,
				Source:     source,
			},
		})
	}

	return episodes
}
