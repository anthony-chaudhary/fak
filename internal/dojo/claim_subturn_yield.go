package dojo

import "fmt"

// claim_subturn_yield.go is the subturn yield calibration cell (#11654):
// measuring yield_compaction_efficacy (the fraction of prompt tokens compacted
// when client resumes following a SubturnYieldMessage) and subturn_resubmission_loop_rate
// (an intentional floor defending against clients that echo uncompacted prompts in loops).

// The anchored literals for this cell.
var _ = RegisterClaim("subturn-yield", "yield_compaction_efficacy", claim(0.40,
	"seed theory (#11654): subturn yield valve activations shed ~40% of context tokens upon client resumption; a genuine estimate the RSI loop recalibrates toward measured reality"))

var _ = RegisterClaim("subturn-yield", "subturn_resubmission_loop_rate", floor(0.0, true,
	"zero-tolerance safety floor (#11654): subturn yield messages must never induce client prompt echo or uncompacted resubmission loops; any uncompacted resubmission breaches this floor"))

// SubturnYieldLedger is the reduced view of subturn yield events and compaction efficacy.
type SubturnYieldLedger struct {
	// YieldEvents is the total count of SubturnYieldMessages emitted.
	YieldEvents int
	// TokensBeforeYield is total tokens across prompts immediately prior to yield.
	TokensBeforeYield int
	// TokensAfterYield is total tokens across prompts on client continuation immediately after yield.
	TokensAfterYield int
	// YieldRecorded indicates whether yield events and continuation tokens were tracked.
	YieldRecorded bool
	// UncompactedResubmits is how many continuations arrived with prompt tokens >= before yield.
	UncompactedResubmits int
}

// SubturnYieldEpisodes folds the subturn-yield ledger into dojo ScoredInputs:
// one for yield_compaction_efficacy, and one for subturn_resubmission_loop_rate.
func SubturnYieldEpisodes(led SubturnYieldLedger) []ScoredInput {
	var episodes []ScoredInput

	// 1. yield_compaction_efficacy
	yieldPred := Registry.MustPredict("subturn-yield", "yield_compaction_efficacy", "fraction")
	if !led.YieldRecorded || led.YieldEvents <= 0 || led.TokensBeforeYield <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: yieldPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   led.YieldEvents,
				Source:   "no subturn yield events recorded in telemetry — yield compaction efficacy is UNMEASURED",
			},
		})
	} else {
		shed := led.TokensBeforeYield - led.TokensAfterYield
		efficacy := float64(shed) / float64(led.TokensBeforeYield)
		if efficacy < -1.0 {
			efficacy = -1.0
		}
		if efficacy > 1.0 {
			efficacy = 1.0
		}
		episodes = append(episodes, ScoredInput{
			Prediction: yieldPred,
			Outcome: Outcome{
				Realized:   efficacy,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     led.YieldEvents,
				Source: fmt.Sprintf("%d tokens shed across %d yield event(s) (before=%d after=%d) (WITNESSED)",
					shed, led.YieldEvents, led.TokensBeforeYield, led.TokensAfterYield),
			},
		})
	}

	// 2. subturn_resubmission_loop_rate
	loopPred := Registry.MustPredict("subturn-yield", "subturn_resubmission_loop_rate", "fraction")
	if !led.YieldRecorded || led.YieldEvents <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: loopPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   led.YieldEvents,
				Source:   "no yield events recorded — subturn resubmission loop rate is UNMEASURED",
			},
		})
	} else {
		loops := led.UncompactedResubmits
		if loops < 0 {
			loops = 0
		}
		loopRate := float64(loops) / float64(led.YieldEvents)
		episodes = append(episodes, ScoredInput{
			Prediction: loopPred,
			Outcome: Outcome{
				Realized:   loopRate,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     led.YieldEvents,
				Source: fmt.Sprintf("%d uncompacted prompt resubmission(s) across %d yield event(s) (WITNESSED)",
					loops, led.YieldEvents),
			},
		})
	}

	return episodes
}
