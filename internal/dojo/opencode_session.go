package dojo

import (
	"fmt"
)

// opencode_session.go defines the opencode-session lever and prediction claims (#11669):
// scoring and tracking cache_read_share, turns_per_task, and compaction_shed_ratio
// with calibration error trending for opencode agentic workflows in the dojo gym.

// Anchored literals for opencode-session claims registered via the additive seam.
var _ = RegisterClaim("opencode-session", "cache_read_share", claim(0.80,
	"seed theory (#11669): ~80% of billed input tokens in opencode agentic sessions are served as prefix cache reads; a genuine estimate the RSI loop recalibrates toward measured reality"))

var _ = RegisterClaim("opencode-session", "turns_per_task", Claim{
	Claimed:       16.0,
	LowerIsBetter: true,
	Basis:         "seed theory (#11669): opencode completes an agentic task in about 16 assistant turns; a genuine estimate the RSI loop recalibrates toward measured reality",
})

var _ = RegisterClaim("opencode-session", "compaction_shed_ratio", claim(0.40,
	"seed theory (#11669): opencode compaction shed line sheds ~40% of context tokens upon trigger threshold; a genuine estimate the RSI loop recalibrates toward measured reality"))

// OpencodeSessionLedger captures telemetry facts from opencode session executions.
type OpencodeSessionLedger struct {
	// SessionID optionally identifies the recorded session.
	SessionID string `json:"session_id,omitempty"`

	// Cache telemetry: input tokens, cache reads, cache creation writes.
	InputTokens         int  `json:"input_tokens"`
	CacheReadTokens     int  `json:"cache_read_tokens"`
	CacheCreationTokens int  `json:"cache_creation_tokens"`
	CacheRecorded       bool `json:"cache_recorded"`

	// Turn telemetry: assistant turns and completed task count.
	TotalTurns     int  `json:"total_turns"`
	CompletedTasks int  `json:"completed_tasks"`
	TurnsRecorded  bool `json:"turns_recorded"`

	// Compaction telemetry: context tokens before/after compaction trigger.
	TokensBeforeCompaction int  `json:"tokens_before_compaction"`
	TokensAfterCompaction  int  `json:"tokens_after_compaction"`
	CompactionEvents       int  `json:"compaction_events"`
	CompactionRecorded     bool `json:"compaction_recorded"`
}

// OpencodeSessionOutcome extracts the measured outcome for a given metric from session telemetry.
func OpencodeSessionOutcome(led OpencodeSessionLedger, metric string) Outcome {
	switch metric {
	case "cache_read_share":
		totalInput := led.InputTokens + led.CacheReadTokens + led.CacheCreationTokens
		if !led.CacheRecorded || totalInput <= 0 {
			return Outcome{
				Measured: false,
				Sample:   totalInput,
				Source:   "no billed input tokens recorded in opencode session telemetry — cache_read_share is UNMEASURED",
			}
		}
		share := float64(led.CacheReadTokens) / float64(totalInput)
		if share < 0.0 {
			share = 0.0
		} else if share > 1.0 {
			share = 1.0
		}
		return Outcome{
			Realized:   share,
			Provenance: Witnessed,
			Measured:   true,
			Sample:     totalInput,
			Source: fmt.Sprintf("%d of %d billed input tokens served from prefix cache in opencode session (WITNESSED)",
				led.CacheReadTokens, totalInput),
		}

	case "turns_per_task":
		if !led.TurnsRecorded || led.CompletedTasks <= 0 || led.TotalTurns < 0 {
			return Outcome{
				Measured: false,
				Sample:   led.CompletedTasks,
				Source:   "no completed tasks recorded in opencode session telemetry — turns_per_task is UNMEASURED",
			}
		}
		tpt := float64(led.TotalTurns) / float64(led.CompletedTasks)
		return Outcome{
			Realized:   tpt,
			Provenance: Witnessed,
			Measured:   true,
			Sample:     led.CompletedTasks,
			Source: fmt.Sprintf("%d assistant turns across %d completed task(s) in opencode session (WITNESSED)",
				led.TotalTurns, led.CompletedTasks),
		}

	case "compaction_shed_ratio":
		if !led.CompactionRecorded || led.TokensBeforeCompaction <= 0 || led.CompactionEvents <= 0 {
			return Outcome{
				Measured: false,
				Sample:   led.TokensBeforeCompaction,
				Source:   "no compaction events recorded in opencode session telemetry — compaction_shed_ratio is UNMEASURED",
			}
		}
		shed := led.TokensBeforeCompaction - led.TokensAfterCompaction
		ratio := float64(shed) / float64(led.TokensBeforeCompaction)
		if ratio < -1.0 {
			ratio = -1.0
		} else if ratio > 1.0 {
			ratio = 1.0
		}
		return Outcome{
			Realized:   ratio,
			Provenance: Witnessed,
			Measured:   true,
			Sample:     led.TokensBeforeCompaction,
			Source: fmt.Sprintf("%d tokens shed across %d compaction event(s) (before=%d after=%d) (WITNESSED)",
				shed, led.CompactionEvents, led.TokensBeforeCompaction, led.TokensAfterCompaction),
		}

	default:
		return Outcome{
			Measured: false,
			Source:   fmt.Sprintf("unknown metric %q for opencode-session", metric),
		}
	}
}

// OpencodeSessionEpisodes folds an opencode session telemetry ledger into three dojo ScoredInputs:
// cache_read_share, turns_per_task, and compaction_shed_ratio.
func OpencodeSessionEpisodes(led OpencodeSessionLedger) []ScoredInput {
	return OpencodeSessionEpisodesWithPredictions(led, nil)
}

// OpencodeSessionEpisodesWithPredictions folds an opencode session telemetry ledger
// using candidate predictions for the opencode-session metrics.
// Any metric omitted from candidatePreds defaults to the registered claim.
func OpencodeSessionEpisodesWithPredictions(led OpencodeSessionLedger, candidatePreds map[string]Prediction) []ScoredInput {
	metrics := []struct {
		name string
		unit string
	}{
		{"cache_read_share", "fraction"},
		{"turns_per_task", "turns"},
		{"compaction_shed_ratio", "fraction"},
	}

	episodes := make([]ScoredInput, 0, len(metrics))
	for _, m := range metrics {
		pred, ok := candidatePreds[m.name]
		if !ok {
			pred = Registry.MustPredict(OpencodeSessionLeverName, m.name, m.unit)
		}
		outcome := OpencodeSessionOutcome(led, m.name)
		episodes = append(episodes, ScoredInput{
			Prediction: pred,
			Outcome:    outcome,
		})
	}
	return episodes
}

// EvaluateCandidatePredictions scores candidate predictions against an opencode session ledger.
func EvaluateCandidatePredictions(scenario string, led OpencodeSessionLedger, candidates []Prediction, band CalibBand) []Episode {
	episodes := make([]Episode, 0, len(candidates))
	for _, cand := range candidates {
		outcome := OpencodeSessionOutcome(led, cand.Metric)
		episodes = append(episodes, Score(scenario, cand, outcome, band))
	}
	return episodes
}

// MultiOpencodeSessionEpisodes aggregates multiple session ledgers into a single set of ScoredInputs.
func MultiOpencodeSessionEpisodes(ledgers []OpencodeSessionLedger) []ScoredInput {
	var agg OpencodeSessionLedger
	for _, l := range ledgers {
		if l.CacheRecorded {
			agg.CacheRecorded = true
			agg.InputTokens += l.InputTokens
			agg.CacheReadTokens += l.CacheReadTokens
			agg.CacheCreationTokens += l.CacheCreationTokens
		}
		if l.TurnsRecorded {
			agg.TurnsRecorded = true
			agg.TotalTurns += l.TotalTurns
			agg.CompletedTasks += l.CompletedTasks
		}
		if l.CompactionRecorded {
			agg.CompactionRecorded = true
			agg.TokensBeforeCompaction += l.TokensBeforeCompaction
			agg.TokensAfterCompaction += l.TokensAfterCompaction
			agg.CompactionEvents += l.CompactionEvents
		}
	}
	return OpencodeSessionEpisodes(agg)
}

// OpencodeSessionLeverName is the canonical name for the opencode session lever.
const OpencodeSessionLeverName = "opencode-session"

// OpencodeSessionLever is the official dojo gym lever for evaluating opencode sessions.
type OpencodeSessionLever struct {
	Ledger      OpencodeSessionLedger
	Predictions map[string]Prediction
}

// NewOpencodeSessionLever creates an OpencodeSessionLever with the provided telemetry ledger.
func NewOpencodeSessionLever(led OpencodeSessionLedger) *OpencodeSessionLever {
	return &OpencodeSessionLever{Ledger: led}
}

// WithPredictions returns a copy of the lever configured with custom candidate predictions.
func (l *OpencodeSessionLever) WithPredictions(preds map[string]Prediction) *OpencodeSessionLever {
	return &OpencodeSessionLever{
		Ledger:      l.Ledger,
		Predictions: preds,
	}
}

// Name returns the lever name "opencode-session".
func (l *OpencodeSessionLever) Name() string {
	return OpencodeSessionLeverName
}

// Episodes returns the scored inputs for the lever over a scenario.
func (l *OpencodeSessionLever) Episodes(s Scenario) ([]ScoredInput, error) {
	if len(l.Predictions) > 0 {
		return OpencodeSessionEpisodesWithPredictions(l.Ledger, l.Predictions), nil
	}
	return OpencodeSessionEpisodes(l.Ledger), nil
}

// DefaultOpencodeSessionLever registers the opencode-session lever in the dojo lever registry.
var DefaultOpencodeSessionLever = RegisterLever(NewOpencodeSessionLever(OpencodeSessionLedger{}))

// OpencodeSessionReport generates a folded dojo claim report from an opencode session ledger,
// optionally attaching calibration error trending against prior ledger rows.
func OpencodeSessionReport(scenario string, led OpencodeSessionLedger, opts FoldOpts, prior []LedgerRow) Report {
	inputs := OpencodeSessionEpisodes(led)
	band := DefaultCalibBand()
	episodes := make([]Episode, 0, len(inputs))
	for _, in := range inputs {
		episodes = append(episodes, Score(scenario, in.Prediction, in.Outcome, band))
	}
	rep := Fold(episodes, opts)
	if len(prior) > 0 {
		row := RowFromReport(rep)
		tr := TrendVsLast(row, prior)
		rep.Trend = &tr
	}
	return rep
}
