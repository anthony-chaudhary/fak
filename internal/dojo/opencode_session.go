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

// OpencodeSessionEpisodes folds an opencode session telemetry ledger into three dojo ScoredInputs:
// cache_read_share, turns_per_task, and compaction_shed_ratio.
func OpencodeSessionEpisodes(led OpencodeSessionLedger) []ScoredInput {
	var episodes []ScoredInput

	// 1. cache_read_share
	cachePred := Registry.MustPredict("opencode-session", "cache_read_share", "fraction")
	totalInput := led.InputTokens + led.CacheReadTokens + led.CacheCreationTokens
	if !led.CacheRecorded || totalInput <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: cachePred,
			Outcome: Outcome{
				Measured: false,
				Sample:   totalInput,
				Source:   "no billed input tokens recorded in opencode session telemetry — cache_read_share is UNMEASURED",
			},
		})
	} else {
		share := float64(led.CacheReadTokens) / float64(totalInput)
		if share < 0.0 {
			share = 0.0
		} else if share > 1.0 {
			share = 1.0
		}
		episodes = append(episodes, ScoredInput{
			Prediction: cachePred,
			Outcome: Outcome{
				Realized:   share,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     totalInput,
				Source: fmt.Sprintf("%d of %d billed input tokens served from prefix cache in opencode session (WITNESSED)",
					led.CacheReadTokens, totalInput),
			},
		})
	}

	// 2. turns_per_task
	turnsPred := Registry.MustPredict("opencode-session", "turns_per_task", "turns")
	if !led.TurnsRecorded || led.TotalTurns < 0 || led.CompletedTasks <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: turnsPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   led.CompletedTasks,
				Source:   "no completed tasks recorded in opencode session telemetry — turns_per_task is UNMEASURED",
			},
		})
	} else {
		tpt := float64(led.TotalTurns) / float64(led.CompletedTasks)
		episodes = append(episodes, ScoredInput{
			Prediction: turnsPred,
			Outcome: Outcome{
				Realized:   tpt,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     led.CompletedTasks,
				Source: fmt.Sprintf("%d assistant turns across %d completed task(s) in opencode session (WITNESSED)",
					led.TotalTurns, led.CompletedTasks),
			},
		})
	}

	// 3. compaction_shed_ratio
	compactionPred := Registry.MustPredict("opencode-session", "compaction_shed_ratio", "fraction")
	if !led.CompactionRecorded || led.TokensBeforeCompaction <= 0 || led.CompactionEvents <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: compactionPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   led.TokensBeforeCompaction,
				Source:   "no compaction events recorded in opencode session telemetry — compaction_shed_ratio is UNMEASURED",
			},
		})
	} else {
		shed := led.TokensBeforeCompaction - led.TokensAfterCompaction
		ratio := float64(shed) / float64(led.TokensBeforeCompaction)
		if ratio < -1.0 {
			ratio = -1.0
		} else if ratio > 1.0 {
			ratio = 1.0
		}
		episodes = append(episodes, ScoredInput{
			Prediction: compactionPred,
			Outcome: Outcome{
				Realized:   ratio,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     led.TokensBeforeCompaction,
				Source: fmt.Sprintf("%d tokens shed across %d compaction event(s) (before=%d after=%d) (WITNESSED)",
					shed, led.CompactionEvents, led.TokensBeforeCompaction, led.TokensAfterCompaction),
			},
		})
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
	Ledger OpencodeSessionLedger
}

// NewOpencodeSessionLever creates an OpencodeSessionLever with the provided telemetry ledger.
func NewOpencodeSessionLever(led OpencodeSessionLedger) *OpencodeSessionLever {
	return &OpencodeSessionLever{Ledger: led}
}

// Name returns the lever name "opencode-session".
func (l *OpencodeSessionLever) Name() string {
	return OpencodeSessionLeverName
}

// Episodes returns the scored inputs for the lever over a scenario.
func (l *OpencodeSessionLever) Episodes(s Scenario) ([]ScoredInput, error) {
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
