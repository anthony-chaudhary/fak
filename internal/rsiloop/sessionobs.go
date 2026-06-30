package rsiloop

// sessionobs.go wires the session->outcome observability scorecard into the RSI
// engine as an objective in its own right. The objective here is the S0 loop-index:
// the full agentic-loop score emitted by internal/loopindex, with its Learn stage
// derived from internal/sessionobs. Higher is better, and the normal shipgate
// keep-bit still decides KEEP/REVERT.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/loopindex"
	"github.com/anthony-chaudhary/fak/internal/sessionobs"
)

// SessionObsMetricName labels the S0 loop-index in the RSI journal. It is the
// full agentic-loop index from internal/loopindex, with the Learn stage derived
// from internal/sessionobs.Score. A candidate must increase it strictly to count
// as an improvement.
const SessionObsMetricName = "s0_loop_index"

// SessionObsState is the scored state of the session-observability toolchain: a
// scrubbed session corpus plus the pipeline facts that say whether the corpus is
// committed and consumed by a loop.
type SessionObsState struct {
	Corpus   []sessionobs.Record
	Pipeline sessionobs.Pipeline
}

// SessionObsProposal is one proposed toolchain change to score as an RSI candidate.
// The harness treats it as data, runs sessionobs.Score, and derives the witness
// fields from that report; the proposal never supplies its own keep-bit.
type SessionObsProposal struct {
	Label string
	State SessionObsState
}

// NewSessionObsHarness drives the S0 loop-index through the generic RSI loop. The
// baseline and each proposal are copied so a caller cannot mutate the evidence
// out from under a running loop.
func NewSessionObsHarness(baseline SessionObsState, proposals []SessionObsProposal) Harness {
	base := cloneSessionObsState(baseline)
	props := cloneSessionObsProposals(proposals)
	return Harness{
		MetricName:      SessionObsMetricName,
		LowerBetter:     false,
		BaselineRefName: "sessionobs-s0",
		BaselineMetric: func() (float64, string, error) {
			s0, sobs := scoreSessionObsS0(base)
			return float64(s0.Corpus.LoopIndex), sessionObsRef(s0, sobs), nil
		},
		Candidates: func() []Candidate {
			out := make([]Candidate, len(props))
			for i, p := range props {
				out[i] = Candidate{Label: p.Label, Payload: i}
			}
			return out
		},
		Measure: func(c Candidate) (Measurement, error) {
			i, ok := c.Payload.(int)
			if !ok || i < 0 || i >= len(props) {
				return Measurement{}, fmt.Errorf("sessionobs proposal index %v out of range", c.Payload)
			}
			p := props[i]
			s0, sobs := scoreSessionObsS0(p.State)
			return Measurement{
				Metric:     float64(s0.Corpus.LoopIndex),
				SuiteGreen: sobs.Corpus.Sessions > 0,
				TruthClean: s0.OK && sobs.OK,
				Score:      sessionObsScorecard(s0, sobs, p.State.Pipeline),
				Note: fmt.Sprintf("loop_index=%d loopindex_debt=%d sessionobs_debt=%d linked=%.0f%% loop_consumes=%v",
					s0.Corpus.LoopIndex, s0.Corpus.LoopIndexDebt, sobs.Corpus.SessionObsDebt,
					100*sobs.Corpus.LinkedFrac, p.State.Pipeline.LoopConsumes),
			}, nil
		},
	}
}

func sessionObsScorecard(s0 loopindex.Report, sobs sessionobs.Report, pipe sessionobs.Pipeline) *Scorecard {
	loopConsumes := 0.0
	if pipe.LoopConsumes {
		loopConsumes = 1.0
	}
	corpusCommitted := 0.0
	if pipe.CorpusCommitted {
		corpusCommitted = 1.0
	}
	return &Scorecard{
		Name:  "sessionobs_s0",
		Value: float64(s0.Corpus.LoopIndex),
		Grade: s0.Corpus.Grade,
		Components: []ScoreComponent{
			{Name: "loop_index", Value: float64(s0.Corpus.LoopIndex), Unit: "score"},
			{Name: "witnessed_index", Value: float64(s0.Corpus.WitnessedIndex), Unit: "score"},
			{Name: "loopindex_debt", Value: float64(s0.Corpus.LoopIndexDebt), Unit: "debt"},
			{Name: "sessionobs_score", Value: float64(sobs.Corpus.Score), Unit: "score"},
			{Name: "sessionobs_debt", Value: float64(sobs.Corpus.SessionObsDebt), Unit: "debt"},
			{Name: "sessions", Value: float64(sobs.Corpus.Sessions), Unit: "sessions"},
			{Name: "linked_frac", Value: sobs.Corpus.LinkedFrac, Unit: "ratio"},
			{Name: "value_frac", Value: sobs.Corpus.ValueFrac, Unit: "ratio"},
			{Name: "waste_frac", Value: sobs.Corpus.WasteFrac, Unit: "ratio"},
			{Name: "corpus_committed", Value: corpusCommitted, Unit: "bool"},
			{Name: "loop_consumes", Value: loopConsumes, Unit: "bool"},
		},
	}
}

// NewSessionObsDemoHarness is the deterministic end-to-end witness used by
// cmd/rsiloop -harness sessionobs. It first proposes a no-op toolchain change
// (REVERT: S0 does not move), then proposes the closed session->outcome->loop
// state (KEEP: linked outcomes, committed corpus, and a consuming loop raise the
// S0 loop-index to 100).
func NewSessionObsDemoHarness() Harness {
	baseline := SessionObsState{
		Corpus: []sessionobs.Record{
			{SessionID: "captured-value", AssistantTurns: 9, ToolCalls: 16, Outcome: sessionobs.OutcomeUnknown},
			{SessionID: "captured-waste", AssistantTurns: 7, ToolCalls: 10, Outcome: sessionobs.OutcomeUnknown},
		},
	}
	closed := SessionObsState{
		Corpus: []sessionobs.Record{
			{
				SessionID:      "captured-value",
				AssistantTurns: 9,
				ToolCalls:      16,
				Outcome:        sessionobs.OutcomeShipped,
				Signals:        sessionobs.Signals{Commits: 1, GoalEvents: 1},
			},
			{
				SessionID:      "captured-waste",
				AssistantTurns: 7,
				ToolCalls:      10,
				Outcome:        sessionobs.OutcomeStopped,
				Signals:        sessionobs.Signals{StopEvents: 1, GuardRefusals: 1},
			},
		},
		Pipeline: sessionobs.Pipeline{CorpusCommitted: true, LoopConsumes: true, Registered: true},
	}
	return NewSessionObsHarness(baseline, []SessionObsProposal{
		{Label: "sessionobs:no-op-toolchain", State: baseline},
		{Label: "sessionobs:link-outcomes-and-consume-s0", State: closed},
	})
}

func scoreSessionObsS0(st SessionObsState) (loopindex.Report, sessionobs.Report) {
	sobs := sessionobs.Score(st.Corpus, st.Pipeline)
	return loopindex.Score(sessionObsS0Loop(sobs)), sobs
}

func sessionObsS0Loop(sobs sessionobs.Report) loopindex.Loop {
	return loopindex.Loop{Stages: []loopindex.Stage{
		closedS0Stage(loopindex.StageOrient, "recall-staleness / context-thrash"),
		closedS0Stage(loopindex.StagePlan, "collision-priced fan-out coverage"),
		closedS0Stage(loopindex.StageAct, "malformed-call repair rate"),
		closedS0Stage(loopindex.StageVerify, "unwitnessed-done rate"),
		closedS0Stage(loopindex.StageShip, "green-gate latency budget"),
		sessionObsLearnStage(sobs),
	}}
}

func closedS0Stage(name, signal string) loopindex.Stage {
	return loopindex.Stage{
		Name: name, Signal: signal, Floor: 0.6,
		Probes: []loopindex.Probe{
			{Name: name + "_keystone", Detail: "stage held fixed for the sessionobs S0 witness", Keystone: true, Pass: true},
			{Name: name + "_support", Detail: "stage held fixed for the sessionobs S0 witness", Pass: true},
		},
	}
}

func sessionObsLearnStage(sobs sessionobs.Report) loopindex.Stage {
	return loopindex.Stage{
		Name: loopindex.StageLearn, Signal: "session->outcome link coverage", Floor: 1.0,
		Probes: []loopindex.Probe{
			{Name: "sessionobs_scorer", Detail: "internal/sessionobs scores the capture->link->learn ladder", Keystone: true, Pass: true},
			{Name: "outcome_link", Detail: "sessions carry linked value-vs-waste outcomes", Keystone: true,
				Pass: sessionObsKPIPassed(sobs, "outcome_link_rate") && sessionObsKPIPassed(sobs, "value_waste_separable")},
			{Name: "consuming_loop", Detail: "a registered RSI loop consumes the session outcome corpus", Keystone: true,
				Pass: sessionObsKPIPassed(sobs, "loop_consumes")},
			{Name: "corpus_committed", Detail: "the scrubbed session corpus is committed and reproducible",
				Pass: sessionObsKPIPassed(sobs, "corpus_committed")},
			{Name: "behavior_signal", Detail: "records carry behavior features a loop can contrast",
				Pass: sessionObsKPIPassed(sobs, "behavior_signal_present")},
		},
	}
}

func sessionObsKPIPassed(rep sessionobs.Report, name string) bool {
	for _, k := range rep.KPIs {
		if k.Name == name {
			return k.Debt == 0 && k.Score > 0
		}
	}
	return false
}

func sessionObsRef(s0 loopindex.Report, sobs sessionobs.Report) string {
	return fmt.Sprintf("sessionobs@sessions=%d,loop_index=%d,sessionobs_debt=%d",
		sobs.Corpus.Sessions, s0.Corpus.LoopIndex, sobs.Corpus.SessionObsDebt)
}

func cloneSessionObsState(in SessionObsState) SessionObsState {
	out := SessionObsState{Pipeline: in.Pipeline}
	if len(in.Corpus) > 0 {
		out.Corpus = append([]sessionobs.Record(nil), in.Corpus...)
	}
	return out
}

func cloneSessionObsProposals(in []SessionObsProposal) []SessionObsProposal {
	out := make([]SessionObsProposal, len(in))
	for i, p := range in {
		out[i] = SessionObsProposal{Label: p.Label, State: cloneSessionObsState(p.State)}
	}
	return out
}
