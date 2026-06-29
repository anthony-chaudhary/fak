// loopverify.go is the verified-vs-naive loop bench (issue #1190, part of the
// #1173 verified-loop epic). It answers the question that justifies the whole
// epic with a number you can move and prove you moved: does a dos-witnessed
// exit-gate actually beat a self-reported one, quoted against the *tuned* naive
// loop rather than a strawman?
//
// It replays the SAME fixed task corpus through two loop drivers:
//
//   - the NAIVE Ralph loop, which terminates the moment the agent self-reports
//     "done" (the dominant ecosystem primitive: trust the model that wrote the
//     code), and
//   - the GATED driver (#1173 #S1+#S2), which terminates only when `dos`
//     witnesses the done from git evidence (commit-audit OK / verify SHIPPED),
//     refusing a self-reported "done" that dos refutes (CLAIM_UNWITNESSED /
//     NOT_SHIPPED) and re-arming the loop instead.
//
// and reports the delta across the four metrics the issue names:
//
//   - false-done rate: turns a loop accepted as done that dos refutes,
//   - slop delta: net slop-scorecard movement attributable to each loop's
//     shipped commits,
//   - iterations-to-witnessed-done and wasted iterations (rework forced by a
//     falsely-accepted "done"), and
//   - gate cost: the dos-adjudication overhead the gated loop pays, so the win
//     is reported NET (the repo's net-true doctrine,
//     docs/standards/net-true-value.md).
//
// PROVENANCE (net-true criterion 4): the corpus here is a SIMULATED fixture, not
// a live agent run — the kernel-native loop DRIVER (#1173 #S1) does not exist
// yet, so a live naive-vs-gated wall-clock is honestly `not yet`. What this bench
// proves today is the MEASUREMENT: a re-runnable command that computes the four
// metrics deterministically over a comparable corpus and shows the gated loop's
// false-done rate is strictly lower. When the live driver lands, the same report
// shape is fed real turn records instead of the fixture; nothing else changes.
//
// Re-run: `go test ./internal/bench/ -run TestLoopVerifyBench` (the report is
// also regenerable into testdata/loopverify_report.json with UPDATE_GOLDEN=1).
package bench

import (
	"encoding/json"
	"fmt"
	"math"
)

// dos verdicts a turn's commit can carry, mirroring the dos truth syscall:
//   - VerdictWitnessed: commit-audit OK / verify SHIPPED — the done is real.
//   - VerdictRefuted:   CLAIM_UNWITNESSED / NOT_SHIPPED — the "done" is a claim
//     the diff does not back (the self-assessment trap #1173 names).
//   - VerdictWorking:   no done claimed this turn (work in progress).
const (
	VerdictWitnessed = "witnessed"
	VerdictRefuted   = "refuted"
	VerdictWorking   = "working"
)

// Turn is one iteration of a loop over a task: what the agent claimed, what dos
// witnessed, the net slop its commit moved, and the adjudication cost the gate
// pays to grade it.
type Turn struct {
	// SelfReportedDone is true when the agent declares the task complete this
	// turn — the only signal the naive loop has.
	SelfReportedDone bool `json:"self_reported_done"`
	// DosVerdict is what dos finds from git evidence for this turn's commit.
	DosVerdict string `json:"dos_verdict"`
	// SlopIntroduced is the net slop-scorecard delta of this turn's commit; a
	// fix turn that removes earlier slop is negative.
	SlopIntroduced int `json:"slop_introduced"`
	// GateCostUnits is the dos-adjudication overhead to grade this turn, in
	// normalized adjudication-runs (the gate cost the gated loop pays; the naive
	// loop pays none because it never adjudicates).
	GateCostUnits float64 `json:"gate_cost_units"`
	// Note is a one-line human description of the turn.
	Note string `json:"note,omitempty"`
}

// Episode is one task run as an ordered sequence of turns.
type Episode struct {
	Name  string `json:"name"`
	Turns []Turn `json:"turns"`
}

// DefaultLoopCorpus is the fixed, small task set both loops replay. It is
// deliberately comparable (same episodes, same turns) so the only difference in
// the report is the EXIT RULE — self-report vs witnessed. It mixes episodes that
// separate the two loops (an early false "done" followed by real fix turns) with
// episodes that do NOT (the first self-report is already witnessed), so the bench
// cannot be accused of rigging a strawman: where the naive loop happens to be
// right, the report shows no gap.
func DefaultLoopCorpus() []Episode {
	return []Episode{
		{
			// Classic self-assessment trap: a `fix:` whose diff only touched a
			// comment is self-reported done but refuted; the real fix (a test
			// that reads back) lands two turns later.
			Name: "fix-gateway-429",
			Turns: []Turn{
				{DosVerdict: VerdictWorking, SlopIntroduced: 0, GateCostUnits: 1, Note: "first attempt"},
				{SelfReportedDone: true, DosVerdict: VerdictRefuted, SlopIntroduced: 3, GateCostUnits: 1, Note: "claims done; diff only touched a comment (CLAIM_UNWITNESSED)"},
				{DosVerdict: VerdictWorking, SlopIntroduced: -1, GateCostUnits: 1, Note: "gate refused; agent reworks"},
				{SelfReportedDone: true, DosVerdict: VerdictWitnessed, SlopIntroduced: -1, GateCostUnits: 1, Note: "test added, diff-witnessed"},
			},
		},
		{
			// No separation: the first self-reported done is already witnessed,
			// so both loops stop at the same turn and the report shows no gap.
			Name: "add-metric-row",
			Turns: []Turn{
				{DosVerdict: VerdictWorking, SlopIntroduced: 1, GateCostUnits: 1, Note: "wire the counter"},
				{SelfReportedDone: true, DosVerdict: VerdictWitnessed, SlopIntroduced: -1, GateCostUnits: 1, Note: "row + test, diff-witnessed"},
			},
		},
		{
			// An --allow-empty / cosmetic "done" on turn 1: naive ships it
			// immediately; gated runs three more turns to a real witnessed done.
			Name: "refactor-evictor",
			Turns: []Turn{
				{SelfReportedDone: true, DosVerdict: VerdictRefuted, SlopIntroduced: 2, GateCostUnits: 1, Note: "claims done; cosmetic-only diff (NOT_SHIPPED)"},
				{DosVerdict: VerdictWorking, SlopIntroduced: 1, GateCostUnits: 1, Note: "gate refused; real extraction begins"},
				{DosVerdict: VerdictWorking, SlopIntroduced: -2, GateCostUnits: 1, Note: "dedup the clone the false done left"},
				{SelfReportedDone: true, DosVerdict: VerdictWitnessed, SlopIntroduced: -1, GateCostUnits: 1, Note: "behavior-preserving split, diff-witnessed"},
			},
		},
		{
			// Trivial true-done on turn 1: both loops agree, zero gap.
			Name: "doc-link-fix",
			Turns: []Turn{
				{SelfReportedDone: true, DosVerdict: VerdictWitnessed, SlopIntroduced: 0, GateCostUnits: 1, Note: "dead link repaired, diff-witnessed"},
			},
		},
	}
}

// EpisodeOutcome is one loop's result on one episode.
type EpisodeOutcome struct {
	Episode          string  `json:"episode"`
	AcceptedTurn     int     `json:"accepted_turn"` // 1-based turn the loop stopped on
	FalseDone        bool    `json:"false_done"`
	Iterations       int     `json:"iterations"`
	SlopShipped      int     `json:"slop_shipped"`
	WastedIterations int     `json:"wasted_iterations"`
	GateCostUnits    float64 `json:"gate_cost_units"`
}

// firstWitnessed returns the 1-based index of the first dos-witnessed turn, or 0
// if the episode never genuinely completes in the corpus.
func firstWitnessed(ep Episode) int {
	for i, t := range ep.Turns {
		if t.DosVerdict == VerdictWitnessed {
			return i + 1
		}
	}
	return 0
}

// runNaive replays an episode through the naive loop: stop at the first
// self-reported "done", trusting it. The cost of a wrong trust is the rework
// (wasted iterations) that the falsely-accepted done forces downstream.
func runNaive(ep Episode) EpisodeOutcome {
	accepted := len(ep.Turns)
	for i, t := range ep.Turns {
		if t.SelfReportedDone {
			accepted = i + 1
			break
		}
	}
	out := EpisodeOutcome{Episode: ep.Name, AcceptedTurn: accepted, Iterations: accepted}
	out.SlopShipped = sumSlop(ep, accepted)
	out.FalseDone = ep.Turns[accepted-1].DosVerdict != VerdictWitnessed
	if out.FalseDone {
		switch w := firstWitnessed(ep); {
		case w > accepted:
			out.WastedIterations = w - accepted // rework to reach the real done
		case w == 0:
			out.WastedIterations = len(ep.Turns) - accepted // never completes in-corpus
		}
	}
	return out // gate cost 0: the naive loop never adjudicates
}

// runGated replays an episode through the dos-gated driver: refuse every
// self-reported done dos refutes and re-arm, stopping only at the first witnessed
// done. It pays the adjudication cost on every turn up to acceptance.
func runGated(ep Episode) EpisodeOutcome {
	accepted := firstWitnessed(ep)
	if accepted == 0 {
		accepted = len(ep.Turns) // never witnessed: the gate would keep re-arming
	}
	out := EpisodeOutcome{Episode: ep.Name, AcceptedTurn: accepted, Iterations: accepted}
	out.SlopShipped = sumSlop(ep, accepted)
	out.FalseDone = false // by construction: only a witnessed done is accepted
	for i := 0; i < accepted; i++ {
		out.GateCostUnits += ep.Turns[i].GateCostUnits
	}
	return out
}

func sumSlop(ep Episode, upto int) int {
	s := 0
	for i := 0; i < upto; i++ {
		s += ep.Turns[i].SlopIntroduced
	}
	return s
}

// ArmSummary aggregates one loop's outcome across the whole corpus.
type ArmSummary struct {
	Loop               string  `json:"loop"`
	Episodes           int     `json:"episodes"`
	FalseDoneEpisodes  int     `json:"false_done_episodes"`
	FalseDoneRate      float64 `json:"false_done_rate"`
	MeanIterations     float64 `json:"mean_iterations_to_accept"`
	SlopShippedTotal   int     `json:"slop_shipped_total"`
	WastedIterations   int     `json:"wasted_iterations_total"`
	GateCostUnitsTotal float64 `json:"gate_cost_units_total"`
	TotalIterationsRun int     `json:"total_iterations_run"`
}

func summarize(loop string, outs []EpisodeOutcome) ArmSummary {
	s := ArmSummary{Loop: loop, Episodes: len(outs)}
	totalIters := 0
	for _, o := range outs {
		if o.FalseDone {
			s.FalseDoneEpisodes++
		}
		totalIters += o.Iterations
		s.SlopShippedTotal += o.SlopShipped
		s.WastedIterations += o.WastedIterations
		s.GateCostUnitsTotal += o.GateCostUnits
	}
	s.TotalIterationsRun = totalIters
	if s.Episodes > 0 {
		s.FalseDoneRate = round4(float64(s.FalseDoneEpisodes) / float64(s.Episodes))
		s.MeanIterations = round4(float64(totalIters) / float64(s.Episodes))
	}
	return s
}

// Delta is the witnessed win of the gated loop over the naive loop, reported NET
// of the gate's own cost (net-true criterion 2).
type Delta struct {
	FalseDoneRateReduction   float64 `json:"false_done_rate_reduction"`
	SlopAvoided              int     `json:"slop_avoided"`
	WastedIterationsAvoided  int     `json:"wasted_iterations_avoided"`
	GateCostUnits            float64 `json:"gate_cost_units"`
	ExtraIterationsToWitness int     `json:"extra_iterations_to_witness"`
	NetTrueFinding           string  `json:"net_true_finding"`
}

// EpisodePair is the per-episode naive-vs-gated detail (transparency: shows
// exactly where the two loops separate and where they agree).
type EpisodePair struct {
	Name  string         `json:"name"`
	Naive EpisodeOutcome `json:"naive"`
	Gated EpisodeOutcome `json:"gated"`
}

// Provenance labels the report per the net-true doctrine.
type Provenance struct {
	Kind        string `json:"kind"`    // SIMULATED until the live driver (#1173 #S1) lands
	Command     string `json:"command"` // re-runnable witness
	GeneratedBy string `json:"generated_by"`
	Note        string `json:"note"`
}

// LoopVerifyReport is the full naive-vs-gated comparison.
type LoopVerifyReport struct {
	Schema         string        `json:"schema"`
	Provenance     Provenance    `json:"provenance"`
	CorpusEpisodes int           `json:"corpus_episodes"`
	CorpusTurns    int           `json:"corpus_turns"`
	Naive          ArmSummary    `json:"naive"`
	Gated          ArmSummary    `json:"gated"`
	Delta          Delta         `json:"delta"`
	Episodes       []EpisodePair `json:"episodes"`
	Verdict        string        `json:"verdict"`
}

// Verdicts the bench can reach.
const (
	// VerdictGatedLower: the gated loop's false-done rate is strictly lower —
	// the acceptance criterion of #1190.
	VerdictGatedLower = "gated_false_done_rate_lower"
	// VerdictCorpusTooEasy: the corpus cannot separate the two loops (no false
	// done for the naive loop to make). An honest finding, not a failure.
	VerdictCorpusTooEasy = "corpus_too_easy_to_separate"
)

// BuildLoopVerifyReport runs the default corpus through both loops and folds the
// result into the report.
func BuildLoopVerifyReport() LoopVerifyReport {
	return BuildLoopVerifyReportFor(DefaultLoopCorpus())
}

// BuildLoopVerifyReportFor folds an arbitrary corpus (the seam the live driver
// will feed real turn records into).
func BuildLoopVerifyReportFor(corpus []Episode) LoopVerifyReport {
	naiveOuts := make([]EpisodeOutcome, 0, len(corpus))
	gatedOuts := make([]EpisodeOutcome, 0, len(corpus))
	pairs := make([]EpisodePair, 0, len(corpus))
	turns := 0
	for _, ep := range corpus {
		turns += len(ep.Turns)
		n := runNaive(ep)
		g := runGated(ep)
		naiveOuts = append(naiveOuts, n)
		gatedOuts = append(gatedOuts, g)
		pairs = append(pairs, EpisodePair{Name: ep.Name, Naive: n, Gated: g})
	}
	naive := summarize("naive", naiveOuts)
	gated := summarize("gated", gatedOuts)

	verdict := VerdictCorpusTooEasy
	if gated.FalseDoneRate < naive.FalseDoneRate {
		verdict = VerdictGatedLower
	}

	d := Delta{
		FalseDoneRateReduction:   round4(naive.FalseDoneRate - gated.FalseDoneRate),
		SlopAvoided:              naive.SlopShippedTotal - gated.SlopShippedTotal,
		WastedIterationsAvoided:  naive.WastedIterations - gated.WastedIterations,
		GateCostUnits:            round4(gated.GateCostUnitsTotal),
		ExtraIterationsToWitness: gated.TotalIterationsRun - naive.TotalIterationsRun,
	}
	d.NetTrueFinding = netTrueFinding(naive, gated, d)

	return LoopVerifyReport{
		Schema: "loopverify.v1",
		Provenance: Provenance{
			Kind:        "SIMULATED",
			Command:     "go test ./internal/bench/ -run TestLoopVerifyBench",
			GeneratedBy: "fak/internal/bench.BuildLoopVerifyReport",
			Note: "Corpus is a labeled fixture: the kernel-native loop driver (#1173 #S1) " +
				"is not built yet, so a live naive-vs-gated wall-clock is `not yet`. This " +
				"witnesses the MEASUREMENT and the metric definitions; live turn records " +
				"feed the same report shape once the driver lands.",
		},
		CorpusEpisodes: len(corpus),
		CorpusTurns:    turns,
		Naive:          naive,
		Gated:          gated,
		Delta:          d,
		Episodes:       pairs,
		Verdict:        verdict,
	}
}

func netTrueFinding(naive, gated ArmSummary, d Delta) string {
	if d.FalseDoneRateReduction <= 0 {
		return "corpus does not separate the loops: the naive loop made no false done, so the gate earns nothing measurable here (a real finding, not a failure)."
	}
	return fmt.Sprintf(
		"the gate cuts the false-done rate %.0f%%→%.0f%%, avoids %d slop unit(s) and %d wasted (rework) iteration(s), "+
			"at a cost of %.0f adjudication-run(s) and %d extra iteration(s) to reach a witnessed done. "+
			"Per-run adjudication wall-clock is `not yet` (host-gated), but each adjudication is bounded and far cheaper "+
			"than the agent turn it gates, so the win is net-positive at any plausible per-run cost.",
		naive.FalseDoneRate*100, gated.FalseDoneRate*100,
		d.SlopAvoided, d.WastedIterationsAvoided,
		d.GateCostUnits, d.ExtraIterationsToWitness,
	)
}

// JSON renders the report as stable, indented JSON (deterministic: no clock, no
// map iteration), so it is a re-derivable witness.
func (r LoopVerifyReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

func round4(f float64) float64 {
	return math.Round(f*1e4) / 1e4
}
