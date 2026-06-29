// loopbench measures what the witnessed exit-gate earns over a naive Ralph
// loop that terminates on the agent's own self-reported "done". It runs the
// SAME fixed task corpus through two policies — naive (accept the first
// self-report) and gated (accept only an externally witnessed done, re-arming
// otherwise via Adjudicate) — and reports the delta the gate removes: the
// false-done rate, the slop shipped under a false done, the wasted iterations,
// and the gate's own per-turn cost so the win is reported NET (the repo's
// net-true doctrine, docs/standards/net-true-value.md).
//
// The corpus is a SIMULATED fixture (hand-authored turn traces representative
// of real loop runs); the report labels its provenance and names the live
// witness that remains `not yet`. Nothing here shells out or needs a host: the
// gated policy drives the real loopgate.Adjudicate decision against a witness
// adapter derived from each turn's ground truth, so the bench exercises the
// shipping gate rather than a strawman reimplementation of it.
package loopgate

import (
	"context"
	"encoding/json"
	"fmt"
)

// BenchTurn is one turn's ground truth in a benchmarked loop episode.
type BenchTurn struct {
	// ClaimedDone is whether the agent self-reported "done" this turn — the
	// only signal a naive Ralph loop consumes.
	ClaimedDone bool `json:"claimed_done"`
	// Witnessed is whether an external witness (commit-audit OK / verify
	// shipped) would corroborate the done claim this turn — the ground truth
	// the gate checks and the naive loop ignores.
	Witnessed bool `json:"witnessed"`
	// SlopUnits is the slop-scorecard delta the turn's commit introduces if it
	// is shipped as the loop's accepted answer.
	SlopUnits int `json:"slop_units"`
	// TurnTokens is the cost of running this turn — used to price the rework a
	// false done pushes downstream.
	TurnTokens int `json:"turn_tokens"`
	// GateCostTokens is the dos-adjudication overhead the gate pays when it
	// adjudicates this turn's claim.
	GateCostTokens int `json:"gate_cost_tokens"`
}

// Episode is one fixed task's turn trace. Turns run in order; the loop reads
// one turn per iteration.
type Episode struct {
	ID    string      `json:"id"`
	Turns []BenchTurn `json:"turns"`
}

// firstWitnessedDone returns the index of the first turn an external witness
// would accept as done, or -1 if the episode never reaches a witnessed done.
func (e Episode) firstWitnessedDone() int {
	for i, t := range e.Turns {
		if t.ClaimedDone && t.Witnessed {
			return i
		}
	}
	return -1
}

// firstClaimedDone returns the index of the first turn the agent self-reports
// done, or -1 if it never does.
func (e Episode) firstClaimedDone() int {
	for i, t := range e.Turns {
		if t.ClaimedDone {
			return i
		}
	}
	return -1
}

// LoopStats is one policy's outcome over the corpus.
type LoopStats struct {
	Policy string `json:"policy"`
	// Episodes is the corpus size scored.
	Episodes int `json:"episodes"`
	// AcceptedDone is the count of episodes the loop terminated on a "done".
	AcceptedDone int `json:"accepted_done"`
	// FalseDone is the count of episodes whose accepted "done" no witness
	// corroborated (the naive loop's CLAIM_UNWITNESSED / NOT_SHIPPED stops).
	FalseDone int `json:"false_done"`
	// FalseDoneRate is FalseDone / AcceptedDone — the headline number.
	FalseDoneRate float64 `json:"false_done_rate"`
	// WitnessedDoneReached is the count of episodes the loop drove to a
	// genuinely witnessed done.
	WitnessedDoneReached int `json:"witnessed_done_reached"`
	// SlopShipped is total slop units of the commits the loop accepted as done.
	SlopShipped int `json:"slop_shipped"`
	// ItersToAcceptedDone is total turns run before each accepted "done"
	// (1-based; the accepted turn counts).
	ItersToAcceptedDone int `json:"iters_to_accepted_done"`
	// WastedIterations is total turns spent shipping-broken after a false done
	// before the true witnessed done is reached — the rework the false stop
	// pushed downstream.
	WastedIterations int `json:"wasted_iterations"`
	// ReworkTokens prices WastedIterations: the turn tokens the downstream
	// rework costs because the loop shipped a false done.
	ReworkTokens int `json:"rework_tokens"`
	// GateCostTokens is the dos-adjudication overhead this policy paid (zero
	// for the naive loop, which never adjudicates).
	GateCostTokens int `json:"gate_cost_tokens"`
}

// NetTrue reports the gate's win net of its own cost, per
// docs/standards/net-true-value.md (criterion 2: net of the cost the change
// itself adds).
type NetTrue struct {
	// ReworkTokensAvoided is the downstream rework the naive loop pays that the
	// gate prevents by refusing the false done.
	ReworkTokensAvoided int `json:"rework_tokens_avoided"`
	// GateCostTokens is what the gate spent adjudicating to prevent it.
	GateCostTokens int `json:"gate_cost_tokens"`
	// NetTokens is ReworkTokensAvoided − GateCostTokens; positive means the
	// gate earns its keep on this corpus.
	NetTokens int `json:"net_tokens"`
	// GateEarnsKeep is NetTokens > 0.
	GateEarnsKeep bool `json:"gate_earns_keep"`
}

// Report is the full verified-vs-naive comparison.
type Report struct {
	Benchmark string `json:"benchmark"`
	// Provenance labels the corpus — SIMULATED here (a fixture), never quoted
	// as a measured live-agent number.
	Provenance string `json:"provenance"`
	// Corpus is the fixed task set, named so a re-run is comparable.
	Corpus string `json:"corpus"`
	// Naive and Gated are the two policies' stats over the same corpus.
	Naive LoopStats `json:"naive"`
	Gated LoopStats `json:"gated"`
	// FalseDoneRateDelta is Naive.FalseDoneRate − Gated.FalseDoneRate — the
	// rate the exit-gate removes. Positive is the claim of the epic.
	FalseDoneRateDelta float64 `json:"false_done_rate_delta"`
	// SlopShippedDelta is Naive.SlopShipped − Gated.SlopShipped.
	SlopShippedDelta int     `json:"slop_shipped_delta"`
	NetTrue          NetTrue `json:"net_true"`
	// Finding is a one-line human summary — including the honest "corpus too
	// easy to separate them" reading if the delta is zero.
	Finding string `json:"finding"`
	// NotYet names the live witness this fixture bench does not yet provide.
	NotYet string `json:"not_yet"`
}

// runNaive terminates each episode on the first self-reported done, ignoring
// the witness. A false done ships the slop of that turn's commit and pushes the
// turns up to the true witnessed done downstream as rework.
func runNaive(corpus []Episode) LoopStats {
	s := LoopStats{Policy: "naive-self-report", Episodes: len(corpus)}
	for _, e := range corpus {
		stop := e.firstClaimedDone()
		if stop < 0 {
			continue // never claimed done; nothing accepted
		}
		s.AcceptedDone++
		s.ItersToAcceptedDone += stop + 1
		s.SlopShipped += e.Turns[stop].SlopUnits
		if e.Turns[stop].Witnessed {
			s.WitnessedDoneReached++
			continue
		}
		// False done: the naive loop stopped early. The true witnessed done is
		// later (if reachable); the gap is the rework it shipped broken.
		s.FalseDone++
		trueDone := e.firstWitnessedDone()
		if trueDone > stop {
			for i := stop + 1; i <= trueDone; i++ {
				s.WastedIterations++
				s.ReworkTokens += e.Turns[i].TurnTokens
			}
		}
	}
	if s.AcceptedDone > 0 {
		s.FalseDoneRate = float64(s.FalseDone) / float64(s.AcceptedDone)
	}
	return s
}

// runGated drives each episode through the real Adjudicate gate: every claimed
// done is adjudicated against a witness derived from that turn's ground truth,
// re-arming on an unwitnessed claim and accepting only a witnessed one. It pays
// GateCostTokens per adjudication.
func runGated(corpus []Episode) LoopStats {
	s := LoopStats{Policy: "gated-witnessed", Episodes: len(corpus)}
	ctx := context.Background()
	for _, e := range corpus {
		accepted := -1
		for i, t := range e.Turns {
			if !t.ClaimedDone {
				continue
			}
			s.GateCostTokens += t.GateCostTokens
			// Witness adapter: the turn is witnessed iff its ground truth says
			// so. The gate makes the keep decision, not the bench.
			dec := Adjudicate(ctx, Turn{ClaimedDone: true, Claim: e.ID, HeadRef: "HEAD"},
				func(_ context.Context, _ Request) (WitnessResult, error) {
					if t.Witnessed {
						return WitnessResult{Outcome: OutcomeWitnessed, Reason: "OK", Rung: "diff-witnessed"}, nil
					}
					return WitnessResult{Outcome: OutcomeNotYet, Reason: "CLAIM_UNWITNESSED", Rung: "subject-only"}, nil
				})
			if dec.Verdict == VerdictWitnessed {
				accepted = i
				break
			}
			// NOT_YET: the gate re-arms; the loop keeps going. No false done is
			// shipped and no downstream rework is incurred.
		}
		if accepted < 0 {
			continue
		}
		s.AcceptedDone++
		s.WitnessedDoneReached++
		s.ItersToAcceptedDone += accepted + 1
		s.SlopShipped += e.Turns[accepted].SlopUnits
	}
	// The gate accepts only witnessed dones, so FalseDone stays zero by
	// construction — that is the property being measured, not assumed.
	if s.AcceptedDone > 0 {
		s.FalseDoneRate = float64(s.FalseDone) / float64(s.AcceptedDone)
	}
	return s
}

// CompareLoops runs the corpus through both policies and folds the four
// metrics, net of gate cost.
func CompareLoops(corpus []Episode) Report {
	naive := runNaive(corpus)
	gated := runGated(corpus)
	net := NetTrue{
		ReworkTokensAvoided: naive.ReworkTokens,
		GateCostTokens:      gated.GateCostTokens,
		NetTokens:           naive.ReworkTokens - gated.GateCostTokens,
	}
	net.GateEarnsKeep = net.NetTokens > 0
	r := Report{
		Benchmark:          "verified-vs-naive-loop",
		Provenance:         "SIMULATED",
		Corpus:             "loopgate.DefaultBenchCorpus (hand-authored episode traces)",
		Naive:              naive,
		Gated:              gated,
		FalseDoneRateDelta: naive.FalseDoneRate - gated.FalseDoneRate,
		SlopShippedDelta:   naive.SlopShipped - gated.SlopShipped,
		NetTrue:            net,
		NotYet: "live-agent run over a real dojo episode corpus (host/agent-gated); " +
			"the SIMULATED fixture proves the gate's mechanism, not a wall-clock tok/s",
	}
	r.Finding = finding(r)
	return r
}

// finding renders the honest one-line reading, including the "too easy to
// separate" case the issue explicitly allows.
func finding(r Report) string {
	if r.FalseDoneRateDelta <= 0 {
		return "corpus does not separate the loops (false-done delta <= 0): " +
			"a real finding the corpus is too easy, not a gate win"
	}
	net := "but the gate cost outweighs the rework avoided on this corpus (net <= 0)"
	if r.NetTrue.GateEarnsKeep {
		net = fmt.Sprintf("net of %d gate-cost tokens it still avoids %d rework tokens (net +%d)",
			r.NetTrue.GateCostTokens, r.NetTrue.ReworkTokensAvoided, r.NetTrue.NetTokens)
	}
	return fmt.Sprintf("gated loop cuts the false-done rate from %.2f to %.2f and removes %d slop units; %s",
		r.Naive.FalseDoneRate, r.Gated.FalseDoneRate, r.SlopShippedDelta, net)
}

// JSON renders the report as the stable, re-runnable bench artifact.
func (r Report) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// DefaultBenchCorpus is the fixed, SIMULATED task set. It mixes honest episodes
// (the agent's "done" coincides with the witnessed done) with trap episodes (a
// premature self-reported "done" the witness refutes), so the naive loop's
// false-done rate is non-zero and separable from the gate's. The numbers are a
// fixture, not a measurement — see Report.Provenance.
func DefaultBenchCorpus() []Episode {
	return []Episode{
		{ID: "honest-fast", Turns: []BenchTurn{
			{ClaimedDone: false, SlopUnits: 1, TurnTokens: 800},
			{ClaimedDone: true, Witnessed: true, SlopUnits: 2, TurnTokens: 900, GateCostTokens: 120},
		}},
		{ID: "honest-slow", Turns: []BenchTurn{
			{ClaimedDone: false, SlopUnits: 1, TurnTokens: 700},
			{ClaimedDone: false, SlopUnits: 1, TurnTokens: 700},
			{ClaimedDone: true, Witnessed: true, SlopUnits: 2, TurnTokens: 850, GateCostTokens: 120},
		}},
		{ID: "trap-premature-done", Turns: []BenchTurn{
			{ClaimedDone: true, Witnessed: false, SlopUnits: 9, TurnTokens: 900, GateCostTokens: 120},
			{ClaimedDone: false, SlopUnits: 2, TurnTokens: 800},
			{ClaimedDone: true, Witnessed: true, SlopUnits: 3, TurnTokens: 850, GateCostTokens: 120},
		}},
		{ID: "trap-double-false-done", Turns: []BenchTurn{
			{ClaimedDone: true, Witnessed: false, SlopUnits: 8, TurnTokens: 950, GateCostTokens: 120},
			{ClaimedDone: true, Witnessed: false, SlopUnits: 6, TurnTokens: 900, GateCostTokens: 120},
			{ClaimedDone: false, SlopUnits: 2, TurnTokens: 800},
			{ClaimedDone: true, Witnessed: true, SlopUnits: 3, TurnTokens: 850, GateCostTokens: 120},
		}},
		{ID: "trap-slop-heavy", Turns: []BenchTurn{
			{ClaimedDone: true, Witnessed: false, SlopUnits: 12, TurnTokens: 1000, GateCostTokens: 120},
			{ClaimedDone: true, Witnessed: true, SlopUnits: 4, TurnTokens: 900, GateCostTokens: 120},
		}},
		{ID: "honest-immediate", Turns: []BenchTurn{
			{ClaimedDone: true, Witnessed: true, SlopUnits: 1, TurnTokens: 600, GateCostTokens: 120},
		}},
	}
}
