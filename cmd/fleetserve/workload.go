package main

// Realistic transcript-derived workload replay for fleetserve.
//
// The synthetic surface (flat -prefix/-decode/-result) answers "how does fak scale a
// FIXED T×C shape", but real agent sessions are not flat: ~94% of turns make a tool call
// (so ingest a tool result), decode is ~700 tokens/turn (not 32), the shared preamble is
// ~20k tokens (not 1024), and a session is a *sequence* of turns of varying shape.
// tools/transcript_workload.py mines that out of the Claude session JSONL into a
// fak.workload.v1 profile; here we PLAY ONE BACK turn-by-turn through the reuse/no-reuse
// kernels. (Turns are keyed by message.id — Claude splits one response across several
// JSONL records; counting per-record gives a bogus ~41% tool-call fraction. See the
// profiler's docstring.)
//
// The tuning knobs (-tune-*) are the "what if" layer: scale the tool-call fraction, the
// result size, the decode length, or the prefix, and watch how fak's prefix-reuse win
// responds. -tune-toolfrac is the headline one — "simulate what % of turns are tool calls"
// — because the result-ingest between turns is exactly the per-agent KV growth that the
// shared-prefix reuse has to amortise.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type statDist struct {
	N      float64 `json:"n"`
	Min    float64 `json:"min"`
	P10    float64 `json:"p10"`
	Median float64 `json:"median"`
	Mean   float64 `json:"mean"`
	P90    float64 `json:"p90"`
	Max    float64 `json:"max"`
}

type replayStep struct {
	Decode int     `json:"decode"`
	Tool   bool    `json:"tool"`
	Result int     `json:"result"`
	Name   *string `json:"name"`
}

type replayTrack struct {
	Session           string       `json:"session"`
	PercentileByTurns int          `json:"percentile_by_turns"`
	PrefixTokens      int          `json:"prefix_tokens"`
	NTurns            int          `json:"n_turns"`
	ToolCallFraction  float64      `json:"tool_call_fraction"`
	Track             []replayStep `json:"track"`
}

type workloadProfile struct {
	Schema                  string        `json:"schema"`
	ToolCallFraction        float64       `json:"tool_call_fraction"`
	PrefixTokens            statDist      `json:"prefix_tokens"`
	DecodeTokensPerTurn     statDist      `json:"decode_tokens_per_turn"`
	ResultTokensPerToolTurn statDist      `json:"result_tokens_per_tool_turn"`
	TurnsPerSession         statDist      `json:"turns_per_session"`
	Replay                  []replayTrack `json:"replay"`
}

// tune holds the benchmark-parameter knobs. All multipliers default to 1.0 (replay as-is).
type tune struct {
	toolFrac float64 // scale the fraction of turns that ingest a tool result
	result   float64 // scale per-tool-turn result tokens (R)
	decode   float64 // scale per-turn decode tokens (D)
	prefix   float64 // scale the shared prefix (P)
	turnCap  int     // cap turns replayed (0 = full track); keeps sweeps tractable
}

// turnStep is one resolved turn of the replay schedule: decode D tokens, then ingest
// resultTokens private tool/result tokens (0 = a text-only / non-tool turn).
type turnStep struct {
	decode int
	result int
}

func loadWorkload(path string) (*workloadProfile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p workloadProfile
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if p.Schema != "fak.workload.v1" {
		return nil, fmt.Errorf("%s: unexpected schema %q (want fak.workload.v1)", path, p.Schema)
	}
	if len(p.Replay) == 0 {
		return nil, fmt.Errorf("%s: no replay tracks", path)
	}
	return &p, nil
}

// pickTrack selects the replay track at the requested turns-percentile (closest), or the
// last (largest) track when pct<=0.
func (p *workloadProfile) pickTrack(pct int) replayTrack {
	if pct <= 0 {
		return p.Replay[len(p.Replay)-1]
	}
	best, bestd := p.Replay[0], 1<<30
	for _, t := range p.Replay {
		d := t.PercentileByTurns - pct
		if d < 0 {
			d = -d
		}
		if d < bestd {
			best, bestd = t, d
		}
	}
	return best
}

// buildSchedule resolves a replay track + tuning into the concrete per-turn schedule and
// the prefix length to prefill. The tool-call fraction knob redistributes which turns are
// result-bearing (evenly, via a Bresenham accumulator) so the knob is monotone and
// deterministic; the real result-size distribution from the track is preserved (cycled).
func buildSchedule(p *workloadProfile, tr replayTrack, tn tune) (prefix int, steps []turnStep, effToolFrac float64) {
	track := tr.Track
	if tn.turnCap > 0 && len(track) > tn.turnCap {
		track = track[:tn.turnCap]
	}
	T := len(track)
	prefix = scaleInt(tr.PrefixTokens, tn.prefix, 1)

	// real per-tool-turn result sizes from this track (already token-estimated), in order
	var baseResults []int
	for _, s := range track {
		if s.Tool && s.Result > 0 {
			baseResults = append(baseResults, s.Result)
		}
	}
	medianResult := int(p.ResultTokensPerToolTurn.Median)
	if medianResult <= 0 {
		medianResult = 1
	}
	if len(baseResults) == 0 {
		baseResults = []int{medianResult}
	}

	// target count of result-bearing turns after the tool-fraction knob
	baseFrac := tr.ToolCallFraction
	target := clamp01(baseFrac * tn.toolFrac)
	targetCount := int(math.Round(target * float64(T)))
	if targetCount < 0 {
		targetCount = 0
	}
	if targetCount > T {
		targetCount = T
	}

	steps = make([]turnStep, T)
	acc, placed, ri := 0.0, 0, 0
	per := 0.0
	if T > 0 {
		per = float64(targetCount) / float64(T)
	}
	for i := 0; i < T; i++ {
		d := scaleInt(track[i].Decode, tn.decode, 1)
		steps[i].decode = d
		// Bresenham: spread targetCount result-bearing turns evenly across T.
		acc += per
		if acc >= 1.0 && placed < targetCount {
			acc -= 1.0
			r := baseResults[ri%len(baseResults)]
			ri++
			steps[i].result = scaleInt(r, tn.result, 1)
			placed++
		}
	}
	// rounding can leave placed one short; top up the last turns deterministically
	for i := T - 1; i >= 0 && placed < targetCount; i-- {
		if steps[i].result == 0 {
			steps[i].result = scaleInt(baseResults[ri%len(baseResults)], tn.result, 1)
			ri++
			placed++
		}
	}
	if T > 0 {
		effToolFrac = float64(placed) / float64(T)
	}
	return prefix, steps, effToolFrac
}

func scaleInt(v int, mul float64, min int) int {
	if mul == 1 {
		if v < min {
			return min
		}
		return v
	}
	r := int(math.Round(float64(v) * mul))
	if r < min {
		return min
	}
	return r
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// scheduleTotals returns total decode steps and total result tokens per agent over the
// schedule (for reporting / amortisation math).
func scheduleTotals(steps []turnStep) (decode, result, toolTurns int) {
	for _, s := range steps {
		decode += s.decode
		result += s.result
		if s.result > 0 {
			toolTurns++
		}
	}
	return
}

// runScheduleTurns replays a per-turn schedule across all C agents in lockstep: decode
// st.decode tokens, then (on tool turns) ingest st.result private tokens per agent. This
// is the per-turn-varying analogue of main.runTurns.
func runScheduleTurns(bs *model.BatchSession, ids0 []int, steps []turnStep, vocab, rep int) (time.Duration, time.Duration) {
	ids := append([]int(nil), ids0...)
	C := len(ids0)
	var decodeTotal, resultTotal time.Duration
	for t, st := range steps {
		t0 := time.Now()
		for s := 0; s < st.decode; s++ {
			bs.StepBatch(ids)
			for i := range ids {
				ids[i] = (ids[i]*48271 + 1) % vocab
			}
		}
		decodeTotal += time.Since(t0)
		if st.result > 0 {
			rp := make([][]int, C)
			for a := range rp {
				rp[a] = lcgIDs(st.result, vocab, 10_000+rep*1_000_000+t*10_000+a*97)
			}
			t1 := time.Now()
			bs.PrefillEachNoLogits(rp)
			resultTotal += time.Since(t1)
		}
	}
	return decodeTotal, resultTotal
}

type workloadPoint struct {
	Concurrency int `json:"concurrency"`
	Turns       int `json:"turns"`
	PrefixLen   int `json:"prefix_len"`
	// effective (post-tune) per-agent workload
	DecodeTokensTotal float64 `json:"decode_tokens_total_per_agent"`
	ResultTokensTotal float64 `json:"result_tokens_total_per_agent"`
	ToolTurns         int     `json:"tool_turns"`
	EffToolFraction   float64 `json:"effective_tool_call_fraction"`
	Reps              int     `json:"reps"`

	reuseMetrics
}

// runWorkloadMode replays one transcript-derived track (after tuning) across a concurrency
// sweep, measuring fak with prefix reuse vs the no-reuse ablation — same kernels, same
// decode, the only difference being whether the P-token preamble is prefilled once and
// cloned (reuse) or re-prefilled per agent (no-reuse).
func runWorkloadMode(m *model.Model, quant bool, vocab int, prof *workloadProfile, trackPct int, tn tune, concs []int, reps int, ablation bool, out string) {
	tr := prof.pickTrack(trackPct)
	prefixLen, steps, effFrac := buildSchedule(prof, tr, tn)
	decTot, resTot, toolTurns := scheduleTotals(steps)
	T := len(steps)
	prefix := lcgIDs(prefixLen, vocab, 1)

	fmt.Fprintf(os.Stderr,
		"workload replay: track %s (p%d) | T=%d prefix=%d decode/agent=%d result/agent=%d "+
			"toolTurns=%d effToolFrac=%.3f | tune{toolfrac=%.2f result=%.2f decode=%.2f prefix=%.2f cap=%d}\n",
		tr.Session, tr.PercentileByTurns, T, prefixLen, decTot, resTot, toolTurns, effFrac,
		tn.toolFrac, tn.result, tn.decode, tn.prefix, tn.turnCap)

	var points []workloadPoint
	for _, C := range concs {
		if C < 1 {
			continue
		}
		ids0 := lcgIDs(C, vocab, 991)
		var reuseTotals, reusePre, reuseClone, reuseDec, reuseResult []time.Duration
		var noReuseTotals, noReusePre, noReuseDec, noReuseResult []time.Duration
		for r := 0; r < reps; r++ {
			// ---- reuse: prefill once, clone C, replay schedule ----
			bs, tPre, tClone := newReuseBatch(m, quant, prefix, C, decTot+resTot)
			tDec, tRes := runScheduleTurns(bs, ids0, steps, vocab, r)
			reuseTotals = append(reuseTotals, tPre+tClone+tDec+tRes)
			reusePre = append(reusePre, tPre)
			reuseClone = append(reuseClone, tClone)
			reuseDec = append(reuseDec, tDec)
			reuseResult = append(reuseResult, tRes)

			if ablation {
				// ---- no-reuse: C independent prefix prefills, same schedule ----
				nbs, nPre := newNoReuseBatch(m, quant, prefix, C, decTot+resTot)
				nDec, nRes := runScheduleTurns(nbs, ids0, steps, vocab, r)
				noReuseTotals = append(noReuseTotals, nPre+nDec+nRes)
				noReusePre = append(noReusePre, nPre)
				noReuseDec = append(noReuseDec, nDec)
				noReuseResult = append(noReuseResult, nRes)
			}
			runtime.GC()
		}

		rTot := msFromDur(minDur(reuseTotals))
		rAgents := float64(C) / (rTot / 1e3)
		rTurns := float64(C*T) / (rTot / 1e3)
		pt := workloadPoint{
			Concurrency: C, Turns: T, PrefixLen: prefixLen,
			DecodeTokensTotal: float64(decTot), ResultTokensTotal: float64(resTot),
			ToolTurns: toolTurns, EffToolFraction: effFrac, Reps: reps,
			reuseMetrics: reuseMetrics{
				ReusePrefillMS: msFromDur(minDur(reusePre)), ReuseCloneMS: msFromDur(minDur(reuseClone)),
				ReuseDecodeMS: msFromDur(minDur(reuseDec)), ReuseResultMS: msFromDur(minDur(reuseResult)),
				ReuseTotalMS: rTot, ReuseAgentsSec: rAgents, ReuseAgentTurnsSec: rTurns,
			},
		}
		if ablation {
			pt.fillNoReuse(noReuseTotals, noReusePre, noReuseDec, noReuseResult, C, T, rAgents)
		}
		points = append(points, pt)
		if ablation {
			fmt.Fprintf(os.Stderr,
				"  C=%-3d | reuse %.0f ms (pre %.0f + clone %.0f + dec %.0f + res %.0f) = %.1f turns/s | "+
					"no-reuse %.0f ms = %.1f turns/s | reuse %.2f×\n",
				C, rTot, pt.ReusePrefillMS, pt.ReuseCloneMS, pt.ReuseDecodeMS, pt.ReuseResultMS,
				rTurns, *pt.NoReuseTotalMS, *pt.NoReuseAgentTurnsSec, *pt.ReuseSpeedup)
		} else {
			fmt.Fprintf(os.Stderr,
				"  C=%-3d | reuse %.0f ms (pre %.0f + clone %.0f + dec %.0f + res %.0f) = %.1f turns/s | no-reuse skipped\n",
				C, rTot, pt.ReusePrefillMS, pt.ReuseCloneMS, pt.ReuseDecodeMS, pt.ReuseResultMS,
				rTurns)
		}
	}

	report := map[string]any{
		"app_version":                   appversion.Current(),
		"engine":                        "fak fleetserve (transcript replay, Q8=" + boolStr(quant) + ")",
		"model":                         "SmolLM2-135M",
		"schema":                        "fak.fleetserve-workload.v1",
		"go_threads":                    runtime.GOMAXPROCS(0),
		"workload_profile":              prof.Schema,
		"track":                         tr.Session,
		"track_percentile":              tr.PercentileByTurns,
		"track_tool_frac":               tr.ToolCallFraction,
		"turns":                         T,
		"prefix_len":                    prefixLen,
		"agent_grid":                    concs,
		"ablation":                      ablation,
		"tune":                          map[string]any{"toolfrac": tn.toolFrac, "result": tn.result, "decode": tn.decode, "prefix": tn.prefix, "turn_cap": tn.turnCap},
		"effective_toolfrac":            effFrac,
		"decode_tokens_total_per_agent": decTot,
		"result_tokens_total_per_agent": resTot,
		"points":                        points,
	}
	writeReport(report, out)
}
