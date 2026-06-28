package main

// Shared measurement plumbing for fleetserve's two run modes (the synthetic surface in
// main.go and the transcript replay in workload.go). Both modes time the same reuse-vs-
// no-reuse shape per concurrency cell, so the metric fields, the no-reuse fill, the
// per-cell setup, and the report emit are factored here to keep the two modes in lockstep.

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// reuseMetrics is the timing payload both point and workloadPoint report. It is embedded
// (anonymous) into each so the JSON output stays flat and byte-identical to the prior
// per-struct field lists.
type reuseMetrics struct {
	// fak with cross-agent prefix reuse (prefill once + clone C + batched decode)
	ReusePrefillMS     float64 `json:"reuse_prefill_ms"` // one prefix prefill
	ReuseCloneMS       float64 `json:"reuse_clone_ms"`   // C deep-copies of the prefix KV
	ReuseDecodeMS      float64 `json:"reuse_decode_ms"`  // batched decode steps
	ReuseResultMS      float64 `json:"reuse_result_prefill_ms"`
	ReuseTotalMS       float64 `json:"reuse_total_ms"`
	ReuseAgentsSec     float64 `json:"reuse_agents_per_sec"`
	ReuseAgentTurnsSec float64 `json:"reuse_agent_turns_per_sec"`

	// fak without reuse (C independent prefix prefills + same batched decode) — the ablation
	// that prices the prefix-reuse lever while keeping the rest of the workload fixed.
	NoReusePrefillMS     *float64 `json:"noreuse_prefill_ms,omitempty"`
	NoReuseDecodeMS      *float64 `json:"noreuse_decode_ms,omitempty"`
	NoReuseResultMS      *float64 `json:"noreuse_result_prefill_ms,omitempty"`
	NoReuseTotalMS       *float64 `json:"noreuse_total_ms,omitempty"`
	NoReuseAgentsSec     *float64 `json:"noreuse_agents_per_sec,omitempty"`
	NoReuseAgentTurnsSec *float64 `json:"noreuse_agent_turns_per_sec,omitempty"`

	ReuseSpeedup *float64 `json:"reuse_speedup_vs_noreuse,omitempty"` // reuse agents/sec ÷ no-reuse agents/sec
}

func msFromDur(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

// fillNoReuse folds the no-reuse ablation timing slices into m's NoReuse* fields and the
// reuse-vs-no-reuse speedup, given the reuse agents/sec for the same cell. C is the
// concurrency (agents) and turns is the per-agent turn count for the turns/sec figures.
func (m *reuseMetrics) fillNoReuse(totals, pre, dec, result []time.Duration, C, turns int, reuseAgentsSec float64) {
	nTot := msFromDur(minDur(totals))
	nAgents := float64(C) / (nTot / 1e3)
	nTurns := float64(C*turns) / (nTot / 1e3)
	nPre := msFromDur(minDur(pre))
	nDec := msFromDur(minDur(dec))
	nResult := msFromDur(minDur(result))
	speedup := reuseAgentsSec / nAgents
	m.NoReusePrefillMS = &nPre
	m.NoReuseDecodeMS = &nDec
	m.NoReuseResultMS = &nResult
	m.NoReuseTotalMS = &nTot
	m.NoReuseAgentsSec = &nAgents
	m.NoReuseAgentTurnsSec = &nTurns
	m.ReuseSpeedup = &speedup
}

// reuseAccum collects the per-rep reuse and no-reuse phase timings for one sweep cell,
// then folds them into the min-based reuseMetrics summary. It holds the accumulate-and-
// aggregate plumbing that the grid (main.go) and workload (workload.go) run modes would
// otherwise each spell out inline.
type reuseAccum struct {
	total, pre, clone, dec, result []time.Duration
	nTotal, nPre, nDec, nResult    []time.Duration
}

// addReuse records one rep's reuse-path timings (prefill, clone, decode, result-prefill).
func (a *reuseAccum) addReuse(tPre, tClone, tDec, tResult time.Duration) {
	a.total = append(a.total, tPre+tClone+tDec+tResult)
	a.pre = append(a.pre, tPre)
	a.clone = append(a.clone, tClone)
	a.dec = append(a.dec, tDec)
	a.result = append(a.result, tResult)
}

// addNoReuse records one rep's no-reuse ablation timings (prefill, decode, result-prefill).
func (a *reuseAccum) addNoReuse(nPre, nDec, nResult time.Duration) {
	a.nTotal = append(a.nTotal, nPre+nDec+nResult)
	a.nPre = append(a.nPre, nPre)
	a.nDec = append(a.nDec, nDec)
	a.nResult = append(a.nResult, nResult)
}

// throughput derives the cell's reuse total time (ms) and the agents/sec and agent-turns/sec
// rates from the min reuse total across reps, for C concurrent agents over turns per agent.
func (a *reuseAccum) throughput(C, turns int) (rTot, rAgents, rTurns float64) {
	rTot = msFromDur(minDur(a.total))
	rAgents = float64(C) / (rTot / 1e3)
	rTurns = float64(C*turns) / (rTot / 1e3)
	return
}

// metrics folds the accumulated reuse timings into the reuseMetrics summary (min across
// reps, in ms) given the precomputed reuse throughput scalars for the cell.
func (a *reuseAccum) metrics(rTot, rAgents, rTurns float64) reuseMetrics {
	return reuseMetrics{
		ReusePrefillMS: msFromDur(minDur(a.pre)), ReuseCloneMS: msFromDur(minDur(a.clone)),
		ReuseDecodeMS: msFromDur(minDur(a.dec)), ReuseResultMS: msFromDur(minDur(a.result)),
		ReuseTotalMS: rTot, ReuseAgentsSec: rAgents, ReuseAgentTurnsSec: rTurns,
	}
}

// newReuseBatch prefills the shared prefix once and clones it into C agents reserving
// reserve tail slots. It is the reuse-path setup shared by both run modes; the returned
// durations are the prefill and clone timings.
func newReuseBatch(m *model.Model, quant bool, prefix []int, C, reserve int) (*model.BatchSession, time.Duration, time.Duration) {
	t0 := time.Now()
	base := m.NewSession()
	base.Quant = quant
	base.Prefill(prefix)
	tPre := time.Since(t0)

	t1 := time.Now()
	bs := m.NewBatchFromPrefixReserve(base.Cache, C, reserve)
	bs.SetQuant(quant)
	tClone := time.Since(t1)
	return bs, tPre, tClone
}

// newNoReuseBatch builds the no-reuse ablation batch: C independent prefills of the same
// prefix, reserving reserve tail slots. The returned duration is the prefill timing.
func newNoReuseBatch(m *model.Model, quant bool, prefix []int, C, reserve int) (*model.BatchSession, time.Duration) {
	prompts := make([][]int, C)
	for b := range prompts {
		prompts[b] = prefix
	}
	n0 := time.Now()
	nbs := m.NewBatchSession(C)
	nbs.SetQuant(quant)
	nbs.PrefillEachNoLogits(prompts)
	nbs.Reserve(reserve)
	return nbs, time.Since(n0)
}

// writeReport marshals report to indented JSON and writes it to out (or stdout when out
// is empty), exiting on a write error. Shared by both run modes' emit tails.
func writeReport(report map[string]any, out string) {
	blob, _ := json.MarshalIndent(report, "", "  ")
	if out != "" {
		if err := os.WriteFile(out, blob, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", out, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", out)
	} else {
		fmt.Println(string(blob))
	}
}
