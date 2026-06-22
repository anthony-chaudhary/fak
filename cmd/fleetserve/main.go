// Command fleetserve measures the CROSS-AGENT SHARED-PREFIX fleet workload — the regime
// where fak's kernel-owned KV cache structurally beats a per-slot serving engine (llama.cpp)
// by more than 2×, not by a faster kernel but by NOT REDOING WORK.
//
// The workload: C concurrent agents that all share one long prefix (a system prompt + tool
// schemas, P tokens), each running T model turns. A turn decodes D assistant tokens; between
// turns, each agent ingests R private tool/result tokens. That gives the direct T×C surface
// the agent-serving claim needs: big shared preamble, short per-agent answers, growing
// per-agent KV across turns.
//
//   - fak (reuse):   prefill the P-token prefix ONCE, Clone its KV into all C agents
//     with NewBatchFromPrefix, then batched-decode all C in lockstep. Prefix
//     prefill work is P tokens, total, regardless of C.
//   - fak (no-reuse): the ablation — C agents each prefill the whole prefix (NewBatchSession
//     and PrefillEach), then the same batched decode. Prefix prefill work = C·P.
//     This isolates the reuse as the win (same kernel, same decode).
//
// The no-reuse ablation isolates the value of prefix reuse inside fak (same kernels, same
// decode). For an external peer, use internal/model/bench_llamacpp_turn_agents.py; it measures
// the same T×C shape directly against llama.cpp's low-level multi-sequence API.
//
// Headline = agents/sec (C ÷ end-to-end wall-clock) and the reuse speedup (reuse ÷ no-reuse).
// Best (min) wall-clock over reps — least-contended sampling, the MODEL-BASELINE methodology.
//
// Usage:
//
//	fleetserve -quant -prefix 1024 -turns 1,2,4 -decode 32 -result 128 -concurrency 1,8,16,32,64 [-out f.json]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func lcgIDs(n, vocab, seed int) []int {
	ids := make([]int, n)
	state := uint64(2463534242 + seed)
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

func minDur(ds []time.Duration) time.Duration {
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return cp[0]
}

type point struct {
	Turns        int `json:"turns"`
	Concurrency  int `json:"concurrency"`
	PrefixLen    int `json:"prefix_len"`
	DecodeSteps  int `json:"decode_steps"`
	ResultTokens int `json:"result_tokens_between_turns"`
	Reps         int `json:"reps"`

	// fak with cross-agent prefix reuse (prefill once + clone C + batched decode)
	ReusePrefillMS     float64 `json:"reuse_prefill_ms"` // one prefix prefill
	ReuseCloneMS       float64 `json:"reuse_clone_ms"`   // C deep-copies of the prefix KV
	ReuseDecodeMS      float64 `json:"reuse_decode_ms"`  // T*D batched decode steps
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

func main() {
	dir := flag.String("dir", "internal/model/.cache/smollm2-135m", "model export dir")
	out := flag.String("out", "", "write JSON result here (default stdout)")
	quant := flag.Bool("quant", true, "use the Q8_0 quantized lane (apples-to-apples with llama.cpp Q8)")
	prefixLen := flag.Int("prefix", 1024, "shared prefix length (system prompt + tool schemas)")
	turnsArg := flag.String("turns", "1", "comma-separated turn counts per agent to sweep")
	decodeSteps := flag.Int("decode", 32, "assistant tokens decoded per agent turn")
	resultTokens := flag.Int("result", 0, "private tool/result tokens appended per agent between turns")
	reps := flag.Int("reps", 3, "reps per concurrency (best/min wall-clock)")
	concArg := flag.String("concurrency", "1,8,16,32,64", "comma-separated concurrency (agents) to sweep")
	ablation := flag.Bool("ablation", true, "also run the no-reuse ablation after each reuse cell")
	// transcript-derived replay mode (see workload.go): -workload feeds a fak.workload.v1
	// profile (from tools/transcript_workload.py) and replays a real per-turn track; the
	// -tune-* knobs are the "what if" layer over that real shape.
	workload := flag.String("workload", "", "fak.workload.v1 profile JSON: replay a real transcript track instead of the flat synthetic shape")
	trackPct := flag.Int("track-pct", 90, "replay the track at this turns-percentile (closest); 0 = largest")
	tuneToolFrac := flag.Float64("tune-toolfrac", 1.0, "scale the tool-call fraction (fraction of turns that ingest a result)")
	tuneResult := flag.Float64("tune-result", 1.0, "scale per-tool-turn result tokens R")
	tuneDecode := flag.Float64("tune-decode", 1.0, "scale per-turn decode tokens D")
	tunePrefix := flag.Float64("tune-prefix", 1.0, "scale the shared prefix P")
	turnCap := flag.Int("turn-cap", 0, "cap replayed turns (0 = full track); keeps sweeps tractable")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*dir = pathutil.ExpandTilde(*dir)

	turnsGrid := parseInts(*turnsArg)
	concs := parseInts(*concArg)
	m, err := model.Load(*dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	if *quant {
		m.Quantize()
	}
	vocab := m.Cfg.VocabSize
	warmFleet(m, *quant, vocab)

	// transcript-derived replay mode short-circuits the synthetic surface below.
	if *workload != "" {
		prof, err := loadWorkload(*workload)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load workload: %v\n", err)
			os.Exit(1)
		}
		tn := tune{
			toolFrac: *tuneToolFrac, result: *tuneResult, decode: *tuneDecode,
			prefix: *tunePrefix, turnCap: *turnCap,
		}
		runWorkloadMode(m, *quant, vocab, prof, *trackPct, tn, concs, *reps, *ablation, *out)
		return
	}

	prefix := lcgIDs(*prefixLen, vocab, 1)

	var points []point
	for _, T := range turnsGrid {
		if T < 1 {
			continue
		}
		for _, C := range concs {
			if C < 1 {
				continue
			}
			// per-user starting tokens (distinct — agents diverge immediately; values don't
			// affect matmul/attention cost, only which weights' rows are read).
			ids0 := lcgIDs(C, vocab, 991)

			var reuseTotals, reusePre, reuseClone, reuseDec, reuseResult []time.Duration
			var noReuseTotals, noReusePre, noReuseDec, noReuseResult []time.Duration
			tailTokens := T * *decodeSteps
			if T > 1 {
				tailTokens += (T - 1) * *resultTokens
			}
			for r := 0; r < *reps; r++ {
				resultPrompts := buildResultPrompts(T, C, *resultTokens, vocab, r)

				// ---- fak REUSE: prefill once, clone C, batched decode ----
				t0 := time.Now()
				base := m.NewSession()
				base.Quant = *quant
				base.Prefill(prefix)
				tPre := time.Since(t0)

				t1 := time.Now()
				bs := m.NewBatchFromPrefixReserve(base.Cache, C, tailTokens)
				bs.SetQuant(*quant)
				tClone := time.Since(t1)

				tDec, tResult := runTurns(bs, ids0, resultPrompts, *decodeSteps, vocab)
				reuseTotals = append(reuseTotals, tPre+tClone+tDec+tResult)
				reusePre = append(reusePre, tPre)
				reuseClone = append(reuseClone, tClone)
				reuseDec = append(reuseDec, tDec)
				reuseResult = append(reuseResult, tResult)

				if *ablation {
					// ---- fak NO-REUSE: C independent prefix prefills, same batched decode ----
					prompts := make([][]int, C)
					for b := range prompts {
						prompts[b] = prefix
					}
					n0 := time.Now()
					nbs := m.NewBatchSession(C)
					nbs.SetQuant(*quant)
					nbs.PrefillEachNoLogits(prompts)
					nbs.Reserve(tailTokens)
					nPre := time.Since(n0)

					nDec, nResult := runTurns(nbs, ids0, resultPrompts, *decodeSteps, vocab)
					noReuseTotals = append(noReuseTotals, nPre+nDec+nResult)
					noReusePre = append(noReusePre, nPre)
					noReuseDec = append(noReuseDec, nDec)
					noReuseResult = append(noReuseResult, nResult)
				}

				runtime.GC()
			}

			ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
			rTot := ms(minDur(reuseTotals))
			rAgents := float64(C) / (rTot / 1e3)
			rTurns := float64(C*T) / (rTot / 1e3)
			pt := point{
				Turns: T, Concurrency: C, PrefixLen: *prefixLen, DecodeSteps: *decodeSteps,
				ResultTokens: *resultTokens, Reps: *reps,
				ReusePrefillMS: ms(minDur(reusePre)), ReuseCloneMS: ms(minDur(reuseClone)),
				ReuseDecodeMS: ms(minDur(reuseDec)), ReuseResultMS: ms(minDur(reuseResult)),
				ReuseTotalMS: rTot, ReuseAgentsSec: rAgents, ReuseAgentTurnsSec: rTurns,
			}
			if *ablation {
				nTot := ms(minDur(noReuseTotals))
				nAgents := float64(C) / (nTot / 1e3)
				nTurns := float64(C*T) / (nTot / 1e3)
				nPre := ms(minDur(noReusePre))
				nDec := ms(minDur(noReuseDec))
				nResult := ms(minDur(noReuseResult))
				speedup := rAgents / nAgents
				pt.NoReusePrefillMS = &nPre
				pt.NoReuseDecodeMS = &nDec
				pt.NoReuseResultMS = &nResult
				pt.NoReuseTotalMS = &nTot
				pt.NoReuseAgentsSec = &nAgents
				pt.NoReuseAgentTurnsSec = &nTurns
				pt.ReuseSpeedup = &speedup
			}
			points = append(points, pt)
			if *ablation {
				fmt.Fprintf(os.Stderr,
					"T=%-2d C=%-3d P=%d D=%d R=%d | reuse %.0f ms (pre %.0f + clone %.0f + dec %.0f + res %.0f) = %.1f turns/s | "+
						"no-reuse %.0f ms = %.1f turns/s | reuse %.2f×\n",
					T, C, *prefixLen, *decodeSteps, *resultTokens, rTot, pt.ReusePrefillMS,
					pt.ReuseCloneMS, pt.ReuseDecodeMS, pt.ReuseResultMS, rTurns, *pt.NoReuseTotalMS,
					*pt.NoReuseAgentTurnsSec, *pt.ReuseSpeedup)
			} else {
				fmt.Fprintf(os.Stderr,
					"T=%-2d C=%-3d P=%d D=%d R=%d | reuse %.0f ms (pre %.0f + clone %.0f + dec %.0f + res %.0f) = %.1f turns/s | no-reuse skipped\n",
					T, C, *prefixLen, *decodeSteps, *resultTokens, rTot, pt.ReusePrefillMS,
					pt.ReuseCloneMS, pt.ReuseDecodeMS, pt.ReuseResultMS, rTurns)
			}
		}
	}

	report := map[string]any{
		"app_version":                 appversion.Current(),
		"engine":                      "fak fleetserve (cross-agent shared-prefix, Q8=" + boolStr(*quant) + ")",
		"model":                       "SmolLM2-135M",
		"go_threads":                  runtime.GOMAXPROCS(0),
		"prefix_len":                  *prefixLen,
		"turn_grid":                   turnsGrid,
		"agent_grid":                  concs,
		"decode_steps_per_turn":       *decodeSteps,
		"result_tokens_between_turns": *resultTokens,
		"ablation":                    *ablation,
		"points":                      points,
	}
	blob, _ := json.MarshalIndent(report, "", "  ")
	if *out != "" {
		if err := os.WriteFile(*out, blob, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", *out, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", *out)
	} else {
		fmt.Println(string(blob))
	}
}

func buildResultPrompts(turns, agents, resultTokens, vocab, rep int) [][][]int {
	if turns <= 1 || resultTokens <= 0 {
		return nil
	}
	out := make([][][]int, turns-1)
	for t := range out {
		out[t] = make([][]int, agents)
		for a := range out[t] {
			out[t][a] = lcgIDs(resultTokens, vocab, 10_000+rep*1_000_000+t*10_000+a*97)
		}
	}
	return out
}

func warmFleet(m *model.Model, quant bool, vocab int) {
	prefix := lcgIDs(8, vocab, 77)
	base := m.NewSession()
	base.Quant = quant
	base.Prefill(prefix)
	bs := m.NewBatchFromPrefix(base.Cache, 4)
	bs.SetQuant(quant)
	bs.StepBatch(lcgIDs(4, vocab, 88))
}

func runTurns(bs *model.BatchSession, ids0 []int, resultPrompts [][][]int, decodeSteps, vocab int) (time.Duration, time.Duration) {
	ids := append([]int(nil), ids0...)
	var decodeTotal, resultTotal time.Duration
	turns := len(resultPrompts) + 1
	for t := 0; t < turns; t++ {
		t0 := time.Now()
		for s := 0; s < decodeSteps; s++ {
			bs.StepBatch(ids)
			for i := range ids {
				ids[i] = (ids[i]*48271 + 1) % vocab
			}
		}
		decodeTotal += time.Since(t0)
		if t < len(resultPrompts) {
			t1 := time.Now()
			bs.PrefillEachNoLogits(resultPrompts[t])
			resultTotal += time.Since(t1)
		}
	}
	return decodeTotal, resultTotal
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func parseInts(s string) []int {
	var out []int
	cur, has := 0, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' {
			cur = cur*10 + int(c-'0')
			has = true
		} else if has {
			out = append(out, cur)
			cur, has = 0, false
		}
	}
	if has {
		out = append(out, cur)
	}
	return out
}
