package main

// Cold concurrent same-read fill-race probe (#878).
//
// The -parallel proof in main.go proves the WARMED case: prewarm one read per hot
// file, then a parallel hot phase collapses to vDSO tier-2. That is honest, but it
// explicitly does NOT prove that COLD concurrent first reads collapse to one engine
// call. This file measures exactly that cold-miss behavior, and nothing else.
//
// The method: for each trial, start from an EMPTY vDSO world (a freshly bumped world
// epoch + a fresh kernel + a never-seen key), park N workers on a single release
// barrier, then free them all at once against the same read key. We count how many
// ENGINE executions happen before the vDSO tier-2 entry is filled. cold_fill_races =
// engine_calls_per_key - 1 (clamped at zero), so 0 means the concurrent cold misses
// collapsed to a single fill (singleflight); >0 means the first burst fanned out.
//
// HONEST SCOPE. This is a BENCHMARK FIRST. It does NOT require the current kernel to
// pass a one-engine-call invariant — singleflight is not built yet, so MEASURED_RACE
// is the expected, acceptable verdict, reported clearly. It measures cold-fill
// COORDINATION only: not warmed hot-cache hit-rate (that is -parallel) and not any
// provider prompt-cache or model-context token saving. The result-token size is a
// modeled knob, as everywhere in this demo; the cold probe counts engine calls.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// coldResultTokens is the per-key result size used by the cold probe — a modeled knob
// (like every result size in this demo), not a measured production payload. The cold
// probe counts ENGINE CALLS per key, not tokens.
const coldResultTokens = 600

// The closed verdict vocabulary for the cold concurrent same-read probe.
//   - coldVerdictRace: some cold key fanned out to MORE than one engine call.
//   - coldVerdictSingleflight: every cold key collapsed to exactly one engine call.
const (
	coldVerdictRace         = "MEASURED_RACE"
	coldVerdictSingleflight = "SINGLEFLIGHT_CONFIRMED"
)

// coldTrialProof is one cold-barrier trial: N workers released at once against a single
// NEVER-SEEN read key, counting how many engine executions happen before the vDSO
// tier-2 entry is filled.
type coldTrialProof struct {
	Trial         int    `json:"trial"`
	Resource      string `json:"resource"`
	ResultTokens  int    `json:"result_tokens"`
	EngineCalls   int64  `json:"engine_calls_per_key"`
	VDSOHits      int64  `json:"vdso_hits_per_key"`
	ColdFillRaces int64  `json:"cold_fill_races"` // engine_calls_per_key - 1, clamped at 0
	RawWallNs     int64  `json:"raw_wall_ns"`
	FakWallNs     int64  `json:"fak_wall_ns"`
	RawP50Ns      int64  `json:"raw_p50_ns"`
	RawP95Ns      int64  `json:"raw_p95_ns"`
	FakP50Ns      int64  `json:"fak_p50_ns"`
	FakP95Ns      int64  `json:"fak_p95_ns"`
}

// coldProof rolls up the cold-fill-race trials and renders the closed verdict.
type coldProof struct {
	Schema                string           `json:"schema"`
	Workers               int              `json:"workers"`
	Trials                int              `json:"trials"`
	EngineDelayMs         int              `json:"engine_delay_ms"`
	ResultTokens          int              `json:"result_tokens"`
	Verdict               string           `json:"verdict"`
	TotalEngineCalls      int64            `json:"total_engine_calls"`
	TotalVDSOHits         int64            `json:"total_vdso_hits"`
	TotalColdFillRaces    int64            `json:"total_cold_fill_races"`
	MinEngineCallsPerKey  int64            `json:"min_engine_calls_per_key"`
	MaxEngineCallsPerKey  int64            `json:"max_engine_calls_per_key"`
	MeanEngineCallsPerKey float64          `json:"mean_engine_calls_per_key"`
	TrialsWithRace        int              `json:"trials_with_race"`
	RawWallTotalNs        int64            `json:"raw_wall_ns_total"`
	FakWallTotalNs        int64            `json:"fak_wall_ns_total"`
	RawP50Ns              int64            `json:"raw_p50_ns"`
	RawP95Ns              int64            `json:"raw_p95_ns"`
	FakP50Ns              int64            `json:"fak_p50_ns"`
	FakP95Ns              int64            `json:"fak_p95_ns"`
	PerTrial              []coldTrialProof `json:"per_trial"`
}

// coldVerdict maps the worst-case engine-calls-per-key across all trials onto the
// closed verdict vocabulary. SINGLEFLIGHT_CONFIRMED is claimed ONLY when no cold key
// ever fanned out (max <= 1); any trial with >1 cold engine call is a MEASURED_RACE.
func coldVerdict(maxEngineCallsPerKey int64) string {
	if maxEngineCallsPerKey <= 1 {
		return coldVerdictSingleflight
	}
	return coldVerdictRace
}

// coldArmResult is one barrier-released arm's per-call timing + per-call source tally.
type coldArmResult struct {
	durations []int64
	sources   []string
	wallNs    int64
}

// runColdBarrier parks every worker on a single release channel, then frees them all at
// once against fn. The barrier is what makes the measurement COLD: all N workers enter
// fn within the same scheduling window, before any vDSO fill exists, so the result
// reflects how many of them miss and fan out to the engine. Each worker's tool call is
// prebuilt (off the critical path) so only fn's work is timed.
func runColdBarrier(ctx context.Context, tcs []*abi.ToolCall, fn func(context.Context, *abi.ToolCall) string) coldArmResult {
	workers := len(tcs)
	out := coldArmResult{
		durations: make([]int64, workers),
		sources:   make([]string, workers),
	}
	var ready, done sync.WaitGroup
	ready.Add(workers)
	done.Add(workers)
	release := make(chan struct{})
	for w := 0; w < workers; w++ {
		go func(idx int) {
			defer done.Done()
			ready.Done() // parked and ready
			<-release    // all workers wake together
			start := time.Now()
			out.sources[idx] = fn(ctx, tcs[idx])
			out.durations[idx] = elapsedNs(start)
		}(w)
	}
	ready.Wait() // every worker is built and parked
	startWall := time.Now()
	close(release) // the barrier
	done.Wait()
	out.wallNs = elapsedNs(startWall)
	return out
}

// buildColdProof runs the cold concurrent same-read probe over `trials` independent
// trials. Each trial starts from an empty vDSO world and releases `workers` workers at
// one barrier against a single never-seen key, on both the raw (no-kernel) arm and the
// fak (kernel.Syscall) arm.
func buildColdProof(ctx context.Context, workers, trials int, engineDelay time.Duration) (coldProof, error) {
	if workers <= 0 {
		return coldProof{}, fmt.Errorf("workers must be positive")
	}
	if trials <= 0 {
		return coldProof{}, fmt.Errorf("trials must be positive")
	}
	configureFileWorld()
	res := abi.ActiveResolver()
	if res == nil {
		return coldProof{}, fmt.Errorf("no active Ref resolver registered")
	}
	prevDelay := setFileEngineDelay(engineDelay)
	defer setFileEngineDelay(prevDelay)

	rawEngine := fileEngine{}
	out := coldProof{
		Schema:        "fak.tokendemo.parallel-cold.v1",
		Workers:       workers,
		Trials:        trials,
		EngineDelayMs: int(engineDelay / time.Millisecond),
		ResultTokens:  coldResultTokens,
	}
	var allRawDur, allFakDur []int64
	minEngine, maxEngine := int64(-1), int64(0)

	buildTCs := func(call turnbench.Call) ([]*abi.ToolCall, error) {
		tcs := make([]*abi.ToolCall, workers)
		for i := range tcs {
			tc, err := traceToolCall(ctx, res, call)
			if err != nil {
				return nil, err
			}
			tcs[i] = tc
		}
		return tcs, nil
	}

	for tr := 0; tr < trials; tr++ {
		// A never-seen key for THIS trial: a distinct path AND a freshly bumped vDSO
		// world below — so the read is provably cold regardless of any prior trial.
		key := fmt.Sprintf("cold/trial-%d/never-seen.dat", tr)
		call := turnbench.Call{
			Tool: "read_file",
			Args: json.RawMessage(`{"path":"` + key + `"}`),
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
				"result_tokens":  strconv.Itoa(coldResultTokens),
			},
		}

		// RAW arm: no kernel — every worker runs the engine. The cold-world baseline.
		rawTCs, err := buildTCs(call)
		if err != nil {
			return coldProof{}, err
		}
		resetFileEngineStats()
		rawRun := runColdBarrier(ctx, rawTCs, func(ctx context.Context, c *abi.ToolCall) string {
			_, _ = rawEngine.Complete(ctx, c)
			return "engine"
		})

		// FAK arm: a fresh EMPTY vDSO world + a fresh kernel, then the barrier release.
		fakTCs, err := buildTCs(call)
		if err != nil {
			return coldProof{}, err
		}
		resetFileEngineStats()
		vdso.Default.BumpWorld()
		ifc.Default.Reset("")
		k := kernel.New("localtools")
		k.SetVDSO(true)
		fakRun := runColdBarrier(ctx, fakTCs, func(ctx context.Context, c *abi.ToolCall) string {
			r, v := k.Syscall(ctx, c)
			_, source, _ := timingClass(r, v)
			return source
		})

		engineCalls := fileEngineCalls()
		vdsoHits := int64(countSource(fakRun.sources, "vdso_tier2"))
		races := engineCalls - 1
		if races < 0 {
			races = 0
		}

		out.PerTrial = append(out.PerTrial, coldTrialProof{
			Trial:         tr,
			Resource:      key,
			ResultTokens:  coldResultTokens,
			EngineCalls:   engineCalls,
			VDSOHits:      vdsoHits,
			ColdFillRaces: races,
			RawWallNs:     rawRun.wallNs,
			FakWallNs:     fakRun.wallNs,
			RawP50Ns:      percentileNs(rawRun.durations, 50),
			RawP95Ns:      percentileNs(rawRun.durations, 95),
			FakP50Ns:      percentileNs(fakRun.durations, 50),
			FakP95Ns:      percentileNs(fakRun.durations, 95),
		})
		out.TotalEngineCalls += engineCalls
		out.TotalVDSOHits += vdsoHits
		out.TotalColdFillRaces += races
		out.RawWallTotalNs += rawRun.wallNs
		out.FakWallTotalNs += fakRun.wallNs
		if races > 0 {
			out.TrialsWithRace++
		}
		if engineCalls > maxEngine {
			maxEngine = engineCalls
		}
		if minEngine < 0 || engineCalls < minEngine {
			minEngine = engineCalls
		}
		allRawDur = append(allRawDur, rawRun.durations...)
		allFakDur = append(allFakDur, fakRun.durations...)
	}

	if minEngine < 0 {
		minEngine = 0
	}
	out.MinEngineCallsPerKey = minEngine
	out.MaxEngineCallsPerKey = maxEngine
	out.MeanEngineCallsPerKey = float64(out.TotalEngineCalls) / float64(trials)
	out.RawP50Ns = percentileNs(allRawDur, 50)
	out.RawP95Ns = percentileNs(allRawDur, 95)
	out.FakP50Ns = percentileNs(allFakDur, 50)
	out.FakP95Ns = percentileNs(allFakDur, 95)
	out.Verdict = coldVerdict(out.MaxEngineCallsPerKey)
	return out, nil
}

func runColdJSON(workers, trials int, delay time.Duration) int {
	proof, err := buildColdProof(context.Background(), workers, trials, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cold proof: %v\n", err)
		return 1
	}
	b, _ := json.MarshalIndent(proof, "", "  ")
	fmt.Println(string(b))
	return 0
}

func runColdPrint(workers, trials int, delay time.Duration) int {
	p := colors()
	proof, err := buildColdProof(context.Background(), workers, trials, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cold proof: %v\n", err)
		return 1
	}

	fmt.Printf("\n  %s — %d workers, %d trials (engine delay %dms)\n",
		p.paint(p.bold, "fak · COLD concurrent same-read fill-race probe"), proof.Workers, proof.Trials, proof.EngineDelayMs)
	fmt.Printf("  %s\n\n", p.paint(p.dim, "all workers released at ONE barrier against a NEVER-SEEN key — counts engine calls before the vDSO fill exists"))

	verdictColor := p.green
	verdictNote := "exactly one cold engine call per key — concurrent cold misses coalesce to a single fill"
	if proof.Verdict == coldVerdictRace {
		verdictColor = p.red
		verdictNote = "the first cold burst fans out to MULTIPLE engine calls — no singleflight on the cold-miss path yet (a measurement, not a regression)"
	}
	fmt.Printf("  verdict: %s\n  %s\n\n", p.paint(p.bold+verdictColor, proof.Verdict), p.paint(p.dim, verdictNote))

	fmt.Printf("  engine calls per key: min %d  max %d  mean %.1f\n",
		proof.MinEngineCallsPerKey, proof.MaxEngineCallsPerKey, proof.MeanEngineCallsPerKey)
	fmt.Printf("  cold fill races: %d total across %d trial%s (%d raced)   vDSO hits per key: %d total\n",
		proof.TotalColdFillRaces, proof.Trials, plural(proof.Trials), proof.TrialsWithRace, proof.TotalVDSOHits)
	fmt.Printf("  wall (sum over trials): raw %.3fms   fak %.3fms\n",
		nsToMs(proof.RawWallTotalNs), nsToMs(proof.FakWallTotalNs))
	fmt.Printf("  per-call p50/p95: raw %s/%sms   fak %s/%sms\n\n",
		formatMs(proof.RawP50Ns), formatMs(proof.RawP95Ns), formatMs(proof.FakP50Ns), formatMs(proof.FakP95Ns))

	fmt.Printf("  %-5s  %-30s  %8s  %8s  %8s  %10s  %10s\n",
		"trial", "resource", "engine", "vdso", "races", "raw_wall", "fak_wall")
	fmt.Printf("  %s\n", strings.Repeat("─", 92))
	for _, tr := range proof.PerTrial {
		color := p.green
		if tr.ColdFillRaces > 0 {
			color = p.red
		}
		fmt.Printf("  %s\n", p.paint(color, fmt.Sprintf("%-5d  %-30s  %8d  %8d  %8d  %8sms  %8sms",
			tr.Trial,
			padTrim(tr.Resource, 30),
			tr.EngineCalls,
			tr.VDSOHits,
			tr.ColdFillRaces,
			formatMs(tr.RawWallNs),
			formatMs(tr.FakWallNs),
		)))
	}
	fmt.Printf("\n  %s\n\n", p.paint(p.dim,
		"scope: COLD-FILL COORDINATION only — not warmed hot-cache hit-rate (-parallel) and not provider/model token savings."))
	return 0
}
