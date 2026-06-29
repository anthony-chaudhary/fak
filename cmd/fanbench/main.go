// Command fanbench runs the ONE-MASTER-GOAL → N-SUBAGENT fan-out sweep — the
// orchestrator-worker topology (one lead decomposes a goal, spawns N sub-agents, folds
// their results) swept from N=1 to N=1000+, the regime no public benchmark maps (see
// experiments/fanout/RESEARCH-BRIEF-fanout-2026-06-17.md). It is the fan-out companion
// to `fleetbench` (A independent agents): here the N sub-agents decompose the SAME
// goal, so cross-agent tier-2 dedup is structurally higher, and the master-goal prefix
// is the ideal shared-prefix-KV-reuse case.
//
// THE BASELINE IS A WARM CACHE. The cost model already prices a warm per-agent prompt cache
// (Anthropic-style cache-read / cache-write prefix economics), so fak's headline number is the
// CROSS-AGENT prefix dedup ON TOP of that warm cache — the (N−1)·prefix_tokens a per-agent
// cache still pays once per sub-agent but the shared goal prefill does only once for the whole
// fleet. There is no cold no-cache re-prefill arm here; every number is already measured against
// the warm per-agent baseline.
//
// Two halves, kept separate (the honesty line):
//   - MEASURED on the real kernel: the cross-agent tool-result dedup (cross_uplift =
//     SHARED-world fan-out − ISOLATED-world sub-agents), a measured path-swap, and the
//     exact prefix-reuse geometry (N−1)·prefix_tokens the kernel does not redo
//     (NewBatchFromPrefix prefills the goal prefix once + clones; witnessed bit-identical
//     by `go test ./cmd/fanbench`).
//   - MODELED by a transparent, knobbed cost model: the token multiplier vs a single
//     agent (with Anthropic-style prompt-cache prefix economics), critical-path vs
//     total-work latency, throughput, and the saturation knee.
//
// Usage:
//
//	fanbench [--profile research|write-heavy|no-share]
//	         [--agents 1,2,4,...,1024]   (explicit) | [--agent-max 1024 --grid log|full|canonical]
//	         [--sub-turns 4]             (explicit grid, comma-separated)
//	         [--trials 12] [--seed N]
//	         [--prefix 2048] | [--prefixes smoke,small,medium,long,big|all|1024,8k,max]
//	         [--model-context 131072] [--model-config /path/to/config.json]
//	         [--suffix 256 --decode 120 --fold 200 --fold-budget 4000]
//	         [--cache-read 0.1 --cache-write 1.25 --turn-latency-ms 1500]
//	         [--out fanout.json] [--csv fanout.csv]
//
// The world version is process-global, so a sweep is SERIAL within one process; shard
// the agent grid across processes (each has its own vdso.Default) and concatenate CSVs.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

const defaultModelContextTokens = 131072

func main() {
	fs := flag.NewFlagSet("fanbench", flag.ExitOnError)
	profileName := fs.String("profile", "research", "workload profile: research | write-heavy | no-share")
	agentsArg := fs.String("agents", "", "explicit fan-out grid, comma-separated (overrides --agent-max/--grid)")
	agentMax := fs.Int("agent-max", 1024, "max fan-out width N (generated grid)")
	gridKind := fs.String("grid", "log", "generated grid: log (powers-of-two ladder to max) | full (1..max) | canonical (D-001: 1,100,500,1000)")
	subTurnsArg := fs.String("sub-turns", "4", "turns per sub-agent, comma-separated grid")
	trials := fs.Int("trials", 12, "seeded trials per cell (more = tighter order stats)")
	seed := fs.Int64("seed", 0x5EED_F1EE, "root seed (whole sweep is reproducible)")

	// Cost-model knobs (the MODELED half; defaults = DefaultFanoutCostModel).
	prefix := fs.Int("prefix", 2048, "master-goal shared prefix tokens P (system + goal + tool schemas)")
	prefixesArg := fs.String("prefixes", "", "prefix grid: comma-separated ints/K-values or presets smoke,small,medium,long,big|max; all = smoke..big (overrides --prefix)")
	modelContext := fs.Int("model-context", defaultModelContextTokens, "model context window tokens used by big/max prefix presets")
	modelConfig := fs.String("model-config", "", "optional HF config.json or snapshot dir; max_position_embeddings overrides --model-context for big/max presets")
	suffix := fs.Int("suffix", 256, "per-sub-agent private prompt tokens (the sub-task slice)")
	decode := fs.Int("decode", 120, "assistant tokens generated per sub-agent turn")
	fold := fs.Int("fold", 200, "master fold tokens to ingest one sub-result")
	foldBudget := fs.Int("fold-budget", 4000, "tokens the master synthesizes per fold turn")
	dollarsIn := fs.Float64("dollars-in", 3.0, "$/Mtok input")
	dollarsOut := fs.Float64("dollars-out", 15.0, "$/Mtok output")
	cacheRead := fs.Float64("cache-read", 0.1, "cached-input price multiple (Anthropic ~0.1)")
	cacheWrite := fs.Float64("cache-write", 1.25, "cache-write price multiple (Anthropic ~1.25)")
	turnLatency := fs.Float64("turn-latency-ms", 1500, "one model round-trip (ms)")

	out := fs.String("out", "fanout.json", "JSON artifact path")
	csv := fs.String("csv", "fanout.csv", "CSV artifact path (one row per cell)")
	scale := fs.Bool("scale", false, "run the dedicated D-001 SCALE harness (internal/bench.RunFanScale): emit the coordination-overhead-vs-N=1-baseline + cross-agent-reuse report at the canonical 1/100/500/1000 grid (or --agents); JSON only, prefix-sweep flags ignored")
	_ = fs.Parse(os.Args[1:])

	p, ok := profileByName(*profileName)
	if !ok {
		fmt.Fprintf(os.Stderr, "fanbench: unknown profile %q (research|write-heavy|no-share)\n", *profileName)
		os.Exit(2)
	}

	cm := turnbench.FanoutCostModel{
		Version:      turnbench.FanoutCostModelVersion,
		PrefixTokens: *prefix, SuffixTokens: *suffix, DecodeTokensPerTurn: *decode,
		FoldTokensPerResult: *fold, FoldTurnTokenBudget: *foldBudget,
		DollarsPerMTokIn: *dollarsIn, DollarsPerMTokOut: *dollarsOut,
		CacheReadMult: *cacheRead, CacheWriteMult: *cacheWrite, TurnLatencyMs: *turnLatency,
	}

	agentGrid := buildAgentGrid(*agentsArg, *agentMax, *gridKind)
	subTurnGrid := parseInts(*subTurnsArg)
	if len(subTurnGrid) == 0 {
		subTurnGrid = []int{4}
	}

	// --scale: run the dedicated D-001 acceptance harness (internal/bench.RunFanScale),
	// which carries the named "coordination overhead vs the N=1 baseline" metric next to
	// the cross-agent reuse uplift, at the canonical 1/100/500/1000 ladder. Deterministic
	// and host-runnable (in-process kernel arithmetic, no model call); the live-model
	// SOTA-framework comparison stays a DeferredRun, never fabricated here.
	if *scale {
		grid := agentGrid
		if strings.TrimSpace(*agentsArg) == "" && strings.ToLower(*gridKind) != "canonical" {
			grid = bench.CanonicalFanScaleGrid // default the scale harness to the acceptance ladder, not the log grid
		}
		fmt.Fprintf(os.Stderr, "fanbench --scale: D-001 harness profile=%s grid=%v sub-turns=%d trials=%d seed=%d => coord-overhead-vs-N=1 + cross-agent reuse (live-model SOTA-framework comparison deferred)\n",
			p.Name, grid, subTurnGrid[0], *trials, *seed)
		if err := runScale(grid, subTurnGrid[0], *trials, *seed, p, cm, *out); err != nil {
			fmt.Fprintln(os.Stderr, "fanbench:", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "fanbench --scale: wrote %s\n", *out)
		return
	}
	if *modelConfig != "" {
		n, err := contextWindowFromConfig(*modelConfig)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fanbench: model-config:", err)
			os.Exit(2)
		}
		*modelContext = n
	}
	prefixGrid, err := buildPrefixGrid(*prefixesArg, *prefix, *modelContext, *suffix, *decode, subTurnGrid)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fanbench:", err)
		os.Exit(2)
	}

	total := len(prefixGrid) * len(agentGrid) * len(subTurnGrid)
	fmt.Fprintf(os.Stderr, "fanbench: profile=%s agents=%v sub-turns=%v prefixes=%v trials=%d => %d cells\n",
		p.Name, agentGrid, subTurnGrid, prefixGrid, *trials, total)
	fmt.Fprintln(os.Stderr, "fanbench: headline = CROSS-AGENT prefix dedup (cross) vs a WARM per-agent prompt cache — no cold no-cache arm; every number is already against the warm baseline")

	t0 := time.Now()
	progress := func(done, total int, c turnbench.FanoutCell) {
		el := time.Since(t0)
		var eta time.Duration
		if done > 0 {
			eta = time.Duration(float64(el) / float64(done) * float64(total-done))
		}
		fmt.Fprintf(os.Stderr, "[%3d/%3d] P=%-7d N=%-4d T=%-2d calls=%-6d shared=%-5d isolated=%-5d cross=%-5d  tax_back=%4.1f%% speedup=%5.1f  elapsed=%s eta=%s\n",
			done, total, c.PrefixTokens, c.Agents, c.SubTurns, c.Calls,
			c.SharedSaved.P50, c.IsolatedSaved.P50, c.CrossUplift.P50,
			c.TaxClawedBack*100, c.ParallelSpeedup,
			el.Round(time.Second), eta.Round(time.Second))
	}

	sw := turnbench.RunFanoutPrefixSweep(context.Background(), p, agentGrid, subTurnGrid, prefixGrid, *trials, *seed, cm, progress)

	if err := benchcli.WriteReport(*out, sw.JSON()); err != nil {
		fmt.Fprintln(os.Stderr, "fanbench:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*csv, sw.CSV(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "fanbench:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "fanbench: wrote %s (%d cells) and %s in %s\n",
		*out, len(sw.Cells), *csv, time.Since(t0).Round(time.Second))
}

// runScale runs the dedicated D-001 fan-out SCALE harness (internal/bench.RunFanScale) at
// the given grid and writes its JSON report — the one artifact that carries the named
// "coordination overhead vs the N=1 baseline" metric (coord_overhead_turns / _frac)
// alongside the cross-agent reuse uplift and the exact (N−1)·prefix reuse geometry. The
// run is deterministic in (profile, grid, subTurns, trials, seed); the published live-model
// LangGraph/AutoGen/CrewAI comparison stays a DeferredRun in the report, never fabricated.
func runScale(grid []int, subTurns, trials int, seed int64, prof turnbench.FanoutProfile, cm turnbench.FanoutCostModel, outPath string) error {
	rep := bench.RunFanScale(context.Background(), bench.FanScaleOptions{
		Profile: prof, Cost: cm, Grid: grid, SubTurns: subTurns, Trials: trials, Seed: seed,
	})
	b, err := benchcli.MarshalReport(rep)
	if err != nil {
		return err
	}
	return os.WriteFile(outPath, append(b, '\n'), 0o644)
}

func profileByName(name string) (turnbench.FanoutProfile, bool) {
	switch strings.ToLower(name) {
	case "research", "research-goal", "read":
		return turnbench.FanoutResearch, true
	case "write-heavy", "write", "write-goal", "wh":
		return turnbench.FanoutWriteHeavy, true
	case "no-share", "noshare", "control":
		return turnbench.FanoutNoShare, true
	}
	return turnbench.FanoutProfile{}, false
}

// buildAgentGrid returns the explicit list if given, else a generated grid up to max.
// "log" is a powers-of-two ladder (1,2,4,8,...,max) that always includes 1 and max —
// the right spacing for a fan-out curve that spans three orders of magnitude; "full"
// is 1..max. "canonical" is the D-001 acceptance ladder.
func buildAgentGrid(explicit string, max int, kind string) []int {
	if strings.TrimSpace(explicit) != "" {
		return parseInts(explicit)
	}
	if max < 1 {
		max = 1
	}
	switch strings.ToLower(kind) {
	case "canonical", "acceptance", "d001":
		return []int{1, 100, 500, 1000}
	case "full":
		out := make([]int, 0, max)
		for i := 1; i <= max; i++ {
			out = append(out, i)
		}
		return out
	}
	out := []int{}
	for v := 1; v <= max; v *= 2 {
		out = append(out, v)
	}
	if out[len(out)-1] != max {
		out = append(out, max)
	}
	return out
}

func buildPrefixGrid(spec string, fallback, modelContext, suffix, decode int, subTurns []int) ([]int, error) {
	if strings.TrimSpace(spec) == "" {
		if fallback < 1 {
			return nil, fmt.Errorf("--prefix must be >=1, got %d", fallback)
		}
		return []int{fallback}, nil
	}

	maxPrefix, err := maxContextPrefix(modelContext, suffix, decode, subTurns)
	if err != nil {
		return nil, err
	}

	var out []int
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		switch strings.ToLower(part) {
		case "all", "range", "default", "defaults":
			out = append(out, 1024, 2048, 8192, 32768, maxPrefix)
		case "smoke":
			out = append(out, 1024)
		case "small", "short":
			out = append(out, 2048)
		case "medium", "med":
			out = append(out, 8192)
		case "long", "large":
			out = append(out, 32768)
		case "big", "max", "context", "full":
			out = append(out, maxPrefix)
		default:
			n, err := parseTokenCount(part)
			if err != nil {
				return nil, fmt.Errorf("bad prefix %q: %w", part, err)
			}
			out = append(out, n)
		}
	}
	out = uniquePositive(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("--prefixes produced no values")
	}
	return out, nil
}

func maxContextPrefix(modelContext, suffix, decode int, subTurns []int) (int, error) {
	if modelContext < 1 {
		return 0, fmt.Errorf("--model-context must be >=1, got %d", modelContext)
	}
	maxSubTurns := 1
	for _, t := range subTurns {
		if t > maxSubTurns {
			maxSubTurns = t
		}
	}
	reserve := suffix + decode*maxSubTurns
	prefix := modelContext - reserve
	if prefix < 1 {
		return 0, fmt.Errorf("big/max prefix cannot fit: model context %d <= suffix %d + max(sub-turns)*decode %d", modelContext, suffix, decode*maxSubTurns)
	}
	return prefix, nil
}

func parseTokenCount(s string) (int, error) {
	mult := 1
	raw := strings.TrimSpace(strings.ToLower(s))
	switch {
	case strings.HasSuffix(raw, "k"):
		mult = 1024
		raw = strings.TrimSuffix(raw, "k")
	case strings.HasSuffix(raw, "m"):
		mult = 1024 * 1024
		raw = strings.TrimSuffix(raw, "m")
	}
	if raw == "" {
		return 0, fmt.Errorf("empty token count")
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, err
	}
	n *= mult
	if n < 1 {
		return 0, fmt.Errorf("must be >=1, got %d", n)
	}
	return n, nil
}

func uniquePositive(vals []int) []int {
	seen := map[int]bool{}
	out := make([]int, 0, len(vals))
	for _, v := range vals {
		if v < 1 || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out
}

func contextWindowFromConfig(path string) (int, error) {
	if st, err := os.Stat(path); err == nil && st.IsDir() {
		path = filepath.Join(path, "config.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var cfg struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
		ModelMaxLength        int `json:"model_max_length"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return 0, err
	}
	if cfg.MaxPositionEmbeddings > 0 {
		return cfg.MaxPositionEmbeddings, nil
	}
	if cfg.ModelMaxLength > 0 {
		return cfg.ModelMaxLength, nil
	}
	return 0, fmt.Errorf("%s has no max_position_embeddings or model_max_length", path)
}

func parseInts(s string) []int {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fanbench: bad int %q in grid\n", part)
			os.Exit(2)
		}
		if n < 1 {
			fmt.Fprintf(os.Stderr, "fanbench: grid values must be >=1, got %d\n", n)
			os.Exit(2)
		}
		out = append(out, n)
	}
	return out
}
