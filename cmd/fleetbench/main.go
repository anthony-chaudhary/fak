// Command fleetbench runs the 2-D turn-tax sweep — turns-per-agent (T) × fleet
// size (A) — over the REAL kernel and writes the surface as JSON + CSV for curve
// fitting. It is the agent-count companion to `fak turntax` (one agent) and the
// stochastic harness (one agent, a distribution): here the kernel's process-global
// tier-2 vDSO cache is shared across A interleaved agents, so cross-agent dedup is a
// live kernel event, and each cell is ablated against the same agents run isolated.
//
// THE BASELINE IS A WARM PER-AGENT KV CACHE. The ISOLATED ablation runs the same agents with
// each reusing its OWN KV across turns but WITHOUT cross-agent sharing — exactly a warm
// per-agent cache (vLLM / SGLang / provider prompt-caching). fak's headline is the CROSS-AGENT
// dedup ON TOP of that warm baseline (cross = shared − isolated): the shared-prefix work a
// per-agent cache still pays once per agent. There is no cold no-cache re-prefill arm here.
//
// Usage:
//
//	fleetbench [--profile read-heavy|write-heavy|no-share]
//	           [--turns 1,2,...,50] [--agents 1,2,...,50]   (explicit grids)
//	           [--turn-max 50 --agent-max 50 --grid full|log]   (generated grids)
//	           [--trials 24] [--seed N]
//	           [--out fleet-sweep.json] [--csv fleet-sweep.csv]
//
// The world version is process-global, so a sweep is SERIAL within one process. To
// use many cores, shard the grid across processes with --turns/--agents subsets
// (each process has its own vdso.Default) and concatenate the CSVs.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/turnbench"
	"github.com/anthony-chaudhary/fak/internal/vdso"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func main() {
	fs := flag.NewFlagSet("fleetbench", flag.ExitOnError)
	profileName := fs.String("profile", "read-heavy", "workload profile: read-heavy | write-heavy | no-share")
	turnsArg := fs.String("turns", "", "explicit turn grid, comma-separated (overrides --turn-max/--grid)")
	agentsArg := fs.String("agents", "", "explicit agent grid, comma-separated (overrides --agent-max/--grid)")
	turnMax := fs.Int("turn-max", 50, "max turns per agent (generated grid)")
	agentMax := fs.Int("agent-max", 50, "max agents (generated grid)")
	gridKind := fs.String("grid", "full", "generated grid: full (1..max) | log (saturation-aware, denser low)")
	trials := fs.Int("trials", 24, "seeded trials per cell (more = tighter order stats)")
	seed := fs.Int64("seed", 0x5EED_F1EE, "root seed (whole sweep is reproducible)")
	writeRate := fs.Float64("write-rate", -1, "override the profile's per-turn write probability (-1 = keep profile)")
	sharedPool := fs.Int("shared-pool", -1, "override the profile's distinct shared-read catalog size (-1 = keep profile)")
	granArg := fs.String("granularity", "global", "vDSO invalidation eraser: global (v0.1 full flush) | namespace | resource (the finer erasers)")
	out := fs.String("out", "fleet-sweep.json", "JSON artifact path")
	csv := fs.String("csv", "fleet-sweep.csv", "CSV artifact path (one row per cell)")
	_ = fs.Parse(os.Args[1:])

	p, ok := profileByName(*profileName)
	if !ok {
		fmt.Fprintf(os.Stderr, "fleetbench: unknown profile %q (read-heavy|write-heavy|no-share)\n", *profileName)
		os.Exit(2)
	}
	// Secondary-axis overrides: re-tag the name so the artifact records the variant.
	if *writeRate >= 0 {
		p.PWrite = *writeRate
		p.Name = fmt.Sprintf("%s-w%.2f", p.Name, *writeRate)
	}
	if *sharedPool >= 0 {
		p.SharedPool = *sharedPool
		p.Name = fmt.Sprintf("%s-pool%d", p.Name, *sharedPool)
	}
	// The eraser axis: configure the process-global vDSO invalidation granularity the
	// whole sweep scores under. Global reproduces the v0.1 full-flush crossover; the
	// finer erasers push it out (the headline of this work).
	gran, ok := vdso.ParseGranularity(*granArg)
	if !ok {
		fmt.Fprintf(os.Stderr, "fleetbench: unknown granularity %q (global|namespace|resource)\n", *granArg)
		os.Exit(2)
	}
	turnbench.SetInvalidation(gran)
	if gran != vdso.Global {
		p.Name = fmt.Sprintf("%s-%s", p.Name, gran.String())
	}

	turnGrid := buildGrid(*turnsArg, *turnMax, *gridKind)
	agentGrid := buildGrid(*agentsArg, *agentMax, *gridKind)
	cm := turnbench.DefaultCostModel()

	total := len(turnGrid) * len(agentGrid)
	fmt.Fprintf(os.Stderr, "fleetbench: profile=%s eraser=%s grid=%s turns=%v agents=%v trials=%d => %d cells\n",
		p.Name, gran.String(), *gridKind, turnGrid, agentGrid, *trials, total)
	fmt.Fprintln(os.Stderr, "fleetbench: headline = CROSS-AGENT dedup (cross) vs a WARM per-agent KV cache (the ISOLATED ablation = per-agent cache); no cold no-cache arm")

	t0 := time.Now()
	progress := func(done, total int, c turnbench.FleetCell) {
		el := time.Since(t0)
		var eta time.Duration
		if done > 0 {
			eta = time.Duration(float64(el) / float64(done) * float64(total-done))
		}
		fmt.Fprintf(os.Stderr, "[%4d/%4d] T=%-3d A=%-3d calls=%-5d shared=%-4d isolated=%-4d cross=%-4d  elapsed=%s eta=%s\n",
			done, total, c.Turns, c.Agents, c.Calls,
			c.SharedSaved.P50, c.IsolatedSaved.P50, c.CrossUplift.P50,
			el.Round(time.Second), eta.Round(time.Second))
	}

	sw := turnbench.RunFleetSweep(context.Background(), p, turnGrid, agentGrid, *trials, *seed, cm, progress)

	if err := os.WriteFile(*out, sw.JSON(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "fleetbench:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*csv, sw.CSV(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "fleetbench:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "fleetbench: wrote %s (%d cells) and %s in %s\n",
		*out, len(sw.Cells), *csv, time.Since(t0).Round(time.Second))
}

func profileByName(name string) (turnbench.FleetProfile, bool) {
	switch strings.ToLower(name) {
	case "read-heavy", "read", "rh":
		return turnbench.FleetReadHeavy, true
	case "write-heavy", "write", "wh":
		return turnbench.FleetWriteHeavy, true
	case "no-share", "noshare", "control":
		return turnbench.FleetNoShare, true
	}
	return turnbench.FleetProfile{}, false
}

// buildGrid returns the explicit list if given, else a generated grid up to max.
// "full" is 1..max; "log" is a saturation-aware ladder (dense at the low end where
// the curvature is, sparse past the knee) that always includes 1 and max.
func buildGrid(explicit string, max int, kind string) []int {
	if strings.TrimSpace(explicit) != "" {
		return parseInts(explicit)
	}
	if max < 1 {
		max = 1
	}
	if strings.ToLower(kind) == "log" {
		base := []int{1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 44, 50}
		var out []int
		for _, v := range base {
			if v <= max {
				out = append(out, v)
			}
		}
		if len(out) == 0 || out[len(out)-1] != max {
			out = append(out, max)
		}
		return out
	}
	out := make([]int, 0, max)
	for i := 1; i <= max; i++ {
		out = append(out, i)
	}
	return out
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
			fmt.Fprintf(os.Stderr, "fleetbench: bad int %q in grid\n", part)
			os.Exit(2)
		}
		out = append(out, n)
	}
	return out
}
