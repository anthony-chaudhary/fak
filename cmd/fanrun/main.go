// Command fanrun runs the MEASURED one-master-goal → N-subagent fan-out — N actual
// agent.RunArm sessions, each a real loop through a real kernel with the vDSO fast path on
// and real tool dispatch, all decomposing ONE shared research goal, WALL-CLOCKED — swept
// from N=1 to N=1024. It is the measured capstone to `cmd/fanbench`, whose headline
// (the 72.8× parallel speedup) is a MODELED projection over a synthetic call stream.
//
// What fanrun measures (every number is a wall-clock, a real kernel counter, or exact
// geometry — there are NO modeled fields):
//   - agents_wall_serial_ms — the real wall-clock of running the N sub-agents end-to-end.
//     SERIAL by construction: the kernel's fast-path cache world-version is process-global,
//     so the N sub-agents share one world epoch (which is exactly what makes cross-agent
//     dedup real) and run one after another. agents_per_sec_serial is N / Σt — NOT a
//     parallel rate. fanrun does not claim 1024-way parallelism.
//   - cross_hits — real vDSO tier-2 hits the siblings get on sub-agent 0's warmed shared
//     reads, reported as the sibling-only delta over the N=1 single-agent baseline (the
//     same shared−isolated discipline as cmd/fanbench's cross_uplift).
//   - prefix_tokens_elided = (N−1)·P — exact geometry: the master-goal prefill the kernel
//     does once and clones, never redoes (bit-identity proven by
//     cmd/fanbench.TestPrefixReuseFanoutWitness). With -model-dir it is also wall-clocked
//     as a reuse-vs-no-reuse prefill race (the cmd/fleetserve methodology).
//
// The sub-agents are driven by a deterministic, offline research planner (no model call, no
// network), so a run is byte-reproducible in its counter+geometry projection and needs no
// GPU, weights, or API key. The read-only research role is faithful to the
// orchestrator-worker pattern (sub-agents gather; the lead folds) and is what keeps the
// shared cache warm — a sub-agent that wrote would bump the world and strand the fleet.
//
// Usage:
//
//	fanrun [--profile research|write-heavy|no-share]
//	       [--agents 1,4,...,1024] | [--agent-max 1024 --grid log|full|canonical]
//	       [--sub-turns 8] [--prefix 2048] [--trials 1] [--seed N]
//	       [--reps 3 --model-dir /path/to/model --quant]   (optional prefill-elision wall-clock)
//	       [--out fanrun.json]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/intlist"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func main() {
	fs := flag.NewFlagSet("fanrun", flag.ExitOnError)
	profileName := fs.String("profile", "research", "sharing regime: research | write-heavy | no-share")
	agentsArg := fs.String("agents", "", "explicit fan-out grid, comma-separated (overrides --agent-max/--grid)")
	agentMax := fs.Int("agent-max", 1024, "max fan-out width N (generated grid)")
	gridKind := fs.String("grid", "log", "generated grid: log (powers-of-two ladder to max) | full (1..max) | canonical (1,100,500,1000)")
	subTurns := fs.Int("sub-turns", 8, "RunArm maxTurns cap per sub-agent (the research gather needs ~6)")
	prefix := fs.Int("prefix", 2048, "master-goal shared prefix tokens P (the reuse lever / (N-1)*P geometry)")
	trials := fs.Int("trials", 1, "determinism witness: >=2 re-runs each wave and asserts identical counts")
	reps := fs.Int("reps", 0, "prefill-timing reps (best-of-min); 0 or no --model-dir => skip the wall-clock half")
	seed := fs.Int64("seed", 0x5EED_F1EE, "root seed")
	modelDir := fs.String("model-dir", "", "optional small CPU model dir for the prefill-elision wall-clock; empty => geometry-only")
	quant := fs.Bool("quant", true, "Q8 lane for the prefill-timing model (fleetserve parity)")
	out := fs.String("out", "fanrun.json", "JSON artifact path")
	_ = fs.Parse(os.Args[1:])

	prof, ok := profileByName(*profileName)
	if !ok {
		fmt.Fprintf(os.Stderr, "fanrun: unknown profile %q (research|write-heavy|no-share)\n", *profileName)
		os.Exit(2)
	}

	grid := buildAgentGrid(*agentsArg, *agentMax, *gridKind)
	opts := bench.FanrunOptions{
		Profile: prof, Grid: grid, SubTurns: *subTurns, Prefix: *prefix,
		Trials: *trials, Reps: *reps, Seed: *seed, ModelDir: *modelDir, Quant: *quant,
	}

	fmt.Fprintf(os.Stderr, "fanrun: profile=%s grid=%v sub-turns=%d prefix=%d trials=%d => %d real-agent waves (SERIAL; cross-agent dedup is the measured win, NOT parallel speedup)\n",
		prof.Name, grid, *subTurns, *prefix, *trials, len(grid))

	t0 := time.Now()
	rep := bench.RunFanoutLive(context.Background(), opts)
	for _, c := range rep.Cells {
		fmt.Fprintf(os.Stderr, "  N=%-4d wall=%8.1fms agents/s(serial)=%6.1f cross_hits=%-5d fills_flat elided=%d\n",
			c.Agents, c.AgentsWallSerialMs, c.AgentsPerSecSerial, c.CrossHits, c.PrefixTokensElided)
	}

	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "fanrun:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, append(b, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "fanrun:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "fanrun: wrote %s (%d cells) in %s\n", *out, len(rep.Cells), time.Since(t0).Round(time.Millisecond))
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

// buildAgentGrid mirrors cmd/fanbench: the explicit list if given, else a generated grid.
// "log" is a powers-of-two ladder (1,2,4,...,max) always including 1 and max — the right
// spacing for a fan-out curve over three orders of magnitude; "full" is 1..max;
// "canonical" is the D-001 acceptance ladder.
func buildAgentGrid(explicit string, max int, kind string) []int {
	if strings.TrimSpace(explicit) != "" {
		return intlist.Parse(explicit)
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
