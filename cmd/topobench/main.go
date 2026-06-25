// Command topobench runs the FLEET-TOPOLOGY genome search (issue #541) — the orthogonal
// STRUCTURE-search axis to the policy-genome search (#503). Where fanbench SWEEPS the
// fan-out width N as an input, topobench SEARCHES the fleet graph itself: it scores a grid
// of (width × lanes) topology genomes against the SAME model-free fan-out replay oracle and
// reports the Pareto frontier of credited-savings-vs-arbiter-collision, gated by the SAME
// divergence frontier as the rest of the replay-as-fitness substrate. The search spends ZERO
// model calls; a genome whose advantage lives past the corpus frontier width is REFUSED
// credit and flagged for live re-validation (a flag, never an executed model run).
//
// The two honest savings halves are the MEASURED ones fanout.go already computes — the exact
// prefix-reuse geometry (N−1)·prefix_tokens and the real cross-agent tier-2 dedup turns — and
// the cost axis is the EXACT dos_arbitrate serialization rule a lane assignment induces
// (Σ C(size_j,2) over the lanes). NO resolve-rate is in the objective.
//
// Usage:
//
//	topobench [--profile research|write-heavy|no-share]
//	          [--frontier-width 16]            (Wmax: the widest run the corpus witnessed)
//	          [--baseline-width 4 --baseline-lanes 1 --baseline-sub-turns 4]
//	          [--widths 1,2,4,8,16] [--lanes 1,2,4,8,16]   (the searched genome grid)
//	          [--named-topology linear|star --named-width N] (score one declared shape)
//	          [--trials 6] [--seed N] [--topk 0]
//	          [--prefix 2048]                  (master-goal shared prefix tokens P)
//	          [--out topo.json] [--csv topo.csv]
//
// The world version is process-global, so the search is SERIAL within one process (the same
// constraint RunFanoutSweep carries); there is no cross-process sharding here because a
// topology search is a small grid, not a 1000-wide sweep.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agenttopo"
	"github.com/anthony-chaudhary/fak/internal/comm"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func main() {
	fs := flag.NewFlagSet("topobench", flag.ExitOnError)
	profileName := fs.String("profile", "research", "workload profile: research | write-heavy | no-share")
	frontierWidth := fs.Int("frontier-width", 16, "Wmax: the widest fan-out the frozen corpus recorded (the divergence gate)")
	baseWidth := fs.Int("baseline-width", 4, "hand-frozen baseline fan-out width (the bar the search beats)")
	baseLanes := fs.Int("baseline-lanes", 1, "hand-frozen baseline lane count (single-lane = the dos.toml partition)")
	baseSubTurns := fs.Int("baseline-sub-turns", 4, "orchestrator->worker depth (turns per worker)")
	namedTopology := fs.String("named-topology", "", "optional declared topology to score: linear | star")
	namedWidth := fs.Int("named-width", 0, "node count for --named-topology (default: baseline-width)")
	widthsArg := fs.String("widths", "", "searched fan-out widths, comma-separated (default: powers-of-two ladder to frontier-width)")
	lanesArg := fs.String("lanes", "", "searched lane partitions, comma-separated (default: same ladder as widths)")
	trials := fs.Int("trials", 6, "seeded trials per cell handed to the measured fan-out replay")
	seed := fs.Int64("seed", 1337, "root seed (the whole search and its frontier are reproducible)")
	topK := fs.Int("topk", 0, "how many frontier genomes to flag for live re-validation (0 = all)")
	prefix := fs.Int("prefix", 2048, "master-goal shared prefix tokens P (the dominant measured savings lever)")
	out := fs.String("out", "topo.json", "JSON artifact path")
	csv := fs.String("csv", "topo.csv", "CSV artifact path (one row per searched genome; empty = skip)")
	_ = fs.Parse(os.Args[1:])

	p, ok := profileByName(*profileName)
	if !ok {
		fmt.Fprintf(os.Stderr, "topobench: unknown profile %q (research|write-heavy|no-share)\n", *profileName)
		os.Exit(2)
	}
	if *frontierWidth < 1 {
		fmt.Fprintf(os.Stderr, "topobench: --frontier-width must be >=1, got %d\n", *frontierWidth)
		os.Exit(2)
	}

	widthGrid := parseGrid(*widthsArg)
	if len(widthGrid) == 0 {
		widthGrid = ladder(*frontierWidth)
	}
	laneGrid := parseGrid(*lanesArg)
	if len(laneGrid) == 0 {
		laneGrid = ladder(*frontierWidth)
	}

	cm := turnbench.DefaultFanoutCostModel()
	cm.PrefixTokens = *prefix

	cfg := turnbench.TopologySearchConfig{
		Profile:       p,
		FrontierWidth: *frontierWidth,
		Baseline:      turnbench.TopologyGenome{Width: *baseWidth, SubTurns: *baseSubTurns, Lanes: *baseLanes},
		WidthGrid:     widthGrid,
		LaneGrid:      laneGrid,
		Trials:        *trials,
		Seed:          *seed,
		TopK:          *topK,
	}

	fmt.Fprintf(os.Stderr, "topobench: profile=%s frontier=%d widths=%v lanes=%v baseline=w%d/l%d/t%d trials=%d => %d genomes\n",
		p.Name, *frontierWidth, widthGrid, laneGrid, *baseWidth, *baseLanes, *baseSubTurns, *trials, len(widthGrid)*len(laneGrid))

	t0 := time.Now()
	rep, err := turnbench.RunTopologySearch(context.Background(), cfg, cm)
	if err != nil {
		fmt.Fprintln(os.Stderr, "topobench:", err)
		os.Exit(1)
	}
	if strings.TrimSpace(*namedTopology) != "" {
		nw := *namedWidth
		if nw <= 0 {
			nw = *baseWidth
		}
		named, err := scoreNamedTopology(context.Background(), *namedTopology, nw, *baseLanes, *baseSubTurns, *frontierWidth, *trials, *seed, p, cm)
		if err != nil {
			fmt.Fprintln(os.Stderr, "topobench:", err)
			os.Exit(2)
		}
		rep.NamedTopology = &named
	}

	if err := os.WriteFile(*out, rep.JSON(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "topobench:", err)
		os.Exit(1)
	}
	if strings.TrimSpace(*csv) != "" {
		if err := os.WriteFile(*csv, rep.CSV(), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "topobench:", err)
			os.Exit(1)
		}
	}

	b := rep.Best
	fmt.Fprintf(os.Stderr,
		"topobench: best=w%d/l%d credited_savings=%d (baseline %d) arbiter_collision=%d (baseline %d) frontier=%d genomes model_calls=%d in %s\n",
		b.Genome.Width, b.Genome.Lanes, b.Fitness.CreditedSavingsTokens, rep.Baseline.Fitness.CreditedSavingsTokens,
		b.Fitness.ArbiterCollisionCost, rep.Baseline.Fitness.ArbiterCollisionCost,
		len(rep.Frontier), rep.ModelCallsSpent, time.Since(t0).Round(time.Millisecond))
	if len(rep.FlaggedForRevalidation) > 0 {
		fmt.Fprintf(os.Stderr, "topobench: %d frontier genome(s) flagged for LIVE re-validation (past the corpus frontier): %v\n",
			len(rep.FlaggedForRevalidation), rep.FlaggedForRevalidation)
	}
	fmt.Fprintf(os.Stderr, "topobench: wrote %s", *out)
	if strings.TrimSpace(*csv) != "" {
		fmt.Fprintf(os.Stderr, " and %s", *csv)
	}
	fmt.Fprintln(os.Stderr)
}

func scoreNamedTopology(ctx context.Context, shape string, width, lanes, subTurns, frontierWidth, trials int, seed int64, p turnbench.FanoutProfile, cm turnbench.FanoutCostModel) (turnbench.TopologyCandidate, error) {
	if width < 1 {
		return turnbench.TopologyCandidate{}, fmt.Errorf("--named-width must be >=1, got %d", width)
	}
	if lanes < 1 {
		lanes = 1
	}
	g, err := comm.New("topobench-"+shape, "", namedMembers(width, lanes))
	if err != nil {
		return turnbench.TopologyCandidate{}, err
	}

	var topo *agenttopo.Topology
	switch strings.ToLower(strings.TrimSpace(shape)) {
	case "linear", "line":
		topo, err = agenttopo.Linear("linear", g)
	case "star":
		root, _ := g.Member(0)
		topo, err = agenttopo.Star("star", g, root.ID)
	default:
		return turnbench.TopologyCandidate{}, fmt.Errorf("unknown --named-topology %q (linear|star)", shape)
	}
	if err != nil {
		return turnbench.TopologyCandidate{}, err
	}

	cand := turnbench.ScoreTopology(ctx, p, turnbench.TopologyGenome{
		Width:    topo.Size(),
		SubTurns: subTurns,
		Lanes:    topo.LaneCount(),
	}, frontierWidth, trials, seed, cm)
	cand.Name = "named-" + topo.Name()
	cand.Summary["declared_shape"] = topo.Name()
	cand.Summary["declared_edges"] = strconv.Itoa(len(topo.Edges()))
	cand.Summary["declared_nodes"] = strconv.Itoa(topo.Size())
	return cand, nil
}

func namedMembers(width, lanes int) []comm.Member {
	if lanes < 1 {
		lanes = 1
	}
	out := make([]comm.Member, width)
	for i := range out {
		out[i] = comm.Member{
			ID:   fmt.Sprintf("node-%04d", i),
			Lane: fmt.Sprintf("lane-%03d", i%lanes),
		}
	}
	return out
}

// profileByName resolves a fan-out workload profile name to its FanoutProfile (the same
// named profiles fanbench exposes).
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

// ladder returns the powers-of-two ladder 1,2,4,...,max that always includes 1 and max — the
// natural genome grid for a topology search whose savings grow with width: it spans the range
// with a handful of points and always includes the corpus frontier width itself.
func ladder(max int) []int {
	if max < 1 {
		max = 1
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

// parseGrid parses a comma-separated positive-int grid, exiting with a clear message on a bad
// or non-positive value (the fanbench parseInts discipline).
func parseGrid(s string) []int {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			fmt.Fprintf(os.Stderr, "topobench: bad int %q in grid\n", part)
			os.Exit(2)
		}
		if n < 1 {
			fmt.Fprintf(os.Stderr, "topobench: grid values must be >=1, got %d\n", n)
			os.Exit(2)
		}
		out = append(out, n)
	}
	return out
}
