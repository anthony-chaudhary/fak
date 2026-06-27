package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/benchpost"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// cmdBenchPost / cmdBenchRequest post fak BENCH-CHANNEL rollups. They are reached as
// `fak bench post` / `fak bench request` (dispatched from cmdBench in main.go) and are
// the outbound bench-channel surface — the twin of `fak scoreboard post`.
//
//	fak bench post --rollup latest                    # latest catalog runs
//	fak bench post --rollup regression                # tok/s drops vs bench_baseline.json
//	fak bench request --now 20260627T143000Z          # next-test-per-machine run-request
//	fak bench request --plan-json plan.json --dry-run # fold a pre-rendered bench_plan payload
//
// A run-request is a POST, not a dispatch: there is no inbound listener, so the message
// asks the bench-nodes to run next; fak does not execute it.
//
// All commands default to the FAK_BENCH_* workspace (token falls back to the scoreboard
// token; channel is never a hard-coded default) and honor --dry-run (render + print, no
// network), matching the scoreboard / *_signal "safe by default" idiom.

// runBenchPost handles `fak bench post`.
func runBenchPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rollup := fs.String("rollup", "latest", "which rollup: latest | regression")
	catalog := fs.String("catalog", "experiments/benchmark/catalog.json", "catalog.json path")
	baseline := fs.String("baseline", "tools/bench_baseline.json", "pinned baseline path (regression rollup)")
	n := fs.Int("n", 8, "latest rollup: number of recent runs to show")
	minDropPct := fs.Float64("min-drop-pct", 15.0, "regression: relative drop %% to flag")
	minAbs := fs.Float64("min-abs", 1.0, "regression: absolute tok/s drop to flag")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_BENCH_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_BENCH_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	cat, err := benchpost.LoadCatalog(*catalog)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench post: load catalog: %v\n", err)
		return 2
	}

	var post benchpost.Post
	switch *rollup {
	case "latest":
		post = benchpost.RollupFromCatalog(cat, *n)
	case "regression":
		bl, err := benchpost.LoadBaseline(*baseline)
		if err != nil {
			fmt.Fprintf(stderr, "fak bench post: load baseline: %v\n", err)
			return 2
		}
		post = benchpost.RegressionFromCatalogVsBaseline(cat, bl, *minDropPct, *minAbs)
	default:
		fmt.Fprintf(stderr, "fak bench post: unknown --rollup %q (want: latest | regression)\n", *rollup)
		return 2
	}
	post.Source = resolveBenchSource(*source)

	return emitBenchPost(stdout, stderr, post, *channel, *token, *dryRun)
}

// runBenchRequest handles `fak bench request` — the next-test-per-machine run-request.
// It folds a bench_plan.py --json payload: either a pre-rendered one (--plan-json, the
// pure path the test exercises) or one produced on the fly by invoking the planner
// (--now <stamp>, the convenience path).
func runBenchRequest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench request", flag.ContinueOnError)
	fs.SetOutput(stderr)
	planJSON := fs.String("plan-json", "", "fold a pre-rendered `bench_plan.py --json` payload from this file (- for stdin)")
	now := fs.String("now", "", "invoke tools/bench_plan.py --now <stamp> --json (e.g. 20260627T143000Z); ignored if --plan-json set")
	top := fs.Int("top", 0, "cap the per-machine list at the top N (0 = all)")
	python := fs.String("python", "python", "python interpreter for the planner (--now path)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_BENCH_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_BENCH_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	plan, err := loadRequestPlan(*planJSON, *now, *python, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench request: %v\n", err)
		return 2
	}
	post := benchpost.RequestFromPlan(plan, *top)
	post.Source = resolveBenchSource(*source)

	return emitBenchPost(stdout, stderr, post, *channel, *token, *dryRun)
}

// loadRequestPlan resolves the bench_plan payload from --plan-json (a file or stdin) or
// by invoking the planner with --now. Exactly one source must be given.
func loadRequestPlan(planJSON, now, python string, stderr io.Writer) (*benchpost.Plan, error) {
	switch {
	case planJSON != "":
		if planJSON == "-" {
			raw, err := io.ReadAll(os.Stdin)
			if err != nil {
				return nil, err
			}
			return benchpost.ParsePlan(raw)
		}
		return benchpost.LoadPlan(planJSON)
	case now != "":
		cmd := exec.CommandContext(ctx(), python, "tools/bench_plan.py", "--now", now, "--json")
		cmd.Stderr = stderr
		out, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("run bench_plan.py: %w", err)
		}
		return benchpost.ParsePlan(out)
	default:
		return nil, fmt.Errorf("nothing to fold: pass --plan-json <file> or --now <stamp>")
	}
}

// emitBenchPost is the shared dry-run / post tail for both bench subcommands.
func emitBenchPost(stdout, stderr io.Writer, post benchpost.Post, channel, token string, dryRun bool) int {
	if dryRun {
		fmt.Fprintln(stdout, post.Text())
		return 0
	}
	ch := channel
	if ch == "" {
		ch = benchpost.ResolveChannel()
	}
	if ch == "" {
		fmt.Fprintln(stderr, "fak bench post: no channel: pass --channel, set FAK_BENCH_CHANNEL, or add it to .env.slack.local")
		return 2
	}
	tok := token
	if tok == "" {
		tok = benchpost.ResolveToken()
	}
	client, err := scoreboard.NewClient(tok)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench post: %v\n", err)
		return 2
	}
	ts, err := client.Post(ctx(), ch, post.Text(), post.Blocks())
	if err != nil {
		fmt.Fprintf(stderr, "fak bench post: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "posted to %s ts=%s\n", ch, ts)
	return 0
}

// resolveBenchSource picks the post source: the flag, else the shared defaultSource
// ($FAK_SCOREBOARD_SOURCE or hostname).
func resolveBenchSource(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return defaultSource()
}
