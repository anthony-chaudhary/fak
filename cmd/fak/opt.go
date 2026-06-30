package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/opttarget"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func cmdOpt(argv []string) { os.Exit(runOpt(os.Stdout, os.Stderr, argv)) }

func runOpt(stdout, stderr io.Writer, argv []string) int {
	cmd := "discover"
	if len(argv) > 0 {
		cmd = argv[0]
		argv = argv[1:]
	}
	switch cmd {
	case "discover":
		return runOptDiscover(stdout, stderr, argv)
	case "run":
		return runOptRun(stdout, stderr, argv)
	case "-h", "--help", "help":
		optUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak opt: unknown command %q\n", cmd)
		optUsage(stderr)
		return 2
	}
}

func runOptDiscover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak opt discover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit the inventory as JSON")
	check := fs.String("check", "", "coverage ratchet: comma-separated target names that must still be present")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak opt discover: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	targets, err := opttarget.DiscoverDir(root)
	if err != nil {
		// A malformed annotation is a WARNING: DiscoverDir still returns the
		// well-formed targets, so we report the defect and continue.
		fmt.Fprintf(stderr, "fak opt discover: %v\n", err)
	}

	if *check != "" {
		var required []string
		for _, name := range strings.Split(*check, ",") {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			required = append(required, name)
		}
		if cerr := opttarget.Check(targets, required); cerr != nil {
			fmt.Fprintln(stdout, cerr)
			return 1
		}
		fmt.Fprintf(stdout, "fak opt discover --check: OK (%d targets, all %d required present)\n",
			len(targets), len(required))
		return 0
	}

	if *asJSON {
		b, merr := opttarget.MarshalInventory(targets)
		if merr != nil {
			fmt.Fprintf(stderr, "fak opt discover: marshal inventory: %v\n", merr)
			return 1
		}
		stdout.Write(b)
		return 0
	}

	fmt.Fprintf(stdout, "fak opt discover: %d target(s)\n", len(targets))
	for _, t := range targets {
		fmt.Fprintf(stdout, "  - %s  metric=%s measurer=%s site=%s:%s\n",
			t.Name, t.Metric, t.Measurer, t.Site.Path, t.Site.Const)
	}
	return 0
}

// runOptRun drives a DECLARED target end-to-end: it discovers the target by name,
// resolves its measurer= key through the registry, Compiles it into a rsiloop.Harness,
// and runs the closed RSI loop with the unchanged non-forgeable keep-bit (rsiloop.Run
// + shipgate.Evaluate). A declared target therefore earns the same keep/revert gate
// the hand-wired worktree demo does — the fuser's whole point. Exit codes mirror
// cmd/rsiloop: 0 normal, 1 error, 2 usage, 3 ESCALATE (breaker tripped — hand to a human).
func runOptRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak opt run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "repo root the worktree forks from (default: repo root)")
	journalPath := fs.String("journal", "-", "append-only JSONL journal path ('-' = stdout)")
	k := fs.Int("k", 3, "escalation breaker: stop after K consecutive non-keeps")
	maxCycles := fs.Int("max", 0, "cap on candidates tried (0 = all)")
	asJSON := fs.Bool("json", false, "emit the run Result as JSON (journal routed off stdout)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintf(stderr, "fak opt run: want exactly one target NAME (got %d positional args)\n", fs.NArg())
		return 2
	}
	name := fs.Arg(0)
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	targets, derr := opttarget.DiscoverDir(root)
	if derr != nil {
		// A malformed annotation elsewhere is a warning; the named target may still
		// be well-formed and present, so report and continue.
		fmt.Fprintf(stderr, "fak opt run: %v\n", derr)
	}
	var target *opttarget.OptTarget
	for i := range targets {
		if targets[i].Name == name {
			target = &targets[i]
			break
		}
	}
	if target == nil {
		names := make([]string, 0, len(targets))
		for _, t := range targets {
			names = append(names, t.Name)
		}
		fmt.Fprintf(stderr, "fak opt run: no discovered target named %q\n", name)
		fmt.Fprintf(stderr, "  available: %s\n", strings.Join(names, ", "))
		return 2
	}

	measurer, merr := opttarget.Resolve(*target, root)
	if merr != nil {
		fmt.Fprintf(stderr, "fak opt run: %v\n", merr)
		return 1
	}
	h, cerr := opttarget.Compile(*target, measurer)
	if cerr != nil {
		fmt.Fprintf(stderr, "fak opt run: compile %q: %v\n", name, cerr)
		return 1
	}

	// Keep the JSON envelope clean: route the JSONL journal off stdout when the user
	// asked for a JSON Result and did not pick an explicit journal file.
	if *asJSON && *journalPath == "-" {
		*journalPath = os.DevNull
	}
	j, jerr := rsiloop.NewJournal(*journalPath)
	if jerr != nil {
		fmt.Fprintf(stderr, "fak opt run: journal: %v\n", jerr)
		return 1
	}
	defer j.Close()

	res, rerr := rsiloop.Run(h, j, *k, *maxCycles)
	if rerr != nil {
		fmt.Fprintf(stderr, "fak opt run: %v\n", rerr)
		return 1
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, res, "fak opt run")
	}

	base := res.FinalBaseline
	if len(res.Rows) > 0 {
		base = res.Rows[0].Baseline
	}
	fmt.Fprintf(stdout, "opt run %q: baseline %s@%s = %.6f\n", name, h.MetricName, res.BaselineRef, base)
	for _, r := range res.Rows {
		cand := fmt.Sprintf("%.6f", r.Candidate_)
		if !r.Measured {
			cand = "(not measured)"
		}
		fmt.Fprintf(stdout, "  cycle %d  %-22s base=%.6f cand=%s improved=%v suite=%v truth=%v -> %s (kept=%v, breaker=%d)\n",
			r.Cycle, r.Candidate, r.Baseline, cand, r.Improved, r.SuiteGreen, r.TruthClean, r.Decision, r.Kept, r.BreakerCount)
	}
	fmt.Fprintf(stdout, "SUMMARY target=%s cycles=%d kept=%d final=%s final_baseline=%.6f escalated=%v\n",
		name, res.Cycles, res.Kept, res.Final.String(), res.FinalBaseline, res.Escalated)
	if res.Escalated {
		return 3 // breaker tripped — hand to a human
	}
	return 0
}

func optUsage(w io.Writer) {
	fmt.Fprint(w, `fak opt - declare and drive the repo's annotated optimization targets (RSI fuser)

  fak opt discover [--workspace DIR] [--json] [--check name,...]
  fak opt run NAME [--workspace DIR] [--journal FILE] [--k N] [--max N] [--json]

  discover walks the workspace for // fak:opttarget annotated consts and reports
  the OptTargets they declare. --json emits the stable JSON inventory; --check
  NAMES,... is the coverage ratchet (exit 1 if any named target is gone).

  run resolves a discovered target's measurer, compiles it into a harness, and
  drives the closed RSI loop (propose/measure/keep-or-revert) with the unchanged
  non-forgeable keep-bit. Exit 3 = ESCALATE (breaker tripped after K non-keeps).
`)
}
