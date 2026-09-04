package main

// scorecardpane.go — the native-fak verbs over internal/scorecardpane: the
// portfolio control-pane fold (`fak scorecard control-pane`) and the repo-hygiene
// scorecard fold (`fak repo-hygiene-scorecard`), ports of
// tools/scorecard_control_pane.py and tools/repo_hygiene_scorecard.py. The Python
// scripts remain as compatibility shims until the baseline can shrink (issue #1449);
// these verbs are the typed, one-process-startup native surface.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scdiff"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
	"github.com/anthony-chaudhary/fak/internal/scorecardportfolio"
)

// cmdScorecardPane dispatches `fak scorecard <sub>`. The control-pane fold is the
// portfolio debt ratchet; it reads each per-scorecard payload, folds total_debt +
// grade_debt, and emits the ratchet verdict.
func cmdScorecardPane(argv []string) {
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "usage: fak scorecard control-pane [--json|--check|--pin] | portfolio-discovery [--workspace PATH] | qa-dogfood [--json|--existing-json PATH] | debt-page [--write-doc|--check-doc|--block]")
		os.Exit(2)
	}
	switch argv[0] {
	case "control-pane":
		os.Exit(runScorecardControlPane(os.Stdout, os.Stderr, argv[1:]))
	case "portfolio-discovery":
		os.Exit(runScorecardPortfolioDiscovery(os.Stdout, os.Stderr, argv[1:]))
	case "qa-dogfood":
		os.Exit(runScorecardQADogfood(os.Stdout, os.Stderr, argv[1:]))
	case "debt-page":
		os.Exit(runScoreboardDebtPage(os.Stdout, os.Stderr, argv[1:]))
	default:
		fmt.Fprintf(os.Stderr, "fak scorecard: unknown subcommand %q\n", argv[0])
		os.Exit(2)
	}
}

func runScorecardPortfolioDiscovery(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak scorecard portfolio-discovery", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	_ = fs.Bool("json", true, "emit schema-versioned JSON (always enabled)")
	if !parseFlags(fs, argv) {
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	report, err := scorecardportfolio.Audit(scorecardportfolio.DefaultConfig(root))
	if err != nil {
		fmt.Fprintf(stderr, "fak scorecard portfolio-discovery: %v\n", err)
		return 1
	}
	if err := scorecardportfolio.WriteJSON(stdout, report); err != nil {
		fmt.Fprintf(stderr, "fak scorecard portfolio-discovery: %v\n", err)
		return 1
	}
	return 0
}

func runScorecardControlPane(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak scorecard control-pane", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "scorecard")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	pin := fs.Bool("pin", false, "pin the current debt as the baseline ("+scorecardpane.BaselineRel+")")
	check := fs.Bool("check", false, "CI ratchet gate: exit non-zero only if debt regressed above baseline")
	baselineFlag := fs.String("baseline", "", "baseline JSON path (default: "+scorecardpane.BaselineRel+")")
	timeoutSec := fs.Int("timeout", 120, "per-scorecard timeout seconds")
	since := fs.String("since", "", "shift-left incremental fold: measure only cards whose corpus changed vs this git ref; carry the rest from the pinned baseline (requires a baseline; cannot combine with --pin/--post)")
	post := fs.Bool("post", false, "opt-in: post the freshly-regenerated portfolio number to #scoreboard (also FAK_SCOREBOARD_AUTOPOST=1); off by default, deduped so an unchanged rerun is silent")
	if !parseFlags(fs, argv) {
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	baselinePath := *baselineFlag
	if baselinePath == "" {
		baselinePath = filepath.Join(root, filepath.FromSlash(scorecardpane.BaselineRel))
	}

	baseline := scorecardpane.LoadBaseline(baselinePath)

	var payload scorecardpane.Payload
	if *since != "" {
		// Shift-left incremental fold. It reproduces carried cards from the baseline,
		// so it is an approximate read, not a full measurement — it may not seed a new
		// floor (--pin) nor be published (--post); both demand a full run.
		if *pin || *post {
			fmt.Fprintln(stderr, "fak scorecard control-pane: --since is an incremental read and cannot be combined with --pin or --post (both require a full measurement)")
			return 2
		}
		if baseline == nil {
			fmt.Fprintf(stderr, "fak scorecard control-pane: --since requires a pinned baseline (%s); run `fak scorecard control-pane --pin` first\n", baselinePath)
			return 2
		}
		changed, err := scdiff.ChangedPaths(root, *since)
		if err != nil {
			fmt.Fprintf(stderr, "fak scorecard control-pane: --since %s: %v\n", *since, err)
			return 2
		}
		metrics, info := scorecardpane.CollectSince(root, "", time.Duration(*timeoutSec)*time.Second, *since, changed, baseline)
		payload = scorecardpane.Fold(metrics, baseline, root, scorecardpane.HeadCommitShort(root))
		payload.Incremental = &info
		fmt.Fprintf(stderr, "fak scorecard control-pane: incremental vs %s — measured %d card(s), carried %d from baseline @%s (%d changed file(s))\n",
			*since, info.Measured, info.Carried, info.BaselineCommit, info.ChangedFiles)
	} else {
		metrics := scorecardpane.Collect(root, "", time.Duration(*timeoutSec)*time.Second)
		payload = scorecardpane.Fold(metrics, baseline, root, scorecardpane.HeadCommitShort(root))

		// Local producer side of #scoreboard (#998): when the operator opts in, publish
		// the freshly-regenerated portfolio number the moment it is folded. Notices go to
		// stderr so a --json/--check stdout stays a clean machine payload.
		if code := autopostControlPane(stderr, root, payload, *post); code != 0 {
			return code
		}
	}

	if *pin {
		doc := scorecardpane.BaselineDoc(payload)
		if err := os.MkdirAll(filepath.Dir(baselinePath), 0o755); err != nil {
			fmt.Fprintf(stderr, "fak scorecard control-pane: mkdir baseline dir: %v\n", err)
			return 1
		}
		f, err := os.Create(baselinePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak scorecard control-pane: write baseline: %v\n", err)
			return 1
		}
		if err := writeIndentedJSONNoEscape(f, doc); err != nil {
			_ = f.Close()
			fmt.Fprintf(stderr, "fak scorecard control-pane: encode baseline: %v\n", err)
			return 1
		}
		_ = f.Close()
		if !*asJSON {
			fmt.Fprintf(stdout, "pinned baseline @%s total_debt=%d -> %s\n", doc.Commit, doc.TotalDebt, baselinePath)
		}
	}

	if *check {
		code, message := scorecardpane.CheckGate(payload)
		if *asJSON {
			gated := payload
			gated.OK = code == 0
			gated.Verdict = "OK"
			if code != 0 {
				gated.Verdict = "ACTION"
			}
			ec := code
			gated.GateExit = &ec
			gated.GateMessage = message
			_ = writeIndentedJSONNoEscape(stdout, gated)
		} else {
			fmt.Fprintln(stdout, message)
		}
		return code
	}

	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak scorecard control-pane: encode json: %v\n", err)
			return 1
		}
	} else if !*pin {
		fmt.Fprint(stdout, scorecardpane.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}

// cmdRepoHygieneScorecard runs the repo-hygiene fold over the git-tracked tree.
func cmdRepoHygieneScorecard(argv []string) {
	os.Exit(runRepoHygieneScorecard(os.Stdout, os.Stderr, argv))
}

func runRepoHygieneScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak repo-hygiene-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the snapshot markdown body")
	stamp := fs.String("stamp", "", "date stamp for the markdown header")
	comparePath := fs.String("compare", "", "print the hygiene-debt delta vs a prior baseline JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	payload := scorecardpane.CollectHygiene(root)

	switch {
	case *comparePath != "":
		baseline, ok := loadHygieneCompareBase(stderr, *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, scorecardpane.RenderHygieneCompare(baseline, payload))
	case *asJSON:
		if err := writeIndentedJSONNoEscape(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak repo-hygiene-scorecard: encode json: %v\n", err)
			return 1
		}
	case *asMarkdown:
		fmt.Fprint(stdout, scorecardpane.RenderHygieneMarkdown(payload, *stamp))
	default:
		fmt.Fprint(stdout, scorecardpane.RenderHygiene(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}

func loadHygieneCompareBase(stderr io.Writer, path string) (scorecardpane.HygienePayload, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak repo-hygiene-scorecard: cannot read baseline %s: %v\n", path, err)
		return scorecardpane.HygienePayload{}, false
	}
	base, err := scorecardpane.ParseHygienePayload(b)
	if err != nil {
		fmt.Fprintf(stderr, "fak repo-hygiene-scorecard: parse baseline %s: %v\n", path, err)
		return scorecardpane.HygienePayload{}, false
	}
	return base, true
}
