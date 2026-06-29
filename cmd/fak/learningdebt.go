package main

// fak learning-debt-dispatch -- the learning-scorecard to backlog bridge.
// It reads a tools/learning_scorecard.py --json payload, folds each HARD defect
// into a stable issue key, and plans at most --cap new GitHub issues. Safe by
// default: dry-run mutates nothing, including the seen-cache; --live is required
// to call gh and record successfully filed keys.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/learningdebt"
)

func cmdLearningDebtDispatch(argv []string) {
	os.Exit(runLearningDebtDispatch(os.Stdout, os.Stderr, argv))
}

func runLearningDebtDispatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("learning-debt-dispatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root for the default cache (default: repo root)")
	scorecard := fs.String("scorecard", "", "tools/learning_scorecard.py --json payload")
	cachePath := fs.String("cache", "", "seen-cache path (default: <workspace>/.fak/learning-debt-dispatch/seen.json)")
	capN := fs.Int("cap", 3, "maximum new issues to file in one run")
	repo := fs.String("repo", "", "owner/repo for gh; default is current repo")
	limit := fs.Int("limit", 300, "existing issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture/list of existing gh issues for dry-run tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to dedup against existing issue bodies")
	live := fs.Bool("live", false, "create GitHub issues with gh and update the seen-cache")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to newly-created issues; repeatable")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *capN < 0 {
		fmt.Fprintln(stderr, "fak learning-debt-dispatch: --cap must be >= 0")
		return 2
	}
	if *scorecard == "" && fs.NArg() == 1 {
		*scorecard = fs.Arg(0)
	} else if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak learning-debt-dispatch: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *scorecard == "" {
		fmt.Fprintln(stderr, "fak learning-debt-dispatch: --scorecard is required")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	if *cachePath == "" {
		*cachePath = filepath.Join(root, learningdebt.DefaultCacheRel)
	}
	scorecardPath, err := filepath.Abs(*scorecard)
	if err != nil {
		fmt.Fprintf(stderr, "learning-debt-dispatch: %v\n", err)
		return 2
	}
	cacheAbs, err := filepath.Abs(*cachePath)
	if err != nil {
		fmt.Fprintf(stderr, "learning-debt-dispatch: %v\n", err)
		return 2
	}

	payload, err := learningdebt.LoadPayload(scorecardPath)
	if err != nil {
		fmt.Fprintf(stderr, "learning-debt-dispatch: %v\n", err)
		return 2
	}
	seen, err := learningdebt.LoadSeen(cacheAbs)
	if err != nil {
		fmt.Fprintf(stderr, "learning-debt-dispatch: load seen-cache: %v\n", err)
		return 2
	}

	var existing []learningdebt.Issue
	switch {
	case *existingJSON != "":
		b, err := os.ReadFile(*existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "learning-debt-dispatch: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "learning-debt-dispatch: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		existing, err = learningdebt.FetchExistingIssues(*repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "learning-debt-dispatch: %v\n", err)
			return 2
		}
	}

	defects := learningdebt.ExtractDefects(payload)
	plan, stats := learningdebt.BuildPlan(defects, seen, existing, *capN, scorecardPath)
	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := learningdebt.Result{
		Schema:    learningdebt.Schema,
		Mode:      mode,
		Scorecard: scorecardPath,
		Cache:     cacheAbs,
		Stats:     stats,
		Planned:   plan,
		Synced:    []learningdebt.SyncRow{},
	}

	exit := 0
	if *live && len(plan) > 0 {
		result.Synced = learningdebt.Sync(plan, *repo, []string(labels), nil)
		learningdebt.MarkSuccessful(&seen, plan, result.Synced, time.Now())
		for _, row := range result.Synced {
			if !row.OK {
				exit = 1
			}
		}
		if err := learningdebt.SaveSeen(cacheAbs, seen); err != nil {
			fmt.Fprintf(stderr, "learning-debt-dispatch: save seen-cache: %v\n", err)
			exit = 1
		}
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "learning-debt-dispatch: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, learningdebt.Render(result))
	}
	return exit
}
