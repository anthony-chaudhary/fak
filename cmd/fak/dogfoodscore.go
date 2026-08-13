package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/dogfoodscore"
)

func cmdDogfoodScore(argv []string) { os.Exit(runDogfoodScore(os.Stdout, os.Stderr, argv)) }

func runDogfoodScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dogfood-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	claudeHome := fs.String("claude-home", "", "user home holding ~/.claude*/projects (default: detected)")
	windowHours := fs.Int("window-hours", dogfoodscore.DefaultConflationWindowHours, "only score sessions newer than this many hours")
	contextEvents := fs.Int("context-events", dogfoodscore.DefaultConflationContextEvents, "max event distance between a Stop-hook error and a success claim for the two to count as one conflation")
	sampleCap := fs.Int("sample-cap", dogfoodscore.DefaultConflationSampleCap, "max conflation hits retained as displayed evidence samples (the score counts every hit)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	kernelValue := fs.Bool("kernel-value", true, "fold durable token/turn/cache dogfood evidence into JSON")
	runsDir := fs.String("runs-dir", filepath.Join(repoRoot(), ".dispatch-runs"), "dispatch receipt archive for --kernel-value")
	cacheReceipt := fs.String("cache-witness", "", "typed fak-micro-cache-affinity-witness/1 JSON receipt")
	cohortMinimum := fs.Int("cohort-minimum", 5, "minimum durable post-default launches for outcome comparison")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dogfood-score: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := dogfoodscore.Build(dogfoodscore.Options{
		Root:          root,
		ClaudeHome:    *claudeHome,
		WindowHours:   *windowHours,
		ContextEvents: *contextEvents,
		SampleCap:     *sampleCap,
	})
	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak dogfood-score", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, dogfoodscore.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if *asJSON {
		var output any = payload
		if *kernelValue {
			output = struct {
				dogfoodscore.ScorecardPayload
				KernelValue dogfoodKernelValue `json:"kernel_value"`
			}{ScorecardPayload: payload, KernelValue: collectDogfoodKernelValue(*runsDir, *cacheReceipt, *cohortMinimum)}
		}
		if err := writeIndentedJSON(stdout, output); err != nil {
			fmt.Fprintf(stderr, "fak dogfood-score: encode json: %v\n", err)
			return 1
		}
	} else if *asMarkdown {
		fmt.Fprint(stdout, dogfoodscore.Markdown(payload))
	} else {
		fmt.Fprintln(stdout, dogfoodscore.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}
