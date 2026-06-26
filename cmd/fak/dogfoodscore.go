package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/dogfoodscore"
)

func cmdDogfoodScore(argv []string) { os.Exit(runDogfoodScore(os.Stdout, os.Stderr, argv)) }

func runDogfoodScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dogfood-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	claudeHome := fs.String("claude-home", "", "user home holding ~/.claude*/projects (default: detected)")
	windowHours := fs.Int("window-hours", dogfoodscore.DefaultConflationWindowHours, "only score sessions newer than this many hours")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
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
		Root:        root,
		ClaudeHome:  *claudeHome,
		WindowHours: *windowHours,
	})
	if *comparePath != "" {
		b, err := os.ReadFile(*comparePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak dogfood-score: read --compare: %v\n", err)
			return 2
		}
		var base map[string]any
		if err := json.Unmarshal(b, &base); err != nil {
			fmt.Fprintf(stderr, "fak dogfood-score: parse --compare: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, dogfoodscore.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
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
