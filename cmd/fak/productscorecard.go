package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/productscorecard"
)

func runProductScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak product-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	chart := fs.Bool("chart", false, "emit an at-a-glance product standing chart")
	critical := fs.Bool("critical", false, "emit the most-critical-areas backlog")
	gaps := fs.Bool("gaps", false, "emit the coverage backlog")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	markdownDir := fs.String("markdown-dir", "", "regenerate the doc folder")
	dataPath := fs.String("data", "", "data directory (default: tools/product_scorecard.data)")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	stamp := fs.String("stamp", "", "optional stamp embedded in generated docs")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak product-scorecard: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	data := *dataPath
	if data != "" {
		if abs, err := filepath.Abs(data); err == nil {
			data = abs
		}
	}
	payload := productscorecard.Collect(root, data)

	if *comparePath != "" {
		b, err := os.ReadFile(*comparePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak product-scorecard: cannot read baseline %s: %v\n", *comparePath, err)
			return 2
		}
		var baseline map[string]any
		if err := json.Unmarshal(b, &baseline); err != nil {
			fmt.Fprintf(stderr, "fak product-scorecard: cannot parse baseline %s: %v\n", *comparePath, err)
			return 2
		}
		fmt.Fprintln(stdout, productscorecard.RenderCompare(baseline, payload))
		return okExit(payload.OK)
	}

	if *markdownDir != "" {
		outDir, err := filepath.Abs(*markdownDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak product-scorecard: --markdown-dir: %v\n", err)
			return 2
		}
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			fmt.Fprintf(stderr, "fak product-scorecard: create markdown dir: %v\n", err)
			return 1
		}
		for rel, content := range productscorecard.RenderDocFolder(payload, *stamp) {
			path := filepath.Join(outDir, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				fmt.Fprintf(stderr, "fak product-scorecard: create markdown parent: %v\n", err)
				return 1
			}
			if err := os.WriteFile(path, []byte(content+"\n"), 0o644); err != nil {
				fmt.Fprintf(stderr, "fak product-scorecard: write markdown: %v\n", err)
				return 1
			}
		}
		if !*asJSON {
			fmt.Fprintf(stdout, "wrote product-scorecard doc folder -> %s\n", outDir)
		}
	}

	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak product-scorecard: encode json: %v\n", err)
			return 1
		}
	case *chart:
		fmt.Fprintln(stdout, productscorecard.RenderChart(payload))
	case *critical:
		fmt.Fprintln(stdout, productscorecard.RenderCritical(payload))
	case *gaps:
		fmt.Fprintln(stdout, productscorecard.RenderGaps(payload))
	case *markdownDir == "":
		fmt.Fprintln(stdout, productscorecard.Render(payload))
	}
	return okExit(payload.OK)
}
