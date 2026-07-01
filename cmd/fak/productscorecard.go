package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/productscorecard"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
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

	health := productScorecardContextHealth(payload)

	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, productScorecardJSONWithContextHealth(payload, health)); err != nil {
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
		fmt.Fprintf(stdout, "context-health: %s\n", health)
	}
	return okExit(payload.OK)
}

// productScorecardContextHealth folds the payload's managed-context SLO rows (see
// internal/productscorecard.ManagedContextSLOReport — the eight managed-context
// glossary areas) into the closed severity vocabulary from issue #1579/#1920
// (internal/scorecardpane.ContextHealthSeverity), so the product scorecard renders
// one reviewable enum instead of a raw debt count. scorecardpane is tier-1
// stdlib-only and does not import internal/productscorecard (see
// internal/architest's tier map), so this CLI layer is where the two meet: it reads
// the already-computed managed_context rows off the payload's corpus map and calls
// the pure classifier. Absent/empty managed-context data classifies as fresh (the
// same "no data means no known defect" posture ManagedContextSLOReport itself takes
// for a nil report).
func productScorecardContextHealth(p productscorecard.Payload) scorecardpane.ContextHealthSeverity {
	mc, ok := p.Corpus["managed_context"].(map[string]any)
	if !ok {
		return scorecardpane.ContextHealthFresh
	}
	rows := productScorecardSLORows(mc["rows"])
	signals := scorecardpane.ContextHealthSignalsFromSLORows(rows)
	return scorecardpane.ClassifyContextHealth(signals)
}

// productScorecardSLORows normalizes the managed-context "rows" value into
// []map[string]any regardless of whether it arrived as a live in-process
// []map[string]any (the shape ManagedContextSLOReport builds directly) or as
// []any-of-map[string]any (the shape a JSON round-trip, e.g. a parsed --compare
// baseline, produces).
func productScorecardSLORows(v any) []map[string]any {
	switch rows := v.(type) {
	case []map[string]any:
		return rows
	case []any:
		out := make([]map[string]any, 0, len(rows))
		for _, r := range rows {
			if m, ok := r.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	default:
		return nil
	}
}

// productScorecardJSONWithContextHealth wraps the product-scorecard payload for
// --json so the closed context_health enum is a first-class, top-level field —
// exactly the issue #1579 done condition ("render a closed context-health enum with
// no free-text severity"), without perturbing any existing field the payload already
// emits (json.Marshal of an embedded struct + a sibling field adds one key).
func productScorecardJSONWithContextHealth(p productscorecard.Payload, health scorecardpane.ContextHealthSeverity) any {
	return struct {
		productscorecard.Payload
		ContextHealth string `json:"context_health"`
	}{Payload: p, ContextHealth: string(health)}
}
