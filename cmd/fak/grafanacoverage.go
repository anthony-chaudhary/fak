package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/nativeperfcoverage"
)

// grafanaCoverageValidate drives the committed native-performance dashboard
// contracts through internal/nativeperfcoverage. Production keeps the real
// Validate; tests replace it to stay hermetic (no promtool binary on PATH).
var grafanaCoverageValidate = func(ctx context.Context, cfg nativeperfcoverage.Config) (nativeperfcoverage.Matrix, error) {
	return nativeperfcoverage.Validate(ctx, cfg)
}

// coveragePromtoolLookPath resolves the promtool binary on PATH. Production
// keeps exec.LookPath; tests replace it to stay hermetic.
var coveragePromtoolLookPath = exec.LookPath

// runGrafanaCoverage verifies all four committed native-performance dashboards
// against their provisioning contracts, controlled fixtures, and the supervised
// Prometheus job, then prints the deterministic coverage matrix (text, or
// --json). Fail-closed: any missing file, unknown metric, or unparsed query
// exits 1 with the spec UID naming the broken triple. Typed environment
// refusal: when no promtool binary can be resolved (--promtool or PATH), it
// prints a PROMTOOL_UNAVAILABLE line and exits 2 instead of degrading to a
// weaker checker.
func runGrafanaCoverage(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak grafana coverage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repository root containing tools/grafana dashboards, contracts, and fixtures")
	prometheusConfig := fs.String("prometheus-config", "tools/grafana/prometheus.yml", "Prometheus config proving the supervised scrape job has an active target")
	jsonOut := fs.Bool("json", false, "emit the deterministic coverage matrix as JSON instead of text")
	promtoolBin := fs.String("promtool", "", "promtool binary that parses every extracted query (default: promtool from PATH)")
	if !parseFlags(fs, argv) {
		return 2
	}

	promtoolPath := *promtoolBin
	if promtoolPath == "" {
		resolved, err := coveragePromtoolLookPath("promtool")
		if err != nil {
			fmt.Fprintf(stderr, "fak grafana coverage: PROMTOOL_UNAVAILABLE: install promtool or pass --promtool /abs/path/to/promtool\n")
			return 2
		}
		promtoolPath = resolved
	}

	matrix, err := grafanaCoverageValidate(context.Background(), nativeperfcoverage.Config{
		Root:             *root,
		PrometheusConfig: *prometheusConfig,
		Checker:          nativeperfcoverage.PromtoolChecker{Path: promtoolPath},
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak grafana coverage: %v\n", err)
		return 1
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(matrix); err != nil {
			fmt.Fprintf(stderr, "fak grafana coverage: encode matrix: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, matrix.Text())
	return 0
}
