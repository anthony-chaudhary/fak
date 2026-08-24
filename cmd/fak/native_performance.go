package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func cmdNativePerformance(args []string) {
	os.Exit(runNativePerformance(os.Stdout, os.Stderr, args))
}

func runNativePerformance(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("native-performance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit the committed native-performance rung graph as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak native-performance [--json]")
		return 2
	}

	graph := nativeperf.ActiveGraph()
	if err := nativeperf.Validate(graph); err != nil {
		fmt.Fprintf(stderr, "fak native-performance: %v\n", err)
		return 1
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(graph); err != nil {
			fmt.Fprintf(stderr, "fak native-performance: encode JSON: %v\n", err)
			return 1
		}
		return 0
	}
	renderNativePerformance(stdout, graph)
	return 0
}

func renderNativePerformance(w io.Writer, graph nativeperf.Graph) {
	e := graph.Envelope
	fmt.Fprintln(w, "NATIVE RAW-MODEL HILL CLIMB")
	fmt.Fprintf(w, "Envelope: %s | %s %s | %s, %d GiB | P%d/T%d | %s/%s\n", e.Model, e.Quantization, e.Backend, e.Hardware, e.MemoryGiB, e.PromptTokens, e.DecodeTokens, e.Engine, e.ForwardPath)
	fmt.Fprintf(w, "Comparison: %s tok/s | %s | %s/%s\n", formatThroughput(graph.Comparison.TokensPerSecond), graph.Comparison.Engine, graph.Comparison.Classification, graph.Comparison.Comparability)
	fmt.Fprintln(w, "Checklist: [x] enabled in this envelope; expected values are hypotheses, never measurements.")

	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "ON\tSTATUS\tRUNG\tDEPENDENCIES\tEXPECTED TOK/S\tWITNESSED TOK/S\tNEXT")
	for _, rung := range graph.Rungs {
		enabled := "[ ]"
		if rung.Enabled {
			enabled = "[x]"
		}
		deps := "-"
		if len(rung.DependencyIDs) != 0 {
			deps = strings.Join(rung.DependencyIDs, ",")
		}
		expected := fmt.Sprintf("%s..%s [%s]", formatThroughput(rung.Expected.FloorTokensPerSecond), formatThroughput(rung.Expected.RoofTokensPerSecond), rung.Expected.Classification)
		witnessed := "pending"
		if rung.Witnessed != nil {
			witnessed = fmt.Sprintf("%s [%s/%s]", formatThroughput(rung.Witnessed.TokensPerSecond), rung.Witnessed.Classification, rung.Witnessed.Comparability)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t#%d\n", enabled, rung.Status, rung.ID, deps, expected, witnessed, rung.NextIssue.Number)
	}
	_ = tw.Flush()
	fmt.Fprintln(w, "Gaps:")
	for _, rung := range graph.Rungs {
		fmt.Fprintf(w, "- %s: %s (next #%d)\n", rung.ID, rung.Gap, rung.NextIssue.Number)
	}
}

func formatThroughput(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
