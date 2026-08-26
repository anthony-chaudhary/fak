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
	"github.com/anthony-chaudhary/fak/internal/systembaseline"
)

func cmdNativePerformance(args []string) {
	os.Exit(runNativePerformance(os.Stdout, os.Stderr, args))
}

func runNativePerformance(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("native-performance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit the committed native-performance graph as JSON")
	nextOut := fs.Bool("next", false, "emit the first dependency-ready unwitnessed lever")
	dotOut := fs.Bool("dot", false, "emit the lever graph as Graphviz DOT")
	baselineLever := fs.String("baseline", "", "emit a pre-change baseline receipt template for LEVER")
	compareBaseline := fs.String("compare", "", "compare baseline receipt FILE with --candidate FILE")
	compareCandidate := fs.String("candidate", "", "candidate receipt FILE used with --compare")
	profilePath := fs.String("profile", "", "validate and classify native profile FILE")
	profileNextPath := fs.String("profile-next", "", "select next lever from native profile FILE")
	gatePath := fs.String("gate", "", "classify a candidate against the last accepted envelope receipt in FILE")
	attachReceiptPath := fs.String("attach-receipt", "", "append --system-baseline attestation FILE to native receipt FILE")
	attachBaselinePath := fs.String("system-baseline", "", "system-baseline attestation FILE used with --attach-receipt")
	outPath := fs.String("out", "", "write attachment result privately to FILE instead of stdout")
	capacityReceiptPath := fs.String("capacity-receipt", "", "validate the #8971 no-FAK_Q4K_FREE_CPU native Metal capacity receipt FILE")
	capacityPlan := fs.Bool("capacity-plan", false, "emit the bounded #8971 no-FAK_Q4K_FREE_CPU capture contract without touching hardware")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) {
		set[f.Name] = true
	})
	modeCount := boolCount(*jsonOut, *nextOut, *dotOut)
	if *baselineLever != "" {
		modeCount++
	}
	if *compareBaseline != "" || *compareCandidate != "" {
		modeCount++
	}
	if set["profile"] {
		modeCount++
	}
	if set["profile-next"] {
		modeCount++
	}
	if set["gate"] {
		modeCount++
	}
	if set["attach-receipt"] || set["system-baseline"] {
		modeCount++
	}
	if set["capacity-receipt"] {
		modeCount++
	}
	if *capacityPlan {
		modeCount++
	}
	attachMode := set["attach-receipt"] || set["system-baseline"]
	if fs.NArg() != 0 || modeCount > 1 || ((*compareBaseline == "") != (*compareCandidate == "")) || (set["profile"] && *profilePath == "") || (set["profile-next"] && *profileNextPath == "") || (set["gate"] && *gatePath == "") || (set["capacity-receipt"] && *capacityReceiptPath == "") || (attachMode && (*attachReceiptPath == "" || *attachBaselinePath == "")) || (set["out"] && !attachMode) {
		fmt.Fprintln(stderr, "usage: fak native-performance [--json | --next | --dot | --baseline LEVER | --compare BASELINE --candidate CANDIDATE | --profile FILE | --profile-next FILE | --gate FILE | --attach-receipt RECEIPT --system-baseline ATTESTATION [--out FILE] | --capacity-plan | --capacity-receipt FILE]")
		return 2
	}

	graph := nativeperf.ActiveGraph()
	if attachMode {
		return attachNativePerformanceSystemBaseline(stdout, stderr, graph, *attachReceiptPath, *attachBaselinePath, *outPath)
	}
	if set["gate"] {
		return runNativePerformanceGate(stdout, stderr, *gatePath)
	}
	if err := nativeperf.Validate(graph); err != nil {
		fmt.Fprintf(stderr, "fak native-performance: %v\n", err)
		return 1
	}
	if set["capacity-receipt"] {
		data, err := os.ReadFile(*capacityReceiptPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: read capacity receipt: %v\n", err)
			return 1
		}
		receipt, err := decodeNativeCapacityReceipt(data)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: %v\n", err)
			return 1
		}
		return encodeNativePerformanceJSON(stdout, stderr, nativeCapacityReadbackFor(receipt))
	}
	if *capacityPlan {
		return encodeNativePerformanceJSON(stdout, stderr, nativeCapacityCapturePlan())
	}
	if set["profile"] || set["profile-next"] {
		path := *profilePath
		if set["profile-next"] {
			path = *profileNextPath
		}
		return runNativePerformanceProfile(stdout, stderr, graph, path, *profileNextPath != "")
	}
	if *baselineLever != "" {
		receipt, err := nativeperf.BaselineTemplate(graph, *baselineLever)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: baseline template: %v\n", err)
			return 1
		}
		return encodeNativePerformanceJSON(stdout, stderr, receipt)
	}
	if *compareBaseline != "" {
		return runNativePerformanceCompare(stdout, stderr, graph, *compareBaseline, *compareCandidate)
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
	if *nextOut {
		next, err := nativeperf.NextLever(graph)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: select next: %v\n", err)
			return 1
		}
		if next == nil {
			fmt.Fprintln(stdout, "NEXT NATIVE-PERFORMANCE ARM: none")
			return 0
		}
		renderNextNativePerformance(stdout, *next)
		return 0
	}
	if *dotOut {
		dot, err := nativeperf.DOT(graph)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: render DOT: %v\n", err)
			return 1
		}
		fmt.Fprint(stdout, dot)
		return 0
	}
	renderNativePerformance(stdout, graph)
	return 0
}

func attachNativePerformanceSystemBaseline(stdout, stderr io.Writer, graph nativeperf.Graph, receiptPath, baselinePath, outPath string) int {
	receiptData, err := os.ReadFile(receiptPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak native-performance: read receipt: %v\n", err)
		return 1
	}
	receipt, err := nativeperf.DecodeReceipt(receiptData)
	if err != nil {
		fmt.Fprintf(stderr, "fak native-performance: %v\n", err)
		return 1
	}
	baselineFile, err := os.Open(baselinePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak native-performance: read system baseline: %v\n", err)
		return 1
	}
	attestation, decodeErr := systembaseline.Decode(baselineFile)
	closeErr := baselineFile.Close()
	if decodeErr != nil {
		fmt.Fprintf(stderr, "fak native-performance: %v\n", decodeErr)
		return 1
	}
	if closeErr != nil {
		fmt.Fprintf(stderr, "fak native-performance: close system baseline: %v\n", closeErr)
		return 1
	}
	upgraded, err := nativeperf.AttachSystemBaseline(graph, receipt, attestation)
	if err != nil {
		fmt.Fprintf(stderr, "fak native-performance: attach system baseline: %v\n", err)
		return 1
	}
	if outPath == "" {
		return encodeNativePerformanceJSON(stdout, stderr, upgraded)
	}
	raw, err := json.MarshalIndent(upgraded, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak native-performance: encode attachment result: %v\n", err)
		return 1
	}
	raw = append(raw, '\n')
	if err := writePrivateJSON(outPath, raw); err != nil {
		fmt.Fprintf(stderr, "fak native-performance: write %s: %v\n", outPath, err)
		return 1
	}
	return 0
}

func boolCount(values ...bool) int {
	count := 0
	for _, value := range values {
		if value {
			count++
		}
	}
	return count
}

func renderNextNativePerformance(w io.Writer, lever nativeperf.Lever) {
	fmt.Fprintln(w, "NEXT NATIVE-PERFORMANCE ARM")
	fmt.Fprintf(w, "Lever: %s\n", lever.ID)
	fmt.Fprintf(w, "Envelope: %s | %s | %s\n", lever.Applicability.EnvelopeID, lever.Applicability.Platform, lever.Applicability.Backend)
	fmt.Fprintf(w, "State: enabled=%t status=%s\n", lever.Enabled, lever.Status)
	fmt.Fprintf(w, "Dependencies: %s\n", joinOrDash(lever.DependencyIDs))
	fmt.Fprintf(w, "Conflicts: %s\n", joinOrDash(lever.ConflictIDs))
	fmt.Fprintf(w, "Expected [%s]: %s\n", lever.Expected.Classification, lever.Expected.Summary)
	fmt.Fprintf(w, "Expected provenance: %s\n", lever.Expected.Provenance)
	fmt.Fprintf(w, "Owning issue: #%d %s\n", lever.OwningIssue.Number, lever.OwningIssue.URL)
	fmt.Fprintf(w, "Required witness: %s\n", lever.NextWitness)
}

func joinOrDash(values []string) string {
	if len(values) == 0 {
		return "-"
	}
	return strings.Join(values, ",")
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
		deps := joinOrDash(rung.DependencyIDs)
		expected := fmt.Sprintf("%s..%s [%s]", formatThroughput(rung.Expected.FloorTokensPerSecond), formatThroughput(rung.Expected.RoofTokensPerSecond), rung.Expected.Classification)
		witnessed := "pending"
		if rung.Witnessed != nil {
			witnessed = fmt.Sprintf("%s [%s/%s]", formatThroughput(rung.Witnessed.TokensPerSecond), rung.Witnessed.Classification, rung.Witnessed.Comparability)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t#%d\n", enabled, rung.Status, rung.ID, deps, expected, witnessed, rung.NextIssue.Number)
	}
	_ = tw.Flush()
	fmt.Fprintln(w, "Feature stack:")
	for _, feature := range graph.Features {
		enabled := "[ ]"
		if feature.Enabled {
			enabled = "[x]"
		}
		fmt.Fprintf(w, "- %s %s (%s; rung=%s): %s\n", enabled, feature.ID, feature.Status, feature.RungID, feature.Observable)
	}
	fmt.Fprintln(w, "Gaps:")
	for _, rung := range graph.Rungs {
		fmt.Fprintf(w, "- %s: %s (next #%d)\n", rung.ID, rung.Gap, rung.NextIssue.Number)
	}

	fmt.Fprintln(w, "Independent levers (envelopes remain separate):")
	for _, envelope := range graph.Envelopes {
		fmt.Fprintf(w, "Envelope %s: %s | %s %s | P%d/T%d | %s/%s\n", envelope.ID, envelope.Model, envelope.Quantization, envelope.Hardware, envelope.PromptTokens, envelope.DecodeTokens, envelope.Engine, envelope.Backend)
		for _, lever := range graph.Levers {
			if lever.Applicability.EnvelopeID != envelope.ID {
				continue
			}
			enabled := "[ ]"
			if lever.Enabled {
				enabled = "[x]"
			}
			witness := "pending"
			if lever.Witnessed != nil {
				witness = lever.Witnessed.Summary
			}
			fmt.Fprintf(w, "- %s %s (%s; deps=%s; conflicts=%s; issue=#%d)\n  expected [%s]: %s (source: %s)\n  witnessed: %s\n  next witness: %s\n", enabled, lever.ID, lever.Status, joinOrDash(lever.DependencyIDs), joinOrDash(lever.ConflictIDs), lever.OwningIssue.Number, lever.Expected.Classification, lever.Expected.Summary, lever.Expected.Provenance, witness, lever.NextWitness)
		}
	}
}

func formatThroughput(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}

func encodeNativePerformanceJSON(stdout, stderr io.Writer, value any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		fmt.Fprintf(stderr, "fak native-performance: encode JSON: %v\n", err)
		return 1
	}
	return 0
}
