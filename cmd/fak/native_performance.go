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
	jsonOut := fs.Bool("json", false, "emit the committed native-performance graph as JSON")
	nextOut := fs.Bool("next", false, "emit the first dependency-ready unwitnessed lever")
	dotOut := fs.Bool("dot", false, "emit the lever graph as Graphviz DOT")
	baselineLever := fs.String("baseline", "", "emit a pre-change baseline receipt template for LEVER")
	compareBaseline := fs.String("compare", "", "compare baseline receipt FILE with --candidate FILE")
	compareCandidate := fs.String("candidate", "", "candidate receipt FILE used with --compare")
	profilePath := fs.String("profile", "", "validate and classify native profile FILE")
	profileNextPath := fs.String("profile-next", "", "select next lever from native profile FILE")
	gatePath := fs.String("gate", "", "classify a candidate against the last accepted envelope receipt in FILE")
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
	if set["capacity-receipt"] {
		modeCount++
	}
	if *capacityPlan {
		modeCount++
	}
	if fs.NArg() != 0 || modeCount > 1 || ((*compareBaseline == "") != (*compareCandidate == "")) || (set["profile"] && *profilePath == "") || (set["profile-next"] && *profileNextPath == "") || (set["gate"] && *gatePath == "") || (set["capacity-receipt"] && *capacityReceiptPath == "") {
		fmt.Fprintln(stderr, "usage: fak native-performance [--json | --next | --dot | --baseline LEVER | --compare BASELINE --candidate CANDIDATE | --profile FILE | --profile-next FILE | --gate FILE | --capacity-plan | --capacity-receipt FILE]")
		return 2
	}

	graph := nativeperf.ActiveGraph()
	if set["gate"] {
		data, err := os.ReadFile(*gatePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: read gate request: %v\n", err)
			return 1
		}
		var request nativeperf.GateRequest
		decoder := json.NewDecoder(strings.NewReader(string(data)))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil {
			fmt.Fprintf(stderr, "fak native-performance: decode gate request: %v\n", err)
			return 1
		}
		verdict, err := nativeperf.Gate(request)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: gate: %v\n", err)
			return 1
		}
		if code := encodeNativePerformanceJSON(stdout, stderr, verdict); code != 0 {
			return code
		}
		if verdict.Classification == nativeperf.GateRegression {
			return 3
		}
		return 0
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
		data, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: read profile: %v\n", err)
			return 1
		}
		profile, err := nativeperf.DecodeProfile(data)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: %v\n", err)
			return 1
		}
		if *profileNextPath != "" {
			lever, classification, err := nativeperf.NextLeverFromProfile(graph, profile)
			if err != nil {
				fmt.Fprintf(stderr, "fak native-performance: profile next: %v\n", err)
				return 1
			}
			return encodeNativePerformanceJSON(stdout, stderr, struct {
				Classification nativeperf.BottleneckClassification `json:"classification"`
				Lever          nativeperf.Lever                    `json:"lever"`
				Override       *nativeperf.SelectionOverride       `json:"selection_override,omitempty"`
			}{classification, *lever, profile.Override})
		}
		classification, err := nativeperf.ClassifyProfile(graph, profile)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: classify profile: %v\n", err)
			return 1
		}
		return encodeNativePerformanceJSON(stdout, stderr, classification)
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
		baselineData, err := os.ReadFile(*compareBaseline)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: read baseline: %v\n", err)
			return 1
		}
		candidateData, err := os.ReadFile(*compareCandidate)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: read candidate: %v\n", err)
			return 1
		}
		baseline, err := nativeperf.DecodeReceipt(baselineData)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: %v\n", err)
			return 1
		}
		candidate, err := nativeperf.DecodeReceipt(candidateData)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: %v\n", err)
			return 1
		}
		comparison, err := nativeperf.CompareReceipts(graph, baseline, candidate)
		if err != nil {
			fmt.Fprintf(stderr, "fak native-performance: compare: %v\n", err)
			return 1
		}
		return encodeNativePerformanceJSON(stdout, stderr, comparison)
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
