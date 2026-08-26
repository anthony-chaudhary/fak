package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func runNativePerformanceGate(stdout, stderr io.Writer, path string) int {
	data, err := os.ReadFile(path)
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

func runNativePerformanceProfile(stdout, stderr io.Writer, graph nativeperf.Graph, path string, next bool) int {
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
	if next {
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

func runNativePerformanceCompare(stdout, stderr io.Writer, graph nativeperf.Graph, baselinePath, candidatePath string) int {
	baselineData, err := os.ReadFile(baselinePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak native-performance: read baseline: %v\n", err)
		return 1
	}
	candidateData, err := os.ReadFile(candidatePath)
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
