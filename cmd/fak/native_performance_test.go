package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func TestNativePerformanceJSONIsValidAndSeparatesEvidence(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var graph nativeperf.Graph
	if err := json.Unmarshal(stdout.Bytes(), &graph); err != nil {
		t.Fatal(err)
	}
	if err := nativeperf.Validate(graph); err != nil {
		t.Fatal(err)
	}
	if graph.Schema != nativeperf.Schema || graph.Rungs[0].Witnessed == nil || graph.Rungs[0].Witnessed.TokensPerSecond != 3.3 {
		t.Fatalf("unexpected native graph: %+v", graph)
	}
	if graph.Comparison.TokensPerSecond != 6.966061 || graph.Rungs[1].Witnessed != nil {
		t.Fatalf("comparison or pending witness was conflated: %+v", graph)
	}
}

func TestNativePerformanceHumanOutputIsChecklistTable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{
		"NATIVE RAW-MODEL HILL CLIMB",
		"P32/T64",
		"6.966061 tok/s",
		"[x]",
		"present",
		"resident-q4k-baseline",
		"3.3 [witnessed/approximate]",
		"coarse-resident-hybrid-graph",
		"5..6.966061 [hypothesis]",
		"Gaps:",
		"#8697",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("human output missing %q:\n%s", want, got)
		}
	}
}

func TestNativePerformanceOutputIsDeterministic(t *testing.T) {
	var first, second bytes.Buffer
	if code := runNativePerformance(&first, &bytes.Buffer{}, []string{"--json"}); code != 0 {
		t.Fatalf("first exit=%d", code)
	}
	if code := runNativePerformance(&second, &bytes.Buffer{}, []string{"--json"}); code != 0 {
		t.Fatalf("second exit=%d", code)
	}
	if first.String() != second.String() {
		t.Fatal("native-performance JSON changed across identical calls")
	}
}

func TestNativePerformanceRejectsPositionalArguments(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"extra"}); code != 2 {
		t.Fatalf("exit=%d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak native-performance [--json]") {
		t.Fatalf("stderr=%q", stderr.String())
	}
}
