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
	if graph.Comparison.TokensPerSecond != 6.966061 || graph.Rungs[1].Witnessed != nil || len(graph.Envelopes) != 2 {
		t.Fatalf("comparison, pending witness, or envelope was conflated: %+v", graph)
	}
}

func TestNativePerformanceHumanOutputIsBackwardReadableAndShowsLevers(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{
		"NATIVE RAW-MODEL HILL CLIMB", "P32/T64", "6.966061 tok/s", "[x]", "present",
		"resident-q4k-baseline", "3.3 [witnessed/approximate]", "coarse-resident-hybrid-graph",
		"5..6.966061 [hypothesis]", "Feature stack:", "[x] resident-q4k-weights",
		"[ ] coarse-token-submission", "Gaps:", "#8697", "Independent levers",
		"metal.command-buffer-amortization", "cuda.dp4a-q4k-mmvq", "expected [hypothesis]",
		"next witness:", "fak-native/cuda",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("human output missing %q:\n%s", want, got)
		}
	}
}

func TestNativePerformanceNextIsDeterministicAndRunnable(t *testing.T) {
	var first, second bytes.Buffer
	if code := runNativePerformance(&first, &bytes.Buffer{}, []string{"--next"}); code != 0 {
		t.Fatalf("first exit=%d", code)
	}
	if code := runNativePerformance(&second, &bytes.Buffer{}, []string{"--next"}); code != 0 {
		t.Fatalf("second exit=%d", code)
	}
	if first.String() != second.String() {
		t.Fatal("--next changed across identical calls")
	}
	for _, want := range []string{"metal.command-buffer-amortization", "Owning issue: #8324", "Required witness:", "OFF/ON", "qwen38-27b-q4km-m3pro-p32-t64"} {
		if !strings.Contains(first.String(), want) {
			t.Fatalf("--next missing %q:\n%s", want, first.String())
		}
	}
}

func TestNativePerformanceDOTIsDeterministicAndSeparated(t *testing.T) {
	var first, second bytes.Buffer
	if code := runNativePerformance(&first, &bytes.Buffer{}, []string{"--dot"}); code != 0 {
		t.Fatalf("first exit=%d", code)
	}
	if code := runNativePerformance(&second, &bytes.Buffer{}, []string{"--dot"}); code != 0 {
		t.Fatalf("second exit=%d", code)
	}
	if first.String() != second.String() {
		t.Fatal("--dot changed across identical calls")
	}
	for _, want := range []string{"digraph native_performance", "cluster_qwen38_27b_q4km_m3pro_p32_t64", "cluster_qwen38_27b_q4k_a100_p1_decode", `label="depends"`, `label="conflicts"`} {
		if !strings.Contains(first.String(), want) {
			t.Fatalf("--dot missing %q:\n%s", want, first.String())
		}
	}
}

func TestNativePerformanceJSONOutputIsDeterministic(t *testing.T) {
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

func TestNativePerformanceRejectsInvalidArguments(t *testing.T) {
	for _, args := range [][]string{{"extra"}, {"--json", "--next"}, {"--next", "--dot"}} {
		var stdout, stderr bytes.Buffer
		if code := runNativePerformance(&stdout, &stderr, args); code != 2 {
			t.Fatalf("args=%v exit=%d, want 2", args, code)
		}
		if !strings.Contains(stderr.String(), "usage: fak native-performance [--json | --next | --dot]") {
			t.Fatalf("args=%v stderr=%q", args, stderr.String())
		}
	}
}
