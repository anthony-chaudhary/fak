package main

import (
	"bytes"
	"encoding/json"
	"os"
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
		if !strings.Contains(stderr.String(), "usage: fak native-performance [--json | --next | --dot | --baseline LEVER | --compare BASELINE --candidate CANDIDATE]") {
			t.Fatalf("args=%v stderr=%q", args, stderr.String())
		}
	}
}

func TestNativePerformanceBaselineTemplate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--baseline", "metal.command-buffer-amortization"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var receipt nativeperf.ExperimentReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != nativeperf.ReceiptSchema || receipt.Role != nativeperf.RoleBaseline || receipt.ChangedLeverID != "metal.command-buffer-amortization" || receipt.Execution.Engine != "fak-native" {
		t.Fatalf("unexpected template: %+v", receipt)
	}
}

func TestNativePerformanceCompareReceipts(t *testing.T) {
	graph := nativeperf.ActiveGraph()
	baseline, err := nativeperf.BaselineTemplate(graph, "metal.command-buffer-amortization")
	if err != nil {
		t.Fatal(err)
	}
	fill := func(r *nativeperf.ExperimentReceipt, role, rev string, delta float64) {
		r.Role, r.Revision, r.Machine.ScrubbedID = role, rev, "lab-class-a"
		r.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
		r.Commands = []string{"fak run-model --native --receipt-out receipt.json"}
		r.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
		for i := range r.Repetitions {
			r.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: 3 + delta}
		}
		if role == nativeperf.RoleCandidate {
			r.ChangedAxes = []string{"lever:" + r.ChangedLeverID}
		}
	}
	candidate := baseline
	candidate.UnchangedControls = append([]string(nil), baseline.UnchangedControls...)
	candidate.Repetitions = append([]nativeperf.Repetition(nil), baseline.Repetitions...)
	candidate.Commands = append([]string(nil), baseline.Commands...)
	candidate.ProfilerArtifacts = append([]nativeperf.ArtifactRef(nil), baseline.ProfilerArtifacts...)
	fill(&baseline, nativeperf.RoleBaseline, "base-rev", 0)
	fill(&candidate, nativeperf.RoleCandidate, "candidate-rev", 1)
	dir := t.TempDir()
	basePath, candidatePath := dir+"/baseline.json", dir+"/candidate.json"
	for path, value := range map[string]nativeperf.ExperimentReceipt{basePath: baseline, candidatePath: candidate} {
		data, _ := json.Marshal(value)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--compare", basePath, "--candidate", candidatePath}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var comparison nativeperf.Comparison
	if err := json.Unmarshal(stdout.Bytes(), &comparison); err != nil {
		t.Fatal(err)
	}
	if comparison.DeltaTokensPerS != 1 || comparison.ChangedLeverID != "metal.command-buffer-amortization" {
		t.Fatalf("comparison=%+v", comparison)
	}
	candidate.Execution.FallbackCount = 1
	data, _ := json.Marshal(candidate)
	_ = os.WriteFile(candidatePath, data, 0600)
	stdout.Reset()
	stderr.Reset()
	if code := runNativePerformance(&stdout, &stderr, []string{"--compare", basePath, "--candidate", candidatePath}); code != 1 || !strings.Contains(stderr.String(), "fallback count") {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
}
