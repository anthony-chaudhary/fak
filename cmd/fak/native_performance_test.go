package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/systembaseline"
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

func TestNativePerformanceNextIsDeterministicGraphProjection(t *testing.T) {
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
	for _, want := range []string{"NEXT GRAPH-READY", "semantic graph projection only", "use --current work_packets", "metal.command-buffer-amortization", "Owning issue: #8324", "Required witness:", "OFF/ON", "qwen38-27b-q4km-m3pro-p32-t64"} {
		if !strings.Contains(first.String(), want) {
			t.Fatalf("--next missing %q:\n%s", want, first.String())
		}
	}
}

func TestNativePerformanceCurrentShowsAllReadyWork(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--current"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var snapshot nativeperf.CurrentSnapshot
	if err := json.Unmarshal(stdout.Bytes(), &snapshot); err != nil {
		t.Fatal(err)
	}
	if err := nativeperf.ValidateCurrentSnapshot(nativeperf.ActiveGraph(), snapshot); err != nil {
		t.Fatal(err)
	}
	got := stdout.String()
	for _, want := range []string{"measurement-control-loop", "active-native-lane-collision", "mac.m1-streamed-q4k-no-copy", "cuda.cache-weight-residency", "metal.command-buffer-amortization", "metal.paged-kv", "metal.chunked-prefill", "cuda.q8_1-activation-quant", "keep/reject"} {
		if !strings.Contains(got, want) {
			t.Errorf("--current missing %q", want)
		}
	}
}

func TestNativePerformanceCurrentMarkdownIsDeterministic(t *testing.T) {
	var first, second bytes.Buffer
	if code := runNativePerformance(&first, &bytes.Buffer{}, []string{"--current-md"}); code != 0 {
		t.Fatalf("first exit=%d", code)
	}
	if code := runNativePerformance(&second, &bytes.Buffer{}, []string{"--current-md"}); code != 0 {
		t.Fatalf("second exit=%d", code)
	}
	if first.String() != second.String() {
		t.Fatal("--current-md changed across identical calls")
	}
	for _, want := range []string{"Current native-performance constraints", "Divide-and-conquer execution", "Graph-dependency-ready arms and collisions", "OSS-to-performance walk"} {
		if !strings.Contains(first.String(), want) {
			t.Errorf("--current-md missing %q", want)
		}
	}
}

func TestNativePerformanceCurrentDocIsFresh(t *testing.T) {
	snapshot, err := nativeperf.BuildCurrentSnapshot(nativeperf.ActiveGraph())
	if err != nil {
		t.Fatal(err)
	}
	want := nativeperf.RenderCurrentMarkdown(snapshot)
	got, err := os.ReadFile(filepath.Join("..", "..", "docs", "benchmarks", "NATIVE-PERFORMANCE-CURRENT.md"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		at := 0
		for at < len(got) && at < len(want) && got[at] == want[at] {
			at++
		}
		gotEnd, wantEnd := at+120, at+120
		if gotEnd > len(got) {
			gotEnd = len(got)
		}
		if wantEnd > len(want) {
			wantEnd = len(want)
		}
		t.Fatalf("docs/benchmarks/NATIVE-PERFORMANCE-CURRENT.md is stale at byte %d (got bytes=%d want=%d): got %q want %q; regenerate with fak native-performance --current-md", at, len(got), len(want), got[at:gotEnd], want[at:wantEnd])
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
	for _, args := range [][]string{
		{"extra"},
		{"--json", "--next"},
		{"--current", "--current-md"},
		{"--current", "--json"},
		{"--next", "--dot"},
		{"--profile="},
		{"--profile-next="},
		{"--capacity-receipt="},
		{"--profile", "profile.json", "--json"},
		{"--profile", "profile.json", "--profile-next", "profile.json"},
		{"--attach-receipt", "receipt.json"},
		{"--system-baseline", "baseline.json"},
		{"--out", "receipt.json"},
	} {
		var stdout, stderr bytes.Buffer
		if code := runNativePerformance(&stdout, &stderr, args); code != 2 {
			t.Fatalf("args=%v exit=%d, want 2", args, code)
		}
		if !strings.Contains(stderr.String(), "usage: fak native-performance [--json | --next | --current | --current-md | --dot") {
			t.Fatalf("args=%v stderr=%q", args, stderr.String())
		}
	}
}

func TestNativePerformanceAttachSystemBaseline(t *testing.T) {
	receipt := gateRequestFixture(t).Candidate
	receipt.Schema = nativeperf.ReceiptSchemaV1
	attestation := nativePerformanceSystemBaselineFixture()
	dir := t.TempDir()
	receiptPath := filepath.Join(dir, "receipt-v1.json")
	baselinePath := filepath.Join(dir, "system-baseline.json")
	writeNativePerformanceJSONFixture(t, receiptPath, receipt)
	writeNativePerformanceJSONFixture(t, baselinePath, attestation)

	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--attach-receipt", receiptPath, "--system-baseline", baselinePath}); code != 0 {
		t.Fatalf("first append exit=%d stderr=%s", code, stderr.String())
	}
	first, err := nativeperf.DecodeReceipt(stdout.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if first.Schema != nativeperf.ReceiptSchemaV2 || len(first.SystemBaselines) != 1 {
		t.Fatalf("first=%+v", first)
	}
	originalData, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	original, err := nativeperf.DecodeReceipt(originalData)
	if err != nil || original.Schema != nativeperf.ReceiptSchemaV1 || len(original.SystemBaselines) != 0 {
		t.Fatalf("input mutated: %+v err=%v", original, err)
	}

	secondInput := filepath.Join(dir, "receipt-one.json")
	secondOutput := filepath.Join(dir, "receipt-two.json")
	if err := os.WriteFile(secondInput, stdout.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	secondAttestation := nativePerformanceSystemBaselineFixture()
	secondAttestation.Window.IntervalNS = 2
	secondAttestation.Seal()
	writeNativePerformanceJSONFixture(t, baselinePath, secondAttestation)
	stdout.Reset()
	stderr.Reset()
	if code := runNativePerformance(&stdout, &stderr, []string{"--attach-receipt", secondInput, "--system-baseline", baselinePath, "--out", secondOutput}); code != 0 || stdout.Len() != 0 {
		t.Fatalf("second append exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	secondData, err := os.ReadFile(secondOutput)
	if err != nil {
		t.Fatal(err)
	}
	second, err := nativeperf.DecodeReceipt(secondData)
	if err != nil || len(second.SystemBaselines) != 2 {
		t.Fatalf("second=%+v err=%v", second, err)
	}
	if info, err := os.Stat(secondOutput); err != nil {
		t.Fatal(err)
	} else if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("output mode=%o", info.Mode().Perm())
	}
}

func TestNativePerformanceAttachRejectsTamperAndOverfill(t *testing.T) {
	receipt := gateRequestFixture(t).Candidate
	attestation := nativePerformanceSystemBaselineFixture()
	dir := t.TempDir()
	receiptPath := filepath.Join(dir, "receipt.json")
	baselinePath := filepath.Join(dir, "system-baseline.json")
	writeNativePerformanceJSONFixture(t, receiptPath, receipt)
	attestation.CommandExitCode = 9
	writeNativePerformanceJSONFixture(t, baselinePath, attestation)
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--attach-receipt", receiptPath, "--system-baseline", baselinePath}); code != 1 || !strings.Contains(stderr.String(), "canonical digest mismatch") {
		t.Fatalf("tamper exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}

	for i := range receipt.Repetitions {
		attestation = nativePerformanceSystemBaselineFixture()
		attestation.Window.IntervalNS = int64(i + 1)
		attestation.Seal()
		var err error
		receipt, err = nativeperf.AttachSystemBaseline(nativeperf.ActiveGraph(), receipt, attestation)
		if err != nil {
			t.Fatal(err)
		}
	}
	writeNativePerformanceJSONFixture(t, receiptPath, receipt)
	overfillAttestation := nativePerformanceSystemBaselineFixture()
	overfillAttestation.Window.IntervalNS = 99
	overfillAttestation.Seal()
	writeNativePerformanceJSONFixture(t, baselinePath, overfillAttestation)
	stdout.Reset()
	stderr.Reset()
	if code := runNativePerformance(&stdout, &stderr, []string{"--attach-receipt", receiptPath, "--system-baseline", baselinePath}); code != 1 || !strings.Contains(stderr.String(), "cover every repetition") {
		t.Fatalf("overfill exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}

func nativePerformanceSystemBaselineFixture() systembaseline.Report {
	metric := systembaseline.Metric{Available: true, Value: 1, Unit: "percent", Source: "test"}
	report := systembaseline.Report{
		Schema: systembaseline.Schema, Verdict: systembaseline.VerdictClean,
		Baseline:        systembaseline.Window{StartedAtUTC: "2026-08-25T23:59:59Z", EndedAtUTC: "2026-08-26T00:00:00Z", DurationNS: 1e9, IntervalNS: 5e8, Samples: 2},
		BaselineHost:    systembaseline.HostTotals{CPUPercent: metric},
		BaselineSampler: systembaseline.SamplerOverhead{CountedSamples: 1, WallNS: 1e7, DutyPercent: metric},
		Window:          systembaseline.Window{StartedAtUTC: "2026-08-26T00:00:00Z", EndedAtUTC: "2026-08-26T00:00:01Z", DurationNS: 1e9, IntervalNS: 5e8, Samples: 2},
		CommandSampler:  systembaseline.SamplerOverhead{CountedSamples: 1, WallNS: 1e7, DutyPercent: metric},
		Coverage:        systembaseline.Coverage{SUTRootPID: 7, DescendantAttribution: "sampled_pid_ppid_tree"},
		Host:            systembaseline.HostTotals{CPUPercent: metric},
		Attribution:     systembaseline.Attribution{SUTCPUPercentOfHost: metric, NonSUTCPUPercentOfHost: metric},
		Policy:          systembaseline.DefaultPolicy(), TopNonSUT: []systembaseline.Consumer{}, Findings: []systembaseline.Finding{},
	}
	report.Seal()
	return report
}

func writeNativePerformanceJSONFixture(t *testing.T, path string, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
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
		r.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
		r.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "r1+gtest"}}
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
	if code := runNativePerformance(&stdout, &stderr, []string{"--compare", basePath, "--candidate", candidatePath}); code != 1 || !strings.Contains(stderr.String(), "fallback count must be zero") {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
}

func TestNativePerformanceProfileFixturesClassifyAndSelectNext(t *testing.T) {
	tests := []struct {
		name          string
		class         string
		lever         string
		overrideIssue int
	}{
		{name: "synthetic-metal-launch-bound.json", class: "launch-bound", lever: "metal.command-buffer-amortization"},
		{name: "synthetic-cuda-bandwidth-bound.json", class: "bandwidth-bound", lever: "cuda.q8_1-activation-quant"},
		{name: "synthetic-metal-bandwidth-override.json", class: "bandwidth-bound", lever: "metal.paged-kv", overrideIssue: 8395},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := nativePerformanceProfileFixture(test.name)
			var first, second, stderr bytes.Buffer
			if code := runNativePerformance(&first, &stderr, []string{"--profile", path}); code != 0 {
				t.Fatalf("profile exit=%d stderr=%s", code, stderr.String())
			}
			if stderr.Len() != 0 {
				t.Fatalf("profile stderr=%q", stderr.String())
			}
			stderr.Reset()
			if code := runNativePerformance(&second, &stderr, []string{"--profile", path}); code != 0 {
				t.Fatalf("second profile exit=%d stderr=%s", code, stderr.String())
			}
			if stderr.Len() != 0 {
				t.Fatalf("second profile stderr=%q", stderr.String())
			}
			if first.String() != second.String() {
				t.Fatalf("profile output changed across identical calls:\nfirst=%s\nsecond=%s", first.String(), second.String())
			}
			var classification nativeperf.BottleneckClassification
			if err := json.Unmarshal(first.Bytes(), &classification); err != nil {
				t.Fatal(err)
			}
			if classification.Schema != nativeperf.ClassificationSchema || classification.Class != test.class || classification.RecommendedLeverID != test.lever {
				t.Fatalf("classification=%+v", classification)
			}

			first.Reset()
			second.Reset()
			stderr.Reset()
			if code := runNativePerformance(&first, &stderr, []string{"--profile-next", path}); code != 0 {
				t.Fatalf("profile-next exit=%d stderr=%s", code, stderr.String())
			}
			if stderr.Len() != 0 {
				t.Fatalf("profile-next stderr=%q", stderr.String())
			}
			stderr.Reset()
			if code := runNativePerformance(&second, &stderr, []string{"--profile-next", path}); code != 0 {
				t.Fatalf("second profile-next exit=%d stderr=%s", code, stderr.String())
			}
			if stderr.Len() != 0 {
				t.Fatalf("second profile-next stderr=%q", stderr.String())
			}
			if first.String() != second.String() {
				t.Fatalf("profile-next output changed across identical calls:\nfirst=%s\nsecond=%s", first.String(), second.String())
			}
			var next struct {
				Classification nativeperf.BottleneckClassification `json:"classification"`
				Lever          nativeperf.Lever                    `json:"lever"`
				Override       *nativeperf.SelectionOverride       `json:"selection_override"`
			}
			if err := json.Unmarshal(first.Bytes(), &next); err != nil {
				t.Fatal(err)
			}
			if next.Classification.RecommendedLeverID != test.lever || next.Lever.ID != test.lever {
				t.Fatalf("profile-next=%+v", next)
			}
			if test.overrideIssue > 0 {
				if next.Override == nil || next.Override.IssueNumber != test.overrideIssue || strings.TrimSpace(next.Override.Reason) == "" {
					t.Fatalf("profile-next lost override provenance: %+v", next)
				}
			} else if next.Override != nil {
				t.Fatalf("profile-next unexpectedly emitted override: %+v", next.Override)
			}
		})
	}
}

func TestNativePerformanceProfileRejectionFixtures(t *testing.T) {
	tests := []struct {
		name string
		mode string
		want string
	}{
		{name: "reject-forward-path-envelope-mismatch.json", mode: "--profile", want: "forward_path must exactly match"},
		{name: "reject-mixed-backend-counters.json", mode: "--profile", want: "only Metal counters"},
		{name: "reject-mixed-envelope-lever.json", mode: "--profile", want: "mixes envelope"},
		{name: "reject-missing-phase.json", mode: "--profile", want: "every ordered phase"},
		{name: "reject-overlapping-phases.json", mode: "--profile", want: "overlaps the previous phase"},
		{name: "reject-non-finite-counter.json", mode: "--profile", want: "cannot unmarshal number"},
		{name: "reject-negative-counter.json", mode: "--profile", want: "finite and non-negative"},
		{name: "reject-invalid-native-identity.json", mode: "--profile", want: "fak-native execution identity"},
		{name: "reject-invalid-fallback-identity.json", mode: "--profile", want: "zero fallback"},
		{name: "reject-missing-attribution-state.json", mode: "--profile", want: "typed unavailable reason"},
		{name: "reject-missing-counter.json", mode: "--profile", want: "missing required field"},
		{name: "reject-unknown-lever.json", mode: "--profile", want: "unknown lever"},
		{name: "reject-mixed-levers.json", mode: "--profile", want: "mixes lever"},
		{name: "reject-unsupported-counter-comparison.json", mode: "--profile", want: "counter comparisons are unsupported"},
		{name: "reject-profile-next-contradiction-without-override.json", mode: "--profile-next", want: "issue-backed reason"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runNativePerformance(&stdout, &stderr, []string{test.mode, nativePerformanceProfileFixture(test.name)}); code != 1 {
				t.Fatalf("exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
			}
			if stdout.Len() != 0 {
				t.Fatalf("failure wrote stdout=%q", stdout.String())
			}
			if !strings.Contains(stderr.String(), test.want) {
				t.Fatalf("stderr=%q, want substring %q", stderr.String(), test.want)
			}
		})
	}
}

func TestNativePerformanceProfileReadAndDecodeErrors(t *testing.T) {
	tests := []struct {
		name string
		path string
		want string
	}{
		{name: "missing", path: filepath.Join(t.TempDir(), "missing.json"), want: "read profile"},
		{name: "malformed", path: nativePerformanceProfileFixture("reject-non-finite-counter.json"), want: "decode profile"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runNativePerformance(&stdout, &stderr, []string{"--profile", test.path}); code != 1 {
				t.Fatalf("exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
			}
			if stdout.Len() != 0 || !strings.Contains(stderr.String(), test.want) {
				t.Fatalf("stdout=%q stderr=%q want=%q", stdout.String(), stderr.String(), test.want)
			}
		})
	}
}

func nativePerformanceProfileFixture(name string) string {
	return filepath.Join("..", "..", "internal", "nativeperf", "testdata", "native-performance-profile", name)
}

func TestNativePerformanceGateExitCodes(t *testing.T) {
	request := gateRequestFixture(t)
	dir := t.TempDir()
	path := dir + "/gate.json"
	write := func() {
		data, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write()
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--gate", path}); code != 0 {
		t.Fatalf("pass exit=%d stderr=%s", code, stderr.String())
	}
	var verdict nativeperf.GateVerdict
	if err := json.Unmarshal(stdout.Bytes(), &verdict); err != nil || verdict.Classification != nativeperf.GatePass {
		t.Fatalf("verdict=%+v err=%v", verdict, err)
	}
	for i := range request.Candidate.Repetitions {
		request.Candidate.Repetitions[i].TokensPerSecond = 80
	}
	write()
	stdout.Reset()
	stderr.Reset()
	if code := runNativePerformance(&stdout, &stderr, []string{"--gate", path}); code != 3 {
		t.Fatalf("regression exit=%d stderr=%s", code, stderr.String())
	}
	if err := json.Unmarshal(stdout.Bytes(), &verdict); err != nil || verdict.Bisect == nil {
		t.Fatalf("verdict=%+v err=%v", verdict, err)
	}
}

func gateRequestFixture(t *testing.T) nativeperf.GateRequest {
	t.Helper()
	baseline, err := nativeperf.BaselineTemplate(nativeperf.ActiveGraph(), "metal.command-buffer-amortization")
	if err != nil {
		t.Fatal(err)
	}
	fill := func(r *nativeperf.ExperimentReceipt, role, revision string) {
		r.Role, r.Revision, r.Machine.ScrubbedID = role, revision, "lab-class-a"
		r.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
		r.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
		r.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "r1+g" + revision}}
		r.Commands = []string{"fak native-run --receipt-out receipt.json"}
		r.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
		for i := range r.Repetitions {
			r.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: 100}
		}
		if role == nativeperf.RoleCandidate {
			r.ChangedAxes = []string{"lever:" + r.ChangedLeverID}
		}
	}
	candidate := baseline
	candidate.UnchangedControls = append([]string(nil), baseline.UnchangedControls...)
	candidate.Repetitions = append([]nativeperf.Repetition(nil), baseline.Repetitions...)
	fill(&baseline, nativeperf.RoleBaseline, "good")
	fill(&candidate, nativeperf.RoleCandidate, "bad")
	return nativeperf.GateRequest{Schema: nativeperf.GateRequestSchema, Policy: nativeperf.GatePolicy{Schema: "fak-native-performance-gate-policy/v1", EnvelopeID: baseline.EnvelopeID, ChangedLeverID: baseline.ChangedLeverID, AcceptedRevision: baseline.Revision, MinimumRepetitions: 3, MaximumNoisePercent: 2, InvestigateDropPercent: 2, RegressionDropPercent: 5, MinimumThroughput: 90, MaximumPeakBytes: 1200, QualityMetric: "exact_match", MinimumQualityScore: 1, QualityHigherIsBetter: true, RequiredEngine: "fak-native", RequiredForwardPath: baseline.Execution.ForwardPath}, LastAccepted: baseline, Candidate: candidate}
}
