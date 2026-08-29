package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func TestCompareProfilesUsesControlMedianForEveryCandidate(t *testing.T) {
	// Paired-row comparison would admit the 105 ms candidate against its 110 ms
	// control. The campaign contract compares it with the 100 ms control median.
	r := compareProfiles([]float64{90, 100, 110}, []float64{80, 105, 70})
	if r.Verdict != "REJECT" || r.EveryCandidateBelowControlMedian {
		t.Fatalf("comparison = %+v, want REJECT against control median", r)
	}
	if got := compareProfiles([]float64{90}, []float64{70}); got.Verdict != "HOLD" {
		t.Fatalf("one-repetition campaign = %+v, want HOLD", got)
	}
}

func TestCompareProfilesUsesFixedFifteenPercentGate(t *testing.T) {
	got := compareProfiles([]float64{98, 100, 102}, []float64{85, 86, 87})
	if got.MinimumMedianImprovementPercent != 15 || got.MedianImprovementPercent < 13.99 || got.MedianImprovementPercent > 14.01 || got.Verdict != "REJECT" {
		t.Fatalf("comparison = %+v, want fixed 15%% gate to reject 14%%", got)
	}
}

func TestNativeProfileForwardPathRequiresExecutedSelectorRoute(t *testing.T) {
	const hybrid = "metal/qwen35-hybrid-session-v1"
	tests := []struct {
		name     string
		selector string
		executed bool
		want     string
		wantErr  bool
	}{
		{name: "OFF hybrid", selector: nativeProfileSelectorOff, want: hybrid},
		{name: "ON sequence", selector: nativeProfileSelectorOn, executed: true, want: model.Qwen35MetalGDNSequenceForwardPath},
		{name: "OFF contamination", selector: nativeProfileSelectorOff, executed: true, wantErr: true},
		{name: "ON did not execute", selector: nativeProfileSelectorOn, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := nativeProfileExecutedForwardPath(hybrid, test.selector, test.executed)
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("route = %q, err = %v, want %q, error=%t", got, err, test.want, test.wantErr)
			}
		})
	}
}

func TestNativeProfileCampaignKeepRejectAndDeterministicJSON(t *testing.T) {
	paths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{80, 82, 81})
	spec := strings.Join(paths, ",")
	keep := compareNativeProfileCampaign(spec)
	if keep.Verdict != "KEEP" || keep.ControlMedianMilliseconds != 100 || keep.CandidateMedianMilliseconds != 81 || keep.EnvelopeID == "" {
		t.Fatalf("valid campaign = %+v", keep)
	}
	one, err := json.Marshal(keep)
	if err != nil {
		t.Fatal(err)
	}
	two, err := json.Marshal(compareNativeProfileCampaign(spec))
	if err != nil {
		t.Fatal(err)
	}
	if string(one) != string(two) {
		t.Fatalf("comparison JSON is not deterministic:\n%s\n%s", one, two)
	}

	rejectPaths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{90, 91, 92})
	if got := compareNativeProfileCampaign(strings.Join(rejectPaths, ",")); got.Verdict != "REJECT" {
		t.Fatalf("below-threshold campaign = %+v, want REJECT", got)
	}
}

func TestWriteProfileComparisonStampsCanonicalLineageReceipt(t *testing.T) {
	t.Setenv("FAK_BENCH_UTC", "2026-08-29T12:34:56Z")
	t.Setenv("FAK_BENCH_COMMIT", strings.Repeat("a", 40))
	t.Setenv("FAK_BENCH_NODE", "modelbench-test-node")
	t.Setenv("FAK_BENCH_RUN_ID", "profile-comparison-test")
	t.Setenv("FAK_BENCH_HARNESS_NAME", "modelbench")
	t.Setenv("FAK_BENCH_ARTIFACT", "")

	out := filepath.Join(t.TempDir(), "profile-comparison.json")
	f := testCompleteBenchFlags()
	*f.out = out
	want := compareProfiles([]float64{100, 102, 98}, []float64{80, 82, 81})
	if err := writeProfileComparison(f, want); err != nil {
		t.Fatalf("writeProfileComparison: %v", err)
	}

	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Schema  string `json:"schema"`
		Verdict string `json:"verdict"`
		Lineage struct {
			Schema    string `json:"lineage_schema"`
			Commit    string `json:"git_commit"`
			Node      string `json:"node"`
			Timestamp string `json:"utc"`
		} `json:"lineage"`
		Artifact struct {
			Schema  string `json:"schema"`
			RunID   string `json:"run_id"`
			Lineage struct {
				SourceArtifact string `json:"source_artifact"`
			} `json:"lineage"`
		} `json:"benchmark_artifact"`
	}
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("decode stamped comparison: %v\n%s", err, b)
	}
	if got.Schema != profileComparisonSchema || got.Verdict != want.Verdict {
		t.Fatalf("comparison payload changed: schema=%q verdict=%q", got.Schema, got.Verdict)
	}
	if got.Lineage.Schema != "fak-bench-lineage/1" ||
		got.Lineage.Commit != strings.Repeat("a", 40) ||
		got.Lineage.Node != "modelbench-test-node" ||
		got.Lineage.Timestamp != "2026-08-29T12:34:56Z" {
		t.Fatalf("lineage = %+v, want canonical fixed test stamp", got.Lineage)
	}
	if got.Artifact.Schema != "fak-benchmark-artifact/1" ||
		got.Artifact.RunID != "profile-comparison-test" ||
		got.Artifact.Lineage.SourceArtifact != filepath.ToSlash(out) {
		t.Fatalf("benchmark artifact receipt = %+v, want source %q", got.Artifact, filepath.ToSlash(out))
	}
}

func TestNativeProfileCampaignInvalidEvidenceHolds(t *testing.T) {
	tests := []struct {
		name   string
		edit   func(t *testing.T, paths []string)
		reason string
	}{
		{name: "wrong schema", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[0], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) { p.Schema = "invented/v1" })
		}, reason: "schema"},
		{name: "missing companion", edit: func(t *testing.T, paths []string) {
			if err := os.Remove(nativeReceiptPath(paths[0])); err != nil {
				t.Fatal(err)
			}
		}, reason: "companion"},
		{name: "mixed identity", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[4], func(_ *nativeperf.ProfileBundle, r *nativeProfileReceipt) {
				r.Source.Revision = strings.Repeat("c", 40)
			})
		}, reason: "identity"},
		{name: "wrong order", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[1], func(_ *nativeperf.ProfileBundle, r *nativeProfileReceipt) {
				r.Controls[nativeProfileSequenceSelector] = nativeProfileSelectorOn
			})
		}, reason: "selector"},
		{name: "reused bound capture", edit: func(t *testing.T, paths []string) {
			paths[1] = paths[0]
		}, reason: "reuses bound receipt"},
		{name: "reused profile capture", edit: func(t *testing.T, paths []string) {
			profileBytes, err := os.ReadFile(paths[0])
			if err != nil {
				t.Fatal(err)
			}
			var reused nativeperf.ProfileBundle
			if err := json.Unmarshal(profileBytes, &reused); err != nil {
				t.Fatal(err)
			}
			rewriteComparisonPair(t, paths[1], func(p *nativeperf.ProfileBundle, r *nativeProfileReceipt) {
				*p = reused
				r.Source.Revision = strings.Repeat("d", 40)
			})
		}, reason: "reuses profile"},
		{name: "missing selector", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[0], func(_ *nativeperf.ProfileBundle, r *nativeProfileReceipt) {
				delete(r.Controls, nativeProfileSequenceSelector)
			})
		}, reason: "selector"},
		{name: "ON selector with hybrid route", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[3], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) {
				p.Execution.ForwardPath = "metal/qwen35-hybrid-session-v1"
			})
		}, reason: "forward path"},
		{name: "OFF selector with candidate route", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[0], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) {
				p.Execution.ForwardPath = model.Qwen35MetalGDNSequenceForwardPath
			})
		}, reason: "forward path"},
		{name: "selector contamination", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[5], func(_ *nativeperf.ProfileBundle, r *nativeProfileReceipt) {
				r.Controls[nativeControlLogicalCPUs] = "999"
			})
		}, reason: "identity"},
		{name: "fallback", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[3], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) { p.Execution.FallbackCount = 1 })
		}, reason: "fallback"},
		{name: "invalid phase", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[2], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) { p.Phases[1].DurationMilliseconds = 0 })
		}, reason: "phase"},
		{name: "phase order", edit: func(t *testing.T, paths []string) {
			rewriteComparisonPair(t, paths[2], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) {
				p.Phases[1].Name, p.Phases[2].Name = p.Phases[2].Name, p.Phases[1].Name
			})
		}, reason: "phase"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			paths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{80, 82, 81})
			test.edit(t, paths)
			got := compareNativeProfileCampaign(strings.Join(paths, ","))
			if got.Verdict != "HOLD" || !strings.Contains(strings.ToLower(got.Reason), test.reason) {
				t.Fatalf("comparison = %+v, want HOLD reason containing %q", got, test.reason)
			}
		})
	}
}

func TestNativeProfileCampaignRequiresExactCardinality(t *testing.T) {
	paths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{80, 82, 81})
	got := compareNativeProfileCampaign(strings.Join(paths[:5], ","))
	if got.Verdict != "HOLD" || !strings.Contains(got.Reason, "exactly 6") {
		t.Fatalf("five-pair campaign = %+v, want cardinality HOLD", got)
	}
}

func writeComparisonCampaign(t *testing.T, control, candidate []float64) []string {
	t.Helper()
	if len(control) != nativeProfileArmRuns || len(candidate) != nativeProfileArmRuns {
		t.Fatal("test campaign requires 3+3 durations")
	}
	dir := t.TempDir()
	paths := make([]string, 0, nativeProfileCampaignRuns)
	for i, duration := range append(append([]float64(nil), control...), candidate...) {
		_, profile, receipt := testNativeProfileReceipt(t)
		setComparisonPrefillDuration(&profile, duration)
		selector := nativeProfileSelectorOff
		if i >= nativeProfileArmRuns {
			selector = nativeProfileSelectorOn
			profile.Execution.ForwardPath = model.Qwen35MetalGDNSequenceForwardPath
		}
		receipt.Controls[nativeProfileSequenceSelector] = selector
		path := filepath.Join(dir, fmt.Sprintf("run-%d.profile.json", i+1))
		writeComparisonPair(t, path, profile, receipt)
		paths = append(paths, path)
	}
	return paths
}

func setComparisonPrefillDuration(profile *nativeperf.ProfileBundle, duration float64) {
	profile.Phases[1].DurationMilliseconds = duration
	start := 0.0
	for i := range profile.Phases {
		profile.Phases[i].StartMilliseconds = start
		start += profile.Phases[i].DurationMilliseconds
	}
}

func rewriteComparisonPair(t *testing.T, path string, edit func(*nativeperf.ProfileBundle, *nativeProfileReceipt)) {
	t.Helper()
	profileBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var profile nativeperf.ProfileBundle
	if err := json.Unmarshal(profileBytes, &profile); err != nil {
		t.Fatal(err)
	}
	receiptBytes, err := os.ReadFile(nativeReceiptPath(path))
	if err != nil {
		t.Fatal(err)
	}
	var receipt nativeProfileReceipt
	if err := json.Unmarshal(receiptBytes, &receipt); err != nil {
		t.Fatal(err)
	}
	edit(&profile, &receipt)
	writeComparisonPair(t, path, profile, receipt)
}

func writeComparisonPair(t *testing.T, path string, profile nativeperf.ProfileBundle, receipt nativeProfileReceipt) {
	t.Helper()
	profileBytes, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	profileBytes = append(profileBytes, '\n')
	profileSHA := sha256.Sum256(profileBytes)
	receipt.ProfileSHA256 = fmt.Sprintf("%x", profileSHA)
	receipt.EnvelopeID = profile.EnvelopeID
	receipt.BindingSHA256, err = nativeReceiptBinding(receipt)
	if err != nil {
		t.Fatal(err)
	}
	receiptBytes, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	receiptBytes = append(receiptBytes, '\n')
	if err := os.WriteFile(path, profileBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(nativeReceiptPath(path), receiptBytes, 0o600); err != nil {
		t.Fatal(err)
	}
}
