package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func TestProfileWelchSignificanceWitnesses(t *testing.T) {
	t.Run("equal distributions hold", func(t *testing.T) {
		got := compareProfiles([]float64{99, 100, 101}, []float64{99, 100, 101}).WelchSignificance
		if got.Verdict != "HOLD" || got.Direction != "none" || got.PValue != 1 {
			t.Fatalf("Welch significance = %+v, want HOLD/not significant", got)
		}
	})

	t.Run("clear speedup is significant in the correct direction", func(t *testing.T) {
		control := []float64{90, 100, 110}
		candidate := []float64{60, 70, 80}
		comparison := compareProfiles(control, candidate)
		got := comparison.WelchSignificance
		wantP := oracleWelchTwoSidedP(t, control, candidate)
		if comparison.Verdict != "KEEP" || got.Verdict != "SIGNIFICANT" || got.Direction != "candidate-faster" ||
			got.Alpha != nativeProfileWelchAlpha || math.Abs(got.PValue-wantP) > 1e-10 {
			t.Fatalf("Welch significance = %+v, want significant candidate-faster with p=%g", got, wantP)
		}
	})

	t.Run("high variance median gain remains advisory", func(t *testing.T) {
		got := compareProfiles([]float64{100, 101, 300}, []float64{70, 80, 90})
		if got.Verdict != "KEEP" {
			t.Fatalf("authoritative comparison verdict = %q, want unchanged KEEP", got.Verdict)
		}
		if got.WelchSignificance.Verdict != "HOLD" || got.WelchSignificance.Direction != "none" || got.WelchSignificance.PValue <= nativeProfileWelchAlpha {
			t.Fatalf("Welch significance = %+v, want HOLD/not significant", got.WelchSignificance)
		}
	})

	t.Run("insufficient samples fail closed", func(t *testing.T) {
		got := welchSignificance([]float64{100}, []float64{80})
		if got.Verdict != "HOLD" || got.Direction != "none" || !strings.Contains(got.Reason, "at least two") {
			t.Fatalf("Welch significance = %+v, want safe insufficient-sample HOLD", got)
		}
	})

	t.Run("invalid samples fail closed", func(t *testing.T) {
		got := welchSignificance([]float64{99, math.NaN(), 101}, []float64{70, 80, 90})
		if got.Verdict != "HOLD" || got.Direction != "none" || got.PValue != 1 || !strings.Contains(got.Reason, "finite and positive") {
			t.Fatalf("Welch significance = %+v, want safe invalid-sample HOLD", got)
		}
	})

	t.Run("degenerate variance fails closed", func(t *testing.T) {
		got := welchSignificance([]float64{100, 100, 100}, []float64{80, 80, 80})
		if got.Verdict != "HOLD" || got.Direction != "none" || !strings.Contains(got.Reason, "variance") {
			t.Fatalf("Welch significance = %+v, want safe degenerate-variance HOLD", got)
		}
	})
}

func TestCompareProfilesUsesControlMedianForEveryCandidate(t *testing.T) {
	// Paired-row comparison would admit the 105 ms candidate against its 110 ms
	// control. The campaign contract compares it with the 100 ms control median.
	r := compareProfiles([]float64{90, 100, 110}, []float64{80, 105, 70})
	if r.Verdict != "REJECT" || r.EveryCandidateBelowControlMedian || r.Phase != profileComparisonPhasePrefill ||
		len(r.ControlPhaseMilliseconds) != 0 || len(r.CandidatePhaseMilliseconds) != 0 {
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

func TestNativeProfileComparisonSelectsSteadyDecode(t *testing.T) {
	paths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{100, 102, 98})
	setComparisonCampaignPhaseDurations(t, paths, "steady-decode", []float64{200, 202, 198}, []float64{150, 152, 151})

	got := compareNativeProfileCampaignPhase(strings.Join(paths, ","), profileComparisonPhaseSteadyDecode)
	if got.Verdict != "KEEP" || got.Phase != profileComparisonPhaseSteadyDecode ||
		!reflect.DeepEqual(got.ControlPhaseMilliseconds, []float64{200, 202, 198}) ||
		!reflect.DeepEqual(got.CandidatePhaseMilliseconds, []float64{150, 152, 151}) ||
		len(got.ControlPrefillMilliseconds) != 0 || len(got.CandidatePrefillMilliseconds) != 0 {
		t.Fatalf("steady-decode comparison = %+v, want typed KEEP", got)
	}
}

func TestNativeProfileComparisonM3DecodeHandoffRequiresExactRoutes(t *testing.T) {
	writeCampaign := func(t *testing.T) []string {
		paths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{100, 102, 98})
		setComparisonCampaignPhaseDurations(t, paths, "steady-decode", []float64{200, 202, 198}, []float64{150, 152, 151})
		for i, path := range paths {
			rewriteComparisonPair(t, path, func(p *nativeperf.ProfileBundle, r *nativeProfileReceipt) {
				p.Execution.ForwardPath = model.Qwen35MetalGDNSequenceForwardPath
				r.Controls[nativeProfileSequenceSelector] = nativeProfileSelectorOn
				mode := model.Qwen35DecodeHandoffControl
				handoff := model.Qwen35DecodeHandoffReceipt{Mode: mode, ResidentGDNAcceptedCalls: 48 * 64}
				if i >= nativeProfileArmRuns {
					mode = model.Qwen35DecodeHandoffMixer
					handoff = model.Qwen35DecodeHandoffReceipt{Mode: mode, MixerAcceptedCalls: 48 * 64}
				}
				r.Controls[nativeProfileDecodeHandoffControl] = mode.String()
				r.Qwen35DecodeHandoff = &handoff
			})
		}
		return paths
	}
	paths := writeCampaign(t)

	got := compareNativeProfileCampaignPhaseAxis(strings.Join(paths, ","), profileComparisonPhaseSteadyDecode, profileComparisonAxisM3DecodeHandoff)
	if got.Verdict != "KEEP" || got.Selector != nativeProfileDecodeHandoffControl ||
		got.ControlSelector != model.Qwen35DecodeHandoffControl.String() || got.CandidateSelector != model.Qwen35DecodeHandoffMixer.String() {
		t.Fatalf("M3 decode-handoff comparison = %+v, want typed KEEP", got)
	}

	tests := []struct {
		name   string
		pair   int
		edit   func(*nativeProfileReceipt)
		reason string
	}{
		{name: "sequence off", pair: 0, edit: func(r *nativeProfileReceipt) { r.Controls[nativeProfileSequenceSelector] = nativeProfileSelectorOff }, reason: "sequence"},
		{name: "control mixer call", pair: 1, edit: func(r *nativeProfileReceipt) { r.Qwen35DecodeHandoff.MixerAcceptedCalls = 1 }, reason: "handoff"},
		{name: "mixer block call", pair: 3, edit: func(r *nativeProfileReceipt) { r.Qwen35DecodeHandoff.BlockAcceptedCalls = 1 }, reason: "handoff"},
		{name: "mixer missing calls", pair: 4, edit: func(r *nativeProfileReceipt) { r.Qwen35DecodeHandoff.MixerAcceptedCalls = 0 }, reason: "handoff"},
		{name: "wrong order", pair: 2, edit: func(r *nativeProfileReceipt) {
			r.Controls[nativeProfileDecodeHandoffControl] = model.Qwen35DecodeHandoffMixer.String()
			r.Qwen35DecodeHandoff = &model.Qwen35DecodeHandoffReceipt{Mode: model.Qwen35DecodeHandoffMixer, MixerAcceptedCalls: 1}
		}, reason: "selector"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			bad := writeCampaign(t)
			rewriteComparisonPair(t, bad[test.pair], func(_ *nativeperf.ProfileBundle, r *nativeProfileReceipt) { test.edit(r) })
			got := compareNativeProfileCampaignPhaseAxis(strings.Join(bad, ","), profileComparisonPhaseSteadyDecode, profileComparisonAxisM3DecodeHandoff)
			if got.Verdict != "HOLD" || !strings.Contains(strings.ToLower(got.Reason), test.reason) {
				t.Fatalf("M3 mismatch = %+v, want HOLD containing %q", got, test.reason)
			}
		})
	}
}

func TestNativeProfileComparisonSelectsDecodeEndToEnd(t *testing.T) {
	paths := writeComparisonCampaign(t, []float64{100, 101, 102}, []float64{80, 81, 82})
	setComparisonCampaignPhaseDurations(t, paths, "steady-decode", []float64{200, 202, 204}, []float64{120, 121, 122})

	got := compareNativeProfileCampaignPhase(strings.Join(paths, ","), profileComparisonPhaseEndToEnd)
	if got.Verdict != "KEEP" || got.Phase != profileComparisonPhaseEndToEnd {
		t.Fatalf("end-to-end comparison = %+v, want typed KEEP", got)
	}
	// The four other canonical phases are 1 ms each, so the full wall includes
	// load setup, first-token, verification, and teardown around the two arms.
	if !reflect.DeepEqual(got.ControlPhaseMilliseconds, []float64{304, 307, 310}) ||
		!reflect.DeepEqual(got.CandidatePhaseMilliseconds, []float64{204, 206, 208}) {
		t.Fatalf("end-to-end durations = control %v candidate %v", got.ControlPhaseMilliseconds, got.CandidatePhaseMilliseconds)
	}
}

func TestNativeProfileComparisonDecodeEndToEndRequiresContiguousCanonicalWall(t *testing.T) {
	paths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{80, 82, 81})
	rewriteComparisonPair(t, paths[1], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) {
		for i := 3; i < len(p.Phases); i++ {
			p.Phases[i].StartMilliseconds++
		}
	})

	got := compareNativeProfileCampaignPhase(strings.Join(paths, ","), profileComparisonPhaseEndToEnd)
	if got.Verdict != "HOLD" || got.Phase != profileComparisonPhaseEndToEnd || !strings.Contains(got.Reason, "contiguous") {
		t.Fatalf("non-contiguous end-to-end comparison = %+v, want typed HOLD", got)
	}
}

func TestNativeProfileComparisonDecodePhaseFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		edit func(*nativeperf.ProfileBundle)
	}{
		{name: "missing", edit: func(p *nativeperf.ProfileBundle) { p.Phases = p.Phases[:3] }},
		{name: "duplicate", edit: func(p *nativeperf.ProfileBundle) { p.Phases[0].Name = "steady-decode" }},
		{name: "mismatched", edit: func(p *nativeperf.ProfileBundle) { p.Phases[3].Name = "decode" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			paths := writeComparisonCampaign(t, []float64{100, 102, 98}, []float64{80, 82, 81})
			rewriteComparisonPair(t, paths[2], func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) { test.edit(p) })
			got := compareNativeProfileCampaignPhase(strings.Join(paths, ","), profileComparisonPhaseSteadyDecode)
			if got.Verdict != "HOLD" || got.Phase != profileComparisonPhaseSteadyDecode || !strings.Contains(strings.ToLower(got.Reason), "phase") {
				t.Fatalf("%s selected phase = %+v, want typed HOLD phase reason", test.name, got)
			}
		})
	}

	got := compareNativeProfileCampaignPhase("unused", profileComparisonPhase("decode"))
	if got.Verdict != "HOLD" || !strings.Contains(got.Reason, "invalid") {
		t.Fatalf("invalid typed selector = %+v, want HOLD", got)
	}
}

func TestProfileComparisonDecodePhaseFlagIsTyped(t *testing.T) {
	phase := profileComparisonPhasePrefill
	if err := phase.Set("steady-decode"); err != nil || phase != profileComparisonPhaseSteadyDecode {
		t.Fatalf("set steady-decode = %q, err=%v", phase, err)
	}
	if err := phase.Set("position-3"); err == nil || phase != profileComparisonPhaseSteadyDecode {
		t.Fatalf("invalid phase changed selection to %q, err=%v", phase, err)
	}
	axis := profileComparisonAxisSequence
	if err := axis.Set("m3-decode-handoff"); err != nil || axis != profileComparisonAxisM3DecodeHandoff {
		t.Fatalf("set M3 axis = %q, err=%v", axis, err)
	}
	if err := axis.Set("mixed"); err == nil || axis != profileComparisonAxisM3DecodeHandoff {
		t.Fatalf("invalid axis changed selection to %q, err=%v", axis, err)
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

// oracleWelchTwoSidedP intentionally uses direct numerical integration of the
// Student-t density rather than the implementation's incomplete-beta path.
func oracleWelchTwoSidedP(t *testing.T, control, candidate []float64) float64 {
	t.Helper()
	meanVariance := func(xs []float64) (float64, float64) {
		var sum float64
		for _, x := range xs {
			sum += x
		}
		mean := sum / float64(len(xs))
		var squared float64
		for _, x := range xs {
			delta := x - mean
			squared += delta * delta
		}
		return mean, squared / float64(len(xs)-1)
	}
	controlMean, controlVariance := meanVariance(control)
	candidateMean, candidateVariance := meanVariance(candidate)
	controlTerm := controlVariance / float64(len(control))
	candidateMeanVariance := candidateVariance / float64(len(candidate))
	tStatistic := math.Abs(controlMean-candidateMean) / math.Sqrt(controlTerm+candidateMeanVariance)
	degreesFreedom := (controlTerm + candidateMeanVariance) * (controlTerm + candidateMeanVariance) /
		(controlTerm*controlTerm/float64(len(control)-1) + candidateMeanVariance*candidateMeanVariance/float64(len(candidate)-1))

	logCoefficient, _ := math.Lgamma((degreesFreedom + 1) / 2)
	denominator, _ := math.Lgamma(degreesFreedom / 2)
	coefficient := math.Exp(logCoefficient-denominator) / math.Sqrt(degreesFreedom*math.Pi)
	density := func(x float64) float64 {
		return coefficient * math.Pow(1+x*x/degreesFreedom, -(degreesFreedom+1)/2)
	}
	const intervals = 20000 // even, deterministic, and ample for this bounded fixture.
	width := tStatistic / intervals
	sum := density(0) + density(tStatistic)
	for i := 1; i < intervals; i++ {
		weight := 2.0
		if i%2 == 1 {
			weight = 4
		}
		sum += weight * density(float64(i)*width)
	}
	p := 1 - 2*sum*width/3
	if p < 0 || p > 1 || math.IsNaN(p) {
		t.Fatalf("oracle produced invalid p-value %g", p)
	}
	return p
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
	setComparisonPhaseDuration(profile, "prefill", duration)
}

func setComparisonCampaignPhaseDurations(t *testing.T, paths []string, phase string, control, candidate []float64) {
	t.Helper()
	durations := append(append([]float64(nil), control...), candidate...)
	if len(paths) != nativeProfileCampaignRuns || len(durations) != nativeProfileCampaignRuns {
		t.Fatal("test campaign phase edit requires 6 paths and 3+3 durations")
	}
	for i, path := range paths {
		rewriteComparisonPair(t, path, func(p *nativeperf.ProfileBundle, _ *nativeProfileReceipt) {
			setComparisonPhaseDuration(p, phase, durations[i])
		})
	}
}

func setComparisonPhaseDuration(profile *nativeperf.ProfileBundle, phaseName string, duration float64) {
	for i := range profile.Phases {
		if profile.Phases[i].Name == phaseName {
			profile.Phases[i].DurationMilliseconds = duration
		}
	}
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
