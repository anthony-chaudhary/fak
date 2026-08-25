package nativeperf

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const profileFixtureDir = "testdata/native-performance-profile"

func loadProfileFixture(t *testing.T, name string) ProfileBundle {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(profileFixtureDir, name))
	if err != nil {
		t.Fatal(err)
	}
	profile, err := DecodeProfile(data)
	if err != nil {
		t.Fatalf("decode %s: %v", name, err)
	}
	return profile
}

func TestSyntheticProfileFixturesClassifyDeterministically(t *testing.T) {
	graph := ActiveGraph()
	tests := []struct {
		name          string
		class         string
		confidence    string
		lever         string
		backend       string
		counterAssert func(*testing.T, ProfileBundle)
	}{
		{
			name:       "synthetic-metal-launch-bound.json",
			class:      "launch-bound",
			confidence: "medium",
			lever:      "metal.command-buffer-amortization",
			backend:    "metal",
			counterAssert: func(t *testing.T, profile ProfileBundle) {
				t.Helper()
				if profile.Metal == nil || profile.CUDA != nil || profile.Metal.CommandBuffers != 40 || profile.Metal.WorkingSetBytes != 24000000000 {
					t.Fatalf("Metal counters were not preserved: %+v", profile)
				}
			},
		},
		{
			name:       "synthetic-cuda-bandwidth-bound.json",
			class:      "bandwidth-bound",
			confidence: "high",
			lever:      "cuda.q8_1-activation-quant",
			backend:    "cuda",
			counterAssert: func(t *testing.T, profile ProfileBundle) {
				t.Helper()
				if profile.CUDA == nil || profile.Metal != nil || profile.CUDA.AchievedBandwidthGBS != 800 || profile.CUDA.PeakComputeTFLOPS != 100 {
					t.Fatalf("CUDA counters were not preserved: %+v", profile)
				}
			},
		},
		{
			name:       "synthetic-metal-bandwidth-override.json",
			class:      "bandwidth-bound",
			confidence: "medium",
			lever:      "metal.paged-kv",
			backend:    "metal-bandwidth",
			counterAssert: func(t *testing.T, profile ProfileBundle) {
				t.Helper()
				if profile.Metal == nil || profile.CUDA != nil || profile.Metal.CommandBuffers != 12 || profile.Metal.WorkingSetBytes <= profile.Metal.ResidentBytes {
					t.Fatalf("Metal bandwidth counters were not preserved: %+v", profile)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.backend, func(t *testing.T) {
			profile := loadProfileFixture(t, test.name)
			if err := ValidateProfile(graph, profile); err != nil {
				t.Fatal(err)
			}
			test.counterAssert(t, profile)

			first, err := ClassifyProfile(graph, profile)
			if err != nil {
				t.Fatal(err)
			}
			second, err := ClassifyProfile(graph, profile)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(first, second) {
				t.Fatalf("classification drift:\nfirst=%+v\nsecond=%+v", first, second)
			}
			if first.Schema != ClassificationSchema || first.Class != test.class || first.Confidence != test.confidence || first.RecommendedLeverID != test.lever {
				t.Fatalf("classification=%+v", first)
			}
		})
	}
}

func TestProfileRejectionFixturesFailClosed(t *testing.T) {
	graph := ActiveGraph()
	tests := []struct {
		name      string
		decodeErr bool
		want      string
	}{
		{name: "reject-forward-path-envelope-mismatch.json", want: "forward_path must exactly match"},
		{name: "reject-missing-phase.json", want: "every ordered phase"},
		{name: "reject-overlapping-phases.json", want: "overlaps the previous phase"},
		{name: "reject-non-finite-counter.json", decodeErr: true, want: "cannot unmarshal number"},
		{name: "reject-negative-counter.json", want: "finite and non-negative"},
		{name: "reject-invalid-native-identity.json", want: "fak-native execution identity"},
		{name: "reject-invalid-fallback-identity.json", want: "zero fallback"},
		{name: "reject-mixed-backend-counters.json", want: "only Metal counters"},
		{name: "reject-missing-attribution-state.json", want: "typed unavailable reason"},
		{name: "reject-unknown-lever.json", want: "unknown lever"},
		{name: "reject-mixed-envelope-lever.json", want: "mixes envelope"},
		{name: "reject-mixed-levers.json", want: "mixes lever"},
		{name: "reject-unsupported-counter-comparison.json", want: "counter comparisons are unsupported"},
		{name: "reject-missing-counter.json", decodeErr: true, want: `metal is missing required field "wait_milliseconds"`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(profileFixtureDir, test.name))
			if err != nil {
				t.Fatal(err)
			}
			profile, err := DecodeProfile(data)
			if test.decodeErr {
				if err == nil || !strings.Contains(err.Error(), test.want) {
					t.Fatalf("decode err=%v, want substring %q", err, test.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			err = ValidateProfile(graph, profile)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validate err=%v, want substring %q", err, test.want)
			}
		})
	}
}

func TestProfilePhaseNumbersRejectNonFiniteNegativeAndOverlap(t *testing.T) {
	graph := ActiveGraph()
	tests := []struct {
		name string
		edit func(*ProfileBundle)
		want string
	}{
		{name: "start-nan", edit: func(p *ProfileBundle) { p.Phases[0].StartMilliseconds = math.NaN() }, want: "start must be finite and non-negative"},
		{name: "start-positive-inf", edit: func(p *ProfileBundle) { p.Phases[0].StartMilliseconds = math.Inf(1) }, want: "start must be finite and non-negative"},
		{name: "start-negative", edit: func(p *ProfileBundle) { p.Phases[0].StartMilliseconds = -1 }, want: "start must be finite and non-negative"},
		{name: "duration-nan", edit: func(p *ProfileBundle) { p.Phases[0].DurationMilliseconds = math.NaN() }, want: "duration must be finite and positive"},
		{name: "duration-positive-inf", edit: func(p *ProfileBundle) { p.Phases[0].DurationMilliseconds = math.Inf(1) }, want: "duration must be finite and positive"},
		{name: "duration-zero", edit: func(p *ProfileBundle) { p.Phases[0].DurationMilliseconds = 0 }, want: "duration must be finite and positive"},
		{name: "duration-negative", edit: func(p *ProfileBundle) { p.Phases[0].DurationMilliseconds = -1 }, want: "duration must be finite and positive"},
		{name: "end-overflow", edit: func(p *ProfileBundle) {
			p.Phases[0].StartMilliseconds = math.MaxFloat64
			p.Phases[0].DurationMilliseconds = math.MaxFloat64
		}, want: "end must be finite"},
		{name: "overlap", edit: func(p *ProfileBundle) { p.Phases[1].StartMilliseconds = 1 }, want: "overlaps the previous phase"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile := loadProfileFixture(t, "synthetic-metal-launch-bound.json")
			test.edit(&profile)
			err := ValidateProfile(graph, profile)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err=%v, want substring %q", err, test.want)
			}
		})
	}
}

func TestMetalCountersRejectInvalidDomains(t *testing.T) {
	graph := ActiveGraph()
	tests := []struct {
		name string
		edit func(*MetalCounters)
		want string
	}{
		{name: "command-buffers-zero", edit: func(c *MetalCounters) { c.CommandBuffers = 0 }, want: "command_buffers must be positive"},
		{name: "encoders-negative", edit: func(c *MetalCounters) { c.Encoders = -1 }, want: "encoders must be positive"},
		{name: "dispatch-zero", edit: func(c *MetalCounters) { c.DispatchMilliseconds = 0 }, want: "dispatch_milliseconds must be finite and positive"},
		{name: "dispatch-nan", edit: func(c *MetalCounters) { c.DispatchMilliseconds = math.NaN() }, want: "dispatch_milliseconds must be finite and positive"},
		{name: "dispatch-inf", edit: func(c *MetalCounters) { c.DispatchMilliseconds = math.Inf(1) }, want: "dispatch_milliseconds must be finite and positive"},
		{name: "wait-negative", edit: func(c *MetalCounters) { c.WaitMilliseconds = -1 }, want: "wait_milliseconds must be finite and non-negative"},
		{name: "wait-nan", edit: func(c *MetalCounters) { c.WaitMilliseconds = math.NaN() }, want: "wait_milliseconds must be finite and non-negative"},
		{name: "wait-inf", edit: func(c *MetalCounters) { c.WaitMilliseconds = math.Inf(1) }, want: "wait_milliseconds must be finite and non-negative"},
		{name: "resident-zero", edit: func(c *MetalCounters) { c.ResidentBytes = 0 }, want: "resident_bytes must be positive"},
		{name: "working-set-zero", edit: func(c *MetalCounters) { c.WorkingSetBytes = 0 }, want: "working_set_bytes must be positive"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile := loadProfileFixture(t, "synthetic-metal-launch-bound.json")
			test.edit(profile.Metal)
			err := ValidateProfile(graph, profile)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err=%v, want substring %q", err, test.want)
			}
		})
	}
}

func TestCUDACountersRejectInvalidDomains(t *testing.T) {
	graph := ActiveGraph()
	tests := []struct {
		name string
		edit func(*CUDACounters)
		want string
	}{
		{name: "launches-zero", edit: func(c *CUDACounters) { c.Launches = 0 }, want: "launches must be positive"},
		{name: "occupancy-negative", edit: func(c *CUDACounters) { c.OccupancyPercent = -1 }, want: "between 0 and 100"},
		{name: "occupancy-over-100", edit: func(c *CUDACounters) { c.OccupancyPercent = 101 }, want: "between 0 and 100"},
		{name: "occupancy-nan", edit: func(c *CUDACounters) { c.OccupancyPercent = math.NaN() }, want: "between 0 and 100"},
		{name: "occupancy-inf", edit: func(c *CUDACounters) { c.OccupancyPercent = math.Inf(1) }, want: "between 0 and 100"},
		{name: "achieved-bandwidth-negative", edit: func(c *CUDACounters) { c.AchievedBandwidthGBS = -1 }, want: "achieved_bandwidth_gbs must be finite and non-negative"},
		{name: "achieved-bandwidth-nan", edit: func(c *CUDACounters) { c.AchievedBandwidthGBS = math.NaN() }, want: "achieved_bandwidth_gbs must be finite and non-negative"},
		{name: "achieved-bandwidth-inf", edit: func(c *CUDACounters) { c.AchievedBandwidthGBS = math.Inf(1) }, want: "achieved_bandwidth_gbs must be finite and non-negative"},
		{name: "peak-bandwidth-zero", edit: func(c *CUDACounters) { c.PeakBandwidthGBS = 0 }, want: "peak_bandwidth_gbs must be finite and positive"},
		{name: "peak-bandwidth-negative", edit: func(c *CUDACounters) { c.PeakBandwidthGBS = -1 }, want: "peak_bandwidth_gbs must be finite and positive"},
		{name: "peak-bandwidth-nan", edit: func(c *CUDACounters) { c.PeakBandwidthGBS = math.NaN() }, want: "peak_bandwidth_gbs must be finite and positive"},
		{name: "peak-bandwidth-inf", edit: func(c *CUDACounters) { c.PeakBandwidthGBS = math.Inf(1) }, want: "peak_bandwidth_gbs must be finite and positive"},
		{name: "achieved-bandwidth-over-peak", edit: func(c *CUDACounters) { c.AchievedBandwidthGBS = c.PeakBandwidthGBS + 1 }, want: "must not exceed peak_bandwidth_gbs"},
		{name: "achieved-compute-negative", edit: func(c *CUDACounters) { c.AchievedComputeTFLOPS = -1 }, want: "achieved_compute_tflops must be finite and non-negative"},
		{name: "achieved-compute-nan", edit: func(c *CUDACounters) { c.AchievedComputeTFLOPS = math.NaN() }, want: "achieved_compute_tflops must be finite and non-negative"},
		{name: "achieved-compute-inf", edit: func(c *CUDACounters) { c.AchievedComputeTFLOPS = math.Inf(1) }, want: "achieved_compute_tflops must be finite and non-negative"},
		{name: "peak-compute-zero", edit: func(c *CUDACounters) { c.PeakComputeTFLOPS = 0 }, want: "peak_compute_tflops must be finite and positive"},
		{name: "peak-compute-negative", edit: func(c *CUDACounters) { c.PeakComputeTFLOPS = -1 }, want: "peak_compute_tflops must be finite and positive"},
		{name: "peak-compute-nan", edit: func(c *CUDACounters) { c.PeakComputeTFLOPS = math.NaN() }, want: "peak_compute_tflops must be finite and positive"},
		{name: "peak-compute-inf", edit: func(c *CUDACounters) { c.PeakComputeTFLOPS = math.Inf(1) }, want: "peak_compute_tflops must be finite and positive"},
		{name: "achieved-compute-over-peak", edit: func(c *CUDACounters) { c.AchievedComputeTFLOPS = c.PeakComputeTFLOPS + 1 }, want: "must not exceed peak_compute_tflops"},
		{name: "synchronization-negative", edit: func(c *CUDACounters) { c.SynchronizationMilliseconds = -1 }, want: "synchronization_milliseconds must be finite and non-negative"},
		{name: "synchronization-nan", edit: func(c *CUDACounters) { c.SynchronizationMilliseconds = math.NaN() }, want: "synchronization_milliseconds must be finite and non-negative"},
		{name: "synchronization-inf", edit: func(c *CUDACounters) { c.SynchronizationMilliseconds = math.Inf(1) }, want: "synchronization_milliseconds must be finite and non-negative"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile := loadProfileFixture(t, "synthetic-cuda-bandwidth-bound.json")
			test.edit(profile.CUDA)
			err := ValidateProfile(graph, profile)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err=%v, want substring %q", err, test.want)
			}
		})
	}
}

func TestProfileDispatchAttributionRequiresAvailableOrTypedUnavailable(t *testing.T) {
	graph := ActiveGraph()
	profile := loadProfileFixture(t, "synthetic-metal-attribution-unavailable.json")
	if err := ValidateProfile(graph, profile); err != nil {
		t.Fatal(err)
	}
	if _, err := ClassifyProfile(graph, profile); err != nil {
		t.Fatal(err)
	}

	profile.AttributionUnavailable.Reason = "not-sure"
	if err := ValidateProfile(graph, profile); err == nil || !strings.Contains(err.Error(), "reason is unknown") {
		t.Fatalf("unknown reason err=%v", err)
	}

	profile = loadProfileFixture(t, "synthetic-metal-launch-bound.json")
	profile.AttributionUnavailable = &AttributionUnavailable{
		Reason: AttributionUnavailableCapture,
		Detail: "Synthetic fixture models a capture export without dispatch attribution.",
	}
	if err := ValidateProfile(graph, profile); err == nil || !strings.Contains(err.Error(), "both available and unavailable") {
		t.Fatalf("mixed availability err=%v", err)
	}
}

func TestProfileNextContradictionRequiresIssueBackedReason(t *testing.T) {
	graph := ActiveGraph()

	metal := loadProfileFixture(t, "synthetic-metal-launch-bound.json")
	lever, classification, err := NextLeverFromProfile(graph, metal)
	if err != nil {
		t.Fatal(err)
	}
	if lever.ID != "metal.command-buffer-amortization" || lever.ID != classification.RecommendedLeverID {
		t.Fatalf("Metal next=%+v classification=%+v", lever, classification)
	}

	cuda := loadProfileFixture(t, "synthetic-cuda-bandwidth-bound.json")
	lever, classification, err = NextLeverFromProfile(graph, cuda)
	if err != nil {
		t.Fatal(err)
	}
	if lever.ID != "cuda.q8_1-activation-quant" || lever.ID != classification.RecommendedLeverID {
		t.Fatalf("CUDA next=%+v classification=%+v", lever, classification)
	}

	contradiction := loadProfileFixture(t, "reject-profile-next-contradiction-without-override.json")
	if err := ValidateProfile(graph, contradiction); err != nil {
		t.Fatal(err)
	}
	if _, _, err := NextLeverFromProfile(graph, contradiction); err == nil || !strings.Contains(err.Error(), "issue-backed reason") {
		t.Fatalf("contradiction err=%v", err)
	}

	override := loadProfileFixture(t, "synthetic-metal-bandwidth-override.json")
	lever, classification, err = NextLeverFromProfile(graph, override)
	if err != nil {
		t.Fatal(err)
	}
	if lever.ID != "metal.paged-kv" || lever.ID != classification.RecommendedLeverID {
		t.Fatalf("override next=%+v classification=%+v", lever, classification)
	}

	override.Override = &SelectionOverride{IssueNumber: 0, Reason: "Not issue-backed."}
	if _, _, err := NextLeverFromProfile(graph, override); err == nil || !strings.Contains(err.Error(), "positive issue number") {
		t.Fatalf("invalid override err=%v", err)
	}
}

func TestProfileNextRejectsRecommendationWithUnmetDependencies(t *testing.T) {
	graph := ActiveGraph()
	profile := loadProfileFixture(t, "synthetic-cuda-bandwidth-bound.json")
	profile.CUDA.AchievedBandwidthGBS = 500
	profile.CUDA.AchievedComputeTFLOPS = 80
	if _, _, err := NextLeverFromProfile(graph, profile); err == nil || !strings.Contains(err.Error(), "not an unwitnessed dependency-ready lever") {
		t.Fatalf("dependency readiness err=%v", err)
	}
}

func TestDecodeProfileRejectsUnknownFieldsAndTrailingDocuments(t *testing.T) {
	valid, err := os.ReadFile(filepath.Join(profileFixtureDir, "synthetic-metal-launch-bound.json"))
	if err != nil {
		t.Fatal(err)
	}
	var object map[string]any
	if err := json.Unmarshal(valid, &object); err != nil {
		t.Fatal(err)
	}
	object["raw_private_log"] = "not allowed"
	unknown, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeProfile(unknown); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown field err=%v", err)
	}
	if _, err := DecodeProfile(append(valid, valid...)); err == nil || !strings.Contains(err.Error(), "multiple JSON values") {
		t.Fatalf("trailing document err=%v", err)
	}
}
