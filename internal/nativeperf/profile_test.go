package nativeperf

import (
	"encoding/json"
	"errors"
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

// fak-test:runtime fast est=1s
func TestProfileCounterScopePrevention(t *testing.T) {
	graph := ActiveGraph()
	loadLegacy := func(t *testing.T, name string) ProfileBundle {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(profileFixtureDir, name))
		if err != nil {
			t.Fatal(err)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatal(err)
		}
		delete(raw, "counter_window")
		data, err = json.Marshal(raw)
		if err != nil {
			t.Fatal(err)
		}
		p, err := DecodeProfile(data)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	tests := []struct {
		name, fixture string
		edit          func(*ProfileBundle)
	}{
		{
			name: "cuda-high-count-bandwidth", fixture: "synthetic-cuda-bandwidth-bound.json",
			edit: func(p *ProfileBundle) { p.CUDA.Launches = 640 },
		},
		{
			name: "cuda-high-count-compute", fixture: "synthetic-cuda-bandwidth-bound.json",
			edit: func(p *ProfileBundle) {
				p.CUDA.Launches = 640
				p.CUDA.AchievedBandwidthGBS = 500
				p.CUDA.AchievedComputeTFLOPS = 80
			},
		},
		{
			name: "metal-high-count-device-active", fixture: "synthetic-metal-launch-bound.json",
			edit: func(p *ProfileBundle) {
				p.Metal.CommandBuffers, p.Metal.Encoders = 640, 640
				p.Metal.DispatchMilliseconds, p.Metal.WaitMilliseconds = 36, 1
			},
		},
		{
			name: "metal-lifecycle-wait-issue-9020", fixture: "synthetic-metal-launch-bound.json",
			edit: func(p *ProfileBundle) {
				// Replay phase/counter numbers from docs/_witnesses/issue-9020-metal-profile.json.
				p.Phases = []ProfilePhase{
					{Name: "load-setup", StartMilliseconds: 0, DurationMilliseconds: 24329.352542},
					{Name: "prefill", StartMilliseconds: 24329.352542, DurationMilliseconds: 32971.988708},
					{Name: "first-token", StartMilliseconds: 57301.34125, DurationMilliseconds: 2523.281666},
					{Name: "steady-decode", StartMilliseconds: 59824.622916, DurationMilliseconds: 16973.805959},
					{Name: "verification", StartMilliseconds: 76798.428875, DurationMilliseconds: 25.327459},
					{Name: "teardown", StartMilliseconds: 76823.75633399999, DurationMilliseconds: 0.15925},
				}
				p.Metal = &MetalCounters{
					CommandBuffers: 14833, Encoders: 23025,
					DispatchMilliseconds: 15322.027541660646, WaitMilliseconds: 39772.66442775726,
					ResidentBytes: 24595492864, WorkingSetBytes: 18288771072,
				}
				p.Attributions = nil
				p.AttributionUnavailable = &AttributionUnavailable{
					Reason: AttributionUnavailableCapture,
					Detail: "fak-native session lifecycle capture does not export per-lever dispatch attribution",
				}
			},
		},
	}
	falseLaunch, falseSync, prevented := 0, 0, 0
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := loadLegacy(t, test.fixture)
			test.edit(&p)
			got, err := ClassifyProfile(graph, p)
			if err != nil {
				t.Fatal(err)
			}
			if got.Class == "launch-bound" {
				falseLaunch++
			}
			if got.Class == "synchronization-bound" {
				falseSync++
			}
			if got.Class == "insufficient-evidence" && got.RecommendedLeverID == "" {
				prevented++
			} else {
				t.Errorf("unscoped counters produced class=%q lever=%q, want insufficient-evidence with no recommendation", got.Class, got.RecommendedLeverID)
			}
		})
	}
	t.Logf("fixed negative families=%d false_launch=%d false_sync=%d prevented=%d", len(tests), falseLaunch, falseSync, prevented)
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

func profileCounterWindowControl(t *testing.T, backend, kind string) ProfileBundle {
	t.Helper()
	fixture := "synthetic-metal-launch-bound.json"
	if backend == "cuda" {
		fixture = "synthetic-cuda-bandwidth-bound.json"
	}
	p := loadProfileFixture(t, fixture)
	w := p.CounterWindow
	w.HostLaunchMilliseconds, w.HostSetupMilliseconds, w.HostTimingSource = nil, nil, ""
	if p.Metal != nil {
		p.Metal.DispatchMilliseconds, p.Metal.WaitMilliseconds = 20, 1
	} else {
		p.CUDA.AchievedBandwidthGBS, p.CUDA.AchievedComputeTFLOPS = 100, 10
		p.CUDA.SynchronizationMilliseconds = 1
	}
	measured := 12.0
	switch kind {
	case "launch":
		w.HostLaunchMilliseconds, w.HostTimingSource = &measured, "host-monotonic-exclusive"
	case "setup":
		w.HostSetupMilliseconds, w.HostTimingSource = &measured, "host-monotonic-exclusive"
	case "sync":
		if p.Metal != nil {
			p.Metal.WaitMilliseconds = measured
		} else {
			p.CUDA.SynchronizationMilliseconds = measured
		}
	default:
		t.Fatalf("unknown test control %q", kind)
	}
	return p
}

// fak-test:runtime fast est=1s
func TestProfileCounterWindowPositiveControls(t *testing.T) {
	graph := ActiveGraph()
	correct, population := 0, 0
	for _, backend := range []string{"metal", "cuda"} {
		for _, control := range []struct{ kind, class string }{
			{"launch", "launch-bound"}, {"setup", "cpu-orchestration-bound"}, {"sync", "synchronization-bound"},
		} {
			population++
			t.Run(backend+"/"+control.kind, func(t *testing.T) {
				p := profileCounterWindowControl(t, backend, control.kind)
				got, err := ClassifyProfile(graph, p)
				if err != nil {
					t.Fatal(err)
				}
				lever := "metal.command-buffer-amortization"
				if backend == "cuda" {
					lever = "cuda.default-decode-routing"
				}
				if got.Class != control.class || got.RecommendedLeverID != lever {
					t.Fatalf("matching measured %s => class=%q lever=%q, want %q/%q", control.kind, got.Class, got.RecommendedLeverID, control.class, lever)
				}
				correct++
			})
		}
	}
	t.Logf("fixed positive controls=%d correctly_classified=%d", population, correct)
}

// fak-test:runtime fast est=1s
func TestProfileMatchingWindowCountsDoNotOverrideDeviceEvidence(t *testing.T) {
	graph := ActiveGraph()
	for _, tc := range []struct {
		name, fixture, class string
		edit                 func(*ProfileBundle)
	}{
		{"cuda-bandwidth", "synthetic-cuda-bandwidth-bound.json", "bandwidth-bound", func(p *ProfileBundle) { p.CUDA.Launches = 640 }},
		{"cuda-compute", "synthetic-cuda-bandwidth-bound.json", "compute-bound", func(p *ProfileBundle) {
			p.CUDA.Launches, p.CUDA.AchievedBandwidthGBS, p.CUDA.AchievedComputeTFLOPS = 640, 500, 80
		}},
		{"metal-device-active", "synthetic-metal-launch-bound.json", "compute-bound", func(p *ProfileBundle) {
			p.Metal.CommandBuffers, p.Metal.Encoders = 640, 640
			p.Metal.DispatchMilliseconds, p.Metal.WaitMilliseconds = 36, 1
			p.CounterWindow.HostLaunchMilliseconds, p.CounterWindow.HostTimingSource = nil, ""
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := loadProfileFixture(t, tc.fixture)
			tc.edit(&p)
			got, err := ClassifyProfile(graph, p)
			if err != nil {
				t.Fatal(err)
			}
			if got.Class != tc.class {
				t.Errorf("high count class=%q, want measured %q", got.Class, tc.class)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestProfileCounterWindowExposureInvariance(t *testing.T) {
	graph := ActiveGraph()
	for _, backend := range []string{"metal", "cuda"} {
		for _, kind := range []string{"launch", "setup", "sync"} {
			baseline, err := ClassifyProfile(graph, profileCounterWindowControl(t, backend, kind))
			if err != nil {
				t.Fatal(err)
			}
			for _, scale := range []float64{1, 2, 100} {
				for _, count := range []int{1, 640} {
					p := profileCounterWindowControl(t, backend, kind)
					for i := range p.Phases {
						p.Phases[i].StartMilliseconds *= scale
						p.Phases[i].DurationMilliseconds *= scale
					}
					p.CounterWindow.StartMilliseconds *= scale
					p.CounterWindow.DurationMilliseconds *= scale
					if p.CounterWindow.HostLaunchMilliseconds != nil {
						*p.CounterWindow.HostLaunchMilliseconds *= scale
					}
					if p.CounterWindow.HostSetupMilliseconds != nil {
						*p.CounterWindow.HostSetupMilliseconds *= scale
					}
					if p.Metal != nil {
						p.Metal.CommandBuffers, p.Metal.Encoders = count, count
						p.Metal.DispatchMilliseconds *= scale
						p.Metal.WaitMilliseconds *= scale
					} else {
						p.CUDA.Launches = count
						p.CUDA.SynchronizationMilliseconds *= scale
					}
					got, err := ClassifyProfile(graph, p)
					if err != nil {
						t.Fatalf("%s/%s scale=%v count=%d: %v", backend, kind, scale, count, err)
					}
					if got.Class != baseline.Class || got.RecommendedLeverID != baseline.RecommendedLeverID {
						t.Errorf("%s/%s same 63 steps scale=%v count=%d changed class %q->%q", backend, kind, scale, count, baseline.Class, got.Class)
					}
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s
func TestProfileCounterWindowRejectsInvalidEvidence(t *testing.T) {
	graph := ActiveGraph()
	ptr := func(v float64) *float64 { return &v }
	for _, tc := range []struct {
		name string
		edit func(*ProfileBundle)
	}{
		{"backend mismatch", func(p *ProfileBundle) { p.CounterWindow.Backend = "cuda" }},
		{"phase mismatch", func(p *ProfileBundle) { p.CounterWindow.Phase = "prefill" }},
		{"start mismatch", func(p *ProfileBundle) { p.CounterWindow.StartMilliseconds++ }},
		{"duration mismatch", func(p *ProfileBundle) { p.CounterWindow.DurationMilliseconds++ }},
		{"zero steps", func(p *ProfileBundle) { p.CounterWindow.DecodeSteps = 0 }},
		{"wrong steps", func(p *ProfileBundle) { p.CounterWindow.DecodeSteps = 62 }},
		{"nonfinite start", func(p *ProfileBundle) { p.CounterWindow.StartMilliseconds = math.NaN() }},
		{"negative start", func(p *ProfileBundle) { p.CounterWindow.StartMilliseconds = -1 }},
		{"nonfinite duration", func(p *ProfileBundle) { p.CounterWindow.DurationMilliseconds = math.Inf(1) }},
		{"zero duration", func(p *ProfileBundle) { p.CounterWindow.DurationMilliseconds = 0 }},
		{"negative launch", func(p *ProfileBundle) { p.CounterWindow.HostLaunchMilliseconds = ptr(-1) }},
		{"nan launch", func(p *ProfileBundle) { p.CounterWindow.HostLaunchMilliseconds = ptr(math.NaN()) }},
		{"infinite launch", func(p *ProfileBundle) { p.CounterWindow.HostLaunchMilliseconds = ptr(math.Inf(1)) }},
		{"negative setup", func(p *ProfileBundle) { p.CounterWindow.HostSetupMilliseconds = ptr(-1) }},
		{"nan setup", func(p *ProfileBundle) { p.CounterWindow.HostSetupMilliseconds = ptr(math.NaN()) }},
		{"infinite setup", func(p *ProfileBundle) { p.CounterWindow.HostSetupMilliseconds = ptr(math.Inf(1)) }},
		{"exclusive sum over wall", func(p *ProfileBundle) {
			p.CounterWindow.HostLaunchMilliseconds, p.CounterWindow.HostSetupMilliseconds = ptr(25), ptr(20)
		}},
		{"launch over wall", func(p *ProfileBundle) { p.CounterWindow.HostLaunchMilliseconds = ptr(41) }},
		{"wait over wall", func(p *ProfileBundle) { p.Metal.WaitMilliseconds = 41 }},
		{"unknown source", func(p *ProfileBundle) { p.CounterWindow.HostTimingSource = "wall-minus-device" }},
		{"missing source", func(p *ProfileBundle) { p.CounterWindow.HostTimingSource = "" }},
		{"source without timing", func(p *ProfileBundle) { p.CounterWindow.HostLaunchMilliseconds = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := loadProfileFixture(t, "synthetic-metal-launch-bound.json")
			tc.edit(&p)
			if err := ValidateProfile(graph, p); !errors.Is(err, ErrProfileWindowInvalid) {
				t.Errorf("invalid window error=%v, want ErrProfileWindowInvalid", err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestDecodeProfileCounterWindowPresenceAndTypes(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(profileFixtureDir, "synthetic-metal-launch-bound.json"))
	if err != nil {
		t.Fatal(err)
	}
	object := func() map[string]json.RawMessage {
		t.Helper()
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatal(err)
		}
		return raw
	}
	decode := func(raw map[string]json.RawMessage) (ProfileBundle, error) {
		t.Helper()
		b, err := json.Marshal(raw)
		if err != nil {
			t.Fatal(err)
		}
		return DecodeProfile(b)
	}
	for _, required := range []string{"backend", "phase", "start_milliseconds", "duration_milliseconds", "decode_steps"} {
		for _, absent := range []bool{false, true} {
			t.Run(required+map[bool]string{false: "/null", true: "/missing"}[absent], func(t *testing.T) {
				raw := object()
				var window map[string]json.RawMessage
				if err := json.Unmarshal(raw["counter_window"], &window); err != nil {
					t.Fatal(err)
				}
				if absent {
					delete(window, required)
				} else {
					window[required] = json.RawMessage("null")
				}
				b, err := json.Marshal(window)
				if err != nil {
					t.Fatal(err)
				}
				raw["counter_window"] = b
				if _, err := decode(raw); !errors.Is(err, ErrProfileWindowInvalid) {
					t.Errorf("decode error=%v, want ErrProfileWindowInvalid", err)
				}
			})
		}
	}
	for _, bad := range []string{`"unknown"`, `[]`, `{"backend":"metal"}`} {
		raw := object()
		raw["counter_window"] = json.RawMessage(bad)
		if _, err := decode(raw); !errors.Is(err, ErrProfileWindowInvalid) {
			t.Errorf("malformed window %s: error=%v, want ErrProfileWindowInvalid", bad, err)
		}
	}
	for _, bad := range []string{`"unknown"`, `1e309`} {
		raw := object()
		var window map[string]json.RawMessage
		if err := json.Unmarshal(raw["counter_window"], &window); err != nil {
			t.Fatal(err)
		}
		window["host_launch_milliseconds"] = json.RawMessage(bad)
		b, err := json.Marshal(window)
		if err != nil {
			t.Fatal(err)
		}
		raw["counter_window"] = b
		if _, err := decode(raw); !errors.Is(err, ErrProfileWindowInvalid) {
			t.Errorf("malformed host timing %s: error=%v, want ErrProfileWindowInvalid", bad, err)
		}
	}
	for _, absent := range []bool{false, true} {
		raw := object()
		if absent {
			delete(raw, "counter_window")
		} else {
			raw["counter_window"] = json.RawMessage("null")
		}
		p, err := decode(raw)
		if err != nil || p.CounterWindow != nil {
			t.Fatalf("legacy window: err=%v window=%v", err, p.CounterWindow)
		}
		p.Override = &SelectionOverride{IssueNumber: 9020, Reason: "Explicit selection cannot substitute for measured counter scope."}
		lever, classification, err := NextLeverFromProfile(ActiveGraph(), p)
		if !errors.Is(err, ErrProfileEvidenceUnavailable) || lever != nil || classification.Class != "insufficient-evidence" {
			t.Errorf("legacy next returned lever=%v class=%q err=%v", lever, classification.Class, err)
		}
	}
	for _, timing := range []string{"null", "0"} {
		raw := object()
		var window map[string]json.RawMessage
		if err := json.Unmarshal(raw["counter_window"], &window); err != nil {
			t.Fatal(err)
		}
		window["host_launch_milliseconds"] = json.RawMessage(timing)
		if timing == "null" {
			delete(window, "host_timing_source")
		}
		b, err := json.Marshal(window)
		if err != nil {
			t.Fatal(err)
		}
		raw["counter_window"] = b
		p, err := decode(raw)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateProfile(ActiveGraph(), p); err != nil {
			t.Fatal(err)
		}
		if timing == "null" && p.CounterWindow.HostLaunchMilliseconds != nil {
			t.Error("null host timing became measured")
		}
		if timing == "0" && (p.CounterWindow.HostLaunchMilliseconds == nil || *p.CounterWindow.HostLaunchMilliseconds != 0) {
			t.Error("measured-zero host timing lost its presence")
		}
	}
}

// fak-test:runtime fast est=1s
func TestDecodeProfileCounterWindowRejectsDuplicateKeys(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(profileFixtureDir, "synthetic-metal-launch-bound.json"))
	if err != nil {
		t.Fatal(err)
	}
	var profile map[string]json.RawMessage
	if err := json.Unmarshal(data, &profile); err != nil {
		t.Fatal(err)
	}
	var matchingWindow map[string]json.RawMessage
	if err := json.Unmarshal(profile["counter_window"], &matchingWindow); err != nil {
		t.Fatal(err)
	}
	delete(profile, "counter_window")
	outer, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	appendPair := func(object []byte, firstKey, firstValue, secondKey, secondValue string) []byte {
		t.Helper()
		trimmed := strings.TrimSpace(string(object))
		return []byte(trimmed[:len(trimmed)-1] + `,"` + firstKey + `":` + firstValue + `,"` + secondKey + `":` + secondValue + `}`)
	}
	assertRejected := func(t *testing.T, raw []byte) {
		t.Helper()
		if _, err := DecodeProfile(raw); !errors.Is(err, ErrProfileWindowInvalid) {
			t.Errorf("duplicate counter-window key error=%v, want ErrProfileWindowInvalid", err)
		}
	}
	prefill := `{"backend":"metal","phase":"prefill","start_milliseconds":12,"duration_milliseconds":10,"decode_steps":1,"host_launch_milliseconds":12,"host_timing_source":"host-monotonic-exclusive"}`
	steady := `{"backend":"metal","phase":"steady-decode","start_milliseconds":28,"duration_milliseconds":40,"decode_steps":63}`
	for _, tc := range []struct{ name, first, second string }{
		{"exact", "counter_window", "counter_window"},
		{"casefold second", "counter_window", "COUNTER_WINDOW"},
		{"casefold first", "COUNTER_WINDOW", "counter_window"},
	} {
		t.Run("root/"+tc.name, func(t *testing.T) {
			assertRejected(t, appendPair(outer, tc.first, prefill, tc.second, steady))
		})
	}
	for _, member := range []struct{ key, firstValue, secondValue string }{
		{"phase", `"prefill"`, `"steady-decode"`},
		{"start_milliseconds", `12`, `28`},
		{"host_launch_milliseconds", `12`, `0`},
		{"host_timing_source", `"unknown"`, `"host-monotonic-exclusive"`},
	} {
		for _, spelling := range []struct{ name, first, second string }{
			{"exact", member.key, member.key},
			{"casefold second", member.key, strings.ToUpper(member.key)},
			{"casefold first", strings.ToUpper(member.key), member.key},
		} {
			t.Run("member/"+member.key+"/"+spelling.name, func(t *testing.T) {
				window := make(map[string]json.RawMessage, len(matchingWindow))
				for key, value := range matchingWindow {
					if key != member.key {
						window[key] = value
					}
				}
				windowBytes, err := json.Marshal(window)
				if err != nil {
					t.Fatal(err)
				}
				duplicate := appendPair(windowBytes, spelling.first, member.firstValue, spelling.second, member.secondValue)
				root := make(map[string]json.RawMessage, len(profile)+1)
				for key, value := range profile {
					root[key] = value
				}
				root["counter_window"] = duplicate
				raw, err := json.Marshal(root)
				if err != nil {
					t.Fatal(err)
				}
				assertRejected(t, raw)
			})
		}
	}
}
