package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

func TestModelObserveReportSpine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "observations.jsonl")
	data := strings.Join([]string{
		`{"schema":"fak-model-perf/1","timestamp":"2026-08-21T00:00:00Z","request_id":"r1","backend":"http://qwen","status":200,"streaming":true,"duration_ms":2500,"ttft_ms":1800,"tpot_ms":36,"output_tokens_per_second":27}`,
		`{"schema":"fak-model-perf/1","timestamp":"2026-08-21T00:00:01Z","request_id":"r2","backend":"http://qwen","status":200,"streaming":true,"duration_ms":2700,"ttft_ms":2000,"tpot_ms":37,"output_tokens_per_second":26}`,
	}, "\n")
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := modelperfobs.ReadObservations(f)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := modelperfobs.WriteMarkdown(&out, modelperfobs.Summarize(rows)); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "Likely bottleneck: **prefill-or-queue**") {
		t.Fatal(out.String())
	}
}

func TestModelObserveStateBenchSpine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache-state.json")
	if err := runModelObserveStateBench([]string{"--output", path}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	report, err := modelperfobs.ReadStateReport(f)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "admitted" || len(report.Arms) != 4 {
		t.Fatalf("cache-state report = verdict %q, arms %d", report.Verdict, len(report.Arms))
	}
}

func TestModelObserveBandwidthJSONSpine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bandwidth.json")
	output := filepath.Join(t.TempDir(), "report.json")
	data := `{"schema":"fak-model-bandwidth/1","engine":"fak-native","trigger":{"symptom_window":2,"resource_window":2,"latency_threshold_ms":100,"resource_utilization":0.8},"samples":[{"phase":"decode","shape":"small","provenance":{"source":"synthetic-test","machine":"fixture-host","device":"cpu-ddr5","collector":"fixture"},"rooflines":{"theoretical_gb_s":100,"measured_sustainable_gb_s":80},"live":{"read_gb_s":70,"write_gb_s":2},"request":{"latency_ms":120,"completion_tokens":1},"device":{},"capacity":{},"transfer":{},"software":{}},{"phase":"decode","shape":"small","provenance":{"source":"synthetic-test","machine":"fixture-host","device":"cpu-ddr5","collector":"fixture"},"rooflines":{"theoretical_gb_s":100,"measured_sustainable_gb_s":80},"live":{"read_gb_s":72,"write_gb_s":2},"request":{"latency_ms":125,"completion_tokens":1},"device":{},"capacity":{},"transfer":{},"software":{}}]}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runModelObserveBandwidth([]string{"--input", path, "--output", output, "--pretty=false"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(got)
	for _, want := range []string{`"schema":"fak-model-bandwidth/1"`, `"engine":"fak-native"`, `"selected_source":"measured-sustainable"`, `"write_gb_s":2`, `"transfer":{}`, `"triggered":true`} {
		if !strings.Contains(text, want) {
			t.Fatalf("report missing %s: %s", want, text)
		}
	}
	if strings.Contains(text, `"host_to_device_gb_s":0`) {
		t.Fatalf("unavailable transfer counter serialized as zero: %s", text)
	}
}

func TestModelObserveBandwidthCollectSpine(t *testing.T) {
	output := filepath.Join(t.TempDir(), "collection.json")
	if err := runModelObserveBandwidth([]string{"collect", "--count", "1", "--interval", "10ms", "--phase", "decode", "--shape", "small", "--theoretical-gb-s", "100", "--measured-gb-s", "80", "--output", output, "--pretty=false"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(got)
	for _, want := range []string{`"schema":"fak-model-bandwidth-collection/1"`, `"engine":"fak-native"`, `"machine_class":"`, `"capture":{`, `"report":{`, `"selected_source":"measured-sustainable"`, `"dram_counters":false`} {
		if !strings.Contains(text, want) {
			t.Fatalf("collection missing %s: %s", want, text)
		}
	}
	for _, forbidden := range []string{`"total_gb_s":0`, `"read_gb_s":0`, `"write_gb_s":0`} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("unavailable DRAM counter serialized as zero: %s", text)
		}
	}
	if strings.Contains(text, `"live":{"process_`) {
		t.Fatalf("process signal mislabeled as live DRAM bandwidth: %s", text)
	}
}

func TestModelObserveBandwidthMeasuresHostRoofline(t *testing.T) {
	old := measureHostMemoryRooflineForObserve
	measureHostMemoryRooflineForObserve = func(_ context.Context, o modelperfobs.RooflineBenchmarkOptions) (modelperfobs.RooflineMeasurement, error) {
		return deterministicRooflineMeasurement(o), nil
	}
	t.Cleanup(func() { measureHostMemoryRooflineForObserve = old })
	output := filepath.Join(t.TempDir(), "collection.json")
	args := []string{"collect", "--count", "1", "--interval", "10ms", "--phase", "other", "--shape", "small", "--nvidia-device", "__fak_no_device__", "--theoretical-gb-s", "999", "--measure-host-roofline", "--roofline-bytes", "1048576", "--roofline-trials", "3", "--roofline-duration", "10ms", "--roofline-threads", "1", "--output", output, "--pretty=false"}
	if err := runModelObserveBandwidth(args); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(got)
	for _, want := range []string{`"schema":"fak-host-memory-roofline/1"`, `"scope":"host-memory"`, `"method":"parallel-copy"`, `"traffic_accounting":"read-plus-write-2-bytes-per-copied-byte"`, `"overhead_accounting":"first-touch-and-calibration-reported-separately-from-measured-trials"`, `"target_duration_ms":10`, `"total_duration_ms":25`, `"warmup_duration_ms":1`, `"warmup_bytes_touched":3145728`, `"duration_ms":5`, `"calibration_duration_ms":40`, `"calibration_traffic_bytes":8388608`, `"runtime_budget_ms":100`, `"command_budget_ms":`, `"dram_isolation":"not-proven"`, `"interpretation":"sustained-host-memory-copy-throughput-not-hardware-counter-dram-bandwidth"`, `"aggregation":"median"`, `"applied_host_observation_count":1`, `"selected_source":"measured-sustainable"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("collection missing %s: %s", want, text)
		}
	}
}

func deterministicRooflineMeasurement(o modelperfobs.RooflineBenchmarkOptions) modelperfobs.RooflineMeasurement {
	return modelperfobs.RooflineMeasurement{
		Schema:                 modelperfobs.RooflineMeasurementSchema,
		Scope:                  "host-memory",
		MachineClass:           "test/test",
		Method:                 "parallel-copy",
		Aggregation:            "median",
		TrafficAccounting:      "read-plus-write-2-bytes-per-copied-byte",
		OverheadAccounting:     "first-touch-and-calibration-reported-separately-from-measured-trials",
		DRAMIsolation:          "not-proven",
		Interpretation:         "sustained-host-memory-copy-throughput-not-hardware-counter-dram-bandwidth",
		WorkingSetBytes:        o.WorkingSetBytes,
		PeakBufferBytes:        o.WorkingSetBytes * 2,
		TargetDurationMS:       o.TargetDuration.Milliseconds(),
		RuntimeBudgetMS:        100,
		TotalDurationMS:        25,
		WarmupDurationMS:       1,
		WarmupBytesTouched:     o.WorkingSetBytes * 3,
		Threads:                o.Threads,
		MeasuredSustainableGBS: 60,
		Trials:                 deterministicRooflineTrials(o, 60),
	}
}

func TestModelObserveBandwidthMeasuresHostRooflineSweep(t *testing.T) {
	old := measureHostMemoryRooflineForObserve
	measureHostMemoryRooflineForObserve = func(_ context.Context, o modelperfobs.RooflineBenchmarkOptions) (modelperfobs.RooflineMeasurement, error) {
		return deterministicRooflineSweepMeasurement(o), nil
	}
	t.Cleanup(func() { measureHostMemoryRooflineForObserve = old })
	output := filepath.Join(t.TempDir(), "collection.json")
	args := []string{"collect", "--count", "1", "--interval", "10ms", "--phase", "other", "--shape", "small", "--nvidia-device", "__fak_no_device__", "--measure-host-roofline", "--roofline-sweep", "--roofline-knee-threshold", "0.01", "--roofline-bytes", "1048576", "--roofline-trials", "3", "--roofline-duration", "10ms", "--roofline-threads", "3", "--output", output, "--pretty=false"}
	if err := runModelObserveBandwidth(args); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var collection modelperfobs.BandwidthCollection
	if err := json.Unmarshal(data, &collection); err != nil {
		t.Fatal(err)
	}
	got := collection.RooflineMeasurement
	if got == nil {
		t.Fatalf("missing roofline measurement: %s", data)
	}
	if got.Schema != modelperfobs.RooflineSweepMeasurementSchema || got.RequestedPointCount != 3 || got.PointCount != 3 || got.OmittedPointCount != 0 || got.KneeThreshold != 0.01 || got.AppliedHostObservationCount != 1 {
		t.Fatalf("sweep summary=%+v", got)
	}
	wantWorkers := []int{1, 2, 3}
	rawPeakIndex := 0
	kneeIndex := -1
	for i, point := range got.Points {
		if point.WorkerCount != wantWorkers[i] || point.MedianGBS <= 0 || point.EfficiencyVersusSustainablePeak <= 0 || len(point.Trials) != 3 || point.WarmupDurationMS <= 0 || point.WarmupBytesTouched == 0 {
			t.Fatalf("point[%d]=%+v", i, point)
		}
		if i == 0 {
			if point.MarginalGainGBS != nil || point.MarginalGainFraction != nil {
				t.Fatalf("first point fabricated marginal gain: %+v", point)
			}
		} else if point.MarginalGainGBS == nil || point.MarginalGainFraction == nil {
			t.Fatalf("point[%d] missing marginal gain: %+v", i, point)
		}
		if point.MedianGBS > got.Points[rawPeakIndex].MedianGBS {
			rawPeakIndex = i
		}
		if point.WorkerCount == got.SaturationKneeWorkerCount {
			kneeIndex = i
		}
	}
	if kneeIndex < 0 || got.RawObservedPeakWorkerCount != got.Points[rawPeakIndex].WorkerCount || got.RawObservedPeakMedianGBS != got.Points[rawPeakIndex].MedianGBS || got.MeasuredSustainableGBS <= 0 || got.MeasuredSustainableGBS > got.RawObservedPeakMedianGBS || len(got.PlateauWorkerCounts) != 2 {
		t.Fatalf("inconsistent sweep summary=%+v", got)
	}
	for i := kneeIndex; i < len(got.Points); i++ {
		if got.Points[i].MedianGBS < got.KneeThreshold*got.MeasuredSustainableGBS {
			t.Fatalf("point after knee fell below stable threshold: summary=%+v", got)
		}
	}
	text := string(data)
	for _, want := range []string{
		`"schema":"fak-host-memory-roofline-sweep/1"`,
		`"traffic_accounting":"read-plus-write-2-bytes-per-copied-byte"`,
		`"overhead_accounting":"first-touch-and-calibration-reported-separately-from-measured-trials"`,
		`"dram_isolation":"not-proven"`,
		`"interpretation":"sustained-host-memory-copy-throughput-not-hardware-counter-dram-bandwidth"`,
		`"raw_observed_peak_worker_count":`,
		`"raw_observed_peak_median_gb_s":`,
		`"plateau_worker_counts":`,
		`"saturation_knee_worker_count":`,
		`"efficiency_versus_sustainable_peak":`,
		`"marginal_gain_gb_s":`,
		`"calibration_duration_ms":`,
		`"runtime_budget_ms":`,
		`"command_budget_ms":`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("sweep collection missing %s: %s", want, text)
		}
	}
}

func deterministicRooflineSweepMeasurement(o modelperfobs.RooflineBenchmarkOptions) modelperfobs.RooflineMeasurement {
	workers := []int{1, 2, 3}
	medians := []float64{10, 20, 30}
	points := make([]modelperfobs.RooflineSweepPoint, len(workers))
	for i, workers := range workers {
		points[i] = modelperfobs.RooflineSweepPoint{
			WorkerCount:                     workers,
			MedianGBS:                       medians[i],
			EfficiencyVersusSustainablePeak: medians[i] / 20,
			WarmupDurationMS:                1,
			WarmupBytesTouched:              o.WorkingSetBytes * 3,
			Trials:                          deterministicRooflineTrials(o, medians[i]),
		}
		if i > 0 {
			gain := medians[i] - medians[i-1]
			fraction := gain / medians[i-1]
			points[i].MarginalGainGBS = &gain
			points[i].MarginalGainFraction = &fraction
		}
	}
	return modelperfobs.RooflineMeasurement{
		Schema:                     modelperfobs.RooflineSweepMeasurementSchema,
		Scope:                      "host-memory",
		MachineClass:               "test/test",
		Method:                     "parallel-copy",
		Aggregation:                "median",
		TrafficAccounting:          "read-plus-write-2-bytes-per-copied-byte",
		OverheadAccounting:         "first-touch-and-calibration-reported-separately-from-measured-trials",
		DRAMIsolation:              "not-proven",
		Interpretation:             "sustained-host-memory-copy-throughput-not-hardware-counter-dram-bandwidth",
		WorkingSetBytes:            o.WorkingSetBytes,
		PeakBufferBytes:            o.WorkingSetBytes * 2,
		TargetDurationMS:           o.TargetDuration.Milliseconds(),
		RuntimeBudgetMS:            100,
		TotalDurationMS:            50,
		Threads:                    o.Threads,
		MeasuredSustainableGBS:     20,
		RequestedPointCount:        len(points),
		PointCount:                 len(points),
		KneeThreshold:              o.KneeThreshold,
		StabilityRule:              "deterministic-test-fixture",
		RawObservedPeakWorkerCount: workers[len(workers)-1],
		RawObservedPeakMedianGBS:   medians[len(medians)-1],
		PlateauWorkerCounts:        []int{2, 3},
		SaturationKneeWorkerCount:  workers[0],
		Points:                     points,
	}
}

func deterministicRooflineTrials(o modelperfobs.RooflineBenchmarkOptions, gbs float64) []modelperfobs.RooflineTrial {
	trials := make([]modelperfobs.RooflineTrial, o.Trials)
	for trial := range trials {
		trials[trial] = modelperfobs.RooflineTrial{
			Index:                   trial + 1,
			Iterations:              2,
			DurationMS:              5,
			TrafficBytes:            o.WorkingSetBytes * 4,
			GBS:                     gbs,
			CalibrationRounds:       modelperfobs.MaxRooflineCalibrationRounds,
			CalibrationDurationMS:   40,
			CalibrationTrafficBytes: o.WorkingSetBytes * 8,
		}
	}
	return trials
}

func TestModelObserveBandwidthValidatesHostRooflineSweepFlags(t *testing.T) {
	base := []string{"collect", "--count", "1", "--interval", "10ms", "--phase", "other", "--shape", "small"}
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "sweep-requires-measurement", args: []string{"--roofline-sweep"}, want: "--roofline-sweep requires --measure-host-roofline"},
		{name: "threads-require-measurement", args: []string{"--roofline-threads", "2"}, want: "--roofline-threads requires --measure-host-roofline"},
		{name: "threshold-requires-sweep", args: []string{"--measure-host-roofline", "--roofline-knee-threshold", "0.8"}, want: "--roofline-knee-threshold requires --roofline-sweep"},
		{name: "threshold-bounded", args: []string{"--measure-host-roofline", "--roofline-sweep", "--roofline-knee-threshold", "0", "--roofline-bytes", "1048576", "--roofline-trials", "3", "--roofline-duration", "10ms", "--roofline-threads", "3"}, want: "roofline knee threshold"},
		{name: "sweep-needs-three-points", args: []string{"--measure-host-roofline", "--roofline-sweep", "--roofline-bytes", "1048576", "--roofline-trials", "3", "--roofline-duration", "10ms", "--roofline-threads", "2"}, want: "at least 3 worker-count points"},
		{name: "operator-measured-conflicts", args: []string{"--measure-host-roofline", "--measured-gb-s", "80"}, want: "--measured-gb-s cannot be combined with --measure-host-roofline"},
		{name: "total-runtime-bounded", args: []string{"--count", "120", "--interval", "1m", "--measure-host-roofline", "--roofline-bytes", "1048576", "--roofline-trials", "3", "--roofline-duration", "10ms", "--roofline-threads", "1"}, want: "collection worst-case runtime budget"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := append(append([]string(nil), base...), tt.args...)
			if err := runModelObserveBandwidth(args); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want substring %q", err, tt.want)
			}
		})
	}
}

func TestModelObserveBandwidthImportsNVIDIAProfile(t *testing.T) {
	output := filepath.Join(t.TempDir(), "collection.json")
	fixture := filepath.Join("..", "..", "internal", "modelperfobs", "testdata", "nvidia-hbm-ncu.csv")
	args := []string{
		"collect",
		"--nvidia-ncu-csv", fixture,
		"--device", "NVIDIA H100 80GB HBM3 (0)",
		"--capture-start", "2026-08-27T10:00:00Z",
		"--capture-end", "2026-08-27T10:01:00Z",
		"--phase", "decode",
		"--shape", "large",
		"--theoretical-gb-s", "3350",
		"--device-roofline-gb-s", "1250",
		"--output", output,
		"--pretty=false",
	}
	if err := runModelObserveBandwidth(args); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(got)
	for _, want := range []string{
		`"engine":"fak-native"`,
		`"collector":"nvidia-nsight-compute"`,
		`"dram_counters":true`,
		`"read_gb_s":750`,
		`"write_gb_s":250`,
		`"total_gb_s":1000`,
		`"utilization":0.8`,
		`"selected_source":"measured-sustainable"`,
		`"schema":"fak-nvidia-ncu-bandwidth-profile/1"`,
		`"engine_evidence":"operator-asserted-not-proven-by-csv"`,
		`"launch_count":2`,
		`"cumulative_duration_ns":2000000`,
		`"capture_ended_at":"2026-08-27T10:01:00Z"`,
		`"device_roofline_evidence":"operator-supplied-matched-device-measurement"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("NVIDIA profile collection missing %s: %s", want, text)
		}
	}
}

func TestModelObserveBandwidthRejectsCrossModeFlags(t *testing.T) {
	fixture := filepath.Join("..", "..", "internal", "modelperfobs", "testdata", "nvidia-hbm-ncu.csv")
	base := []string{"collect", "--nvidia-ncu-csv", fixture, "--device", "NVIDIA H100", "--capture-start", "2026-08-27T10:00:00Z", "--capture-end", "2026-08-27T10:01:00Z", "--phase", "decode", "--shape", "large"}
	for _, incompatible := range [][]string{{"--count", "2"}, {"--measured-gb-s", "80"}, {"--nvidia-device", "0"}, {"--logical-bytes", "10"}, {"--roofline-sweep"}} {
		args := append(append([]string(nil), base...), incompatible...)
		if err := runModelObserveBandwidth(args); err == nil || !strings.Contains(err.Error(), "cannot be combined") {
			t.Fatalf("args=%v error=%v", incompatible, err)
		}
	}
	if err := runModelObserveBandwidth([]string{"collect", "--device", "NVIDIA H100", "--count", "1", "--interval", "10ms", "--phase", "other", "--shape", "small"}); err == nil || !strings.Contains(err.Error(), "requires --nvidia-ncu-csv") {
		t.Fatalf("profile-only flag error=%v", err)
	}
}

func TestModelObserveBandwidthImportsHostControllerCounters(t *testing.T) {
	output := filepath.Join(t.TempDir(), "collection.json")
	fixture := filepath.Join("..", "..", "internal", "modelperfobs", "testdata", "host-controller-direct.json")
	args := []string{
		"collect",
		"--host-counter-import", fixture,
		"--host-counter-provider", "fixture-imc",
		"--host-counter-scope", "system",
		"--phase", "decode",
		"--shape", "small",
		"--measured-gb-s", "4",
		"--output", output,
		"--pretty=false",
	}
	if err := runModelObserveBandwidth(args); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{
		`"collector":"fixture-imc"`,
		`"dram_counters":true`,
		`"source":"host-controller-direct-bytes"`,
		`"device":"host-memory"`,
		`"read_gb_s":1.5`,
		`"write_gb_s":0.5`,
		`"total_gb_s":2`,
		`"selected_source":"measured-sustainable"`,
		`"host_controller_artifact":{"schema":"fak-host-controller-counters/1"`,
		`"scope":{"kind":"system"}`,
		`"byte_provenance":"direct-bytes"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("host controller collection missing %s: %s", want, text)
		}
	}
	for _, forbidden := range []string{`"process_read_bytes":0`, `"process_write_bytes":0`, `"physical_read_bytes":0`, `"physical_write_bytes":0`} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("unavailable process I/O serialized as DRAM evidence: %s", text)
		}
	}
}

func TestModelObserveBandwidthImportsPerfHostCounters(t *testing.T) {
	output := filepath.Join(t.TempDir(), "collection.json")
	fixture := filepath.Join("..", "..", "internal", "modelperfobs", "testdata", "host-controller-perf.csv")
	args := []string{
		"collect",
		"--host-counter-import", fixture,
		"--host-counter-format", "perf-csv",
		"--host-counter-provider", "linux-perf",
		"--host-counter-scope", "controller",
		"--host-counter-scope-id", "imc0",
		"--host-counter-bytes-per-event", "64",
		"--capture-start", "2026-08-27T10:00:00Z",
		"--capture-end", "2026-08-27T10:00:00.1Z",
		"--phase", "decode",
		"--shape", "small",
		"--output", output,
		"--pretty=false",
	}
	if err := runModelObserveBandwidth(args); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{`"import_format":"perf-csv"`, `"bytes_per_event":64`, `"running_ratio":1`, `"byte_provenance":"converted-events"`, `"total_gb_s":0.96`} {
		if !strings.Contains(text, want) {
			t.Fatalf("perf host collection missing %s: %s", want, text)
		}
	}
}

func TestModelObserveBandwidthValidatesHostCounterImportFlags(t *testing.T) {
	fixture := filepath.Join("..", "..", "internal", "modelperfobs", "testdata", "host-controller-direct.json")
	base := []string{"collect", "--host-counter-import", fixture, "--host-counter-provider", "fixture-imc", "--host-counter-scope", "system", "--phase", "other", "--shape", "small"}
	for _, incompatible := range [][]string{
		{"--count", "2"},
		{"--interval", "10ms"},
		{"--nvidia-device", "0"},
		{"--amd-device", "0"},
		{"--nvidia-ncu-csv", "profile.csv"},
		{"--logical-bytes", "10"},
		{"--physical-read-bytes", "10"},
		{"--measure-host-roofline"},
		{"--roofline-sweep"},
	} {
		args := append(append([]string(nil), base...), incompatible...)
		if err := runModelObserveBandwidth(args); err == nil || !strings.Contains(err.Error(), "cannot be combined") {
			t.Fatalf("args=%v error=%v", incompatible, err)
		}
	}
	for _, tt := range []struct {
		args []string
		want string
	}{
		{args: []string{"collect", "--host-counter-import", fixture, "--host-counter-scope", "system", "--phase", "other", "--shape", "small"}, want: "--host-counter-provider is required"},
		{args: []string{"collect", "--host-counter-import", fixture, "--host-counter-provider", "fixture-imc", "--phase", "other", "--shape", "small"}, want: "--host-counter-scope is required"},
		{args: []string{"collect", "--host-counter-provider", "fixture-imc", "--phase", "other", "--shape", "small"}, want: "requires --host-counter-import"},
		{args: []string{"collect", "--host-counter-import", fixture, "--host-counter-provider", "fixture-imc", "--host-counter-scope", "system", "--host-counter-scope-id", "0", "--phase", "other", "--shape", "small"}, want: "system host counter scope must not have an id"},
		{args: []string{"collect", "--host-counter-import", fixture, "--host-counter-provider", "fixture-imc", "--host-counter-scope", "controller", "--host-counter-scope-id", "imc0", "--measured-gb-s", "10", "--phase", "other", "--shape", "small"}, want: "require system-aggregate"},
		{args: []string{"collect", "--host-counter-import", fixture, "--host-counter-provider", "fixture-imc", "--host-counter-scope", "controller", "--host-counter-scope-id", "imc0", "--theoretical-gb-s", "10", "--phase", "other", "--shape", "small"}, want: "require system-aggregate"},
	} {
		if err := runModelObserveBandwidth(tt.args); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Fatalf("args=%v error=%v want=%q", tt.args, err, tt.want)
		}
	}
}
