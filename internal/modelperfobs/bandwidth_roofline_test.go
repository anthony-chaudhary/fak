package modelperfobs

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestRooflineBenchmarkBounds(t *testing.T) {
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: MinRooflineTrials, TargetDuration: 10 * time.Millisecond, Threads: 1}
	if err := ValidateRooflineBenchmarkOptions(o); err != nil {
		t.Fatal(err)
	}
	o.WorkingSetBytes = MinRooflineWorkingSet - 1
	if err := ValidateRooflineBenchmarkOptions(o); err == nil {
		t.Fatal("expected working-set bound")
	}
	o = RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: MinRooflineTrials, TargetDuration: 10 * time.Millisecond, Threads: 3, Sweep: true, KneeThreshold: DefaultRooflineKneeThreshold}
	if err := ValidateRooflineBenchmarkOptions(o); err != nil {
		t.Fatal(err)
	}
	o.KneeThreshold = 0
	if err := ValidateRooflineBenchmarkOptions(o); err == nil || !strings.Contains(err.Error(), "knee threshold") {
		t.Fatalf("invalid knee threshold error = %v", err)
	}
	o.KneeThreshold = DefaultRooflineKneeThreshold
	o.Threads = 2
	if err := ValidateRooflineBenchmarkOptions(o); err == nil || !strings.Contains(err.Error(), "at least 3") {
		t.Fatalf("underspecified sweep error = %v", err)
	}
	o = RooflineBenchmarkOptions{WorkingSetBytes: MaxRooflineWorkingSet, Trials: MaxRooflineTrials, TargetDuration: 2 * time.Second, Threads: MaxRooflineThreads, Sweep: true, KneeThreshold: DefaultRooflineKneeThreshold}
	if err := ValidateRooflineBenchmarkOptions(o); err == nil || !strings.Contains(err.Error(), "worst-case runtime budget") {
		t.Fatalf("unbounded runtime error = %v", err)
	}
}

func TestMeasureHostMemoryRooflineAccounting(t *testing.T) {
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: 3, TargetDuration: 10 * time.Millisecond, Threads: 1}
	got, err := MeasureHostMemoryRoofline(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != RooflineMeasurementSchema || got.Scope != "host-memory" || got.MeasuredSustainableGBS <= 0 || got.DRAMIsolation != "not-proven" || len(got.Caveats) < 4 || len(got.Trials) != 3 {
		t.Fatalf("%+v", got)
	}
	if got.PeakBufferBytes != o.WorkingSetBytes*2 || got.RuntimeBudgetMS <= 0 || got.TotalDurationMS <= 0 || got.OverheadAccounting == "" {
		t.Fatalf("bounded accounting missing: %+v", got)
	}
	for _, trial := range got.Trials {
		want := o.WorkingSetBytes * 2 * trial.Iterations
		if trial.TrafficBytes != want || trial.GBS <= 0 || trial.DurationMS <= 0 || trial.CalibrationRounds < 1 || trial.CalibrationDurationMS <= 0 || trial.CalibrationTrafficBytes == 0 {
			t.Fatalf("trial=%+v want traffic=%d", trial, want)
		}
	}
}

func TestMeasureHostMemoryRooflineRecordsBoundedBatchBelowCalibrationTarget(t *testing.T) {
	stubFastRooflineCopyBatches(t)
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: 3, TargetDuration: 10 * time.Millisecond, Threads: 1}
	got, err := MeasureHostMemoryRoofline(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	for _, trial := range got.Trials {
		if trial.CalibrationRounds != MaxRooflineCalibrationRounds || trial.DurationMS != 5 || trial.DurationMS >= float64(o.TargetDuration/time.Millisecond) {
			t.Fatalf("sub-target bounded batch was not recorded: %+v", trial)
		}
		if trial.Iterations == 0 || trial.CalibrationDurationMS != 40 || trial.CalibrationTrafficBytes == 0 {
			t.Fatalf("calibration accounting=%+v", trial)
		}
	}
}

func TestMeasureHostMemoryRooflineAllocatesAndFirstTouchesPerSweepPoint(t *testing.T) {
	stubFastRooflineCopyBatches(t)
	old := allocateRooflinePointBuffers
	allocations := 0
	allocateRooflinePointBuffers = func(n int) ([]byte, []byte) {
		allocations++
		src, dst := make([]byte, n), make([]byte, n)
		return src, dst
	}
	t.Cleanup(func() { allocateRooflinePointBuffers = old })
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: 3, TargetDuration: 10 * time.Millisecond, Threads: 3, Sweep: true, KneeThreshold: 0.01}
	got, err := MeasureHostMemoryRoofline(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if allocations != 3 {
		t.Fatalf("allocation calls=%d want one buffer pair per point", allocations)
	}
	if got.PeakBufferBytes != 2*o.WorkingSetBytes || !containsString(got.Caveats, "buffers-reallocated-and-first-touched-per-worker-count") || !containsString(got.Caveats, "per-point-buffer-release-is-go-runtime-mediated") {
		t.Fatalf("bounded memory/caveat missing: %+v", got)
	}
	for _, point := range got.Points {
		if point.WarmupBytesTouched != 3*o.WorkingSetBytes {
			t.Fatalf("point warmup accounting=%+v", point)
		}
	}
}

func TestMeasureHostMemoryRooflineOmitsFailedPointAndContinues(t *testing.T) {
	stubFastRooflineCopyBatches(t)
	old := allocateRooflinePointBuffers
	allocation := 0
	allocateRooflinePointBuffers = func(n int) ([]byte, []byte) {
		allocation++
		if allocation == 2 {
			return make([]byte, n), make([]byte, n-1)
		}
		return make([]byte, n), make([]byte, n)
	}
	t.Cleanup(func() { allocateRooflinePointBuffers = old })
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: 3, TargetDuration: 10 * time.Millisecond, Threads: 8, Sweep: true, KneeThreshold: 0.01}
	got, err := MeasureHostMemoryRoofline(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.RequestedPointCount != 4 || got.PointCount != 3 || got.OmittedPointCount != 1 || len(got.OmittedPoints) != 1 {
		t.Fatalf("omission summary=%+v", got)
	}
	if got.OmittedPoints[0].WorkerCount != 2 || !strings.Contains(got.OmittedPoints[0].Reason, "buffers do not match") {
		t.Fatalf("omission=%+v", got.OmittedPoints[0])
	}
	wantWorkers := []int{1, 4, 8}
	for i, point := range got.Points {
		if point.WorkerCount != wantWorkers[i] || point.MedianGBS <= 0 {
			t.Fatalf("point[%d]=%+v want workers=%d", i, point, wantWorkers[i])
		}
	}
}

func TestRooflineSweepWorkerCountsAreBoundedAndEndExactly(t *testing.T) {
	tests := []struct {
		threads int
		want    []int
	}{
		{threads: 1, want: []int{1}},
		{threads: 3, want: []int{1, 2, 3}},
		{threads: 6, want: []int{1, 2, 4, 6}},
		{threads: MaxRooflineThreads, want: []int{1, 2, 4, 8, 16, 32, 64, 128, 256}},
	}
	for _, tt := range tests {
		got, err := rooflineSweepWorkerCounts(tt.threads)
		if err != nil {
			t.Fatalf("threads=%d: %v", tt.threads, err)
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Fatalf("threads=%d counts=%v want=%v", tt.threads, got, tt.want)
		}
		if len(got) > MaxRooflineSweepPoints {
			t.Fatalf("threads=%d produced %d points; max=%d", tt.threads, len(got), MaxRooflineSweepPoints)
		}
	}
}

func TestMedianRooflineGBSUsesTrueEvenMedian(t *testing.T) {
	trials := testRooflinePoint(1, 10, 20, 40, 100).Trials
	got, err := medianRooflineGBS(trials)
	if err != nil {
		t.Fatal(err)
	}
	if got != 30 {
		t.Fatalf("median=%g want=30", got)
	}
}

func TestBuildRooflineSweepMeasurementUsesStablePlateauAndMarginalGains(t *testing.T) {
	o := testSweepOptions()
	results := []rooflinePointResult{
		testRooflineResult(1, 65, 70, 75),
		testRooflineResult(2, 88, 90, 92),
		testRooflineResult(4, 95, 100, 105),
		testRooflineResult(8, 80, 100, 120),
	}
	got, err := buildRooflineSweepMeasurement(o, results)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != RooflineSweepMeasurementSchema || got.PointCount != 4 || got.RequestedPointCount != 4 || got.OmittedPointCount != 0 {
		t.Fatalf("summary=%+v", got)
	}
	if got.RawObservedPeakWorkerCount != 4 || got.RawObservedPeakMedianGBS != 100 || got.MeasuredSustainableGBS != 100 || !reflect.DeepEqual(got.PlateauWorkerCounts, []int{4, 8}) {
		t.Fatalf("peak summary=%+v", got)
	}
	if got.RawObservedPeakStatus != "measured" || got.SaturationKneeStatus != "measured" || got.EnvelopeDigest == "" {
		t.Fatalf("shared sweep evidence=%+v", got)
	}
	if got.SaturationKneeWorkerCount != 2 || got.StabilityRule != rooflineStabilityRule {
		t.Fatalf("knee/stability=%+v", got)
	}
	wantEfficiencies := []float64{0.70, 0.90, 1, 1}
	wantGains := []float64{20, 10, 0}
	for i, point := range got.Points {
		if math.Abs(point.EfficiencyVersusSustainablePeak-wantEfficiencies[i]) > 1e-12 {
			t.Fatalf("point %d efficiency=%g want=%g", i, point.EfficiencyVersusSustainablePeak, wantEfficiencies[i])
		}
		if i == 0 {
			if point.MarginalGainGBS != nil || point.MarginalGainFraction != nil {
				t.Fatalf("first point fabricated marginal gain: %+v", point)
			}
		} else if point.MarginalGainGBS == nil || *point.MarginalGainGBS != wantGains[i-1] || point.MarginalGainFraction == nil {
			t.Fatalf("point %d marginal gain=%+v", i, point)
		}
	}
}

func TestBuildRooflineSweepMeasurementRejectsIsolatedSpike(t *testing.T) {
	o := testSweepOptions()
	got, err := buildRooflineSweepMeasurement(o, []rooflinePointResult{
		testRooflineResult(1, 65, 70, 75),
		testRooflineResult(2, 90, 95, 100),
		testRooflineResult(4, 290, 300, 310),
		testRooflineResult(8, 91, 96, 101),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.RawObservedPeakWorkerCount != 4 || got.RawObservedPeakMedianGBS != 300 || got.MeasuredSustainableGBS != 96 || !reflect.DeepEqual(got.PlateauWorkerCounts, []int{4, 8}) || got.SaturationKneeWorkerCount != 2 {
		t.Fatalf("isolated spike became sustainable: %+v", got)
	}
	if got.Points[2].EfficiencyVersusSustainablePeak <= 1 {
		t.Fatalf("raw spike was clipped instead of disclosed: %+v", got.Points[2])
	}
}

func TestBuildRooflineSweepMeasurementRejectsLaterCollapse(t *testing.T) {
	o := testSweepOptions()
	_, err := buildRooflineSweepMeasurement(o, []rooflinePointResult{
		testRooflineResult(1, 65, 70, 75),
		testRooflineResult(2, 95, 100, 105),
		testRooflineResult(4, 93, 98, 103),
		testRooflineResult(8, 35, 40, 45),
	})
	if err == nil || !strings.Contains(err.Error(), "no stable knee") || !strings.Contains(err.Error(), "later valid point") {
		t.Fatalf("collapse error=%v", err)
	}
}

func TestBuildRooflineSweepMeasurementOmitsMalformedPointWithoutZeroPlaceholder(t *testing.T) {
	o := testSweepOptions()
	got, err := buildRooflineSweepMeasurement(o, []rooflinePointResult{
		testRooflineResult(1, 65, 70, 75),
		testRooflineResult(2, 0, 0, 0),
		testRooflineResult(4, 95, 100, 105),
		testRooflineResult(8, 93, 98, 103),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.RequestedPointCount != 4 || got.PointCount != 3 || got.OmittedPointCount != 1 || len(got.OmittedPoints) != 1 {
		t.Fatalf("omission summary=%+v", got)
	}
	if got.RawObservedPeakStatus != "not_identifiable" || got.SaturationKneeStatus != "not_identifiable" {
		t.Fatalf("omitted point retained an exact finding: %+v", got)
	}
	if got.OmittedPoints[0].WorkerCount != 2 || got.OmittedPoints[0].Reason == "" || !strings.Contains(got.OmittedPoints[0].Reason, "positive") {
		t.Fatalf("omission=%+v", got.OmittedPoints[0])
	}
	for _, point := range got.Points {
		if point.WorkerCount == 0 || point.MedianGBS == 0 || point.WorkerCount == 2 {
			t.Fatalf("malformed point retained or zero-filled: %+v", point)
		}
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"worker_count":0`) || strings.Contains(string(data), `"median_gb_s":0`) {
		t.Fatalf("omitted point serialized as zero placeholder: %s", data)
	}
}

func TestBuildRooflineSweepMeasurementRequiresEnoughValidPointsAndTerminalPoint(t *testing.T) {
	o := testSweepOptions()
	_, err := buildRooflineSweepMeasurement(o, []rooflinePointResult{
		testRooflineResult(1, 65, 70, 75),
		{workerCount: 2, err: errors.New("malformed point")},
		{workerCount: 4, err: errors.New("copy timeout")},
		testRooflineResult(8, 90, 95, 100),
	})
	if err == nil || !strings.Contains(err.Error(), "requires at least 3") || !strings.Contains(err.Error(), "workers=2") {
		t.Fatalf("insufficient-point error=%v", err)
	}
	_, err = buildRooflineSweepMeasurement(o, []rooflinePointResult{
		testRooflineResult(1, 65, 70, 75),
		testRooflineResult(2, 85, 90, 95),
		testRooflineResult(4, 95, 100, 105),
		{workerCount: 8, err: context.DeadlineExceeded},
	})
	if err == nil || !strings.Contains(err.Error(), "terminal worker_count=8") || !strings.Contains(err.Error(), "deadline exceeded") {
		t.Fatalf("terminal-point error=%v", err)
	}
}

func TestRooflineCopyAndCalibrationHonorCanceledContext(t *testing.T) {
	src, dst := make([]byte, MinRooflineWorkingSet), make([]byte, MinRooflineWorkingSet)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	started := time.Now()
	if _, err := runRooflineCopyBatch(ctx, src, dst, 1, math.MaxUint32, 10*time.Millisecond); !errors.Is(err, context.Canceled) {
		t.Fatalf("copy batch error=%v", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("canceled copy batch ran too long: %s", elapsed)
	}
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: 3, TargetDuration: 10 * time.Millisecond, Threads: 1}
	if _, _, _, _, err := calibrateRooflineIterations(ctx, o, src, dst); !errors.Is(err, context.Canceled) {
		t.Fatalf("calibration error=%v", err)
	}
}

func TestApplyHostRooflineMeasurementDoesNotContaminateAMDDeviceObservation(t *testing.T) {
	read, write := 842.0, 126.0
	collection := BandwidthCollection{Capture: BandwidthCapture{
		Schema:  BandwidthSchema,
		Engine:  "fak-native",
		Trigger: TriggerConfig{SymptomWindow: 2, ResourceWindow: 2, LatencyThresholdMS: 100, ResourceUtilization: .8},
		Samples: []BandwidthSample{
			{Phase: PhaseDecode, Shape: ShapeMedium, Provenance: BandwidthProvenance{Source: "live-amd-rocm", Device: "AMD ROCm GPU 0"}, Live: LiveBandwidth{ReadGBS: &read, WriteGBS: &write}},
		},
	}}
	measurement := newRooflineMeasurement(RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: 3, TargetDuration: 10 * time.Millisecond, Threads: 1}, RooflineMeasurementSchema, time.Second)
	measurement.MeasuredSustainableGBS = 1000
	if err := ApplyHostRooflineMeasurement(&collection, measurement); err != nil {
		t.Fatal(err)
	}
	if collection.RooflineMeasurement == nil || collection.RooflineMeasurement.AppliedHostObservationCount != 0 {
		t.Fatalf("measurement application=%+v", collection.RooflineMeasurement)
	}
	amd := collection.Report.Observations[0]
	if amd.Rooflines.MeasuredSustainableGBS != nil || amd.Rooflines.SelectedGBS != nil || amd.Live.Utilization != nil || amd.Live.TotalGBS == nil || *amd.Live.TotalGBS != 968 {
		t.Fatalf("host roofline contaminated AMD device observation: roofline=%+v live=%+v", amd.Rooflines, amd.Live)
	}
}

func TestRooflineSweepSchemaPreservesEvidenceAndCaveats(t *testing.T) {
	o := testSweepOptions()
	got, err := buildRooflineSweepMeasurement(o, []rooflinePointResult{
		testRooflineResult(1, 65, 70, 75),
		testRooflineResult(2, 88, 90, 92),
		testRooflineResult(4, 95, 100, 105),
		testRooflineResult(8, 96, 101, 106),
	})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{
		`"schema":"fak-host-memory-roofline-sweep/1"`,
		`"method":"parallel-copy"`,
		`"aggregation":"median"`,
		`"traffic_accounting":"read-plus-write-2-bytes-per-copied-byte"`,
		`"overhead_accounting":"first-touch-and-calibration-reported-separately-from-measured-trials"`,
		`"dram_isolation":"not-proven"`,
		`"interpretation":"sustained-host-memory-copy-throughput-not-hardware-counter-dram-bandwidth"`,
		`"knee_threshold":0.9`,
		`"raw_observed_peak_worker_count":8`,
		`"raw_observed_peak_median_gb_s":101`,
		`"raw_observed_peak_status":"measured"`,
		`"plateau_worker_counts":[4,8]`,
		`"saturation_knee_worker_count":2`,
		`"saturation_knee_status":"measured"`,
		`"efficiency_versus_sustainable_peak":`,
		`"marginal_gain_gb_s":`,
		`"calibration_duration_ms":`,
		`"calibration_traffic_bytes":`,
		`"buffers-reallocated-and-first-touched-per-worker-count"`,
		`"invalid-points-are-omitted-and-reported-not-zero-filled"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("schema missing %s: %s", want, text)
		}
	}
}

func testSweepOptions() RooflineBenchmarkOptions {
	return RooflineBenchmarkOptions{
		WorkingSetBytes: MinRooflineWorkingSet,
		Trials:          3,
		TargetDuration:  10 * time.Millisecond,
		Threads:         8,
		Sweep:           true,
		KneeThreshold:   0.90,
	}
}

func testRooflineResult(workers int, values ...float64) rooflinePointResult {
	return rooflinePointResult{workerCount: workers, point: testRooflinePoint(workers, values...)}
}

func testRooflinePoint(workers int, values ...float64) RooflineSweepPoint {
	trials := make([]RooflineTrial, len(values))
	for i, value := range values {
		trials[i] = RooflineTrial{
			Index:                   i + 1,
			Iterations:              1,
			DurationMS:              1,
			TrafficBytes:            2,
			GBS:                     value,
			CalibrationRounds:       1,
			CalibrationDurationMS:   1,
			CalibrationTrafficBytes: 2,
		}
	}
	return RooflineSweepPoint{WorkerCount: workers, WarmupDurationMS: 1, WarmupBytesTouched: 3, Trials: trials}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func stubFastRooflineCopyBatches(t *testing.T) {
	t.Helper()
	old := executeRooflineCopyBatch
	executeRooflineCopyBatch = func(ctx context.Context, _, _ []byte, _ int, _ uint64, target time.Duration) (time.Duration, error) {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		return target / 2, nil
	}
	t.Cleanup(func() { executeRooflineCopyBatch = old })
}
