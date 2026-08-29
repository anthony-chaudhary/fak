package modelperfobs

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sweepcert"
)

const RooflineMeasurementSchema = "fak-host-memory-roofline/1"
const RooflineSweepMeasurementSchema = "fak-host-memory-roofline-sweep/1"

const (
	MinRooflineWorkingSet        = 1 << 20
	MaxRooflineWorkingSet        = 512 << 20
	MinRooflineTrials            = 3
	MaxRooflineTrials            = 15
	MaxRooflineThreads           = 256
	MaxRooflineSweepPoints       = 9
	MinRooflineSweepValidPoints  = 3
	MaxRooflineCalibrationRounds = 8
	DefaultRooflineKneeThreshold = 0.90

	MaxRooflineRuntimeBudget = 5 * time.Minute
	rooflineWarmupBudget     = 10 * time.Second
	rooflineCopyChunk        = 8 << 20
)

const rooflineStabilityRule = "maximum-lower-median-of-two-adjacent-valid-points; knee-is-earliest-valid-point-with-it-and-all-later-valid-points-at-or-above-threshold-times-sustainable-peak"

type RooflineBenchmarkOptions struct {
	WorkingSetBytes uint64
	Trials          int
	TargetDuration  time.Duration
	Threads         int
	Sweep           bool
	KneeThreshold   float64
}

type RooflineTrial struct {
	Index                   int     `json:"index"`
	Iterations              uint64  `json:"iterations"`
	DurationMS              float64 `json:"duration_ms"`
	TrafficBytes            uint64  `json:"traffic_bytes"`
	GBS                     float64 `json:"gb_s"`
	CalibrationRounds       int     `json:"calibration_rounds"`
	CalibrationDurationMS   float64 `json:"calibration_duration_ms"`
	CalibrationTrafficBytes uint64  `json:"calibration_traffic_bytes"`
}

type RooflineSweepPoint struct {
	WorkerCount                     int             `json:"worker_count"`
	MedianGBS                       float64         `json:"median_gb_s"`
	EfficiencyVersusSustainablePeak float64         `json:"efficiency_versus_sustainable_peak"`
	MarginalGainGBS                 *float64        `json:"marginal_gain_gb_s,omitempty"`
	MarginalGainFraction            *float64        `json:"marginal_gain_fraction,omitempty"`
	WarmupDurationMS                float64         `json:"warmup_duration_ms"`
	WarmupBytesTouched              uint64          `json:"warmup_bytes_touched"`
	BufferCleanupDurationMS         float64         `json:"buffer_cleanup_duration_ms,omitempty"`
	Trials                          []RooflineTrial `json:"trials"`
}

type RooflineSweepOmission struct {
	WorkerCount int    `json:"worker_count"`
	Reason      string `json:"reason"`
}

type RooflineMeasurement struct {
	Schema                      string                  `json:"schema"`
	Scope                       string                  `json:"scope"`
	MachineClass                string                  `json:"machine_class"`
	Method                      string                  `json:"method"`
	Aggregation                 string                  `json:"aggregation"`
	TrafficAccounting           string                  `json:"traffic_accounting"`
	OverheadAccounting          string                  `json:"overhead_accounting"`
	DRAMIsolation               string                  `json:"dram_isolation"`
	Interpretation              string                  `json:"interpretation"`
	Caveats                     []string                `json:"caveats"`
	WorkingSetBytes             uint64                  `json:"working_set_bytes"`
	PeakBufferBytes             uint64                  `json:"peak_buffer_bytes"`
	TargetDurationMS            int64                   `json:"target_duration_ms"`
	RuntimeBudgetMS             int64                   `json:"runtime_budget_ms"`
	CommandBudgetMS             int64                   `json:"command_budget_ms,omitempty"`
	TotalDurationMS             float64                 `json:"total_duration_ms"`
	WarmupDurationMS            float64                 `json:"warmup_duration_ms,omitempty"`
	WarmupBytesTouched          uint64                  `json:"warmup_bytes_touched,omitempty"`
	Threads                     int                     `json:"threads"`
	MeasuredSustainableGBS      float64                 `json:"measured_sustainable_gb_s"`
	AppliedHostObservationCount int                     `json:"applied_host_observation_count"`
	Trials                      []RooflineTrial         `json:"trials,omitempty"`
	RequestedPointCount         int                     `json:"requested_point_count,omitempty"`
	PointCount                  int                     `json:"point_count,omitempty"`
	OmittedPointCount           int                     `json:"omitted_point_count,omitempty"`
	OmittedPoints               []RooflineSweepOmission `json:"omitted_points,omitempty"`
	KneeThreshold               float64                 `json:"knee_threshold,omitempty"`
	StabilityRule               string                  `json:"stability_rule,omitempty"`
	RawObservedPeakWorkerCount  int                     `json:"raw_observed_peak_worker_count,omitempty"`
	RawObservedPeakMedianGBS    float64                 `json:"raw_observed_peak_median_gb_s,omitempty"`
	EnvelopeDigest              string                  `json:"envelope_digest,omitempty"`
	RawObservedPeakStatus       string                  `json:"raw_observed_peak_status,omitempty"`
	PlateauWorkerCounts         []int                   `json:"plateau_worker_counts,omitempty"`
	SaturationKneeWorkerCount   int                     `json:"saturation_knee_worker_count,omitempty"`
	SaturationKneeStatus        string                  `json:"saturation_knee_status,omitempty"`
	SaturationKneeReason        string                  `json:"saturation_knee_reason,omitempty"`
	Points                      []RooflineSweepPoint    `json:"points,omitempty"`
}

type rooflinePointResult struct {
	workerCount int
	point       RooflineSweepPoint
	err         error
}

var allocateRooflinePointBuffers = func(n int) ([]byte, []byte) {
	return make([]byte, n), make([]byte, n)
}

var executeRooflineCopyBatch = runRooflineCopyBatch

func DefaultRooflineBenchmarkOptions() RooflineBenchmarkOptions {
	return RooflineBenchmarkOptions{
		WorkingSetBytes: 64 << 20,
		Trials:          5,
		TargetDuration:  100 * time.Millisecond,
		Threads:         runtime.GOMAXPROCS(0),
		KneeThreshold:   DefaultRooflineKneeThreshold,
	}
}

func ValidateRooflineBenchmarkOptions(o RooflineBenchmarkOptions) error {
	if o.WorkingSetBytes < MinRooflineWorkingSet || o.WorkingSetBytes > MaxRooflineWorkingSet {
		return fmt.Errorf("roofline working set must be %d..%d bytes", MinRooflineWorkingSet, MaxRooflineWorkingSet)
	}
	if o.Trials < MinRooflineTrials || o.Trials > MaxRooflineTrials {
		return fmt.Errorf("roofline trials must be %d..%d", MinRooflineTrials, MaxRooflineTrials)
	}
	if o.Threads < 1 || o.Threads > MaxRooflineThreads {
		return fmt.Errorf("roofline threads must be 1..%d", MaxRooflineThreads)
	}
	if uint64(o.Threads) > o.WorkingSetBytes {
		return fmt.Errorf("roofline threads exceed working set bytes")
	}
	if o.TargetDuration < 10*time.Millisecond || o.TargetDuration > 2*time.Second {
		return fmt.Errorf("roofline target duration must be 10ms..2s")
	}
	if o.Sweep {
		if math.IsNaN(o.KneeThreshold) || math.IsInf(o.KneeThreshold, 0) || o.KneeThreshold <= 0 || o.KneeThreshold > 1 {
			return fmt.Errorf("roofline knee threshold must be greater than 0 and at most 1")
		}
		counts, err := rooflineSweepWorkerCounts(o.Threads)
		if err != nil {
			return err
		}
		if len(counts) < MinRooflineSweepValidPoints {
			return fmt.Errorf("roofline sweep requires at least %d worker-count points", MinRooflineSweepValidPoints)
		}
		if len(counts) > MaxRooflineSweepPoints {
			return fmt.Errorf("roofline sweep point count %d exceeds maximum %d", len(counts), MaxRooflineSweepPoints)
		}
	}
	budget, err := rooflineRuntimeBudgetUnchecked(o)
	if err != nil {
		return err
	}
	if budget > MaxRooflineRuntimeBudget {
		return fmt.Errorf("roofline worst-case runtime budget %s exceeds maximum %s", budget, MaxRooflineRuntimeBudget)
	}
	return nil
}

// RooflineRuntimeBudget returns the hard context budget used by the benchmark.
// It includes first-touch, bounded calibration batches, and measured copy batches.
func RooflineRuntimeBudget(o RooflineBenchmarkOptions) (time.Duration, error) {
	if err := ValidateRooflineBenchmarkOptions(o); err != nil {
		return 0, err
	}
	return rooflineRuntimeBudgetUnchecked(o)
}

func rooflineRuntimeBudgetUnchecked(o RooflineBenchmarkOptions) (time.Duration, error) {
	points := 1
	if o.Sweep {
		counts, err := rooflineSweepWorkerCounts(o.Threads)
		if err != nil {
			return 0, err
		}
		points = len(counts)
	}
	batchBudget := rooflineCopyBatchBudget(o.TargetDuration)
	perTrial, ok := multiplyDuration(batchBudget, MaxRooflineCalibrationRounds+1)
	if !ok {
		return 0, fmt.Errorf("roofline runtime budget overflow")
	}
	trialBudget, ok := multiplyDuration(perTrial, o.Trials)
	if !ok || trialBudget > time.Duration(math.MaxInt64)-rooflineWarmupBudget {
		return 0, fmt.Errorf("roofline runtime budget overflow")
	}
	perPoint := rooflineWarmupBudget + trialBudget
	total, ok := multiplyDuration(perPoint, points)
	if !ok {
		return 0, fmt.Errorf("roofline runtime budget overflow")
	}
	return total, nil
}

func multiplyDuration(d time.Duration, n int) (time.Duration, bool) {
	if d < 0 || n < 0 || (n > 0 && d > time.Duration(math.MaxInt64)/time.Duration(n)) {
		return 0, false
	}
	return d * time.Duration(n), true
}

func rooflineCopyBatchBudget(target time.Duration) time.Duration {
	slop := target / 2
	if slop < 50*time.Millisecond {
		slop = 50 * time.Millisecond
	}
	return target + slop
}

// MeasureHostMemoryRoofline measures sustained copy bandwidth. Each copy counts
// one byte read and one byte written. It is a host-memory benchmark, not a live
// process DRAM counter and not a GPU HBM measurement.
func MeasureHostMemoryRoofline(ctx context.Context, o RooflineBenchmarkOptions) (RooflineMeasurement, error) {
	budget, err := RooflineRuntimeBudget(o)
	if err != nil {
		return RooflineMeasurement{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	started := time.Now()
	if !o.Sweep {
		point, err := measureHostMemoryRooflinePoint(ctx, o)
		if err != nil {
			return RooflineMeasurement{}, err
		}
		measurement := newRooflineMeasurement(o, RooflineMeasurementSchema, budget)
		measurement.MeasuredSustainableGBS = point.MedianGBS
		measurement.WarmupDurationMS = point.WarmupDurationMS
		measurement.WarmupBytesTouched = point.WarmupBytesTouched
		measurement.Trials = point.Trials
		measurement.TotalDurationMS = durationMS(time.Since(started))
		return measurement, nil
	}
	counts, err := rooflineSweepWorkerCounts(o.Threads)
	if err != nil {
		return RooflineMeasurement{}, err
	}
	results := make([]rooflinePointResult, 0, len(counts))
	for i, workers := range counts {
		pointOptions := o
		pointOptions.Threads = workers
		pointOptions.Sweep = false
		point, pointErr := measureHostMemoryRooflinePoint(ctx, pointOptions)
		if i < len(counts)-1 {
			cleanupStarted := time.Now()
			runtime.GC()
			point.BufferCleanupDurationMS = durationMS(time.Since(cleanupStarted))
		}
		results = append(results, rooflinePointResult{workerCount: workers, point: point, err: pointErr})
	}
	measurement, err := buildRooflineSweepMeasurement(o, results)
	if err != nil {
		return RooflineMeasurement{}, err
	}
	measurement.RuntimeBudgetMS = budget.Milliseconds()
	measurement.TotalDurationMS = durationMS(time.Since(started))
	return measurement, nil
}

func measureHostMemoryRooflinePoint(ctx context.Context, o RooflineBenchmarkOptions) (RooflineSweepPoint, error) {
	select {
	case <-ctx.Done():
		return RooflineSweepPoint{}, ctx.Err()
	default:
	}
	n := int(o.WorkingSetBytes)
	src, dst := allocateRooflinePointBuffers(n)
	warmupStarted := time.Now()
	warmupRun, cancelWarmup := context.WithTimeout(ctx, rooflineWarmupBudget)
	err := firstTouchRooflineBuffers(warmupRun, src, dst, o.Threads)
	cancelWarmup()
	warmupDuration := time.Since(warmupStarted)
	if err != nil {
		return RooflineSweepPoint{}, fmt.Errorf("roofline first-touch worker_count=%d: %w", o.Threads, err)
	}
	warmupBytes, ok := multiplyUint64(o.WorkingSetBytes, 3)
	if !ok {
		return RooflineSweepPoint{}, fmt.Errorf("roofline first-touch byte count overflow")
	}
	point := RooflineSweepPoint{
		WorkerCount:        o.Threads,
		WarmupDurationMS:   durationMS(warmupDuration),
		WarmupBytesTouched: warmupBytes,
		Trials:             make([]RooflineTrial, 0, o.Trials),
	}
	for trial := 0; trial < o.Trials; trial++ {
		select {
		case <-ctx.Done():
			return RooflineSweepPoint{}, ctx.Err()
		default:
		}
		iterations, calibrationRounds, calibrationDuration, calibrationTraffic, err := calibrateRooflineIterations(ctx, o, src, dst)
		if err != nil {
			return RooflineSweepPoint{}, fmt.Errorf("roofline trial %d calibration: %w", trial+1, err)
		}
		elapsed, err := executeRooflineCopyBatch(ctx, src, dst, o.Threads, iterations, o.TargetDuration)
		if err != nil {
			return RooflineSweepPoint{}, fmt.Errorf("roofline trial %d measured batch: %w", trial+1, err)
		}
		traffic, err := rooflineTrafficBytes(o.WorkingSetBytes, iterations)
		if err != nil {
			return RooflineSweepPoint{}, fmt.Errorf("roofline trial %d: %w", trial+1, err)
		}
		gbs := float64(traffic) / elapsed.Seconds() / 1e9
		result := RooflineTrial{
			Index:                   trial + 1,
			Iterations:              iterations,
			DurationMS:              durationMS(elapsed),
			TrafficBytes:            traffic,
			GBS:                     gbs,
			CalibrationRounds:       calibrationRounds,
			CalibrationDurationMS:   durationMS(calibrationDuration),
			CalibrationTrafficBytes: calibrationTraffic,
		}
		if err := validateRooflineTrial(result, trial+1); err != nil {
			return RooflineSweepPoint{}, fmt.Errorf("roofline trial %d: %w", trial+1, err)
		}
		point.Trials = append(point.Trials, result)
	}
	median, err := medianRooflineGBS(point.Trials)
	if err != nil {
		return RooflineSweepPoint{}, err
	}
	point.MedianGBS = median
	runtime.KeepAlive(src)
	runtime.KeepAlive(dst)
	return point, nil
}

func firstTouchRooflineBuffers(ctx context.Context, src, dst []byte, threads int) error {
	if len(src) == 0 || len(dst) != len(src) {
		return fmt.Errorf("roofline point buffers do not match working set")
	}
	var wg sync.WaitGroup
	errCh := make(chan error, threads)
	for worker := 0; worker < threads; worker++ {
		lo := len(src) * worker / threads
		hi := len(src) * (worker + 1) / threads
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for block := lo; block < hi; block += rooflineCopyChunk {
				select {
				case <-ctx.Done():
					errCh <- ctx.Err()
					return
				default:
				}
				end := block + rooflineCopyChunk
				if end > hi {
					end = hi
				}
				for i := block; i < end; i++ {
					src[i] = byte(i*131 + 17)
				}
				copy(dst[block:end], src[block:end])
			}
		}(lo, hi)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}

func calibrateRooflineIterations(ctx context.Context, o RooflineBenchmarkOptions, src, dst []byte) (uint64, int, time.Duration, uint64, error) {
	iterations := uint64(1)
	var totalDuration time.Duration
	var totalTraffic uint64
	for round := 1; round <= MaxRooflineCalibrationRounds; round++ {
		elapsed, err := executeRooflineCopyBatch(ctx, src, dst, o.Threads, iterations, o.TargetDuration)
		if err != nil {
			return 0, round, totalDuration, totalTraffic, err
		}
		traffic, err := rooflineTrafficBytes(o.WorkingSetBytes, iterations)
		if err != nil {
			return 0, round, totalDuration, totalTraffic, err
		}
		var ok bool
		totalDuration, ok = addDuration(totalDuration, elapsed)
		if !ok {
			return 0, round, 0, 0, fmt.Errorf("calibration duration overflow")
		}
		totalTraffic, ok = addRooflineUint64(totalTraffic, traffic)
		if !ok {
			return 0, round, 0, 0, fmt.Errorf("calibration traffic byte count overflow")
		}
		if elapsed <= 0 {
			if round == MaxRooflineCalibrationRounds {
				return 0, round, totalDuration, totalTraffic, fmt.Errorf("calibration batch duration must be positive")
			}
			if iterations > math.MaxUint64/2 {
				return 0, round, totalDuration, totalTraffic, fmt.Errorf("iteration count overflow")
			}
			iterations *= 2
			continue
		}
		// TargetDuration guides calibration; it is not a validity threshold. At
		// the round limit, retain the last iteration count whose batch completed
		// within the hard batch/context bounds instead of using a new, unexecuted
		// estimate or omitting an otherwise valid fast-copy point.
		if elapsed >= o.TargetDuration || round == MaxRooflineCalibrationRounds {
			return iterations, round, totalDuration, totalTraffic, nil
		}
		nextFloat := float64(iterations) * float64(o.TargetDuration) / float64(elapsed) * 1.05
		if math.IsNaN(nextFloat) || math.IsInf(nextFloat, 0) || nextFloat > float64(math.MaxUint64) {
			return 0, round, totalDuration, totalTraffic, fmt.Errorf("iteration estimate overflow")
		}
		next := uint64(nextFloat)
		if next <= iterations {
			if iterations == math.MaxUint64 {
				return 0, round, totalDuration, totalTraffic, fmt.Errorf("iteration count overflow")
			}
			next = iterations + 1
		}
		if _, err := rooflineTrafficBytes(o.WorkingSetBytes, next); err != nil {
			return 0, round, totalDuration, totalTraffic, err
		}
		iterations = next
	}
	return 0, MaxRooflineCalibrationRounds, totalDuration, totalTraffic, fmt.Errorf("calibration produced no bounded batch")
}

func runRooflineCopyBatch(ctx context.Context, src, dst []byte, threads int, iterations uint64, target time.Duration) (time.Duration, error) {
	if len(src) == 0 || len(dst) != len(src) {
		return 0, fmt.Errorf("roofline point buffers do not match working set")
	}
	if threads < 1 {
		return 0, fmt.Errorf("copy batch workers must be positive")
	}
	if iterations == 0 {
		return 0, fmt.Errorf("copy batch iterations must be positive")
	}
	batchRun, cancel := context.WithTimeout(ctx, rooflineCopyBatchBudget(target))
	defer cancel()
	started := time.Now()
	var wg sync.WaitGroup
	errCh := make(chan error, threads)
	for worker := 0; worker < threads; worker++ {
		lo := len(src) * worker / threads
		hi := len(src) * (worker + 1) / threads
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for iteration := uint64(0); iteration < iterations; iteration++ {
				if err := copyRooflineRange(batchRun, dst, src, lo, hi); err != nil {
					errCh <- err
					return
				}
			}
		}(lo, hi)
	}
	wg.Wait()
	elapsed := time.Since(started)
	close(errCh)
	for err := range errCh {
		if err != nil {
			return elapsed, err
		}
	}
	if err := batchRun.Err(); err != nil {
		return elapsed, err
	}
	return elapsed, nil
}

func copyRooflineRange(ctx context.Context, dst, src []byte, lo, hi int) error {
	for block := lo; block < hi; block += rooflineCopyChunk {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		end := block + rooflineCopyChunk
		if end > hi {
			end = hi
		}
		copy(dst[block:end], src[block:end])
	}
	return nil
}

func rooflineTrafficBytes(workingSetBytes, iterations uint64) (uint64, error) {
	bytesPerIteration, ok := multiplyUint64(workingSetBytes, 2)
	if !ok || bytesPerIteration == 0 || iterations > math.MaxUint64/bytesPerIteration {
		return 0, fmt.Errorf("traffic byte count overflow")
	}
	return bytesPerIteration * iterations, nil
}

func multiplyUint64(v, n uint64) (uint64, bool) {
	if n != 0 && v > math.MaxUint64/n {
		return 0, false
	}
	return v * n, true
}

func addRooflineUint64(a, b uint64) (uint64, bool) {
	if a > math.MaxUint64-b {
		return 0, false
	}
	return a + b, true
}

func addDuration(a, b time.Duration) (time.Duration, bool) {
	if b > 0 && a > time.Duration(math.MaxInt64)-b {
		return 0, false
	}
	return a + b, true
}

func rooflineSweepWorkerCounts(threads int) ([]int, error) {
	if threads < 1 || threads > MaxRooflineThreads {
		return nil, fmt.Errorf("roofline threads must be 1..%d", MaxRooflineThreads)
	}
	counts := []int{1}
	for counts[len(counts)-1] < threads {
		next := counts[len(counts)-1] * 2
		if next > threads {
			next = threads
		}
		counts = append(counts, next)
		if len(counts) > MaxRooflineSweepPoints {
			return nil, fmt.Errorf("roofline sweep point count exceeds maximum %d", MaxRooflineSweepPoints)
		}
	}
	return counts, nil
}

func buildRooflineSweepMeasurement(o RooflineBenchmarkOptions, results []rooflinePointResult) (RooflineMeasurement, error) {
	counts, err := rooflineSweepWorkerCounts(o.Threads)
	if err != nil {
		return RooflineMeasurement{}, err
	}
	if len(results) != len(counts) {
		return RooflineMeasurement{}, fmt.Errorf("roofline sweep produced %d point results; expected %d", len(results), len(counts))
	}
	points := make([]RooflineSweepPoint, 0, len(results))
	omissions := make([]RooflineSweepOmission, 0)
	terminalValid := false
	for i, result := range results {
		expectedWorkers := counts[i]
		if result.workerCount != expectedWorkers {
			omissions = append(omissions, newRooflineOmission(expectedWorkers, fmt.Errorf("result worker_count=%d; expected %d", result.workerCount, expectedWorkers)))
			continue
		}
		if result.err != nil {
			omissions = append(omissions, newRooflineOmission(expectedWorkers, result.err))
			continue
		}
		point := result.point
		if point.WorkerCount != expectedWorkers {
			omissions = append(omissions, newRooflineOmission(expectedWorkers, fmt.Errorf("point worker_count=%d; expected %d", point.WorkerCount, expectedWorkers)))
			continue
		}
		if len(point.Trials) != o.Trials {
			omissions = append(omissions, newRooflineOmission(expectedWorkers, fmt.Errorf("trial count=%d; expected %d", len(point.Trials), o.Trials)))
			continue
		}
		median, medianErr := medianRooflineGBS(point.Trials)
		if medianErr != nil {
			omissions = append(omissions, newRooflineOmission(expectedWorkers, medianErr))
			continue
		}
		point.MedianGBS = median
		points = append(points, point)
		terminalValid = i == len(results)-1
	}
	if len(points) < MinRooflineSweepValidPoints {
		return RooflineMeasurement{}, fmt.Errorf("roofline sweep has %d valid points; requires at least %d (%s)", len(points), MinRooflineSweepValidPoints, formatRooflineOmissions(omissions))
	}
	if !terminalValid {
		return RooflineMeasurement{}, fmt.Errorf("roofline sweep terminal worker_count=%d was omitted (%s)", counts[len(counts)-1], formatRooflineOmissions(omissions))
	}

	rawPeakIndex := 0
	for i := 1; i < len(points); i++ {
		if points[i].MedianGBS > points[rawPeakIndex].MedianGBS {
			rawPeakIndex = i
		}
	}
	sustainable := math.Inf(-1)
	plateauIndex := -1
	for i := 1; i < len(points); i++ {
		candidate := math.Min(points[i-1].MedianGBS, points[i].MedianGBS)
		if candidate > sustainable {
			sustainable = candidate
			plateauIndex = i - 1
		}
	}
	if plateauIndex < 0 || math.IsNaN(sustainable) || math.IsInf(sustainable, 0) || sustainable <= 0 {
		return RooflineMeasurement{}, fmt.Errorf("roofline sweep has no stable adjacent-point plateau")
	}

	floor := o.KneeThreshold * sustainable
	evidence, err := rooflineSweepEvidence(o, counts, points, omissions)
	if err != nil {
		return RooflineMeasurement{}, err
	}
	rawPeakFinding := sweepcert.ObservedExtremum(evidence, "median_gb_s", sweepcert.Maximum)
	kneeFinding := sweepcert.StableSuffixThreshold(evidence, "median_gb_s", sweepcert.AtOrAbove, floor)
	if kneeFinding.Status == sweepcert.FindingNotIdentifiable && len(omissions) == 0 {
		return RooflineMeasurement{}, fmt.Errorf("roofline sweep has no stable knee: %s; later valid point fell below %.6g GB/s (threshold %.6g of sustainable peak %.6g)", kneeFinding.Reason, floor, o.KneeThreshold, sustainable)
	}

	for i := range points {
		points[i].EfficiencyVersusSustainablePeak = points[i].MedianGBS / sustainable
		if i > 0 {
			gain := points[i].MedianGBS - points[i-1].MedianGBS
			fraction := gain / points[i-1].MedianGBS
			points[i].MarginalGainGBS = &gain
			points[i].MarginalGainFraction = &fraction
		}
	}
	budget, err := rooflineRuntimeBudgetUnchecked(o)
	if err != nil {
		return RooflineMeasurement{}, err
	}
	measurement := newRooflineMeasurement(o, RooflineSweepMeasurementSchema, budget)
	measurement.MeasuredSustainableGBS = sustainable
	measurement.RequestedPointCount = len(counts)
	measurement.PointCount = len(points)
	measurement.OmittedPointCount = len(omissions)
	measurement.OmittedPoints = omissions
	measurement.KneeThreshold = o.KneeThreshold
	measurement.StabilityRule = rooflineStabilityRule
	measurement.EnvelopeDigest = evidence.EnvelopeDigest
	measurement.RawObservedPeakWorkerCount = points[rawPeakIndex].WorkerCount
	measurement.RawObservedPeakMedianGBS = points[rawPeakIndex].MedianGBS
	measurement.RawObservedPeakStatus = string(rawPeakFinding.Status)
	measurement.PlateauWorkerCounts = []int{points[plateauIndex].WorkerCount, points[plateauIndex+1].WorkerCount}
	measurement.SaturationKneeStatus = string(kneeFinding.Status)
	measurement.SaturationKneeReason = kneeFinding.Reason
	if kneeFinding.PointID != "" {
		for _, point := range points {
			if kneeFinding.PointID == "workers:"+strconv.Itoa(point.WorkerCount) {
				measurement.SaturationKneeWorkerCount = point.WorkerCount
				break
			}
		}
	}
	measurement.Points = points
	return measurement, nil
}

func rooflineSweepEvidence(o RooflineBenchmarkOptions, counts []int, points []RooflineSweepPoint, omissions []RooflineSweepOmission) (sweepcert.Evidence, error) {
	coordinates := make([]float64, len(counts))
	for i, count := range counts {
		coordinates[i] = float64(count)
	}
	envelope := sweepcert.Envelope{
		Axis: sweepcert.Axis{Name: "host_memory_workers", Unit: "workers", Coordinates: coordinates, LowerClosed: true, UpperClosed: true},
		Bindings: []sweepcert.Binding{
			{Name: "model", Value: "none-host-memory-copy"},
			{Name: "workload", Value: "parallel-copy-read-plus-write"},
			{Name: "engine", Value: "go-runtime/" + runtime.Version()},
			{Name: "configuration", Value: fmt.Sprintf("working_set=%d;trials=%d;target_ms=%d;knee=%g", o.WorkingSetBytes, o.Trials, o.TargetDuration.Milliseconds(), o.KneeThreshold)},
			{Name: "capacity", Value: strconv.Itoa(o.Threads)},
			{Name: "reset_order", Value: "ascending-workers;reallocate-first-touch;gc-between-points"},
			{Name: "environment", Value: runtime.GOOS + "/" + runtime.GOARCH},
		},
	}
	digest, err := sweepcert.CanonicalEnvelopeDigest(envelope)
	if err != nil {
		return sweepcert.Evidence{}, err
	}
	evidence := sweepcert.Evidence{Envelope: envelope, EnvelopeDigest: digest}
	byWorkers := make(map[int]RooflineSweepPoint, len(points))
	for _, point := range points {
		byWorkers[point.WorkerCount] = point
	}
	omitted := make(map[int]bool, len(omissions))
	for _, omission := range omissions {
		omitted[omission.WorkerCount] = true
	}
	for _, count := range counts {
		point := sweepcert.Point{ID: "workers:" + strconv.Itoa(count), Coordinate: float64(count), Status: sweepcert.PointNotMeasured, EnvelopeDigest: digest}
		if measured, ok := byWorkers[count]; ok {
			value := measured.MedianGBS
			point.Status = sweepcert.PointMeasured
			point.Observations = map[string]sweepcert.Observation{"median_gb_s": {Status: sweepcert.ObservationMeasured, Value: &value, Provenance: sweepcert.Provenance{Source: "host-memory-roofline", Method: "median-parallel-copy", Unit: "GB/s", EnvelopeDigest: digest}}}
		} else if !omitted[count] {
			return sweepcert.Evidence{}, fmt.Errorf("roofline evidence has no point or omission for worker_count=%d", count)
		}
		evidence.Points = append(evidence.Points, point)
	}
	return evidence, nil
}

func newRooflineOmission(workers int, err error) RooflineSweepOmission {
	reason := "unknown point failure"
	if err != nil {
		reason = strings.Join(strings.Fields(err.Error()), " ")
		if reason == "" {
			reason = "unknown point failure"
		}
	}
	const maxReasonBytes = 240
	if len(reason) > maxReasonBytes {
		reason = reason[:maxReasonBytes]
	}
	return RooflineSweepOmission{WorkerCount: workers, Reason: reason}
}

func formatRooflineOmissions(omissions []RooflineSweepOmission) string {
	if len(omissions) == 0 {
		return "no omitted-point reason recorded"
	}
	parts := make([]string, len(omissions))
	for i, omission := range omissions {
		parts[i] = fmt.Sprintf("workers=%d: %s", omission.WorkerCount, omission.Reason)
	}
	return strings.Join(parts, "; ")
}

func medianRooflineGBS(trials []RooflineTrial) (float64, error) {
	if len(trials) == 0 {
		return 0, fmt.Errorf("roofline point has no trials")
	}
	values := make([]float64, len(trials))
	for i, trial := range trials {
		if err := validateRooflineTrial(trial, i+1); err != nil {
			return 0, err
		}
		values[i] = trial.GBS
	}
	sort.Float64s(values)
	middle := len(values) / 2
	median := values[middle]
	if len(values)%2 == 0 {
		median = values[middle-1]/2 + values[middle]/2
	}
	if math.IsNaN(median) || math.IsInf(median, 0) || median <= 0 {
		return 0, fmt.Errorf("roofline point median must be finite and positive")
	}
	return median, nil
}

func validateRooflineTrial(trial RooflineTrial, expectedIndex int) error {
	if trial.Index != expectedIndex {
		return fmt.Errorf("roofline trial index=%d; expected %d", trial.Index, expectedIndex)
	}
	if trial.Iterations == 0 {
		return fmt.Errorf("roofline trial iterations must be positive")
	}
	if math.IsNaN(trial.DurationMS) || math.IsInf(trial.DurationMS, 0) || trial.DurationMS <= 0 {
		return fmt.Errorf("roofline trial duration must be finite and positive")
	}
	if trial.TrafficBytes == 0 {
		return fmt.Errorf("roofline trial traffic bytes must be positive")
	}
	if math.IsNaN(trial.GBS) || math.IsInf(trial.GBS, 0) || trial.GBS <= 0 {
		return fmt.Errorf("roofline trial bandwidth must be finite and positive")
	}
	if trial.CalibrationRounds < 1 || trial.CalibrationRounds > MaxRooflineCalibrationRounds {
		return fmt.Errorf("roofline trial calibration rounds must be 1..%d", MaxRooflineCalibrationRounds)
	}
	if math.IsNaN(trial.CalibrationDurationMS) || math.IsInf(trial.CalibrationDurationMS, 0) || trial.CalibrationDurationMS <= 0 {
		return fmt.Errorf("roofline trial calibration duration must be finite and positive")
	}
	if trial.CalibrationTrafficBytes == 0 {
		return fmt.Errorf("roofline trial calibration traffic bytes must be positive")
	}
	return nil
}

func newRooflineMeasurement(o RooflineBenchmarkOptions, schema string, budget time.Duration) RooflineMeasurement {
	peakBufferBytes, _ := multiplyUint64(o.WorkingSetBytes, 2)
	caveats := []string{
		"cache-residency-not-measured",
		"numa-placement-os-managed",
		"copy-kernel-and-scheduler-dependent",
		"peak-buffer-bytes-excludes-go-runtime-and-os-retention",
	}
	if o.Sweep {
		caveats = append(caveats,
			"buffers-reallocated-and-first-touched-per-worker-count",
			"per-point-buffer-release-is-go-runtime-mediated",
			"invalid-points-are-omitted-and-reported-not-zero-filled",
		)
	}
	return RooflineMeasurement{
		Schema:             schema,
		Scope:              "host-memory",
		MachineClass:       runtime.GOOS + "/" + runtime.GOARCH,
		Method:             "parallel-copy",
		Aggregation:        "median",
		TrafficAccounting:  "read-plus-write-2-bytes-per-copied-byte",
		OverheadAccounting: "first-touch-and-calibration-reported-separately-from-measured-trials",
		DRAMIsolation:      "not-proven",
		Interpretation:     "sustained-host-memory-copy-throughput-not-hardware-counter-dram-bandwidth",
		Caveats:            caveats,
		WorkingSetBytes:    o.WorkingSetBytes,
		PeakBufferBytes:    peakBufferBytes,
		TargetDurationMS:   o.TargetDuration.Milliseconds(),
		RuntimeBudgetMS:    budget.Milliseconds(),
		Threads:            o.Threads,
	}
}

// ApplyHostRooflineMeasurement attaches a host-memory benchmark artifact and
// applies its selected roofline only to live host, imported host-controller,
// or package/system Apple observations. Device observations retain no host
// measured roofline.
func ApplyHostRooflineMeasurement(collection *BandwidthCollection, measurement RooflineMeasurement) error {
	if collection == nil {
		return fmt.Errorf("bandwidth collection is nil")
	}
	if measurement.Scope != "host-memory" || measurement.MeasuredSustainableGBS <= 0 || math.IsNaN(measurement.MeasuredSustainableGBS) || math.IsInf(measurement.MeasuredSustainableGBS, 0) {
		return fmt.Errorf("invalid host roofline measurement")
	}
	artifact := collection.HostControllerArtifact
	if artifact != nil && artifact.Scope.Kind != "system" {
		return fmt.Errorf("host memory roofline requires system-aggregate host counter scope, got %s", artifact.Scope.Kind)
	}
	appleArtifact := collection.AppleMemoryArtifact
	if appleArtifact != nil {
		if err := validateAppleMemoryScope(appleArtifact.Scope); err != nil {
			return fmt.Errorf("host memory roofline requires system/package Apple memory scope: %w", err)
		}
	}
	capture := collection.Capture
	capture.Samples = append([]BandwidthSample(nil), collection.Capture.Samples...)
	matching := make([]int, 0, len(capture.Samples))
	for i := range capture.Samples {
		sample := capture.Samples[i]
		isLiveHost := sample.Provenance.Source == "live-host" && sample.Provenance.Device == "host-memory"
		isImportedHostController := artifact != nil &&
			(sample.Provenance.Source == "host-controller-direct-bytes" || sample.Provenance.Source == "host-controller-converted-events") &&
			sample.Provenance.Device == "host-memory" &&
			sample.Provenance.Collector == artifact.Provider && sample.Provenance.SampledAt == artifact.CaptureEndedAt
		isImportedAppleMemory := appleArtifact != nil &&
			(sample.Provenance.Source == "apple-memory-direct-byte-rates" || sample.Provenance.Source == "apple-memory-monotonic-byte-deltas") &&
			sample.Provenance.Device == "apple-unified-memory" &&
			sample.Provenance.Collector == appleArtifact.Provider && sample.Provenance.SampledAt == appleArtifact.CaptureEndedAt &&
			sample.Live.Scope != nil &&
			(sample.Live.Scope.Kind == "system" || sample.Live.Scope.Kind == "package") &&
			sample.Live.Scope.Kind == appleArtifact.Scope.Kind && sample.Live.Scope.ID == appleArtifact.Scope.ID
		if isLiveHost || isImportedHostController || isImportedAppleMemory {
			matching = append(matching, i)
		}
	}
	if artifact != nil && len(matching) != 1 {
		return fmt.Errorf("system host controller artifact matches %d imported samples, want exactly one", len(matching))
	}
	if appleArtifact != nil && len(matching) != 1 {
		return fmt.Errorf("system/package Apple memory artifact matches %d imported samples, want exactly one", len(matching))
	}
	for _, i := range matching {
		capture.Samples[i].Rooflines.MeasuredSustainableGBS = cloneFloat(&measurement.MeasuredSustainableGBS)
	}
	measurement.AppliedHostObservationCount = len(matching)
	report, err := AnalyzeBandwidth(capture)
	if err != nil {
		return err
	}
	collection.Capture = capture
	collection.RooflineMeasurement = &measurement
	collection.Report = report
	return nil
}

func durationMS(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}
