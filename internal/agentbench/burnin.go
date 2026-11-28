package agentbench

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
)

const (
	defaultBurnInMaxGoroutineDelta  = 16
	defaultBurnInMaxHeapGrowthBytes = 64 << 20
	defaultBurnInMaxFDDelta         = 16
)

type burnInLeakBounds struct {
	MaxGoroutineDelta  int   `json:"max_goroutine_delta"`
	MaxHeapGrowthBytes int64 `json:"max_heap_growth_bytes"`
	MaxFDDelta         int   `json:"max_fd_delta"`
}

type burnInRuntimeSample struct {
	Goroutines int    `json:"goroutines"`
	HeapAlloc  uint64 `json:"heap_alloc_bytes"`
	OpenFDs    int    `json:"open_fds"`
	FDStatus   string `json:"fd_status"`
}

type burnInEpochLeakDelta struct {
	Ordinal        int      `json:"ordinal"`
	GoroutineDelta int      `json:"goroutine_delta"`
	HeapDeltaBytes int64    `json:"heap_delta_bytes"`
	FDDelta        int      `json:"fd_delta"`
	FDStatus       string   `json:"fd_status"`
	Verdict        string   `json:"verdict"`
	Breaches       []string `json:"breaches,omitempty"`
}

type burnInLeakReport struct {
	Verdict           string                 `json:"verdict"`
	Passed            bool                   `json:"passed"`
	Bounds            burnInLeakBounds       `json:"bounds"`
	Baseline          burnInRuntimeSample    `json:"baseline"`
	Final             burnInRuntimeSample    `json:"final"`
	MaxGoroutineDelta int                    `json:"max_goroutine_delta_observed"`
	MaxHeapDeltaBytes int64                  `json:"max_heap_delta_bytes_observed"`
	MaxFDDelta        int                    `json:"max_fd_delta_observed"`
	FDStatus          string                 `json:"fd_status"`
	EpochDeltas       []burnInEpochLeakDelta `json:"epoch_deltas,omitempty"`
	Breaches          []string               `json:"breaches,omitempty"`
}

type burnInIdentity struct {
	ModelID        string `json:"model_id"`
	ConfigDigest   string `json:"config_digest"`
	WorkloadDigest string `json:"workload_digest"`
}

type burnInNormalAdmission struct {
	Qualified   bool           `json:"qualified"`
	Concurrency int            `json:"concurrency"`
	Identity    burnInIdentity `json:"identity"`
}

type burnInEpochSpec struct {
	Ordinal                 int    `json:"ordinal"`
	Seed                    uint64 `json:"seed"`
	Concurrency             int    `json:"concurrency"`
	Sessions                int    `json:"sessions"`
	TurnsPerSession         int    `json:"turns_per_session"`
	Controls                int    `json:"controls"`
	Tasks                   int    `json:"tasks"`
	Areas                   int    `json:"areas"`
	MaxPriming              int    `json:"max_priming"`
	MaxRetries              int    `json:"max_retries"`
	RequirePressureRevisit  bool   `json:"require_pressure_revisit"`
	RequireSourceEditReread bool   `json:"require_source_edit_reread"`
	RequireCancelRecovery   bool   `json:"require_cancel_recovery"`
}

type burnInEpochReceipt struct {
	Ordinal                         int                    `json:"ordinal"`
	Complete                        bool                   `json:"complete"`
	Partial                         bool                   `json:"partial"`
	SteadyRequests                  int                    `json:"steady_requests"`
	ControlRequests                 int                    `json:"control_requests"`
	TaskAttempts                    int                    `json:"task_attempts"`
	AcceptedTasks                   int                    `json:"accepted_tasks"`
	PrimingRequests                 int                    `json:"priming_requests"`
	ReplayRetries                   int                    `json:"replay_retries"`
	AreaTransitions                 []burnInAreaTransition `json:"area_transitions,omitempty"`
	PressureCohorts                 []int                  `json:"pressure_cohorts,omitempty"`
	EventsPath                      string                 `json:"events_path,omitempty"`
	EventsDigest                    string                 `json:"events_digest,omitempty"`
	AreasVisited                    int                    `json:"areas_visited"`
	PressureStatus                  string                 `json:"pressure_status"`
	SourceEditRereadPassed          bool                   `json:"source_edit_reread_passed"`
	IntentionalCancellationObserved bool                   `json:"intentional_cancellation_observed"`
	RecoveryPassed                  bool                   `json:"recovery_passed"`
	CompletedAt                     time.Time              `json:"completed_at"`
	Errors                          []string               `json:"errors,omitempty"`
}

type burnInControllerConfig struct {
	Plan               profileplan.Plan
	Identity           burnInIdentity
	Normal             *burnInNormalAdmission
	RunNormalPreflight func(context.Context) (burnInNormalAdmission, error)
	Now                func() time.Time
	RunEpoch           func(context.Context, burnInEpochSpec, time.Time) burnInEpochReceipt
	Drain              func(context.Context, time.Duration) error
	AllowShortSchedule bool
	MaxEpochs          int
	LeakBounds         *burnInLeakBounds
	SampleRuntime      func() burnInRuntimeSample
}

type burnInReceipt struct {
	Status               string                `json:"status"`
	Qualified            bool                  `json:"qualified"`
	Identity             burnInIdentity        `json:"identity"`
	NormalAdmission      burnInNormalAdmission `json:"normal_admission"`
	NormalPreflightRun   bool                  `json:"normal_preflight_run"`
	FullEpochs           int                   `json:"full_epochs"`
	PartialEpochs        int                   `json:"partial_epochs"`
	SteadyRequests       int                   `json:"steady_requests"`
	ControlRequests      int                   `json:"control_requests"`
	TaskAttempts         int                   `json:"task_attempts"`
	AcceptedTasks        int                   `json:"accepted_tasks"`
	PrimingRequests      int                   `json:"priming_requests"`
	ReplayRetries        int                   `json:"replay_retries"`
	FullDurationObserved bool                  `json:"full_duration_observed"`
	DrainSucceeded       bool                  `json:"drain_succeeded"`
	LeakOracle           burnInLeakReport      `json:"leak_oracle"`
	Epochs               []burnInEpochReceipt  `json:"epochs"`
	Errors               []string              `json:"errors,omitempty"`
}

func runBurnInController(ctx context.Context, cfg burnInControllerConfig) (burnInReceipt, error) {
	r := burnInReceipt{Status: "INCOMPLETE", Identity: cfg.Identity}
	validSchedule := cfg.Plan.Deadline == 2*time.Hour || cfg.Plan.Deadline == 8*time.Hour
	validDrain := cfg.Plan.DrainGrace == 90*time.Second
	if cfg.AllowShortSchedule {
		validSchedule = cfg.Plan.Deadline > 0
		validDrain = cfg.Plan.DrainGrace > 0
	}
	if cfg.Plan.Profile != "burn-in" || !validSchedule || cfg.Plan.Epoch == nil || !validDrain || cfg.Now == nil || cfg.RunEpoch == nil || cfg.Drain == nil {
		return r, errors.New("burn-in controller configuration is incomplete")
	}
	admission := burnInNormalAdmission{}
	if cfg.Normal != nil {
		admission = *cfg.Normal
	}
	if !matchingBurnInNormal(admission, cfg.Identity, cfg.Plan.PreferredConcurrency) {
		if cfg.RunNormalPreflight == nil {
			return r, errors.New("burn-in requires matching qualifying normal admission")
		}
		r.NormalPreflightRun = true
		var err error
		admission, err = cfg.RunNormalPreflight(ctx)
		if err != nil || !matchingBurnInNormal(admission, cfg.Identity, cfg.Plan.PreferredConcurrency) {
			return r, errors.New("separately accounted normal preflight did not qualify")
		}
	}
	r.NormalAdmission = admission
	bounds := defaultBurnInLeakBounds()
	if cfg.LeakBounds != nil {
		bounds = *cfg.LeakBounds
	}
	sampleRuntime := sampleBurnInRuntime
	if cfg.SampleRuntime != nil {
		sampleRuntime = cfg.SampleRuntime
	}
	leak := newBurnInLeakReport(bounds, sampleRuntime())
	started := cfg.Now()
	cutoff := started.Add(cfg.Plan.Deadline)
	// Map the injected observation clock to one real execution deadline. Epochs
	// and cleanup share this bound; cleanup cannot renew grace already consumed.
	executionDeadline := time.Now().Add(cfg.Plan.Deadline + cfg.Plan.DrainGrace)
	epochCtx, cancelEpochs := context.WithDeadline(ctx, executionDeadline)
	defer cancelEpochs()
	for ordinal := 1; cfg.Now().Before(cutoff); ordinal++ {
		epochStarted := cfg.Now()
		spec := burnInSpec(cfg.Plan, ordinal)
		epoch := cfg.RunEpoch(epochCtx, spec, cutoff)
		r.Epochs = append(r.Epochs, epoch)
		r.SteadyRequests += epoch.SteadyRequests
		r.ControlRequests += epoch.ControlRequests
		r.TaskAttempts += epoch.TaskAttempts
		r.AcceptedTasks += epoch.AcceptedTasks
		r.PrimingRequests += epoch.PrimingRequests
		r.ReplayRetries += epoch.ReplayRetries
		if validFullBurnInEpoch(spec, epoch) {
			r.FullEpochs++
		} else {
			r.PartialEpochs++
			cutoffPartial := epoch.Partial && !epoch.Complete && !epoch.CompletedAt.Before(cutoff)
			if len(epoch.Errors) == 0 && !cutoffPartial {
				r.Errors = append(r.Errors, fmt.Sprintf("epoch %d incomplete", ordinal))
			}
		}
		r.Errors = append(r.Errors, epoch.Errors...)
		delta := leak.recordEpoch(spec.Ordinal, sampleRuntime())
		r.Errors = append(r.Errors, delta.Breaches...)
		if cfg.MaxEpochs > 0 && ordinal >= cfg.MaxEpochs {
			break
		}
		observed := cfg.Now()
		if epoch.CompletedAt.After(observed) {
			observed = epoch.CompletedAt
		}
		if !observed.Before(cutoff) {
			break
		}
		if !observed.After(epochStarted) {
			r.Errors = append(r.Errors, fmt.Sprintf("epoch %d made no clock progress", ordinal))
			break
		}
		if err := epochCtx.Err(); err != nil {
			r.Errors = append(r.Errors, err.Error())
			break
		}
	}
	r.FullDurationObserved = !cfg.Now().Before(cutoff) || (cfg.AllowShortSchedule && cfg.MaxEpochs > 0 && len(r.Epochs) >= cfg.MaxEpochs)
	remaining := min(cfg.Plan.DrainGrace, max(time.Duration(0), cutoff.Add(cfg.Plan.DrainGrace).Sub(cfg.Now())))
	drainDeadline := time.Now().Add(remaining)
	if executionDeadline.Before(drainDeadline) {
		drainDeadline = executionDeadline
	}
	drainCtx, cancel := context.WithDeadline(context.WithoutCancel(ctx), drainDeadline)
	drainErr := cfg.Drain(drainCtx, remaining)
	if drainErr == nil {
		drainErr = drainCtx.Err()
	}
	cancel()
	r.DrainSucceeded = drainErr == nil
	if drainErr != nil {
		r.Errors = append(r.Errors, "drain: "+drainErr.Error())
	}
	finalBreaches := leak.finalize(sampleRuntime())
	r.Errors = append(r.Errors, finalBreaches...)
	r.LeakOracle = leak
	cutoffPartialOK := r.PartialEpochs == 0
	if r.PartialEpochs == 1 && len(r.Epochs) > 0 {
		last := r.Epochs[len(r.Epochs)-1]
		cutoffPartialOK = last.Partial && !last.Complete && !last.CompletedAt.Before(cutoff) && len(last.Errors) == 0
	}
	if r.FullDurationObserved && r.FullEpochs >= cfg.Plan.MinimumEpochs && cutoffPartialOK && r.DrainSucceeded && r.LeakOracle.Passed && len(r.Errors) == 0 {
		r.Status, r.Qualified = "PASSED", true
		return r, nil
	}
	if !r.LeakOracle.Passed {
		return r, errors.New("burn-in runtime leak oracle failed")
	}
	return r, errors.New("burn-in coverage incomplete")
}

func matchingBurnInNormal(got burnInNormalAdmission, identity burnInIdentity, concurrency int) bool {
	return got.Qualified && got.Concurrency == concurrency && got.Identity == identity && identity.ModelID != "" && identity.ConfigDigest != "" && identity.WorkloadDigest != ""
}

func burnInSpec(plan profileplan.Plan, ordinal int) burnInEpochSpec {
	e := plan.Epoch
	return burnInEpochSpec{Ordinal: ordinal, Seed: plan.Seed + uint64(ordinal-1), Concurrency: plan.PreferredConcurrency, Sessions: e.Sessions, TurnsPerSession: e.TurnsPerSession, Controls: e.ControlTurns, Tasks: e.TaskFixtures, Areas: 4, MaxPriming: e.MaxPrimingRequests, MaxRetries: e.MaxReplayRetries, RequirePressureRevisit: true, RequireSourceEditReread: true, RequireCancelRecovery: true}
}

func validFullBurnInEpoch(spec burnInEpochSpec, got burnInEpochReceipt) bool {
	pressureValid := got.PressureStatus == "passed" || got.PressureStatus == "unknown"
	return got.Ordinal == spec.Ordinal && got.Complete && !got.Partial &&
		got.SteadyRequests == spec.Sessions*spec.TurnsPerSession && got.ControlRequests == spec.Controls &&
		got.TaskAttempts == spec.Tasks && got.AcceptedTasks == spec.Tasks &&
		got.PrimingRequests >= 0 && got.PrimingRequests <= spec.MaxPriming && got.ReplayRetries >= 0 && got.ReplayRetries <= spec.MaxRetries &&
		got.AreasVisited == spec.Areas && pressureValid && got.SourceEditRereadPassed && got.IntentionalCancellationObserved && got.RecoveryPassed && len(got.Errors) == 0
}

func defaultBurnInLeakBounds() burnInLeakBounds {
	return burnInLeakBounds{
		MaxGoroutineDelta:  defaultBurnInMaxGoroutineDelta,
		MaxHeapGrowthBytes: defaultBurnInMaxHeapGrowthBytes,
		MaxFDDelta:         defaultBurnInMaxFDDelta,
	}
}

func sampleBurnInRuntime() burnInRuntimeSample {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	sample := burnInRuntimeSample{
		Goroutines: runtime.NumGoroutine(),
		HeapAlloc:  ms.HeapAlloc,
		OpenFDs:    -1,
		FDStatus:   "skipped_non_linux",
	}
	if runtime.GOOS == "linux" {
		entries, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			sample.FDStatus = "unavailable"
		} else {
			sample.OpenFDs = len(entries)
			sample.FDStatus = "measured"
		}
	}
	return sample
}

func newBurnInLeakReport(bounds burnInLeakBounds, baseline burnInRuntimeSample) burnInLeakReport {
	return burnInLeakReport{
		Verdict:  "PASS",
		Passed:   true,
		Bounds:   bounds,
		Baseline: baseline,
		Final:    baseline,
		FDStatus: baseline.FDStatus,
	}
}

func (r *burnInLeakReport) recordEpoch(ordinal int, cur burnInRuntimeSample) burnInEpochLeakDelta {
	delta := evaluateLeakDelta(ordinal, r.Baseline, cur, r.Bounds)
	r.EpochDeltas = append(r.EpochDeltas, delta)
	r.updatePeaks(delta)
	if len(delta.Breaches) > 0 {
		r.Passed = false
		r.Verdict = "FAIL"
		r.Breaches = append(r.Breaches, delta.Breaches...)
	}
	return delta
}

func (r *burnInLeakReport) finalize(final burnInRuntimeSample) []string {
	r.Final = final
	if final.FDStatus != "" && r.FDStatus == "measured" && final.FDStatus != "measured" {
		r.FDStatus = final.FDStatus
	}
	delta := evaluateLeakDelta(0, r.Baseline, final, r.Bounds)
	r.updatePeaks(delta)
	if len(delta.Breaches) > 0 {
		r.Passed = false
		r.Verdict = "FAIL"
		r.Breaches = append(r.Breaches, delta.Breaches...)
	}
	return delta.Breaches
}

func (r *burnInLeakReport) updatePeaks(delta burnInEpochLeakDelta) {
	if delta.GoroutineDelta > r.MaxGoroutineDelta {
		r.MaxGoroutineDelta = delta.GoroutineDelta
	}
	if delta.HeapDeltaBytes > r.MaxHeapDeltaBytes {
		r.MaxHeapDeltaBytes = delta.HeapDeltaBytes
	}
	if delta.FDStatus == "measured" && delta.FDDelta > r.MaxFDDelta {
		r.MaxFDDelta = delta.FDDelta
	}
}

func evaluateLeakDelta(ordinal int, baseline, cur burnInRuntimeSample, bounds burnInLeakBounds) burnInEpochLeakDelta {
	gDelta := cur.Goroutines - baseline.Goroutines
	hDelta := int64(cur.HeapAlloc) - int64(baseline.HeapAlloc)
	fdDelta := 0
	fdStatus := cur.FDStatus
	if baseline.FDStatus == "measured" && cur.FDStatus == "measured" {
		fdDelta = cur.OpenFDs - baseline.OpenFDs
		fdStatus = "measured"
	} else if baseline.FDStatus != "measured" {
		fdStatus = baseline.FDStatus
	}
	scope := "final"
	if ordinal > 0 {
		scope = fmt.Sprintf("epoch %d", ordinal)
	}
	var breaches []string
	if gDelta > bounds.MaxGoroutineDelta {
		breaches = append(breaches, fmt.Sprintf("leak oracle (%s): goroutine delta %d exceeds bound %d", scope, gDelta, bounds.MaxGoroutineDelta))
	}
	if hDelta > bounds.MaxHeapGrowthBytes {
		breaches = append(breaches, fmt.Sprintf("leak oracle (%s): heap growth %d bytes exceeds bound %d", scope, hDelta, bounds.MaxHeapGrowthBytes))
	}
	if fdStatus == "measured" && fdDelta > bounds.MaxFDDelta {
		breaches = append(breaches, fmt.Sprintf("leak oracle (%s): open fd delta %d exceeds bound %d", scope, fdDelta, bounds.MaxFDDelta))
	}
	verdict := "PASS"
	if len(breaches) > 0 {
		verdict = "FAIL"
	}
	return burnInEpochLeakDelta{
		Ordinal:        ordinal,
		GoroutineDelta: gDelta,
		HeapDeltaBytes: hDelta,
		FDDelta:        fdDelta,
		FDStatus:       fdStatus,
		Verdict:        verdict,
		Breaches:       breaches,
	}
}
