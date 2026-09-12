package agentbench

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
)

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
	Epochs               []burnInEpochReceipt  `json:"epochs"`
	Errors               []string              `json:"errors,omitempty"`
}

func runBurnInController(ctx context.Context, cfg burnInControllerConfig) (burnInReceipt, error) {
	r := burnInReceipt{Status: "INCOMPLETE", Identity: cfg.Identity}
	if cfg.Plan.Profile != "burn-in" || (cfg.Plan.Deadline != 2*time.Hour && cfg.Plan.Deadline != 8*time.Hour) || cfg.Plan.Epoch == nil || cfg.Plan.DrainGrace != 90*time.Second || cfg.Now == nil || cfg.RunEpoch == nil || cfg.Drain == nil {
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
	started := cfg.Now()
	cutoff := started.Add(cfg.Plan.Deadline)
	for ordinal := 1; cfg.Now().Before(cutoff); ordinal++ {
		epochStarted := cfg.Now()
		spec := burnInSpec(cfg.Plan, ordinal)
		epoch := cfg.RunEpoch(ctx, spec, cutoff)
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
		if err := ctx.Err(); err != nil {
			r.Errors = append(r.Errors, err.Error())
			break
		}
	}
	r.FullDurationObserved = !cfg.Now().Before(cutoff)
	drainCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cfg.Plan.DrainGrace)
	drainErr := cfg.Drain(drainCtx, cfg.Plan.DrainGrace)
	cancel()
	r.DrainSucceeded = drainErr == nil
	if drainErr != nil {
		r.Errors = append(r.Errors, "drain: "+drainErr.Error())
	}
	cutoffPartialOK := r.PartialEpochs == 0
	if r.PartialEpochs == 1 && len(r.Epochs) > 0 {
		last := r.Epochs[len(r.Epochs)-1]
		cutoffPartialOK = last.Partial && !last.Complete && !last.CompletedAt.Before(cutoff) && len(last.Errors) == 0
	}
	if r.FullDurationObserved && r.FullEpochs >= cfg.Plan.MinimumEpochs && cutoffPartialOK && r.DrainSucceeded && len(r.Errors) == 0 {
		r.Status, r.Qualified = "PASSED", true
		return r, nil
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
