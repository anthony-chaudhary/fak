package agentbench

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
)

func TestBurnInControllerRequiresFullDurationCoverageAndDrain(t *testing.T) {
	plan, err := profileplan.Build(profileplan.Options{Profile: "burn-in", PreferredConcurrency: 4, Duration: 2 * time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	identity := burnInIdentity{ModelID: "fixture-model", ConfigDigest: "config", WorkloadDigest: "workload"}
	normal := burnInNormalAdmission{Qualified: true, Concurrency: 4, Identity: identity}
	start := time.Date(2026, 9, 10, 8, 0, 0, 0, time.UTC)

	t.Run("complete full duration", func(t *testing.T) {
		now := start
		var specs []burnInEpochSpec
		drained := time.Duration(0)
		receipt, err := runBurnInController(context.Background(), burnInControllerConfig{
			Plan: plan, Identity: identity, Normal: &normal, Now: func() time.Time { return now },
			RunEpoch: func(_ context.Context, spec burnInEpochSpec, cutoff time.Time) burnInEpochReceipt {
				specs = append(specs, spec)
				now = now.Add(30 * time.Minute)
				return completeBurnInEpoch(spec, now)
			},
			Drain: func(_ context.Context, grace time.Duration) error { drained = grace; return nil },
		})
		if err != nil || !receipt.Qualified || receipt.Status != "PASSED" || receipt.FullEpochs != 4 || receipt.PartialEpochs != 0 || receipt.SteadyRequests != 4*256 || receipt.ControlRequests != 4*24 || receipt.TaskAttempts != 4*6 || receipt.AcceptedTasks != 4*6 || receipt.PrimingRequests != 8 || receipt.ReplayRetries != 4 || !receipt.FullDurationObserved || !receipt.DrainSucceeded || drained != 90*time.Second {
			t.Fatalf("complete burn-in receipt = %+v err=%v drain=%v", receipt, err, drained)
		}
		for i, spec := range specs {
			if spec.Ordinal != i+1 || spec.Seed != 0xA63EB001+uint64(i) || spec.Concurrency != 4 || spec.Sessions != 8 || spec.TurnsPerSession != 32 || spec.Controls != 24 || spec.Tasks != 6 || spec.Areas != 4 || spec.MaxPriming != 8 || spec.MaxRetries != 8 || !spec.RequirePressureRevisit || !spec.RequireSourceEditReread || !spec.RequireCancelRecovery {
				t.Fatalf("epoch %d immutable runtime geometry = %+v", i+1, spec)
			}
		}
	})

	t.Run("healthy cutoff partial is retained and allowed", func(t *testing.T) {
		now := start
		receipt, err := runBurnInController(context.Background(), burnInControllerConfig{
			Plan: plan, Identity: identity, Normal: &normal, Now: func() time.Time { return now },
			RunEpoch: func(_ context.Context, spec burnInEpochSpec, cutoff time.Time) burnInEpochReceipt {
				if spec.Ordinal <= 2 {
					now = now.Add(50 * time.Minute)
					return completeBurnInEpoch(spec, now)
				}
				now = cutoff
				return burnInEpochReceipt{Ordinal: spec.Ordinal, Partial: true, SteadyRequests: 80, ControlRequests: 7, TaskAttempts: 1, AcceptedTasks: 1, PrimingRequests: 1, CompletedAt: cutoff}
			},
			Drain: func(context.Context, time.Duration) error { return nil },
		})
		if err != nil || !receipt.Qualified || receipt.Status != "PASSED" || receipt.FullEpochs != 2 || receipt.PartialEpochs != 1 || receipt.SteadyRequests != 2*256+80 || receipt.AcceptedTasks != 2*6+1 || !receipt.DrainSucceeded {
			t.Fatalf("healthy duration-cutoff partial was hidden or failed qualification: %+v err=%v", receipt, err)
		}
	})

	t.Run("malformed full epochs cannot qualify", func(t *testing.T) {
		for name, mutate := range map[string]func(*burnInEpochReceipt){
			"tasks attempted but not accepted": func(epoch *burnInEpochReceipt) { epoch.AcceptedTasks = 0 },
			"excess steady requests":           func(epoch *burnInEpochReceipt) { epoch.SteadyRequests++ },
			"priming over budget":              func(epoch *burnInEpochReceipt) { epoch.PrimingRequests = 9 },
			"retries over budget":              func(epoch *burnInEpochReceipt) { epoch.ReplayRetries = 9 },
			"failed pressure":                  func(epoch *burnInEpochReceipt) { epoch.PressureStatus = "failed" },
		} {
			t.Run(name, func(t *testing.T) {
				now := start
				receipt, err := runBurnInController(context.Background(), burnInControllerConfig{
					Plan: plan, Identity: identity, Normal: &normal, Now: func() time.Time { return now },
					RunEpoch: func(_ context.Context, spec burnInEpochSpec, _ time.Time) burnInEpochReceipt {
						now = now.Add(time.Hour)
						epoch := completeBurnInEpoch(spec, now)
						mutate(&epoch)
						return epoch
					}, Drain: func(context.Context, time.Duration) error { return nil },
				})
				if err == nil || receipt.Qualified || receipt.Status != "INCOMPLETE" || receipt.PartialEpochs != 2 {
					t.Fatalf("malformed epoch qualified: %+v err=%v", receipt, err)
				}
			})
		}
	})

	t.Run("partial cutoff and drain failure retain diagnostics", func(t *testing.T) {
		now := start
		receipt, err := runBurnInController(context.Background(), burnInControllerConfig{
			Plan: plan, Identity: identity, Normal: &normal, Now: func() time.Time { return now },
			RunEpoch: func(_ context.Context, spec burnInEpochSpec, cutoff time.Time) burnInEpochReceipt {
				if spec.Ordinal <= 2 {
					now = now.Add(50 * time.Minute)
					return completeBurnInEpoch(spec, now)
				}
				now = cutoff
				return burnInEpochReceipt{Ordinal: spec.Ordinal, Partial: true, SteadyRequests: 80, ControlRequests: 7, TaskAttempts: 1, Errors: []string{"request 81 failed before cutoff"}, CompletedAt: now}
			},
			Drain: func(_ context.Context, grace time.Duration) error { return errors.New("drain timed out") },
		})
		if err == nil || receipt.Qualified || receipt.Status != "INCOMPLETE" || receipt.FullEpochs != 2 || receipt.PartialEpochs != 1 || len(receipt.Errors) < 2 || receipt.SteadyRequests != 2*256+80 || receipt.ControlRequests != 2*24+7 || receipt.TaskAttempts != 2*6+1 || !receipt.FullDurationObserved || receipt.DrainSucceeded {
			t.Fatalf("partial burn-in receipt lost cutoff/drain evidence: %+v err=%v", receipt, err)
		}
	})

	t.Run("normal admission is exact or separately accounted", func(t *testing.T) {
		mismatch := normal
		mismatch.Identity.ConfigDigest = "different"
		called := 0
		preflightNow := start
		cfg := burnInControllerConfig{Plan: plan, Identity: identity, Normal: &mismatch, Now: func() time.Time { return preflightNow }, RunNormalPreflight: func(context.Context) (burnInNormalAdmission, error) { called++; return normal, nil }, RunEpoch: func(_ context.Context, spec burnInEpochSpec, cutoff time.Time) burnInEpochReceipt {
			preflightNow = cutoff
			return burnInEpochReceipt{Ordinal: spec.Ordinal, Partial: true, CompletedAt: cutoff}
		}, Drain: func(context.Context, time.Duration) error { return nil }}
		receipt, err := runBurnInController(context.Background(), cfg)
		if err == nil || called != 1 || !receipt.NormalPreflightRun || receipt.NormalAdmission.Identity != identity {
			t.Fatalf("mismatched normal was invented or preflight unaccounted: receipt=%+v called=%d err=%v", receipt, called, err)
		}
	})
}

func completeBurnInEpoch(spec burnInEpochSpec, at time.Time) burnInEpochReceipt {
	return burnInEpochReceipt{Ordinal: spec.Ordinal, Complete: true, SteadyRequests: 256, ControlRequests: 24, TaskAttempts: spec.Tasks, AcceptedTasks: spec.Tasks, PrimingRequests: 2, ReplayRetries: 1, AreasVisited: 4, PressureStatus: "unknown", SourceEditRereadPassed: true, IntentionalCancellationObserved: true, RecoveryPassed: true, CompletedAt: at}
}
