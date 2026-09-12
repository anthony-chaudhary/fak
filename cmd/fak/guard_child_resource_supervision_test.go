package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

type writerFunc func([]byte) (int, error)

func (fn writerFunc) Write(p []byte) (int, error) {
	return fn(p)
}

func guardInspectionFailureForSupervisionTest() guardChildWaitEvent {
	return guardResourceMonitorFailure(42, procguard.MemorySnapshot{
		Metric:    procguard.MemoryMetricCommit,
		Processes: []procguard.MemoryProcess{{PID: 42}},
	}, "CHILD_RESOURCE_INSPECTION_DENIED", "OpenProcess: Access is denied")
}

func guardMeasuredBreachForSupervisionTest() guardChildWaitEvent {
	decision := decideGuardResource(guardResourcePolicy{MaxTreeBytes: 100}, procguard.MemorySnapshot{
		Metric:    procguard.MemoryMetricCommit,
		TreeBytes: 101,
		Processes: []procguard.MemoryProcess{{PID: 42, Bytes: 101}},
	})
	return guardChildWaitEvent{Kind: guardChildResourceLimit, Reason: guardResourceReason(decision), Resource: &decision}
}

func guardHeadroomBreachForSupervisionTest() guardChildWaitEvent {
	decision := decideGuardResource(guardResourcePolicy{MinSystemHeadroom: 100}, procguard.MemorySnapshot{
		Metric:      procguard.MemoryMetricCommit,
		SystemBytes: 950,
		SystemLimit: 1000,
		Processes:   []procguard.MemoryProcess{{PID: 42, Bytes: 50}},
	})
	return guardChildWaitEvent{Kind: guardChildResourceLimit, Reason: guardResourceReason(decision), Resource: &decision}
}

func TestGuardPrelaunchSystemCommitHeadroom(t *testing.T) {
	const required = uint64(10)
	for _, tt := range []struct {
		name         string
		snapshot     procguard.MemorySnapshot
		supported    bool
		detail       string
		wantStart    bool
		wantErr      string
		wantObserved string
		wantRequired string
		wantDetail   string
		wantAbsent   string
	}{
		{
			name:         "below reserve refuses before child work",
			snapshot:     procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, SystemBytes: 91, SystemLimit: 100, Processes: []procguard.MemoryProcess{{CommandLine: "fixture child arguments"}}},
			supported:    true,
			wantErr:      procguard.SystemCommitHeadroomReason,
			wantObserved: "observed 9 bytes",
			wantRequired: "required 10 bytes",
			wantAbsent:   "fixture child arguments",
		},
		{
			name:         "exact reserve refuses before child work",
			snapshot:     procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, SystemBytes: 90, SystemLimit: 100},
			supported:    true,
			wantErr:      procguard.SystemCommitHeadroomReason,
			wantObserved: "observed 10 bytes",
			wantRequired: "required 10 bytes",
		},
		{
			name:      "above reserve starts",
			snapshot:  procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, SystemBytes: 89, SystemLimit: 100},
			supported: true,
			wantStart: true,
		},
		{
			name:      "unsupported platform preserves launch",
			detail:    "system commit accounting unsupported on this platform",
			wantStart: true,
		},
		{
			name:       "supported telemetry failure fails closed",
			snapshot:   procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit},
			supported:  true,
			detail:     "GetPerformanceInfo failed",
			wantErr:    "CHILD_RESOURCE_COLLECTOR_FAILURE",
			wantDetail: "GetPerformanceInfo failed",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			collectorCalls := 0
			starts := 0
			providerRequests := 0
			invoked, err := guardStartChildAfterSystemCommitAdmission(
				guardResourcePolicy{MinSystemHeadroom: required},
				func() (procguard.MemorySnapshot, bool, string) {
					collectorCalls++
					return tt.snapshot, tt.supported, tt.detail
				},
				func() error {
					starts++
					providerRequests++
					return nil
				},
			)
			if collectorCalls != 1 {
				t.Fatalf("collector calls=%d, want 1", collectorCalls)
			}
			if invoked != tt.wantStart || starts != guardTestBoolInt(tt.wantStart) || providerRequests != guardTestBoolInt(tt.wantStart) {
				t.Fatalf("invoked=%v starts=%d provider_requests=%d want_start=%v", invoked, starts, providerRequests, tt.wantStart)
			}
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected admission error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("admission error=%v, want %q", err, tt.wantErr)
			}
			if tt.wantObserved != "" && !strings.Contains(err.Error(), tt.wantObserved) {
				t.Fatalf("admission error=%q, want observed evidence %q", err, tt.wantObserved)
			}
			if tt.wantRequired != "" && !strings.Contains(err.Error(), tt.wantRequired) {
				t.Fatalf("admission error=%q, want required evidence %q", err, tt.wantRequired)
			}
			if tt.wantDetail != "" && !strings.Contains(err.Error(), tt.wantDetail) {
				t.Fatalf("admission error=%q, want telemetry detail %q", err, tt.wantDetail)
			}
			if tt.wantAbsent != "" && strings.Contains(err.Error(), tt.wantAbsent) {
				t.Fatalf("admission error exposed process command: %q", err)
			}
		})
	}
}

func guardTestBoolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func TestGuardChildResourceSupervisionContainsOnlyMeasuredBreaches(t *testing.T) {
	inspectionFailure := guardInspectionFailureForSupervisionTest()
	if guardChildResourceNeedsContainment(inspectionFailure) {
		t.Fatalf("inspection failure requested containment: %+v", inspectionFailure)
	}
	inspectionFailure.Resource.Stop = true
	if guardChildResourceNeedsContainment(inspectionFailure) {
		t.Fatalf("typed inspection failure requested containment despite not being a measured limit: %+v", inspectionFailure)
	}
	inspectionFailure.Resource.Stop = false
	var diagnostic bytes.Buffer
	guardReportChildResourceMonitorFailure(&diagnostic, inspectionFailure)
	got := diagnostic.String()
	if !strings.Contains(got, "CHILD_RESOURCE_INSPECTION_DENIED") || !strings.Contains(got, "child remains running") || strings.Contains(got, "runaway") {
		t.Fatalf("diagnostic=%q", got)
	}

	breach := guardMeasuredBreachForSupervisionTest()
	if !guardChildResourceNeedsContainment(breach) {
		t.Fatalf("measured breach did not request containment: %+v", breach)
	}

	headroomBreach := guardHeadroomBreachForSupervisionTest()
	if !guardChildResourceNeedsContainment(headroomBreach) {
		t.Fatalf("headroom breach did not request containment: %+v", headroomBreach)
	}
}

func TestGuardChildNormalSupervisionWaitsForChildAfterMonitorFailure(t *testing.T) {
	wait := make(chan error, 1)
	resources := make(chan guardChildWaitEvent, 1)
	resources <- guardInspectionFailureForSupervisionTest()
	childErr := errors.New("child completed after monitor degraded")

	// The child may complete only after the monitor diagnostic is emitted. This
	// models the ordering under test instead of preloading both select cases.
	diagnosticObserved := make(chan struct{})
	diagnostic := writerFunc(func(p []byte) (int, error) {
		select {
		case <-diagnosticObserved:
		default:
			close(diagnosticObserved)
		}
		return len(p), nil
	})
	go func() {
		<-diagnosticObserved
		wait <- childErr
	}()

	runErr, event, contain := waitGuardChildWithoutRestart(wait, resources, diagnostic)
	if contain {
		t.Fatal("normal supervision requested containment for monitor failure")
	}
	if !errors.Is(runErr, childErr) {
		t.Fatalf("runErr=%v, want child completion %v", runErr, childErr)
	}
	if event.Resource == nil || event.Resource.Stop {
		t.Fatalf("monitor event=%+v, want non-terminal diagnostic", event)
	}
	select {
	case <-diagnosticObserved:
	default:
		t.Fatal("monitor diagnostic was not emitted before child completion")
	}
}

func TestGuardChildNormalSupervisionReturnsMeasuredBreachForContainment(t *testing.T) {
	wait := make(chan error)
	resources := make(chan guardChildWaitEvent, 1)
	resources <- guardMeasuredBreachForSupervisionTest()

	runErr, event, contain := waitGuardChildWithoutRestart(wait, resources, nil)
	if runErr != nil || !contain || !guardChildResourceNeedsContainment(event) {
		t.Fatalf("runErr=%v contain=%v event=%+v", runErr, contain, event)
	}
}

func TestGuardChildRestartSupervisionContinuesAfterMonitorFailure(t *testing.T) {
	wait := make(chan error)
	restarts := make(chan guardBudgetRestartEvent, 1)
	ticks := make(chan time.Time)
	resources := make(chan guardChildWaitEvent, 1)
	resources <- guardInspectionFailureForSupervisionTest()
	wantRestart := guardBudgetRestartEvent{Reason: "budget_rotation"}
	restarts <- wantRestart

	var diagnostic bytes.Buffer
	event := waitGuardChildSupervised(wait, restarts, ticks, nil, resources, &diagnostic)
	if event.Kind != guardChildRestart || event.Restart.Reason != wantRestart.Reason {
		t.Fatalf("event=%+v, want restart after monitor failure", event)
	}
	if !strings.Contains(diagnostic.String(), "child remains running") {
		t.Fatalf("diagnostic=%q", diagnostic.String())
	}
}

func TestGuardChildRestartSupervisionKeepsTimeBudgetAfterMonitorFailure(t *testing.T) {
	wait := make(chan error)
	restarts := make(chan guardBudgetRestartEvent)
	ticks := make(chan time.Time, 1)
	resources := make(chan guardChildWaitEvent, 1)
	resources <- guardInspectionFailureForSupervisionTest()
	ticks <- time.Unix(1, 0)

	event := waitGuardChildSupervised(wait, restarts, ticks, func(time.Time) (bool, string) {
		return true, "TIME_BUDGET_EXHAUSTED"
	}, resources, nil)
	if event.Kind != guardChildTimeBudget || event.Reason != "TIME_BUDGET_EXHAUSTED" {
		t.Fatalf("event=%+v, want time-budget event after monitor failure", event)
	}
}

func TestGuardChildRestartSupervisionReturnsMeasuredBreachForContainment(t *testing.T) {
	wait := make(chan error)
	restarts := make(chan guardBudgetRestartEvent)
	ticks := make(chan time.Time)
	resources := make(chan guardChildWaitEvent, 1)
	resources <- guardMeasuredBreachForSupervisionTest()

	event := waitGuardChildSupervised(wait, restarts, ticks, nil, resources, nil)
	if !guardChildResourceNeedsContainment(event) {
		t.Fatalf("event=%+v, want measured containment event", event)
	}
}

func TestGuardReportChildResourceReaped(t *testing.T) {
	t.Run("headroom logs host reserve breached", func(t *testing.T) {
		var buf bytes.Buffer
		ev := guardHeadroomBreachForSupervisionTest()
		guardReportChildResourceReaped(&buf, ev)
		got := buf.String()
		if !strings.Contains(got, "host operating system commit reserve breached safety floor") {
			t.Fatalf("unexpected message: %q", got)
		}
		if !strings.Contains(got, "child stopped to preserve host stability") {
			t.Fatalf("missing stability note: %q", got)
		}
		if strings.Contains(got, "runaway") {
			t.Fatalf("falsely called headroom a runaway: %q", got)
		}
	})

	t.Run("tree limit logs child resource runaway", func(t *testing.T) {
		var buf bytes.Buffer
		ev := guardMeasuredBreachForSupervisionTest()
		guardReportChildResourceReaped(&buf, ev)
		got := buf.String()
		if !strings.Contains(got, "reaped child resource runaway") {
			t.Fatalf("unexpected message: %q", got)
		}
	})
}

func TestGuardSystemCommitHeadroomClassificationAndExitCode(t *testing.T) {
	headroomErr := errors.New("child resource limit: SYSTEM_COMMIT_HEADROOM tree_commit=10 threshold=100")
	if !isGuardSystemCommitHeadroom(headroomErr) {
		t.Fatalf("isGuardSystemCommitHeadroom(%v) = false, want true", headroomErr)
	}
	rssErr := errors.New("child resource limit: CHILD_TREE_RSS_LIMIT")
	if isGuardSystemCommitHeadroom(rssErr) {
		t.Fatalf("isGuardSystemCommitHeadroom(%v) = true, want false", rssErr)
	}
	if isGuardSystemCommitHeadroom(nil) {
		t.Fatal("isGuardSystemCommitHeadroom(nil) = true, want false")
	}

	t.Run("supervised by goal yields cleanly", func(t *testing.T) {
		t.Setenv("FAK_GOAL_ID", "goal-test-1")
		if !isGuardGoalOrSessionSupervised() {
			t.Fatal("isGuardGoalOrSessionSupervised() = false with FAK_GOAL_ID set")
		}
	})

	t.Run("unsupervised exits with refusal exit code 3", func(t *testing.T) {
		t.Setenv("FAK_GOAL_ID", "")
		t.Setenv("DISPATCH_GOAL", "")
		t.Setenv("FAK_SESSION_ID", "")
		t.Setenv("FAK_GOAL_LOOP", "")
		t.Setenv("FAK_GOAL_SPEC", "")
		t.Setenv("FAK_GOAL_RUN", "")
		t.Setenv("FAK_LOOP_ID", "")
		t.Setenv("DISPATCH_LANE", "")
		if isGuardGoalOrSessionSupervised() {
			t.Fatal("isGuardGoalOrSessionSupervised() = true in clean environment")
		}
		if recoverExitRefusal != 3 {
			t.Fatalf("recoverExitRefusal = %d, want 3", recoverExitRefusal)
		}
	})

	status := guardResourceRestartGiveUpStatus(guardResourceRetryVerdict{ResourceType: "SYSTEM_COMMIT_HEADROOM"}, "test-trace")
	if !strings.Contains(status, "host commit capacity reached the safety floor") || !strings.Contains(status, "fak recover SYSTEM_COMMIT_HEADROOM") {
		t.Fatalf("unexpected status string: %q", status)
	}
}

func TestGuardGoalParkOnSystemCommitHeadroom(t *testing.T) {
	goalParkTestRoot(t)

	t.Setenv("DISPATCH_GOAL", "headroom-park-goal")
	t.Setenv("DISPATCH_LANE", "headroom-lane")
	t.Setenv("DISPATCH_ACCOUNT", "account-a")
	t.Setenv("FAK_GOAL_ID", "headroom-park-goal")

	parked := guardParkGoalOnHeadroom("headroom-park-goal", "account-a", "claude")
	if !parked {
		t.Fatal("guardParkGoalOnHeadroom returned false")
	}

	rec, err := goalParkStore().Load("headroom-park-goal")
	if err != nil {
		t.Fatalf("failed to load parked goal: %v", err)
	}
	if rec.Goal != "headroom-park-goal" {
		t.Errorf("Goal = %q, want %q", rec.Goal, "headroom-park-goal")
	}
	if rec.Lane != "headroom-lane" {
		t.Errorf("Lane = %q, want %q", rec.Lane, "headroom-lane")
	}
	if rec.Account != "account-a" {
		t.Errorf("Account = %q, want %q", rec.Account, "account-a")
	}
	if rec.Reason != "SYSTEM_COMMIT_HEADROOM" {
		t.Errorf("Reason = %q, want %q", rec.Reason, "SYSTEM_COMMIT_HEADROOM")
	}
	if rec.ParkedAt <= 0 {
		t.Errorf("ParkedAt = %d, want > 0", rec.ParkedAt)
	}
	wantUntil := rec.ParkedAt + int64((15 * time.Minute).Seconds())
	if rec.ParkedUntil != wantUntil {
		t.Errorf("ParkedUntil = %d, want %d", rec.ParkedUntil, wantUntil)
	}
	if len(rec.Command) == 0 || rec.Command[0] != "claude" {
		t.Errorf("Command = %v, want prefix claude", rec.Command)
	}
}
