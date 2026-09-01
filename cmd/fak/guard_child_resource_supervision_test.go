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
