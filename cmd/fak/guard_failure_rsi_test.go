package main

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func guardTestContainmentSurvivorError(pids ...int) error {
	raw := fmt.Errorf("verify child resource reap: %s: owned processes still alive: %v; recovery: stop the verified processes", guardFailureRSIReasonContainmentSurvivors, pids)
	return guardTypeContainmentSurvivorError(raw)
}

func TestGuardFailureRSITypedReceiptLaunchesScrubbedNormalizedRequest(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	var got []guardCrashRSIRequest
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		got = append(got, req)
		return nil
	}

	var stderr bytes.Buffer
	if !guardMaybeLaunchFailureRSI(&stderr, new(guardRSISession), "RID-private-trace", "codex", guardTestContainmentSurvivorError(12345, 67890)) {
		t.Fatal("typed receipt failure did not launch")
	}
	if len(got) != 1 {
		t.Fatalf("launches=%d, want 1", len(got))
	}
	req := got[0]
	if req.Trigger != guardFailureRSITriggerChildResourceReceipt || req.Reason != guardFailureRSIReasonContainmentSurvivors || req.Subsystem != guardFailureRSISubsystemChildResource {
		t.Fatalf("typed request fields=%+v", req)
	}
	for _, want := range []string{req.Tag, req.Source, req.Trigger, req.Reason, req.Subsystem, req.Agent, req.BuildCommit, req.BuildModule, req.ReceiptIdentity, req.Signature} {
		if want == "" || !strings.Contains(req.Prompt, want) {
			t.Fatalf("prompt missing bounded field %q: %s", want, req.Prompt)
		}
	}
	for _, forbidden := range []string{"12345", "67890", "RID-private-trace", "owned processes still alive", "stop the verified processes"} {
		if strings.Contains(req.Prompt, forbidden) || strings.Contains(req.Signature, forbidden) || strings.Contains(req.Tag, forbidden) {
			t.Fatalf("request leaked volatile/private field %q: %+v", forbidden, req)
		}
	}
	if !strings.Contains(stderr.String(), req.Tag) {
		t.Fatalf("status missing tag: %s", stderr.String())
	}
}

func TestGuardFailureRSIVolatilePIDsNormalizeToOneSignature(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	one, ok := guardFailureRSIAdmission("same-trace", "claude", guardTestContainmentSurvivorError(101))
	if !ok {
		t.Fatal("first typed receipt failure was not admitted")
	}
	two, ok := guardFailureRSIAdmission("same-trace", "claude", guardTestContainmentSurvivorError(202, 303))
	if !ok {
		t.Fatal("second typed receipt failure was not admitted")
	}
	if one.Signature != two.Signature || one.Tag != two.Tag || one.Prompt != two.Prompt {
		t.Fatalf("PID-only change altered normalized request:\none=%+v\ntwo=%+v", one, two)
	}
}

func TestGuardRSISessionAllowsOnlyOneCrashOrFailureLaunch(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { launches++; return nil }

	session := new(guardRSISession)
	guardMaybeLaunchCrashRSI(nil, session, "trace", "codex", "NONZERO_EXIT", 2, 0)
	guardMaybeLaunchFailureRSI(nil, session, "trace", "codex", guardTestContainmentSurvivorError(404))
	if launches != 1 {
		t.Fatalf("crash then failure launches=%d, want 1", launches)
	}

	launches = 0
	session = new(guardRSISession)
	guardMaybeLaunchFailureRSI(nil, session, "trace", "codex", guardTestContainmentSurvivorError(505))
	guardMaybeLaunchCrashRSI(nil, session, "trace", "codex", "NONZERO_EXIT", 2, 0)
	guardMaybeLaunchFailureRSI(nil, session, "trace", "codex", guardTestContainmentSurvivorError(606))
	if launches != 1 {
		t.Fatalf("failure then repeats/crash launches=%d, want 1", launches)
	}
}

func TestGuardFailureRSIRecursionUntypedAndInfrastructureFailuresSkip(t *testing.T) {
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { launches++; return nil }

	t.Setenv(guardCrashRSIMarkerEnv, "guard-failure-rsi/already")
	guardMaybeLaunchFailureRSI(nil, new(guardRSISession), "trace", "codex", guardTestContainmentSurvivorError(1))
	t.Setenv(guardCrashRSIMarkerEnv, "")
	for _, err := range []error{
		errors.New("child resource receipt missing decision"),
		errors.New("verify child resource reap: collector unavailable"),
		errors.New("append child resource receipt: disk full"),
		errors.New("infrastructure mentioned CHILD_RESOURCE_CONTAINMENT_SURVIVORS without a typed prefix"),
		fmt.Errorf("verify child resource reap: %s: owned processes still alive: [9191]", guardFailureRSIReasonContainmentSurvivors),
	} {
		guardMaybeLaunchFailureRSI(nil, new(guardRSISession), "trace", "codex", err)
	}
	if launches != 0 {
		t.Fatalf("recursive/untyped/infrastructure launches=%d", launches)
	}
}

func TestGuardFailureRSILaunchFailurePreservesOriginalErrorAndConsumesSession(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error {
		launches++
		return errors.New("synthetic launch failure")
	}

	receiptErr := fmt.Errorf("verify child resource reap: %s: owned processes still alive: [777]; recovery: stop the verified processes", guardFailureRSIReasonContainmentSurvivors)
	want := "child resource receipt failed after containment: " + receiptErr.Error()
	var stderr bytes.Buffer
	session := new(guardRSISession)
	original := guardHandleResourceReceiptFailure(&stderr, session, "trace", "codex", receiptErr)
	guardMaybeLaunchCrashRSI(&stderr, session, "trace", "codex", "NONZERO_EXIT", 2, 0)
	if launches != 1 {
		t.Fatalf("launch attempts=%d, want exactly 1", launches)
	}
	if original.Error() != want || !errors.Is(original, receiptErr) || !strings.Contains(original.Error(), guardFailureRSIReasonContainmentSurvivors) || !strings.Contains(original.Error(), "777") {
		t.Fatalf("original containment/recovery error changed: %q, want %q", original, want)
	}
	if !strings.Contains(stderr.String(), "synthetic launch failure") {
		t.Fatalf("missing fail-open diagnostic: %s", stderr.String())
	}
}
