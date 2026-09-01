package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
)

func recoveryFailureRequest(status string) sessionrecovery.Request {
	return sessionrecovery.Request{
		ThreadID: "thread-sensitive-12345",
		CWD:      `C:\secret\recovery\workspace`,
		Provider: "codex",
		Harness:  "codex",
		Status:   status,
		Reason:   "launch pid 4242 failed at 2026-08-31T10:11:12Z: secret raw error",
	}
}

func TestSessionRecoveryFailureRSIAdmissionAcceptsTypedFailures(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	for _, status := range []string{"prompt_failed", "launch_failed", "verification_failed", "launched_unproven", "provider_reaped"} {
		t.Run(status, func(t *testing.T) {
			req, ok := sessionRecoveryFailureRSIAdmission(recoveryFailureRequest(status))
			if !ok {
				t.Fatalf("status %q was not admitted", status)
			}
			if req.Trigger != sessionRecoveryFailureRSITrigger || req.Reason != status || req.Subsystem != sessionRecoveryFailureRSISubsystem {
				t.Fatalf("typed request = %#v", req)
			}
			if req.ReceiptIdentity != sessionRecoveryFailureRSIReceiptIdentity {
				t.Fatalf("receipt identity = %q", req.ReceiptIdentity)
			}
		})
	}
}

func TestSessionRecoveryFailureRSIAdmissionRefusesNonFailuresAndUnsupportedHarness(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	for _, status := range []string{"", "candidate", "launching", "launched", "productive", "completed", "already_receipted", "receipt_failed"} {
		t.Run(status, func(t *testing.T) {
			if _, ok := sessionRecoveryFailureRSIAdmission(recoveryFailureRequest(status)); ok {
				t.Fatalf("status %q admitted", status)
			}
		})
	}
	unsupported := recoveryFailureRequest("launch_failed")
	unsupported.Provider = "other"
	unsupported.Harness = "other"
	if _, ok := sessionRecoveryFailureRSIAdmission(unsupported); ok {
		t.Fatal("unsupported harness admitted")
	}
	emptyCorrelation := recoveryFailureRequest("launch_failed")
	emptyCorrelation.ThreadID = " "
	if _, ok := sessionRecoveryFailureRSIAdmission(emptyCorrelation); ok {
		t.Fatal("empty correlation admitted")
	}
}

func TestSessionRecoveryFailureRSIAdmissionRecursionMarkerSuppresses(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "session-recovery-failure-rsi/already-running")
	if _, ok := sessionRecoveryFailureRSIAdmission(recoveryFailureRequest("launch_failed")); ok {
		t.Fatal("recursive recovery RSI admitted")
	}
}

func TestSessionRecoveryFailureRSISignatureNormalizesCorrelationAndScrubsRawValues(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	first := recoveryFailureRequest("launch_failed")
	second := recoveryFailureRequest("launch_failed")
	second.ThreadID = "different-sensitive-thread"
	firstReq, ok := sessionRecoveryFailureRSIAdmission(first)
	if !ok {
		t.Fatal("first request refused")
	}
	secondReq, ok := sessionRecoveryFailureRSIAdmission(second)
	if !ok {
		t.Fatal("second request refused")
	}
	if firstReq.Signature != secondReq.Signature {
		t.Fatalf("signatures differ: %q != %q", firstReq.Signature, secondReq.Signature)
	}
	if firstReq.Source == secondReq.Source {
		t.Fatal("source digest did not distinguish correlation IDs")
	}
	if firstReq.Workspace == "" {
		t.Fatal("launcher workspace is empty")
	}
	combined := firstReq.Tag + "\n" + firstReq.Source + "\n" + firstReq.Signature + "\n" + firstReq.Prompt
	for _, raw := range []string{first.ThreadID, first.CWD, "4242", "2026-08-31T10:11:12Z", first.Reason, firstReq.Workspace} {
		if strings.Contains(combined, raw) {
			t.Fatalf("request exposed raw value %q in %q", raw, combined)
		}
	}
	if !strings.Contains(firstReq.Prompt, "recovered-session launch-failure RSI") || !strings.Contains(firstReq.Prompt, "read-only root-cause analysis") {
		t.Fatalf("prompt is not bounded and recovery-specific: %q", firstReq.Prompt)
	}
}

func TestSessionMaybeLaunchFailureRSIClaimsOnceBeforeLaunch(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	calls := 0
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		calls++
		return nil
	}
	session := new(guardRSISession)
	if !sessionMaybeLaunchFailureRSI(nil, session, recoveryFailureRequest("launch_failed")) {
		t.Fatal("first launch refused")
	}
	if sessionMaybeLaunchFailureRSI(nil, session, recoveryFailureRequest("verification_failed")) {
		t.Fatal("second launch admitted")
	}
	if calls != 1 {
		t.Fatalf("launch calls = %d, want 1", calls)
	}
}

func TestSessionMaybeLaunchFailureRSILaunchFailureIsAdvisoryAndConsumesSlot(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	calls := 0
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		calls++
		return errors.New("launcher unavailable")
	}
	var stderr bytes.Buffer
	session := new(guardRSISession)
	if sessionMaybeLaunchFailureRSI(&stderr, session, recoveryFailureRequest("launch_failed")) {
		t.Fatal("failed launcher reported success")
	}
	if sessionMaybeLaunchFailureRSI(&stderr, session, recoveryFailureRequest("launch_failed")) {
		t.Fatal("failed launch did not consume slot")
	}
	if calls != 1 {
		t.Fatalf("launch calls = %d, want 1", calls)
	}
	if !strings.Contains(stderr.String(), "failure RSI launch skipped") {
		t.Fatalf("missing advisory diagnostic: %q", stderr.String())
	}
}
