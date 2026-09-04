package abi

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestThreadPriorityInvariants(t *testing.T) {
	tests := []struct {
		p        ThreadPriority
		wantRank int
		wantName string
	}{
		{ThreadPriorityP0System, 0, "P0_SYSTEM"},
		{ThreadPriorityP1Interactive, 1, "P1_INTERACTIVE"},
		{ThreadPriorityP2Batch, 2, "P2_BATCH"},
		{ThreadPriorityP3Speculative, 3, "P3_SPECULATIVE"},
	}

	for _, tc := range tests {
		if tc.p.Rank() != tc.wantRank {
			t.Errorf("priority %v Rank() = %d, want %d", tc.p, tc.p.Rank(), tc.wantRank)
		}
		if tc.p.String() != tc.wantName {
			t.Errorf("priority %v String() = %q, want %q", tc.p, tc.p.String(), tc.wantName)
		}
		if !tc.p.IsValid() {
			t.Errorf("expected priority %v to be valid", tc.p)
		}
	}

	// Rank ordering: P0 < P1 < P2 < P3 (0 is highest priority)
	if !(ThreadPriorityP0System.Rank() < ThreadPriorityP1Interactive.Rank() &&
		ThreadPriorityP1Interactive.Rank() < ThreadPriorityP2Batch.Rank() &&
		ThreadPriorityP2Batch.Rank() < ThreadPriorityP3Speculative.Rank()) {
		t.Fatal("priority ranks must be strictly monotonic: P0 < P1 < P2 < P3")
	}

	// Invalid priority handling
	invalidPriority := ThreadPriority(99)
	if invalidPriority.IsValid() {
		t.Fatal("invalid priority should return false for IsValid()")
	}
	if !strings.Contains(invalidPriority.String(), "UNKNOWN") && !strings.Contains(invalidPriority.String(), "99") {
		t.Errorf("unexpected string representation for unknown priority: %q", invalidPriority.String())
	}
}

func TestCapacityAndLimitConstants(t *testing.T) {
	if MaxQueueCapacity != 512 {
		t.Fatalf("MaxQueueCapacity = %d, want 512", MaxQueueCapacity)
	}

	if DefaultWorkerPoolP0System <= 0 ||
		DefaultWorkerPoolP1Interactive <= 0 ||
		DefaultWorkerPoolP2Batch <= 0 ||
		DefaultWorkerPoolP3Speculative <= 0 {
		t.Fatal("all default worker pool allocations must be strictly positive")
	}

	totalDefaultTierWorkers := DefaultWorkerPoolP0System +
		DefaultWorkerPoolP1Interactive +
		DefaultWorkerPoolP2Batch +
		DefaultWorkerPoolP3Speculative

	if totalDefaultTierWorkers > DefaultMaxTotalWorkers {
		t.Fatalf("sum of tier workers (%d) exceeds DefaultMaxTotalWorkers (%d)",
			totalDefaultTierWorkers, DefaultMaxTotalWorkers)
	}

	// Byte limits must be positive and follow expected size hierarchy
	if MaxJournalSizeBytes <= 0 {
		t.Fatal("MaxJournalSizeBytes must be positive")
	}
	if MaxScratchStorageBytes <= 0 {
		t.Fatal("MaxScratchStorageBytes must be positive")
	}
	if MaxPerRunScratchBytes <= 0 {
		t.Fatal("MaxPerRunScratchBytes must be positive")
	}
	if MaxInMemoryCacheBytes <= 0 {
		t.Fatal("MaxInMemoryCacheBytes must be positive")
	}
	if MaxPerRunLogBytes <= 0 {
		t.Fatal("MaxPerRunLogBytes must be positive")
	}

	if MaxPerRunScratchBytes >= MaxScratchStorageBytes {
		t.Fatal("individual run scratch must be strictly smaller than total workspace scratch ceiling")
	}
}

func TestAdmissionErrorFormattingAndSentinels(t *testing.T) {
	if ErrQueueFull == nil {
		t.Fatal("ErrQueueFull sentinel must not be nil")
	}
	if ErrResourceConstrained == nil {
		t.Fatal("ErrResourceConstrained sentinel must not be nil")
	}

	if ErrQueueFull.Code != AdmissionCodeQueueFull {
		t.Errorf("ErrQueueFull.Code = %q, want %q", ErrQueueFull.Code, AdmissionCodeQueueFull)
	}
	if ErrResourceConstrained.Code != AdmissionCodeResourceConstrained {
		t.Errorf("ErrResourceConstrained.Code = %q, want %q", ErrResourceConstrained.Code, AdmissionCodeResourceConstrained)
	}

	// Error formatting with retry
	errStr := ErrQueueFull.Error()
	if !strings.Contains(errStr, AdmissionCodeQueueFull) || !strings.Contains(errStr, "retry after") {
		t.Errorf("unexpected Error() string: %q", errStr)
	}

	// Error formatting without retry
	errNoRetry := &AdmissionError{Code: "ERR_TEST", Message: "test message", RetryAfterMS: 0}
	if errNoRetry.Error() != "ERR_TEST: test message" {
		t.Errorf("unexpected Error() string: %q", errNoRetry.Error())
	}

	// Nil error
	var nilErr *AdmissionError
	if nilErr.Error() != "<nil>" {
		t.Errorf("nil AdmissionError.Error() = %q, want \"<nil>\"", nilErr.Error())
	}
}

func TestAdmissionErrorEqualityAndErrorsIs(t *testing.T) {
	customQueueFull := NewQueueFullError(250)
	if !errors.Is(customQueueFull, ErrQueueFull) {
		t.Fatal("customQueueFull should match ErrQueueFull via errors.Is (same Code)")
	}

	customResourceErr := NewResourceConstrainedError("memory pressure", 500)
	if !errors.Is(customResourceErr, ErrResourceConstrained) {
		t.Fatal("customResourceErr should match ErrResourceConstrained via errors.Is (same Code)")
	}

	if errors.Is(customQueueFull, ErrResourceConstrained) {
		t.Fatal("customQueueFull must not match ErrResourceConstrained")
	}

	if errors.Is(ErrQueueFull, errors.New("other error")) {
		t.Fatal("ErrQueueFull must not match unrelated error")
	}

	// Factory functions handle non-positive defaults
	zeroDelayQueue := NewQueueFullError(0)
	if zeroDelayQueue.RetryAfterMS != DefaultRetryAfterQueueFullMS {
		t.Errorf("NewQueueFullError(0) RetryAfterMS = %d, want default %d",
			zeroDelayQueue.RetryAfterMS, DefaultRetryAfterQueueFullMS)
	}

	defaultMsgResource := NewResourceConstrainedError("", 0)
	if defaultMsgResource.Message == "" {
		t.Error("NewResourceConstrainedError(\"\", 0) should use default message")
	}
	if defaultMsgResource.RetryAfterMS != DefaultRetryAfterResourceConstrainedMS {
		t.Errorf("NewResourceConstrainedError(\"\", 0) RetryAfterMS = %d, want default %d",
			defaultMsgResource.RetryAfterMS, DefaultRetryAfterResourceConstrainedMS)
	}
}

func TestAdmissionErrorJSONRoundTrip(t *testing.T) {
	orig := &AdmissionError{
		Code:         AdmissionCodeQueueFull,
		Message:      "queue cap reached",
		RetryAfterMS: 75,
	}

	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var decoded AdmissionError
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if decoded.Code != orig.Code {
		t.Errorf("decoded.Code = %q, want %q", decoded.Code, orig.Code)
	}
	if decoded.Message != orig.Message {
		t.Errorf("decoded.Message = %q, want %q", decoded.Message, orig.Message)
	}
	if decoded.RetryAfterMS != orig.RetryAfterMS {
		t.Errorf("decoded.RetryAfterMS = %d, want %d", decoded.RetryAfterMS, orig.RetryAfterMS)
	}
}
