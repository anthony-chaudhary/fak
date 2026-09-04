package nativeperf

import (
	"errors"
	"math"
	"testing"
)

func TestTabulateClockCoverageBuckets(t *testing.T) {
	observations := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 1980.0, 9500.0),
		BindPostPassClockObservation("pass-1", 1, "gpu-0", 1975.0, 9500.0),
		BindFailedClockObservation("pass-2", 2, "gpu-0", errors.New("nvml error")),
		BindUnreadableClockObservation("pass-3", 3, "gpu-0", "counter unsupported"),
	}

	buckets := TabulateClockCoverage(observations)
	if buckets.Total != 4 {
		t.Fatalf("expected total 4, got %d", buckets.Total)
	}
	if buckets.Observed != 2 {
		t.Fatalf("expected observed 2, got %d", buckets.Observed)
	}
	if buckets.Failed != 1 {
		t.Fatalf("expected failed 1, got %d", buckets.Failed)
	}
	if buckets.Unreadable != 1 {
		t.Fatalf("expected unreadable 1, got %d", buckets.Unreadable)
	}
	if math.Abs(buckets.CoverageRatio()-0.5) > 1e-6 {
		t.Fatalf("expected coverage ratio 0.5, got %f", buckets.CoverageRatio())
	}
	if buckets.IsComplete() {
		t.Fatalf("expected IsComplete() to be false")
	}

	completeObs := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 2000.0, 9000.0),
		BindPostPassClockObservation("pass-1", 1, "gpu-0", 2000.0, 9000.0),
	}
	completeBuckets := TabulateClockCoverage(completeObs)
	if !completeBuckets.IsComplete() {
		t.Fatalf("expected completeBuckets.IsComplete() to be true")
	}
}

func TestValidateClockObservations_PassWithinTolerance(t *testing.T) {
	observations := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 2000.0, 9500.0),
		BindPostPassClockObservation("pass-1", 1, "gpu-0", 2010.0, 9520.0),
		BindPostPassClockObservation("pass-2", 2, "gpu-0", 1995.0, 9480.0),
	}

	policy := ClockTolerancePolicy{
		MinCoverageRatio:           1.0,
		MaxSMClockDriftPercent:     2.0,
		MaxMemoryClockDriftPercent: 2.0,
	}

	res, err := ValidateClockObservations(observations, policy)
	if err != nil {
		t.Fatalf("expected validation to pass, got error: %v", err)
	}
	if !res.Valid {
		t.Fatalf("expected res.Valid to be true")
	}
	if res.ObservedPasses != 3 {
		t.Fatalf("expected 3 observed passes, got %d", res.ObservedPasses)
	}
	if len(res.Violations) != 0 {
		t.Fatalf("expected zero violations, got: %v", res.Violations)
	}
}

func TestValidateClockObservations_RejectsSMDriftExceeded(t *testing.T) {
	// Mean SM clock ~ 1800, min 1500, max 2100 -> drift > 15%
	observations := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 1500.0, 9500.0),
		BindPostPassClockObservation("pass-1", 1, "gpu-0", 2100.0, 9500.0),
	}

	policy := ClockTolerancePolicy{
		MinCoverageRatio:           1.0,
		MaxSMClockDriftPercent:     5.0,
		MaxMemoryClockDriftPercent: 5.0,
	}

	res, err := ValidateClockObservations(observations, policy)
	if err == nil {
		t.Fatalf("expected error due to SM clock drift, got nil")
	}
	if res.Valid {
		t.Fatalf("expected res.Valid to be false")
	}

	var clockErr *ClockValidationError
	if !errors.As(err, &clockErr) {
		t.Fatalf("expected error to be *ClockValidationError, got %T", err)
	}
	if res.SMDriftPercent <= 5.0 {
		t.Fatalf("expected SMDriftPercent > 5.0, got %f", res.SMDriftPercent)
	}
}

func TestValidateClockObservations_RejectsMemoryDriftExceeded(t *testing.T) {
	observations := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 2000.0, 8000.0),
		BindPostPassClockObservation("pass-1", 1, "gpu-0", 2000.0, 9500.0),
	}

	policy := ClockTolerancePolicy{
		MinCoverageRatio:           1.0,
		MaxSMClockDriftPercent:     5.0,
		MaxMemoryClockDriftPercent: 5.0,
	}

	res, err := ValidateClockObservations(observations, policy)
	if err == nil {
		t.Fatalf("expected error due to memory clock drift, got nil")
	}
	if res.Valid {
		t.Fatalf("expected res.Valid to be false")
	}
	if res.MemoryDriftPercent <= 5.0 {
		t.Fatalf("expected MemoryDriftPercent > 5.0, got %f", res.MemoryDriftPercent)
	}
}

func TestValidateClockObservations_TargetClockTolerance(t *testing.T) {
	observations := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 1950.0, 9500.0),
		BindPostPassClockObservation("pass-1", 1, "gpu-0", 1960.0, 9500.0),
	}

	// Target is 2200 MHz; 1950 vs 2200 is ~11.36% drift
	policy := ClockTolerancePolicy{
		MinCoverageRatio:           1.0,
		TargetSMClockMHz:           2200.0,
		MaxSMClockDriftPercent:     5.0,
		MaxMemoryClockDriftPercent: 5.0,
	}

	res, err := ValidateClockObservations(observations, policy)
	if err == nil {
		t.Fatalf("expected target drift failure, got nil")
	}
	if res.Valid {
		t.Fatalf("expected res.Valid to be false")
	}
}

func TestValidateClockObservations_CoverageViolations(t *testing.T) {
	// Only 1 out of 2 passes observed
	observations := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 2000.0, 9500.0),
		BindFailedClockObservation("pass-1", 1, "gpu-0", errors.New("read error")),
	}

	policy := ClockTolerancePolicy{
		MinCoverageRatio: 1.0,
	}

	res, err := ValidateClockObservations(observations, policy)
	if err == nil {
		t.Fatalf("expected coverage failure due to failed pass, got nil")
	}
	if res.Valid {
		t.Fatalf("expected invalid result")
	}
}

func TestValidateClockObservations_UnreadableAllowedAndDisallowed(t *testing.T) {
	observations := []PostPassClockObservation{
		BindUnreadableClockObservation("pass-0", 0, "gpu-0", "unsupported on platform"),
		BindUnreadableClockObservation("pass-1", 1, "gpu-0", "unsupported on platform"),
	}

	// Disallowed by default
	disallowedPolicy := ClockTolerancePolicy{
		MinCoverageRatio: 0.0,
		AllowUnreadable:  false,
	}
	res, err := ValidateClockObservations(observations, disallowedPolicy)
	if err == nil || res.Valid {
		t.Fatalf("expected failure when unreadable is disallowed")
	}

	// Allowed when policy opts in and MinCoverageRatio is 0
	allowedPolicy := ClockTolerancePolicy{
		MinCoverageRatio: 0.0,
		AllowUnreadable:  true,
	}
	res2, err2 := ValidateClockObservations(observations, allowedPolicy)
	if err2 != nil || !res2.Valid {
		t.Fatalf("expected success when unreadable is allowed, got err: %v", err2)
	}
}

func TestClockObservations_SerializationRoundTrip(t *testing.T) {
	orig := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 2000.0, 9500.0),
		BindPostPassClockObservation("pass-1", 1, "gpu-0", 2010.0, 9520.0),
	}

	bytes, err := MarshalClockObservations(orig)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	parsed, err := UnmarshalClockObservations(bytes)
	if err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if len(parsed) != len(orig) {
		t.Fatalf("expected %d, got %d", len(orig), len(parsed))
	}
	if parsed[0].PassID != orig[0].PassID || parsed[0].SMClockMHz != orig[0].SMClockMHz {
		t.Fatalf("data mismatch after roundtrip: %+v vs %+v", parsed[0], orig[0])
	}
}

func TestClockObservation_PredicatesAndFiniteChecks(t *testing.T) {
	valid := BindPostPassClockObservation("pass-0", 0, "gpu-0", 2100.0, 9500.0)
	if !valid.IsObserved() || valid.IsFailed() || valid.IsUnreadable() {
		t.Fatalf("expected valid to be observed only")
	}

	unreadable := BindUnreadableClockObservation("pass-0", 0, "gpu-0", "unsupported")
	if unreadable.IsObserved() || unreadable.IsFailed() || !unreadable.IsUnreadable() {
		t.Fatalf("expected unreadable to be unreadable only")
	}

	failed := BindFailedClockObservation("pass-0", 0, "gpu-0", errors.New("i/o error"))
	if failed.IsObserved() || !failed.IsFailed() || failed.IsUnreadable() {
		t.Fatalf("expected failed to be failed only")
	}

	nanObs := BindPostPassClockObservation("pass-0", 0, "gpu-0", math.NaN(), 9500.0)
	if nanObs.IsObserved() || !nanObs.IsFailed() {
		t.Fatalf("expected NaN clock rate to not be observed and be marked failed")
	}

	infObs := BindPostPassClockObservation("pass-0", 0, "gpu-0", 2100.0, math.Inf(1))
	if infObs.IsObserved() || !infObs.IsFailed() {
		t.Fatalf("expected +Inf clock rate to not be observed and be marked failed")
	}

	zeroObs := BindPostPassClockObservation("pass-0", 0, "gpu-0", 0.0, 9500.0)
	if zeroObs.IsObserved() || !zeroObs.IsFailed() {
		t.Fatalf("expected zero clock rate to not be observed and be marked failed")
	}

	negObs := BindPostPassClockObservation("pass-0", 0, "gpu-0", -100.0, 9500.0)
	if negObs.IsObserved() || !negObs.IsFailed() {
		t.Fatalf("expected negative clock rate to not be observed and be marked failed")
	}
}

func TestClockCoverageBuckets_RatiosAndEmpty(t *testing.T) {
	var empty ClockCoverageBuckets
	if empty.CoverageRatio() != 0 || empty.ObservedRatio() != 0 || empty.FailedRatio() != 0 || empty.UnreadableRatio() != 0 {
		t.Fatalf("expected 0 ratios for empty buckets")
	}
	if empty.IsComplete() {
		t.Fatalf("empty buckets should not be complete")
	}

	buckets := ClockCoverageBuckets{
		Observed:   2,
		Failed:     1,
		Unreadable: 1,
		Total:      4,
	}
	if math.Abs(buckets.ObservedRatio()-0.5) > 1e-6 {
		t.Fatalf("expected observed ratio 0.5, got %f", buckets.ObservedRatio())
	}
	if math.Abs(buckets.FailedRatio()-0.25) > 1e-6 {
		t.Fatalf("expected failed ratio 0.25, got %f", buckets.FailedRatio())
	}
	if math.Abs(buckets.UnreadableRatio()-0.25) > 1e-6 {
		t.Fatalf("expected unreadable ratio 0.25, got %f", buckets.UnreadableRatio())
	}
}

func TestClockValidationError_Formatting(t *testing.T) {
	errEmpty := &ClockValidationError{}
	if errEmpty.Error() != "nativeperf: clock validation failed" {
		t.Fatalf("unexpected empty error string: %s", errEmpty.Error())
	}

	errWithViolations := &ClockValidationError{
		Result: ClockValidationResult{
			Violations: []string{"violation 1", "violation 2"},
		},
	}
	expected := "nativeperf: clock validation failed: violation 1; violation 2"
	if errWithViolations.Error() != expected {
		t.Fatalf("expected %q, got %q", expected, errWithViolations.Error())
	}
}

func TestGPUClockEvidence_ContainerAndRoundTrip(t *testing.T) {
	obs := []PostPassClockObservation{
		BindPostPassClockObservation("pass-0", 0, "gpu-0", 2000.0, 9500.0),
	}
	ev := NewGPUClockEvidence("gpu-0", obs)
	if ev.Schema != GPUClockObservationSchema {
		t.Fatalf("expected schema %q, got %q", GPUClockObservationSchema, ev.Schema)
	}
	if ev.DeviceID != "gpu-0" {
		t.Fatalf("expected device id gpu-0, got %q", ev.DeviceID)
	}
	if len(ev.Observations) != 1 {
		t.Fatalf("expected 1 observation, got %d", len(ev.Observations))
	}
}

func TestValidateClockObservations_EmptyInputs(t *testing.T) {
	// Zero observations with MinCoverageRatio > 0
	policyReject := ClockTolerancePolicy{MinCoverageRatio: 0.5}
	res, err := ValidateClockObservations(nil, policyReject)
	if err == nil || res.Valid {
		t.Fatalf("expected rejection for empty observations with MinCoverageRatio > 0")
	}

	// Zero observations with MinCoverageRatio == 0
	policyAllow := ClockTolerancePolicy{MinCoverageRatio: 0}
	res2, err2 := ValidateClockObservations(nil, policyAllow)
	if err2 != nil || !res2.Valid {
		t.Fatalf("expected acceptance for empty observations with MinCoverageRatio == 0, got err: %v", err2)
	}
}
