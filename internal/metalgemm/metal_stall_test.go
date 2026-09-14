package metalgemm

import (
	"errors"
	"math"
	"testing"
)

func TestCheckCommandBufferWaitClassifiesStall(t *testing.T) {
	const limit = 10_000.0

	cases := []struct {
		name     string
		op       string
		waitedMS float64
		limitMS  float64
		wantErr  bool
	}{
		{name: "under limit is healthy", op: "q4_k-gemv", waitedMS: 0, limitMS: limit, wantErr: false},
		{name: "just under limit is healthy", op: "q8-gemv", waitedMS: limit - 0.001, limitMS: limit, wantErr: false},
		{name: "at limit is a stall", op: "q4_k-gemm", waitedMS: limit, limitMS: limit, wantErr: true},
		{name: "over limit is a stall", op: "q2_k-gemv", waitedMS: limit + 0.001, limitMS: limit, wantErr: true},
		{name: "far over limit is a stall", op: "decode", waitedMS: 30_000, limitMS: limit, wantErr: true},

		{name: "negative waited is a stall", op: "neg-wait", waitedMS: -1, limitMS: limit, wantErr: true},
		{name: "plus infinity waited is a stall", op: "posinf-wait", waitedMS: math.Inf(1), limitMS: limit, wantErr: true},
		{name: "minus infinity waited is a stall", op: "neginf-wait", waitedMS: math.Inf(-1), limitMS: limit, wantErr: true},
		{name: "NaN waited is a stall", op: "nan-wait", waitedMS: math.NaN(), limitMS: limit, wantErr: true},

		{name: "zero limit is a stall", op: "zero-limit", waitedMS: 1, limitMS: 0, wantErr: true},
		{name: "negative limit is a stall", op: "neg-limit", waitedMS: 1, limitMS: -1, wantErr: true},
		{name: "NaN limit is a stall", op: "nan-limit", waitedMS: 1, limitMS: math.NaN(), wantErr: true},
		{name: "plus infinity limit is a stall", op: "posinf-limit", waitedMS: 1, limitMS: math.Inf(1), wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckCommandBufferWait(tc.op, tc.waitedMS, tc.limitMS)
			if tc.wantErr && err == nil {
				t.Fatalf("waited %vms at limit %vms: want stall error, got nil", tc.waitedMS, tc.limitMS)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("waited %vms under limit %vms: want nil, got %v", tc.waitedMS, tc.limitMS, err)
			}
			if !tc.wantErr {
				return
			}
			if !IsMetalCommandBufferStall(err) {
				t.Fatalf("IsMetalCommandBufferStall(%v) = false, want true", err)
			}
			var stall MetalCommandBufferStallError
			if !errors.As(err, &stall) {
				t.Fatalf("errors.As(%v, MetalCommandBufferStallError) = false, want true", err)
			}
			if stall.Operation != tc.op {
				t.Errorf("Operation = %q, want %q", stall.Operation, tc.op)
			}
			if !sameFloat(stall.WaitedMilliseconds, tc.waitedMS) {
				t.Errorf("WaitedMilliseconds = %v, want %v", stall.WaitedMilliseconds, tc.waitedMS)
			}
			if !sameFloat(stall.LimitMilliseconds, tc.limitMS) {
				t.Errorf("LimitMilliseconds = %v, want %v", stall.LimitMilliseconds, tc.limitMS)
			}
		})
	}
}

func TestMetalCommandBufferStallErrorMessage(t *testing.T) {
	err := CheckCommandBufferWait("q4_k-gemv", 12_500, 10_000)
	if err == nil {
		t.Fatal("want stall error, got nil")
	}
	msg := err.Error()
	for _, want := range []string{"q4_k-gemv", "12500.000ms", "10000.000ms"} {
		if !contains(msg, want) {
			t.Errorf("error message %q missing %q", msg, want)
		}
	}
}

func TestIsMetalCommandBufferStallRejectsOtherErrors(t *testing.T) {
	if IsMetalCommandBufferStall(nil) {
		t.Error("IsMetalCommandBufferStall(nil) = true, want false")
	}
	if IsMetalCommandBufferStall(errors.New("unrelated")) {
		t.Error("IsMetalCommandBufferStall(unrelated) = true, want false")
	}
	if IsMetalCommandBufferStall(ExecutionEventsUnavailableError{}) {
		t.Error("IsMetalCommandBufferStall(ExecutionEventsUnavailableError) = true, want false")
	}
}

func sameFloat(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return a == b
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
