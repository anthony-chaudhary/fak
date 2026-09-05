package researcharm

import (
	"context"
	"errors"
	"net/http/httptest"
	"testing"
	"time"
)

func TestExtractOriginExplicitHeaders(t *testing.T) {
	req := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "orch-eval-sweep")
	req.Header.Set("X-Fak-Caller-PID", "45678")

	origin := ExtractOrigin(req, "trace-123")
	if origin.ArmID != "orch-eval-sweep" {
		t.Errorf("expected ArmID orch-eval-sweep, got %s", origin.ArmID)
	}
	if origin.CallerPID != 45678 {
		t.Errorf("expected CallerPID 45678, got %d", origin.CallerPID)
	}
	if !origin.Explicit {
		t.Errorf("expected Explicit true, got false")
	}
}

func TestExtractOriginInferredFromTrace(t *testing.T) {
	tests := []struct {
		traceID   string
		wantArm   string
		wantGroup string
	}{
		{"orch-62ef4cd4e1ce-worker-1", "orch", "orch"},
		{"codex-clear-54312fdf8801ad7e", "codex", "codex"},
		{"guard-b600cbf88306f2c8", "guard", "guard"},
		{"gw-2", "gw-bench", "bench"},
		{"fleet-9817-worker", "fleet-worker", "fleet"},
		{"npc1_10005000000000000000000000000000", "npc-sim", "sim"},
	}

	for _, tt := range tests {
		req := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/chat/completions", nil)
		origin := ExtractOrigin(req, tt.traceID)
		if origin.ArmID != tt.wantArm {
			t.Errorf("trace %q: expected arm %s, got %s", tt.traceID, tt.wantArm, origin.ArmID)
		}
		if origin.ArmGroup != tt.wantGroup {
			t.Errorf("trace %q: expected group %s, got %s", tt.traceID, tt.wantGroup, origin.ArmGroup)
		}
	}
}

func TestExtractOriginInferredFromUserAgent(t *testing.T) {
	req := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/chat/completions", nil)
	req.Header.Set("User-Agent", "curl/8.4.0")

	origin := ExtractOrigin(req, "")
	if origin.ArmID != "curl-client" {
		t.Errorf("expected curl-client, got %s", origin.ArmID)
	}
}

func TestCoordinatorConcurrencyLimit(t *testing.T) {
	coord := NewCoordinator(2) // Max 2 per arm by default

	req1 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req1.Header.Set("X-Fak-Project-Arm", "arm-a")

	l1, err := coord.Admit(context.Background(), req1, "/v1/chat/completions", "trace-1")
	if err != nil {
		t.Fatalf("first admit failed: %v", err)
	}

	l2, err := coord.Admit(context.Background(), req1, "/v1/chat/completions", "trace-2")
	if err != nil {
		t.Fatalf("second admit failed: %v", err)
	}

	// Third request should exceed concurrency limit
	_, err = coord.Admit(context.Background(), req1, "/v1/chat/completions", "trace-3")
	if !errors.Is(err, ErrArmConcurrencyExceeded) {
		t.Fatalf("expected ErrArmConcurrencyExceeded, got %v", err)
	}

	// Request from arm-b should still succeed (fairness between arms)
	req2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req2.Header.Set("X-Fak-Project-Arm", "arm-b")

	lb1, err := coord.Admit(context.Background(), req2, "/v1/chat/completions", "trace-b1")
	if err != nil {
		t.Fatalf("arm-b admit failed: %v", err)
	}

	// Completing l1 frees a slot for arm-a
	l1.Done(50, nil)

	l3, err := coord.Admit(context.Background(), req1, "/v1/chat/completions", "trace-3-retry")
	if err != nil {
		t.Fatalf("retry admit after Done failed: %v", err)
	}

	l2.Done(30, nil)
	lb1.Done(40, nil)
	l3.Done(20, nil)

	snap := coord.Snapshot()
	if snap.TotalInflight != 0 {
		t.Errorf("expected 0 in-flight, got %d", snap.TotalInflight)
	}
	for _, arm := range snap.Arms {
		if arm.ID == "arm-a" {
			if arm.TotalRequests != 3 {
				t.Errorf("arm-a total requests: got %d, want 3", arm.TotalRequests)
			}
			if arm.TotalTokens != 100 {
				t.Errorf("arm-a total tokens: got %d, want 100", arm.TotalTokens)
			}
		}
	}
}

func TestCoordinatorExclusiveLease(t *testing.T) {
	coord := NewCoordinator(10)

	// Arm "bench-exclusive" acquires exclusive lease
	lease, err := coord.AcquireLease(LeaseRequest{
		ArmID:     "bench-exclusive",
		HolderPID: 12345,
		Mode:      LeaseModeExclusive,
		TTL:       1 * time.Minute,
	})
	if err != nil {
		t.Fatalf("failed to acquire exclusive lease: %v", err)
	}
	if lease.Token == "" {
		t.Fatal("expected lease token, got empty")
	}

	// Request from another arm should be rejected
	reqOther := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	reqOther.Header.Set("X-Fak-Project-Arm", "arm-other")

	_, err = coord.Admit(context.Background(), reqOther, "/v1/chat/completions", "trace-other")
	if !errors.Is(err, ErrExclusiveLeaseHeld) {
		t.Fatalf("expected ErrExclusiveLeaseHeld, got %v", err)
	}

	// Request from lease holder arm should succeed
	reqHolder := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	reqHolder.Header.Set("X-Fak-Project-Arm", "bench-exclusive")

	l, err := coord.Admit(context.Background(), reqHolder, "/v1/chat/completions", "trace-holder")
	if err != nil {
		t.Fatalf("holder admit failed: %v", err)
	}
	l.Done(10, nil)

	// Release lease with invalid token should fail
	if err := coord.ReleaseLease(lease.ID, "bad-token"); !errors.Is(err, ErrInvalidLeaseToken) {
		t.Fatalf("expected ErrInvalidLeaseToken, got %v", err)
	}

	// Release lease with correct token
	if err := coord.ReleaseLease(lease.ID, lease.Token); err != nil {
		t.Fatalf("failed to release lease: %v", err)
	}

	// Now other arm can admit
	lOther, err := coord.Admit(context.Background(), reqOther, "/v1/chat/completions", "trace-other-2")
	if err != nil {
		t.Fatalf("admit after lease release failed: %v", err)
	}
	lOther.Done(15, nil)
}

func TestCoordinatorEnforceLeases(t *testing.T) {
	coord := NewCoordinator(10)
	coord.SetEnforceLeases(true)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "unleased-arm")

	// Admission should fail because no lease is held
	_, err := coord.Admit(context.Background(), req, "/v1/chat/completions", "trace-1")
	if !errors.Is(err, ErrLeaseRequired) {
		t.Fatalf("expected ErrLeaseRequired, got %v", err)
	}

	// Acquire shared lease
	_, err = coord.AcquireLease(LeaseRequest{
		ArmID: "unleased-arm",
		Mode:  LeaseModeShared,
		TTL:   5 * time.Minute,
	})
	if err != nil {
		t.Fatalf("failed to acquire shared lease: %v", err)
	}

	// Now admission succeeds
	l, err := coord.Admit(context.Background(), req, "/v1/chat/completions", "trace-2")
	if err != nil {
		t.Fatalf("admit with lease failed: %v", err)
	}
	l.Done(5, nil)
}
