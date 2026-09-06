package researcharm

import (
	"context"
	"errors"
	"net/http/httptest"
	"testing"
	"time"
)

func BenchmarkExtractOriginExplicitHeaders(b *testing.B) {
	req := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "orch-eval-sweep")
	req.Header.Set("X-Fak-Caller-PID", "45678")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		origin := ExtractOrigin(req, "trace-123")
		if !origin.Explicit {
			b.Fatal("expected explicit origin")
		}
	}
}

func BenchmarkExtractOriginInferredTrace(b *testing.B) {
	req := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/chat/completions", nil)
	req.RemoteAddr = "10.0.0.1:12345"
	traceID := "orch-62ef4cd4e1ce-worker-1"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		origin := ExtractOrigin(req, traceID)
		if origin.ArmID != "orch" {
			b.Fatalf("expected arm orch, got %s", origin.ArmID)
		}
	}
}

func BenchmarkExtractOriginInferredUserAgent(b *testing.B) {
	req := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/chat/completions", nil)
	req.RemoteAddr = "10.0.0.1:12345"
	req.Header.Set("User-Agent", "curl/8.4.0")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		origin := ExtractOrigin(req, "")
		if origin.ArmID != "curl-client" {
			b.Fatalf("expected arm curl-client, got %s", origin.ArmID)
		}
	}
}

func BenchmarkCoordinatorConcurrencyLimit(b *testing.B) {
	coord := NewCoordinator(2)
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "arm-bench")
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lease, err := coord.Admit(ctx, req, "/v1/chat/completions", "trace-limit")
		if err != nil {
			b.Fatalf("admit failed: %v", err)
		}
		lease.Done(25, nil)
	}
}

func BenchmarkCoordinatorConcurrencyExceeded(b *testing.B) {
	coord := NewCoordinator(1)
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "arm-saturated")
	ctx := context.Background()

	held, err := coord.Admit(ctx, req, "/v1/chat/completions", "trace-held")
	if err != nil {
		b.Fatalf("initial admit failed: %v", err)
	}
	defer held.Done(0, nil)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := coord.Admit(ctx, req, "/v1/chat/completions", "trace-blocked")
		if !errors.Is(err, ErrArmConcurrencyExceeded) {
			b.Fatalf("expected ErrArmConcurrencyExceeded, got %v", err)
		}
	}
}

func BenchmarkCoordinatorExclusiveLease(b *testing.B) {
	coord := NewCoordinator(10)
	lease, err := coord.AcquireLease(LeaseRequest{
		ArmID:     "bench-exclusive",
		HolderPID: 12345,
		Mode:      LeaseModeExclusive,
		TTL:       5 * time.Minute,
	})
	if err != nil {
		b.Fatalf("failed to acquire exclusive lease: %v", err)
	}
	defer coord.ReleaseLease(lease.ID, lease.Token)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "bench-exclusive")
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l, err := coord.Admit(ctx, req, "/v1/chat/completions", "trace-excl")
		if err != nil {
			b.Fatalf("admit failed: %v", err)
		}
		l.Done(10, nil)
	}
}

func BenchmarkCoordinatorExclusiveLeaseForeignRejected(b *testing.B) {
	coord := NewCoordinator(10)
	lease, err := coord.AcquireLease(LeaseRequest{
		ArmID:     "bench-exclusive",
		HolderPID: 12345,
		Mode:      LeaseModeExclusive,
		TTL:       5 * time.Minute,
	})
	if err != nil {
		b.Fatalf("failed to acquire exclusive lease: %v", err)
	}
	defer coord.ReleaseLease(lease.ID, lease.Token)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "foreign-arm")
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := coord.Admit(ctx, req, "/v1/chat/completions", "trace-foreign")
		if !errors.Is(err, ErrExclusiveLeaseHeld) {
			b.Fatalf("expected ErrExclusiveLeaseHeld, got %v", err)
		}
	}
}

func BenchmarkCoordinatorAcquireReleaseLease(b *testing.B) {
	coord := NewCoordinator(10)
	req := LeaseRequest{
		ArmID:     "bench-arm",
		HolderPID: 1234,
		Mode:      LeaseModeExclusive,
		TTL:       time.Minute,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lease, err := coord.AcquireLease(req)
		if err != nil {
			b.Fatalf("acquire lease failed: %v", err)
		}
		if err := coord.ReleaseLease(lease.ID, lease.Token); err != nil {
			b.Fatalf("release lease failed: %v", err)
		}
	}
}

func BenchmarkCoordinatorSnapshot(b *testing.B) {
	coord := NewCoordinator(10)
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "arm-snap")
	ctx := context.Background()

	l, err := coord.Admit(ctx, req, "/v1/chat/completions", "trace-snap")
	if err != nil {
		b.Fatalf("admit failed: %v", err)
	}
	defer l.Done(10, nil)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = coord.Snapshot()
	}
}
