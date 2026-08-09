package microagent

import (
	"context"
	"net/http"
	"testing"
	"time"
)

func TestAPIAdmissionEnforcesProviderEnvelopes(t *testing.T) {
	a, err := NewAPIAdmission(APIProviderShape{Name: "opaque", ReuseEvidence: "opaque", RequestsPerMinute: 2, TokensPerMinute: 100, Concurrency: 1, MaxSpendMicros: 50, PromptMicrosPerToken: 1, OutputMicrosPerToken: 2})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := a.Acquire(context.Background(), 20, 10)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := a.Acquire(ctx, 1, 1); err == nil {
		t.Fatal("expected bounded concurrency wait cancellation")
	}
	lease.Release()
	if _, err := a.Acquire(context.Background(), 20, 10); err == nil {
		t.Fatal("expected spend refusal")
	}
	if got := a.ReuseClaim(100); got != "not-observable" {
		t.Fatalf("opaque cache claim=%q", got)
	}
}

func TestAPIAdmissionReconcilesConservativeReservation(t *testing.T) {
	a, _ := NewAPIAdmission(APIProviderShape{Name: "explicit", RequestsPerMinute: 2, TokensPerMinute: 50, Concurrency: 1, MaxSpendMicros: 100, PromptMicrosPerToken: 1, OutputMicrosPerToken: 2})
	lease, err := a.Acquire(context.Background(), 40, 10)
	if err != nil {
		t.Fatal(err)
	}
	lease.Reconcile(10, 2)
	lease.Release()
	second, err := a.Acquire(context.Background(), 20, 5)
	if err != nil {
		t.Fatalf("reconciled reservation should admit second request: %v", err)
	}
	second.Release()
}
func TestAPIAdmissionExplicitCacheAndRetryAfter(t *testing.T) {
	a, err := NewAPIAdmission(APIProviderShape{Name: "explicit", ReuseControl: "provider-explicit", ReuseEvidence: "billed-cached-tokens", RequestsPerMinute: 10, TokensPerMinute: 100, Concurrency: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := a.ReuseClaim(10); got != "provider-billed-cache-hit" {
		t.Fatalf("cache claim=%q", got)
	}
	h := http.Header{"Retry-After": []string{"7"}}
	if got := RetryAfter(h, time.Now()); got != 7*time.Second {
		t.Fatalf("retry=%s", got)
	}
}

func TestAPIAdmissionCancellationAndTPM(t *testing.T) {
	a, _ := NewAPIAdmission(APIProviderShape{Name: "opaque", RequestsPerMinute: 10, TokensPerMinute: 10, Concurrency: 2})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := a.Acquire(ctx, 1, 1); err == nil {
		t.Fatal("expected cancellation")
	}
	if _, err := a.Acquire(context.Background(), 9, 2); err == nil {
		t.Fatal("expected TPM refusal")
	}
}
