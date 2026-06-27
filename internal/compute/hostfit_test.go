package compute

import (
	"errors"
	"strings"
	"testing"
)

// TestRefuseMemoryPlanForHostMem covers the pure-CPU reference-path host-RAM fit refusal
// (refuseMemoryPlanForHostMem, the injectable core of RefuseMemoryPlanIfTooBigForHost, #974):
// it must refuse a plan that exceeds MemAvailable-less-headroom with a typed host-scoped
// FitError, pass a plan that fits, and FAIL OPEN when the host cannot report its memory.
func TestRefuseMemoryPlanForHostMem(t *testing.T) {
	const gib = int64(1) << 30
	// Weights + KV demands the way serveGGUFPathMemoryPlan builds them for the CPU path; Total()
	// sums them regardless of scope (on a CPU serve they are all anonymous host RAM).
	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 433 * gib, Detail: "gguf-load", Scope: MemoryScopeDevice},
		{Class: MemoryKVCache, Bytes: 5 * gib, Detail: "kv", Scope: MemoryScopeDevice},
	}
	const headroom = 0.15

	// The GLM-5.2 wedge scenario: ~495 GiB usable (hugepage-locked co-tenant capped MemAvailable),
	// 438 GiB plan. budget = 495*0.85 = ~420.75 GiB < 438 GiB plan -> refuse before the load wedges.
	err := refuseMemoryPlanForHostMem(plan, 1024*gib, 495*gib, true, headroom)
	if err == nil {
		t.Fatal("438 GiB plan into 495 GiB MemAvailable (15% headroom): want FitTooBig refusal, got nil")
	}
	var fe *FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want *FitError, got %T: %v", err, err)
	}
	if fe.Verdict != FitTooBig {
		t.Errorf("verdict = %s, want too_big", fe.Verdict)
	}
	if fe.Scope != MemoryScopeHost {
		t.Errorf("scope = %q, want host (the demand is anonymous host RAM)", fe.Scope)
	}
	if fe.Want != 438*gib {
		t.Errorf("want bytes = %d, want %d (plan grand total)", fe.Want, 438*gib)
	}
	if fe.Avail != int64(float64(495*gib)*(1-headroom)) {
		t.Errorf("avail = %d, want headroom-adjusted budget %d", fe.Avail, int64(float64(495*gib)*(1-headroom)))
	}
	if msg := fe.Error(); !strings.Contains(msg, "host has") || !strings.Contains(msg, "FitTooBig") {
		t.Errorf("error message %q missing host-scope sizing/FitTooBig", msg)
	}

	// A box with ample headroom (1 TiB usable): the same plan fits and must NOT refuse.
	if err := refuseMemoryPlanForHostMem(plan, 1100*gib, 1024*gib, true, headroom); err != nil {
		t.Errorf("438 GiB plan into 1024 GiB MemAvailable: want nil (fits), got %v", err)
	}

	// Fail-open: a platform that cannot report host memory (known=false) must never refuse, so the
	// portable floor and any non-reporting host load exactly as before the pre-flight existed.
	if err := refuseMemoryPlanForHostMem(plan, 0, FreeUnknown, false, headroom); err != nil {
		t.Errorf("unreported host memory: want fail-open nil, got %v", err)
	}

	// An empty plan asks for nothing -> FitOK, never a refusal, even on a tiny box.
	if err := refuseMemoryPlanForHostMem(nil, 8*gib, 1*gib, true, headroom); err != nil {
		t.Errorf("empty plan: want nil, got %v", err)
	}
}
