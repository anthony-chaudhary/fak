package gateway

import (
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// upstreamErrorStatus must turn an in-kernel device OOM into a specific, actionable client
// response — a distinct 503 + "in_kernel_oom" code + a message that names the condition and the
// fix — while leaving genuine upstream errors classified exactly as before, with no provider
// body leaked. No GPU is needed: the OOM error is an ordinary agent.InKernelOOMError value.

func TestUpstreamErrorStatus_InKernelOOMIsActionable(t *testing.T) {
	const bytes = 4 << 30
	status, code, msg := upstreamErrorStatus(&agent.InKernelOOMError{Bytes: bytes, Class: compute.MemoryKVCache})

	if status != http.StatusServiceUnavailable {
		t.Fatalf("in-kernel OOM should be 503 (retryable local exhaustion), got %d", status)
	}
	if code != "in_kernel_oom" {
		t.Fatalf("in-kernel OOM should carry the distinct code, got %q", code)
	}
	// The message must be actionable: name the failure AND at least one concrete remedy.
	if !strings.Contains(msg, "out of memory") {
		t.Fatalf("message does not name the failure: %q", msg)
	}
	if !strings.Contains(msg, "kv cache") {
		t.Fatalf("message does not surface the memory class: %q", msg)
	}
	if !strings.Contains(msg, "smaller model") && !strings.Contains(msg, "reduce the prompt") {
		t.Fatalf("message is not actionable (no remedy): %q", msg)
	}
}

func TestUpstreamErrorStatus_InKernelCapacityPrecheckIsActionable(t *testing.T) {
	status, code, msg := upstreamErrorStatus(&agent.InKernelCapacityError{
		Want:  96 << 20,
		Avail: 64 << 20,
		Class: compute.MemoryKVCache,
		Scope: compute.MemoryScopeDevice,
		Site:  "capacity-precheck",
	})

	if status != http.StatusServiceUnavailable {
		t.Fatalf("capacity precheck refusal should be 503 (retryable local exhaustion), got %d", status)
	}
	if code != "in_kernel_oom" {
		t.Fatalf("capacity precheck refusal should carry the distinct code, got %q", code)
	}
	for _, want := range []string{"capacity precheck", "kv cache", "reduce the prompt"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("capacity message missing %q: %q", want, msg)
		}
	}
}

// A genuine upstream error must NOT be misclassified as an in-kernel OOM, and its raw provider
// body must never cross the trust boundary into the client message (#82/#346 invariant).
func TestUpstreamErrorStatus_RealUpstreamErrorDoesNotLeakOrMisclassify(t *testing.T) {
	const secret = "SECRET_UPSTREAM_BODY_should_never_reach_client"

	// 5xx upstream → opaque 502, no body, not the OOM code.
	status, code, msg := upstreamErrorStatus(&agent.UpstreamStatusError{Status: 500, Body: secret})
	if code == "in_kernel_oom" {
		t.Fatal("a real upstream 500 must not be classified as in_kernel_oom")
	}
	if status != http.StatusBadGateway {
		t.Fatalf("a real upstream 5xx should be 502, got %d", status)
	}
	if strings.Contains(msg, secret) {
		t.Fatalf("upstream body leaked into the client message: %q", msg)
	}

	// 4xx upstream → surfaced status, still no body leak, still not the OOM code.
	status, code, msg = upstreamErrorStatus(&agent.UpstreamStatusError{Status: 404, Body: secret})
	if code == "in_kernel_oom" {
		t.Fatal("a real upstream 404 must not be classified as in_kernel_oom")
	}
	if status != 404 {
		t.Fatalf("a real upstream 4xx should surface its status, got %d", status)
	}
	if strings.Contains(msg, secret) {
		t.Fatalf("upstream body leaked into the client message: %q", msg)
	}
}
