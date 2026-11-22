package agent

import (
	"testing"
	"time"
)

// TestResolveFirstTokenWatchdogCPUOffloadExperts pins the host-bound prefill rule: an
// in-kernel chat whose routed experts are CPU-offloaded/streamed on a device backend is
// prefill-bound by the host exactly like a CPU-backend chat, so absent an in-band
// FAK_STREAM_STALL_TIMEOUT_S it resolves the same CPUBackendFirstTokenWatchdogTimeout under a
// distinct source; the env override still wins and a plain device chat keeps the 60s default.
func TestResolveFirstTokenWatchdogCPUOffloadExperts(t *testing.T) {
	tests := []struct {
		name       string
		env        string
		cpu        bool
		offload    bool
		wantWindow time.Duration
		wantSource FirstTokenWatchdogSource
	}{
		{name: "offload on device backend", offload: true, wantWindow: CPUBackendFirstTokenWatchdogTimeout, wantSource: FirstTokenWatchdogSourceCPUOffloadExperts},
		{name: "plain device backend", wantWindow: 60 * time.Second, wantSource: FirstTokenWatchdogSourceDefault},
		{name: "cpu backend wins source over offload", cpu: true, offload: true, wantWindow: CPUBackendFirstTokenWatchdogTimeout, wantSource: FirstTokenWatchdogSourceCPUBackend},
		{name: "env wins over offload default", env: "120", offload: true, wantWindow: 120 * time.Second, wantSource: FirstTokenWatchdogSourceEnv},
		{name: "out-of-band env ignored on offload", env: "601", offload: true, wantWindow: CPUBackendFirstTokenWatchdogTimeout, wantSource: FirstTokenWatchdogSourceCPUOffloadExperts},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotWindow, gotSource := resolveFirstTokenWatchdogHostBound(tt.env, tt.cpu, tt.offload)
			if gotWindow != tt.wantWindow || gotSource != tt.wantSource {
				t.Fatalf("resolveFirstTokenWatchdogHostBound(%q, cpu=%v, offload=%v) = (%s, %q), want (%s, %q)", tt.env, tt.cpu, tt.offload, gotWindow, gotSource, tt.wantWindow, tt.wantSource)
			}
		})
	}
	if FirstTokenWatchdogSourceCPUOffloadExperts == FirstTokenWatchdogSourceCPUBackend {
		t.Fatal("cpu-offload-experts source must be distinct from the cpu-backend source")
	}

	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "")
	if got, src := ResolveFirstTokenWatchdogHostBound(false, true); got != CPUBackendFirstTokenWatchdogTimeout || src != FirstTokenWatchdogSourceCPUOffloadExperts {
		t.Fatalf("exported resolver, unset env, offload: got (%s, %q)", got, src)
	}
}
