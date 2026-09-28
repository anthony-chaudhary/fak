package agent

import (
	"testing"
	"time"
)

// TestResolveFirstTokenWatchdog pins the backend-aware first-token resolution rule: an
// in-band FAK_STREAM_STALL_TIMEOUT_S always wins, a CPU-backend chat otherwise defaults to the
// band ceiling, every other backend keeps the 60s default, and the band semantics are the
// stream idle deadline's (out-of-band or unparseable values are ignored, never clamped).
func TestResolveFirstTokenWatchdog(t *testing.T) {
	tests := []struct {
		name       string
		env        string
		cpu        bool
		wantWindow time.Duration
		wantSource FirstTokenWatchdogSource
	}{
		{name: "cpu backend default", cpu: true, wantWindow: 600 * time.Second, wantSource: FirstTokenWatchdogSourceCPUBackend},
		{name: "non-cpu backend default", cpu: false, wantWindow: 60 * time.Second, wantSource: FirstTokenWatchdogSourceDefault},
		{name: "env wins over cpu default", env: "90", cpu: true, wantWindow: 90 * time.Second, wantSource: FirstTokenWatchdogSourceEnv},
		{name: "env wins over plain default", env: "300", cpu: false, wantWindow: 300 * time.Second, wantSource: FirstTokenWatchdogSourceEnv},
		{name: "env band floor honored", env: "5", cpu: true, wantWindow: 5 * time.Second, wantSource: FirstTokenWatchdogSourceEnv},
		{name: "env band ceiling honored", env: "600", cpu: false, wantWindow: 600 * time.Second, wantSource: FirstTokenWatchdogSourceEnv},
		{name: "env below band ignored on cpu", env: "4", cpu: true, wantWindow: 600 * time.Second, wantSource: FirstTokenWatchdogSourceCPUBackend},
		{name: "env above band ignored off cpu", env: "601", cpu: false, wantWindow: 60 * time.Second, wantSource: FirstTokenWatchdogSourceDefault},
		{name: "env unparseable ignored on cpu", env: "ten", cpu: true, wantWindow: 600 * time.Second, wantSource: FirstTokenWatchdogSourceCPUBackend},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotWindow, gotSource := resolveFirstTokenWatchdog(tt.env, tt.cpu)
			if gotWindow != tt.wantWindow || gotSource != tt.wantSource {
				t.Fatalf("resolveFirstTokenWatchdog(%q, cpu=%v) = (%s, %q), want (%s, %q)", tt.env, tt.cpu, gotWindow, gotSource, tt.wantWindow, tt.wantSource)
			}
		})
	}
}

// TestResolveFirstTokenWatchdogReadsEnv proves the exported resolver reads the operator knob
// and that the backend-blind FirstTokenWatchdogTimeout and the inter-byte stall deadline are
// unchanged by the CPU default.
func TestResolveFirstTokenWatchdogReadsEnv(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "")
	if got, src := ResolveFirstTokenWatchdog(true); got != CPUBackendFirstTokenWatchdogTimeout || src != FirstTokenWatchdogSourceCPUBackend {
		t.Fatalf("unset env, cpu: got (%s, %q)", got, src)
	}
	if got := FirstTokenWatchdogTimeout(); got != 60*time.Second {
		t.Fatalf("backend-blind FirstTokenWatchdogTimeout = %s, want 60s", got)
	}
	if got := streamStallTimeout(); got != 60*time.Second {
		t.Fatalf("inter-byte streamStallTimeout = %s, want 60s (must not follow the CPU first-token default)", got)
	}

	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "120")
	if got, src := ResolveFirstTokenWatchdog(true); got != 120*time.Second || src != FirstTokenWatchdogSourceEnv {
		t.Fatalf("env=120, cpu: got (%s, %q), want (2m0s, env)", got, src)
	}
	if got := FirstTokenWatchdogTimeout(); got != 120*time.Second {
		t.Fatalf("env=120: FirstTokenWatchdogTimeout = %s, want 2m0s", got)
	}
}

// TestCPUBackendFirstTokenWatchdogIsBandCeiling keeps the CPU default expressible by the
// operator knob: it must equal the FAK_STREAM_STALL_TIMEOUT_S ceiling, never exceed it.
func TestCPUBackendFirstTokenWatchdogIsBandCeiling(t *testing.T) {
	if CPUBackendFirstTokenWatchdogTimeout != time.Duration(streamStallMaxS)*time.Second {
		t.Fatalf("CPUBackendFirstTokenWatchdogTimeout = %s, want the %ds band ceiling", CPUBackendFirstTokenWatchdogTimeout, streamStallMaxS)
	}
}
