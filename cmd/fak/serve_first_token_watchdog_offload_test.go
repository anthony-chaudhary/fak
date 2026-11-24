package main

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// TestResolveServeFirstTokenWatchdogCPUOffloadExperts drives the serve seam buildGateway calls
// for `fak serve --backend cuda --cpu-offload-experts`: the host-streamed experts bound prefill
// (DS4.1 Q2_K on a 16 GB card measured ~0.4 tok/s), so the 60s device default would cut the
// first real request. The seam must resolve the CPU-backend window with the distinct
// cpu-offload-experts source and print it at startup; FAK_STREAM_STALL_TIMEOUT_S still wins.
func TestResolveServeFirstTokenWatchdogCPUOffloadExperts(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "")
	empty := ""
	upstream := "http://127.0.0.1:1"
	on, off := true, false
	newRuntime := func(backend compute.Backend) *serveRuntime {
		return &serveRuntime{inKernelModel: &fakmodel.Model{}, inKernelTok: &tokenizer.Tokenizer{}, chatBackend: backend}
	}
	device := func() compute.Backend { return &serveBackendAutoSentinel{name: "cuda"} }
	tests := []struct {
		name       string
		rt         *serveRuntime
		baseURL    *string
		offload    *bool
		wantWindow time.Duration
		wantSource agent.FirstTokenWatchdogSource
	}{
		{name: "cuda + cpu-offload-experts", rt: newRuntime(device()), baseURL: &empty, offload: &on, wantWindow: agent.CPUBackendFirstTokenWatchdogTimeout, wantSource: agent.FirstTokenWatchdogSourceCPUOffloadExperts},
		{name: "cuda without offload", rt: newRuntime(device()), baseURL: &empty, offload: &off, wantWindow: 60 * time.Second, wantSource: agent.FirstTokenWatchdogSourceDefault},
		{name: "cuda offload flag unset pointer", rt: newRuntime(device()), baseURL: &empty, wantWindow: 60 * time.Second, wantSource: agent.FirstTokenWatchdogSourceDefault},
		{name: "cpu backend with offload keeps cpu source", rt: newRuntime(nil), baseURL: &empty, offload: &on, wantWindow: agent.CPUBackendFirstTokenWatchdogTimeout, wantSource: agent.FirstTokenWatchdogSourceCPUBackend},
		{name: "proxy upstream with offload", rt: newRuntime(device()), baseURL: &upstream, offload: &on, wantWindow: agent.CPUBackendFirstTokenWatchdogTimeout, wantSource: agent.FirstTokenWatchdogSourceProxyUpstream},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sf := &serveFlags{baseURL: tt.baseURL, cpuOffloadExperts: tt.offload}
			if got := tt.rt.resolveServeFirstTokenWatchdog(sf); got != tt.wantWindow {
				t.Fatalf("window = %s, want %s", got, tt.wantWindow)
			}
			if len(tt.rt.startupMessages) != 1 {
				t.Fatalf("startup messages = %+v, want exactly one", tt.rt.startupMessages)
			}
			msg := tt.rt.startupMessages[0]
			if msg.Kind != "first-token-watchdog" || !strings.Contains(msg.Text, tt.wantWindow.String()) || !strings.Contains(msg.Text, "source: "+string(tt.wantSource)+")") {
				t.Fatalf("startup message = %+v, want first-token-watchdog naming %s and source %q", msg, tt.wantWindow, tt.wantSource)
			}
		})
	}

	t.Run("env override wins on cuda offload", func(t *testing.T) {
		t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "90")
		rt := newRuntime(device())
		if got := rt.resolveServeFirstTokenWatchdog(&serveFlags{baseURL: &empty, cpuOffloadExperts: &on}); got != 90*time.Second {
			t.Fatalf("window = %s, want 1m30s from the env override", got)
		}
		if msg := rt.startupMessages[0]; !strings.Contains(msg.Text, "source: env") {
			t.Fatalf("startup message = %+v, want source env", msg)
		}
	})
}
