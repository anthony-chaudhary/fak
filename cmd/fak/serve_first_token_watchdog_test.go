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

func TestServeChatDecodesOnCPU(t *testing.T) {
	device := &serveBackendAutoSentinel{name: "vulkan"}
	tests := []struct {
		name     string
		inKernel bool
		backend  compute.Backend
		metal    bool
		want     bool
	}{
		{name: "in-kernel cpu floor", inKernel: true, want: true},
		{name: "in-kernel device backend", inKernel: true, backend: device, want: false},
		{name: "in-kernel metal forward", inKernel: true, metal: true, want: false},
		{name: "proxy or mock chat", inKernel: false, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := serveChatDecodesOnCPU(tt.inKernel, tt.backend, tt.metal); got != tt.want {
				t.Fatalf("serveChatDecodesOnCPU(%v, %v, %v) = %v, want %v", tt.inKernel, tt.backend, tt.metal, got, tt.want)
			}
		})
	}
}

// TestResolveServeFirstTokenWatchdogPassesCPUBackend drives the serve seam buildGateway calls:
// a pure in-kernel chat on the CPU floor must resolve the CPU-backend default and record it as
// a startup message, while a device backend or a proxy upstream keeps the 60s default.
func TestResolveServeFirstTokenWatchdogPassesCPUBackend(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "")
	empty := ""
	upstream := "http://127.0.0.1:1"
	newRuntime := func(backend compute.Backend) *serveRuntime {
		return &serveRuntime{inKernelModel: &fakmodel.Model{}, inKernelTok: &tokenizer.Tokenizer{}, chatBackend: backend}
	}
	tests := []struct {
		name       string
		rt         *serveRuntime
		baseURL    *string
		wantWindow time.Duration
		wantSource agent.FirstTokenWatchdogSource
	}{
		{name: "cpu backend in-kernel", rt: newRuntime(nil), baseURL: &empty, wantWindow: agent.CPUBackendFirstTokenWatchdogTimeout, wantSource: agent.FirstTokenWatchdogSourceCPUBackend},
		{name: "device backend in-kernel", rt: newRuntime(&serveBackendAutoSentinel{name: "vulkan"}), baseURL: &empty, wantWindow: 60 * time.Second, wantSource: agent.FirstTokenWatchdogSourceDefault},
		{name: "proxy upstream", rt: newRuntime(nil), baseURL: &upstream, wantWindow: 60 * time.Second, wantSource: agent.FirstTokenWatchdogSourceDefault},
		{name: "no model loaded", rt: &serveRuntime{}, baseURL: &empty, wantWindow: 60 * time.Second, wantSource: agent.FirstTokenWatchdogSourceDefault},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sf := &serveFlags{baseURL: tt.baseURL}
			if got := tt.rt.resolveServeFirstTokenWatchdog(sf); got != tt.wantWindow {
				t.Fatalf("window = %s, want %s", got, tt.wantWindow)
			}
			if len(tt.rt.startupMessages) != 1 {
				t.Fatalf("startup messages = %+v, want exactly one", tt.rt.startupMessages)
			}
			msg := tt.rt.startupMessages[0]
			if msg.Kind != "first-token-watchdog" || !strings.Contains(msg.Text, tt.wantWindow.String()) || !strings.Contains(msg.Text, "source: "+string(tt.wantSource)) {
				t.Fatalf("startup message = %+v, want first-token-watchdog naming %s and source %q", msg, tt.wantWindow, tt.wantSource)
			}
		})
	}

	t.Run("env override wins on cpu backend", func(t *testing.T) {
		t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "90")
		rt := newRuntime(nil)
		if got := rt.resolveServeFirstTokenWatchdog(&serveFlags{baseURL: &empty}); got != 90*time.Second {
			t.Fatalf("window = %s, want 1m30s from the env override", got)
		}
		if msg := rt.startupMessages[0]; !strings.Contains(msg.Text, "source: env") {
			t.Fatalf("startup message = %+v, want source env", msg)
		}
	})
}
