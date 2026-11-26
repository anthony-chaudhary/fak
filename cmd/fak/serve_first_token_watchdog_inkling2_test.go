package main

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// A cold prompt filling a long Metal Inkling-2 context prefills for minutes, so
// the 60s idle default would cut every full-context request before its first token.
func TestResolveServeFirstTokenWatchdogInkling2MetalContext(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "")
	empty := ""
	tests := []struct {
		name       string
		rt         *serveRuntime
		wantWindow time.Duration
		wantSource agent.FirstTokenWatchdogSource
	}{
		{name: "metal 32k", rt: &serveRuntime{inkling2Metal: true, inkling2MaxPromptTokens: 32768}, wantWindow: 327 * time.Second, wantSource: agent.FirstTokenWatchdogSourceMetalContext},
		{name: "metal 16k", rt: &serveRuntime{inkling2Metal: true, inkling2MaxPromptTokens: 16384}, wantWindow: 163 * time.Second, wantSource: agent.FirstTokenWatchdogSourceMetalContext},
		{name: "metal 4k keeps default", rt: &serveRuntime{inkling2Metal: true, inkling2MaxPromptTokens: 4096}, wantWindow: 60 * time.Second, wantSource: agent.FirstTokenWatchdogSourceDefault},
		{name: "cuda inkling2 keeps default", rt: &serveRuntime{inkling2MaxPromptTokens: 32768}, wantWindow: 60 * time.Second, wantSource: agent.FirstTokenWatchdogSourceDefault},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.rt.resolveServeFirstTokenWatchdog(&serveFlags{baseURL: &empty}); got != tt.wantWindow {
				t.Fatalf("window = %s, want %s", got, tt.wantWindow)
			}
			msg := tt.rt.startupMessages[0]
			if !strings.Contains(msg.Text, "source: "+string(tt.wantSource)+")") {
				t.Fatalf("startup message = %+v, want source %q", msg, tt.wantSource)
			}
		})
	}

	t.Run("env override wins", func(t *testing.T) {
		t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "90")
		rt := &serveRuntime{inkling2Metal: true, inkling2MaxPromptTokens: 32768}
		if got := rt.resolveServeFirstTokenWatchdog(&serveFlags{baseURL: &empty}); got != 90*time.Second {
			t.Fatalf("window = %s, want the 90s override", got)
		}
	})
}

func TestEnsureServeInkling2WriteTimeout(t *testing.T) {
	t.Setenv("FAK_HTTP_WRITE_TIMEOUT_S", "")
	(&serveRuntime{}).ensureServeInkling2WriteTimeout()
	if got := os.Getenv("FAK_HTTP_WRITE_TIMEOUT_S"); got != "" {
		t.Fatalf("FAK_HTTP_WRITE_TIMEOUT_S = %q, a non-Inkling-2 serve must leave it unset", got)
	}
	rt := &serveRuntime{inkling2Planner: &agent.MockPlanner{}}
	rt.ensureServeInkling2WriteTimeout()
	if got := os.Getenv("FAK_HTTP_WRITE_TIMEOUT_S"); got != "0" {
		t.Fatalf("FAK_HTTP_WRITE_TIMEOUT_S = %q, want 0 (no deadline) for an Inkling-2 serve", got)
	}
	t.Setenv("FAK_HTTP_WRITE_TIMEOUT_S", "45")
	rt.ensureServeInkling2WriteTimeout()
	if got := os.Getenv("FAK_HTTP_WRITE_TIMEOUT_S"); got != "45" {
		t.Fatalf("FAK_HTTP_WRITE_TIMEOUT_S = %q, an operator value must not change", got)
	}
}
