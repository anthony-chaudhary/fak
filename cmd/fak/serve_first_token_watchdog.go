package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// serveChatDecodesOnCPU reports whether the served chat is the in-kernel planner decoding on
// the CPU backend: a pure in-kernel chat (model + tokenizer loaded, no proxy upstream) with
// no compute-HAL device backend and no Apple-Silicon Metal forward. A nil backend is the
// CPU floor resolveServeBackendSelection returns for an explicit or defaulted "cpu".
func serveChatDecodesOnCPU(pureInKernelChat bool, backend compute.Backend, useMetal bool) bool {
	return pureInKernelChat && backend == nil && !useMetal
}

// servesPureInKernelChat reports whether every chat turn decodes in-kernel: a model and
// tokenizer are resident and no --base-url / --replica-base-url proxies turns upstream.
func (rt *serveRuntime) servesPureInKernelChat(sf *serveFlags) bool {
	if rt == nil || rt.inKernelModel == nil || rt.inKernelTok == nil || sf == nil {
		return false
	}
	if sf.baseURL != nil && strings.TrimSpace(*sf.baseURL) != "" {
		return false
	}
	return len(sf.replicaBaseURLs.Values()) == 0
}

// resolveServeFirstTokenWatchdog resolves the buffered first-token watchdog window for the
// chat backend this serve resolved and records the effective window and its source as a
// startup message, so an operator can see why a slow first token was (or was not) cut off.
// A CPU-backend in-kernel chat defaults to agent.CPUBackendFirstTokenWatchdogTimeout; an
// explicit FAK_STREAM_STALL_TIMEOUT_S still wins; every other deployment keeps the default.
func (rt *serveRuntime) resolveServeFirstTokenWatchdog(sf *serveFlags) time.Duration {
	cpu := serveChatDecodesOnCPU(rt.servesPureInKernelChat(sf), rt.chatBackend, rt.useMetal)
	window, source := agent.ResolveFirstTokenWatchdog(cpu)
	rt.addStartupMessage(newServeStartupMessage("serve", "first-token-watchdog", "info",
		fmt.Sprintf("buffered first-token watchdog window %s (source: %s)", window, source)))
	return window
}
