package main

import (
	"fmt"
	"os"
	"strconv"
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

// servesProxyUpstream reports whether chat turns are proxied to an upstream engine through
// --base-url or --replica-base-url.
func servesProxyUpstream(sf *serveFlags) bool {
	if sf == nil {
		return false
	}
	if sf.baseURL != nil && strings.TrimSpace(*sf.baseURL) != "" {
		return true
	}
	return len(sf.replicaBaseURLs.Values()) > 0
}

// serveProxyTimeoutFloorS is the upstream HTTP and response-write deadline, in seconds, that a
// proxying serve raises its defaults to. A buffered agent turn is written only after the
// upstream finishes the whole completion, and a local engine can need well over a minute for
// one turn, so the 60 s planner and 90 s write defaults cut legitimate turns off. It is the
// floor `fak guard` already applies to a wrapped session.
const serveProxyTimeoutFloorS = guardTimeoutFloorS

// ensureServeProxyTimeoutFloors raises FAK_PLANNER_TIMEOUT_S and FAK_HTTP_WRITE_TIMEOUT_S to
// serveProxyTimeoutFloorS for a proxying serve before any planner or listener reads them. An
// operator-set value, including an explicit 0, is never changed.
func ensureServeProxyTimeoutFloors(sf *serveFlags) {
	if !servesProxyUpstream(sf) {
		return
	}
	for _, name := range []string{"FAK_PLANNER_TIMEOUT_S", "FAK_HTTP_WRITE_TIMEOUT_S"} {
		if strings.TrimSpace(os.Getenv(name)) == "" {
			_ = os.Setenv(name, strconv.Itoa(serveProxyTimeoutFloorS))
		}
	}
}

// resolveServeFirstTokenWatchdog resolves the buffered first-token watchdog window for the
// chat backend this serve resolved and records the effective window and its source as a
// startup message, so an operator can see why a slow first token was (or was not) cut off.
// A CPU-backend in-kernel chat, a device-backend in-kernel chat under --cpu-offload-experts
// (host-bound prefill) and a proxied upstream default to
// agent.CPUBackendFirstTokenWatchdogTimeout; an explicit FAK_STREAM_STALL_TIMEOUT_S still
// wins; every other deployment keeps the default. A proxied buffered turn produces nothing
// before the upstream completes it, so the stream-idle default would cap the whole turn at
// 60 s, below the planner timeout that already bounds the upstream call.
func (rt *serveRuntime) resolveServeFirstTokenWatchdog(sf *serveFlags) time.Duration {
	inKernel := rt.servesPureInKernelChat(sf)
	cpu := serveChatDecodesOnCPU(inKernel, rt.chatBackend, rt.useMetal)
	offload := inKernel && sf.cpuOffloadExperts != nil && *sf.cpuOffloadExperts
	window, source := agent.ResolveFirstTokenWatchdogHostBound(cpu, offload)
	if source == agent.FirstTokenWatchdogSourceDefault && servesProxyUpstream(sf) {
		window, source = agent.CPUBackendFirstTokenWatchdogTimeout, agent.FirstTokenWatchdogSourceProxyUpstream
	}
	if w := rt.inkling2PrefillWindow(); source == agent.FirstTokenWatchdogSourceDefault && w > window {
		window, source = w, agent.FirstTokenWatchdogSourceMetalContext
	}
	rt.addStartupMessage(newServeStartupMessage("serve", "first-token-watchdog", "info",
		fmt.Sprintf("buffered first-token watchdog window %s (source: %s)", window, source)))
	return window
}

// inkling2MetalPrefillFloorTokPerS is a pessimistic Inkling-2 Metal prefill
// rate: a full-context prompt arrives cold, and under serve the prefill chunk
// can drop to 512 rows, so a 32k prompt needs minutes before its first token.
const inkling2MetalPrefillFloorTokPerS = 100

// inkling2PrefillWindow is how long a cold prompt filling the served Metal
// Inkling-2 context may take to its first token at the floor rate, or zero when
// no Metal Inkling-2 model is served or the default window already covers it.
func (rt *serveRuntime) inkling2PrefillWindow() time.Duration {
	if rt == nil || !rt.inkling2Metal || rt.inkling2MaxPromptTokens <= 0 {
		return 0
	}
	w := time.Duration(rt.inkling2MaxPromptTokens/inkling2MetalPrefillFloorTokPerS) * time.Second
	if w <= inkling2ServeIdleDefault {
		return 0
	}
	return min(w, agent.CPUBackendFirstTokenWatchdogTimeout)
}

// ensureServeInkling2WriteTimeout lifts the HTTP write deadline for an
// Inkling-2 serve when FAK_HTTP_WRITE_TIMEOUT_S is unset. Its planner is a local
// model the gateway does not classify as in-kernel, so it would get the 90s
// proxy deadline, while a buffered turn is written only after its whole prefill
// and decode, which a long context stretches to minutes.
func (rt *serveRuntime) ensureServeInkling2WriteTimeout() {
	if rt == nil || rt.inkling2Planner == nil {
		return
	}
	if strings.TrimSpace(os.Getenv("FAK_HTTP_WRITE_TIMEOUT_S")) == "" {
		_ = os.Setenv("FAK_HTTP_WRITE_TIMEOUT_S", "0")
	}
}

// inkling2ServeIdleDefault mirrors the agent's 60s stream idle default, which
// a short context's prefill already fits.
const inkling2ServeIdleDefault = 60 * time.Second
