package main

import (
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// Late startup observations share one read-modify-publish boundary so concurrent
// feature callbacks cannot overwrite another feature's update.
var serveFeatureUpdateMu sync.Mutex

func updateServeFeature(srv *gateway.Server, feature gateway.ServeFeature, state gateway.FeatureState, description string) {
	if srv == nil {
		return
	}
	serveFeatureUpdateMu.Lock()
	defer serveFeatureUpdateMu.Unlock()
	catalog := srv.FeatureSnapshot()
	for i := range catalog.Features {
		row := &catalog.Features[i]
		if row.Feature != feature || (row.State == state && row.Description == description) {
			continue
		}
		row.State, row.Description = state, description
		// The row and schema were validated at initial publication; these
		// callbacks supply only fixed states and scrubbed descriptions.
		_ = srv.SetFeatureCatalog(catalog)
		return
	}
}

// evaluateServeFeatures performs no I/O. A nil live runtime describes pre-boot
// intent; active means installed for eligible requests, not exercised by a turn.
func evaluateServeFeatures(sf *serveFlags, manifest deploymanifest.Manifest, explicit map[string]bool, getenv func(string) string, live *serveRuntime) (gateway.FeatureCatalog, error) {
	if sf == nil {
		return gateway.FeatureCatalog{}, fmt.Errorf("serve feature evaluation requires flags")
	}
	if getenv == nil {
		getenv = func(string) string { return "" }
	}
	ready := live != nil && live.srv != nil
	b := func(p *bool) bool { return p != nil && *p }
	str := func(p *string) string {
		if p == nil {
			return ""
		}
		return strings.TrimSpace(*p)
	}
	positive := func(p *int) bool { return p != nil && *p > 0 }
	envOn := func(key string) bool {
		switch strings.ToLower(strings.TrimSpace(getenv(key))) {
		case "1", "true", "yes", "on":
			return true
		}
		return false
	}
	source := func(flagName, section, key string) gateway.FeatureProvenance {
		if explicit[flagName] {
			return gateway.FeatureCLIFlag
		}
		if section != "" && manifest.Present(section, key) {
			return gateway.FeatureManifest
		}
		return gateway.FeatureAutoDefault
	}
	rows := make([]gateway.FeatureStatus, 0, 42)
	add := func(id gateway.ServeFeature, enabled, installed bool, src gateway.FeatureProvenance, description string) {
		state := gateway.FeatureDisabled
		if enabled {
			state = gateway.FeatureConfiguredStandby
			if ready && installed {
				state = gateway.FeatureConfiguredActive
			}
		}
		rows = append(rows, gateway.FeatureStatus{Feature: id, State: state, Provenance: src, Description: description})
	}
	simple := func(id gateway.ServeFeature, flagName string, enabled, installed bool, description string) {
		add(id, enabled, installed, source(flagName, "", ""), description)
	}
	orEnv := func(id gateway.ServeFeature, flagName, envName string, configured, installed bool, description string) {
		src := source(flagName, "", "")
		enabled := configured
		if !configured && envOn(envName) {
			enabled = true
			src = gateway.FeatureEnvVar
		}
		add(id, enabled, installed, src, description)
	}

	// These gateway fields are installed synchronously by buildGateway. A
	// passthrough-specific transform remains standby on an owned native loop.
	passthrough := !b(sf.native) && (str(sf.baseURL) != "" || len(sf.replicaBaseURLs) > 0)
	anthropic := passthrough && str(sf.provider) == "anthropic"
	nativeModel := ready && live.inKernelModel != nil && live.inKernelTok != nil && str(sf.baseURL) == "" && len(sf.replicaBaseURLs) == 0
	device := ready && live.chatBackend != nil
	metal := ready && live.useMetal
	simple(gateway.FeatureVDSO, "vdso", b(sf.vdso), true, "Kernel vDSO deduplication for eligible tool calls.")
	simple(gateway.FeatureCompactHistory, "compact-history-budget", positive(sf.compactHistoryBudget), anthropic, "History compaction on eligible Anthropic passthrough turns.")
	simple(gateway.FeatureCompactAnchorHead, "compact-anchor-head", b(sf.compactAnchorHead), anthropic && positive(sf.compactHistoryBudget), "Stable head anchoring requires enabled history compaction.")
	orEnv(gateway.FeatureElideResults, "elide-result-bytes", "FAK_ABLATE_UNCACHED_TRIM", positive(sf.elideResultBytes), !b(sf.native), "Oversized result elision installed on eligible decoded and passthrough turns.")
	simple(gateway.FeatureContextView, "ctx-view-budget", positive(sf.ctxViewBudget), true, "Context view planner installed for eligible buffered turns.")
	simple(gateway.FeaturePositiveResidualSubstitution, "positive-residual-substitution", b(sf.positiveResidualSubstitution), anthropic && positive(sf.compactHistoryBudget), "Positive residual substitution requires history compaction.")
	orEnv(gateway.FeatureVCacheAnchor, "vcache-anchor", "FAK_ABLATE_BP_PLAN", b(sf.vcacheAnchor), anthropic, "Provider prefix anchoring requires Anthropic passthrough.")
	orEnv(gateway.FeatureDeferColdTools, "defer-cold-tools", "FAK_DEFER_COLD_TOOLS", b(sf.deferColdTools), anthropic, "Cold tool deferral requires eligible Anthropic passthrough.")
	simple(gateway.FeatureDeferTools, "defer-tools", b(sf.deferTools), true, "MCP tool discovery defers the cold tail.")
	simple(gateway.FeatureElideStaleReads, "elide-stale-reads", b(sf.elideStaleReads), !b(sf.native), "Restorable stale-read elision installed on eligible decoded and passthrough turns.")
	floorSource := source("policy", "policy", "floor")
	if explicit["profile"] {
		floorSource = gateway.FeatureCLIFlag
	} else if str(sf.policyPath) == "" && str(sf.profile) == "" && strings.TrimSpace(getenv("FAK_PROFILE")) != "" {
		floorSource = gateway.FeatureEnvVar
	}
	add(gateway.FeaturePolicyFloor, true, true, floorSource, "Configured capability floor, with the selected permission profile or built-in default.")
	auth := str(sf.requireKeyEnv) != "" || len(sf.keyPrincipal) > 0
	authSource := source("require-key-env", "auth", "require_key_env")
	if explicit["key-principal"] {
		authSource = gateway.FeatureCLIFlag
	}
	add(gateway.FeatureInboundAuth, auth, auth, authSource, "Inbound credentials resolved at boot; credential values are never reported.")
	simple(gateway.FeatureToolExposure, "expose", len(sf.expose) > 0, true, "Validated tool exposure allowlist.")
	simple(gateway.FeatureRouteManifest, "route-manifest", str(sf.routeManifest) != "", true, "Validated model-routing manifest installed at boot.")
	simple(gateway.FeatureRouteAccounts, "route-accounts", str(sf.routeAccounts) != "", str(sf.routeManifest) != "", "Account roster requires a routing manifest for dispatch.")
	engineCache := str(sf.engineCacheEngine) != ""
	simple(gateway.FeatureEngineCache, "engine-cache-engine", engineCache, true, "Remote engine cache client installed; remote connectivity and request success are not implied.")
	simple(gateway.FeatureRemoteKV, "remote-kv-mode", str(sf.remoteKVMode) != "disabled" && str(sf.remoteKVBackend) != "none", false, "Remote KV availability requires a retained connectivity and attachment witness.")
	// The existing resolver gives non-nil flag pointers priority even when empty.
	// Do not invent environment fallback that this boot path does not implement.
	if explicit["remote-kv-url"] || explicit["remote-kv-backend"] {
		rows[len(rows)-1].Provenance = gateway.FeatureCLIFlag
	}
	backendSource := source("backend", "", "")
	backendName := strings.ToLower(str(sf.backendName))
	if backendName == "" && strings.TrimSpace(getenv("FAK_BACKEND")) != "" {
		backendName = strings.ToLower(strings.TrimSpace(getenv("FAK_BACKEND")))
		backendSource = gateway.FeatureEnvVar
	}
	if device && backendSource == gateway.FeatureAutoDefault {
		backendSource = gateway.FeatureHardwareProbed
	}
	add(gateway.FeatureDeviceBackend, backendName != "cpu", device && nativeModel, backendSource, "Automatic or selected device backend; execution requires an initialized backend and in-kernel model.")
	cudaGraphSource := source("cuda-graph", "", "")
	cudaGraphRequested := b(sf.cudaGraph)
	// The compute runtime accepts exactly "1"; generic boolean spellings would
	// report a request that its initialization gate never consumes.
	if !cudaGraphRequested && getenv("FAK_CUDA_GRAPH") == "1" {
		cudaGraphRequested = true
		cudaGraphSource = gateway.FeatureEnvVar
	}
	add(gateway.FeatureCUDAGraph, cudaGraphRequested, false, cudaGraphSource, "CUDA graph capture is request-dependent; no capture witness is retained here.")
	metalSource := source("metal", "", "")
	metalRequested := b(sf.metal)
	if !metalRequested && getenv("FAK_METAL") != "" {
		metalRequested = true
		metalSource = gateway.FeatureEnvVar
	}
	if metal && metalSource == gateway.FeatureAutoDefault {
		metalSource = gateway.FeatureHardwareProbed
	}
	add(gateway.FeatureMetal, metalRequested || metal, metal && nativeModel, metalSource, "Metal execution requires initialized Metal and an in-kernel model.")
	simple(gateway.FeatureNativeModel, "gguf", str(sf.ggufPath) != "" || nativeModel, nativeModel, "Loaded in-kernel model and tokenizer with local chat routing.")
	if explicit["model"] && str(sf.ggufPath) != "" && !explicit["gguf"] {
		rows[len(rows)-1].Provenance = gateway.FeatureCLIFlag
	}
	simple(gateway.FeatureNativeHarness, "native", b(sf.native), true, "Owned native agent loop configured for eligible messages.")
	simple(gateway.FeatureNativeSpeculate, "native-speculate", b(sf.nativeSpeculate), b(sf.native) && b(sf.nativeCodeTools) && str(sf.nativeCodeWorkspace) != "", "Effect-free speculation requires the owned loop and installed coding tools.")
	governorSource := source("memory-governor", "", "")
	governorEnabled := b(sf.memoryGovernor) || metalRequested || metal
	if !governorEnabled && getenv("FAK_MEMORY_GOVERNOR") == "1" {
		governorEnabled = true
		governorSource = gateway.FeatureEnvVar
	}
	if !b(sf.memoryGovernor) && (metalRequested || metal) {
		governorSource = metalSource
	}
	governorInstalled := ready && live.srv.MemoryGovernor() != nil
	add(gateway.FeatureMemoryGovernor, governorEnabled, governorInstalled, governorSource, "Memory admission governor attached at boot; hardware execution is not implied.")

	add(gateway.FeatureApplianceObservability, b(sf.applianceObservability) || (ready && live.strixPreflight.Detected), true, source("appliance-observability", "observability", "appliance_profile"), "Appliance dashboard catalog profile installed at gateway construction; Grafana starts on demand.")
	if ready && live.strixPreflight.Detected && !b(sf.applianceObservability) {
		rows[len(rows)-1].Provenance = gateway.FeatureHardwareProbed
	}
	mockPlanner := str(sf.ggufPath) == "" && str(sf.baseURL) == "" && len(sf.replicaBaseURLs) == 0 && !nativeModel
	simple(gateway.FeatureMockPlanner, "mock", b(sf.mock) || mockPlanner, mockPlanner, "Deterministic mock planner when no model or upstream is configured.")
	simple(gateway.FeatureEngineCacheExactSpan, "engine-cache-require-exact-span", b(sf.engineCacheRequireExactSpan), engineCache, "Exact cache eviction requirement installed on the client; engine qualification occurs on use.")
	add(gateway.FeatureAllowLAN, b(sf.allowLAN), auth, source("allow-lan", "auth", "allow_lan"), "Private-network authentication exemption when inbound authentication is configured.")
	simple(gateway.FeatureBudgetReset, "reset-on-budget", b(sf.resetOnBudget), positive(sf.contextBudgetTokens), "Trace continuation reset requires a positive context budget.")
	simple(gateway.FeatureCPUOffloadExperts, "cpu-offload-experts", b(sf.cpuOffloadExperts), false, "Expert CPU offload requires model/backend qualification; no placement witness is retained.")
	simple(gateway.FeatureGPUDirectOverflow, "gpudirect-overflow", b(sf.gpudirectOverflow), false, "Flag is registered but no serve overflow implementation consumes it.")
	simple(gateway.FeatureNotifyNative, "notify-native", b(sf.notifyNative), true, "Native session-boundary notifications are wired during gateway setup.")
	simple(gateway.FeatureDebugStats, "debug-stats", b(sf.debugStats), true, "Per-turn debug sink installed at gateway construction.")
	simple(gateway.FeatureDojo, "dojo", b(sf.dojoMode), false, "Episode marker is attempted later during route-watcher startup; scoring is not wired.")
	simple(gateway.FeatureNativeCodeTools, "native-code-tools", b(sf.nativeCodeTools), b(sf.native) && str(sf.nativeCodeWorkspace) != "", "Bounded coding tools require an owned loop and resolved workspace.")
	simple(gateway.FeatureVDSOProxyFill, "vdso-proxy-fill", b(sf.vdsoProxyFill), b(sf.vdso) && passthrough, "Admitted proxy tool results may warm vDSO.")
	simple(gateway.FeatureFleetBus, "fleet-bus", b(sf.fleetBus), false, "Fleet registration is attempted after catalog publication; no joined receipt is retained.")
	simple(gateway.FeatureNativeMetalGDNSequence, "native-qwen35-metal-gdn-sequence", b(sf.nativeQwen35MetalGDNSequence), false, "Experimental GDN sequence path requires model and Metal qualification.")
	simple(gateway.FeatureNativeQ4KGateUpSlab, "native-q4k-gateup-slab", b(sf.nativeQ4KGateUpOutputSlab), nativeModel && live.inKernelQ4K, "Gate/up slab requires a loaded native Q4_K model.")
	simple(gateway.FeatureVulkanQ4KProfile, "vulkan-q4k-profile", b(sf.vulkanQ4KProfile), device && live.chatBackend.Name() == "vulkan", "Vulkan timing instrumentation configured on an initialized backend.")
	simple(gateway.FeatureVulkanStageQ4K, "vulkan-stage-q4k", b(sf.vulkanStageQ4K), false, "Vulkan Q4_K staging requires an eligible model; placement is not witnessed here.")
	simple(gateway.FeatureKeepAwake, "keep-awake", str(sf.keepAwake) != "" && str(sf.keepAwake) != KeepAwakeOff, false, "OS keep-awake acquisition is best effort; success is not retained in this catalog.")

	// A known incompatible backend is unavailable, while missing initialization
	// remains standby. Never infer hardware success from an option name.
	if ready {
		for i := range rows {
			if rows[i].State != gateway.FeatureConfiguredStandby {
				continue
			}
			incompatible := (rows[i].Feature == gateway.FeatureCUDAGraph && (!device || live.chatBackend.Name() != "cuda")) ||
				((rows[i].Feature == gateway.FeatureVulkanStageQ4K || rows[i].Feature == gateway.FeatureVulkanQ4KProfile) && (!device || live.chatBackend.Name() != "vulkan")) ||
				(rows[i].Feature == gateway.FeatureMetal && !metal)
			if incompatible {
				rows[i].State = gateway.FeatureRefusedUnavailable
			}
		}
	}
	return gateway.NewFeatureCatalog(rows)
}
