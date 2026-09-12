package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
)

// ServeFeature identifies a runtime mechanism, independently of request execution.
type ServeFeature string

const (
	FeatureVDSO                         ServeFeature = "vdso"
	FeatureCompactHistory               ServeFeature = "compact_history"
	FeatureCompactAnchorHead            ServeFeature = "compact_anchor_head"
	FeatureElideResults                 ServeFeature = "elide_results"
	FeatureContextView                  ServeFeature = "context_view"
	FeaturePositiveResidualSubstitution ServeFeature = "positive_residual_substitution"
	FeatureVCacheAnchor                 ServeFeature = "vcache_anchor"
	FeatureDeferColdTools               ServeFeature = "defer_cold_tools"
	FeatureDeferTools                   ServeFeature = "defer_tools"
	FeatureElideStaleReads              ServeFeature = "elide_stale_reads"
	FeaturePolicyFloor                  ServeFeature = "policy_floor"
	FeatureInboundAuth                  ServeFeature = "inbound_auth"
	FeatureToolExposure                 ServeFeature = "tool_exposure"
	FeatureRouteManifest                ServeFeature = "route_manifest"
	FeatureRouteAccounts                ServeFeature = "route_accounts"
	FeatureEngineCache                  ServeFeature = "engine_cache"
	FeatureRemoteKV                     ServeFeature = "remote_kv"
	FeatureDeviceBackend                ServeFeature = "device_backend"
	FeatureCUDAGraph                    ServeFeature = "cuda_graph"
	FeatureMetal                        ServeFeature = "metal"
	FeatureNativeModel                  ServeFeature = "native_model"
	FeatureNativeHarness                ServeFeature = "native_harness"
	FeatureNativeSpeculate              ServeFeature = "native_speculate"
	FeatureMemoryGovernor               ServeFeature = "memory_governor"
	FeatureApplianceObservability       ServeFeature = "appliance_observability"
	FeatureMockPlanner                  ServeFeature = "mock_planner"
	FeatureEngineCacheExactSpan         ServeFeature = "engine_cache_exact_span"
	FeatureAllowLAN                     ServeFeature = "allow_lan"
	FeatureBudgetReset                  ServeFeature = "budget_reset"
	FeatureCPUOffloadExperts            ServeFeature = "cpu_offload_experts"
	FeatureGPUDirectOverflow            ServeFeature = "gpudirect_overflow"
	FeatureNotifyNative                 ServeFeature = "notify_native"
	FeatureDebugStats                   ServeFeature = "debug_stats"
	FeatureDojo                         ServeFeature = "dojo"
	FeatureNativeCodeTools              ServeFeature = "native_code_tools"
	FeatureVDSOProxyFill                ServeFeature = "vdso_proxy_fill"
	FeatureFleetBus                     ServeFeature = "fleet_bus"
	FeatureNativeMetalGDNSequence       ServeFeature = "native_metal_gdn_sequence"
	FeatureNativeQ4KGateUpSlab          ServeFeature = "native_q4k_gateup_slab"
	FeatureVulkanQ4KProfile             ServeFeature = "vulkan_q4k_profile"
	FeatureVulkanStageQ4K               ServeFeature = "vulkan_stage_q4k"
	FeatureKeepAwake                    ServeFeature = "keep_awake"
)

// FeatureState describes enablement, not whether a particular request used a feature.
type FeatureState string

const (
	FeatureDisabled           FeatureState = "disabled"
	FeatureConfiguredActive   FeatureState = "configured_active"
	FeatureConfiguredStandby  FeatureState = "configured_standby"
	FeatureRefusedUnavailable FeatureState = "refused_unavailable"
)

// FeatureProvenance names the source of the effective enablement decision.
type FeatureProvenance string

const (
	FeatureCLIFlag        FeatureProvenance = "cli_flag"
	FeatureManifest       FeatureProvenance = "manifest"
	FeatureEnvVar         FeatureProvenance = "env_var"
	FeatureAutoDefault    FeatureProvenance = "auto_default"
	FeatureHardwareProbed FeatureProvenance = "hardware_probed"
	FeatureSchema                           = "fak-serve-features/1"
)

// FeatureStatus contains no configuration values, credentials, or request data.
type FeatureStatus struct {
	Feature     ServeFeature      `json:"feature"`
	State       FeatureState      `json:"state"`
	Provenance  FeatureProvenance `json:"provenance"`
	Description string            `json:"description"`
}

// FeatureCatalog is a point-in-time enablement matrix.
type FeatureCatalog struct {
	Schema   string          `json:"schema"`
	Features []FeatureStatus `json:"features"`
}

func knownServeFeature(feature ServeFeature) bool {
	switch feature {
	case FeatureVDSO, FeatureCompactHistory, FeatureCompactAnchorHead, FeatureElideResults, FeatureContextView, FeaturePositiveResidualSubstitution, FeatureVCacheAnchor, FeatureDeferColdTools, FeatureDeferTools, FeatureElideStaleReads, FeaturePolicyFloor, FeatureInboundAuth, FeatureToolExposure, FeatureRouteManifest, FeatureRouteAccounts, FeatureEngineCache, FeatureRemoteKV, FeatureDeviceBackend, FeatureCUDAGraph, FeatureMetal, FeatureNativeModel, FeatureNativeHarness, FeatureNativeSpeculate, FeatureMemoryGovernor, FeatureApplianceObservability, FeatureMockPlanner, FeatureEngineCacheExactSpan, FeatureAllowLAN, FeatureBudgetReset, FeatureCPUOffloadExperts, FeatureGPUDirectOverflow, FeatureNotifyNative, FeatureDebugStats, FeatureDojo, FeatureNativeCodeTools, FeatureVDSOProxyFill, FeatureFleetBus, FeatureNativeMetalGDNSequence, FeatureNativeQ4KGateUpSlab, FeatureVulkanQ4KProfile, FeatureVulkanStageQ4K, FeatureKeepAwake:
		return true
	}
	return false
}

// NewFeatureCatalog validates and copies evaluated statuses in stable feature order.
// A partial catalog contains only evaluated features; an empty one encodes as [].
func NewFeatureCatalog(statuses []FeatureStatus) (FeatureCatalog, error) {
	result := FeatureCatalog{Schema: FeatureSchema, Features: make([]FeatureStatus, 0, len(statuses))}
	seen := make(map[ServeFeature]bool, len(statuses))
	for _, status := range statuses {
		if !knownServeFeature(status.Feature) {
			return FeatureCatalog{}, fmt.Errorf("unknown serve feature %q", status.Feature)
		}
		if seen[status.Feature] {
			return FeatureCatalog{}, fmt.Errorf("duplicate serve feature %q", status.Feature)
		}
		switch status.State {
		case FeatureDisabled, FeatureConfiguredActive, FeatureConfiguredStandby, FeatureRefusedUnavailable:
		default:
			return FeatureCatalog{}, fmt.Errorf("unknown feature state %q", status.State)
		}
		switch status.Provenance {
		case FeatureCLIFlag, FeatureManifest, FeatureEnvVar, FeatureAutoDefault, FeatureHardwareProbed:
		default:
			return FeatureCatalog{}, fmt.Errorf("unknown feature provenance %q", status.Provenance)
		}
		seen[status.Feature] = true
		result.Features = append(result.Features, status)
	}
	sort.Slice(result.Features, func(i, j int) bool { return result.Features[i].Feature < result.Features[j].Feature })
	return result, nil
}

// SetFeatureCatalog atomically publishes a detached, validated snapshot.
func (s *Server) SetFeatureCatalog(catalog FeatureCatalog) error {
	if catalog.Schema != FeatureSchema {
		return fmt.Errorf("unsupported feature schema %q", catalog.Schema)
	}
	copy, err := NewFeatureCatalog(catalog.Features)
	if err != nil {
		return err
	}
	s.featureCatalog.Store(&copy)
	return nil
}

// FeatureSnapshot returns a detached copy safe for a caller to modify.
func (s *Server) FeatureSnapshot() FeatureCatalog {
	current := s.featureCatalog.Load()
	if current == nil {
		return FeatureCatalog{Schema: FeatureSchema, Features: []FeatureStatus{}}
	}
	return FeatureCatalog{Schema: FeatureSchema, Features: append([]FeatureStatus{}, current.Features...)}
}

func (s *Server) handleFeatures(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(s.FeatureSnapshot())
}
