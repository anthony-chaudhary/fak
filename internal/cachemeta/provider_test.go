package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestFromProviderCacheIsCostTelemetryNotTrust(t *testing.T) {
	e := FromProviderCache(ProviderCache{
		Provider:       "anthropic",
		ModelID:        "claude-opus",
		CachedTokens:   1200,
		WriteTokens:    300,
		PromptTokens:   1500,
		SerializerID:   "serde-7",
		BreakpointMode: "explicit",
		Retention:      "1h",
	})
	if e.Plane != PlaneProvider || e.Residency.Tier != TierProvider {
		t.Fatalf("provider entry must be provider-plane/provider-residency: %+v", e)
	}
	if e.ID.MediaType != MediaPromptPrefix || e.ID.Unit != UnitTokens {
		t.Fatalf("provider prefix media/unit wrong: %+v", e.ID)
	}
	if e.Metrics.PrefillTokensSaved != 1200 {
		t.Fatalf("cached tokens must fold into PrefillTokensSaved, got %d", e.Metrics.PrefillTokensSaved)
	}
	if e.Security.AdmissionVerdict != AdmissionDefer {
		t.Fatalf("provider telemetry must defer admission, got %q", e.Security.AdmissionVerdict)
	}
	if e.Validity.TTLMillis != 60*60*1000 {
		t.Fatalf("1h retention should map to 3600000ms, got %d", e.Validity.TTLMillis)
	}
	// Refusal rule 6: a provider cache hit is cost/latency evidence only.
	v := ProviderCacheVerdict(e)
	if v.CanServe() {
		t.Fatalf("provider cache verdict must NOT be serveable as a local hit: %+v", v)
	}
	if v.Kind != LookupTransform || v.Meta["provider_cache"] != "cost_latency_only" {
		t.Fatalf("provider verdict should be a cost-only transform: %+v", v)
	}
}

func TestProviderCacheEndpointAndReasoningModeAreVaryAxes(t *testing.T) {
	// GLM-5.2 (Z.AI) §A2: the Coding-Plan vs general endpoint and the
	// reasoning/thinking mode are silent cache-breakers. Two telemetry events that
	// differ ONLY by endpoint or reasoning mode must NOT collapse to one entry
	// identity, or a metrics sink would blend two request shapes into one hit rate.
	base := ProviderCache{
		Provider:     "openai",
		ModelID:      "glm-5.2",
		CachedTokens: 100,
		PromptTokens: 200,
		SerializerID: "serde-1",
	}
	general := FromProviderCache(base) // no endpoint/reasoning axes contributed
	coding := base
	coding.Endpoint = "coding"
	coding.ReasoningMode = "max"
	codingEntry := FromProviderCache(coding)

	if general.ID.Digest == codingEntry.ID.Digest {
		t.Fatalf("endpoint+reasoning axes must change the entry identity; both = %s", general.ID.Digest)
	}
	if codingEntry.Labels["endpoint"] != "coding" || codingEntry.Labels["reasoning_mode"] != "max" {
		t.Fatalf("endpoint/reasoning labels not surfaced: %+v", codingEntry.Labels)
	}
	// The base (no-axis) entry must NOT emit the optional labels at all.
	if _, ok := general.Labels["endpoint"]; ok {
		t.Fatalf("empty endpoint axis must not emit a label: %+v", general.Labels)
	}
	// Endpoint alone and reasoning alone are each distinct breakers.
	epOnly := base
	epOnly.Endpoint = "general"
	rmOnly := base
	rmOnly.ReasoningMode = "disabled"
	if FromProviderCache(epOnly).ID.Digest == general.ID.Digest {
		t.Fatal("endpoint axis alone must break identity")
	}
	if FromProviderCache(rmOnly).ID.Digest == general.ID.Digest {
		t.Fatal("reasoning-mode axis alone must break identity")
	}
	// Still cost-only telemetry, never a serveable local hit.
	if v := ProviderCacheVerdict(codingEntry); v.CanServe() {
		t.Fatalf("GLM provider entry must not be serveable: %+v", v)
	}
}

func TestProviderCacheFoldedIntoMetricsDoesNotImplyLocalHit(t *testing.T) {
	// A fleet benchmark must be able to tell provider savings apart from local hits:
	// the two live on different planes, so a metrics sink can split them.
	local := FromKVPrefix(KVPrefix{Tokens: []int{1, 2}, ModelID: "m"})
	remote := FromProviderCache(ProviderCache{Provider: "openai", CachedTokens: 900})
	if local.Plane == remote.Plane {
		t.Fatalf("local KV and provider telemetry must be distinct planes: %s", local.Plane)
	}
	if remote.Security.Taint != abi.TaintTrusted || remote.Security.Scope != abi.ScopeFleet {
		t.Fatalf("provider telemetry should be fleet-visible trusted observation: %+v", remote.Security)
	}
}
