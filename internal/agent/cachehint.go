package agent

import (
	"encoding/json"
	"fmt"

	capmatrix "github.com/anthony-chaudhary/fak/internal/capabilitymatrix"
)

// CacheIntentVersion is the provider-neutral cache-hint contract version.
const CacheIntentVersion = "fak-cache-intent/1"

type CacheHorizon string
type CacheResidency string
type CachePreference string
type CacheHintStatus string

const (
	CacheHorizonProviderDefault CacheHorizon = "provider-default"
	CacheHorizonFiveMinutes     CacheHorizon = "5m"
	CacheHorizonOneHour         CacheHorizon = "1h"
	CacheHorizonTwentyFourHours CacheHorizon = "24h"

	CacheResidencyMemory   CacheResidency = "memory"
	CacheResidencyExtended CacheResidency = "extended"

	CachePreferenceAutomatic CachePreference = "automatic"
	CachePreferenceExplicit  CachePreference = "explicit"

	CacheHintSupported         CacheHintStatus = "supported"
	CacheHintDowngraded        CacheHintStatus = "downgraded"
	CacheHintProviderDefaulted CacheHintStatus = "provider-defaulted"
	CacheHintRejected          CacheHintStatus = "rejected"
)

// CacheIntent describes desired provider-owned prompt-cache behavior without
// exposing a provider's request field names. PrivacyCeiling is mandatory: a
// provider route may use storage no less restrictive than this value.
type CacheIntent struct {
	Version        string          `json:"version"`
	Enabled        bool            `json:"enabled"`
	Horizon        CacheHorizon    `json:"horizon,omitempty"`
	AffinityID     string          `json:"affinity_id,omitempty"`
	PrivacyCeiling CacheResidency  `json:"privacy_ceiling,omitempty"`
	Preference     CachePreference `json:"preference,omitempty"`
	Advisory       bool            `json:"advisory,omitempty"`
}

// CacheHintResult is the request-side portion of the route/usage receipt.
type CacheHintResult struct {
	Requested          *CacheIntent    `json:"requested,omitempty"`
	Status             CacheHintStatus `json:"status,omitempty"`
	Provider           Provider        `json:"provider,omitempty"`
	Model              string          `json:"model,omitempty"`
	API                string          `json:"api,omitempty"`
	Emitted            map[string]any  `json:"emitted,omitempty"`
	EffectiveHorizon   CacheHorizon    `json:"effective_horizon,omitempty"`
	EffectiveResidency CacheResidency  `json:"effective_residency,omitempty"`
	Reason             string          `json:"reason,omitempty"`
}

func negotiateCacheIntent(provider Provider, model string, in *CacheIntent) (CacheHintResult, error) {
	if in == nil {
		return CacheHintResult{}, nil
	}
	r := CacheHintResult{Requested: in, Provider: provider, Model: model, API: string(provider), Emitted: map[string]any{}}
	reject := func(reason string) (CacheHintResult, error) {
		r.Status, r.Reason = CacheHintRejected, reason
		return r, fmt.Errorf("cache hint rejected: %s", reason)
	}
	if in.Version != CacheIntentVersion {
		return reject("unsupported contract version")
	}
	if !in.Enabled {
		r.Status = CacheHintSupported
		return r, nil
	}
	if in.PrivacyCeiling == "" {
		return reject("privacy ceiling is required")
	}
	if in.PrivacyCeiling != CacheResidencyMemory && in.PrivacyCeiling != CacheResidencyExtended {
		return reject("unknown privacy ceiling")
	}
	advisory := func(reason string) (CacheHintResult, error) {
		if !in.Advisory {
			return reject(reason)
		}
		r.Status, r.Reason = CacheHintDowngraded, reason
		return r, nil
	}
	switch provider {
	case ProviderOpenAIResponses, ProviderOpenAI:
		// The 24h route may persist beyond in-memory residency and is therefore
		// incompatible with a memory-only ceiling.
		h := in.Horizon
		if h == "" || h == CacheHorizonProviderDefault {
			r.Status, r.EffectiveHorizon, r.EffectiveResidency = CacheHintProviderDefaulted, CacheHorizonProviderDefault, CacheResidencyMemory
		} else if h == CacheHorizonTwentyFourHours {
			if in.PrivacyCeiling == CacheResidencyMemory {
				return reject("24h retention exceeds memory-only privacy ceiling")
			}
			if !openAISupportsExtendedRetention(model) {
				return advisory("model/API does not support 24h prompt-cache retention")
			}
			r.Status, r.EffectiveHorizon, r.EffectiveResidency = CacheHintSupported, h, CacheResidencyExtended
			r.Emitted["retention"] = "24h"
		} else {
			return advisory("requested horizon is unsupported by OpenAI")
		}
		if in.AffinityID != "" {
			r.Emitted["affinity"] = in.AffinityID
		}
		if in.Preference == CachePreferenceExplicit {
			return advisory("OpenAI manages cache breakpoints automatically")
		}
	case ProviderAnthropic:
		if in.PrivacyCeiling != CacheResidencyMemory { /* memory is stricter and honors extended ceilings */
		}
		r.EffectiveResidency = CacheResidencyMemory
		if in.Preference == CachePreferenceAutomatic {
			r.Status, r.EffectiveHorizon = CacheHintProviderDefaulted, CacheHorizonProviderDefault
			if in.Horizon != "" && in.Horizon != CacheHorizonProviderDefault {
				return advisory("automatic Anthropic caching cannot guarantee a requested TTL")
			}
			return r, nil
		}
		h := in.Horizon
		if h == "" || h == CacheHorizonProviderDefault {
			h = CacheHorizonFiveMinutes
		}
		if h != CacheHorizonFiveMinutes && h != CacheHorizonOneHour {
			return advisory("requested horizon is unsupported by Anthropic")
		}
		if !anthropicSupportsTTL(model) {
			return advisory("model/API does not support explicit Anthropic TTL controls")
		}
		r.Status, r.EffectiveHorizon = CacheHintSupported, h
		r.Emitted["ttl"] = string(h)
		if in.AffinityID != "" {
			r.Reason = "Anthropic has no stable routing-key field; affinity remains local"
		}
	default:
		return advisory("provider has no explicit cache-hint contract")
	}
	return r, nil
}

func openAISupportsExtendedRetention(model string) bool {
	return capmatrix.Lookup(model).ExtendedRetention
}

func anthropicSupportsTTL(model string) bool {
	return capmatrix.Lookup(model).AnthropicTTL
}

func applyCacheHintToJSON(body []byte, result CacheHintResult) ([]byte, error) {
	if result.Requested == nil || !result.Requested.Enabled || result.Status == CacheHintRejected {
		return body, nil
	}
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, err
	}
	switch result.Provider {
	case ProviderOpenAI, ProviderOpenAIResponses:
		if v, ok := result.Emitted["affinity"]; ok {
			obj["prompt_cache_key"] = v
		}
		if v, ok := result.Emitted["retention"]; ok {
			obj["prompt_cache_retention"] = v
		}
	case ProviderAnthropic:
		ttl, ok := result.Emitted["ttl"].(string)
		if !ok {
			break
		}
		if err := applyAnthropicTTL(obj, ttl); err != nil {
			return nil, err
		}
	case ProviderGemini, ProviderXAI:
		// Neither provider advertises an explicit cache-hint contract;
		// negotiation records the advisory downgrade and the body stays intact.
	}
	return json.Marshal(obj)
}

func applyAnthropicTTL(obj map[string]any, ttl string) error {
	count := 0
	var walk func(any) error
	walk = func(v any) error {
		switch x := v.(type) {
		case []any:
			for _, e := range x {
				if err := walk(e); err != nil {
					return err
				}
			}
		case map[string]any:
			if cc, ok := x["cache_control"].(map[string]any); ok {
				count++
				if count > 4 {
					return fmt.Errorf("anthropic cache breakpoint limit exceeded: %d", count)
				}
				cc["ttl"] = ttl
			}
			for k, e := range x {
				if k != "cache_control" {
					if err := walk(e); err != nil {
						return err
					}
				}
			}
		}
		return nil
	}
	if err := walk(obj); err != nil {
		return err
	}
	return validateAnthropicTTLOrder(obj)
}

func validateAnthropicTTLOrder(obj map[string]any) error {
	rank := map[string]int{"1h": 2, "5m": 1}
	last := 3
	count := 0
	var walk func(any) error
	walk = func(v any) error {
		switch x := v.(type) {
		case []any:
			for _, e := range x {
				if err := walk(e); err != nil {
					return err
				}
			}
		case map[string]any:
			if cc, ok := x["cache_control"].(map[string]any); ok {
				count++
				if count > 4 {
					return fmt.Errorf("anthropic cache breakpoint limit exceeded: %d", count)
				}
				t, _ := cc["ttl"].(string)
				if t == "" {
					t = "5m"
				}
				if rank[t] == 0 {
					return fmt.Errorf("unsupported anthropic cache ttl %q", t)
				}
				if rank[t] > last {
					return fmt.Errorf("anthropic mixed TTL order requires 1h before 5m")
				}
				last = rank[t]
			}
			for k, e := range x {
				if k != "cache_control" {
					if err := walk(e); err != nil {
						return err
					}
				}
			}
		}
		return nil
	}
	return walk(obj)
}
