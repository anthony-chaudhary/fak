package modelroute

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// KnowledgeState prevents an absent provider fact from being mistaken for a zero value.
type KnowledgeState string

const (
	KnowledgeKnown         KnowledgeState = "known"
	KnowledgeUnknown       KnowledgeState = "unknown"
	KnowledgeNotApplicable KnowledgeState = "not_applicable"
)

// FactSource binds one provider fact to the official or witnessed source that established it.
type FactSource struct {
	URL          string `json:"url"`
	Ref          string `json:"ref"`
	Path         string `json:"path,omitempty"`
	Symbol       string `json:"symbol,omitempty"`
	ObservedAt   string `json:"observed_at"`
	CopiedPath   string `json:"copied_path,omitempty"`
	CopiedSHA256 string `json:"copied_sha256,omitempty"`
	License      string `json:"license,omitempty"`
}

// ContractFact carries an explicit knowledge state and provenance for one provider fact.
type ContractFact[T any] struct {
	State  KnowledgeState `json:"state"`
	Value  T              `json:"value,omitempty"`
	Source FactSource     `json:"source"`
}

// ProviderContract is the lifecycle-wide provider truth from which routing profiles are projected.
type ProviderContract struct {
	Provider            string                            `json:"provider"`
	Family              string                            `json:"family"`
	ModelScope          string                            `json:"model_scope"`
	Endpoint            ContractFact[string]              `json:"endpoint"`
	AuthEnv             ContractFact[string]              `json:"auth_env"`
	APIDialect          ContractFact[string]              `json:"api_dialect"`
	ContextTokens       ContractFact[int]                 `json:"context_tokens"`
	MaxOutputTokens     ContractFact[int]                 `json:"max_output_tokens"`
	ToolCalls           ContractFact[bool]                `json:"tool_calls"`
	ParallelToolCalls   ContractFact[bool]                `json:"parallel_tool_calls"`
	StructuredOutput    ContractFact[bool]                `json:"structured_output"`
	StreamingEvents     ContractFact[string]              `json:"streaming_events"`
	PromptCaching       ContractFact[bool]                `json:"prompt_caching"`
	CacheTTLSeconds     ContractFact[int]                 `json:"cache_ttl_seconds"`
	MaxCacheBreakpoints ContractFact[int]                 `json:"max_cache_breakpoints"`
	UsageDetails        ContractFact[string]              `json:"cache_accounting"`
	RetryStatusCodes    ContractFact[string]              `json:"retry_status_codes"`
	RateLimitHeaders    ContractFact[string]              `json:"rate_limit_headers"`
	SessionResume       ContractFact[string]              `json:"session_resume"`
	SupportMaturity     ContractFact[string]              `json:"support_maturity"`
	ServiceTiers        ContractFact[ServiceTierContract] `json:"service_tiers"`
}

var providerContracts = []ProviderContract{
	openAIProviderContract(),
	anthropicProviderContract(),
	openRouterProviderContract(),
}

// DefaultProviderContracts returns a detached, deterministically ordered registry snapshot.
func DefaultProviderContracts() []ProviderContract {
	out := append([]ProviderContract(nil), providerContracts...)
	sort.Slice(out, func(i, j int) bool { return out[i].Provider < out[j].Provider })
	return out
}

// LookupProviderContract resolves a canonical provider name without guessing aliases.
func LookupProviderContract(provider string) (ProviderContract, bool) {
	provider = strings.ToLower(strings.TrimSpace(provider))
	for _, contract := range providerContracts {
		if contract.Provider == provider {
			return contract, true
		}
	}
	return ProviderContract{}, false
}

// ProviderContractsJSON renders the registry for operator inspection and stable snapshots.
func ProviderContractsJSON() ([]byte, error) {
	if err := ValidateProviderContracts(DefaultProviderContracts()); err != nil {
		return nil, err
	}
	return json.MarshalIndent(DefaultProviderContracts(), "", "  ")
}

// ValidateProviderContracts rejects duplicate scopes, implicit unknowns, and unproven facts.
func ValidateProviderContracts(contracts []ProviderContract) error {
	if len(contracts) == 0 {
		return errors.New("provider contract registry is empty")
	}
	seen := make(map[string]struct{}, len(contracts))
	for i, contract := range contracts {
		key := contract.Provider + "\x00" + contract.ModelScope
		if contract.Provider == "" || contract.Family == "" || contract.ModelScope == "" {
			return fmt.Errorf("provider contract %d has incomplete scope", i)
		}
		if _, ok := seen[key]; ok {
			return fmt.Errorf("duplicate provider contract scope %q", key)
		}
		seen[key] = struct{}{}
		facts := []factValidator{
			wrapFact("endpoint", contract.Endpoint), wrapFact("auth_env", contract.AuthEnv), wrapFact("api_dialect", contract.APIDialect),
			wrapFact("context_tokens", contract.ContextTokens), wrapFact("max_output_tokens", contract.MaxOutputTokens),
			wrapFact("tool_calls", contract.ToolCalls), wrapFact("parallel_tool_calls", contract.ParallelToolCalls), wrapFact("structured_output", contract.StructuredOutput),
			wrapFact("streaming_events", contract.StreamingEvents), wrapFact("prompt_caching", contract.PromptCaching), wrapFact("cache_ttl_seconds", contract.CacheTTLSeconds),
			wrapFact("max_cache_breakpoints", contract.MaxCacheBreakpoints), wrapFact("cache_accounting", contract.UsageDetails), wrapFact("retry_status_codes", contract.RetryStatusCodes),
			wrapFact("rate_limit_headers", contract.RateLimitHeaders), wrapFact("session_resume", contract.SessionResume), wrapFact("support_maturity", contract.SupportMaturity),
			wrapServiceTierFact("service_tiers", contract.ServiceTiers),
		}
		for _, fact := range facts {
			if err := fact.validate(); err != nil {
				return fmt.Errorf("provider %s field %s: %w", contract.Provider, fact.name, err)
			}
		}
	}
	return nil
}

type factValidator struct {
	name     string
	state    KnowledgeState
	source   FactSource
	hasValue bool
}

func wrapFact[T comparable](name string, fact ContractFact[T]) factValidator {
	var zero T
	return factValidator{name: name, state: fact.State, source: fact.Source, hasValue: fact.Value != zero}
}

func wrapServiceTierFact(name string, fact ContractFact[ServiceTierContract]) factValidator {
	return factValidator{name: name, state: fact.State, source: fact.Source, hasValue: len(fact.Value.Supported) != 0 || fact.Value.RequestField != "" || fact.Value.RealizedField != ""}
}

func (f factValidator) validate() error {
	if f.state != KnowledgeKnown && f.state != KnowledgeUnknown && f.state != KnowledgeNotApplicable {
		return fmt.Errorf("invalid knowledge state %q", f.state)
	}
	if f.source.URL == "" || f.source.Ref == "" || f.source.ObservedAt == "" {
		return errors.New("fact provenance requires url, ref, and observed_at")
	}
	if f.state != KnowledgeKnown && f.hasValue {
		return errors.New("unknown or not-applicable fact carries an optimistic value")
	}
	return nil
}

func known[T any](value T, source FactSource) ContractFact[T] {
	return ContractFact[T]{State: KnowledgeKnown, Value: value, Source: source}
}

func unknown[T any](source FactSource) ContractFact[T] {
	return ContractFact[T]{State: KnowledgeUnknown, Source: source}
}

func notApplicable[T any](source FactSource) ContractFact[T] {
	return ContractFact[T]{State: KnowledgeNotApplicable, Source: source}
}

func openAIProviderContract() ProviderContract {
	sdk := FactSource{URL: "https://github.com/openai/openai-go", Ref: "6dfcd9b01bc201830df3aff3820a1664ea05e21b", Path: "responses/response.go", ObservedAt: "2026-08-21"}
	retry := FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "internal/requestconfig/requestconfig.go", Symbol: "shouldRetry", ObservedAt: sdk.ObservedAt, CopiedPath: "testdata/provider_contracts/upstream/openai_should_retry.go.txt", CopiedSHA256: "19d8259d9f2e08ada9df5708305af0f8e747904c0eb6f00f617a84c8005311b7", License: "Apache-2.0"}
	audit := FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "README.md", Symbol: "model-specific limits not encoded by SDK", ObservedAt: sdk.ObservedAt}
	return ProviderContract{
		Provider: "openai", Family: "openai", ModelScope: "*",
		Endpoint:   known("https://api.openai.com/v1", FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "option/requestoption.go", Symbol: "WithBaseURL", ObservedAt: sdk.ObservedAt}),
		AuthEnv:    known("OPENAI_API_KEY", FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "client.go", Symbol: "NewClient", ObservedAt: sdk.ObservedAt}),
		APIDialect: known("openai-responses", sdk), ContextTokens: unknown[int](audit), MaxOutputTokens: unknown[int](audit),
		ToolCalls: known(true, sdk), ParallelToolCalls: known(true, sdk), StructuredOutput: known(true, sdk), StreamingEvents: known("responses semantic events", sdk),
		PromptCaching:   known(true, FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: sdk.Path, Symbol: "ResponseUsageInputTokensDetails", ObservedAt: sdk.ObservedAt}),
		CacheTTLSeconds: notApplicable[int](sdk), MaxCacheBreakpoints: notApplicable[int](sdk), UsageDetails: known("input_tokens_details.cache_write_tokens+cached_tokens", sdk),
		RetryStatusCodes: known("408,409,429,5xx; x-should-retry overrides", retry), RateLimitHeaders: unknown[string](retry), SessionResume: known("previous_response_id", sdk),
		SupportMaturity: known("production", audit),
		ServiceTiers:    known(ServiceTierContract{Supported: []ServiceTierBinding{{Mode: ServiceModeStandard, WireValue: "default"}, {Mode: ServiceModeFast, WireValue: "priority"}}, RequestField: "service_tier", RealizedField: "service_tier", CacheOnSwitch: "preserved", PremiumPrice: KnowledgeUnknown, Fallback: "provider_may_realize_default"}, FactSource{URL: "https://platform.openai.com/docs/api-reference/responses/create#responses-create-service_tier", Ref: "observed-2026-08-25", Path: "service_tier", ObservedAt: "2026-08-25"}),
	}
}

func anthropicProviderContract() ProviderContract {
	sdk := FactSource{URL: "https://github.com/anthropics/anthropic-sdk-go", Ref: "da00f432230f3cdc1dff4e0c1e201d8e95558449", Path: "message.go", ObservedAt: "2026-08-21"}
	retry := FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "internal/requestconfig/requestconfig.go", Symbol: "shouldRetry", ObservedAt: sdk.ObservedAt, CopiedPath: "testdata/provider_contracts/upstream/anthropic_should_retry.go.txt", CopiedSHA256: "c2628df609bbcf284028519341ea4aa1b9725ba1f034402383a1b0ac64a4dedf", License: "MIT"}
	docs := FactSource{URL: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching", Ref: "observed-2026-08-21", Path: "cache duration and breakpoints", ObservedAt: "2026-08-21"}
	audit := FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "README.md", Symbol: "model-specific limits not encoded by SDK", ObservedAt: sdk.ObservedAt}
	return ProviderContract{
		Provider: "anthropic", Family: "anthropic", ModelScope: "*",
		Endpoint:   known("https://api.anthropic.com", FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "client.go", Symbol: "NewClient", ObservedAt: sdk.ObservedAt}),
		AuthEnv:    known("ANTHROPIC_API_KEY", FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "client.go", Symbol: "NewClient", ObservedAt: sdk.ObservedAt}),
		APIDialect: known("anthropic-messages", sdk), ContextTokens: unknown[int](audit), MaxOutputTokens: unknown[int](audit),
		ToolCalls: known(true, sdk), ParallelToolCalls: known(true, sdk), StructuredOutput: unknown[bool](sdk), StreamingEvents: known("message_start/content_block/message_delta/message_stop", sdk),
		PromptCaching:   known(true, FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: sdk.Path, Symbol: "Usage", ObservedAt: sdk.ObservedAt}),
		CacheTTLSeconds: known(300, docs), MaxCacheBreakpoints: known(4, docs), UsageDetails: known("cache_creation_input_tokens+cache_read_input_tokens", sdk),
		RetryStatusCodes: known("408,409,429,5xx; x-should-retry overrides", retry), RateLimitHeaders: unknown[string](retry), SessionResume: notApplicable[string](sdk),
		SupportMaturity: known("production", audit),
		ServiceTiers:    known(ServiceTierContract{Supported: []ServiceTierBinding{{Mode: ServiceModeStandard}, {Mode: ServiceModeFast, WireValue: "auto"}}, RequestField: "service_tier", RealizedField: "usage.service_tier", CacheOnSwitch: "invalidates", PremiumPrice: KnowledgeUnknown, Fallback: "provider_may_realize_standard"}, FactSource{URL: "https://docs.anthropic.com/en/api/service-tiers", Ref: "observed-2026-08-25", Path: "service_tier", ObservedAt: "2026-08-25"}),
	}
}

func openRouterProviderContract() ProviderContract {
	sdk := FactSource{URL: "https://github.com/OpenRouterTeam/ai-sdk-provider", Ref: "b96b20799eadeb72a180ef021b85254fc1500746", Path: "src/index.ts", ObservedAt: "2026-08-18"}
	docs := FactSource{URL: "https://openrouter.ai/docs", Ref: "observed-2026-08-18", Path: "provider selection and routing", ObservedAt: "2026-08-18"}
	catalog := FactSource{URL: "https://openrouter.ai/api/v1/models", Ref: "observed-2026-08-18", Path: "live catalog snapshot", ObservedAt: "2026-08-18"}
	return ProviderContract{
		Provider: "openrouter", Family: "openrouter", ModelScope: "*",
		Endpoint:   known("https://openrouter.ai/api/v1", FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "src/index.ts", ObservedAt: sdk.ObservedAt}),
		AuthEnv:    known("OPENROUTER_API_KEY", FactSource{URL: sdk.URL, Ref: sdk.Ref, Path: "src/index.ts", ObservedAt: sdk.ObservedAt}),
		APIDialect: known("openai-compatible", sdk), ContextTokens: unknown[int](catalog), MaxOutputTokens: unknown[int](catalog),
		ToolCalls: known(true, sdk), ParallelToolCalls: known(true, sdk), StructuredOutput: unknown[bool](sdk), StreamingEvents: known("chat.completion.chunk", sdk),
		PromptCaching:   known(false, FactSource{URL: docs.URL, Ref: docs.Ref, Path: "prompt-caching", ObservedAt: docs.ObservedAt}),
		CacheTTLSeconds: notApplicable[int](sdk), MaxCacheBreakpoints: notApplicable[int](sdk), UsageDetails: notApplicable[string](sdk),
		RetryStatusCodes: known("408,409,429,5xx", sdk), RateLimitHeaders: unknown[string](sdk), SessionResume: notApplicable[string](sdk),
		SupportMaturity: known("production", docs),
		ServiceTiers:    known(ServiceTierContract{Supported: []ServiceTierBinding{{Mode: ServiceModeStandard}, {Mode: ServiceModeFast, WireValue: "priority"}}, RequestField: "provider.sort", RealizedField: "", CacheOnSwitch: "preserved", PremiumPrice: KnowledgeUnknown, Fallback: "provider_may_realize_default"}, FactSource{URL: docs.URL, Ref: docs.Ref, Path: "provider-selection", ObservedAt: docs.ObservedAt}),
	}
}

func providerProfileFromContract(contract ProviderContract) (ProviderProfile, error) {
	for name, state := range map[string]KnowledgeState{"endpoint": contract.Endpoint.State, "auth_env": contract.AuthEnv.State, "prompt_caching": contract.PromptCaching.State} {
		if state != KnowledgeKnown {
			return ProviderProfile{}, fmt.Errorf("provider %s required profile field %s is %s", contract.Provider, name, state)
		}
	}
	profile := ProviderProfile{Provider: contract.Provider, Endpoint: contract.Endpoint.Value, AuthEnv: contract.AuthEnv.Value, SupportsPromptCaching: contract.PromptCaching.Value}
	if contract.CacheTTLSeconds.State == KnowledgeKnown {
		profile.CacheTTLSeconds = contract.CacheTTLSeconds.Value
	}
	if contract.MaxCacheBreakpoints.State == KnowledgeKnown {
		profile.MaxCacheBreakpoints = contract.MaxCacheBreakpoints.Value
	}

	return profile, nil
}
