package modelroute

import "fmt"

type ServiceMode string

const (
	ServiceModeStandard ServiceMode = "standard"
	ServiceModeFast     ServiceMode = "fast"
	ServiceModeUnknown  ServiceMode = "unknown"
)

type ServiceFallbackPolicy string

const (
	ServiceRequire                ServiceFallbackPolicy = "require"
	ServiceAllowDeclaredDowngrade ServiceFallbackPolicy = "allow_declared_downgrade"
	ServiceStandardOnly           ServiceFallbackPolicy = "standard_only"
)

type ServiceTierBinding struct {
	Mode      ServiceMode `json:"mode"`
	WireValue string      `json:"wire_value,omitempty"`
}
type ServiceTierContract struct {
	Supported     []ServiceTierBinding `json:"supported,omitempty"`
	RequestField  string               `json:"request_field,omitempty"`
	RealizedField string               `json:"realized_field,omitempty"`
	CacheOnSwitch string               `json:"cache_on_switch"`
	PremiumPrice  KnowledgeState       `json:"premium_price"`
	Fallback      string               `json:"fallback"`
}
type ServiceTierRequest struct {
	Mode   ServiceMode
	Policy ServiceFallbackPolicy
}
type ServiceTierReceipt struct {
	Requested         ServiceMode           `json:"requested"`
	Wire              string                `json:"wire,omitempty"`
	Realized          ServiceMode           `json:"realized"`
	Policy            ServiceFallbackPolicy `json:"policy"`
	CapabilitySource  FactSource            `json:"capability_source"`
	DowngradeReason   string                `json:"downgrade_reason,omitempty"`
	CacheInvalidated  bool                  `json:"cache_invalidated"`
	CacheRewarmTokens int64                 `json:"cache_rewarm_tokens,omitempty"`
	PremiumPrice      KnowledgeState        `json:"premium_price"`
}

func (c ServiceTierContract) Binding(mode ServiceMode) (ServiceTierBinding, bool) {
	for _, b := range c.Supported {
		if b.Mode == mode {
			return b, true
		}
	}
	return ServiceTierBinding{}, false
}
func BindServiceTier(contract ProviderContract, req ServiceTierRequest) (string, ServiceTierReceipt, error) {
	if req.Mode == "" {
		req.Mode = ServiceModeStandard
	}
	if req.Policy == "" {
		req.Policy = ServiceRequire
	}
	r := ServiceTierReceipt{Requested: req.Mode, Realized: ServiceModeUnknown, Policy: req.Policy, CapabilitySource: contract.ServiceTiers.Source, PremiumPrice: KnowledgeUnknown}
	if req.Policy != ServiceRequire && req.Policy != ServiceAllowDeclaredDowngrade && req.Policy != ServiceStandardOnly {
		return "", r, fmt.Errorf("service tier: invalid fallback policy %q", req.Policy)
	}
	if req.Policy == ServiceStandardOnly && req.Mode != ServiceModeStandard {
		return "", r, fmt.Errorf("service tier: standard_only rejects %q", req.Mode)
	}
	if contract.ServiceTiers.State != KnowledgeKnown {
		return "", r, fmt.Errorf("service tier: provider %s capability is %s", contract.Provider, contract.ServiceTiers.State)
	}
	b, ok := contract.ServiceTiers.Value.Binding(req.Mode)
	if !ok {
		return "", r, fmt.Errorf("service tier: provider %s does not support %q", contract.Provider, req.Mode)
	}
	r.Wire = b.WireValue
	r.PremiumPrice = contract.ServiceTiers.Value.PremiumPrice
	return b.WireValue, r, nil
}

func RealizeServiceTier(r ServiceTierReceipt, realized ServiceMode, cacheRewarmTokens int64) (ServiceTierReceipt, error) {
	if realized == "" {
		realized = ServiceModeUnknown
	}
	r.Realized = realized
	if realized == ServiceModeUnknown && r.Policy == ServiceRequire {
		return r, fmt.Errorf("service tier: required %q, provider did not report realized tier", r.Requested)
	}
	if realized != ServiceModeUnknown && realized != r.Requested {
		r.DowngradeReason = "provider_realized_different_tier"
		if r.Policy == ServiceRequire {
			return r, fmt.Errorf("service tier: required %q, provider realized %q", r.Requested, realized)
		}
	}
	if cacheRewarmTokens > 0 {
		r.CacheInvalidated = true
		r.CacheRewarmTokens = cacheRewarmTokens
	}
	return r, nil
}

// SupportedServiceTierMetadata returns only contract-declared portable modes.
func SupportedServiceTierMetadata(contract ProviderContract) ([]string, []map[string]string) {
	if contract.ServiceTiers.State != KnowledgeKnown {
		return []string{}, []map[string]string{}
	}
	modes := make([]string, 0, len(contract.ServiceTiers.Value.Supported))
	rows := make([]map[string]string, 0, len(contract.ServiceTiers.Value.Supported))
	for _, binding := range contract.ServiceTiers.Value.Supported {
		modes = append(modes, string(binding.Mode))
		rows = append(rows, map[string]string{"mode": string(binding.Mode), "wire_value": binding.WireValue})
	}
	return modes, rows
}
