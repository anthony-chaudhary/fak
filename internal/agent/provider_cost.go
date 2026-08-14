package agent

import (
	"bytes"
	"encoding/json"
)

// attachProviderReportedCost copies a provider-authored dollar amount without
// pricing tokens locally. OpenAI-compatible gateways commonly expose one of
// these additive fields at the response top level; absent, null, negative, or
// non-numeric values remain unsupported rather than becoming a zero-cost claim.
func attachProviderReportedCost(comp *Completion, raw []byte) {
	if comp == nil {
		return
	}
	if comp.Usage.CostUSD != nil {
		comp.Usage.CostStatus = "provider-reported"
		if comp.Usage.CostProvenance == "" {
			comp.Usage.CostProvenance = "response.usage.cost_usd"
		}
		return
	}
	if len(bytes.TrimSpace(raw)) == 0 {
		return
	}
	type wireUsage struct {
		CostUSD *float64 `json:"cost_usd"`
		Cost    *float64 `json:"cost"`
	}
	var wire struct {
		CostUSD      *float64  `json:"cost_usd"`
		TotalCostUSD *float64  `json:"total_cost_usd"`
		ResponseCost *float64  `json:"response_cost"`
		Usage        wireUsage `json:"usage"`
		HiddenParams *struct {
			ResponseCost *float64 `json:"response_cost"`
		} `json:"_hidden_params"`
		Response *struct {
			CostUSD      *float64  `json:"cost_usd"`
			TotalCostUSD *float64  `json:"total_cost_usd"`
			ResponseCost *float64  `json:"response_cost"`
			Usage        wireUsage `json:"usage"`
		} `json:"response"`
	}
	if json.Unmarshal(raw, &wire) != nil {
		return
	}
	candidates := []struct {
		value  *float64
		source string
	}{
		{wire.CostUSD, "response.cost_usd"},
		{wire.TotalCostUSD, "response.total_cost_usd"},
		{wire.ResponseCost, "response.response_cost"},
		{wire.Usage.CostUSD, "response.usage.cost_usd"},
		{wire.Usage.Cost, "response.usage.cost"},
	}
	if wire.HiddenParams != nil {
		candidates = append(candidates, struct {
			value  *float64
			source string
		}{wire.HiddenParams.ResponseCost, "response._hidden_params.response_cost"})
	}
	if wire.Response != nil {
		candidates = append(candidates,
			struct {
				value  *float64
				source string
			}{wire.Response.CostUSD, "response.response.cost_usd"},
			struct {
				value  *float64
				source string
			}{wire.Response.TotalCostUSD, "response.response.total_cost_usd"},
			struct {
				value  *float64
				source string
			}{wire.Response.ResponseCost, "response.response.response_cost"},
			struct {
				value  *float64
				source string
			}{wire.Response.Usage.CostUSD, "response.response.usage.cost_usd"},
			struct {
				value  *float64
				source string
			}{wire.Response.Usage.Cost, "response.response.usage.cost"},
		)
	}
	for _, candidate := range candidates {
		if candidate.value == nil || *candidate.value < 0 {
			continue
		}
		cost := *candidate.value
		comp.Usage.CostUSD = &cost
		comp.Usage.CostStatus = "provider-reported"
		comp.Usage.CostProvenance = candidate.source
		return
	}
}
