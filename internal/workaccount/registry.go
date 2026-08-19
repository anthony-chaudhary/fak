// Package workaccount declares every shipped fak mechanism whose work effect must be
// visible, explicitly excluded, or honestly unavailable in the WORK DONE product seam.
package workaccount

import (
	"fmt"
	"sort"
)

type Status string

const (
	Accounted   Status = "accounted"
	Overlapping Status = "overlapping"
	Excluded    Status = "intentionally_excluded"
	Unavailable Status = "not_yet_measurable"
)

type Mechanism struct {
	ID               string   `json:"id"`
	Label            string   `json:"label"`
	Producer         string   `json:"producer"`
	Status           Status   `json:"status"`
	Units            []string `json:"units,omitempty"`
	SourceID         string   `json:"source_id,omitempty"`
	ExclusivityGroup string   `json:"exclusivity_group,omitempty"`
	Reason           string   `json:"reason,omitempty"`
}

//enumlint:exempt The shipped registry has no unavailable mechanism; unavailable is retained for honest future declarations.
func Registry() []Mechanism {
	return []Mechanism{
		{ID: "provider_prompt_cache", Label: "provider prompt cache", Producer: "internal/gateway.AdjudicationSummary.CachedPromptTokens", Status: Accounted, Units: []string{"input_tokens"}, SourceID: "provider_cache", ExclusivityGroup: "input_token_equiv_owner/v1"},
		{ID: "response_vdso_memo", Label: "response/vDSO memoization", Producer: "internal/gateway.cacheAttributionVars(vdsoHits)", Status: Accounted, Units: []string{"model_calls"}, SourceID: "fak_response_reuse", ExclusivityGroup: "avoided_model_call_path/v1"},
		{ID: "inline_tool_serving", Label: "inline tool serving", Producer: "internal/gateway.gatewayMetrics.servedInlineSnapshot", Status: Accounted, Units: []string{"model_calls"}, SourceID: "inline_tool_local", ExclusivityGroup: "avoided_model_call_path/v1"},
		{ID: "context_compaction", Label: "context compaction", Producer: "internal/gateway.AdjudicationSummary.CompactionShedTokens", Status: Accounted, Units: []string{"input_tokens"}, SourceID: "context_reduction", ExclusivityGroup: "input_token_equiv_owner/v1"},
		{ID: "kv_prefix_reuse", Label: "fak KV-prefix reuse", Producer: "internal/gateway.AdjudicationSummary.KVPrefixReusedTokens", Status: Accounted, Units: []string{"input_tokens"}, SourceID: "fak_prefix_reuse", ExclusivityGroup: "input_token_equiv_owner/v1"},
		{ID: "context_elision", Label: "stale-read context elision", Producer: "internal/agent anthropic elision path", Status: Accounted, Units: []string{"input_tokens", "events"}, SourceID: "context_reduction", ExclusivityGroup: "input_token_equiv_owner/v1"},
		{ID: "schema_tool_filtering", Label: "schema/tool filtering", Producer: "internal/gateway tool projection/filter path", Status: Accounted, Units: []string{"input_tokens", "events"}, SourceID: "context_reduction", ExclusivityGroup: "input_token_equiv_owner/v1"},
		{ID: "cold_tool_defer", Label: "cold-tool defer routing", Producer: "internal/gateway.AdjudicationSummary.DeferColdCount", Status: Overlapping, Units: []string{"input_tokens", "model_calls", "latency", "events"}, SourceID: "unknown", ExclusivityGroup: "routing_defer_counterfactual/v1", Reason: "observed decision count and paired-run calibrated deltas are published separately; modeled deltas overlap provider execution and are not added to WORK DONE totals"},
		{ID: "model_routing", Label: "model routing/defer", Producer: "internal/gateway routing decision records", Status: Overlapping, Units: []string{"input_tokens", "model_calls", "latency", "events"}, SourceID: "unknown", ExclusivityGroup: "routing_defer_counterfactual/v1", Reason: "observed decision count and compatible paired-run calibration are published separately; modeled deltas are counterfactual and not additive with provider/cache effects"},
		{ID: "safety_intervention", Label: "safety interventions", Producer: "internal/gateway.AdjudicationSummary", Status: Excluded, Units: []string{"operator_interventions"}, Reason: "safety BLOCK/FIX/DEFER counts are visible in the Safety panel but are not token or call savings"},
	}
}

func Validate(rows []Mechanism) error {
	seen := map[string]bool{}
	for i, row := range rows {
		if row.ID == "" || row.Label == "" || row.Producer == "" {
			return fmt.Errorf("row %d: id, label, and producer are required", i)
		}
		if seen[row.ID] {
			return fmt.Errorf("%s: duplicate mechanism id", row.ID)
		}
		seen[row.ID] = true
		switch row.Status {
		case Accounted:
			if len(row.Units) == 0 || row.SourceID == "" || row.ExclusivityGroup == "" {
				return fmt.Errorf("%s: accounted mechanism requires units, source_id, and exclusivity_group", row.ID)
			}
		case Overlapping:
			if len(row.Units) == 0 || row.ExclusivityGroup == "" || row.Reason == "" {
				return fmt.Errorf("%s: overlapping mechanism requires units, exclusivity_group, and reason", row.ID)
			}
		case Excluded, Unavailable:
			if row.Reason == "" {
				return fmt.Errorf("%s: %s mechanism requires reason", row.ID, row.Status)
			}
		default:
			return fmt.Errorf("%s: unclassified status %q", row.ID, row.Status)
		}
	}
	return nil
}

type Report struct {
	Schema     string         `json:"schema"`
	Mechanisms []Mechanism    `json:"mechanisms"`
	Counts     map[Status]int `json:"counts"`
}

func BuildReport(rows []Mechanism) (Report, error) {
	if err := Validate(rows); err != nil {
		return Report{}, err
	}
	rows = append([]Mechanism(nil), rows...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].ID < rows[j].ID })
	r := Report{Schema: "fak.info.work-accounting-coverage/1", Mechanisms: rows, Counts: map[Status]int{}}
	for _, row := range rows {
		r.Counts[row.Status]++
	}
	return r, nil
}
