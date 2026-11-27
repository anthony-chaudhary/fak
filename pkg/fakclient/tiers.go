package fakclient

import (
	"context"
	"net/http"
	"strconv"
	"strings"
)

// TierStatus describes the live capacity and utilization of one KV-cache tier.
// Mirrors internal/kvbudget.TierStatus field-for-field by JSON tag.
type TierStatus struct {
	Name             string `json:"name"`
	Level            int    `json:"level"`
	MaxContext       int    `json:"max_context"`
	MinSlots         int    `json:"min_slots"`
	ActiveSlots      int    `json:"active_slots"`
	AvailableNow     int    `json:"available_now"`
	ElasticAvailable int    `json:"elastic_available"`
	Preemptible      bool   `json:"preemptible"`
}

// RecommendedFanout summarizes the safe concurrent fan-out shape for an
// agentic orchestrator reading GET /v1/fak/capacity/tiers.
// Mirrors internal/kvbudget.FanoutRecommendation field-for-field by JSON tag.
type RecommendedFanout struct {
	MasterMaxCtx int `json:"master_max_ctx"`
	WorkerSlots  int `json:"worker_slots"`
	WorkerMaxCtx int `json:"worker_max_ctx"`
	MicroSlots   int `json:"micro_slots"`
	MicroMaxCtx  int `json:"micro_max_ctx"`
}

// CapacityTiersSnapshot is the point-in-time response of GET /v1/fak/capacity/tiers.
// Mirrors internal/kvbudget.TierPoolSnapshot field-for-field by JSON tag.
type CapacityTiersSnapshot struct {
	Enabled           bool              `json:"enabled"`
	TotalTokens       int               `json:"total_tokens"`
	FreeTokens        int               `json:"free_tokens"`
	Tiers             []TierStatus      `json:"tiers"`
	RecommendedFanout RecommendedFanout `json:"recommended_fanout"`
}

// CapacityTiers queries the gateway's live KV-cache tier capacity and fan-out
// recommendation (GET /v1/fak/capacity/tiers).
func (c *Client) CapacityTiers(ctx context.Context) (CapacityTiersSnapshot, error) {
	var resp CapacityTiersSnapshot
	if err := c.do(ctx, http.MethodGet, "/v1/fak/capacity/tiers", nil, &resp); err != nil {
		return CapacityTiersSnapshot{}, err
	}
	return resp, nil
}

// ParseAvailableTiersHeader parses compact X-Fak-Available-Tiers header strings
// such as "master=0@131072,worker=4@32768,micro=32@4096" into TierStatus entries
// with Name, AvailableNow, and MaxContext populated. Malformed segments are skipped.
func ParseAvailableTiersHeader(h string) []TierStatus {
	h = strings.TrimSpace(h)
	if h == "" {
		return nil
	}
	parts := strings.Split(h, ",")
	out := make([]TierStatus, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		name, rest, ok := strings.Cut(part, "=")
		if !ok {
			continue
		}
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		availStr, maxCtxStr, ok := strings.Cut(rest, "@")
		if !ok {
			continue
		}
		avail, err1 := strconv.Atoi(strings.TrimSpace(availStr))
		maxCtx, err2 := strconv.Atoi(strings.TrimSpace(maxCtxStr))
		if err1 != nil || err2 != nil || avail < 0 || maxCtx <= 0 {
			continue
		}
		out = append(out, TierStatus{
			Name:         name,
			AvailableNow: avail,
			MaxContext:   maxCtx,
		})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
