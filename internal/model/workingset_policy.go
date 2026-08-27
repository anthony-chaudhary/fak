package model

import (
	"fmt"
	"sort"
)

type WorkingSet struct {
	ID             string  `json:"id"`
	Tenant         string  `json:"tenant"`
	Bytes          int64   `json:"bytes"`
	ReuseValue     float64 `json:"reuse_value"`
	ReloadBytes    int64   `json:"reload_bytes"`
	AcceptedTokens int     `json:"accepted_tokens"`
}
type WorkingSetDecision struct {
	Schema                     string   `json:"schema"`
	Engine                     string   `json:"engine"`
	CapacityBytes              int64    `json:"capacity_bytes"`
	Resident                   []string `json:"resident"`
	Evicted                    []string `json:"evicted"`
	ResidentBytes              int64    `json:"resident_bytes"`
	EvictionAmplificationBytes int64    `json:"eviction_amplification_bytes"`
	MinTenantShare             float64  `json:"min_tenant_share"`
	ReloadBytesPerAccepted     float64  `json:"reload_bytes_per_accepted_token"`
	StopRule                   string   `json:"stop_rule"`
	Rollback                   string   `json:"rollback"`
}

// PlaceWorkingSets chooses value-per-byte while reserving each active tenant a fair
// share before filling spare capacity globally. Eviction and reload cost remain explicit.
func PlaceWorkingSets(sets []WorkingSet, capacity int64) (WorkingSetDecision, error) {
	if capacity < 0 {
		return WorkingSetDecision{}, fmt.Errorf("model: negative working-set capacity")
	}
	r := WorkingSetDecision{Schema: "fak-working-set-decision/1", Engine: "fak-native", CapacityBytes: capacity, StopRule: "reject placement when reload amplification or tails exceed control", Rollback: "isolate tenants with static partitions"}
	tenants := map[string][]WorkingSet{}
	for _, s := range sets {
		if s.ID == "" || s.Tenant == "" || s.Bytes <= 0 || s.ReuseValue < 0 || s.ReloadBytes < 0 || s.AcceptedTokens < 0 {
			return WorkingSetDecision{}, fmt.Errorf("model: invalid working set")
		}
		tenants[s.Tenant] = append(tenants[s.Tenant], s)
	}
	selected := map[string]bool{}
	share := int64(0)
	if len(tenants) > 0 {
		share = capacity / int64(len(tenants))
	}
	for _, list := range tenants {
		sort.SliceStable(list, func(i, j int) bool {
			return list[i].ReuseValue/float64(list[i].Bytes) > list[j].ReuseValue/float64(list[j].Bytes)
		})
		used := int64(0)
		for _, s := range list {
			if used+s.Bytes <= share {
				selected[s.ID] = true
				used += s.Bytes
				r.ResidentBytes += s.Bytes
			}
		}
	}
	remaining := append([]WorkingSet(nil), sets...)
	sort.SliceStable(remaining, func(i, j int) bool {
		return remaining[i].ReuseValue/float64(remaining[i].Bytes) > remaining[j].ReuseValue/float64(remaining[j].Bytes)
	})
	for _, s := range remaining {
		if !selected[s.ID] && r.ResidentBytes+s.Bytes <= capacity {
			selected[s.ID] = true
			r.ResidentBytes += s.Bytes
		}
	}
	tenantBytes := map[string]int64{}
	accepted := 0
	for _, s := range sets {
		accepted += s.AcceptedTokens
		if selected[s.ID] {
			r.Resident = append(r.Resident, s.ID)
			tenantBytes[s.Tenant] += s.Bytes
		} else {
			r.Evicted = append(r.Evicted, s.ID)
			r.EvictionAmplificationBytes += s.ReloadBytes
		}
	}
	if r.ResidentBytes > 0 {
		r.MinTenantShare = 1
		for t := range tenants {
			v := float64(tenantBytes[t]) / float64(r.ResidentBytes)
			if v < r.MinTenantShare {
				r.MinTenantShare = v
			}
		}
	}
	if accepted > 0 {
		r.ReloadBytesPerAccepted = float64(r.EvictionAmplificationBytes) / float64(accepted)
	}
	return r, nil
}
