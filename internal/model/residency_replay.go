package model

import "fmt"

// ResidencyArrival is one multi-tenant request against a base, adapter, quant variant, and hot set.
type ResidencyArrival struct {
	Tenant           string `json:"tenant"`
	Base             string `json:"base"`
	Adapter          string `json:"adapter"`
	Variant          string `json:"variant"`
	HotSet           string `json:"hot_set"`
	BaseBytes        int64  `json:"base_bytes"`
	DeltaBytes       int64  `json:"delta_bytes"`
	AcceptedTokens   int    `json:"accepted_tokens"`
	QueueNanoseconds int64  `json:"queue_nanoseconds"`
}

type ResidencyReplayReceipt struct {
	Schema                 string         `json:"schema"`
	Engine                 string         `json:"engine"`
	CapacityBytes          int64          `json:"capacity_bytes"`
	PeakResidentBytes      int64          `json:"peak_resident_bytes"`
	BaseLoads              int            `json:"base_loads"`
	DeltaLoads             int            `json:"delta_loads"`
	ReloadBytes            int64          `json:"reload_bytes"`
	EvictionBytes          int64          `json:"eviction_bytes"`
	AcceptedTokens         int            `json:"accepted_tokens"`
	ReloadBytesPerAccepted float64        `json:"reload_bytes_per_accepted_token"`
	MaxQueueNanoseconds    int64          `json:"max_queue_nanoseconds"`
	TenantRequests         map[string]int `json:"tenant_requests"`
	StopRule               string         `json:"stop_rule"`
	Rollback               string         `json:"rollback"`
}

// ReplayResidency reuses shared bases while bounding resident deltas. The oldest
// delta is evicted first; base eviction is avoided whenever one shared base plus
// the arriving delta fits, preventing adapter churn from forcing base reloads.
func ReplayResidency(arrivals []ResidencyArrival, capacity int64) (ResidencyReplayReceipt, error) {
	if capacity < 0 {
		return ResidencyReplayReceipt{}, fmt.Errorf("model: negative residency capacity")
	}
	r := ResidencyReplayReceipt{Schema: "fak-residency-replay/1", Engine: "fak-native", CapacityBytes: capacity, TenantRequests: map[string]int{}, StopRule: "reject placement when one base+delta exceeds capacity or reload tails regress", Rollback: "pin one model per worker"}
	bases := map[string]int64{}
	deltas := map[string]int64{}
	order := []string{}
	resident := int64(0)
	for _, a := range arrivals {
		if a.Tenant == "" || a.Base == "" || a.BaseBytes <= 0 || a.DeltaBytes < 0 || a.AcceptedTokens < 0 || a.QueueNanoseconds < 0 {
			return ResidencyReplayReceipt{}, fmt.Errorf("model: invalid residency arrival")
		}
		if a.BaseBytes+a.DeltaBytes > capacity {
			return ResidencyReplayReceipt{}, fmt.Errorf("model: base plus delta exceeds capacity")
		}
		r.TenantRequests[a.Tenant]++
		r.AcceptedTokens += a.AcceptedTokens
		if a.QueueNanoseconds > r.MaxQueueNanoseconds {
			r.MaxQueueNanoseconds = a.QueueNanoseconds
		}
		if _, ok := bases[a.Base]; !ok {
			resident = evictResidencyDeltas(resident, a.BaseBytes, capacity, &order, deltas, &r)
			if resident+a.BaseBytes > capacity {
				return ResidencyReplayReceipt{}, fmt.Errorf("model: base residency exceeds capacity")
			}
			bases[a.Base] = a.BaseBytes
			resident += a.BaseBytes
			r.BaseLoads++
			r.ReloadBytes += a.BaseBytes
		}
		key := a.Base + "|" + a.Adapter + "|" + a.Variant + "|" + a.HotSet
		if _, ok := deltas[key]; !ok {
			resident = evictResidencyDeltas(resident, a.DeltaBytes, capacity, &order, deltas, &r)
			if resident+a.DeltaBytes > capacity {
				return ResidencyReplayReceipt{}, fmt.Errorf("model: delta residency exceeds capacity")
			}
			deltas[key] = a.DeltaBytes
			order = append(order, key)
			resident += a.DeltaBytes
			r.DeltaLoads++
			r.ReloadBytes += a.DeltaBytes
		}
		if resident > r.PeakResidentBytes {
			r.PeakResidentBytes = resident
		}
	}
	if r.AcceptedTokens > 0 {
		r.ReloadBytesPerAccepted = float64(r.ReloadBytes) / float64(r.AcceptedTokens)
	}
	return r, nil
}

func evictResidencyDeltas(resident, incoming, capacity int64, order *[]string, deltas map[string]int64, receipt *ResidencyReplayReceipt) int64 {
	for resident+incoming > capacity && len(*order) > 0 {
		key := (*order)[0]
		*order = (*order)[1:]
		resident -= deltas[key]
		receipt.EvictionBytes += deltas[key]
		delete(deltas, key)
	}
	return resident
}
