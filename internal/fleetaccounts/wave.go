package fleetaccounts

import (
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
)

// WaveRequest carries the account-wave allocation inputs.
type WaveRequest struct {
	Count             int
	TaskText          string
	TaskClass         string
	WorkKind          string
	Product           string
	AllowTierFallback bool
	StrictTier        bool
	WaveID            string
}

// WaveLane is the flat resolve record for one allocated account plus its
// distinct-pool and rank-stamped wave membership.
type WaveLane struct {
	Resolved
	Pool   string `json:"pool"`
	Rank   int    `json:"rank"`
	WaveID string `json:"wave_id"`
	Size   int    `json:"size"`
}

// WaveResult is the native account-wave allocation shape.
type WaveResult struct {
	OK                    bool             `json:"ok"`
	Requested             int              `json:"requested"`
	Granted               int              `json:"granted"`
	Shortfall             int              `json:"shortfall"`
	DistinctPools         int              `json:"distinct_pools"`
	Size                  int              `json:"size"`
	WaveID                string           `json:"wave_id"`
	TargetTier            int              `json:"target_tier"`
	Reason                string           `json:"reason"`
	Lanes                 []WaveLane       `json:"lanes"`
	BlockedTargetAccounts []BlockedAccount `json:"blocked_target_accounts"`
}

// AllocateWave allocates up to Count distinct available account pools for a
// parallel fan-out. It is the multi-account sibling of Resolve: lanes are flat
// resolve records, but no two lanes share the same PoolKey.
func AllocateWave(rows []Account, req WaveRequest, pol Policy) WaveResult {
	n := req.Count
	if n < 0 {
		n = 0
	}
	cls, strict := req.TaskClass, req.StrictTier
	wk := strings.ToLower(strings.TrimSpace(req.WorkKind))
	if gardeningWorkKinds[wk] || engineeringWorkKinds[wk] {
		cls, strict = wk, false
	}
	task := ClassifyTask(req.TaskText, cls, pol)
	target := task.TargetTier

	wantedProduct := strings.ToLower(strings.TrimSpace(req.Product))
	workers, available := routableAndAvailable(rows, wantedProduct)

	fallbackPolicy := strings.ToLower(pol.Routing.HardTier1Fallback)
	effectiveAllow := req.AllowTierFallback || in(fallbackPolicy, "allow", "fallback", "tier2", "t2")
	tierOrder := []int{target}
	if target == 2 && !strict {
		tierOrder = append(tierOrder, 1)
	} else if effectiveAllow {
		tierOrder = append(tierOrder, 2)
	}

	var lanes []WaveLane
	seenPools := map[string]bool{}
	for _, tier := range tierOrder {
		if len(lanes) >= n {
			break
		}
		var candidates []Account
		for _, r := range available {
			if tierOf(r) == tier {
				candidates = append(candidates, r)
			}
		}
		sort.SliceStable(candidates, func(i, j int) bool {
			return rankLess(candidates[i], candidates[j])
		})
		for _, r := range candidates {
			if len(lanes) >= n {
				break
			}
			pool := PoolKey(r)
			if seenPools[pool] {
				continue
			}
			seenPools[pool] = true
			reason := "wave lane (target tier)"
			if tier != target {
				reason = "wave lane (fallback tier)"
			}
			tt := target
			lanes = append(lanes, WaveLane{
				Resolved: flattenResolved(r, true, reason, r.ModelTier, &tt, tier != target, ""),
				Pool:     pool,
			})
		}
	}

	granted := len(lanes)
	shortfall := n - granted
	if shortfall < 0 {
		shortfall = 0
	}
	waveID := strings.TrimSpace(req.WaveID)
	if waveID == "" {
		waveID = waveIDForPools(lanePools(lanes))
	}
	for i := range lanes {
		lanes[i].Rank = i
		lanes[i].WaveID = waveID
		lanes[i].Size = granted
	}

	blocked := make([]BlockedAccount, 0)
	for _, r := range workers {
		if tierOf(r) == target && !accountCanBeOffered(r) {
			blocked = append(blocked, publicBlocked(r))
		}
	}

	reason := ""
	switch {
	case granted == 0:
		reason = fmt.Sprintf("no available account for a wave (target tier %d", target)
		if wantedProduct != "" {
			reason += ", product " + wantedProduct
		}
		reason += ")"
	case shortfall > 0:
		reason = fmt.Sprintf("granted %d of %d distinct pools; %d short (roster has no more distinct available pools at the requested tiers)", granted, n, shortfall)
	default:
		reason = fmt.Sprintf("granted %d distinct pools", granted)
	}

	if lanes == nil {
		lanes = []WaveLane{}
	}
	return WaveResult{
		OK:                    granted > 0,
		Requested:             n,
		Granted:               granted,
		Shortfall:             shortfall,
		DistinctPools:         granted,
		Size:                  granted,
		WaveID:                waveID,
		TargetTier:            target,
		Reason:                reason,
		Lanes:                 lanes,
		BlockedTargetAccounts: blocked,
	}
}

func lanePools(lanes []WaveLane) []string {
	out := make([]string, 0, len(lanes))
	for _, lane := range lanes {
		out = append(out, lane.Pool)
	}
	return out
}

func waveIDForPools(pools []string) string {
	if len(pools) == 0 {
		return ""
	}
	sort.Strings(pools)
	sum := sha256.Sum256([]byte(strings.Join(pools, ",")))
	return "wave-" + fmt.Sprintf("%x", sum[:6])
}
