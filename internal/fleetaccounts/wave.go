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
	Leases            []Lease
	AllowTierFallback bool
	StrictTier        bool
	WaveID            string
}

// WaveLane is the flat resolve record for one allocated account plus its
// distinct-pool and rank-stamped wave membership.
type WaveLane struct {
	Resolved
	Pool        string `json:"pool"`
	SessionSlot int    `json:"session_slot,omitempty"`
	SessionCap  int    `json:"session_cap,omitempty"`
	Rank        int    `json:"rank"`
	WaveID      string `json:"wave_id"`
	Size        int    `json:"size"`
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
	// DroppedSeats lists seats out of the pool pending an interactive re-login
	// (#2075): the reason a fan-out is smaller than the roster suggests, surfaced
	// so the shrink is never silent.
	DroppedSeats []DroppedSeat `json:"dropped_seats,omitempty"`
}

// AllocateWave allocates up to Count available account session slots for a
// parallel fan-out. It is the multi-account sibling of Resolve: lanes are flat
// resolve records, and a healthy Claude account can appear once per session slot
// up to its product cap.
func AllocateWave(rows []Account, req WaveRequest, pol Policy) WaveResult {
	n := req.Count
	if n < 0 {
		n = 0
	}
	class, strict := dispatchTaskClass(req.TaskClass, req.WorkKind, req.StrictTier)
	task := ClassifyTask(req.TaskText, class, pol)
	target := task.TargetTier

	wantedProduct := strings.ToLower(strings.TrimSpace(req.Product))
	workers, available := routableAndAvailable(rows, wantedProduct)

	// The same fallback ladder as RouteAccount: no non-apex target lists tier 0, so an
	// apex (Fable 5) seat is never swept into a hard/light wave. An explicit apex target
	// degrades DOWN to frontier (never sideways to light) when no apex seat is offered.
	tierOrder, _ := routingTierOrder(target, strict, req.AllowTierFallback, pol)

	var lanes []WaveLane
	usedPools := map[string]bool{}
	leaseWorkers, _ := leaseWorkersByPool(workers, req.Leases)
	load := map[string]int{}
	for pool, workers := range leaseWorkers {
		load[pool] = len(workers)
	}
	// Floor each pool's load at its registry-live session count: sessions launched
	// outside this dispatcher (interactive resumes, the watchdog) hold no seat
	// lease, and a wave that only counts its own leases would grant a full wave
	// onto a seat already running at capacity. max() rather than sum, because a
	// dispatch-launched session appears in both counts.
	for _, r := range uniquePoolAccounts(workers) {
		if live := derefInt(r.LiveSessions); live > load[PoolKey(r)] {
			load[PoolKey(r)] = live
		}
	}
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
		candidates = uniquePoolAccounts(candidates)
		sort.SliceStable(candidates, func(i, j int) bool {
			return rankLess(candidates[i], candidates[j])
		})
		for len(lanes) < n {
			best := -1
			for i, r := range candidates {
				pool := PoolKey(r)
				if load[pool] >= AccountSessionCap(r) {
					continue
				}
				if best < 0 || waveSlotLess(r, candidates[best], load[pool], load[PoolKey(candidates[best])]) {
					best = i
				}
			}
			if best < 0 {
				break
			}
			r := candidates[best]
			pool := PoolKey(r)
			capacity := AccountSessionCap(r)
			slot := load[pool] + 1
			load[pool] = slot
			usedPools[pool] = true
			reason := "wave lane (target tier)"
			if tier != target {
				reason = "wave lane (fallback tier)"
			}
			tt := target
			lanes = append(lanes, WaveLane{
				Resolved:    flattenResolved(r, true, reason, r.ModelTier, &tt, tier != target, ""),
				Pool:        pool,
				SessionSlot: slot,
				SessionCap:  capacity,
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
		reason = fmt.Sprintf("granted %d of %d session slot(s) across %d distinct pool(s); %d short (roster has no more available session slots at the requested tiers)", granted, n, len(usedPools), shortfall)
	default:
		reason = fmt.Sprintf("granted %d session slot(s) across %d distinct pool(s)", granted, len(usedPools))
	}

	// A pool shrunk by a stale credential must never shrink silently (#2075):
	// name the seats waiting on a re-login so the operator sees why the fan-out
	// is smaller than the roster suggests.
	var dropped []DroppedSeat
	if wantedProduct == "" || wantedProduct == "claude" {
		dropped = DroppedSeats(rows)
	}
	if len(dropped) > 0 {
		tags := make([]string, 0, len(dropped))
		for _, d := range dropped {
			tags = append(tags, d.Tag)
		}
		reason += fmt.Sprintf("; %d seat(s) dropped pending re-login: %s",
			len(dropped), strings.Join(tags, ", "))
	}

	if lanes == nil {
		lanes = []WaveLane{}
	}
	return WaveResult{
		OK:                    granted > 0,
		Requested:             n,
		Granted:               granted,
		Shortfall:             shortfall,
		DistinctPools:         len(usedPools),
		Size:                  granted,
		WaveID:                waveID,
		TargetTier:            target,
		Reason:                reason,
		Lanes:                 lanes,
		BlockedTargetAccounts: blocked,
		DroppedSeats:          dropped,
	}
}

func waveSlotLess(a, b Account, loadA, loadB int) bool {
	if loadA != loadB {
		return loadA < loadB
	}
	return rankLess(a, b)
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
