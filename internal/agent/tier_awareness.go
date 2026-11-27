package agent

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

// AgentRole identifies an agent's position in the L1/L2/L3 KV-cache tier hierarchy.
type AgentRole string

const (
	RoleMaster AgentRole = "master"
	RoleWorker AgentRole = "worker"
	RoleMicro  AgentRole = "micro"
)

// TierBudgetPlan describes the active context ceiling, compaction watermark,
// and safe parallel fan-out budget for an agent given a live capacity snapshot.
type TierBudgetPlan struct {
	Role                AgentRole
	AssignedTier        string
	ContextCap          int
	CompactionWatermark int // 80% of ContextCap
	MaxParallelWorkers  int
	WorkerMaxCtx        int
	MaxParallelMicros   int
	MicroMaxCtx         int
	Degraded            bool
	Advice              string
}

func normalizeAgentRole(role AgentRole) AgentRole {
	switch AgentRole(strings.ToLower(strings.TrimSpace(string(role)))) {
	case RoleMaster:
		return RoleMaster
	case RoleWorker:
		return RoleWorker
	case RoleMicro:
		return RoleMicro
	default:
		return RoleMaster
	}
}

func findTierStatus(tiers []fakclient.TierStatus, name string) (fakclient.TierStatus, bool) {
	for _, t := range tiers {
		if t.Name == name {
			return t, true
		}
	}
	return fakclient.TierStatus{}, false
}

// PlanAgentTopology maps a CapacityTiersSnapshot and AgentRole into a
// deterministic TierBudgetPlan with safe parallel fan-out counts and an 80%
// compaction watermark.
func PlanAgentTopology(snap fakclient.CapacityTiersSnapshot, role AgentRole) TierBudgetPlan {
	normRole := normalizeAgentRole(role)
	assignedTier := string(normRole)

	masterTier, hasMaster := findTierStatus(snap.Tiers, string(RoleMaster))
	workerTier, hasWorker := findTierStatus(snap.Tiers, string(RoleWorker))
	microTier, hasMicro := findTierStatus(snap.Tiers, string(RoleMicro))

	workerMaxCtx := snap.RecommendedFanout.WorkerMaxCtx
	if workerMaxCtx <= 0 && hasWorker {
		workerMaxCtx = workerTier.MaxContext
	}
	maxWorkers := snap.RecommendedFanout.WorkerSlots
	if maxWorkers <= 0 && hasWorker && workerTier.AvailableNow > 0 {
		maxWorkers = workerTier.AvailableNow
	}

	microMaxCtx := snap.RecommendedFanout.MicroMaxCtx
	if microMaxCtx <= 0 && hasMicro {
		microMaxCtx = microTier.MaxContext
	}
	maxMicros := snap.RecommendedFanout.MicroSlots
	if maxMicros <= 0 && hasMicro && microTier.AvailableNow > 0 {
		maxMicros = microTier.AvailableNow
	}

	var contextCap int
	switch normRole {
	case RoleMaster:
		if hasMaster && masterTier.MaxContext > 0 {
			contextCap = masterTier.MaxContext
		} else {
			contextCap = snap.RecommendedFanout.MasterMaxCtx
		}
	case RoleWorker:
		contextCap = workerMaxCtx
	case RoleMicro:
		contextCap = microMaxCtx
	}

	watermark := (contextCap * 80) / 100

	plan := TierBudgetPlan{
		Role:                normRole,
		AssignedTier:        assignedTier,
		ContextCap:          contextCap,
		CompactionWatermark: watermark,
		MaxParallelWorkers:  maxWorkers,
		WorkerMaxCtx:        workerMaxCtx,
		MaxParallelMicros:   maxMicros,
		MicroMaxCtx:         microMaxCtx,
	}

	if !snap.Enabled || len(snap.Tiers) == 0 {
		plan.Degraded = true
		plan.Advice = "Tiered KV capacity is unavailable; keep prompts compact and avoid wide parallel fan-out."
		return plan
	}

	switch normRole {
	case RoleMaster:
		if !hasMaster || masterTier.AvailableNow <= 0 {
			plan.Degraded = true
			switch {
			case maxWorkers > 0:
				plan.Advice = fmt.Sprintf(
					"Master tier has 0 available slots; spawn up to %d L2 worker(s) (%d tok) or compact context to fit L2.",
					maxWorkers, workerMaxCtx,
				)
			case maxMicros > 0:
				plan.Advice = fmt.Sprintf(
					"Master and worker tiers are full; decompose into up to %d L3 micro tasks (%d tok) or compact.",
					maxMicros, microMaxCtx,
				)
			default:
				plan.Advice = "All KV tiers are saturated (0 available slots); compact context aggressively and wait for slot release."
			}
		} else {
			plan.Advice = fmt.Sprintf(
				"Compact at %d tok (80%% of %d); delegate bounded work to %d worker(s) or %d micro(s).",
				watermark, contextCap, maxWorkers, maxMicros,
			)
		}
	case RoleWorker:
		if !hasWorker || workerTier.AvailableNow <= 0 {
			plan.Degraded = true
			if maxMicros > 0 {
				plan.Advice = fmt.Sprintf(
					"Worker tier has 0 available slots; offload leaf checks to %d L3 micro slot(s) (%d tok) or compact.",
					maxMicros, microMaxCtx,
				)
			} else {
				plan.Advice = "Worker tier is saturated (0 available slots); compact context and wait for slot release."
			}
		} else {
			plan.Advice = fmt.Sprintf(
				"Stay under %d tok (compact at %d); offload <=%d tok leaf tasks to micro slots.",
				contextCap, watermark, microMaxCtx,
			)
		}
	case RoleMicro:
		if !hasMicro || microTier.AvailableNow <= 0 {
			plan.Degraded = true
			plan.Advice = "Micro tier has 0 available slots; keep turn minimal or wait for slot release."
		} else {
			plan.Advice = fmt.Sprintf(
				"Execute focused leaf subtask within %d tok (compact at %d); do not fan out.",
				contextCap, watermark,
			)
		}
	}

	return plan
}

// FormatTierAwarenessPrompt renders a compact, deterministic system-prompt block
// (<= 8 lines) informing the agent of its context ceiling, compaction watermark,
// available parallel slots, and decomposition rules.
func FormatTierAwarenessPrompt(snap fakclient.CapacityTiersSnapshot, role AgentRole) string {
	plan := PlanAgentTopology(snap, role)
	status := "healthy"
	if plan.Degraded {
		status = "DEGRADED"
	}

	lines := []string{
		fmt.Sprintf("[KV Tier Topology | role=%s tier=%s status=%s]", plan.Role, plan.AssignedTier, status),
		fmt.Sprintf("- Context ceiling: %d tokens; compact before %d tokens (80%%).", plan.ContextCap, plan.CompactionWatermark),
		fmt.Sprintf("- Parallel capacity: %d worker slots @ %d tok | %d micro slots @ %d tok.", plan.MaxParallelWorkers, plan.WorkerMaxCtx, plan.MaxParallelMicros, plan.MicroMaxCtx),
		fmt.Sprintf("- Decomposition: use worker (<=%d tok) for multi-file edits; micro (<=%d tok) for single-file/leaf checks.", plan.WorkerMaxCtx, plan.MicroMaxCtx),
		fmt.Sprintf("- Guidance: %s", plan.Advice),
	}
	return strings.Join(lines, "\n")
}

// SelectTierForSubtask chooses the best-fit tier for a subtask of estimatedTokens
// given the current snapshot and caller role, returning the tier name, its
// context cap, and actionable placement advice.
func SelectTierForSubtask(snap fakclient.CapacityTiersSnapshot, estimatedTokens int, role AgentRole) (tierName string, capTokens int, advice string) {
	if estimatedTokens < 0 {
		estimatedTokens = 0
	}
	if len(snap.Tiers) == 0 {
		return string(normalizeAgentRole(role)), 0, "no tier capacity metadata available; compact before dispatch"
	}

	// Sort a copy ascending by MaxContext so we always pick the smallest fitting tier.
	sorted := make([]fakclient.TierStatus, len(snap.Tiers))
	copy(sorted, snap.Tiers)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].MaxContext != sorted[j].MaxContext {
			return sorted[i].MaxContext < sorted[j].MaxContext
		}
		return sorted[i].Level > sorted[j].Level
	})

	var smallestFitting *fakclient.TierStatus
	for i := range sorted {
		t := &sorted[i]
		if t.MaxContext >= estimatedTokens {
			if smallestFitting == nil {
				smallestFitting = t
			}
			if t.AvailableNow > 0 {
				if t.Name == smallestFitting.Name {
					return t.Name, t.MaxContext, fmt.Sprintf("fits %s tier (%d <= %d tok, %d slots free)", t.Name, estimatedTokens, t.MaxContext, t.AvailableNow)
				}
				return t.Name, t.MaxContext, fmt.Sprintf("escalated to %s tier (%d slots free) because %s tier has 0 available slots", t.Name, t.AvailableNow, smallestFitting.Name)
			}
		}
	}

	// If estimatedTokens exceeds all tiers, pick the largest available tier and advise splitting.
	if smallestFitting == nil {
		for i := len(sorted) - 1; i >= 0; i-- {
			t := &sorted[i]
			if t.AvailableNow > 0 {
				return t.Name, t.MaxContext, fmt.Sprintf("subtask (%d tok) exceeds max tier %s (%d tok); split or compact to fit", estimatedTokens, t.Name, t.MaxContext)
			}
		}
		largest := sorted[len(sorted)-1]
		return largest.Name, largest.MaxContext, fmt.Sprintf("subtask (%d tok) exceeds all tiers and 0 slots are free; split below %d tok and wait", estimatedTokens, largest.MaxContext)
	}

	// A tier fits, but neither it nor any larger tier has available slots.
	// Check if a smaller tier has available slots so the caller can split to fit.
	for i := len(sorted) - 1; i >= 0; i-- {
		t := &sorted[i]
		if t.AvailableNow > 0 {
			return t.Name, t.MaxContext, fmt.Sprintf("%s tier has 0 slots; split subtask (%d tok) to fit %s (%d tok, %d slots free)", smallestFitting.Name, estimatedTokens, t.Name, t.MaxContext, t.AvailableNow)
		}
	}

	return smallestFitting.Name, smallestFitting.MaxContext, fmt.Sprintf("all tiers saturated (0 slots free); wait or compact %s subtask", smallestFitting.Name)
}
