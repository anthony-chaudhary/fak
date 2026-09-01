package workerworktree

import (
	"sort"
	"strconv"
	"strings"
)

const (
	CapacityAdvisorySchema = "fak-worker-worktree-capacity/1"
	// AdvisoryCapacitySetpoint is the fail-open operator checkpoint for sanctioned
	// worker worktrees. It is deliberately not an admission limit: prepare remains
	// available when the census is unavailable or growth has not been justified.
	AdvisoryCapacitySetpoint = 50
)

type CapacityStatus string

const (
	CapacityWithinSetpoint      CapacityStatus = "WITHIN_SETPOINT"
	CapacityJustificationNeeded CapacityStatus = "JUSTIFICATION_REQUESTED"
	CapacityGrowthJustified     CapacityStatus = "GROWTH_JUSTIFIED"
	CapacityInventoryUnknown    CapacityStatus = "INVENTORY_UNKNOWN"
)

type CapacityCensus struct {
	Known bool
	Paths []string
}

type RetainedTree struct {
	Path         string
	ColdReapable bool
	OwnerDead    bool
	Clean        bool
}

type ContractionRecommendation struct {
	Path   string `json:"path"`
	Basis  string `json:"basis"`
	Action string `json:"action"`
}

// CapacityAdvisory is deterministic lifecycle evidence for a fail-open capacity
// decision. CurrentCount is the observed census; ProspectiveCount includes the
// operation being considered (and equals CurrentCount for list).
type CapacityAdvisory struct {
	Schema                     string                      `json:"schema"`
	Setpoint                   int                         `json:"setpoint"`
	CurrentCount               int                         `json:"current_count"`
	ProspectiveCount           int                         `json:"prospective_count"`
	InventoryKnown             bool                        `json:"inventory_known"`
	AboveSetpoint              bool                        `json:"above_setpoint"`
	Allowed                    bool                        `json:"allowed"`
	Status                     CapacityStatus              `json:"status"`
	Reason                     string                      `json:"reason"`
	Message                    string                      `json:"message"`
	ContractionRecommendations []ContractionRecommendation `json:"contraction_recommendations"`
}

// CapacityCensusFor enumerates only sanctioned worker worktrees while retaining
// whether git supplied a trustworthy inventory. Count historically returns
// (0,nil) on a git error, which is correct for its compatibility contract but
// insufficient for a fail-open warning that must distinguish unknown from empty.
func CapacityCensusFor(root string, git GitRunner) CapacityCensus {
	if git == nil {
		git = defaultGit
	}
	rc, out := run(git, root, []string{"worktree", "list", "--porcelain"})
	if rc != 0 {
		return CapacityCensus{Known: false, Paths: []string{}}
	}
	paths := make([]string, 0)
	for _, path := range parseWorktreePaths(out) {
		if IsWorkerWorktree(path) {
			paths = append(paths, path)
		}
	}
	sortCapacityPaths(paths)
	return CapacityCensus{Known: true, Paths: paths}
}

// AssessCapacity reports an advisory only. Allowed is always true, including
// unknown inventories and unjustified growth, so capacity telemetry can never
// wedge worker preparation.
func AssessCapacity(currentCount, prospectiveCount int, inventoryKnown bool, reason string, trees []RetainedTree) CapacityAdvisory {
	if currentCount < 0 {
		currentCount = 0
	}
	if prospectiveCount < currentCount {
		prospectiveCount = currentCount
	}
	reason = strings.Join(strings.Fields(reason), " ")
	out := CapacityAdvisory{
		Schema:                     CapacityAdvisorySchema,
		Setpoint:                   AdvisoryCapacitySetpoint,
		CurrentCount:               currentCount,
		ProspectiveCount:           prospectiveCount,
		InventoryKnown:             inventoryKnown,
		Allowed:                    true,
		Reason:                     reason,
		ContractionRecommendations: []ContractionRecommendation{},
	}
	if !inventoryKnown {
		out.Status = CapacityInventoryUnknown
		out.Message = "worker worktree inventory is unknown; capacity advisory failed open and the operation remains allowed"
		return out
	}

	out.AboveSetpoint = prospectiveCount > AdvisoryCapacitySetpoint
	if !out.AboveSetpoint {
		out.Status = CapacityWithinSetpoint
		out.Message = capacityWithinMessage(prospectiveCount)
		return out
	}

	out.ContractionRecommendations = safeContractionRecommendations(trees)
	if reason == "" {
		out.Status = CapacityJustificationNeeded
		out.Message = capacityJustificationMessage(prospectiveCount)
		return out
	}
	out.Status = CapacityGrowthJustified
	out.Message = capacityJustifiedMessage(prospectiveCount, reason)
	return out
}

func capacityWithinMessage(count int) string {
	return "worker worktree capacity " + strconv.Itoa(count) + " is at or below advisory setpoint " + strconv.Itoa(AdvisoryCapacitySetpoint)
}

func capacityJustificationMessage(count int) string {
	return "worker worktree capacity " + strconv.Itoa(count) + " exceeds advisory setpoint " + strconv.Itoa(AdvisoryCapacitySetpoint) + "; provide --capacity-reason to justify growth; the operation remains allowed"
}

func capacityJustifiedMessage(count int, reason string) string {
	return "worker worktree capacity " + strconv.Itoa(count) + " exceeds advisory setpoint " + strconv.Itoa(AdvisoryCapacitySetpoint) + " with recorded justification: " + reason + "; the operation remains allowed"
}

func safeContractionRecommendations(trees []RetainedTree) []ContractionRecommendation {
	out := make([]ContractionRecommendation, 0)
	seen := map[string]bool{}
	for _, tree := range trees {
		path := strings.TrimSpace(tree.Path)
		if path == "" || !tree.Clean {
			continue
		}
		recommendation := ContractionRecommendation{Path: path}
		switch {
		case tree.ColdReapable:
			recommendation.Basis = "COLD_REAPABLE"
			recommendation.Action = "fak worktree worker reap --all-cold"
		case tree.OwnerDead:
			recommendation.Basis = "OWNER_DEAD_CLEAN"
			recommendation.Action = "fak worktree worker gc --dry-run"
		default:
			continue
		}
		key := recommendation.Path + "\x00" + recommendation.Basis
		if !seen[key] {
			seen[key] = true
			out = append(out, recommendation)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		left := strings.ToLower(strings.ReplaceAll(out[i].Path, "\\", "/"))
		right := strings.ToLower(strings.ReplaceAll(out[j].Path, "\\", "/"))
		if left == right {
			if out[i].Path == out[j].Path {
				return out[i].Basis < out[j].Basis
			}
			return out[i].Path < out[j].Path
		}
		return left < right
	})
	return out
}

func sortCapacityPaths(paths []string) {
	sort.Slice(paths, func(i, j int) bool {
		left := strings.ToLower(strings.ReplaceAll(paths[i], "\\", "/"))
		right := strings.ToLower(strings.ReplaceAll(paths[j], "\\", "/"))
		if left == right {
			return paths[i] < paths[j]
		}
		return left < right
	})
}
