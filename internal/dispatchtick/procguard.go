package dispatchtick

import (
	"fmt"
	"sort"
	"strings"
)

const ProcGuardSchema = "fleet-proc-resource-guard/1"

const (
	ProcGuardDefaultMaxThreads      = 2000
	ProcGuardDefaultMaxHandles      = 0
	ProcGuardDefaultMaxWorkingSetMB = 0
)

var ProtectedProcessNames = map[string]bool{
	"system":             true,
	"idle":               true,
	"registry":           true,
	"smss":               true,
	"csrss":              true,
	"wininit":            true,
	"winlogon":           true,
	"services":           true,
	"lsass":              true,
	"fontdrvhost":        true,
	"dwm":                true,
	"sihost":             true,
	"memory compression": true,
	"init":               true,
	"systemd":            true,
	"launchd":            true,
	"kernel_task":        true,
	"kthreadd":           true,
}

type ProcInfo struct {
	PID          int
	Name         string
	Threads      *int
	Handles      *int
	WorkingSetMB *int
}

type ProcGuardThresholds struct {
	MaxThreads      int
	MaxHandles      int
	MaxWorkingSetMB int
}

type ProcGuardInput struct {
	Processes     []ProcInfo
	CollectError  string
	Thresholds    ProcGuardThresholds
	ProtectedPIDs []int
	AllowNames    []string
}

type ProcGuardFinding struct {
	PID          int      `json:"pid"`
	Name         string   `json:"name"`
	Threads      *int     `json:"threads,omitempty"`
	Handles      *int     `json:"handles,omitempty"`
	WorkingSetMB *int     `json:"ws_mb,omitempty"`
	Reasons      []string `json:"reasons"`
	Protected    bool     `json:"protected"`
}

type ProcGuardResult struct {
	Schema                 string              `json:"schema"`
	OK                     bool                `json:"ok"`
	Thresholds             ProcGuardThresholds `json:"thresholds"`
	Scanned                int                 `json:"scanned"`
	FlaggedCount           int                 `json:"flagged_count"`
	ActionableFlaggedCount int                 `json:"actionable_flagged_count"`
	Flagged                []ProcGuardFinding  `json:"flagged"`
	CollectError           string              `json:"collect_error,omitempty"`
	NextAction             string              `json:"next_action"`
}

func DefaultProcGuardThresholds() ProcGuardThresholds {
	return ProcGuardThresholds{
		MaxThreads:      ProcGuardDefaultMaxThreads,
		MaxHandles:      ProcGuardDefaultMaxHandles,
		MaxWorkingSetMB: ProcGuardDefaultMaxWorkingSetMB,
	}
}

func EvaluateProcGuard(in ProcGuardInput) ProcGuardResult {
	thresholds := in.Thresholds
	if thresholds == (ProcGuardThresholds{}) {
		thresholds = DefaultProcGuardThresholds()
	}
	protectedPIDs := map[int]bool{}
	for _, pid := range in.ProtectedPIDs {
		if pid > 0 {
			protectedPIDs[pid] = true
		}
	}
	allow := map[string]bool{}
	for _, name := range in.AllowNames {
		name = strings.ToLower(strings.TrimSpace(name))
		if name != "" {
			allow[name] = true
		}
	}

	flagged := []ProcGuardFinding{}
	for _, proc := range in.Processes {
		name := strings.TrimSpace(proc.Name)
		if allow[strings.ToLower(name)] {
			continue
		}
		reasons := []string{}
		if thresholds.MaxThreads > 0 && proc.Threads != nil && *proc.Threads > thresholds.MaxThreads {
			reasons = append(reasons, fmt.Sprintf("threads %d > %d", *proc.Threads, thresholds.MaxThreads))
		}
		if thresholds.MaxHandles > 0 && proc.Handles != nil && *proc.Handles > thresholds.MaxHandles {
			reasons = append(reasons, fmt.Sprintf("handles %d > %d", *proc.Handles, thresholds.MaxHandles))
		}
		if thresholds.MaxWorkingSetMB > 0 && proc.WorkingSetMB != nil && *proc.WorkingSetMB > thresholds.MaxWorkingSetMB {
			reasons = append(reasons, fmt.Sprintf("ws_mb %d > %d", *proc.WorkingSetMB, thresholds.MaxWorkingSetMB))
		}
		if len(reasons) == 0 {
			continue
		}
		protected := protectedPIDs[proc.PID] || ProtectedProcessNames[strings.ToLower(name)]
		flagged = append(flagged, ProcGuardFinding{
			PID:          proc.PID,
			Name:         name,
			Threads:      proc.Threads,
			Handles:      proc.Handles,
			WorkingSetMB: proc.WorkingSetMB,
			Reasons:      reasons,
			Protected:    protected,
		})
	}
	sort.Slice(flagged, func(i, j int) bool {
		ti, tj := ptrIntValue(flagged[i].Threads), ptrIntValue(flagged[j].Threads)
		if ti != tj {
			return ti > tj
		}
		return flagged[i].PID < flagged[j].PID
	})
	actionable := 0
	for _, row := range flagged {
		if !row.Protected {
			actionable++
		}
	}
	collectError := strings.TrimSpace(in.CollectError)
	ok := collectError == "" && actionable == 0
	return ProcGuardResult{
		Schema:                 ProcGuardSchema,
		OK:                     ok,
		Thresholds:             thresholds,
		Scanned:                len(in.Processes),
		FlaggedCount:           len(flagged),
		ActionableFlaggedCount: actionable,
		Flagged:                flagged,
		CollectError:           collectError,
		NextAction:             procGuardNextAction(flagged, collectError),
	}
}

func (r ProcGuardResult) ActionableNames() []string {
	return actionableFindingNames(r.Flagged)
}

// actionableFindingNames returns the sorted display names of the non-protected
// flagged findings, formatting each as "name(pid N)" when a PID is present.
func actionableFindingNames(flagged []ProcGuardFinding) []string {
	names := []string{}
	for _, row := range flagged {
		if row.Protected {
			continue
		}
		name := row.Name
		if row.PID > 0 {
			name = fmt.Sprintf("%s(pid %d)", name, row.PID)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func procGuardNextAction(flagged []ProcGuardFinding, collectError string) string {
	if strings.TrimSpace(collectError) != "" {
		return "process scan failed; rerun the guard and inspect collect_error"
	}
	names := actionableFindingNames(flagged)
	if len(names) == 0 {
		return "no runaway process; no action"
	}
	return "inspect or reap flagged runaway process(es): " + strings.Join(names, ", ")
}

func ptrIntValue(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}
