package wipinventory

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ReconcileSchema is the schema identifier for WIP inventory reconciliation reports.
const ReconcileSchema = "fak-wip-reconcile/1"

// TransitionType is an alias for TransitionKind to match the reconciliation contract.
type TransitionType = TransitionKind

// RawSurfacesReport aggregates record counts across all physical surfaces.
type RawSurfacesReport struct {
	Issues           int `json:"issues"`
	Sessions         int `json:"sessions"`
	Checkpoints      int `json:"checkpoints"`
	LaneLeases       int `json:"lane_leases"`
	ManagedWorktrees int `json:"managed_worktrees"`
	UnlinkedFiles    int `json:"unlinked_files"`
	UnlinkedDirs     int `json:"unlinked_dirs"`
}

// LogicalUnitsReport summarizes logical WIP unit conservation.
type LogicalUnitsReport struct {
	Total                int `json:"total"`
	Active               int `json:"active"`
	Terminal             int `json:"terminal"`
	SingleRepresentation int `json:"single_representation"`
	SplitRepresentations int `json:"split_representations"`
}

// UnresolvedJoinDebtItem categorizes one piece of unresolved join debt.
type UnresolvedJoinDebtItem struct {
	Reason      string   `json:"reason"`
	Surface     string   `json:"surface"`
	Identifiers []string `json:"identifiers"`
	Details     string   `json:"details"`
	Sample      string   `json:"sample"`
}

// ReconciliationReport is the top-level reconciled WIP report.
type ReconciliationReport struct {
	Schema             string                   `json:"schema"`
	Repo               string                   `json:"repo"`
	HEAD               string                   `json:"head"`
	ObservedAt         time.Time                `json:"observed_at"`
	RawSurfaces        RawSurfacesReport        `json:"raw_surfaces"`
	LogicalUnits       LogicalUnitsReport       `json:"logical_units"`
	TransitionCounts   map[TransitionType]int   `json:"transition_counts"`
	UnresolvedJoinDebt []UnresolvedJoinDebtItem `json:"unresolved_join_debt"`
	SourceErrors       []string                 `json:"source_errors"`
}

// ManagedWorktreeBinding implements ManagedWorktreeWIPBinding for inventory joins.
type ManagedWorktreeBinding struct {
	WorktreeID string `json:"worktree_id"`
	WIPUnitID  string `json:"wip_unit_id"`
	WorkerID   string `json:"worker_id"`
	Lane       string `json:"lane"`
	LeaseID    string `json:"lease_id"`
	Registered bool   `json:"registered"`
}

func (m ManagedWorktreeBinding) WIPWorktreeID() string { return m.WorktreeID }
func (m ManagedWorktreeBinding) WIPUnit() string       { return m.WIPUnitID }
func (m ManagedWorktreeBinding) WIPWorkerID() string   { return m.WorkerID }
func (m ManagedWorktreeBinding) WIPLane() string       { return m.Lane }
func (m ManagedWorktreeBinding) WIPLeaseID() string    { return m.LeaseID }
func (m ManagedWorktreeBinding) WIPRegistered() bool   { return m.Registered }

// InventoryInputs contains raw surface artifacts supplied to ReconcileInventory.
type InventoryInputs struct {
	Histories     []History
	Transitions   []Transition
	Bindings      ExecutionBindingReport
	Checkpoints   []CheckpointWIPBinding
	Leases        []LiveLaneLease
	Worktrees     []ManagedWorktreeBinding
	Issues        []IssueReference
	Sessions      []string
	UnlinkedFiles []string
	UnlinkedDirs  []string
	SourceErrors  []string
	Now           time.Time
	HEAD          string
	Runner        Runner
}

// ReconcileInventory reconciles raw physical artifacts across issues, sessions,
// checkpoints, lane leases, and worktrees into conserved logical units and
// categorizes unresolved debt without mutating repository state.
func ReconcileInventory(ctx context.Context, repo string, inputs InventoryInputs) (*ReconciliationReport, error) {
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}

	if repo == "" {
		repo = "."
	}
	cleanRepo := filepath.ToSlash(filepath.Clean(repo))

	now := inputs.Now.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}

	head := inputs.HEAD
	if head == "" {
		runner := inputs.Runner
		if runner == nil {
			runner = GitRunner{}
		}
		if out, err := runner.Run(cleanRepo, "rev-parse", "HEAD"); err == nil {
			head = strings.TrimSpace(string(out))
		}
	}

	histories := append([]History(nil), inputs.Histories...)
	if len(histories) == 0 && len(inputs.Transitions) > 0 {
		histories = []History{{Schema: WIPUnitSchema, Transitions: inputs.Transitions}}
	}

	sourceErrors := append([]string(nil), inputs.SourceErrors...)

	// 1. Join issue sessions.
	issueJoin, issueErr := JoinIssueSessions(histories, inputs.Bindings)
	if issueErr != nil {
		sourceErrors = append(sourceErrors, "issue_session_join: "+issueErr.Error())
	}

	// 2. Join checkpoints, leases, and managed worktrees.
	cpJoin := JoinCheckpointWorktrees(inputs.Checkpoints, inputs.Leases, inputs.Worktrees)

	// 3. Transition accounting across histories.
	transitionCounts := make(map[TransitionType]int)
	var accountingDebts []AccountingDebt
	accountedUnitStates := make(map[WIPUnitID]AccountedUnitState)

	for _, history := range histories {
		for _, tr := range history.Transitions {
			transitionCounts[tr.Kind]++
		}
		acc := AccountHistory(history)
		accountingDebts = append(accountingDebts, acc.Debt...)
		for _, u := range acc.Units {
			accountedUnitStates[u.ID] = u.State
		}
	}

	// 4. Raw surface counts.
	issueSet := make(map[string]bool)
	for _, issue := range inputs.Issues {
		key := issueKey(issue)
		if key != "" && key != "#0" {
			issueSet[key] = true
		}
	}
	for _, binding := range inputs.Bindings.Bindings {
		if binding.Issue != nil {
			key := issueKey(IssueReference{Repository: binding.Issue.Repository, Number: binding.Issue.Number})
			if key != "" && key != "#0" {
				issueSet[key] = true
			}
		}
	}
	for _, history := range histories {
		for _, tr := range history.Transitions {
			for _, ref := range tr.References {
				if ref.Kind == SurfaceIssue && ref.Issue != nil {
					key := issueKey(*ref.Issue)
					if key != "" && key != "#0" {
						issueSet[key] = true
					}
				}
			}
		}
	}
	rawIssues := len(issueSet)
	if len(inputs.Issues) > rawIssues {
		rawIssues = len(inputs.Issues)
	}

	sessionSet := make(map[string]bool)
	for _, s := range inputs.Sessions {
		s = strings.TrimSpace(s)
		if s != "" {
			sessionSet[s] = true
		}
	}
	for _, binding := range inputs.Bindings.Bindings {
		for _, sid := range binding.SessionIDs {
			sid = strings.TrimSpace(sid)
			if sid != "" {
				sessionSet[sid] = true
			}
		}
		if len(binding.SessionIDs) == 0 && strings.TrimSpace(binding.RootRegistrationID) != "" {
			sessionSet[strings.TrimSpace(binding.RootRegistrationID)] = true
		}
	}
	for _, cp := range inputs.Checkpoints {
		sid := strings.TrimSpace(cp.SessionID)
		if sid != "" {
			sessionSet[sid] = true
		}
	}
	for _, lease := range inputs.Leases {
		sid := strings.TrimSpace(lease.SessionID)
		if sid != "" {
			sessionSet[sid] = true
		}
	}
	rawSessions := len(sessionSet)

	rawSurfaces := RawSurfacesReport{
		Issues:           rawIssues,
		Sessions:         rawSessions,
		Checkpoints:      len(inputs.Checkpoints),
		LaneLeases:       len(inputs.Leases),
		ManagedWorktrees: len(inputs.Worktrees),
		UnlinkedFiles:    len(inputs.UnlinkedFiles),
		UnlinkedDirs:     len(inputs.UnlinkedDirs),
	}

	// 5. Logical units calculation.
	logicalUnitSet := make(map[WIPUnitID]bool)
	for unitID := range accountedUnitStates {
		if unitID.Validate() == nil {
			logicalUnitSet[unitID] = true
		}
	}
	for _, u := range issueJoin.Units {
		if u.UnitID.Validate() == nil {
			logicalUnitSet[u.UnitID] = true
		}
	}
	for _, cp := range inputs.Checkpoints {
		if cp.WIPUnitID.Validate() == nil {
			logicalUnitSet[cp.WIPUnitID] = true
		}
	}
	for _, lease := range inputs.Leases {
		if lease.WIPUnitID.Validate() == nil {
			logicalUnitSet[lease.WIPUnitID] = true
		}
	}
	for _, wt := range inputs.Worktrees {
		if id, err := ParseWIPUnitID(wt.WIPUnit()); err == nil {
			logicalUnitSet[id] = true
		}
	}

	activeUnits := 0
	terminalUnits := 0
	singleRepUnits := 0
	splitRepUnits := 0

	for id := range logicalUnitSet {
		state, ok := accountedUnitStates[id]
		if ok && (state == AccountedUnitLanded || state == AccountedUnitAbandoned || state == AccountedUnitSuperseded) {
			terminalUnits++
		} else {
			activeUnits++
		}

		repCount := 0

		hasIssue := false
		for _, u := range issueJoin.Units {
			if u.UnitID == id && u.Issue.Number > 0 {
				hasIssue = true
				break
			}
		}
		if hasIssue {
			repCount++
		}

		hasSession := false
		for _, u := range issueJoin.Units {
			if u.UnitID == id && (u.RootRegistrationID != "" || len(u.SessionIDs) > 0) {
				hasSession = true
				break
			}
		}
		if !hasSession {
			for _, cp := range inputs.Checkpoints {
				if cp.WIPUnitID == id && cp.SessionID != "" {
					hasSession = true
					break
				}
			}
		}
		if !hasSession {
			for _, lease := range inputs.Leases {
				if lease.WIPUnitID == id && lease.SessionID != "" {
					hasSession = true
					break
				}
			}
		}
		if hasSession {
			repCount++
		}

		hasCheckpoint := false
		for _, cp := range inputs.Checkpoints {
			if cp.WIPUnitID == id && cp.CheckpointID != "" {
				hasCheckpoint = true
				break
			}
		}
		if hasCheckpoint {
			repCount++
		}

		hasLease := false
		for _, lease := range inputs.Leases {
			if lease.WIPUnitID == id && lease.LeaseID != "" {
				hasLease = true
				break
			}
		}
		if hasLease {
			repCount++
		}

		hasWorktree := false
		for _, wt := range inputs.Worktrees {
			if wt.WIPUnit() == string(id) && wt.WIPWorktreeID() != "" {
				hasWorktree = true
				break
			}
		}
		if hasWorktree {
			repCount++
		}

		if repCount == 0 {
			repCount = 1
		}

		if repCount > 1 {
			splitRepUnits++
		} else {
			singleRepUnits++
		}
	}

	logicalUnits := LogicalUnitsReport{
		Total:                len(logicalUnitSet),
		Active:               activeUnits,
		Terminal:             terminalUnits,
		SingleRepresentation: singleRepUnits,
		SplitRepresentations: splitRepUnits,
	}

	// 6. Unresolved join debt.
	debtItems := make([]UnresolvedJoinDebtItem, 0)

	// Debt from unjoined sessions in issue/session join.
	for _, d := range issueJoin.Debt {
		var ids []string
		if d.RootRegistrationID != "" {
			ids = append(ids, d.RootRegistrationID)
		}
		ids = append(ids, d.RegistrationIDs...)
		ids = append(ids, d.SessionIDs...)
		ids = append(ids, d.AttemptIDs...)
		ids = sortedUnique(ids)
		sample := ""
		if len(ids) > 0 {
			sample = ids[0]
		}
		details := strings.Join(d.Details, "; ")
		if details == "" {
			details = fmt.Sprintf("session with status %s has no matching WIP unit binding", d.Status)
		}
		debtItems = append(debtItems, UnresolvedJoinDebtItem{
			Reason:      "unjoined_sessions",
			Surface:     "dispatch_session",
			Identifiers: ids,
			Details:     details,
			Sample:      sample,
		})
	}

	// Debt from issues with unresolved binding status.
	for _, u := range issueJoin.Units {
		if u.Status == IssueSessionJoined {
			continue
		}
		reason := "unjoined_issue"
		if u.Status == IssueSessionAmbiguous {
			reason = "ambiguous_issue_binding"
		} else if u.Status == IssueSessionConflicting {
			reason = "conflicting_parentage"
		} else if u.Status == IssueSessionStale {
			reason = "stale_session_binding"
		}
		key := issueKey(u.Issue)
		var ids []string
		if key != "" {
			ids = append(ids, key)
		}
		if u.UnitID != "" {
			ids = append(ids, string(u.UnitID))
		}
		ids = sortedUnique(ids)
		sample := ""
		if len(ids) > 0 {
			sample = ids[0]
		}
		details := strings.Join(u.Details, "; ")
		if details == "" {
			details = fmt.Sprintf("issue has unresolved status %s", u.Status)
		}
		debtItems = append(debtItems, UnresolvedJoinDebtItem{
			Reason:      reason,
			Surface:     "issue",
			Identifiers: ids,
			Details:     details,
			Sample:      sample,
		})
	}

	// Debt from checkpoints, leases, and worktrees.
	for _, d := range cpJoin.Debt {
		var reason string
		var surface string

		switch d.Kind {
		case DebtMissingRegistration:
			if strings.HasPrefix(d.Artifact, "checkpoint:") {
				reason = "orphaned_checkpoints"
				surface = "checkpoint"
			} else if strings.HasPrefix(d.Artifact, "worktree:") {
				reason = "unlinked_worktrees"
				surface = "managed_worktree"
			} else if strings.HasPrefix(d.Artifact, "lease:") {
				reason = "missing_registration"
				surface = "lane_lease"
			} else {
				reason = "missing_registration"
				surface = "artifact"
			}
		case DebtStaleLease:
			reason = "stale_leases"
			if strings.HasPrefix(d.Artifact, "checkpoint:") {
				surface = "checkpoint"
			} else if strings.HasPrefix(d.Artifact, "worktree:") {
				surface = "managed_worktree"
			} else {
				surface = "lane_lease"
			}
		case DebtConflictingOwners:
			reason = "conflicting_parentage"
			if strings.HasPrefix(d.Artifact, "checkpoint:") {
				surface = "checkpoint"
			} else if strings.HasPrefix(d.Artifact, "worktree:") {
				surface = "managed_worktree"
			} else {
				surface = "lane_lease"
			}
		case DebtSharedSnapshot:
			reason = "shared_snapshot"
			surface = "checkpoint"
		case DebtForeignPaths:
			reason = "foreign_paths"
			surface = "checkpoint"
		default:
			reason = string(d.Kind)
			surface = "artifact"
		}

		var ids []string
		if d.Artifact != "" {
			ids = append(ids, d.Artifact)
		}
		for _, id := range d.WIPUnitIDs {
			ids = append(ids, string(id))
		}
		ids = sortedUnique(ids)
		sample := ""
		if len(ids) > 0 {
			sample = ids[0]
		}
		debtItems = append(debtItems, UnresolvedJoinDebtItem{
			Reason:      reason,
			Surface:     surface,
			Identifiers: ids,
			Details:     d.Detail,
			Sample:      sample,
		})
	}

	// Debt from accounting transitions.
	for _, d := range accountingDebts {
		reason := string(d.Kind)
		if d.Kind == AccountingDebtAmbiguousMultiParentOwner {
			reason = "conflicting_parentage"
		}
		var ids []string
		if d.TransitionIndex >= 0 {
			ids = append(ids, fmt.Sprintf("transition:%d", d.TransitionIndex))
		}
		if d.TransitionKind != "" {
			ids = append(ids, string(d.TransitionKind))
		}
		ids = sortedUnique(ids)
		sample := ""
		if len(ids) > 0 {
			sample = ids[0]
		}
		debtItems = append(debtItems, UnresolvedJoinDebtItem{
			Reason:      reason,
			Surface:     "history",
			Identifiers: ids,
			Details:     d.Detail,
			Sample:      sample,
		})
	}

	// Debt from unlinked files.
	if len(inputs.UnlinkedFiles) > 0 {
		files := append([]string(nil), inputs.UnlinkedFiles...)
		sort.Strings(files)
		debtItems = append(debtItems, UnresolvedJoinDebtItem{
			Reason:      "unlinked_files",
			Surface:     "unlinked_files",
			Identifiers: files,
			Details:     fmt.Sprintf("%d unlinked file(s) found", len(files)),
			Sample:      files[0],
		})
	}

	// Debt from unlinked dirs.
	if len(inputs.UnlinkedDirs) > 0 {
		dirs := append([]string(nil), inputs.UnlinkedDirs...)
		sort.Strings(dirs)
		debtItems = append(debtItems, UnresolvedJoinDebtItem{
			Reason:      "unlinked_dirs",
			Surface:     "unlinked_dirs",
			Identifiers: dirs,
			Details:     fmt.Sprintf("%d unlinked directory(ies) found", len(dirs)),
			Sample:      dirs[0],
		})
	}

	// Deterministic ordering of debt items.
	sort.Slice(debtItems, func(i, j int) bool {
		if debtItems[i].Reason != debtItems[j].Reason {
			return debtItems[i].Reason < debtItems[j].Reason
		}
		if debtItems[i].Surface != debtItems[j].Surface {
			return debtItems[i].Surface < debtItems[j].Surface
		}
		if debtItems[i].Sample != debtItems[j].Sample {
			return debtItems[i].Sample < debtItems[j].Sample
		}
		return debtItems[i].Details < debtItems[j].Details
	})

	sort.Strings(sourceErrors)
	if sourceErrors == nil {
		sourceErrors = make([]string, 0)
	}

	report := &ReconciliationReport{
		Schema:             ReconcileSchema,
		Repo:               cleanRepo,
		HEAD:               head,
		ObservedAt:         now,
		RawSurfaces:        rawSurfaces,
		LogicalUnits:       logicalUnits,
		TransitionCounts:   transitionCounts,
		UnresolvedJoinDebt: debtItems,
		SourceErrors:       sourceErrors,
	}

	return report, nil
}

// JSON formats the reconciliation report as deterministic indented JSON.
func (r *ReconciliationReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// SummaryText returns compact human text representation of the reconciliation report.
func (r *ReconciliationReport) SummaryText() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Active logical WIP: %d (total: %d, terminal: %d, single: %d, split: %d)\n",
		r.LogicalUnits.Active,
		r.LogicalUnits.Total,
		r.LogicalUnits.Terminal,
		r.LogicalUnits.SingleRepresentation,
		r.LogicalUnits.SplitRepresentations,
	))
	sb.WriteString(fmt.Sprintf("Raw surfaces: issues=%d sessions=%d checkpoints=%d lane_leases=%d managed_worktrees=%d unlinked_files=%d unlinked_dirs=%d\n",
		r.RawSurfaces.Issues,
		r.RawSurfaces.Sessions,
		r.RawSurfaces.Checkpoints,
		r.RawSurfaces.LaneLeases,
		r.RawSurfaces.ManagedWorktrees,
		r.RawSurfaces.UnlinkedFiles,
		r.RawSurfaces.UnlinkedDirs,
	))

	var transitions []string
	for k, v := range r.TransitionCounts {
		if v > 0 {
			transitions = append(transitions, fmt.Sprintf("%s=%d", k, v))
		}
	}
	sort.Strings(transitions)
	if len(transitions) == 0 {
		sb.WriteString("Transitions: none\n")
	} else {
		sb.WriteString(fmt.Sprintf("Transitions: %s\n", strings.Join(transitions, " ")))
	}

	if len(r.UnresolvedJoinDebt) == 0 {
		sb.WriteString("Unresolved join debt: none\n")
	} else {
		sb.WriteString(fmt.Sprintf("Unresolved join debt (%d items):\n", len(r.UnresolvedJoinDebt)))
		for _, d := range r.UnresolvedJoinDebt {
			sampleStr := ""
			if d.Sample != "" {
				sampleStr = fmt.Sprintf(" sample=%s", d.Sample)
			}
			sb.WriteString(fmt.Sprintf("  - [%s] surface=%s%s: %s\n", d.Reason, d.Surface, sampleStr, d.Details))
		}
	}
	if len(r.SourceErrors) > 0 {
		sb.WriteString(fmt.Sprintf("Source errors (%d items):\n", len(r.SourceErrors)))
		for _, err := range r.SourceErrors {
			sb.WriteString(fmt.Sprintf("  - %s\n", err))
		}
	}
	return sb.String()
}
