package sessionrecovery

import (
	"path"
	"sort"
	"strings"
	"time"
)

// HostEvidenceRow is the closed host-resurrection evidence needed to join a dead
// guard row to Codex's authoritative state_5 thread namespace.
type HostEvidenceRow struct {
	Handle       string
	TraceID      string
	ResumeHandle string
	PID          int
	StartedAt    string
	CWD          string
	Command      []string
}

type HostCohortEntry struct {
	Handle    string
	PID       int
	StartedAt string
}

const (
	IdentityArgvExact   = "argv_exact_uuid"
	IdentityTraceLedger = "trace_ledger_uuid"
)

// MergeCodexHostCohort accounts for every exact handle+pid+start member of the
// pre-crash cohort. A Codex guard handle is never itself a thread identity. The
// only admissible joins are a full UUID already present in argv or a durable
// trace->UUID identity row. Every candidate UUID is finally verified against
// state_5-derived inventory; cwd and timestamps are never identity evidence.
func MergeCodexHostCohort(report InventoryReport, guards []HostEvidenceRow, cohort []HostCohortEntry, uuidByTrace map[string]string) InventoryReport {
	state := make(map[string]int)
	for i := range report.Sessions {
		if report.Sessions[i].Thread == nil || !isUUID(report.Sessions[i].Thread.ID) {
			continue
		}
		state[strings.ToLower(report.Sessions[i].Thread.ID)] = i
	}
	sort.SliceStable(guards, func(i, j int) bool { return guards[i].Handle < guards[j].Handle })
	sort.SliceStable(cohort, func(i, j int) bool {
		if cohort[i].Handle != cohort[j].Handle {
			return cohort[i].Handle < cohort[j].Handle
		}
		if cohort[i].PID != cohort[j].PID {
			return cohort[i].PID < cohort[j].PID
		}
		return cohort[i].StartedAt < cohort[j].StartedAt
	})
	for _, member := range cohort {
		var exact []HostEvidenceRow
		var sameHandle []HostEvidenceRow
		for _, guard := range guards {
			if guard.Handle != member.Handle {
				continue
			}
			sameHandle = append(sameHandle, guard)
			if guard.PID == member.PID && guard.StartedAt == member.StartedAt {
				exact = append(exact, guard)
			}
		}
		if len(exact) == 0 {
			guard := HostEvidenceRow{Handle: member.Handle, PID: member.PID, StartedAt: member.StartedAt}
			reason := "host_evidence_missing"
			if len(sameHandle) > 0 {
				guard.CWD, guard.Command = sameHandle[0].CWD, sameHandle[0].Command
				reason = "host_evidence_pid_start_mismatch"
			}
			report.Sessions = append(report.Sessions, blockedCohortRow(guard, guardProvider(guard.Command), reason, ""))
			continue
		}
		if len(exact) > 1 {
			report.Sessions = append(report.Sessions, blockedCohortRow(exact[0], guardProvider(exact[0].Command), "ambiguous_exact_host_evidence", ""))
			continue
		}
		guard := exact[0]
		provider := guardProvider(guard.Command)
		if provider != ProviderCodex && provider != ProviderClaude {
			report.Sessions = append(report.Sessions, blockedCohortRow(guard, provider, "exact_resume_provider_blocked:"+provider, ""))
			continue
		}
		id, provenance, reason := resolveHostIdentity(guard, uuidByTrace, provider)
		idx, verified := state[strings.ToLower(id)]
		if verified {
			stateProvider := normalizeProvider(report.Sessions[idx].Provider, report.Sessions[idx].Thread.Source)
			verified = stateProvider == provider
		}
		if id == "" || !verified {
			if reason == "" {
				reason = "exact_uuid_not_in_state_5"
			}
			report.Sessions = append(report.Sessions, blockedCohortRow(guard, provider, reason, provenance))
			continue
		}
		row := &report.Sessions[idx]
		row.Provider = provider
		appendHostHandle(row, guard.Handle)
		row.IdentityProvenance = provenance
		if row.Category == "" {
			row.Category = CategorySubstantive
		}
		if row.Action == "" {
			row.Action = ActionRecover
		}
		if row.Reason == "" {
			row.Reason = "host_crash_recovery"
		}
		if row.Thread.Source == "" {
			row.Thread.Source = "host_resurrection"
		}
		if row.Thread.CWD == "" {
			row.Thread.CWD = guard.CWD
		}
		if len(row.ProcessTrees) > 0 {
			row.Category = CategoryLive
			row.Action = ActionLeaveLive
			row.Reason = "live_process_tree"
		}
	}
	return report
}

func appendHostHandle(row *Session, handle string) {
	if row.HostHandle == "" {
		row.HostHandle = handle
	}
	for _, existing := range row.HostHandles {
		if existing == handle {
			return
		}
	}
	row.HostHandles = append(row.HostHandles, handle)
}

func blockedCohortRow(guard HostEvidenceRow, provider, reason, provenance string) Session {
	if provider == "" {
		provider = "unknown"
	}
	return Session{
		Thread:   &Thread{ID: guard.Handle, Source: "host_resurrection", CWD: guard.CWD, CreatedAt: guard.StartedAt},
		Provider: provider, Category: CategoryIdentityBlocked, Action: ActionResolveIdentity,
		Reason: reason, HostHandle: guard.Handle, HostHandles: []string{guard.Handle}, IdentityProvenance: provenance,
	}
}

func resolveHostIdentity(guard HostEvidenceRow, uuidByTrace map[string]string, provider string) (string, string, string) {
	exact := map[string]bool{}
	for i := 1; i < len(guard.Command); i++ {
		previous := strings.ToLower(strings.TrimSpace(guard.Command[i-1]))
		if (previous == "resume" || (provider == ProviderClaude && previous == "--resume")) && isUUID(guard.Command[i]) {
			exact[strings.ToLower(strings.TrimSpace(guard.Command[i]))] = true
		}
	}
	for _, arg := range guard.Command {
		if provider == ProviderClaude && strings.HasPrefix(strings.ToLower(strings.TrimSpace(arg)), "--resume=") {
			id := strings.TrimSpace(strings.SplitN(arg, "=", 2)[1])
			if isUUID(id) {
				exact[strings.ToLower(id)] = true
			}
		}
	}
	ledger := map[string]bool{}
	for _, key := range []string{guard.TraceID, guard.Handle, guard.ResumeHandle} {
		if id := strings.TrimSpace(uuidByTrace[strings.TrimSpace(key)]); isUUID(id) {
			ledger[strings.ToLower(id)] = true
		}
	}
	identities := map[string]bool{}
	for id := range exact {
		identities[id] = true
	}
	for id := range ledger {
		identities[id] = true
	}
	if len(identities) == 1 {
		for id := range identities {
			if exact[id] {
				return id, IdentityArgvExact, ""
			}
			return id, IdentityTraceLedger, ""
		}
	}
	if len(identities) > 1 {
		return "", "", "ambiguous_exact_identity"
	}
	return "", "", "exact_identity_missing"
}

func guardProvider(command []string) string {
	for _, arg := range command {
		if provider := providerName(arg); provider != "" {
			return provider
		}
	}
	return "unknown"
}

func normalizedCWD(value string) string {
	value = strings.ReplaceAll(strings.TrimSpace(value), `\`, "/")
	lower := strings.ToLower(value)
	switch {
	case strings.HasPrefix(lower, "//?/unc/"):
		value = "//" + value[len("//?/unc/"):]
	case strings.HasPrefix(lower, "//?/"):
		value = value[len("//?/"):]
	}
	if value == "" {
		return ""
	}
	return strings.ToLower(path.Clean(value))
}

func parseTime(value string) (time.Time, bool) {
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339} {
		if parsed, err := time.Parse(layout, strings.TrimSpace(value)); err == nil {
			return parsed, true
		}
	}
	return time.Time{}, false
}

func isUUID(value string) bool {
	value = strings.TrimSpace(value)
	if len(value) != 36 {
		return false
	}
	for i, c := range value {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return false
			}
			continue
		}
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}
