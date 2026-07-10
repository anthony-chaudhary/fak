package fleetpane

import (
	"fmt"
	"strings"
	"time"
)

func PaneText(status StatusDoc) string {
	var lines []string
	lines = append(lines, "FLEET CONTROL PANE @ "+status.GeneratedUTC)
	lines = append(lines, fmt.Sprintf("verdict: %s  host=%s", status.Verdict, status.Machine["host"]))
	if available, _ := status.Git["available"].(bool); available {
		counts := mapInt(status.Git["counts"])
		lines = append(lines, fmt.Sprintf("git: %s %s  dirty=%d M=%d D=%d ?=%d",
			status.Git["branch"], status.Git["head"], intValueDefault(status.Git["dirty_total"], 0),
			counts["modified"], counts["deleted"], counts["untracked"]))
	} else {
		lines = append(lines, "git: unavailable ("+stringValue(status.Git["reason"])+")")
	}
	if exists, _ := status.Registry["exists"].(bool); exists {
		lines = append(lines, fmt.Sprintf("sessions: %d age=%vm categories=%s",
			intValueDefault(status.Registry["sessions"], 0), status.Registry["age_min"], CompactCounts(mapValue(status.Registry["categories"]))))
		lines = append(lines, "actions: "+CompactCounts(mapValue(status.Registry["actions"])))
	} else {
		lines = append(lines, "sessions: registry missing")
	}
	if available, _ := status.Supervisor["available"].(bool); available {
		payload := mapValue(status.Supervisor["payload"])
		lines = append(lines, fmt.Sprintf("supervisor: verdict=%v", payload["verdict"]))
	} else {
		lines = append(lines, "supervisor: unavailable ("+stringValue(status.Supervisor["reason"])+")")
	}
	if states, ok := status.Loops["states"]; ok {
		lines = append(lines, "loops: "+CompactCounts(mapValue(states)))
	}
	lines = append(lines, WorkerHealthText(status.WorkerHealth))
	if len(status.Actions) > 0 {
		lines = append(lines, "recommended:")
		for _, action := range status.Actions {
			lines = append(lines, "  - "+action)
		}
	}
	return strings.Join(lines, "\n")
}

func LoopListText(doc LoopListDoc) string {
	lines := []string{fmt.Sprintf("loops: count=%d enabled=%d disabled=%d blocked=%d", doc.Count, doc.Enabled, doc.Disabled, doc.Blocked)}
	for _, loop := range doc.Loops {
		state := "disabled"
		if loop.Enabled && loop.Ready {
			state = "ready"
		} else if loop.Enabled {
			state = "blocked"
		}
		lines = append(lines, fmt.Sprintf("- %s: %s source=%s status=%s", loop.Name, state, loop.Source, loop.StatusReady.Detail))
		if loop.AutoRecover {
			lines = append(lines, fmt.Sprintf("  recover: %s", loop.RecoverReady.Detail))
		}
	}
	if len(doc.Commands) > 0 {
		lines = append(lines, "next:")
		for _, command := range doc.Commands {
			lines = append(lines, "  "+command)
		}
	}
	return strings.Join(lines, "\n")
}

func LoopCheckText(doc LoopCheckDoc) string {
	if !doc.OK && doc.Reason != "" {
		return fmt.Sprintf("loop[%s]: %s (%s)", doc.Loop, doc.Verdict, doc.Reason)
	}
	detail := doc.Check.Detail
	if detail == "" {
		detail = doc.Check.Reason
	}
	return strings.TrimSpace(fmt.Sprintf("loop[%s]: verdict=%s state=%s %s", doc.Loop, doc.Verdict, doc.Check.State, detail))
}

func LoopAuditText(doc LoopAuditDoc) string {
	lines := []string{fmt.Sprintf("loop-audit: ok=%v healthy=%d action=%d broken=%d total=%d", doc.OK, doc.Counts["healthy"], doc.Counts["action"], doc.Counts["broken"], doc.Counts["total"])}
	for _, row := range doc.Loops {
		suffix := ""
		if row.Detail != "" {
			suffix = " " + row.Detail
		}
		lines = append(lines, fmt.Sprintf("- %s: %s/%s%s", row.Name, row.Bucket, row.State, suffix))
	}
	if len(doc.Missing) > 0 {
		lines = append(lines, "missing: "+strings.Join(doc.Missing, ", "))
	}
	return strings.Join(lines, "\n")
}

const fleetSnapshotStaleMargin = 2.0 // poller presumed dead at 2x the machine-stale horizon

// fleetSnapshotStaleFloorMin is the fallback stale horizon when a doc carries no StaleMin.
const fleetSnapshotStaleFloorMin = 15.0

// FleetSnapshotAge reports the aggregate snapshot's age at now and whether it is
// stale (poller presumed dead). ok is false when GeneratedUTC is empty or unparseable,
// in which case callers omit the age and do not flag stale. Staleness is a pure
// function of the snapshot's age — a quiet-but-fresh fleet never reads stale.
func FleetSnapshotAge(doc FleetDoc, now time.Time) (age time.Duration, stale bool, ok bool) {
	if doc.GeneratedUTC == "" {
		return 0, false, false
	}
	captured, err := time.Parse(time.RFC3339, doc.GeneratedUTC)
	if err != nil {
		return 0, false, false
	}
	age = now.Sub(captured)
	if age < 0 {
		age = 0
	}
	staleMin := doc.StaleMin
	if staleMin <= 0 {
		staleMin = fleetSnapshotStaleFloorMin
	}
	threshold := time.Duration(staleMin * fleetSnapshotStaleMargin * float64(time.Minute))
	return age, age > threshold, true
}

// humanizeAge renders a duration in a compact truncated form: "12s", "6m", "2h".
func humanizeAge(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	}
	return fmt.Sprintf("%dh", int(d.Hours()))
}

func FleetText(doc FleetDoc) string { return FleetTextAt(doc, time.Now().UTC()) }

func FleetTextAt(doc FleetDoc, now time.Time) string {
	age, stale, ok := FleetSnapshotAge(doc, now)
	header := "FLEET CONTROL PANE AGGREGATE @ " + doc.GeneratedUTC
	if ok {
		header += " (updated " + humanizeAge(age) + " ago)"
	}
	verdict := doc.Verdict
	if stale {
		verdict = fmt.Sprintf("STALE_SNAPSHOT (frozen: %s)", doc.Verdict)
	}
	lines := []string{
		header,
		fmt.Sprintf("verdict: %s  machines=%d  states=%s", verdict, len(doc.Machines), CompactCounts(intMapAny(doc.States))),
		"versions: " + CompactCounts(intMapAny(doc.Versions)),
		fmt.Sprintf("totals: sessions=%d actions=%d auth_sessions=%d auto_resume=%d loops=%d dirty=%d version=%d",
			doc.Totals["sessions"], doc.Totals["actions"], doc.Totals["auth_blocked"], doc.Totals["auto_resume"], doc.Totals["loop_actions"], doc.Totals["dirty_paths"], doc.Totals["version_mismatches"]),
	}
	for _, machine := range doc.Machines {
		accounts := mapValue(machine["accounts"])
		loops := mapValue(machine["loops"])
		gitStatus := mapValue(machine["git"])
		lines = append(lines, fmt.Sprintf("- %s: state=%s age=%vm ver=%s sessions=%d actions=%d acct=%d/%d loops=%d/%d git=%v sync=%v dirty=%d",
			machine["id"], machine["state"], machine["age_min"], stringValueDefault(machine["app_version"], "unknown"),
			intValueDefault(machine["sessions"], 0), intValueDefault(machine["actions_count"], 0),
			intValueDefault(accounts["available"], 0), intValueDefault(accounts["total"], 0),
			intValueDefault(loops["action"], 0), intValueDefault(loops["count"], 0),
			gitStatus["branch"], stringValueDefault(gitStatus["safe_ff_state"], "unknown"), intValueDefault(gitStatus["dirty_total"], 0)))
	}
	if len(doc.Machines) == 0 {
		lines = append(lines, "no machine snapshots found in "+doc.MachineDir)
	}
	return strings.Join(lines, "\n")
}
