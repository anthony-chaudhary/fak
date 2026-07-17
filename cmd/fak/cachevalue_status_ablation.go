package main

// Optional cache-value ablation digest -- summarizes the ablation report that
// attributes savings to individual cache levers.
// Split out of cachevalue_status.go along this concern seam so the cachevalue
// status dispatch surface stays steerable as new digests land (steerability
// dispatch_god_file). Behavior-preserving code motion -- same package, no logic change.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ablate"
)

func loadCachevalueAblationStatus(path string) (cachevalueAblationDigest, []cachevalueStatusRow) {
	digest := cachevalueAblationDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{ablationReportUnavailableRow(path, err.Error())}
	}
	var rep ablate.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{ablationReportUnavailableRow(path, "decode: "+err.Error())}
	}
	rows := rowsFromAblationReport(&rep, path)
	digest.Runs = len(rep.Runs)
	digest.DroppedArms = len(rep.Dropped)
	for _, drop := range rep.Dropped {
		if strings.TrimSpace(drop.Stage) != "" || drop.ExitCode != nil ||
			strings.TrimSpace(drop.StderrTail) != "" || strings.TrimSpace(drop.StdoutTail) != "" ||
			strings.TrimSpace(drop.ExpectedWorkloadHash) != "" || strings.TrimSpace(drop.ActualWorkloadHash) != "" ||
			drop.DurationSeconds > 0 {
			digest.DroppedWithDiagnostics++
		}
		switch strings.TrimSpace(drop.Stage) {
		case "workload_hash":
			digest.DroppedWorkloadMismatches++
		case "child_exit":
			digest.DroppedChildExits++
		}
	}
	for _, run := range rep.Runs {
		for _, effect := range run.CacheEffects {
			digest.CacheEffects++
			switch strings.ToLower(strings.TrimSpace(effect.Status)) {
			case "active":
				digest.ActiveEffects++
			case "unavailable":
				digest.UnavailableEffects++
			}
		}
	}
	switch {
	case len(rep.Runs) == 0:
		digest.Status = "missing"
	case digest.DroppedArms > 0 || digest.UnavailableEffects > 0:
		digest.Status = "partial"
	default:
		digest.Status = "measured"
	}
	return digest, rows
}

func ablationReportUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "diagnostics",
		Component:     "ablation_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "ablation_artifact",
		SessionImpact: "cache ablation evidence could not be folded, so subprocess/cache-effect attribution is incomplete",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak ablate --sweep " + strings.Join(cacheAblationFeatures(), ",") + " --json",
	}
}

func rowsFromAblationReport(rep *ablate.Report, path string) []cachevalueStatusRow {
	if rep == nil {
		return []cachevalueStatusRow{ablationReportUnavailableRow(path, "nil report")}
	}
	rows := []cachevalueStatusRow{{
		Plane:         "diagnostics",
		Component:     "ablation_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "WITNESSED",
		Status:        ablationReportStatus(rep),
		FailureDomain: "fak_diagnostics",
		SessionImpact: "ablation artifact is folded into this status so cache-effect and subprocess-arm holes are visible",
		Reason:        fmt.Sprintf("%d arm(s), %d dropped arm(s), workload=%s", len(rep.Runs), len(rep.Dropped), rep.WorkloadHash),
		NextAction:    ablationReportNextAction(rep),
	}}
	for _, run := range rep.Runs {
		for _, effect := range run.CacheEffects {
			rows = append(rows, rowFromAblationEffect(run.ArmID, effect))
		}
	}
	for _, drop := range rep.Dropped {
		rows = append(rows, rowFromDroppedAblationArm(drop))
	}
	return rows
}

func rowFromAblationEffect(armID string, effect ablate.CacheEffect) cachevalueStatusRow {
	status := strings.ToLower(strings.TrimSpace(effect.Status))
	if status == "" {
		status = "unknown"
	}
	component := "ablation_effect:" + nonEmpty(armID, "unknown_arm") + ":" + nonEmpty(effect.Feature, effect.Component)
	return cachevalueStatusRow{
		Plane:         nonEmpty(effect.Plane, "diagnostics"),
		Component:     component,
		Owner:         nonEmpty(effect.Owner, "unknown"),
		Dependency:    nonEmpty(effect.Dependency, "unknown"),
		Fidelity:      nonEmpty(effect.Fidelity, "unknown"),
		Evidence:      nonEmpty(effect.Evidence, "unknown"),
		Status:        status,
		FailureDomain: cachevalueFailureDomain(effect.Owner, effect.Component),
		SessionImpact: ablationEffectImpact(effect),
		Reason:        fmt.Sprintf("arm=%s feature=%s component=%s: %s", armID, effect.Feature, effect.Component, effect.Reason),
		NextAction:    ablationEffectNextAction(effect),
	}
}

func rowFromDroppedAblationArm(drop ablate.DroppedArm) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "diagnostics",
		Component:     "ablation_dropped_arm:" + nonEmpty(drop.ArmID, "unknown"),
		Owner:         "fak",
		Dependency:    "subprocess_reexec",
		Fidelity:      "diagnostic",
		Evidence:      "WITNESSED",
		Status:        "dropped",
		FailureDomain: droppedAblationFailureDomain(drop),
		SessionImpact: "one ablation child failed, so that cache/fak feature has no measured arm in this report",
		Reason:        droppedAblationReason(drop),
		NextAction:    droppedAblationNextAction(drop),
	}
}

func droppedAblationFailureDomain(drop ablate.DroppedArm) string {
	switch strings.TrimSpace(drop.Stage) {
	case "workload_hash":
		return "fak_diagnostics_workload_guard"
	case "child_exit":
		return "fak_diagnostics_subprocess_exit"
	case "decode_stdout":
		return "fak_diagnostics_subprocess_codec"
	default:
		return "fak_diagnostics_subprocess"
	}
}

func droppedAblationReason(drop ablate.DroppedArm) string {
	parts := []string{}
	if strings.TrimSpace(drop.Stage) != "" {
		parts = append(parts, "stage="+drop.Stage)
	}
	if drop.ExitCode != nil {
		parts = append(parts, fmt.Sprintf("exit=%d", *drop.ExitCode))
	}
	if drop.DurationSeconds > 0 {
		parts = append(parts, fmt.Sprintf("duration=%.3fs", drop.DurationSeconds))
	}
	if strings.TrimSpace(drop.StderrTail) != "" {
		parts = append(parts, "stderr="+drop.StderrTail)
	}
	if strings.TrimSpace(drop.StdoutTail) != "" {
		parts = append(parts, "stdout="+drop.StdoutTail)
	}
	if strings.TrimSpace(drop.ExpectedWorkloadHash) != "" || strings.TrimSpace(drop.ActualWorkloadHash) != "" {
		parts = append(parts, fmt.Sprintf("hash child=%s parent=%s", cachevalueEmptyDash(drop.ActualWorkloadHash), cachevalueEmptyDash(drop.ExpectedWorkloadHash)))
	}
	if strings.TrimSpace(drop.Reason) != "" {
		parts = append(parts, drop.Reason)
	}
	return strings.Join(parts, " ")
}

func droppedAblationNextAction(drop ablate.DroppedArm) string {
	switch strings.TrimSpace(drop.Stage) {
	case "workload_hash":
		return "inspect trace/session replay drift before comparing arm " + nonEmpty(drop.ArmID, "unknown")
	case "decode_stdout":
		return "rerun fak ablate for arm " + nonEmpty(drop.ArmID, "unknown") + " and inspect child stdout/stderr"
	case "child_exit":
		return "rerun fak ablate for arm " + nonEmpty(drop.ArmID, "unknown") + " after fixing the child process failure"
	default:
		return "rerun fak ablate for arm " + nonEmpty(drop.ArmID, "unknown")
	}
}

func ablationReportStatus(rep *ablate.Report) string {
	if rep == nil || len(rep.Runs) == 0 {
		return "missing"
	}
	if len(rep.Dropped) > 0 {
		return "partial"
	}
	return "measured"
}

func ablationReportNextAction(rep *ablate.Report) string {
	if rep == nil || len(rep.Runs) == 0 {
		return "fak ablate --sweep " + strings.Join(cacheAblationFeatures(), ",") + " --json"
	}
	if len(rep.Dropped) > 0 {
		return "rerun dropped ablation arms before treating the sweep as complete"
	}
	return ""
}

func ablationEffectImpact(effect ablate.CacheEffect) string {
	switch strings.ToLower(strings.TrimSpace(effect.Owner)) {
	case "provider":
		return "ablation effect belongs to provider prompt-cache behavior, not fak-native cache machinery"
	case "external":
		return "ablation effect depends on an external cache/compression sidecar"
	case "fak":
		return "ablation effect is fak-owned; compare this arm against the baseline before blaming provider cache"
	default:
		return "ablation effect owner is unknown"
	}
}

func ablationEffectNextAction(effect ablate.CacheEffect) string {
	switch strings.ToLower(strings.TrimSpace(effect.Status)) {
	case "unavailable":
		return "check dependency " + nonEmpty(effect.Dependency, "unknown")
	case "no-op":
		return "confirm the selected engine/component can exercise " + nonEmpty(effect.Feature, effect.Component)
	default:
		return ""
	}
}
