package repoguard

import "strings"

// ReasonForegroundPowerShellInventory identifies expensive host-wide inventory
// reads that are left unbounded in the foreground. On the Windows fleet these
// calls dominate the two-minute timeout residue measured in #4595.
const ReasonForegroundPowerShellInventory = "FOREGROUND_POWERSHELL_INVENTORY"

// ClassifyForegroundPowerShellInventory returns a fix-hint only for inventory
// commands whose result set is not visibly bounded. Backgrounded commands do not
// hold the turn open and explicit filters/limits are treated as intentional.
func ClassifyForegroundPowerShellInventory(command string) []Violation {
	return classifyForegroundPowerShellInventory(command)
}

func classifyForegroundPowerShellInventory(command string) []Violation {
	trimmed := strings.TrimSpace(command)
	if trimmed == "" || hasBackgroundAmpersand(trimmed) || powerShellStartsBackgroundJob(trimmed) {
		return nil
	}
	lower := strings.ToLower(trimmed)

	switch {
	case strings.Contains(lower, "get-winevent"):
		if hasAnyFold(lower, "-maxevents", "-filterhashtable", "-filterxml", "-filterxpath", "-listlog", "-listprovider") {
			return nil
		}
		return []Violation{powerShellInventoryViolation("Get-WinEvent", trimmed,
			"bound the event query with -MaxEvents N or a server-side -FilterHashtable, or run the long inventory in the background")}
	case strings.Contains(lower, "get-ciminstance") || strings.Contains(lower, "get-wmiobject"):
		if hasAnyFold(lower, "-filter", "-query") || powerShellPipelineBounded(lower) {
			return nil
		}
		op := "Get-CimInstance"
		if strings.Contains(lower, "get-wmiobject") {
			op = "Get-WmiObject"
		}
		return []Violation{powerShellInventoryViolation(op, trimmed,
			"add -Filter/-Query or a bounded Select-Object -First N, or run the host-wide inventory in the background")}
	default:
		return nil
	}
}

func powerShellStartsBackgroundJob(command string) bool {
	lower := strings.ToLower(command)
	return strings.Contains(lower, "start-job") || strings.Contains(lower, "start-threadjob") || strings.Contains(lower, "-asjob")
}

func powerShellPipelineBounded(command string) bool {
	if !strings.Contains(command, "select-object") && !strings.Contains(command, "select ") {
		return false
	}
	return strings.Contains(command, "-first") || strings.Contains(command, "-last") || strings.Contains(command, "-skip")
}

func hasAnyFold(s string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

func powerShellInventoryViolation(op, command, fix string) Violation {
	return Violation{
		Reason:   ReasonForegroundPowerShellInventory,
		Op:       op,
		Target:   command,
		Resolved: "<foreground-inventory>",
		Why:      "an unbounded host-wide PowerShell inventory can exceed the foreground turn budget and be killed before returning evidence",
		Fix:      fix,
	}
}
