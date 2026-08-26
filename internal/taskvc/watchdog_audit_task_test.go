package taskvc

import (
	"os"
	"strings"
	"testing"
)

func TestWatchdogWatchdogAuditTaskIsOrthogonal(t *testing.T) {
	raw, err := os.ReadFile("../../tools/scheduled-tasks/FleetWatchdogWatchdogAudit.xml")
	if err != nil {
		t.Fatal(err)
	}
	xml := string(raw)
	for _, want := range []string{"<LogonType>S4U</LogonType>", "%USERPROFILE%\\bin\\fak.exe", "watchdog-audit-run", "tools\\watchdog_watchdog_audit.ps1", "%LOCALAPPDATA%\\fak-watchdog-audit\\audit.jsonl", "--max-bytes 4194304"} {
		if !strings.Contains(xml, want) {
			t.Errorf("task capture missing %q", want)
		}
	}
	for _, forbidden := range []string{"<LogonType>InteractiveToken</LogonType>", "%LOCALAPPDATA%\\fak-watchdog-audit\\run-watchdog-audit.ps1", "exits success"} {
		if strings.Contains(xml, forbidden) {
			t.Errorf("task capture retains stale contract %q", forbidden)
		}
	}
}
