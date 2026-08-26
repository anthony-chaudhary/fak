//go:build windows

package main

import (
	"bytes"
	"os/exec"
	"strings"
	"testing"
)

func TestHostdiagWindowsCollectorParses(t *testing.T) {
	script := strings.Replace(hostdiagEventPS, "__MILLIS__", "60000", 1)
	command := `$source=[Console]::In.ReadToEnd(); $errors=$null; [System.Management.Automation.Language.Parser]::ParseInput($source,[ref]$null,[ref]$errors)>$null; if($errors.Count){$errors|ForEach-Object{$_.Message}; exit 1}`
	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command)
	cmd.Stdin = strings.NewReader(script)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("collector PowerShell does not parse: %v: %s", err, stderr.String())
	}
}

func TestHostdiagWindowsCollectorBoundsHostLifecycleEvidence(t *testing.T) {
	for _, want := range []string{
		"ProviderName='User32';Id=1074", "event_name='HOST_RESTART_INITIATED'",
		"ProviderName='EventLog';Id=6008", "event_name='HOST_UNEXPECTED_SHUTDOWN'",
		"ProviderName='Microsoft-Windows-Kernel-Power';Id=41", "event_name='HOST_UNCLEAN_RESTART'",
	} {
		if !strings.Contains(hostdiagEventPS, want) {
			t.Fatalf("collector missing %q", want)
		}
	}
	for _, name := range []string{"HOST_RESTART_INITIATED", "HOST_UNEXPECTED_SHUTDOWN", "HOST_UNCLEAN_RESTART"} {
		marker := "event_name='" + name + "'"
		at := strings.Index(hostdiagEventPS, marker)
		start := strings.LastIndex(hostdiagEventPS[:at], "[pscustomobject]")
		end := strings.Index(hostdiagEventPS[at:], "}")
		row := hostdiagEventPS[start : at+end+1]
		if strings.Contains(row, "message=") {
			t.Fatalf("%s retained raw message: %s", name, row)
		}
	}
}
