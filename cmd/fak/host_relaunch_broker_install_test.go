package main

import (
	"os"
	"strings"
	"testing"
)

func TestStallMonitorInstallsInteractiveHostRelaunchBroker(t *testing.T) {
	raw, err := os.ReadFile("../../tools/fak_stall_monitor.ps1")
	if err != nil {
		t.Fatal(err)
	}
	script := string(raw)
	want := []string{
		"$brokerPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited",
		"if ($broker.Principal.LogonType -ne 'InteractiveToken')",
		"Test-Path -LiteralPath $broker.Actions.Execute -PathType Leaf",
		"Test-Path -LiteralPath $brokerSpool -PathType Container",
		"[uint32]$brokerInfo.LastTaskResult -eq 0x80070002",
	}
	for _, fragment := range want {
		if !strings.Contains(script, fragment) {
			t.Errorf("installer is missing %q", fragment)
		}
	}
	if strings.Contains(script, "$brokerPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U") {
		t.Error("host relaunch broker must not use S4U: session 0 cannot activate Windows Terminal")
	}
}
