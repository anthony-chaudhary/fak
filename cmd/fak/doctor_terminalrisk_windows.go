//go:build windows

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/terminalrisk"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func gatherTerminalRiskFacts(settingsPath string) (terminalrisk.Facts, error) {
	raw, err := os.ReadFile(settingsPath)
	if err != nil {
		return terminalrisk.Facts{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	script := `$amd=@(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue|?{$_.Name -match '(?i)AMD|Radeon'}).Count -gt 0;$crash=@(Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Application Error';Id=1000;StartTime=(Get-Date).AddDays(-30)} -ErrorAction SilentlyContinue|?{$_.Message -match '(?i)WindowsTerminal\.exe' -and $_.Message -match '(?i)Microsoft\.Terminal\.Control\.dll' -and $_.Message -match '0xc0000005'}).Count -gt 0;[pscustomobject]@{amd=$amd;crash=$crash}|ConvertTo-Json -Compress`
	cmd := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return terminalrisk.Facts{}, fmt.Errorf("terminal risk probe: %w", err)
	}
	var p struct {
		AMD   bool `json:"amd"`
		Crash bool `json:"crash"`
	}
	if err := json.Unmarshal(out, &p); err != nil {
		return terminalrisk.Facts{}, err
	}
	return terminalrisk.Facts{AMDPresent: p.AMD, PriorWTRenderCrash: p.Crash, SettingsPath: settingsPath, Settings: raw}, nil
}
func defaultWTSettingsPath() string {
	return filepath.Join(os.Getenv("LOCALAPPDATA"), "Packages", "Microsoft.WindowsTerminal_8wekyb3d8bbwe", "LocalState", "settings.json")
}
func terminalRiskRDPAdvice() string {
	return "RDP review: AVCHardwareEncodePreferred=0 may still enumerate AMD HW encoders (bEnumerateHWBeforeSW); inspect policy manually."
}
