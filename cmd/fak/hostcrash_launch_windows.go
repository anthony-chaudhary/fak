//go:build windows

package main

import (
	"encoding/base64"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"unicode/utf16"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func encodedPowerShellCommand(script string) string {
	units := utf16.Encode([]rune(script))
	b := make([]byte, len(units)*2)
	for i, v := range units {
		b[2*i], b[2*i+1] = byte(v), byte(v>>8)
	}
	return base64.StdEncoding.EncodeToString(b)
}

var runInteractiveBrokerTask = func(taskName string) error {
	script := "Start-ScheduledTask -TaskName '" + strings.ReplaceAll(taskName, "'", "''") + "'"
	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encodedPowerShellCommand(script))
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run()
}

func hostRelaunchSpoolDir() string {
	if dir := strings.TrimSpace(os.Getenv("FAK_HOST_RELAUNCH_DIR")); dir != "" {
		return dir
	}
	base, err := os.UserConfigDir()
	if err != nil || base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "fak", "host", "relaunch")
}

// The S4U watchdog persists before signaling the desktop broker. If TermService
// is unavailable, Start-ScheduledTask may fail but the request remains queued and
// the broker's AtLogOn trigger drains it after the interactive host recovers.
func launchHostSessionPlatform(req hostresurrect.Request) (int, error) {
	if len(req.Command) == 0 {
		return 0, errors.New("empty relaunch command")
	}
	if _, err := hostresurrect.Enqueue(hostRelaunchSpoolDir(), req); err != nil {
		return 0, err
	}
	_ = runInteractiveBrokerTask("FakHostRelaunchBroker")
	return 0, nil
}
