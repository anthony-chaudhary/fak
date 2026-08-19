//go:build windows

package sessionjournal

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

var (
	modKernel32SJ        = syscall.NewLazyDLL("kernel32.dll")
	procGetTickCount64   = modKernel32SJ.NewProc("GetTickCount64")
	queryWindowsBootTime = windowsBootTimeWMI
)

type persistedBootMarker struct {
	BootTime string `json:"boot_time"`
	Source   string `json:"source"`
}

func approximateWindowsBootTime(now time.Time) time.Time {
	r, _, _ := procGetTickCount64.Call()
	if r == 0 {
		return time.Time{}
	}
	return now.Add(-time.Duration(uint64(r)) * time.Millisecond).UTC()
}

func windowsBootTimeWMI() (time.Time, error) {
	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
		"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	out, err := cmd.Output()
	if err != nil {
		return time.Time{}, err
	}
	boot, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(string(out)))
	if err != nil {
		return time.Time{}, fmt.Errorf("parse LastBootUpTime: %w", err)
	}
	return boot.UTC(), nil
}

func readBootMarker(path string) (time.Time, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return time.Time{}, false
	}
	var marker persistedBootMarker
	if json.Unmarshal(b, &marker) != nil || marker.Source != "wmi-lastbootuptime" {
		return time.Time{}, false
	}
	boot, err := time.Parse(time.RFC3339Nano, marker.BootTime)
	return boot.UTC(), err == nil
}

func writeBootMarker(path string, boot time.Time) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, _ := json.Marshal(persistedBootMarker{BootTime: boot.UTC().Format(time.RFC3339Nano), Source: "wmi-lastbootuptime"})
	tmp, err := os.CreateTemp(filepath.Dir(path), ".boot-marker-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(append(b, '\n')); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	_ = os.Remove(path) // Windows rename cannot replace an existing destination.
	return os.Rename(name, path)
}

// BootTime returns WMI's exact LastBootUpTime. A host-global persisted marker
// avoids a PowerShell process on every heartbeat; GetTickCount64 is used only to
// detect that the marker belongs to an earlier boot.
func BootTime(now time.Time) (time.Time, string) {
	markerPath := bootMarkerPath()
	approximate := approximateWindowsBootTime(now)
	if marked, ok := readBootMarker(markerPath); ok && !approximate.IsZero() && absDuration(marked.Sub(approximate)) < 2*time.Minute {
		return marked, "wmi-lastbootuptime-marker"
	}
	if exact, err := queryWindowsBootTime(); err == nil && !exact.IsZero() {
		_ = writeBootMarker(markerPath, exact)
		return exact, "wmi-lastbootuptime"
	}
	if !approximate.IsZero() {
		return approximate, "gettickcount64"
	}
	return time.Time{}, "unknown"
}

func absDuration(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}
