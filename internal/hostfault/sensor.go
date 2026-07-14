package hostfault

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
)

const HostCrashSignalSchema = "fak.host-crash.v1"

type HostCrashClass string

const (
	HostCrashWTRenderAV       HostCrashClass = "WT_RENDER_AV"
	HostCrashTermServiceAMDAV HostCrashClass = "TERMSERVICE_AMD_AV"
	HostCrashGeneric          HostCrashClass = "HOST_CRASH_GENERIC"
)

// ApplicationError1000 is the stable subset of a Windows Application Error
// Event 1000 needed by the terminal-host crash sensor.
type ApplicationError1000 struct {
	TimeMS      int64  `json:"time_ms"`
	App         string `json:"app"`
	Module      string `json:"module"`
	Exception   string `json:"exception"`
	FaultOffset string `json:"fault_offset,omitempty"`
	ProcessID   string `json:"process_id,omitempty"`
	ReportID    string `json:"report_id,omitempty"`
}

type HostCrashSignal struct {
	Schema         string         `json:"schema"`
	WhenMS         int64          `json:"when_unix_ms"`
	HostPID        int64          `json:"host_pid,omitempty"`
	FaultingApp    string         `json:"faulting_app"`
	FaultingModule string         `json:"faulting_module"`
	Exception      string         `json:"exception"`
	FaultOffset    string         `json:"fault_offset,omitempty"`
	Class          HostCrashClass `json:"class"`
	EventID        string         `json:"event_id"`
}

// ClassifyApplicationError fails closed for relevance: every terminal-family
// Event 1000 emits a signal, even when its module/signature is new. Unrelated
// application crashes are not host-crash signals.
func ClassifyApplicationError(e ApplicationError1000) (HostCrashSignal, bool) {
	app := strings.ToLower(strings.TrimSpace(e.App))
	module := strings.ToLower(strings.TrimSpace(e.Module))
	exception := normalizeHex(e.Exception)
	terminalFamily := app == "windowsterminal.exe" || app == "svchost.exe" || app == "openconsole.exe" || app == "pwsh.exe" || app == "powershell.exe"
	if !terminalFamily {
		return HostCrashSignal{}, false
	}
	class := HostCrashGeneric
	if app == "windowsterminal.exe" && exception == "0xc0000005" && (module == "microsoft.terminal.control.dll" || module == "terminalapp.dll") {
		class = HostCrashWTRenderAV
	} else if app == "svchost.exe" && exception == "0xc0000005" && module == "amdxx64.dll" {
		class = HostCrashTermServiceAMDAV
	}
	pidText := strings.TrimSpace(e.ProcessID)
	base := 10
	if strings.HasPrefix(strings.ToLower(pidText), "0x") {
		base = 16
		pidText = pidText[2:]
	}
	pid, _ := strconv.ParseInt(pidText, base, 64)
	identity := fmt.Sprintf("%d\x00%s\x00%s\x00%s\x00%s\x00%s\x00%s", e.TimeMS, app, module, exception, strings.ToLower(e.FaultOffset), strings.ToLower(e.ProcessID), strings.ToLower(e.ReportID))
	sum := sha256.Sum256([]byte(identity))
	return HostCrashSignal{
		Schema: HostCrashSignalSchema, WhenMS: e.TimeMS, HostPID: pid,
		FaultingApp: e.App, FaultingModule: e.Module, Exception: exception,
		FaultOffset: e.FaultOffset, Class: class, EventID: hex.EncodeToString(sum[:12]),
	}, true
}

func normalizeHex(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return ""
	}
	if strings.HasPrefix(s, "0x") {
		return s
	}
	return "0x" + s
}
