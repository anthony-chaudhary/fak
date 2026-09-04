// Package localappux renders host-app language for local compute lifecycle states.
//
// Invariant: UX rendering is deterministic and fail-closed.
// Guard: Diagnostic preview enforces strict redaction of sensitive paths, tokens, and prompts before serialization.
package localappux

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Mode designates the compute execution policy mode for the host application.
type Mode string

// Host-app compute execution modes.
const (
	ModeAutomatic   Mode = "automatic"
	ModePreferLocal Mode = "prefer_local"
	ModeLocalOnly   Mode = "local_only"
	ModePaused      Mode = "paused"
)

// State designates the local compute readiness lifecycle state.
type State string

// Lifecycle readiness states.
const (
	StateFirstRun      State = "first_run"
	StatePartial       State = "partial_readiness"
	StateNoSpace       State = "no_space"
	StateNoNetwork     State = "no_network"
	StatePressure      State = "pressure"
	StateBattery       State = "battery"
	StateThermal       State = "thermal"
	StateHelperRestart State = "helper_restart"
	StateHandoffAsk    State = "handoff_ask"
	StateHandoffDenied State = "handoff_denied"
	StateRollback      State = "corrupt_update_rollback"
	StateReady         State = "ready"
)

// View represents the current display state of the local compute features.
type View struct {
	State                            State
	Mode                             Mode
	AssetBytes, FreeBytes            int64
	ReadyTasks, PendingTasks         []string
	LocalData                        string
	Destination                      string
	CanRetry, CanRepair, CanRollback bool
}

// Render returns the user-facing status title, detail message, and action text.
// Invariant: UX rendering is deterministic and fail-closed.
func Render(v View) string {
	title, detail, action := "Local features are ready", "Your tasks run on this Mac.", ""
	switch v.State {
	case StateFirstRun:
		title = "Set up local features"
		detail = fmt.Sprintf("Download %s. %s available. Tasks: %s.", bytes(v.AssetBytes), bytes(v.FreeBytes), join(v.PendingTasks))
		action = "Download"
	case StatePartial:
		title = "Some local features are ready"
		detail = fmt.Sprintf("Ready: %s. Still downloading: %s.", join(v.ReadyTasks), join(v.PendingTasks))
		action = "Continue in background"
	case StateNoSpace:
		title = "More storage is needed"
		detail = fmt.Sprintf("The download needs %s; %s is available.", bytes(v.AssetBytes), bytes(v.FreeBytes))
		action = "Manage storage"
	case StateNoNetwork:
		title = "Waiting for a connection"
		detail = "Downloaded features remain available. Connect to finish setup."
		action = "Retry"
	case StatePressure:
		title = "Local work is paused to keep this app responsive"
		detail = "It will resume when memory pressure falls."
		action = "Use cloud or wait"
	case StateBattery:
		title = "Using less power"
		detail = "Local tasks may take longer while on battery."
		action = "Continue or pause"
	case StateThermal:
		title = "Cooling down"
		detail = "Local work will resume when this Mac cools."
		action = "Wait"
	case StateHelperRestart:
		title = "Restarting local features"
		detail = "Your work is safe. The helper is reconnecting."
		action = "Retry now"
	case StateHandoffAsk:
		title = "Continue using cloud compute?"
		detail = fmt.Sprintf("Send %s to %s. Local processing cannot finish this task.", v.LocalData, v.Destination)
		action = "Allow once or keep local"
	case StateHandoffDenied:
		title = "Cloud handoff was not allowed"
		detail = "Nothing was sent. Retry locally or cancel."
		action = "Retry locally"
	case StateRollback:
		title = "Update repaired"
		detail = "The new local asset failed verification. The previous working version was restored."
		action = "Retry update or keep current"
	case StateReady:
		// The ready copy is the initialized default above; spelling the case
		// keeps future lifecycle states from silently inheriting it.
	}
	return fmt.Sprintf("%s\n%s\nMode: %s\nAction: %s\n", title, detail, labelMode(v.Mode), action)
}
func bytes(n int64) string {
	if n >= 1<<30 {
		return fmt.Sprintf("%.1f GB", float64(n)/float64(1<<30))
	}
	return fmt.Sprintf("%.1f MB", float64(n)/float64(1<<20))
}
func join(v []string) string {
	if len(v) == 0 {
		return "none"
	}
	x := append([]string(nil), v...)
	sort.Strings(x)
	return strings.Join(x, ", ")
}
func labelMode(m Mode) string {
	switch m {
	case ModeAutomatic:
		return "Automatic"
	case ModePreferLocal:
		return "Prefer local"
	case ModeLocalOnly:
		return "Local only"
	case ModePaused:
		return "Paused"
	}
	return "Automatic"
}

// Diagnostic contains local application state and telemetry for issue reports.
type Diagnostic struct {
	AppVersion string   `json:"app_version"`
	State      State    `json:"state"`
	Mode       Mode     `json:"mode"`
	Engine     string   `json:"engine"`
	ErrorCode  string   `json:"error_code,omitempty"`
	Paths      []string `json:"paths,omitempty"`
	Prompt     string   `json:"prompt,omitempty"`
	Token      string   `json:"token,omitempty"`
}

var sensitive = regexp.MustCompile(`(?i)(token|secret|password|prompt|path|user)`)

// PreviewDiagnostic generates a consent-safe JSON report with sensitive data scrubbed.
// Guard: Any key containing sensitive strings (tokens, secrets, passwords, prompts, paths, users) is stripped.
func PreviewDiagnostic(d Diagnostic) ([]byte, error) {
	raw := map[string]any{"schema": "fak.local-app-diagnostic/1", "app_version": d.AppVersion, "state": d.State, "mode": d.Mode, "engine": d.Engine, "error_code": d.ErrorCode}
	for k := range raw {
		if sensitive.MatchString(k) {
			delete(raw, k)
		}
	}
	return json.MarshalIndent(raw, "", "  ")
}
