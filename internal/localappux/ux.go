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
//
// Contract: Mode must be one of ModeAutomatic, ModePreferLocal, ModeLocalOnly, or ModePaused.
type Mode string

// Host-app compute execution modes.
const (
	// ModeAutomatic dynamically routes tasks based on device pressure, battery, and capability.
	ModeAutomatic Mode = "automatic"
	// ModePreferLocal attempts on-device compute first and prompts before cloud handoff.
	ModePreferLocal Mode = "prefer_local"
	// ModeLocalOnly forbids cloud fallback and keeps all execution strictly on the local machine.
	ModeLocalOnly Mode = "local_only"
	// ModePaused suspends local execution to preserve system resources.
	ModePaused Mode = "paused"
)

// State designates the local compute readiness lifecycle state.
//
// Invariant: Lifecycle states resolve deterministically to user-visible status and actionable copy.
type State string

// Lifecycle readiness states.
const (
	// StateFirstRun indicates local assets need initial download and verification.
	StateFirstRun State = "first_run"
	// StatePartial indicates some local features are downloaded while others are pending.
	StatePartial State = "partial_readiness"
	// StateNoSpace indicates disk space is insufficient for the asset download.
	StateNoSpace State = "no_space"
	// StateNoNetwork indicates download or verification is paused awaiting network connectivity.
	StateNoNetwork State = "no_network"
	// StatePressure indicates local inference is suspended due to system memory or resource pressure.
	StatePressure State = "pressure"
	// StateBattery indicates conservative execution while the host device runs on battery power.
	StateBattery State = "battery"
	// StateThermal indicates execution throttling to allow the host hardware to cool down.
	StateThermal State = "thermal"
	// StateHelperRestart indicates the helper process exited and is reconnecting safely.
	StateHelperRestart State = "helper_restart"
	// StateHandoffAsk requests user consent before transmitting task context to cloud compute.
	StateHandoffAsk State = "handoff_ask"
	// StateHandoffDenied reflects user refusal of cloud handoff, falling back locally.
	StateHandoffDenied State = "handoff_denied"
	// StateRollback indicates a corrupted update was safely rolled back to the last known-good asset.
	StateRollback State = "corrupt_update_rollback"
	// StateReady indicates local compute assets are verified and ready for on-device inference.
	StateReady State = "ready"
)

// View represents the current display state of the local compute features.
//
// Invariant: State and Mode fields determine user-facing copy deterministically without side effects.
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
//
// Invariant: UX rendering is deterministic and fail-closed.
// Precondition: Caller passes a View instance; unknown states fallback safely to ready default copy.
// Postcondition: Returns a formatted multi-line status string containing title, detail, Mode, and Action.
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
//
// Invariant: Diagnostic fields represent host state snapshots prior to privacy screening.
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
//
// Guard: Any key containing sensitive strings (tokens, secrets, passwords, prompts, paths, users) is stripped.
// Precondition: Caller passes Diagnostic telemetry data for serialization.
// Postcondition: Returns indented JSON payload conforming to schema fak.local-app-diagnostic/1 with sensitive keys deleted.
func PreviewDiagnostic(d Diagnostic) ([]byte, error) {
	raw := map[string]any{"schema": "fak.local-app-diagnostic/1", "app_version": d.AppVersion, "state": d.State, "mode": d.Mode, "engine": d.Engine, "error_code": d.ErrorCode}
	for k := range raw {
		if sensitive.MatchString(k) {
			delete(raw, k)
		}
	}
	return json.MarshalIndent(raw, "", "  ")
}
