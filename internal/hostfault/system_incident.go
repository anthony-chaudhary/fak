package hostfault

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"time"
)

const (
	SystemIncidentSchema  = "fak.host-system-incident.v1"
	maxSystemMessageBytes = 64 << 10
)

type SystemIncident struct {
	Schema        string   `json:"schema"`
	EventID       string   `json:"event_id"`
	TimeMS        int64    `json:"time_ms"`
	TimeUTC       string   `json:"time_utc"`
	Source        string   `json:"source"`
	WindowsID     int      `json:"windows_event_id"`
	Class         string   `json:"class"`
	BugcheckCode  string   `json:"bugcheck_code,omitempty"`
	Parameters    []string `json:"parameters,omitempty"`
	DumpPath      string   `json:"dump_path,omitempty"`
	ReportID      string   `json:"report_id,omitempty"`
	Message       string   `json:"message,omitempty"`
	Observational bool     `json:"observational"`
}

type WindowsSystemEvent struct {
	TimeMS       int64    `json:"time_ms"`
	Source       string   `json:"source"`
	WindowsID    int      `json:"windows_event_id"`
	RecordID     string   `json:"record_id,omitempty"`
	BugcheckCode string   `json:"bugcheck_code,omitempty"`
	Parameters   []string `json:"parameters,omitempty"`
	DumpPath     string   `json:"dump_path,omitempty"`
	ReportID     string   `json:"report_id,omitempty"`
	Message      string   `json:"message,omitempty"`
}

// ClassifyWindowsSystemEvent converts only the three machine-restart evidence
// sources into observational records. These records are deliberately a distinct
// type from HostCrashSignal, so they cannot enter the resurrection planner.
func ClassifyWindowsSystemEvent(e WindowsSystemEvent) (SystemIncident, bool) {
	if e.TimeMS <= 0 {
		return SystemIncident{}, false
	}
	source := strings.TrimSpace(e.Source)
	class := ""
	switch {
	case e.WindowsID == 1001 && strings.EqualFold(source, "Microsoft-Windows-WER-SystemErrorReporting"):
		class = "bugcheck"
	case e.WindowsID == 41 && strings.EqualFold(source, "Microsoft-Windows-Kernel-Power"):
		class = "unclean_restart"
	case e.WindowsID == 6008 && strings.EqualFold(source, "EventLog"):
		class = "unexpected_shutdown"
	default:
		return SystemIncident{}, false
	}

	params := make([]string, 0, len(e.Parameters))
	for _, parameter := range e.Parameters {
		if parameter = strings.TrimSpace(parameter); parameter != "" {
			params = append(params, parameter)
		}
	}
	message := strings.TrimSpace(e.Message)
	if len(message) > maxSystemMessageBytes {
		message = message[:maxSystemMessageBytes]
	}
	parts := []string{
		strconv.FormatInt(e.TimeMS, 10), strings.ToLower(source), strconv.Itoa(e.WindowsID),
		strings.TrimSpace(e.RecordID), strings.ToLower(strings.TrimSpace(e.BugcheckCode)),
		strings.Join(params, "\x1f"), strings.ToLower(strings.TrimSpace(e.DumpPath)),
		strings.ToLower(strings.TrimSpace(e.ReportID)), message,
	}
	hash := sha256.Sum256([]byte(strings.Join(parts, "\x1e")))
	return SystemIncident{
		Schema:        SystemIncidentSchema,
		EventID:       "sys-" + hex.EncodeToString(hash[:12]),
		TimeMS:        e.TimeMS,
		TimeUTC:       time.UnixMilli(e.TimeMS).UTC().Format(time.RFC3339Nano),
		Source:        source,
		WindowsID:     e.WindowsID,
		Class:         class,
		BugcheckCode:  strings.TrimSpace(e.BugcheckCode),
		Parameters:    params,
		DumpPath:      strings.TrimSpace(e.DumpPath),
		ReportID:      strings.TrimSpace(e.ReportID),
		Message:       message,
		Observational: true,
	}, true
}
