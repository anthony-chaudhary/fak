// Package selfcheck provides the shared self-check contract for factory CLIs.
//
// It exposes a small, stable vocabulary (PASS/WARN/FAIL), a per-check Item with
// an optional remediation line, and an aggregating Report that renders as text or
// JSON. Adopting CLIs wire a `selfcheck` verb that reports whether their own
// subsystem is usable right now, exiting non-zero on any FAIL so agents can gate
// before dispatching work.
//
// The vocabulary mirrors platform/release/doctor.go, which remains the reference
// implementation for the aggregate diagnosis; this package is the reusable seam.
package selfcheck

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// Status is the diagnostic result of an individual self-check item.
type Status string

const (
	// StatusPass means the check succeeded and the subsystem is usable.
	StatusPass Status = "PASS"
	// StatusWarn means the subsystem is usable but degraded; WARN does not fail the run.
	StatusWarn Status = "WARN"
	// StatusFail means the subsystem is unusable; any FAIL fails the run.
	StatusFail Status = "FAIL"
)

// Item records a single self-check evaluation.
type Item struct {
	Category    string `json:"category"`
	Name        string `json:"name"`
	Status      Status `json:"status"`
	Message     string `json:"message"`
	Remediation string `json:"remediation,omitempty"`
}

// Report contains all self-check items and aggregated summary counts.
type Report struct {
	Command     string    `json:"command"`
	Timestamp   time.Time `json:"timestamp"`
	PassCount   int       `json:"pass_count"`
	WarnCount   int       `json:"warn_count"`
	FailCount   int       `json:"fail_count"`
	Items       []Item    `json:"items"`
	OverallPass bool      `json:"overall_pass"`
}

// NewReport creates an empty report for the named command, timestamped in UTC.
// An empty report has no failures, so it is PASS by construction; this keeps
// ExitCode()==0 for a report with zero items rather than failing on the zero value.
func NewReport(command string) *Report {
	return &Report{Command: command, Timestamp: time.Now().UTC(), OverallPass: true}
}

// Add appends an item and updates the aggregate counts and overall verdict.
func (r *Report) Add(item Item) {
	r.Items = append(r.Items, item)
	switch item.Status {
	case StatusPass:
		r.PassCount++
	case StatusWarn:
		r.WarnCount++
	case StatusFail:
		r.FailCount++
	}
	r.OverallPass = r.FailCount == 0
}

// Pass is a convenience constructor for a passing item.
func Pass(category, name, message string) Item {
	return Item{Category: category, Name: name, Status: StatusPass, Message: message}
}

// Warn is a convenience constructor for a degraded-but-usable item.
func Warn(category, name, message, remediation string) Item {
	return Item{Category: category, Name: name, Status: StatusWarn, Message: message, Remediation: remediation}
}

// Fail is a convenience constructor for an unusable item with a remediation line.
func Fail(category, name, message, remediation string) Item {
	return Item{Category: category, Name: name, Status: StatusFail, Message: message, Remediation: remediation}
}

// ExitCode returns 0 when no item failed and 1 otherwise.
func (r *Report) ExitCode() int {
	if r.OverallPass {
		return 0
	}
	return 1
}

// String renders the report as a human-readable table with a summary line.
func (r *Report) String() string {
	var sb strings.Builder
	title := "Self-Check Report"
	if r.Command != "" {
		title = fmt.Sprintf("Self-Check Report: %s", r.Command)
	}
	sb.WriteString(fmt.Sprintf("=== %s (%s) ===\n\n", title, r.Timestamp.UTC().Format(time.RFC3339)))
	for _, item := range r.Items {
		sb.WriteString(fmt.Sprintf("[%-4s] %-24s | %s\n", item.Status, item.Category+": "+item.Name, item.Message))
		if item.Remediation != "" {
			sb.WriteString(fmt.Sprintf("       Remediation: %s\n", item.Remediation))
		}
	}
	sb.WriteString("\n----------------------------------------------------------------------\n")
	verdict := "PASS"
	if !r.OverallPass {
		verdict = "FAIL"
	}
	sb.WriteString(fmt.Sprintf("Summary: %d passed, %d warnings, %d failures (OVERALL: %s)\n",
		r.PassCount, r.WarnCount, r.FailCount, verdict))
	return sb.String()
}

// WriteJSON encodes the report as indented JSON to w.
func (r *Report) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// Render writes the report to w as JSON when jsonOut is set, else as text, and
// returns the process exit code (0 when no item failed).
func (r *Report) Render(w io.Writer, jsonOut bool) (int, error) {
	if jsonOut {
		if err := r.WriteJSON(w); err != nil {
			return 1, err
		}
		return r.ExitCode(), nil
	}
	if _, err := io.WriteString(w, r.String()); err != nil {
		return 1, err
	}
	return r.ExitCode(), nil
}
