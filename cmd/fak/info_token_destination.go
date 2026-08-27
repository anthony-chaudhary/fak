package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// infoTokenDestinationSnapshotSchema versions the info-side freshness wrapper around
// trajectory.AuditSummaryRow. The summary is a real recorded trajectory artifact: info never
// inspects transcript payloads, estimates a category mix from billed token counters, or turns an
// absent recorder into zeroes. A gateway/recording sidecar may publish this wrapper in the
// optional /debug/vars token_destination field; ordinary gateways omit it.
const infoTokenDestinationSnapshotSchema = "fak-info-token-destination/1"

// infoTokenDestinationSource reads a trajectory audit JSONL summary that another recorder
// refreshes. Stat is paid every info tick; the artifact itself is re-read only when its mtime
// changes. A zero Path leaves any snapshot supplied directly by /debug/vars untouched.
type infoTokenDestinationSource struct {
	Path       string
	MaxAge     time.Duration
	now        func() time.Time
	lastMod    time.Time
	lastSize   int64
	lastResult *infoTokenDestinationSnapshot
}

func (s *infoTokenDestinationSource) decorate(v *guardInfoVars) {
	if s == nil || strings.TrimSpace(s.Path) == "" {
		return
	}
	v.TokenDestination = s.snapshot()
}

func (s *infoTokenDestinationSource) snapshot() *infoTokenDestinationSnapshot {
	path := strings.TrimSpace(s.Path)
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return unavailableTokenDestinationSnapshot("recorded snapshot file unavailable")
	}
	if s.lastResult == nil || !info.ModTime().Equal(s.lastMod) || info.Size() != s.lastSize {
		summary, readErr := trajectory.ReadAuditBaseline(path)
		if readErr != nil {
			s.lastResult = unavailableTokenDestinationSnapshot("recorded snapshot file invalid")
		} else {
			window := fmt.Sprintf("%d recorded sessions", summary.Transcripts)
			if summary.Transcripts == 1 {
				window = "1 recorded session"
			}
			s.lastResult = &infoTokenDestinationSnapshot{
				Schema:       infoTokenDestinationSnapshotSchema,
				Availability: guardvars.AvailabilityObserved,
				RecordedAt:   info.ModTime().UTC().Format(time.RFC3339),
				Revision:     fmt.Sprintf("mtime:%d:size:%d", info.ModTime().Unix(), info.Size()),
				Window:       window,
				Summary:      *summary,
			}
		}
		s.lastMod, s.lastSize = info.ModTime(), info.Size()
	}

	result := *s.lastResult
	if result.Availability == guardvars.AvailabilityObserved && s.MaxAge > 0 {
		now := time.Now().UTC()
		if s.now != nil {
			now = s.now().UTC()
		}
		age := now.Sub(info.ModTime().UTC())
		if age > s.MaxAge {
			result.Availability = guardvars.AvailabilityStale
			result.Reason = fmt.Sprintf("recorded snapshot age %s exceeds %s", age.Round(time.Second), s.MaxAge)
		}
	}
	return &result
}

func unavailableTokenDestinationSnapshot(reason string) *infoTokenDestinationSnapshot {
	return &infoTokenDestinationSnapshot{
		Schema:       infoTokenDestinationSnapshotSchema,
		Availability: guardvars.AvailabilityUnavailable,
		Reason:       reason,
	}
}

// infoTokenDestinationSnapshot is the payload-free data seam between a recorded trajectory
// audit and the guard info TUI. Window names the recorder's actual bounded scope (for example,
// "last 12 turns"); RecordedAt and Availability keep a retained sample from being presented as
// live. Summary uses only trajectory's already-exported, versioned summary and distribution rows.
type infoTokenDestinationSnapshot struct {
	Schema       string                     `json:"schema"`
	Availability guardvars.Availability     `json:"availability"`
	RecordedAt   string                     `json:"recorded_at,omitempty"`
	Revision     string                     `json:"revision,omitempty"`
	Window       string                     `json:"window,omitempty"`
	Reason       string                     `json:"reason,omitempty"`
	Summary      trajectory.AuditSummaryRow `json:"summary,omitempty"`
}

// infoTokenDestinationRows renders the destination-mix rows placed immediately after the
// #9408 per-turn cost rows. The first row reuses trajectory's canonical compact renderer; the
// second states the non-billing byte basis, recorded window, and timestamp. Unavailable and stale
// lead their rows so a narrow pane cannot truncate away the honesty state.
func infoTokenDestinationRows(snapshot *infoTokenDestinationSnapshot, width int) []string {
	if snapshot == nil {
		return []string{infoTokenDestinationFit(" tokens→ unavailable · live recorded snapshot not available", width)}
	}
	if problem := infoTokenDestinationProblem(snapshot); problem != "" {
		return []string{infoTokenDestinationFit(" tokens→ unavailable · "+problem, width)}
	}

	compact := trajectory.CompactAuditDistributionLine(
		snapshot.Summary.Distribution,
		snapshot.Summary.ToolDistribution,
		0, // info applies a rune/cell-aware cap below; trajectory's renderer owns the content.
	)
	if snapshot.Availability == guardvars.AvailabilityStale {
		compact = "tokens→ STALE · " + strings.TrimPrefix(compact, "tokens→ · ")
	}
	mixRow := infoTokenDestinationFit(" "+compact, width)
	rows := []string{mixRow}
	// trajectory's compact form correctly puts category mix first, but that can push its
	// top-tool tail past a narrow pane's edge. Preserve the requested top-tool signal on a
	// second bounded row only when it did not survive the compact row.
	if len(snapshot.Summary.ToolDistribution) > 0 {
		top := snapshot.Summary.ToolDistribution[0]
		topCompact := fmt.Sprintf("top-tool %s %.0f%%", top.Name, top.Share*100)
		if !strings.Contains(mixRow, topCompact) {
			rows = append(rows, infoTokenDestinationFit(
				fmt.Sprintf(" %s of attributed tool bytes", topCompact), width,
			))
		}
	}

	basis := " basis   model-visible attributed UTF-8 bytes (not billed tokens)"
	if window := strings.TrimSpace(snapshot.Window); window != "" {
		basis += " · " + window
	}
	if recorded := infoTokenDestinationRecordedAt(snapshot.RecordedAt); recorded != "" {
		basis += " · recorded " + recorded
	}
	if snapshot.Availability == guardvars.AvailabilityStale {
		basis += " · STALE: " + strings.TrimSpace(snapshot.Reason)
	}
	rows = append(rows, infoTokenDestinationFit(basis, width))
	return rows
}

// infoTokenDestinationProblem validates every claim-bearing field before rendering. A
// malformed or unversioned snapshot degrades to unavailable; it never gets a best-effort mix.
func infoTokenDestinationProblem(snapshot *infoTokenDestinationSnapshot) string {
	if snapshot.Schema != infoTokenDestinationSnapshotSchema {
		return "invalid recorded snapshot schema"
	}
	switch snapshot.Availability {
	case guardvars.AvailabilityUnavailable:
		reason := strings.TrimSpace(snapshot.Reason)
		if reason == "" {
			reason = "live recorded snapshot not available"
		}
		return reason
	case guardvars.AvailabilityEmpty:
		return "recorded snapshot has no model-visible payload yet"
	case guardvars.AvailabilityObserved, guardvars.AvailabilityStale:
		// Continue below: both observed and retained-stale samples may carry the last real
		// distribution, but stale is labeled before any percentage is shown.
	default:
		return "invalid recorded snapshot availability"
	}
	if snapshot.Availability == guardvars.AvailabilityStale && strings.TrimSpace(snapshot.Reason) == "" {
		return "stale recorded snapshot omitted its reason"
	}
	if strings.TrimSpace(snapshot.Window) == "" {
		return "recorded snapshot omitted its window"
	}
	if _, err := time.Parse(time.RFC3339, strings.TrimSpace(snapshot.RecordedAt)); err != nil {
		return "recorded snapshot omitted a valid recorded_at"
	}
	summary := snapshot.Summary
	if summary.Schema != trajectory.AuditSchema || summary.Kind != "summary" {
		return "invalid trajectory audit summary"
	}
	if summary.DistributionUnit != trajectory.AuditDistributionUnit || strings.TrimSpace(summary.DistributionProvenance) == "" {
		return "trajectory distribution basis unavailable"
	}
	if len(summary.Distribution) == 0 {
		return "recorded snapshot has no model-visible distribution"
	}
	for _, row := range append(append([]trajectory.AuditDistributionRow(nil), summary.Distribution...), summary.ToolDistribution...) {
		if strings.TrimSpace(row.Name) == "" || row.Bytes < 0 || row.Share < 0 || row.Share > 1 {
			return "invalid trajectory distribution row"
		}
	}
	return ""
}

func infoTokenDestinationRecordedAt(raw string) string {
	at, err := time.Parse(time.RFC3339, strings.TrimSpace(raw))
	if err != nil {
		return ""
	}
	return at.UTC().Format(time.RFC3339)
}

// infoTokenDestinationFit is cell-aware, unlike a byte slice. Keeping the ellipsis inside
// the helper makes the captured narrow state visibly truncated rather than silently incomplete.
func infoTokenDestinationFit(row string, width int) string {
	if width <= 0 || dispWidthTUI(row) <= width {
		return row
	}
	if width <= 1 {
		return "…"
	}
	return takeCellsTUI(row, width-1) + "…"
}
