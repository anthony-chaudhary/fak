package fleetpane

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// health.go folds the headless fleet monitor's worker-health classification
// histogram (internal/fleetmon's `fak-fleet-monitor/1` payload) into the
// operator control pane, so `fak fleetpane status` exposes ONE health summary in
// both its JSON and its human readout (#2035).
//
// The pane consumes the monitor as JSON — the same way it already consumes the
// supervisor and registry payloads — so this fold needs no code dependency on
// internal/fleetmon; it reads the stable wire schema. The class-name strings and
// the replacement-eligible set below MIRROR internal/fleetmon (the Classification
// constants in monitor.go and eligibleClasses in replace.go); fleetmon is the
// source of truth, and both sets are covered by health_test.go so a drift is
// caught rather than silently rendered.

// MonitorSchema is the wire schema tag the fleet monitor stamps on its payload.
const MonitorSchema = "fak-fleet-monitor/1"

// healthClassOrder is the closed set of monitor worker classes, in the operator
// readout order. It mirrors internal/fleetmon.Classification; every class is
// always present in WorkerHealth.Counts (zero-filled) so the JSON and the human
// line expose an identical, stable key set.
var healthClassOrder = []string{
	"healthy",
	"completed-final",
	"dead",
	"stale-transcript",
	"auth-or-rate-blocked",
	"stale-child-command",
	"attention",
}

// healthReplacementClasses is the set of classes for which the launcher permits a
// replacement — it mirrors internal/fleetmon/replace.go's eligibleClasses (dead,
// auth-or-rate-blocked, stale-transcript). Their sum is the pane's derived
// "replacement-needed" rollup: how many workers the operator can relaunch.
var healthReplacementClasses = []string{"dead", "auth-or-rate-blocked", "stale-transcript"}

// WorkerHealth is the monitor's worker-health classification, folded for the
// operator pane. Counts always carries every healthClassOrder key (zero-filled)
// plus any class the monitor emits that this build does not yet know;
// ReplacementNeeded is the derived rollup of the replacement-eligible classes.
type WorkerHealth struct {
	Available         bool           `json:"available"`
	Reason            string         `json:"reason,omitempty"`
	MonitorSchema     string         `json:"monitor_schema,omitempty"`
	GeneratedAt       string         `json:"generated_at,omitempty"`
	Source            []string       `json:"source_cmd,omitempty"`
	Total             int            `json:"total"`
	Counts            map[string]int `json:"counts"`
	ReplacementNeeded int            `json:"replacement_needed"`
}

// monitorPayloadShape is the subset of the fak-fleet-monitor/1 payload the pane
// folds: the class histogram and the run total (plus schema/time for context).
type monitorPayloadShape struct {
	Schema      string         `json:"schema"`
	GeneratedAt string         `json:"generated_at"`
	Total       int            `json:"total"`
	ByClass     map[string]int `json:"by_class"`
}

// emptyHealthCounts returns the zero-filled canonical count map.
func emptyHealthCounts() map[string]int {
	counts := make(map[string]int, len(healthClassOrder))
	for _, c := range healthClassOrder {
		counts[c] = 0
	}
	return counts
}

// WorkerHealthFromMonitorJSON folds a fak-fleet-monitor/1 payload into a pane
// health summary. Empty or unparseable input yields an unavailable summary with a
// reason rather than a misleading all-zero histogram — the honest "no monitor
// witness" rung, matched to how the pane reports an unavailable supervisor.
func WorkerHealthFromMonitorJSON(raw []byte) WorkerHealth {
	health := WorkerHealth{Counts: emptyHealthCounts()}
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		health.Reason = "no monitor JSON"
		return health
	}
	var payload monitorPayloadShape
	if err := json.Unmarshal([]byte(trimmed), &payload); err != nil {
		health.Reason = "monitor JSON was not parseable: " + err.Error()
		return health
	}
	health.Available = true
	health.MonitorSchema = payload.Schema
	health.GeneratedAt = payload.GeneratedAt
	health.Total = payload.Total
	for class, n := range payload.ByClass {
		health.Counts[class] += n // pass through any class, canonical or not
	}
	for _, c := range healthReplacementClasses {
		health.ReplacementNeeded += health.Counts[c]
	}
	return health
}

// WorkerHealthText renders the one-line health readout the pane prints. It reads
// the SAME WorkerHealth.Counts the JSON exposes — every count key is rendered, in
// canonical order first then any unknown classes sorted — so the human and
// machine outputs cannot drift.
func WorkerHealthText(h WorkerHealth) string {
	if !h.Available {
		reason := h.Reason
		if reason == "" {
			reason = "no monitor witness"
		}
		return "health: unavailable (" + reason + ")"
	}
	parts := make([]string, 0, len(h.Counts)+2)
	parts = append(parts, fmt.Sprintf("total=%d", h.Total))
	known := make(map[string]bool, len(healthClassOrder))
	for _, c := range healthClassOrder {
		known[c] = true
		parts = append(parts, fmt.Sprintf("%s=%d", c, h.Counts[c]))
	}
	extras := make([]string, 0)
	for c := range h.Counts {
		if !known[c] {
			extras = append(extras, c)
		}
	}
	sort.Strings(extras)
	for _, c := range extras {
		parts = append(parts, fmt.Sprintf("%s=%d", c, h.Counts[c]))
	}
	parts = append(parts, fmt.Sprintf("replacement-needed=%d", h.ReplacementNeeded))
	return "health: " + strings.Join(parts, " ")
}
