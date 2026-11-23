package main

// In-run adaptive scale: a clean-room ramp-until-fail controller for speed A/B
// campaigns, inspired by the AdaptiveScale strategy described in
// docs/research/STUDY-aiperf-2026-11-09.md (candidate aiperf-lifecycle-11).
// INSPIRE-ONLY: the upstream implementation was not consulted; the study
// note's behavioral description is the only borrowed input.
//
// The controller walks per-window SLA assessments of a benchmark session. Each
// window records the session-concurrency cap that was live plus its observed
// tail latencies. Every passing window ramps the cap toward the next scheduled
// level; the first window that misses the filter marks the failing boundary,
// and the controller sustains the last passing level. Decisions are emitted as
// JSONL events so a receipt can replay why the cap moved.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	adaptiveScaleSchemaName    = "fak.speed-ab.ramp.v1"
	adaptiveScaleActionRamp    = "ramp"
	adaptiveScaleActionSustain = "sustain"
)

var (
	errAdaptiveScaleSchema      = errors.New("adaptive scale: schema must be " + adaptiveScaleSchemaName)
	errAdaptiveScaleWindows     = errors.New("adaptive scale: at least one assessment window is required")
	errAdaptiveScaleLevels      = errors.New("adaptive scale: window levels must be strictly increasing positive integers")
	errAdaptiveScaleSLA         = errors.New("adaptive scale: SLA thresholds must be finite and non-negative")
	errAdaptiveScaleMeasurement = errors.New("adaptive scale: window measurements must be finite and non-negative")
)

// adaptiveScaleSLA is the per-window filter a ramp must satisfy. A zero
// threshold is legal: only zero-observed windows pass it.
type adaptiveScaleSLA struct {
	MaxP95TTFTMS float64 `json:"max_p95_ttft_ms"`
	MaxP95ITLMS  float64 `json:"max_p95_itl_ms"`
	MaxErrors    int     `json:"max_errors"`
}

// adaptiveScaleWindow is one assessment window: the session-concurrency cap
// that was live and the SLA measurements taken against it.
type adaptiveScaleWindow struct {
	Level   int     `json:"level"`
	P95TTFT float64 `json:"p95_ttft_ms"`
	P95ITL  float64 `json:"p95_itl_ms"`
	Errors  int     `json:"errors"`
}

// adaptiveScaleDecision is one JSONL decision event.
type adaptiveScaleDecision struct {
	Window    int    `json:"window"`
	Level     int    `json:"level"`
	Action    string `json:"action"`
	NextLevel int    `json:"next_level,omitempty"`
}

// adaptiveScaleResult is the final controller state.
type adaptiveScaleResult struct {
	Schema    string                  `json:"schema"`
	Boundary  int                     `json:"boundary"`
	FailedAt  int                     `json:"failed_at"`
	Sustained int                     `json:"sustained"`
	Events    []adaptiveScaleDecision `json:"events"`
}

// adaptiveScaleInput is the ramp document accepted by --ramp mode.
type adaptiveScaleInput struct {
	Schema  string                `json:"schema"`
	Windows []adaptiveScaleWindow `json:"windows"`
	SLA     adaptiveScaleSLA      `json:"sla"`
}

func adaptiveScaleFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// withinSLA reports whether the window's measurements satisfy the filter.
// Thresholds are inclusive: a measurement exactly at the cap passes.
func (w adaptiveScaleWindow) withinSLA(sla adaptiveScaleSLA) bool {
	return w.P95TTFT <= sla.MaxP95TTFTMS && w.P95ITL <= sla.MaxP95ITLMS && w.Errors <= sla.MaxErrors
}

// adaptiveScaleRamp runs the ramp-until-fail controller over ordered
// assessment windows and returns the discovered boundary plus the sustained
// level. It refuses a malformed schedule or filter with sentinel errors
// instead of guessing.
func adaptiveScaleRamp(windows []adaptiveScaleWindow, sla adaptiveScaleSLA) (adaptiveScaleResult, error) {
	res := adaptiveScaleResult{Schema: adaptiveScaleSchemaName, Events: []adaptiveScaleDecision{}}
	if len(windows) == 0 {
		return res, errAdaptiveScaleWindows
	}
	if !adaptiveScaleFinite(sla.MaxP95TTFTMS) || !adaptiveScaleFinite(sla.MaxP95ITLMS) || sla.MaxP95TTFTMS < 0 || sla.MaxP95ITLMS < 0 || sla.MaxErrors < 0 {
		return res, errAdaptiveScaleSLA
	}
	prev := 0
	for _, w := range windows {
		if w.Level <= prev {
			return res, errAdaptiveScaleLevels
		}
		prev = w.Level
		if !adaptiveScaleFinite(w.P95TTFT) || !adaptiveScaleFinite(w.P95ITL) || w.P95TTFT < 0 || w.P95ITL < 0 || w.Errors < 0 {
			return res, errAdaptiveScaleMeasurement
		}
	}
	for i, w := range windows {
		if !w.withinSLA(sla) {
			res.FailedAt = w.Level
			res.Sustained = res.Boundary
			res.Events = append(res.Events, adaptiveScaleDecision{Window: i + 1, Level: w.Level, Action: adaptiveScaleActionSustain, NextLevel: res.Boundary})
			return res, nil
		}
		res.Boundary = w.Level
		action, next := adaptiveScaleActionSustain, 0
		if i+1 < len(windows) {
			action, next = adaptiveScaleActionRamp, windows[i+1].Level
		}
		res.Events = append(res.Events, adaptiveScaleDecision{Window: i + 1, Level: w.Level, Action: action, NextLevel: next})
	}
	res.Sustained = res.Boundary
	return res, nil
}

// writeAdaptiveScaleJSONL renders decision events one JSON object per line.
func writeAdaptiveScaleJSONL(w io.Writer, events []adaptiveScaleDecision) error {
	enc := json.NewEncoder(w)
	for _, ev := range events {
		if err := enc.Encode(ev); err != nil {
			return fmt.Errorf("adaptive scale: encode decision event: %w", err)
		}
	}
	return nil
}

// parseAdaptiveScaleInput decodes and schema-checks a --ramp document. Window
// validation belongs to adaptiveScaleRamp.
func parseAdaptiveScaleInput(b []byte) (adaptiveScaleInput, error) {
	var in adaptiveScaleInput
	if err := json.Unmarshal(b, &in); err != nil {
		return adaptiveScaleInput{}, fmt.Errorf("adaptive scale: decode ramp document: %w", err)
	}
	if in.Schema != adaptiveScaleSchemaName {
		return adaptiveScaleInput{}, errAdaptiveScaleSchema
	}
	return in, nil
}
