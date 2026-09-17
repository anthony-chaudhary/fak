package causalreceipt

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// Produce folds a bounded compute-trace artifact plus a caller-supplied launch
// context into one validated causal receipt. It is a pure, deterministic
// transform: no clock, no global state, no I/O.
//
// Grouping: one Phase per (Engine, Route) pair, where Engine is the trace
// event's Backend (e.g. "cuda", "metal", "cpu-ref") and Backend is the event's
// Route (e.g. "metal_command_buffer"). This is the natural grouping the
// computetrace.Event data supports: every retained event carries Backend and
// Route, and device work is dispatched per backend/route. Events with no
// Backend group under engine "unknown" rather than being dropped, because
// Validate requires a non-empty Engine per phase and fabricating a plausible
// engine would be worse than a named unknown.
//
// Interval axis: computetrace exposes a single device duration per event
// (DeviceDurationNS). The receipt's six-way decomposition has no
// device-duration axis, so the entire device time is assigned to RecomputeNS
// (the only axis that unambiguously means "device time spent computing") and
// every other interval is left zero. Nothing is fabricated: an event with no
// DeviceDurationNS contributes zero overhead.
//
// Tokens and CacheReuseBytes have no counterpart in computetrace.Event, so they
// stay zero. BytesRead/BytesWritten/Bytes do exist and feed Phase.Bytes.
//
// Launch-context seam (ARK-1): causalreceipt must not import
// internal/perfrsiscore (cycle risk), so the launch context enters as a neutral
// attributes map[string]string. When ARK-1's *perfrsiscore.LaunchContext is
// wired to this producer, flatten the context into that map and call Produce
// unchanged.
func Produce(a computetrace.Artifact, attributes map[string]string) (Receipt, error) {
	if a.Schema != computetrace.Schema {
		return Receipt{}, fmt.Errorf("%w: unsupported compute trace schema %q", ErrInvalid, a.Schema)
	}

	type group struct {
		phase Phase
	}
	grouped := map[string]*group{}
	var order []string
	for _, e := range a.Events {
		engine := e.Backend
		if engine == "" {
			engine = "unknown"
		}
		if strings.Contains(engine, "native") && engine != "fak-native" {
			engine = "fak-native"
		}
		key := engine + "\x00" + e.Route
		g, ok := grouped[key]
		if !ok {
			g = &group{phase: Phase{
				Kind:    "compute",
				Engine:  engine,
				Backend: e.Route,
				Outcome: outcomeFor(e.Status),
			}}
			grouped[key] = g
			order = append(order, key)
		}
		p := &g.phase
		if !e.StartedAt.IsZero() {
			if p.Started.IsZero() || e.StartedAt.Before(p.Started) {
				p.Started = e.StartedAt
			}
			if end := e.StartedAt.Add(time.Duration(e.DurationNS)); end.After(p.Ended) {
				p.Ended = end
			}
		}
		p.RecomputeNS += e.DeviceDurationNS
		p.Bytes += e.BytesRead + e.BytesWritten + e.Bytes
		if p.Outcome != "failed" && outcomeFor(e.Status) == "failed" {
			p.Outcome = "failed"
		}
	}

	rcpt := Receipt{
		Schema: Schema,
		IDs: IDs{
			Work:         attributeOr(attributes, "work_id", "compute-trace"),
			Turn:         attributeOr(attributes, "turn_id", "compute-trace"),
			Graph:        attributeOr(attributes, "graph_id", "compute-trace"),
			Request:      attributeOr(attributes, "request_id", firstEventString(a, func(e computetrace.Event) string { return e.RequestID })),
			ModelSession: attributes["model_session"],
		},
		Phases:         make([]Phase, 0, len(order)),
		ModuleVersions: moduleVersions(attributes),
		Attributes:     receiptAttributes(attributes),
	}
	for n, key := range order {
		phase := grouped[key].phase
		phase.ID = fmt.Sprintf("phase-%d", n+1)
		rcpt.Phases = append(rcpt.Phases, phase)
	}
	if err := Validate(rcpt); err != nil {
		return Receipt{}, err
	}
	return rcpt, nil
}

func outcomeFor(status string) string {
	switch strings.ToLower(status) {
	case "", "ok", "success", "completed":
		return "completed"
	case "failed", "error", "timeout":
		return "failed"
	default:
		return "completed"
	}
}

// moduleVersions associates the producer's code-version SHA with the compute
// trace source, so a produced receipt records which code emitted it.
func moduleVersions(attributes map[string]string) map[string]string {
	if v := attributes["code_version"]; v != "" {
		return map[string]string{"internal/computetrace": v}
	}
	return nil
}

// receiptAttributes pins the launch-context digest under a non-sensitive key
// (the privacy guard rejects content-bearing names) and drops the raw
// code_version, which belongs in ModuleVersions rather than Attributes.
func receiptAttributes(attributes map[string]string) map[string]string {
	out := copyAttributes(attributes)
	if out == nil {
		return nil
	}
	delete(out, "code_version")
	if v := attributes["launch_digest"]; v != "" {
		delete(out, "launch_digest")
		out["launch_context_digest"] = v
	}
	return out
}

func attributeOr(attrs map[string]string, key, fallback string) string {
	if v := attrs[key]; v != "" {
		return v
	}
	return fallback
}

func copyAttributes(attrs map[string]string) map[string]string {
	if len(attrs) == 0 {
		return nil
	}
	out := make(map[string]string, len(attrs))
	for k, v := range attrs {
		out[k] = v
	}
	return out
}

func firstEventString(a computetrace.Artifact, pick func(computetrace.Event) string) string {
	sorted := append([]computetrace.Event(nil), a.Events...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Sequence < sorted[j].Sequence })
	for _, e := range sorted {
		if v := pick(e); v != "" {
			return v
		}
	}
	return "compute-trace"
}
