package sessionctl

// fatigue.go — the READ-SIDE confirm-fatigue detector (#4427, part of #2753):
// measure approval-without-inspection per confirm gate, and NAME the gates that
// have become rubber stamps.
//
// The thesis: a confirm gate that fires on every call and is waved through every
// time, with no intervening inspection, is not a safety control — it is a rubber
// stamp. A flood of per-call prompts trains the operator (or the auto-accept dial)
// to blanket-approve, which silently defeats the whole guard system. fak already
// emits the raw per-stop telemetry (the fak.guard-stop.v1 stream written by the
// Stop hook) and already owns the coarsening targets (the permission regimes of
// #2389, the set-autonomy dial of #2759) — but nothing folded that stream into an
// approval-without-inspection rate, so a gate could fire hundreds of times, be
// approved every time with zero inspection, and leave no signal at all.
//
// This file is that missing fold, and NOTHING more. Three fences bind it:
//
//   - READ-ONLY. It never mutates policy, never coarsens a gate, and never writes
//     an event. It NAMES a target; the coarsening mechanism is #2389/#2405 and the
//     autonomy dial is #2759. A high rate on a genuinely-reversible gate is the
//     signal working as intended, not a bug in that gate.
//   - NO SCHEMA CHANGE. It decodes the EXISTING fak.guard-stop.v1 rows and adds no
//     field to them. Fields it cannot find degrade to "unknown" rather than being
//     invented.
//   - NOT the static operator-surface score. `operator-heaviness-score` counts the
//     STATIC operator surface (verbs, flags, refusals). This counts RUNTIME gate
//     firing. The two measure different things and must not be double-counted.
//
// The honest-direction rule for the "unknown" case is the load-bearing detail: a
// stop whose transcript could not be read tells us nothing about whether an
// inspection happened, so it is counted in Fires but NOT in
// ApprovedWithoutInspection. Unknown therefore DILUTES the rate and can only ever
// make a gate look less rubber-stamped than it is. The detector under-reports
// rather than over-flags — the safe direction for a signal whose whole purpose is
// to justify relaxing a control.

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// FatigueEventSchema is the existing per-stop telemetry schema this detector reads.
// It is consumed, never written, and never extended here.
const FatigueEventSchema = "fak.guard-stop.v1"

// FatigueReportSchema versions the read-side report shape so a consumer can evolve
// independently of the event stream it folds.
const FatigueReportSchema = "fak.gate-fatigue.v1"

// RubberStampedFlag is the closed token a flagged row carries: this gate is a
// coarsening target, not an incident. It names the gate; it does not act on it.
const RubberStampedFlag = "RUBBER_STAMPED"

const (
	// DefaultFatigueThreshold is the SOFT bar an approval-without-inspection rate
	// must reach before a gate is named. It is deliberately below 1.0: a gate
	// inspected one time in ten is already functionally a rubber stamp.
	DefaultFatigueThreshold = 0.8
	// DefaultFatigueMinFires is the floor that keeps a gate which fired twice and
	// was waved through twice from reading as a habituated rubber stamp. Fatigue is
	// a claim about a HABIT, and a habit needs a sample.
	DefaultFatigueMinFires = 10
)

// FatigueTranscript is the bounded transcript context carried on a stop row. Read
// distinguishes "a transcript was parsed" from "no transcript / unreadable" — the
// difference between knowing there was no inspection and not knowing either way.
type FatigueTranscript struct {
	Read        bool   `json:"read"`
	LastToolUse string `json:"last_tool_use,omitempty"`
}

// FatigueEvent is one decoded fak.guard-stop.v1 row, narrowed to the fields the
// fold reads. It intentionally mirrors a SUBSET of the emitted row: unknown fields
// decode away, so the writer can add fields without breaking this reader.
type FatigueEvent struct {
	Schema      string             `json:"schema"`
	Session     string             `json:"session,omitempty"`
	Disposition string             `json:"disposition"`
	Kind        string             `json:"kind,omitempty"`
	Stage       string             `json:"stage,omitempty"`
	Mode        string             `json:"mode,omitempty"`
	Blocked     bool               `json:"blocked,omitempty"`
	Transcript  *FatigueTranscript `json:"transcript,omitempty"`
}

// FatigueRow is the per-gate fold: how often this gate fired, how often it was
// waved through, and how often that happened with nothing inspected in between.
type FatigueRow struct {
	// Key is the gate identity: stage/kind/disposition, with "-" for an absent axis.
	Key         string `json:"key"`
	Stage       string `json:"stage,omitempty"`
	Kind        string `json:"kind,omitempty"`
	Disposition string `json:"disposition,omitempty"`

	// Fires is every recorded firing of this gate — the denominator.
	Fires int `json:"fires"`
	// Approved counts firings the gate waved through (it did not block).
	Approved int `json:"approved"`
	// ApprovedWithoutInspection counts approvals we can POSITIVELY show had no
	// inspect-class tool call in the turn that ended at the gate.
	ApprovedWithoutInspection int `json:"approved_without_inspection"`
	// InspectionUnknown counts approvals whose transcript could not be read. These
	// are excluded from ApprovedWithoutInspection on purpose (see the file header):
	// unknown is not evidence of habituation.
	InspectionUnknown int `json:"inspection_unknown,omitempty"`

	// Rate is ApprovedWithoutInspection / Fires — the fatigue score.
	Rate float64 `json:"fatigue"`
	// RubberStamped is the soft-threshold verdict: over the rate bar AND past the
	// minimum-fires floor.
	RubberStamped bool `json:"rubber_stamped"`
	// Flag carries RubberStampedFlag when RubberStamped, else empty.
	Flag string `json:"flag,omitempty"`
	// Coarsen names what an operator would do about it — the suggestion, not an act.
	Coarsen string `json:"coarsen_target,omitempty"`
}

// FatigueOptions tunes the fold. The zero value selects the documented defaults.
type FatigueOptions struct {
	// Threshold is the soft rate bar (<= 0 selects DefaultFatigueThreshold).
	Threshold float64
	// MinFires is the sample floor (<= 0 selects DefaultFatigueMinFires).
	MinFires int
	// Session, when non-empty, folds only that session's stream.
	Session string
}

func (o FatigueOptions) normalize() FatigueOptions {
	if o.Threshold <= 0 {
		o.Threshold = DefaultFatigueThreshold
	}
	if o.MinFires <= 0 {
		o.MinFires = DefaultFatigueMinFires
	}
	o.Session = strings.TrimSpace(o.Session)
	return o
}

// FatigueReport is the whole read-side view: every gate's row, plus the flagged
// subset an operator would act on.
type FatigueReport struct {
	Schema    string       `json:"schema"`
	Session   string       `json:"session,omitempty"`
	Threshold float64      `json:"threshold"`
	MinFires  int          `json:"min_fires"`
	Events    int          `json:"events"`
	Rows      []FatigueRow `json:"rows,omitempty"`
	// Flagged lists the Keys of the rubber-stamped gates, most fatigued first — the
	// coarsening worklist, one row per gate identity, never one per firing.
	Flagged []string `json:"flagged,omitempty"`
}

// inspectTools is the closed inspect-class set. This is an APPROXIMATION of
// inspection, and deliberately a coarse one: seeing a Read before a confirm proves
// the operator looked at something, never that they understood it. It is good
// enough for a habituation signal and is not evidence of comprehension.
var inspectTools = map[string]bool{
	"read":         true,
	"grep":         true,
	"glob":         true,
	"notebookread": true,
	"diff":         true,
	"view":         true,
}

// inspected reports whether this stop shows an inspection, and whether we know at
// all. A row with no readable transcript returns (false, false) — unknown.
func (e FatigueEvent) inspected() (yes, known bool) {
	if e.Transcript == nil || !e.Transcript.Read {
		return false, false
	}
	tool := strings.ToLower(strings.TrimSpace(e.Transcript.LastToolUse))
	return inspectTools[tool], true
}

// approved reports whether the gate waved this firing through. The gate blocking is
// the gate doing its job; the gate NOT blocking is the approval this detector counts.
func (e FatigueEvent) approved() bool { return !e.Blocked }

// fatigueKey renders the gate identity. An absent axis renders "-" so two different
// gates can never collapse onto one key by both omitting a field.
func fatigueKey(e FatigueEvent) string {
	parts := []string{strings.TrimSpace(e.Stage), strings.TrimSpace(e.Kind), strings.TrimSpace(e.Disposition)}
	for i, p := range parts {
		if p == "" {
			parts[i] = "-"
		}
	}
	return strings.Join(parts, "/")
}

// ParseFatigueEvents decodes a fak.guard-stop.v1 JSONL stream, keeping only rows of
// that schema. Blank, malformed, and foreign lines are skipped — a ledger shared
// with another writer still folds cleanly.
func ParseFatigueEvents(content string) []FatigueEvent {
	return jsonlledger.Parse(content, func(e FatigueEvent) bool {
		return e.Schema == FatigueEventSchema
	})
}

// FoldFatigue folds a slice of stop events into the per-gate fatigue view. It is a
// pure function of its inputs: no clock, no I/O, no mutation of anything.
func FoldFatigue(events []FatigueEvent, opts FatigueOptions) FatigueReport {
	opts = opts.normalize()
	rep := FatigueReport{
		Schema:    FatigueReportSchema,
		Session:   opts.Session,
		Threshold: opts.Threshold,
		MinFires:  opts.MinFires,
	}
	byKey := map[string]*FatigueRow{}
	var order []string
	for _, e := range events {
		if e.Schema != "" && e.Schema != FatigueEventSchema {
			continue
		}
		if opts.Session != "" && e.Session != opts.Session {
			continue
		}
		rep.Events++
		key := fatigueKey(e)
		row := byKey[key]
		if row == nil {
			row = &FatigueRow{Key: key, Stage: e.Stage, Kind: e.Kind, Disposition: e.Disposition}
			byKey[key] = row
			order = append(order, key)
		}
		row.Fires++
		if !e.approved() {
			continue
		}
		row.Approved++
		yes, known := e.inspected()
		switch {
		case !known:
			row.InspectionUnknown++
		case !yes:
			row.ApprovedWithoutInspection++
		}
	}

	for _, key := range order {
		row := byKey[key]
		if row.Fires > 0 {
			row.Rate = float64(row.ApprovedWithoutInspection) / float64(row.Fires)
		}
		row.RubberStamped = row.Fires >= opts.MinFires && row.Rate >= opts.Threshold
		if row.RubberStamped {
			row.Flag = RubberStampedFlag
			row.Coarsen = "promote this per-call confirm to a coarse, witnessed regime (#2389) or an autonomy level (#2759)"
			rep.Flagged = append(rep.Flagged, row.Key)
		}
		rep.Rows = append(rep.Rows, *row)
	}
	// Most fatigued first, then most-fired, then by key — a stable read order that
	// puts the strongest coarsening target at the top.
	sort.SliceStable(rep.Rows, func(i, j int) bool {
		a, b := rep.Rows[i], rep.Rows[j]
		if a.Rate != b.Rate {
			return a.Rate > b.Rate
		}
		if a.Fires != b.Fires {
			return a.Fires > b.Fires
		}
		return a.Key < b.Key
	})
	sort.SliceStable(rep.Flagged, func(i, j int) bool { return rep.Flagged[i] < rep.Flagged[j] })
	return rep
}
