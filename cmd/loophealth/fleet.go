package main

// The fleet fold (#1196): one read-only pass over EVERY loop's ledger.
//
// The loop ladder runs on disjoint journals — loopmgr, nightrun, dojo, cadence,
// rsiloop, guardrsi, dispatch — so there is no single pane for "is the fleet of
// loops healthy". This fold answers that in one read: per loop it surfaces the
// last tick, the run/keep/witness counts, the cadence (observed from the loop's
// own ticks, or an expected default when too few), and a derived health state.
// A loop whose last tick is older than its cadence is DARK — a first-class,
// surfaced state, not silence — and a dark loop exits nonzero so a scheduler can
// gate on it. Every adapter is read-only: an unreadable or missing ledger is
// skipped-and-surfaced (rendered `unknown`), never fatal — the `fak loop rollup`
// discipline. This fold appends no event and issues no control verb: adding a
// ledger changes only the view, never any loop's behavior.

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// FleetSchema is the versioned tag a consumer pins the fleet-health shape to.
const FleetSchema = "fak-loop-fleet-health/1"

// The four health states a loop can be in.
const (
	stateLive    = "live"    // last tick well within its cadence
	stateStale   = "stale"   // past the warn fraction of its cadence (running late)
	stateDark    = "dark"    // last tick older than its cadence (missed it entirely)
	stateUnknown = "unknown" // no readable last tick (absent/empty/unrecognized)
)

// fleetWarnFraction is the fraction of a loop's cadence after which it reads
// `stale` (running late) while not yet `dark` (past its cadence entirely).
const fleetWarnFraction = 0.5

// ledgerAdapter is a read-only reader for one known loop ledger: the loop kind,
// the default path (relative to the workspace root), the expected cadence used
// only when the ledger has too few ticks to derive its own, and a pure row
// classifier that pulls a tick timestamp + keep/witness bits out of one row.
type ledgerAdapter struct {
	Kind     string
	Path     string        // relative to the workspace root
	Expected time.Duration // fallback cadence when fewer than 2 observed ticks
	Classify func(row map[string]any) (ts time.Time, ok, kept, witnessed bool)
}

// fleetAdapters is the registry of the loop ledgers the issue enumerates. Each
// is read-only and tolerant: a missing or unrecognized ledger folds to
// `unknown` rather than aborting the fleet view.
func fleetAdapters() []ledgerAdapter {
	return []ledgerAdapter{
		{Kind: "loopmgr", Path: filepath.Join(".fak", "loops.jsonl"), Expected: time.Hour, Classify: classifyLoopmgr},
		{Kind: "nightrun", Path: filepath.Join("docs", "nightrun", "collected.jsonl"), Expected: 24 * time.Hour, Classify: classifyNightrun},
		{Kind: "dojo", Path: filepath.Join("docs", "dojo", "history.jsonl"), Expected: 24 * time.Hour, Classify: classifyDojo},
		{Kind: "cadence", Path: filepath.Join("docs", "cadence", "history.jsonl"), Expected: 24 * time.Hour, Classify: classifyCadence},
		{Kind: "rsiloop", Path: filepath.Join(".dos", "rsiloop-journal.jsonl"), Expected: 24 * time.Hour, Classify: classifyRSILoop},
		{Kind: "guardrsi", Path: "guard-audit.jsonl", Expected: 24 * time.Hour, Classify: classifyGuardRSI},
		{Kind: "dispatch", Path: filepath.Join(".dispatch-runs", "progress.jsonl"), Expected: time.Hour, Classify: classifyDispatch},
	}
}

// fleetRow is one loop's folded health: its kind, ledger path, derived state,
// the tick/keep/witness counts, the last tick, its age, and both the observed
// and expected cadence behind the state.
type fleetRow struct {
	Kind             string  `json:"kind"`
	Path             string  `json:"path"`
	State            string  `json:"state"`
	Runs             int     `json:"runs"`
	Keep             int     `json:"keep"`
	Witness          int     `json:"witness"`
	LastTickUnixNano int64   `json:"last_tick_unix_nano,omitempty"`
	LastTick         string  `json:"last_tick,omitempty"`
	AgeSeconds       float64 `json:"age_seconds,omitempty"`
	CadenceObserved  float64 `json:"cadence_observed_seconds,omitempty"`
	CadenceExpected  float64 `json:"cadence_expected_seconds"`
	Note             string  `json:"note,omitempty"`
}

// fleetReport is the machine-readable fleet fold: one row per known loop ledger
// plus the count of dark loops (the gate signal).
type fleetReport struct {
	Schema     string     `json:"schema"`
	Root       string     `json:"root"`
	TSUnixNano int64      `json:"ts_unix_nano"`
	Loops      []fleetRow `json:"loops"`
	DarkCount  int        `json:"dark"`
	Note       string     `json:"note"`
}

// foldFleet is the pure cross-loop fold: read every adapter's ledger under root,
// classify its rows, and derive one health row per loop. now is injected so the
// fold is deterministic (no wall-clock, no RNG): the same bytes + same now yield
// the same report. It is read-only — it opens journals and writes nothing.
func foldFleet(root string, adapters []ledgerAdapter, now time.Time) fleetReport {
	rep := fleetReport{Schema: FleetSchema, Root: root, TSUnixNano: now.UTC().UnixNano()}
	for _, a := range adapters {
		row := foldOneLedger(root, a, now)
		if row.State == stateDark {
			rep.DarkCount++
		}
		rep.Loops = append(rep.Loops, row)
	}
	rep.Note = fmt.Sprintf("%d loop ledger(s) folded, %d dark", len(rep.Loops), rep.DarkCount)
	return rep
}

// foldOneLedger reads and folds a single adapter's ledger. A missing or
// unreadable ledger is surfaced as `unknown` with a note, never an error.
func foldOneLedger(root string, a ledgerAdapter, now time.Time) fleetRow {
	row := fleetRow{
		Kind:            a.Kind,
		Path:            a.Path,
		State:           stateUnknown,
		CadenceExpected: a.Expected.Seconds(),
	}
	path := a.Path
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, a.Path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			row.Note = "ledger absent — never ticked or not on this node"
		} else {
			row.Note = "unreadable: " + err.Error()
		}
		return row
	}

	var ticks []time.Time
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// UseNumber so a nanosecond timestamp survives intact (float64 would
		// round the low bits of a 19-digit ts_unix_nano).
		dec := json.NewDecoder(strings.NewReader(line))
		dec.UseNumber()
		var r map[string]any
		if dec.Decode(&r) != nil {
			continue // tolerate a malformed line rather than brick the fold
		}
		ts, ok, kept, witnessed := a.Classify(r)
		if !ok {
			continue
		}
		row.Runs++
		if kept {
			row.Keep++
		}
		if witnessed {
			row.Witness++
		}
		ticks = append(ticks, ts)
	}
	if len(ticks) == 0 {
		row.Note = "no parseable tick rows — empty or unrecognized shape"
		return row
	}

	sort.Slice(ticks, func(i, j int) bool { return ticks[i].Before(ticks[j]) })
	last := ticks[len(ticks)-1]
	age := now.Sub(last)
	row.LastTickUnixNano = last.UTC().UnixNano()
	row.LastTick = last.UTC().Format(time.RFC3339)
	row.AgeSeconds = mathx.Round3(age.Seconds())

	observed := observedCadence(ticks)
	effective := observed
	if observed > 0 {
		row.CadenceObserved = mathx.Round3(observed.Seconds())
	} else {
		effective = a.Expected // too few ticks to observe a cadence
	}
	row.State = healthState(age, effective)
	return row
}

// observedCadence is the mean interval between a loop's ticks: the span of the
// sorted timestamps over the gaps between them. Fewer than two ticks (or a
// zero-span burst) has no measurable cadence and returns 0.
func observedCadence(sorted []time.Time) time.Duration {
	if len(sorted) < 2 {
		return 0
	}
	span := sorted[len(sorted)-1].Sub(sorted[0])
	if span <= 0 {
		return 0
	}
	return span / time.Duration(len(sorted)-1)
}

// healthState derives the loop's state from how old its last tick is relative to
// its cadence: dark once the age exceeds the cadence (it missed its cadence
// entirely — the issue's first-class dark state), stale past the warn fraction,
// else live. A non-positive cadence is unknown (nothing to compare against).
func healthState(age, cadence time.Duration) string {
	if cadence <= 0 {
		return stateUnknown
	}
	switch {
	case age > cadence:
		return stateDark
	case age > time.Duration(float64(cadence)*fleetWarnFraction):
		return stateStale
	default:
		return stateLive
	}
}

// fleetMain folds the fleet under root and emits it (human table or --json),
// returning exit 3 when any loop is dark so a scheduler can gate on it, 2 on an
// encode error, else 0.
func fleetMain(stdout, stderr io.Writer, root string, asJSON bool, now time.Time) int {
	rep := foldFleet(root, fleetAdapters(), now)
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(stderr, "loophealth:", err)
			return 2
		}
	} else {
		renderFleet(stdout, rep)
	}
	if rep.DarkCount > 0 {
		return 3
	}
	return 0
}

// renderFleet prints the fleet fold as one aligned row per loop, reusing the
// `fak loop rollup` tabwriter idiom, then the dark-loop tally.
func renderFleet(w io.Writer, rep fleetReport) {
	fmt.Fprintf(w, "fak loop health — fleet fold across %d loop ledger(s) under %s\n\n", len(rep.Loops), rep.Root)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "LOOP\tSTATE\tRUNS\tKEEP\tWITNESS\tCADENCE\tEXPECTED\tLAST\tNOTE")
	for _, l := range rep.Loops {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\t%s\n",
			l.Kind, strings.ToUpper(l.State), l.Runs, l.Keep, l.Witness,
			humanDur(l.CadenceObserved), humanDur(l.CadenceExpected), lastTickLabel(l), l.Note)
	}
	_ = tw.Flush()
	fmt.Fprintf(w, "\n%d dark loop(s) — exit 3 when any loop is dark\n", rep.DarkCount)
}

func lastTickLabel(l fleetRow) string {
	if l.LastTickUnixNano == 0 {
		return "-"
	}
	return l.LastTick
}

// humanDur renders a duration (seconds) in its dominant unit, "-" when there is
// no measurable value.
func humanDur(sec float64) string {
	switch {
	case sec <= 0:
		return "-"
	case sec >= 86400:
		return fmt.Sprintf("%.1fd", sec/86400)
	case sec >= 3600:
		return fmt.Sprintf("%.1fh", sec/3600)
	case sec >= 60:
		return fmt.Sprintf("%.1fm", sec/60)
	default:
		return fmt.Sprintf("%.0fs", sec)
	}
}

// --- per-ledger classifiers (each read-only, each tolerant of a foreign row) ---

func classifyLoopmgr(r map[string]any) (time.Time, bool, bool, bool) {
	ts, ok := tickTime(r, "ts_unix_nano")
	if !ok {
		return time.Time{}, false, false, false
	}
	kind := strings.ToLower(asString(r["kind"]))
	status := strings.ToLower(asString(r["status"]))
	kept := status == "claimed-done" || status == "witnessed"
	return ts, true, kept, kind == "witness"
}

func classifyNightrun(r map[string]any) (time.Time, bool, bool, bool) {
	ts, ok := tickTime(r, "generated_at", "date")
	if !ok {
		return time.Time{}, false, false, false
	}
	outcome := strings.ToLower(asString(r["outcome"]))
	kept := outcome == "collected"
	_, witnessed := r["number"].(json.Number)
	return ts, true, kept, witnessed
}

func classifyDojo(r map[string]any) (time.Time, bool, bool, bool) {
	ts, ok := tickTime(r, "generated_at", "date")
	if !ok {
		return time.Time{}, false, false, false
	}
	kept := strings.EqualFold(asString(r["verdict"]), "OK")
	return ts, true, kept, asInt(r["calibrated"]) > 0
}

func classifyCadence(r map[string]any) (time.Time, bool, bool, bool) {
	ts, ok := tickTime(r, "generated_at", "date")
	if !ok {
		return time.Time{}, false, false, false
	}
	kept := strings.EqualFold(asString(r["verdict"]), "OK")
	return ts, true, kept, asInt(r["work_ships"]) > 0
}

func classifyRSILoop(r map[string]any) (time.Time, bool, bool, bool) {
	ts, ok := tickTime(r, "ts_unix_nano", "ts", "generated_at")
	if !ok {
		return time.Time{}, false, false, false
	}
	verdict := strings.ToUpper(asString(r["verdict"]))
	return ts, true, verdict == "KEEP", verdict == "KEEP" || verdict == "REVERT"
}

func classifyGuardRSI(r map[string]any) (time.Time, bool, bool, bool) {
	ts, ok := tickTime(r, "ts_unix_nano", "ts")
	if !ok {
		return time.Time{}, false, false, false
	}
	kept := !strings.EqualFold(asString(r["decision"]), "deny")
	return ts, true, kept, asString(r["hash"]) != ""
}

func classifyDispatch(r map[string]any) (time.Time, bool, bool, bool) {
	ts, ok := tickTime(r, "utc", "ts")
	if !ok {
		return time.Time{}, false, false, false
	}
	return ts, true, asBool(r["ok"]), asInt(r["closed_now"]) > 0
}

// tickTime returns the first present timestamp among fields: an integer field is
// read as unix nanoseconds (ts_unix_nano), a string field as an RFC3339 or
// date-only stamp. Returns false when no field yields a usable time.
func tickTime(r map[string]any, fields ...string) (time.Time, bool) {
	for _, f := range fields {
		v, present := r[f]
		if !present {
			continue
		}
		switch x := v.(type) {
		case json.Number:
			if i, err := x.Int64(); err == nil && i > 0 {
				return time.Unix(0, i).UTC(), true
			}
		case float64:
			if x > 0 {
				return time.Unix(0, int64(x)).UTC(), true
			}
		case string:
			if t, ok := parseTimeString(x); ok {
				return t, true
			}
		}
	}
	return time.Time{}, false
}

// parseTimeString parses an RFC3339 (with or without sub-seconds) or a date-only
// stamp, normalized to UTC.
func parseTimeString(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), true
		}
	}
	return time.Time{}, false
}

func asInt(v any) int64 {
	switch n := v.(type) {
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return i
		}
		if f, err := n.Float64(); err == nil {
			return int64(f)
		}
	case float64:
		return int64(n)
	}
	return 0
}

func asBool(v any) bool {
	b, ok := v.(bool)
	return ok && b
}
