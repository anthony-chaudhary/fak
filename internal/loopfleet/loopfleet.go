// Package loopfleet is the cross-ledger loop-health fold (#1196, part of #1173 —
// the verified loop): one read-only pane that answers "show me EVERY loop's
// health — last tick, run count, keep/witness rate, and whether it has gone DARK"
// across the repo's fragmented loop ledgers.
//
// The loop ladder runs on disjoint journals, each owned by a different package and
// written in its own schema:
//
//	loopmgr   .fak/loops.jsonl            (fak.loop-event.v1)
//	nightrun  docs/nightrun/collected.jsonl (fak-nightrun-collect/1)
//	dojo      docs/dojo/history.jsonl       (fak-dojo-ledger/1)
//	cadence   docs/cadence/history.jsonl    (fak-cadence-ledger/1)
//	dispatch  .dispatch-runs/progress.jsonl (fleet-issue-resolve-progress/1)
//
// loopmgr.FoldHealth already unifies the two ledgers reachable WITHIN loopmgr
// (the loop-event ledger + the job registry) but, by design, does not import the
// other packages' journals. This package is the cross-ledger surface that comment
// names: each ledger gets a small read-only adapter, and one Fold joins them into a
// single per-loop health view that reuses loopmgr's HealthState vocabulary so the
// whole ladder speaks one word for "live"/"stale"/"dark"/"unknown".
//
// It is PURE and READ-ONLY: every adapter only os.ReadFile's its journal, `now` is
// supplied (no clock read in the fold), and nothing is appended or mutated. An
// unreadable or missing ledger is skipped-and-surfaced (recorded in Report.Skipped),
// never fatal — the `fak loop rollup` discipline. Adding a ledger to the set changes
// only the VIEW, never any loop's behavior.
//
// Scope (honest): this folds 5 of the 7 ledgers the issue names. The two not yet
// wired — rsiloop (its Row carries no timestamp, so no last-tick is derivable) and
// guardrsi (a config-dir directory of guard-audit.jsonl files, not a single
// repo-relative file) — are deliberate follow-ons; each would be one more adapter
// here with the same read-only contract. The acceptance bar is "≥3 of the listed
// ledgers", which 5 clears.
package loopfleet

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// Schema is the versioned payload tag so a `--json` consumer can pin the shape.
const Schema = "fak.loop-fleet-health.v1"

// LoopHealth is one loop's row in the cross-ledger pane: its identity (kind +
// source ledger), the inputs the verdict was derived from (last tick, cadence,
// counts), and the derived HealthState — so a reader sees not just "dark" but the
// cadence and last tick that made it dark.
type LoopHealth struct {
	// Kind is the loop's stable identity. For the single-purpose ledgers it is the
	// ledger name ("nightrun", "dojo", "cadence", "dispatch"); for loopmgr — whose
	// one ledger holds many loops — it is "loopmgr:<loop_id>".
	Kind string `json:"kind"`
	// Ledger names the journal this row was folded from.
	Ledger string `json:"ledger"`
	// State is the derived verdict (live/stale/dark/unknown), loopmgr's vocabulary.
	State loopmgr.HealthState `json:"state"`
	// Dark is the surfaced boolean the issue makes first-class: true iff State==dark,
	// so a `--json` consumer (or a scheduler) gates on one field without re-deriving.
	Dark bool `json:"dark"`
	// LastTickUnixNano is the loop's most recent ledger event time, 0 if never ticked.
	LastTickUnixNano int64 `json:"last_tick_unix_nano,omitempty"`
	// AgeSeconds is now-lastTick in whole seconds; 0 when the loop has never ticked.
	AgeSeconds int64 `json:"age_seconds,omitempty"`
	// CadenceSeconds is the expected interval the verdict compared the age against.
	CadenceSeconds int64 `json:"cadence_seconds,omitempty"`
	// Runs is the count of recorded runs/ticks — the denominator of the keep rate.
	Runs int `json:"runs"`
	// Keep is the count of runs that landed a kept/positive outcome.
	Keep int `json:"keep"`
	// Witness is the count of runs that carried an independent witness (an artifact,
	// a measurement, a witnessed-done verdict) — the evidence behind the keep.
	Witness int `json:"witness"`
	// KeepRate is Keep/Runs rounded to 3 decimals; -1 when Runs==0 (no denominator),
	// so a brand-new loop is never slandered as 0% kept on an empty base.
	KeepRate float64 `json:"keep_rate"`
}

// Skipped is a ledger that could not be folded — missing, unreadable, or holding no
// parseable rows. It is surfaced (never silent) so "absent" reads as a known gap,
// not as a healthy zero.
type Skipped struct {
	Ledger string `json:"ledger"`
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// Rollup is the fleet-wide tally: how many loops, how many in each state, and how
// many ledgers were folded vs skipped. Dark>0 is the signal a scheduler gates on.
type Rollup struct {
	Loops   int `json:"loops"`
	Live    int `json:"live"`
	Stale   int `json:"stale"`
	Dark    int `json:"dark"`
	Unknown int `json:"unknown"`
	Ledgers int `json:"ledgers"` // ledgers present and folded
	Skipped int `json:"skipped"` // ledgers missing/unreadable/empty
}

// Report is the full read-only fold: schema tag, the time it was folded, the
// per-loop rows (stable kind order), the surfaced skipped ledgers, and the rollup.
type Report struct {
	Schema     string       `json:"schema"`
	TSUnixNano int64        `json:"ts_unix_nano"`
	Loops      []LoopHealth `json:"loops"`
	Skipped    []Skipped    `json:"skipped"`
	Rollup     Rollup       `json:"rollup"`
}

// rawLoop is the normalized intermediate every adapter produces: one loop's
// identity plus the raw counts and last tick, before the health state is derived.
type rawLoop struct {
	kind             string
	lastTickUnixNano int64
	runs             int
	keep             int
	witness          int
}

// adapter is one ledger's read-only descriptor: its id, its repo-relative path, the
// cadence a loop of this kind is expected to tick within, and a fold that turns the
// parsed JSONL rows into zero or more rawLoops. Keeping the path + cadence here (not
// in the fold) makes the adapter table the single place a ledger is declared.
type adapter struct {
	id      string
	relPath string
	cadence int64
	fold    func(rows []map[string]any) []rawLoop
}

// adapters is the registered ledger set. To wire another journal, add a row here —
// the Fold loop, the skip-and-surface handling, and the rollup all flow from it.
func adapters() []adapter {
	return []adapter{
		{id: "nightrun", relPath: filepath.Join("docs", "nightrun", "collected.jsonl"), cadence: dailyCadence, fold: foldNightrun},
		{id: "dojo", relPath: filepath.Join("docs", "dojo", "history.jsonl"), cadence: dailyCadence, fold: foldDojo},
		{id: "cadence", relPath: filepath.Join("docs", "cadence", "history.jsonl"), cadence: dailyCadence, fold: foldCadence},
		{id: "dispatch", relPath: filepath.Join(".dispatch-runs", "progress.jsonl"), cadence: hourlyCadence, fold: foldDispatch},
	}
}

// Cadence horizons a loop of a kind is expected to tick within. A daily loop quiet
// for more than a day is slipping; a fast dispatch loop quiet for an hour is.
const (
	dailyCadence  int64 = 24 * 60 * 60
	hourlyCadence int64 = 60 * 60
)

// Fold is the cross-ledger health fold. It runs every adapter against root,
// derives one LoopHealth per loop found, surfaces every skipped ledger, and tallies
// the rollup. It is PURE: `now` is supplied, the only I/O is reading the journals,
// and no input is mutated. A zero-value HealthThresholds is fine — the classifier
// fills the defaults.
func Fold(root string, now time.Time, th loopmgr.HealthThresholds) Report {
	var loops []LoopHealth
	var skipped []Skipped

	// loopmgr's own ledger folds through its hash-chain-aware reader, not the generic
	// JSONL path — one row per loop_id it holds.
	mgrLoops, mgrSkip := foldLoopmgr(root, now, th)
	loops = append(loops, mgrLoops...)
	if mgrSkip != nil {
		skipped = append(skipped, *mgrSkip)
	}

	for _, a := range adapters() {
		path := filepath.Join(root, a.relPath)
		rows, reason := readJSONL(path)
		if reason != "" {
			skipped = append(skipped, Skipped{Ledger: a.id, Path: a.relPath, Reason: reason})
			continue
		}
		raws := a.fold(rows)
		if len(raws) == 0 {
			skipped = append(skipped, Skipped{Ledger: a.id, Path: a.relPath, Reason: "no parseable rows"})
			continue
		}
		for _, raw := range raws {
			loops = append(loops, deriveRow(a.id, raw, a.cadence, now, th))
		}
	}

	sort.Slice(loops, func(i, j int) bool { return loops[i].Kind < loops[j].Kind })
	return Report{
		Schema:     Schema,
		TSUnixNano: now.UTC().UnixNano(),
		Loops:      loops,
		Skipped:    skipped,
		Rollup:     rollup(loops, len(skipped)),
	}
}

// foldLoopmgr folds .fak/loops.jsonl into one row per loop_id via loopmgr's own
// snapshot reader. A missing/unreadable ledger is surfaced as a single Skipped.
func foldLoopmgr(root string, now time.Time, th loopmgr.HealthThresholds) ([]LoopHealth, *Skipped) {
	rel := filepath.Join(".fak", "loops.jsonl")
	path := filepath.Join(root, rel)
	if _, err := os.Stat(path); err != nil {
		reason := "absent"
		if !os.IsNotExist(err) {
			reason = err.Error()
		}
		return nil, &Skipped{Ledger: "loopmgr", Path: rel, Reason: reason}
	}
	st, err := loopmgr.SnapshotFile(path, now)
	if err != nil {
		return nil, &Skipped{Ledger: "loopmgr", Path: rel, Reason: err.Error()}
	}
	if len(st.Loops) == 0 {
		return nil, &Skipped{Ledger: "loopmgr", Path: rel, Reason: "no parseable rows"}
	}
	cadence := defaultCadence(th)
	out := make([]LoopHealth, 0, len(st.Loops))
	for _, snap := range st.Loops {
		out = append(out, deriveRow("loopmgr", rawLoop{
			kind:             "loopmgr:" + snap.LoopID,
			lastTickUnixNano: snap.LastEventUnixNano,
			runs:             int(snap.Ended),
			keep:             int(snap.Witnessed),
			witness:          int(snap.Witnessed),
		}, cadence, now, th))
	}
	return out, nil
}

// deriveRow turns a rawLoop + its cadence into a finished LoopHealth: keep rate,
// age, and the derived health state.
func deriveRow(ledger string, raw rawLoop, cadence int64, now time.Time, th loopmgr.HealthThresholds) LoopHealth {
	row := LoopHealth{
		Kind:             raw.kind,
		Ledger:           ledger,
		LastTickUnixNano: raw.lastTickUnixNano,
		CadenceSeconds:   cadence,
		Runs:             raw.runs,
		Keep:             raw.keep,
		Witness:          raw.witness,
	}
	if raw.runs > 0 {
		row.KeepRate = round3(float64(raw.keep) / float64(raw.runs))
	} else {
		row.KeepRate = -1
	}
	if raw.lastTickUnixNano > 0 {
		age := now.UTC().UnixNano() - raw.lastTickUnixNano
		if age < 0 {
			age = 0
		}
		row.AgeSeconds = age / int64(time.Second)
	}
	row.State = classify(raw.lastTickUnixNano, cadence, now, th)
	// A loop with recorded runs but NO usable last tick (every ledger row missing
	// or carrying an unparseable timestamp) is not "never fired" — classify would
	// call it DARK, which reads as "registered but never ticked" and would trip a
	// scheduler into reviving a loop that demonstrably ran. We have proof it ran
	// (runs > 0) but cannot place it on the freshness timeline, so the honest
	// verdict is UNKNOWN (decline to judge liveness), not DARK. A genuinely empty
	// loop (runs == 0, no tick) stays DARK — that one really has never fired.
	if raw.runs > 0 && raw.lastTickUnixNano <= 0 && row.State == loopmgr.HealthDark {
		row.State = loopmgr.HealthUnknown
	}
	row.Dark = row.State == loopmgr.HealthDark
	return row
}

// classify derives a loop's liveness from its last tick against a cadence. It
// MIRRORS loopmgr.deriveState (unexported) so the cross-ledger pane and loopmgr's
// own fold draw the dark line identically; it references loopmgr's exported
// thresholds + constants so the defaults can never drift apart:
//   - never ticked, known cadence: DARK (registered/ledgered but never fired).
//   - never ticked, no cadence:    UNKNOWN (decline to judge).
//   - ticked, age <= cadence:                LIVE.
//   - ticked, cadence < age <= darkMul*cadence: STALE (slipping).
//   - ticked, age > darkMul*cadence:         DARK (gone quiet past its cadence).
func classify(lastTickUnixNano, cadenceSeconds int64, now time.Time, th loopmgr.HealthThresholds) loopmgr.HealthState {
	if lastTickUnixNano <= 0 {
		if cadenceSeconds <= 0 {
			return loopmgr.HealthUnknown
		}
		return loopmgr.HealthDark
	}
	if cadenceSeconds <= 0 {
		return loopmgr.HealthUnknown
	}
	ageNanos := now.UTC().UnixNano() - lastTickUnixNano
	if ageNanos < 0 {
		ageNanos = 0
	}
	cadenceNanos := cadenceSeconds * int64(time.Second)
	darkNanos := darkMultiple(th) * cadenceNanos
	switch {
	case ageNanos <= cadenceNanos:
		return loopmgr.HealthLive
	case ageNanos <= darkNanos:
		return loopmgr.HealthStale
	default:
		return loopmgr.HealthDark
	}
}

func darkMultiple(th loopmgr.HealthThresholds) int64 {
	if th.DarkMultiple >= 1 {
		return th.DarkMultiple
	}
	return loopmgr.DefaultDarkMultiple
}

func defaultCadence(th loopmgr.HealthThresholds) int64 {
	if th.DefaultCadenceSeconds > 0 {
		return th.DefaultCadenceSeconds
	}
	return loopmgr.DefaultHealthCadenceSeconds
}

func rollup(loops []LoopHealth, skipped int) Rollup {
	r := Rollup{Loops: len(loops), Skipped: skipped}
	ledgers := map[string]struct{}{}
	for _, l := range loops {
		ledgers[l.Ledger] = struct{}{}
		switch l.State {
		case loopmgr.HealthLive:
			r.Live++
		case loopmgr.HealthStale:
			r.Stale++
		case loopmgr.HealthDark:
			r.Dark++
		case loopmgr.HealthUnknown:
			r.Unknown++
		}
	}
	r.Ledgers = len(ledgers)
	return r
}

// --- per-ledger folds (each read-only; each documents its keep/witness mapping) ---

// foldNightrun: one collection loop. A run kept = outcome reads as success; a run
// witnessed = it wrote an artifact (the log path the row carries).
func foldNightrun(rows []map[string]any) []rawLoop {
	lp := rawLoop{kind: "nightrun"}
	for _, r := range rows {
		lp.runs++
		bumpLastTick(&lp, asString(r["generated_at"]))
		if isSuccess(asString(r["outcome"])) {
			lp.keep++
		}
		if strings.TrimSpace(asString(r["artifact"])) != "" {
			lp.witness++
		}
	}
	return single(lp)
}

// foldDojo: the dojo gym loop. A run kept = verdict OK; witnessed = it measured at
// least one episode (the calibration evidence).
func foldDojo(rows []map[string]any) []rawLoop {
	lp := rawLoop{kind: "dojo"}
	for _, r := range rows {
		lp.runs++
		bumpLastTick(&lp, asString(r["generated_at"]))
		if strings.EqualFold(asString(r["verdict"]), "OK") {
			lp.keep++
		}
		if f, ok := asFloat(r["measured"]); ok && f > 0 {
			lp.witness++
		}
	}
	return single(lp)
}

// foldCadence: the cadence-report loop. A run kept = verdict OK; witnessed = it
// bound a commit (the run's git anchor).
func foldCadence(rows []map[string]any) []rawLoop {
	lp := rawLoop{kind: "cadence"}
	for _, r := range rows {
		lp.runs++
		bumpLastTick(&lp, asString(r["generated_at"]))
		if strings.EqualFold(asString(r["verdict"]), "OK") {
			lp.keep++
		}
		if strings.TrimSpace(asString(r["commit"])) != "" {
			lp.witness++
		}
	}
	return single(lp)
}

// foldDispatch: the issue-resolve dispatch loop. A tick kept = ok==true; witnessed
// = the tick closed or witnessed an issue this round.
func foldDispatch(rows []map[string]any) []rawLoop {
	lp := rawLoop{kind: "dispatch"}
	for _, r := range rows {
		lp.runs++
		bumpLastTick(&lp, asString(r["utc"]))
		if b, ok := r["ok"].(bool); ok && b {
			lp.keep++
		}
		if positive(r["closed_now"]) || positive(r["witnessed_open"]) {
			lp.witness++
		}
	}
	return single(lp)
}

// --- shared row helpers ---

// bumpLastTick advances lp.lastTickUnixNano to the row's RFC3339 timestamp if it is
// newer (ledgers are append-ordered but a max is order-independent and robust).
func bumpLastTick(lp *rawLoop, rfc3339 string) {
	if ts, ok := parseRFC3339(rfc3339); ok && ts > lp.lastTickUnixNano {
		lp.lastTickUnixNano = ts
	}
}

func single(lp rawLoop) []rawLoop {
	if lp.runs == 0 {
		return nil
	}
	return []rawLoop{lp}
}

func isSuccess(outcome string) bool {
	switch strings.ToLower(strings.TrimSpace(outcome)) {
	case "ok", "success", "succeeded", "passed", "pass":
		return true
	}
	return false
}

// readJSONL reads a JSONL ledger into parsed rows, tolerating malformed lines. The
// returned reason is "" on success, else the surfaced skip reason ("absent", an IO
// error, or "no parseable rows" for a present-but-blank file).
func readJSONL(path string) ([]map[string]any, string) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, "absent"
		}
		return nil, err.Error()
	}
	var rows []map[string]any
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue // tolerate a malformed line rather than brick the fold
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, "no parseable rows"
	}
	return rows, ""
}

func parseRFC3339(s string) (int64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return 0, false
	}
	return t.UTC().UnixNano(), true
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}

func positive(v any) bool {
	f, ok := asFloat(v)
	return ok && f > 0
}

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }
