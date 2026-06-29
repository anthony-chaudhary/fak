package loopmgr

import (
	"math"
	"sort"
	"time"
)

// Health is the read-only fold that answers "show me EVERY loop's health" in one
// pane: per loop a last-tick, a run count, a keep/witness rate, and whether it has
// gone DARK (registered/ledgered but not ticking past its cadence). It is the
// observability rung for the loop ladder (#1196, part of #1173 — the verified loop).
//
// It is a PURE PROJECTION over what loopmgr already records — the folded loop
// ledger (Summarize -> Status) and the on-disk job Registry (the cadence
// DEFINITION) — not new tracking. FoldHealth reads those two inputs plus `now` and
// returns rows; it appends no event, mutates nothing, and issues no control verb.
// Adding a loop to either input changes only the VIEW, never any loop's behavior.
//
// Scope boundary (honest): the issue names up to SEVEN fragmented loop ledgers
// (loopmgr, nightrun, dojo, cadence, rsiloop, guardrsi, dispatch) that live across
// different packages. This fold unifies the two reachable IN loopmgr — the loop
// event ledger and the job registry. Folding the other five behind one read-only
// adapter interface, without loopmgr importing those packages, is the cross-ledger
// unification follow-on (the `fak loop health` CLI in cmd/fak is the surface that
// wires the adapters; this core is what it calls).

// HealthSchema is the versioned payload tag so a consumer (a `--json` CLI, the
// scorecard) can pin the shape it folds.
const HealthSchema = "fak.loop-health.v1"

// MetricLearningDebt is the loop metric key the learning-docs freshness loop
// records when it measures the current docs scorecard debt.
const MetricLearningDebt = "learning_debt"

// HealthState is the derived per-loop health verdict. It is a CLOSED set with no
// zero-value default in the rendered row: every row carries an explicit state so a
// reader never has to guess what an empty string means. A DARK loop — one that is
// registered/ledgered but has not ticked within its cadence (or has never ticked at
// all) — is a first-class, surfaced state, not silence.
type HealthState string

const (
	// HealthLive: the loop ticked within its cadence window (or, when no cadence is
	// known, ticked recently enough by the default staleness horizon).
	HealthLive HealthState = "live"
	// HealthStale: the loop has ticked, but its last tick is older than its cadence
	// (or the default horizon) by less than the dark multiple — slipping, not yet dark.
	HealthStale HealthState = "stale"
	// HealthDark: the loop is registered or has a ledger entry but is not ticking —
	// last tick older than DarkMultiple cadences, or it has NEVER ticked. The state
	// the issue makes first-class: a dark loop is surfaced, never silent.
	HealthDark HealthState = "dark"
	// HealthUnknown: the loop has no cadence and no usable last tick, so liveness
	// cannot be derived. Distinct from dark — we decline to judge rather than judge
	// wrongly. (Never returned for a registered job; a registered job has a cadence.)
	HealthUnknown HealthState = "unknown"
)

// HealthThresholds tunes the staleness derivation. The zero value is usable:
// FoldHealth fills any unset field with a documented default, so a caller can pass
// HealthThresholds{} and get sane behavior.
type HealthThresholds struct {
	// DefaultCadenceSeconds is the staleness horizon for a loop that has a ledger
	// entry but NO registered cadence (e.g. an ad-hoc `fak loop run`). 0 -> the
	// DefaultHealthCadenceSeconds default. A loop quieter than this many seconds is
	// stale; quieter than DarkMultiple of it is dark.
	DefaultCadenceSeconds int64 `json:"default_cadence_seconds,omitempty"`

	// DarkMultiple is how many whole cadences a loop may miss before it is DARK
	// rather than merely STALE. Must be >= 1; 0 -> DefaultDarkMultiple. A last tick
	// older than DarkMultiple*cadence is dark.
	DarkMultiple int64 `json:"dark_multiple,omitempty"`
}

// Defaults for the staleness derivation.
const (
	// DefaultHealthCadenceSeconds is the staleness horizon (1h) applied to a loop
	// with a ledger entry but no registered cadence.
	DefaultHealthCadenceSeconds int64 = 3600
	// DefaultDarkMultiple: a loop dark once it has missed this many whole cadences.
	DefaultDarkMultiple int64 = 2
)

func (t HealthThresholds) defaultCadenceSeconds() int64 {
	if t.DefaultCadenceSeconds > 0 {
		return t.DefaultCadenceSeconds
	}
	return DefaultHealthCadenceSeconds
}

func (t HealthThresholds) darkMultiple() int64 {
	if t.DarkMultiple >= 1 {
		return t.DarkMultiple
	}
	return DefaultDarkMultiple
}

// HealthRow is one loop's health line. It carries the loop's identity, the inputs
// the verdict was derived from (last tick, cadence, counts), and the derived
// HealthState — so a reader sees not just "dark" but the cadence and last tick that
// made it dark.
type HealthRow struct {
	// LoopID is the loop / job identity (the ledger's LoopID, the registry's JobID).
	LoopID string `json:"loop_id"`

	// State is the derived verdict (live/stale/dark/unknown).
	State HealthState `json:"state"`

	// Dark is the surfaced boolean the issue asks for: true iff State == HealthDark.
	// Carried explicitly so a `--json` consumer can gate on one field without
	// re-deriving the state string.
	Dark bool `json:"dark"`

	// Registered is true when this loop has a cadence definition in the registry —
	// i.e. it is a scheduled job, not just an ad-hoc ledger entry. A registered loop
	// with no ledger tick at all is the canonical "dark loop": known to the schedule,
	// never observed firing.
	Registered bool `json:"registered"`

	// Ledgered is true when this loop has at least one event in the loop ledger.
	Ledgered bool `json:"ledgered"`

	// CadenceSeconds is the cadence the verdict used: the registered interval when
	// known, else the default horizon. 0 only when neither is available (unknown).
	CadenceSeconds int64 `json:"cadence_seconds,omitempty"`

	// CadenceSource names where CadenceSeconds came from: "registry" (a registered
	// job's interval) or "default" (the no-cadence horizon). Empty when unknown.
	CadenceSource string `json:"cadence_source,omitempty"`

	// LastTickUnixNano is the loop's most recent ledger event time, 0 if it has
	// never ticked.
	LastTickUnixNano int64 `json:"last_tick_unix_nano,omitempty"`

	// AgeSeconds is now - last tick in whole seconds, the staleness the verdict read.
	// 0 when the loop has never ticked (Dark by the never-ticked rule, not by age).
	AgeSeconds int64 `json:"age_seconds,omitempty"`

	// Runs is the run count: ended runs from the ledger fold (the unit the keep-rate
	// is over). A loop that has fired but not yet ended a run reads 0 here.
	Runs uint64 `json:"runs"`

	// Witnessed is the count of runs that ended with an independent witnessed-done
	// verdict — the numerator of the keep rate.
	Witnessed uint64 `json:"witnessed"`

	// KeepRate is the witness/keep rate: Witnessed / Runs, the fraction of ended runs
	// that reached a witnessed-done verdict. -1 when Runs == 0 (no denominator) so a
	// brand-new loop is never reported as 0% kept on an empty base — the absence of a
	// rate is distinct from a rate of zero.
	KeepRate float64 `json:"keep_rate"`

	// LastState is the loop's last folded state string (the ledger's word on what it
	// was doing), carried for context. Not the health verdict — that is State.
	LastState string `json:"last_state,omitempty"`

	// LearningDebt is the latest learning_debt metric recorded by the loop ledger.
	// Nil means the loop has not measured that metric yet; 0 is a real debt-free
	// measurement and must still be rendered.
	LearningDebt *int64 `json:"learning_debt,omitempty"`
}

// HealthRollup is the fleet-wide summary across all rows: how many loops, and how
// many in each derived state. The roll-up answers "is the fleet of loops healthy"
// in one line; Dark > 0 is the signal a scheduler gates on.
type HealthRollup struct {
	Loops      int `json:"loops"`
	Live       int `json:"live"`
	Stale      int `json:"stale"`
	Dark       int `json:"dark"`
	Unknown    int `json:"unknown"`
	Registered int `json:"registered"`
	Ledgered   int `json:"ledgered"`
}

// HealthReport is the full read-only fold: the schema tag, the time it was folded,
// the per-loop rows (stable loop-id order), and the roll-up.
type HealthReport struct {
	Schema     string       `json:"schema"`
	TSUnixNano int64        `json:"ts_unix_nano"`
	Rows       []HealthRow  `json:"rows"`
	Rollup     HealthRollup `json:"rollup"`
}

// FoldHealth is the read-only health fold. It joins the folded loop ledger (st, from
// Summarize) with the job registry (reg, the cadence definition) and derives one
// HealthRow per loop seen in EITHER input, plus a roll-up. It is PURE: no clock read
// (now is supplied), no I/O, no mutation of its inputs.
//
// The union is deliberate. A loop present in the ledger but absent from the registry
// is an ad-hoc loop (judged against the default horizon). A loop present in the
// registry but absent from the ledger is the canonical DARK loop — registered to a
// schedule, never observed ticking — which a ledger-only fold would render as silence.
// Folding the union surfaces it.
func FoldHealth(st Status, reg Registry, now time.Time, th HealthThresholds) HealthReport {
	snaps := map[string]LoopSnapshot{}
	for _, loop := range st.Loops {
		snaps[loop.LoopID] = loop
	}
	jobs := map[string]Job{}
	for _, job := range reg.List() {
		jobs[job.JobID()] = job
	}

	ids := unionIDs(snaps, jobs)
	rows := make([]HealthRow, 0, len(ids))
	for _, id := range ids {
		snap, ledgered := snaps[id]
		job, registered := jobs[id]
		rows = append(rows, healthRow(id, snap, ledgered, job, registered, now, th))
	}

	return HealthReport{
		Schema:     HealthSchema,
		TSUnixNano: now.UTC().UnixNano(),
		Rows:       rows,
		Rollup:     rollup(rows),
	}
}

// unionIDs returns the sorted union of loop ids across the ledger snapshots and the
// registered jobs, so a loop in either input gets a row and the order is deterministic.
func unionIDs(snaps map[string]LoopSnapshot, jobs map[string]Job) []string {
	seen := map[string]struct{}{}
	for id := range snaps {
		seen[id] = struct{}{}
	}
	for id := range jobs {
		seen[id] = struct{}{}
	}
	ids := make([]string, 0, len(seen))
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// healthRow derives one loop's row from its (optional) ledger snapshot and its
// (optional) registry job. The state derivation is fixed-order so the verdict is
// deterministic: never-ticked-but-known -> dark; otherwise classify by age against
// the cadence.
func healthRow(id string, snap LoopSnapshot, ledgered bool, job Job, registered bool, now time.Time, th HealthThresholds) HealthRow {
	row := HealthRow{
		LoopID:           id,
		Registered:       registered,
		Ledgered:         ledgered,
		LastTickUnixNano: snap.LastEventUnixNano,
		Runs:             snap.Ended,
		Witnessed:        snap.Witnessed,
		LastState:        snap.State,
	}
	if v, ok := snap.Metrics[MetricLearningDebt]; ok {
		vv := v
		row.LearningDebt = &vv
	}

	// Keep rate: witnessed / ended runs. -1 (no rate) on an empty denominator so a
	// new loop is not slandered as 0% kept.
	if row.Runs > 0 {
		row.KeepRate = round3(float64(row.Witnessed) / float64(row.Runs))
	} else {
		row.KeepRate = -1
	}

	// Cadence: a registered job's interval is the truth; else the default horizon.
	cadence := int64(0)
	if registered && job.Schedule.IntervalSeconds > 0 {
		cadence = job.Schedule.IntervalSeconds
		row.CadenceSource = "registry"
	} else {
		cadence = th.defaultCadenceSeconds()
		row.CadenceSource = "default"
	}
	row.CadenceSeconds = cadence

	row.State = deriveState(snap.LastEventUnixNano, cadence, now, th)
	if row.State != HealthDark || snap.LastEventUnixNano > 0 {
		// Age is meaningful only when the loop has ticked; a never-ticked dark loop
		// leaves AgeSeconds at 0 (the never-ticked rule, not an age).
		if snap.LastEventUnixNano > 0 {
			age := now.UTC().UnixNano() - snap.LastEventUnixNano
			if age < 0 {
				age = 0
			}
			row.AgeSeconds = age / int64(time.Second)
		}
	}
	row.Dark = row.State == HealthDark
	return row
}

// deriveState classifies a loop's liveness from its last tick against a cadence.
// The rules, in order:
//   - never ticked (lastTick <= 0): DARK — a registered/ledgered loop that has never
//     fired is dark, not unknown; if cadence is also unknown (<=0) it is UNKNOWN.
//   - ticked, age <= cadence: LIVE.
//   - ticked, cadence < age <= DarkMultiple*cadence: STALE (slipping).
//   - ticked, age > DarkMultiple*cadence: DARK (gone quiet past its cadence).
func deriveState(lastTickUnixNano, cadenceSeconds int64, now time.Time, th HealthThresholds) HealthState {
	if lastTickUnixNano <= 0 {
		if cadenceSeconds <= 0 {
			return HealthUnknown
		}
		return HealthDark
	}
	if cadenceSeconds <= 0 {
		return HealthUnknown
	}
	ageNanos := now.UTC().UnixNano() - lastTickUnixNano
	if ageNanos < 0 {
		ageNanos = 0
	}
	cadenceNanos := cadenceSeconds * int64(time.Second)
	darkNanos := th.darkMultiple() * cadenceNanos
	switch {
	case ageNanos <= cadenceNanos:
		return HealthLive
	case ageNanos <= darkNanos:
		return HealthStale
	default:
		return HealthDark
	}
}

// rollup tallies the rows into the fleet summary.
func rollup(rows []HealthRow) HealthRollup {
	var r HealthRollup
	r.Loops = len(rows)
	for _, row := range rows {
		switch row.State {
		case HealthLive:
			r.Live++
		case HealthStale:
			r.Stale++
		case HealthDark:
			r.Dark++
		case HealthUnknown:
			r.Unknown++
		}
		if row.Registered {
			r.Registered++
		}
		if row.Ledgered {
			r.Ledgered++
		}
	}
	return r
}

// round3 rounds to 3 decimals so a keep rate is a stable, comparable value across
// runs. Kept inline to honor loopmgr's stdlib-only ledger ethos.
func round3(v float64) float64 {
	return math.Round(v*1000) / 1000
}
