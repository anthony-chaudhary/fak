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
	// HealthRetired: the loop is registered but the operator put it DOWN (stopped or
	// disabled — not armed), so it is not expected to tick. Distinct from dark: a dark
	// loop is SUPPOSED to be running and has gone quiet (an alarm); a retired loop is
	// intentionally quiet (not an alarm). Never returned by DeriveState — healthRow
	// assigns it, and only in place of what would otherwise be a dark verdict, so a
	// retired job neither reddens `loop health --check` nor inflates rollup.Dark.
	HealthRetired HealthState = "retired"
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

	// WitnessRefused is the count of ended runs whose independent witness was actively
	// REFUSED (the referee said "not done") — a stronger, non-infra failure signal than a
	// merely-unavailable witness. Lifted straight from the loop snapshot.
	WitnessRefused uint64 `json:"witness_refused"`

	// WitnessUnavailable is the count of ended runs whose witness could not be reached at
	// all (an infra gap, not a refusal). Kept distinct from WitnessRefused so a reader can
	// tell "the witness said no" apart from "no witness ever ran".
	WitnessUnavailable uint64 `json:"witness_unavailable"`

	// WitnessGap is Runs - Witnessed, floored at 0: ended runs that never reached an
	// independent witnessed-done verdict — the "talking, not proven done" magnitude. A
	// loop that keeps ending runs it cannot prove done reads a high gap while its
	// Dark/State stay clean, so the worst-first walk that consumes this fold can finally
	// prefer it over a loop with honest, witnessed failures. Derived from counters that
	// already exist; the KeepRate is the same signal as a rate, the gap as a count.
	WitnessGap uint64 `json:"witness_gap"`

	// WitnessCollapse is true when a MAJORITY of ended runs went unwitnessed
	// (Runs > 0 && Witnessed*2 < Runs). It is ADVISORY and DESCRIPTIVE: the tunable
	// keep-below-floor gate lives in the governor (Policy.MinWitnessRate ->
	// ReasonWitnessCollapse); this is a fixed "majority unwitnessed" descriptor for the
	// health pane, carried explicitly like Dark so a --json consumer can gate on one
	// field without re-deriving it from KeepRate.
	WitnessCollapse bool `json:"witness_collapse"`

	// Failed is the count of ended runs that ended failed or canceled. Runs counts an
	// end regardless of outcome, so without this a loop that has failed on every
	// recorded run is indistinguishable, in this pane, from one that worked (#6497).
	Failed uint64 `json:"failed"`

	// ConsecutiveFailures is the current trailing streak of failing ends, reset by any
	// completed run. Carried on the row so an operator sees "failing right now" rather
	// than a lifetime total that a long-ago streak could explain away.
	ConsecutiveFailures uint64 `json:"consecutive_failures"`

	// FailureAlert is the surfaced boolean the issue asks for: true once the loop has
	// failed FailureAlertThreshold times in a row — the first REPEATED failure. Like
	// Dark and WitnessCollapse it is carried explicitly so a --json consumer gates on
	// one field instead of re-deriving the threshold.
	FailureAlert bool `json:"failure_alert,omitempty"`

	// NeverSucceeded is true when the loop has ended at least one run and EVERY one
	// failed — the exact shape #6497 was filed about (four runs, four failures, zero
	// witnessed successes, while the OS scheduler looked green).
	NeverSucceeded bool `json:"never_succeeded,omitempty"`

	// Effects, NoFuel and Unattributed are the utility partition of the completed
	// runs: produced something, typed-declared there was nothing to do, or ended with
	// a bare child exit 0 that proves neither. Runs == Failed+Effects+NoFuel+Unattributed.
	Effects      uint64 `json:"effects"`
	NoFuel       uint64 `json:"no_fuel"`
	Unattributed uint64 `json:"unattributed"`

	// CostMilliUSD is the summed reported cost across ended runs (thousandths of a US
	// dollar); CostedRuns is how many runs reported one, so a zero cost reads as
	// "never measured" rather than "free".
	CostMilliUSD int64  `json:"cost_milli_usd,omitempty"`
	CostedRuns   uint64 `json:"costed_runs,omitempty"`

	// LastState is the loop's last folded state string (the ledger's word on what it
	// was doing), carried for context. Not the health verdict — that is State.
	LastState string `json:"last_state,omitempty"`

	// LearningDebt is the latest learning_debt metric recorded by the loop ledger.
	// Nil means the loop has not measured that metric yet; 0 is a real debt-free
	// measurement and must still be rendered.
	LearningDebt *int64 `json:"learning_debt,omitempty"`

	// OSFiredNoLedgerRow is the #4989 OS-scheduler rung: true iff this row is DARK in
	// the ledger plane AND a mapped OS task fired successfully (LastTaskResult 0x0)
	// within its cadence window. It sits ALONGSIDE Dark/State (never overloads them):
	// the ledger gap is a real fact, but the loop is demonstrably alive at the OS
	// layer, so the row is fired-but-no-ledger-row (likely `--check` instead of
	// `tick`, or a genuinely no-op tick), NOT a loop that stopped running. Only ever
	// set on a DARK row; fails closed (an absent/unhealthy/unplaceable/stale OS
	// witness never promotes a row out of DARK).
	OSFiredNoLedgerRow bool `json:"os_fired_no_ledger_row,omitempty"`

	// OSTaskLabel is the corroborating OS task's label (e.g. "FleetStaleWorkGarden"),
	// carried onto the row for the reader ONLY when OSFiredNoLedgerRow is set. Empty
	// on every other row, including a fail-closed DARK one.
	OSTaskLabel string `json:"os_task_label,omitempty"`

	// OSLastRunUnixNano is the corroborating OS task's last-run time, carried for the
	// reader ONLY when OSFiredNoLedgerRow is set. 0 otherwise.
	OSLastRunUnixNano int64 `json:"os_last_run_unix_nano,omitempty"`
}

// HealthRollup is the fleet-wide summary across all rows: how many loops, and how
// many in each derived state. The roll-up answers "is the fleet of loops healthy"
// in one line; Dark > 0 is the signal a scheduler gates on.
type HealthRollup struct {
	Loops   int `json:"loops"`
	Live    int `json:"live"`
	Stale   int `json:"stale"`
	Dark    int `json:"dark"`
	Unknown int `json:"unknown"`
	// Retired is the count of registered-but-not-armed (stopped/disabled) jobs. It is
	// a SIBLING of Dark, never part of it: an operator-retired loop is intentionally
	// quiet, so it must not redden `loop health --check` (which gates on Dark) nor be
	// counted among the loops that have gone silently dark.
	Retired    int `json:"retired,omitempty"`
	Registered int `json:"registered"`
	Ledgered   int `json:"ledgered"`
	// WitnessGap is the fleet-wide sum of per-loop WitnessGap: total ended-but-unwitnessed
	// runs across all loops — the aggregate "talking, not proven done" magnitude.
	WitnessGap int `json:"witness_gap"`
	// WitnessCollapse is the count of loops whose ended runs are majority-unwitnessed
	// (per-row WitnessCollapse). A scheduler can gate on WitnessCollapse > 0 the way it
	// already gates on Dark > 0.
	WitnessCollapse int `json:"witness_collapse"`
	// Runs is the fleet-wide sum of ended runs — the denominator the utility counters
	// below partition, carried so a reader never has to re-sum the rows.
	Runs int `json:"runs"`
	// Failed is the fleet-wide sum of failed/canceled ended runs; FailureAlert and
	// NeverSucceeded are the counts of LOOPS in each alarming shape (#6497). A pane
	// gates on FailureAlert > 0 exactly as it already gates on Dark > 0 — and unlike
	// Dark, this one fires for a loop that is ticking perfectly on cadence and failing
	// every single time.
	Failed         int `json:"failed"`
	FailureAlert   int `json:"failure_alert"`
	NeverSucceeded int `json:"never_succeeded"`
	// Effects, NoFuel and Unattributed are the fleet-wide utility partition of the
	// completed runs. A fleet whose completed runs are all Unattributed is a fleet that
	// cannot prove it did anything, however green its schedulers look.
	Effects      int `json:"effects"`
	NoFuel       int `json:"no_fuel"`
	Unattributed int `json:"unattributed"`
	// CostMilliUSD is the fleet-wide reported cost; CostedRuns is how many runs
	// reported one, so a zero total is readable as unmeasured rather than free.
	CostMilliUSD int64 `json:"cost_milli_usd,omitempty"`
	CostedRuns   int   `json:"costed_runs,omitempty"`
	// OSFiredNoLedgerRow is the count of rows promoted by the #4989 OS-scheduler rung:
	// ledger-DARK loops whose mapped OS task fired 0x0 within cadence. It is a SUBSET of
	// Dark (each such row is still counted in Dark), never a sibling tally — a reader
	// sees "N of the dark loops are actually firing at the OS layer, just not writing a
	// ledger row".
	OSFiredNoLedgerRow int `json:"os_fired_no_ledger_row,omitempty"`
}

// HealthReport is the full read-only fold: the schema tag, the time it was folded,
// the per-loop rows (stable loop-id order), and the roll-up.
type HealthReport struct {
	Schema     string       `json:"schema"`
	TSUnixNano int64        `json:"ts_unix_nano"`
	Rows       []HealthRow  `json:"rows"`
	Rollup     HealthRollup `json:"rollup"`
}

// OSTaskInfo is the corroborating OS-scheduler signal for one loop (#4989): did its
// mapped OS task fire successfully (LastTaskResult 0x0), and when. It is the SECOND
// liveness plane — the ledger plane reads last-appended-row-vs-cadence and so cannot
// see a task that fired on cadence but wrote no ledger row. It is trusted to promote a
// row out of false-DARK ONLY when it fired AND its last run is placeable in time AND
// that run falls within the loop's cadence window; every other shape fails closed (see
// overlayOSTask), because the rung's one unacceptable failure is fabricating liveness
// for a loop that is really dead.
type OSTaskInfo struct {
	// TaskLabel is the OS task's name (e.g. "FleetStaleWorkGarden"), carried onto the
	// row for the reader when the witness corroborates.
	TaskLabel string
	// Fired is true iff the task's LastTaskResult decoded to success (0x0). A task
	// that ran but failed, or whose result is unknown, is not a witness.
	Fired bool
	// LastRunUnixNano is when the task last ran. 0 means unreadable — which fails
	// closed: an unplaceable run cannot prove the task fired within cadence.
	LastRunUnixNano int64
}

// FoldHealth is the read-only health fold. It joins the folded loop ledger (st, from
// Summarize) with the job registry (reg, the cadence definition) and derives one
// HealthRow per loop seen in EITHER input, plus a roll-up. It is PURE: no clock read
// (now is supplied), no I/O, no mutation of its inputs.
//
// It is exactly FoldHealthWithOS with no OS witness — the ledger plane alone — so the
// existing ledger-only callers stay byte-identical.
func FoldHealth(st Status, reg Registry, now time.Time, th HealthThresholds) HealthReport {
	return FoldHealthWithOS(st, reg, now, th, nil)
}

// FoldHealthWithOS is FoldHealth plus the #4989 OS-scheduler rung. After each row is
// derived from the ledger plane, a DARK row whose loop id has a corroborating
// OSTaskInfo (fired 0x0, last run placeable AND within the loop's cadence) is
// surfaced as OSFiredNoLedgerRow — fired-but-no-ledger-row — instead of being left as
// a false-DARK. The promotion sits ALONGSIDE the liveness verdict: State/Dark are
// untouched (a --json consumer gating on Dark keeps its meaning) and the OS tally is a
// SUBSET of Dark in the roll-up, never a sibling. Fail-closed at every edge: a nil
// witness map, an unmapped loop, a non-fired task, an unreadable last-run, or a
// last-run past cadence all leave the row a plain DARK with no OS evidence attached. A
// non-DARK row is never touched — the rung explains a dark ledger, it does not
// decorate a healthy row.
//
// The union is deliberate. A loop present in the ledger but absent from the registry
// is an ad-hoc loop (judged against the default horizon). A loop present in the
// registry but absent from the ledger is the canonical DARK loop — registered to a
// schedule, never observed ticking — which a ledger-only fold would render as silence.
// Folding the union surfaces it.
func FoldHealthWithOS(st Status, reg Registry, now time.Time, th HealthThresholds, osTasks map[string]OSTaskInfo) HealthReport {
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
		row := healthRow(id, snap, ledgered, job, registered, now, th)
		if w, ok := osTasks[id]; ok {
			overlayOSTask(&row, w, now)
		}
		rows = append(rows, row)
	}

	return HealthReport{
		Schema:     HealthSchema,
		TSUnixNano: now.UTC().UnixNano(),
		Rows:       rows,
		Rollup:     rollup(rows),
	}
}

// overlayOSTask folds one loop's OS-scheduler witness onto its row (#4989). It is the
// sole promotion path and is fail-closed by construction: it attaches OS evidence ONLY
// when the row is DARK (the rung explains a dark ledger, nothing else), the task fired
// 0x0, its last run is readable (>0), the row carries a cadence to bound the run
// against, AND that run falls within one cadence window ending at now. Any other shape
// returns with the row unchanged — no OS evidence attached, the row stays a plain
// DARK. State and Dark are never mutated.
func overlayOSTask(row *HealthRow, w OSTaskInfo, now time.Time) {
	if row.State != HealthDark {
		return // the rung only overlays a dark row; a live/stale row is left untouched
	}
	if !w.Fired || w.LastRunUnixNano <= 0 {
		return // a non-fired task or an unplaceable run is not a witness — fail closed
	}
	if row.CadenceSeconds <= 0 {
		return // no cadence to bound the OS run against — fail closed
	}
	age := now.UTC().UnixNano() - w.LastRunUnixNano
	if age < 0 {
		return // a run stamped in the future has not happened yet — it cannot corroborate a within-cadence fire; fail closed, the row stays a plain DARK
	}
	if age > row.CadenceSeconds*int64(time.Second) {
		return // the OS task's own last run is past cadence — it stopped firing too
	}
	row.OSFiredNoLedgerRow = true
	row.OSTaskLabel = w.TaskLabel
	row.OSLastRunUnixNano = w.LastRunUnixNano
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

	// Surface the claimed-vs-witnessed gap alongside the keep rate: the counts a
	// worst-first walk needs to tell a loop "talking, not proven done" apart from one
	// with honest, witnessed failures. Additive — the source counters already exist on
	// the snapshot, so this fold reads them, it does not track anything new.
	row.WitnessRefused = snap.WitnessRefused
	row.WitnessUnavailable = snap.WitnessUnavailable
	if row.Runs > row.Witnessed {
		row.WitnessGap = row.Runs - row.Witnessed
	}
	row.WitnessCollapse = row.Runs > 0 && row.Witnessed*2 < row.Runs

	// The #6497 utility partition and its two alarms. Same shape as the witness block
	// above: the snapshot already carries the counters, so this fold reads them and
	// names the one derived bit (FailureAlert / NeverSucceeded) each surface would
	// otherwise re-derive. A loop can be LIVE, on cadence, and still be alerting here —
	// that disagreement is the whole point, since the liveness verdict answers "is it
	// ticking" and these answer "is it working".
	row.Failed = snap.Failed
	row.ConsecutiveFailures = snap.ConsecutiveFailures
	row.FailureAlert = snap.FailureAlert()
	row.NeverSucceeded = snap.NeverSucceeded()
	row.Effects = snap.Effects
	row.NoFuel = snap.NoFuel
	row.Unattributed = snap.Unattributed
	row.CostMilliUSD = snap.CostMilliUSD
	row.CostedRuns = snap.CostedRuns

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

	row.State = DeriveState(snap.LastEventUnixNano, cadence, now, th)
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
	// A registered job the operator put DOWN (stopped or disabled — not armed) is not
	// expected to tick, so a dark verdict for it is a false alarm that reddens
	// `loop health --check` and inflates rollup.Dark for a loop that is intentionally
	// quiet. Reclassify that ONE case to RETIRED — a distinct, non-dark surfaced state —
	// so the alarm fires only for loops that are SUPPOSED to be running. A non-armed job
	// that still reads live/stale keeps its liveness verdict (it is genuinely ticking).
	if registered && !job.State.Armed() && row.State == HealthDark {
		row.State = HealthRetired
	}
	row.Dark = row.State == HealthDark
	return row
}

// DeriveState classifies a loop's liveness from its last tick against a cadence.
// The rules, in order:
//   - never ticked (lastTick <= 0): DARK — a registered/ledgered loop that has never
//     fired is dark, not unknown; if cadence is also unknown (<=0) it is UNKNOWN.
//   - ticked, age <= cadence: LIVE.
//   - ticked, cadence < age <= DarkMultiple*cadence: STALE (slipping).
//   - ticked, age > DarkMultiple*cadence: DARK (gone quiet past its cadence).
//
// Exported so a cross-package fold (e.g. internal/loopfleet's cross-ledger pane) can
// call the SAME classifier instead of re-implementing it, guaranteeing the two draw
// the dark line identically by construction rather than by comment convention.
func DeriveState(lastTickUnixNano, cadenceSeconds int64, now time.Time, th HealthThresholds) HealthState {
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
		case HealthRetired:
			r.Retired++
		}
		if row.Registered {
			r.Registered++
		}
		if row.Ledgered {
			r.Ledgered++
		}
		r.WitnessGap += int(row.WitnessGap)
		if row.WitnessCollapse {
			r.WitnessCollapse++
		}
		r.Runs += int(row.Runs)
		r.Failed += int(row.Failed)
		if row.FailureAlert {
			r.FailureAlert++
		}
		if row.NeverSucceeded {
			r.NeverSucceeded++
		}
		r.Effects += int(row.Effects)
		r.NoFuel += int(row.NoFuel)
		r.Unattributed += int(row.Unattributed)
		r.CostMilliUSD += row.CostMilliUSD
		r.CostedRuns += int(row.CostedRuns)
		if row.OSFiredNoLedgerRow {
			r.OSFiredNoLedgerRow++
		}
	}
	return r
}

// round3 rounds to 3 decimals so a keep rate is a stable, comparable value across
// runs. Kept inline to honor loopmgr's stdlib-only ledger ethos.
func round3(v float64) float64 {
	return math.Round(v*1000) / 1000
}
