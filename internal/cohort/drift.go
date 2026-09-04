package cohort

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// drift.go is the per-cohort quality-drift monitor of the quality middle ladder
// (#4581, under epic #4509): the layer that watches input and output quality
// signals BUCKETED BY COHORT and localizes a drift to the cohort that actually
// drifted, without smearing it across the whole run. It sits between fak
// primitive correctness and coarse end-benchmark scores: a cheap, deterministic
// fold that a PR / nightly / release tier can each run at its own cadence.
//
// It is a pure adjudicator. Each cohort's observed signals are compared ONLY
// against that same cohort's own baseline and declared tolerance, so a drift in
// one cohort can never mark a sibling cohort drifted (no global misattribution).
// It fails closed: a cohort whose provenance is incomplete, whose baseline is
// absent, whose tolerance for a baselined signal is undeclared, or whose observed
// value for a baselined signal is missing is NOT reported stable — "no evidence"
// and "unclear evidence" are never a pass (#4581 acceptance).
//
// Honesty caveat: this monitors SUPPLIED signals; it does not itself sample a
// model, tokenize, or run an engine. The provenance a case carries
// (model/tokenizer/engine/seed-or-oracle/revision/baseline) is the producer's
// attestation of where the numbers came from — this leaf checks the attestation
// is complete and the numbers are within tolerance, not that the producer told
// the truth about the run. The replay artifact it emits is scrubbed by
// construction: it carries only cohort labels, signal names, and the numeric
// baseline/observed/delta/tolerance — never raw model text — so it is safe to
// attach to an issue.

// DriftSchema is the versioned tag on a cohort drift report. Consumers pin the
// major so a schema bump is a conscious migration, not a silent field drift.
const DriftSchema = "fak-cohort-drift/1"

// Well-known drift signals: input mix, output length, task proxy, degeneration,
// and rubric score. The monitor evaluates any signal a cohort defines a baseline for.
const (
	// SignalMix tracks input task distribution stability.
	SignalMix = "mix"
	// SignalLength monitors generated token output length.
	SignalLength = "length"
	// SignalLanguageTask monitors language and task proxy stability.
	SignalLanguageTask = "language_task"
	// SignalDegeneration detects repetitive response degeneration.
	SignalDegeneration = "degeneration"
	// SignalRubric evaluates qualitative scoring against expected rubrics.
	SignalRubric = "rubric"
)

// DriftTier is the cost and cadence class assigned to a cohort drift case.
type DriftTier string

const (
	// DriftTierPR evaluates fast checks during pull request review.
	DriftTierPR DriftTier = "pr"
	// DriftTierNightly runs comprehensive drift monitoring in scheduled runs.
	DriftTierNightly DriftTier = "nightly"
	// DriftTierRelease performs validation before tagging a formal release.
	DriftTierRelease DriftTier = "release"
)

// valid reports whether the tier is one of the three declared cadence classes.
// An unrecognized tier is not silently accepted — an unattributable cadence is
// treated as inconclusive by the monitor.
func (t DriftTier) valid() bool {
	switch t {
	case DriftTierPR, DriftTierNightly, DriftTierRelease:
		return true
	}
	return false
}

// DriftState represents the adjudication state for a cohort's observed evidence.
type DriftState string

const (
	// DriftStable indicates all baselined signals held within declared tolerances.
	DriftStable DriftState = "stable"
	// DriftDrifted indicates at least one signal exceeded its allowable divergence tolerance.
	DriftDrifted DriftState = "drifted"
	// DriftMissing indicates no baseline records were supplied for the cohort.
	DriftMissing DriftState = "missing"
	// DriftInconclusive indicates incomplete provenance, unassigned cadence, or missing observations.
	DriftInconclusive DriftState = "inconclusive"
)

// stable reports whether a state is the single passing state.
func (s DriftState) stable() bool { return s == DriftStable }

// DriftProvenance is the per-case provenance #4581 requires every drift record to
// carry: which model/tokenizer/engine produced the signals, the seed (stochastic)
// OR deterministic oracle they were judged by, the code/module revision they were
// produced at, and the tolerance/baseline they were compared against. Absent
// provenance is treated as inconclusive — signals you cannot attribute cannot
// pass.
type DriftProvenance struct {
	Model     string `json:"model"`
	Tokenizer string `json:"tokenizer"`
	Engine    string `json:"engine"`
	// Seed pins a stochastic case; Oracle names the deterministic comparator. At
	// least one must be set — a case that is neither seeded nor oracle-judged is
	// not reproducible, so it cannot qualify.
	Seed   int64  `json:"seed,omitempty"`
	Oracle string `json:"oracle,omitempty"`
	// Revision is the code/module revision the signals were produced at.
	Revision string `json:"revision"`
	// Baseline is the tolerance/baseline provenance the cohort was judged against.
	Baseline string `json:"baseline"`
}

// complete reports whether the provenance names everything #4581 requires. The
// returned reason localizes the FIRST missing field so a gap is actionable rather
// than a bare "incomplete".
func (p DriftProvenance) complete() (bool, string) {
	switch {
	case p.Model == "":
		return false, "missing model"
	case p.Tokenizer == "":
		return false, "missing tokenizer"
	case p.Engine == "":
		return false, "missing engine/backend"
	case p.Seed == 0 && p.Oracle == "":
		return false, "missing seed or deterministic oracle"
	case p.Revision == "":
		return false, "missing code/module revision"
	case p.Baseline == "":
		return false, "missing tolerance/baseline provenance"
	}
	return true, ""
}

// CohortObservation is one cohort's tracked signals for a run: its declared tier
// and runtime cost, its provenance, and the per-signal baseline, observed value,
// and allowed absolute tolerance. A signal is judged only when the cohort carries
// a baseline for it; a baselined signal missing a tolerance or an observation is
// inconclusive, not stable.
type CohortObservation struct {
	Cohort      string             `json:"cohort"`
	Tier        DriftTier          `json:"tier"`
	CostSeconds float64            `json:"cost_seconds"`
	Provenance  DriftProvenance    `json:"provenance"`
	Baseline    map[string]float64 `json:"baseline"`
	Observed    map[string]float64 `json:"observed"`
	Tolerance   map[string]float64 `json:"tolerance"`
}

// SignalDivergence is the first actionable divergence within a cohort: the signal
// that offended, its baseline and observed values, the absolute delta, and the
// tolerance it exceeded. For an inconclusive (missing-observation) divergence the
// Observed field is left zero and Detail says so.
type SignalDivergence struct {
	Signal    string  `json:"signal"`
	Baseline  float64 `json:"baseline"`
	Observed  float64 `json:"observed"`
	Delta     float64 `json:"delta"`
	Tolerance float64 `json:"tolerance"`
	Detail    string  `json:"detail"`
}

// DriftReplay is the scrubbed replay artifact emitted for a non-stable cohort. It
// carries only labels and numbers — cohort, revision, and the offending signal's
// baseline/observed/delta/tolerance — so it is safe to attach to an issue without
// leaking model text or host detail. It replays: the same inputs reproduce it
// byte for byte.
type DriftReplay struct {
	Schema    string  `json:"schema"`
	Cohort    string  `json:"cohort"`
	Revision  string  `json:"revision"`
	Signal    string  `json:"signal"`
	Baseline  float64 `json:"baseline"`
	Observed  float64 `json:"observed"`
	Delta     float64 `json:"delta"`
	Tolerance float64 `json:"tolerance"`
	Note      string  `json:"note"`
}

// CohortDrift is the per-cohort verdict: its state, the reason it reached that
// state, and — on a non-stable state — the localized first divergence and scrubbed
// replay artifact.
type CohortDrift struct {
	Cohort          string            `json:"cohort"`
	Tier            DriftTier         `json:"tier"`
	CostSeconds     float64           `json:"cost_seconds"`
	State           DriftState        `json:"state"`
	Reason          string            `json:"reason"`
	FirstDivergence *SignalDivergence `json:"first_divergence,omitempty"`
	Replay          *DriftReplay      `json:"replay,omitempty"`
}

// DriftReport is the machine-readable output of the monitor: whether every cohort
// held stable, the non-stable cohorts in submission order (Drifts[0] is the first
// actionable cohort), and the labels that held stable. Clean is true iff no cohort
// drifted, went missing, or was inconclusive — the fail-closed contract of #4581.
type DriftReport struct {
	Schema   string        `json:"schema"`
	Revision string        `json:"revision"`
	Clean    bool          `json:"clean"`
	Drifts   []CohortDrift `json:"drifts,omitempty"`
	Stable   []string      `json:"stable,omitempty"`
}

// MonitorDrift adjudicates each cohort observation independently against its own
// baseline and declared tolerance, localizing drift to the drifting cohort so a
// synthetic drift in one cohort never misattributes to another. It is a pure
// function of its inputs — same inputs, same report — so a verdict replays.
//
// Per cohort, in order of decreasing precedence: incomplete provenance or an
// invalid tier is Inconclusive; a cohort with no baseline at all is Missing; then
// each baselined signal is scanned in sorted name order and the FIRST offending
// one decides the verdict — a signal with no declared tolerance or no observation
// is Inconclusive, and a signal whose |observed - baseline| exceeds its tolerance
// is Drifted. A cohort with no offending signal is Stable.
func MonitorDrift(rev string, obs []CohortObservation) DriftReport {
	report := DriftReport{Schema: DriftSchema, Revision: rev}
	for _, o := range obs {
		d := judgeCohort(rev, o)
		if d.State.stable() {
			report.Stable = append(report.Stable, o.Cohort)
			continue
		}
		report.Drifts = append(report.Drifts, d)
	}
	sort.Strings(report.Stable)
	report.Clean = len(report.Drifts) == 0
	return report
}

// judgeCohort adjudicates a single cohort against its own evidence.
func judgeCohort(rev string, o CohortObservation) CohortDrift {
	d := CohortDrift{Cohort: o.Cohort, Tier: o.Tier, CostSeconds: o.CostSeconds}

	if ok, why := o.Provenance.complete(); !ok {
		d.State, d.Reason = DriftInconclusive, "incomplete provenance: "+why
		return d
	}
	if !o.Tier.valid() {
		d.State, d.Reason = DriftInconclusive, fmt.Sprintf("unassigned or unknown tier %q", o.Tier)
		return d
	}
	if len(o.Baseline) == 0 {
		d.State, d.Reason = DriftMissing, "no baseline recorded for cohort"
		return d
	}

	// stopAt ends the scan on the first signal that is not cleanly within tolerance.
	// The scan walks signals in sorted order and reports the earliest divergence,
	// with a scrubbed replay attached, so two runs over the same inputs name the same signal.
	// Each arm below supplies only what actually differs — the resulting state,
	// the operator-facing reason, the divergence, and the replay note.
	stopAt := func(state DriftState, reason string, div *SignalDivergence, note string) CohortDrift {
		d.State, d.Reason = state, reason
		d.FirstDivergence = div
		d.Replay = replayOf(rev, o.Cohort, div, note)
		return d
	}
	for _, sig := range sortedKeys(o.Baseline) {
		base := o.Baseline[sig]
		tol, hasTol := o.Tolerance[sig]
		if !hasTol {
			return stopAt(DriftInconclusive, "signal "+sig+": no tolerance declared",
				&SignalDivergence{
					Signal: sig, Baseline: base,
					Detail: "no tolerance declared for baselined signal",
				}, "inconclusive: undeclared tolerance")
		}
		obsVal, hasObs := o.Observed[sig]
		if !hasObs {
			return stopAt(DriftInconclusive, "signal "+sig+": no observation recorded",
				&SignalDivergence{
					Signal: sig, Baseline: base, Tolerance: tol,
					Detail: "no observation recorded for baselined signal",
				}, "inconclusive: missing observation")
		}
		delta := math.Abs(obsVal - base)
		if delta > tol {
			return stopAt(DriftDrifted,
				fmt.Sprintf("signal %s drifted %.6g beyond tolerance %.6g", sig, delta, tol),
				&SignalDivergence{
					Signal: sig, Baseline: base, Observed: obsVal, Delta: delta, Tolerance: tol,
					Detail: fmt.Sprintf("|%.6g - %.6g| = %.6g > tolerance %.6g", obsVal, base, delta, tol),
				}, "drifted beyond tolerance")
		}
	}

	d.State, d.Reason = DriftStable, fmt.Sprintf("%d signal(s) within tolerance", len(o.Baseline))
	return d
}

// replayOf builds the scrubbed replay artifact for an offending signal. It copies
// only labels and numbers already in the divergence, so it can never carry model
// text or host detail.
func replayOf(rev, cohort string, div *SignalDivergence, note string) *DriftReplay {
	return &DriftReplay{
		Schema: DriftSchema, Cohort: cohort, Revision: rev,
		Signal: div.Signal, Baseline: div.Baseline, Observed: div.Observed,
		Delta: div.Delta, Tolerance: div.Tolerance, Note: note,
	}
}

// sortedKeys returns the map keys in ascending order so the signal scan — and thus
// the first divergence — is deterministic regardless of Go map iteration order.
func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ExplainDrift renders a DriftReport as an operator readout: CLEAN with the stable
// cohorts, or DRIFT naming the first actionable cohort first and then every other
// non-stable cohort with its localized divergence. It is the bridge from a machine
// verdict to "here is exactly which cohort to look at".
func ExplainDrift(r DriftReport) string {
	var b strings.Builder
	if r.Clean {
		fmt.Fprintf(&b, "CLEAN  revision %s — %d cohort(s) within tolerance\n", r.Revision, len(r.Stable))
		for _, c := range r.Stable {
			fmt.Fprintf(&b, "  ok   %s\n", c)
		}
		return b.String()
	}
	fmt.Fprintf(&b, "DRIFT  revision %s — %d cohort(s) not stable\n", r.Revision, len(r.Drifts))
	for i, d := range r.Drifts {
		marker := "  "
		if i == 0 {
			marker = "->" // the first actionable divergence
		}
		fmt.Fprintf(&b, "%s %-8s %-12s %s\n", marker, d.Tier, d.State, d.Cohort)
		fmt.Fprintf(&b, "     reason: %s\n", d.Reason)
		if dv := d.FirstDivergence; dv != nil {
			fmt.Fprintf(&b, "     signal %s: baseline %.6g observed %.6g (%s)\n",
				dv.Signal, dv.Baseline, dv.Observed, dv.Detail)
		}
	}
	return b.String()
}
