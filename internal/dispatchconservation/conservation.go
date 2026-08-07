// Package dispatchconservation is the worker-unit conservation ledger for the
// dispatch fleet: over a window, units_spent = accounted + leaked, so a
// worker-unit that dies ungraded reads as a LEAK count, not as silence.
//
// The fleet's promise is work conservation: if N open issues meet N worker-units
// inside a window, every unit must end in exactly one accountable outcome — a
// witnessed ship, a graded refusal (no-commit with a reason), or a spawn failure
// — and anything else is a LEAK the operator cannot see today. `fak
// dispatch-throughput` measures a close RATE; `fak dispatch-status` diagnoses the
// current tick; neither answers the identity question "we spent K units this
// window — where did each go?". This package folds the durable `.dispatch-runs`
// artifacts into that identity:
//
//	units_spent(window) = shipped_witnessed
//	                    + committed_unwitnessed
//	                    + no_commit{self_modify|policy_block|auth_wall|usage_cap|
//	                                model_unknown|rate_limit|off_trunk|banner_noop|
//	                                restart_exhausted|preview_confirm_feedback|
//	                                missing_log_artifact|clean_exit_no_commit|
//	                                died_before_epilogue|guard_child_spawn_failed|
//	                                unknown}
//	                    + spawn_failed
//	                    + leaked_unswept          <- the number this tool exists for
//
// plus the issue side (windowed closes, contract-hold pressure, and re-storm
// churn: issues burning 2+ units in one window). Read-only by construction: it
// parses the artifacts directly and never imports the hot dispatcher modules, so
// it stays runnable while a peer edits them and cannot perturb what it measures.
//
// Honesty rules: a `.pid` we cannot disprove is LIVE (never leaked); a graded
// `.witness` is final; a dead worker with a real log and no witness is
// `leaked_unswept` — either the witness sweep has not reached it yet or its
// artifacts were pruned first. Coverage is reported, never assumed.
//
// This is the Go port of the retired tools/dispatch_conservation.py, moved to Go
// under the pythongate de-Python ratchet (a NEW tool must be written in Go).
package dispatchconservation

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Schema is the machine-readable report's schema tag.
const Schema = "fleet-dispatch-conservation/1"

// DefaultRunsDirname is the on-disk dispatch-runs artifact directory (under cwd).
const DefaultRunsDirname = ".dispatch-runs"

// DefaultWindowH is the default reporting window, in hours.
const DefaultWindowH = 6.0

// Closed outcome vocabulary. Every finished in-window unit folds to exactly one.
const (
	OutcomeLive        = "live"
	OutcomeWitnessed   = "shipped_witnessed"
	OutcomeUnwitnessed = "committed_unwitnessed"
	OutcomeNoCommit    = "no_commit"
	OutcomeSpawnFailed = "spawn_failed"
	OutcomeLeaked      = "leaked_unswept"
)

// noCommitReasons is the witness sweep's no-commit reason vocabulary; anything
// outside it folds to "unknown".
//
// This is a literal copy of the reason strings the sidecar writers actually stamp,
// from BOTH producers — the same "not an import" discipline logRE below documents:
//
//   - internal/dispatchtick/witness.go (NoCommit* consts): the Go model-arm classes
//     self_modify, policy_block, auth_wall, usage_cap, model_unknown, rate_limit,
//     off_trunk, banner_noop, unknown.
//   - tools/issue_resolve_dispatch.py (NO_COMMIT_* consts, classify_no_commit_reason):
//     the Python witness sweep that writes the .witness sidecars THIS package reads.
//     It emits three classes the dispatchtick list never had — restart_exhausted,
//     preview_confirm_feedback and missing_log_artifact.
//
// Those three were absent here, so every unit carrying them hit the `!noCommitReasons`
// fold below and was booked as "unknown" — the ledger under-reported the one cause it
// most needs to name. Measured on this repo's own .dispatch-runs over a 78h window
// (2026-08-04..08-07): the sidecars held unknown=143 and restart_exhausted=26, and
// `fak dispatch-conservation --window-h 78` printed a single inflated unknown=169 with
// restart_exhausted absent from the breakdown entirely. A guard-restart storm
// (dominant_cause=BUDGET_CONTEXT_EXHAUSTED, cmd/fak/guard_child.go guardEquivalentRestartStatus)
// is a NAMED, actionable terminal; folding it into the residual bucket hid 15% of the
// window's no-commit units behind the label that means "we could not tell".
//
// Keep this set a SUPERSET of both producers: a reason that only one side knows is
// still a real, typed outcome, and silently renaming it to "unknown" is exactly the
// unaccounted-unit blindness this package exists to remove.
var noCommitReasons = map[string]bool{
	"self_modify": true, "policy_block": true, "auth_wall": true,
	"usage_cap": true, "model_unknown": true, "rate_limit": true,
	"off_trunk": true, "banner_noop": true, "unknown": true,
	// tools/issue_resolve_dispatch.py-only classes (see above).
	"restart_exhausted": true, "preview_confirm_feedback": true,
	"missing_log_artifact": true,
	// Refinements of the producers' own "unknown", derived HERE from the worker log
	// tail (see refineUnknownNoCommit). The sweep has no signature for these, so it
	// stamps "unknown"; this package can still name them.
	//
	// These three are deliberately NOT mirrored into internal/dispatchtick's NoCommit*
	// consts or into tools/issue_resolve_dispatch.py, unlike the classes above. That
	// list is the PRODUCER vocabulary — the strings a sidecar writer stamps into a
	// .witness — and neither producer stamps these. Adding them there would advertise a
	// class no writer emits, and dispatchtick.failKindForNoCommitReason would then be
	// switching on a reason it can never receive. The asymmetry is the point: everything
	// above is "a producer knows this and we must not fold it"; these are "no producer
	// can tell, and the ledger reads the artifact to find out".
	ReasonCleanExitNoCommit:     true,
	ReasonDiedBeforeEpilogue:    true,
	ReasonGuardChildSpawnFailed: true,
}

// Refinements of the sweep's residual "unknown" no-commit reason.
//
// WHY these exist. Over 2026-08-04T00:00Z..08-07T06:15Z (the clean current-fleet
// window — a -compact-solvency-floor flag-parse regression killed workers at spawn
// 07-28..08-03, so a trailing-7d slice measures the outage, not the fleet), the
// sidecars held 283 finished resolve units and `unknown` was the LARGEST terminal
// disposition: 149 runs (52.8%) and 55.2 of 102.3 wall-to-witness seat-hours (54.0%).
// Nothing downstream can route, hold or cool down on a fall-through, so the fleet's
// biggest bucket was also its blindest. #5867.
//
// The residual is not opaque — the worker log tail already separates it:
//
//   - ReasonCleanExitNoCommit (114/149): the guard wrote its exit summary, so the
//     session ENDED NORMALLY and simply landed nothing. Actionable as a throughput
//     problem (same wall-clock as a winning run, ~half the turns), not as a wall.
//   - ReasonDiedBeforeEpilogue (30/149): no guard exit summary at all. Corroborated
//     30/30 by the log's last line being an in-flight `fak-turn trace=… ` row — these
//     were killed MID-TURN. Actionable only as supervision/retention, never as a
//     reason: no exit evidence was ever written.
//   - ReasonGuardChildSpawnFailed (2/149): the guard could not exec the agent at all
//     ("fak guard: could not run", cmd/fak/guard_child_supervision.go). A mechanical
//     config failure, not an agent outcome.
const (
	ReasonCleanExitNoCommit     = "clean_exit_no_commit"
	ReasonDiedBeforeEpilogue    = "died_before_epilogue"
	ReasonGuardChildSpawnFailed = "guard_child_spawn_failed"
)

// guardEpilogueMarker is the guard exit summary's section rule — guardSection() in
// cmd/fak/guard_format_layout.go renders "── guard · <name> ──…"
// once per section. Its presence in the tail proves the guard reached its epilogue.
//
// It is deliberately the SECTION RULE and not the "guard · cache window" section
// #5867 proposed. That last section is emitted by formatVCacheSnapshotPointer, which
// returns "" when the session recorded zero cache turns
// (cmd/fak/guard_child_supervision.go: `if turns <= 0`), so it MISSES clean exits that
// never saw a cached turn. Measured over the full retained history (2382 graded
// resolve units): the cache-window marker finds 0/69 auth_wall runs, every one of
// which carries a complete epilogue the section rule does find (69/69). Inside the
// unknown bucket the same gate hides 7 runs — 4 quiet clean exits, 2 spawn failures
// and 1 abnormal child exit — so keying on it would have booked 7 real epilogues as
// deaths. Section-rule presence is the same signal without the zero-turn blind spot.
const guardEpilogueMarker = "── guard · "

// guardChildSpawnFailedMarker is the guard's own "I never started the agent" line
// (cmd/fak/guard_child_supervision.go: `fak guard: could not run %q: %v`). It is
// checked FIRST because such a run also prints a partial epilogue, and "the child
// never ran" is the more specific — and more fixable — statement.
//
// It is the only auxiliary marker that survived a specificity check. The obvious
// fourth candidate, the abnormal-exit banner "exited abnormally (code" from
// formatGuardResumeGuidance, is REJECTED: over the same 2382 units it fires 927
// times and only 21 of those are in the unknown bucket (2.3%), including 16 of 85
// CLAIM_WITNESSED runs that crashed AND still shipped. A nonzero child exit simply
// does not predict a missing commit, so a `guard_child_crashed` class would look
// actionable and route nothing — strictly worse than leaving those runs in a class
// keyed on evidence that does hold. "fak guard: could not run" fires 3 times in the
// whole history and all 3 are unknown no-commits (100%).
const guardChildSpawnFailedMarker = "fak guard: could not run"

// unknownRefineTailBytes bounds the refinement read. 4096 is the witness sweep's own
// classifier window (internal/dispatchtick.WitnessTailBytes,
// tools/issue_resolve_dispatch.py `_CAP_TAIL_BYTES`), kept identical so this package
// never claims to see something the producer could not have seen. Verified sufficient:
// re-classifying all 149 in-window unknown units at 24 KiB instead moves ZERO of them.
const unknownRefineTailBytes = 4096

// refineUnknownNoCommit names a residual "unknown" no-commit from the worker log tail.
//
// It runs ONLY on a reason the producer itself stamped "unknown" — never on a reason
// string this package merely failed to recognize. That distinction is the #5866 lesson
// kept intact: after this refinement a bare "unknown" left in a report means "a sidecar
// writer stamped a reason our vocabulary has never heard of", a vocabulary-drift alarm
// worth acting on, rather than a shrug that swallows both cases at once.
//
// Fail-open and read-only, like every classifier here: an unreadable or absent log
// yields no marker and the unit keeps its plain "unknown".
func refineUnknownNoCommit(log string) string {
	tail := ReadTailBytes(log, unknownRefineTailBytes)
	switch {
	case strings.Contains(tail, guardChildSpawnFailedMarker):
		return ReasonGuardChildSpawnFailed
	case strings.Contains(tail, guardEpilogueMarker):
		return ReasonCleanExitNoCommit
	default:
		return ReasonDiedBeforeEpilogue
	}
}

// Log-name grammar shared with the dispatcher (kept as a literal copy, not an
// import: the dispatcher module is a live-edited surface).
var (
	logRE         = regexp.MustCompile(`^(resolve|repair)-(\d+)-(\d{8})-(\d{6})\.log$`)
	spawnHeaderRE = regexp.MustCompile(`^# fak-spawn \S+ issue=\d+ lane=(\S+) backend=(\S+)`)
)

// AliveProbe is a host liveness snapshot. Scanned=false means the host could not
// be scanned ("cannot disprove liveness"), which classifies every .pid-bearing
// unit as live — the conservative direction (a blind probe never invents a leak).
type AliveProbe struct {
	Scanned bool
	PIDs    map[int]bool
}

func (a AliveProbe) alive(pid int) bool {
	if !a.Scanned {
		return true // cannot disprove liveness
	}
	return a.PIDs[pid]
}

// DefaultAliveProbe reports that the host could not be scanned. Go has no
// portable process-enumeration primitive (unlike the retired tool's optional
// psutil), so the default cannot disprove liveness: pid-bearing units count live,
// never leaked. This equals the psutil-absent behavior of the original tool and
// honors the "never invent a leak from a blind probe" rule.
func DefaultAliveProbe() AliveProbe { return AliveProbe{Scanned: false} }

// Unit is one classified worker-unit.
type Unit struct {
	Log        string
	Kind       string
	Issue      int
	Lane       string
	Backend    string
	SpawnedUTC string
	Outcome    string
	Reason     string // no_commit bucket
	SHA        string // witnessed/unwitnessed sha (may be empty)
	PID        int
	stamp      time.Time
}

// ParseLogStampUTC returns the spawn time from a worker log name (the stamp is
// authoritative for windowing: log mtime moves every write, but the unit was
// SPENT at spawn). ok is false when the name is not a worker log.
func ParseLogStampUTC(name string) (time.Time, bool) {
	m := logRE.FindStringSubmatch(name)
	if m == nil {
		return time.Time{}, false
	}
	t, err := time.ParseInLocation("20060102150405", m[3]+m[4], time.UTC)
	if err != nil {
		return time.Time{}, false
	}
	return t, true
}

// ReadSpawnHeader returns lane/backend from the `# fak-spawn` first line ("" when
// absent), plus whether the log ever grew past the header — the spawn-failure
// signal: a header-only (or empty) log is a worker that died at/before exec.
func ReadSpawnHeader(log string) (lane, backend string, body bool) {
	raw, err := os.ReadFile(log)
	if err != nil {
		return "", "", false
	}
	lines := strings.Split(string(raw), "\n")
	if len(lines) > 0 {
		first := strings.TrimSpace(lines[0])
		if m := spawnHeaderRE.FindStringSubmatch(first); m != nil {
			lane, backend = m[1], m[2]
		} else if first != "" {
			body = true // no header but real content: still a run
		}
	}
	for _, line := range lines[1:] {
		if strings.TrimSpace(line) != "" {
			body = true
			break
		}
	}
	return lane, backend, body
}

// readWitness returns the worker's `.witness` verdict sidecar as a JSON object,
// or ok=false when the sweep has not graded it (or the record is unreadable or
// not a JSON object — treated the same, honestly).
func readWitness(log string) (map[string]any, bool) {
	raw, err := os.ReadFile(sidecar(log, ".witness"))
	if err != nil {
		return nil, false
	}
	var rec map[string]any
	if err := json.Unmarshal(raw, &rec); err != nil || rec == nil {
		return nil, false
	}
	return rec, true
}

// ClassifyUnit folds one worker log + sidecars into the closed outcome
// vocabulary. Precedence: a graded witness is final (the sweep only grades dead
// pids); else an alive .pid is live; else header-only means spawn-failed; else
// the unit leaked — dead, real work attempted, never graded.
func ClassifyUnit(log string, alive AliveProbe) Unit {
	name := filepath.Base(log)
	m := logRE.FindStringSubmatch(name)
	issue, _ := strconv.Atoi(m[2])
	lane, backend, body := ReadSpawnHeader(log)
	stamp, _ := ParseLogStampUTC(name)
	u := Unit{
		Log: name, Kind: m[1], Issue: issue, Lane: lane, Backend: backend,
		SpawnedUTC: stamp.UTC().Format("2006-01-02T15:04:05Z"),
		stamp:      stamp,
	}
	if w, ok := readWitness(log); ok {
		switch asString(w["claim"]) {
		case "CLAIM_WITNESSED":
			u.Outcome = OutcomeWitnessed
			u.SHA = asString(w["sha"])
		case "CLAIM_UNWITNESSED":
			u.Outcome = OutcomeUnwitnessed
			u.SHA = asString(w["sha"])
		default:
			u.Outcome = OutcomeNoCommit
			reason := asString(w["reason"])
			if !noCommitReasons[reason] {
				reason = "unknown"
			}
			// The sweep's residual "unknown" is the fleet's largest terminal bucket
			// (#5867). Split it from the log tail the sweep itself read, so a bare
			// "unknown" survives only as a vocabulary-drift alarm (see
			// refineUnknownNoCommit) — never as the label for half the fleet.
			if asString(w["reason"]) == "unknown" {
				reason = refineUnknownNoCommit(log)
			}
			u.Reason = reason
		}
		return u
	}
	if raw, err := os.ReadFile(sidecar(log, ".pid")); err == nil {
		if pid, err := strconv.Atoi(strings.TrimSpace(string(raw))); err == nil && alive.alive(pid) {
			u.Outcome = OutcomeLive
			u.PID = pid
			return u
		}
	}
	if !body {
		u.Outcome = OutcomeSpawnFailed
		return u
	}
	u.Outcome = OutcomeLeaked
	return u
}

// CollectUnits returns every worker-unit spent inside the window (spawn stamp >=
// since), classified and sorted oldest-first so reports read chronologically.
func CollectUnits(runsDir string, since time.Time, alive AliveProbe) []Unit {
	logs, _ := filepath.Glob(filepath.Join(runsDir, "*.log"))
	units := make([]Unit, 0, len(logs))
	for _, log := range logs {
		stamp, ok := ParseLogStampUTC(filepath.Base(log))
		if !ok || stamp.Before(since) {
			continue
		}
		units = append(units, ClassifyUnit(log, alive))
	}
	sort.SliceStable(units, func(i, j int) bool { return units[i].stamp.Before(units[j].stamp) })
	return units
}

// Closes is the windowed issue-close picture from progress.jsonl.
type Closes struct {
	ClosedInWindow int
	OpenNow        *int
	BaselineOpen   *int
}

type progressRow struct {
	UTC          string `json:"utc"`
	ClosedNow    *int   `json:"closed_now"`
	OpenNow      *int   `json:"open_now"`
	BaselineOpen *int   `json:"baseline_open"`
}

// WindowedCloses sums closed_now over in-window progress rows and tracks the
// newest open/baseline picture (the last in-window row, in file order, that
// carries them).
func WindowedCloses(progressPath string, since time.Time) Closes {
	var out Closes
	for _, line := range readLines(progressPath) {
		var rec progressRow
		if json.Unmarshal([]byte(line), &rec) != nil {
			continue
		}
		ts, ok := parseISOUTC(rec.UTC)
		if !ok || ts.Before(since) {
			continue
		}
		if rec.ClosedNow != nil && *rec.ClosedNow > 0 {
			out.ClosedInWindow += *rec.ClosedNow
		}
		if rec.OpenNow != nil {
			out.OpenNow = rec.OpenNow
		}
		if rec.BaselineOpen != nil {
			out.BaselineOpen = rec.BaselineOpen
		}
	}
	return out
}

// Holds is the windowed contract-hold pressure: issues the gate kept OUT of
// dispatch (they spent no unit, but they are demand the window did not serve).
type Holds struct {
	Rows           int `json:"rows"`
	DistinctIssues int `json:"distinct_issues"`
}

type holdRow struct {
	TS    *float64 `json:"ts"`
	UTC   string   `json:"utc"`
	Issue *int     `json:"issue"`
}

// WindowedContractHolds counts contract-hold rows recorded inside the window and
// the distinct issues they name.
func WindowedContractHolds(holdsPath string, since time.Time) Holds {
	issues := map[int]bool{}
	rows := 0
	for _, line := range readLines(holdsPath) {
		var rec holdRow
		if json.Unmarshal([]byte(line), &rec) != nil {
			continue
		}
		var ts time.Time
		if rec.TS != nil {
			ts = time.Unix(int64(*rec.TS), 0).UTC()
		} else {
			var ok bool
			if ts, ok = parseISOUTC(rec.UTC); !ok {
				continue
			}
		}
		if ts.Before(since) {
			continue
		}
		rows++
		if rec.Issue != nil {
			issues[*rec.Issue] = true
		}
	}
	return Holds{Rows: rows, DistinctIssues: len(issues)}
}

// Report is the conservation identity over one window.
type Report struct {
	Schema        string       `json:"schema"`
	UTC           string       `json:"utc"`
	WindowH       float64      `json:"window_h"`
	Verdict       string       `json:"verdict"`
	Units         Units        `json:"units"`
	IdentityHolds bool         `json:"identity_holds"`
	Yield         Yield        `json:"yield"`
	Churn         Churn        `json:"churn"`
	ContractHolds Holds        `json:"contract_holds"`
	LeakedUnits   []LeakedUnit `json:"leaked_units"`
}

// Units is the closed-outcome fold over resolve worker-units (repair counted
// separately).
type Units struct {
	ResolveTotal         int            `json:"resolve_total"`
	Live                 int            `json:"live"`
	Spent                int            `json:"spent"`
	ShippedWitnessed     int            `json:"shipped_witnessed"`
	CommittedUnwitnessed int            `json:"committed_unwitnessed"`
	NoCommit             int            `json:"no_commit"`
	NoCommitReasons      map[string]int `json:"no_commit_reasons"`
	SpawnFailed          int            `json:"spawn_failed"`
	LeakedUnswept        int            `json:"leaked_unswept"`
	RepairTotal          int            `json:"repair_total"`
}

// Yield is the ship-per-spend and issue-close side of the report.
type Yield struct {
	WitnessedPerSpent    *float64 `json:"witnessed_per_spent"`
	IssuesClosedInWindow int      `json:"issues_closed_in_window"`
	OpenNow              *int     `json:"open_now"`
	BaselineOpen         *int     `json:"baseline_open"`
}

// Churn surfaces issues that burned 2+ finished units in one window (the
// cooldown-loop signature).
type Churn struct {
	IssuesWith2Plus int          `json:"issues_with_2plus_units"`
	Worst           []ChurnEntry `json:"worst"`
}

// ChurnEntry is one issue's re-storm burn count.
type ChurnEntry struct {
	Issue int `json:"issue"`
	Units int `json:"units"`
}

// LeakedUnit is one dead-and-ungraded worker-unit, listed so silence reads as a
// number.
type LeakedUnit struct {
	Log        string `json:"log"`
	Issue      int    `json:"issue"`
	Lane       string `json:"lane"`
	Backend    string `json:"backend"`
	SpawnedUTC string `json:"spawned_utc"`
}

// FoldConservation folds classified units + the issue side into the conservation
// identity over one window. Pure: data in, report out.
func FoldConservation(units []Unit, closes Closes, holds Holds, windowH float64, nowISO string) Report {
	var resolve, repair []Unit
	for _, u := range units {
		switch u.Kind {
		case "resolve":
			resolve = append(resolve, u)
		case "repair":
			repair = append(repair, u)
		}
	}

	byOutcome := map[string]int{}
	noCommit := map[string]int{}
	for _, u := range resolve {
		byOutcome[u.Outcome]++
		if u.Outcome == OutcomeNoCommit {
			noCommit[u.Reason]++
		}
	}

	live := byOutcome[OutcomeLive]
	spent := len(resolve) - live
	witnessed := byOutcome[OutcomeWitnessed]
	unwitnessed := byOutcome[OutcomeUnwitnessed]
	refused := byOutcome[OutcomeNoCommit]
	spawnFailed := byOutcome[OutcomeSpawnFailed]
	leaked := byOutcome[OutcomeLeaked]

	// Re-storm churn: issues that burned 2+ finished units in ONE window are the
	// cooldown-loop signature — each extra unit is capacity another issue never got.
	attempts := map[int]int{}
	for _, u := range resolve {
		if u.Outcome != OutcomeLive {
			attempts[u.Issue]++
		}
	}
	churned := make([]ChurnEntry, 0)
	for issue, c := range attempts {
		if c >= 2 {
			churned = append(churned, ChurnEntry{Issue: issue, Units: c})
		}
	}
	sort.Slice(churned, func(i, j int) bool { return churned[i].Issue < churned[j].Issue })
	worst := append([]ChurnEntry(nil), churned...)
	sort.SliceStable(worst, func(i, j int) bool { return worst[i].Units > worst[j].Units })
	if len(worst) > 5 {
		worst = worst[:5]
	}
	if worst == nil {
		worst = []ChurnEntry{}
	}

	verdict := "CONSERVED"
	if leaked != 0 {
		verdict = "LEAKING"
	}

	var wps *float64
	if spent != 0 {
		v := math.Round(float64(witnessed)/float64(spent)*10000) / 10000
		wps = &v
	}

	leakedUnits := make([]LeakedUnit, 0)
	for _, u := range resolve {
		if u.Outcome == OutcomeLeaked {
			leakedUnits = append(leakedUnits, LeakedUnit{
				Log: u.Log, Issue: u.Issue, Lane: u.Lane,
				Backend: u.Backend, SpawnedUTC: u.SpawnedUTC,
			})
		}
	}
	if noCommit == nil {
		noCommit = map[string]int{}
	}

	return Report{
		Schema:  Schema,
		UTC:     nowISO,
		WindowH: windowH,
		Verdict: verdict,
		Units: Units{
			ResolveTotal:         len(resolve),
			Live:                 live,
			Spent:                spent,
			ShippedWitnessed:     witnessed,
			CommittedUnwitnessed: unwitnessed,
			NoCommit:             refused,
			NoCommitReasons:      noCommit,
			SpawnFailed:          spawnFailed,
			LeakedUnswept:        leaked,
			RepairTotal:          len(repair),
		},
		IdentityHolds: spent == witnessed+unwitnessed+refused+spawnFailed+leaked,
		Yield: Yield{
			WitnessedPerSpent:    wps,
			IssuesClosedInWindow: closes.ClosedInWindow,
			OpenNow:              closes.OpenNow,
			BaselineOpen:         closes.BaselineOpen,
		},
		Churn: Churn{
			IssuesWith2Plus: len(churned),
			Worst:           worst,
		},
		ContractHolds: holds,
		LeakedUnits:   leakedUnits,
	}
}

// Render is the human summary. ASCII-only: the Windows console renders under cp1252.
func Render(r Report) string {
	u := r.Units
	y := r.Yield
	identity := "holds"
	if !r.IdentityHolds {
		identity = "BROKEN"
	}
	lines := []string{
		fmt.Sprintf("dispatch conservation -- %s  window=%gh  %s", r.Verdict, r.WindowH, r.UTC),
		fmt.Sprintf("  units: %d resolve (%d live) + %d repair; spent=%d",
			u.ResolveTotal, u.Live, u.RepairTotal, u.Spent),
		fmt.Sprintf("  spent = %d shipped + %d unwitnessed-commit + %d no-commit + %d spawn-failed + %d LEAKED  (identity %s)",
			u.ShippedWitnessed, u.CommittedUnwitnessed, u.NoCommit, u.SpawnFailed, u.LeakedUnswept, identity),
	}
	if len(u.NoCommitReasons) > 0 {
		reasons := make([]string, 0, len(u.NoCommitReasons))
		for _, k := range sortedKeys(u.NoCommitReasons) {
			reasons = append(reasons, fmt.Sprintf("%s=%d", k, u.NoCommitReasons[k]))
		}
		lines = append(lines, "  no-commit reasons: "+strings.Join(reasons, ", "))
	}
	yieldLine := fmt.Sprintf("  yield: witnessed/spent=%s  closes-in-window=%d",
		fmtPtrFloat(y.WitnessedPerSpent), y.IssuesClosedInWindow)
	if y.OpenNow != nil {
		yieldLine += fmt.Sprintf("  open_now=%d (baseline %s)", *y.OpenNow, fmtPtrInt(y.BaselineOpen))
	}
	lines = append(lines, yieldLine)
	if r.Churn.IssuesWith2Plus > 0 {
		worst := make([]string, 0, len(r.Churn.Worst))
		for _, w := range r.Churn.Worst {
			worst = append(worst, fmt.Sprintf("#%dx%d", w.Issue, w.Units))
		}
		lines = append(lines, fmt.Sprintf("  churn: %d issue(s) burned 2+ units (%s)",
			r.Churn.IssuesWith2Plus, strings.Join(worst, ", ")))
	}
	if r.ContractHolds.Rows > 0 {
		lines = append(lines, fmt.Sprintf("  contract holds: %d row(s), %d distinct issue(s) kept out of dispatch",
			r.ContractHolds.Rows, r.ContractHolds.DistinctIssues))
	}
	for _, leak := range r.LeakedUnits {
		lines = append(lines, fmt.Sprintf("  LEAK %s  issue=#%d lane=%s spawned=%s",
			leak.Log, leak.Issue, leak.Lane, leak.SpawnedUTC))
	}
	if r.Verdict == "CONSERVED" {
		lines = append(lines, "  every finished unit is accounted; leaks would be listed above")
	}
	return strings.Join(lines, "\n")
}

// Run is the CLI entry point. aliveProvider snapshots host liveness (injectable
// for tests); now is the reporting clock (injectable). It returns the process
// exit code.
func Run(argv []string, aliveProvider func() AliveProbe, now time.Time, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("dispatch-conservation", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", "", fmt.Sprintf("dispatch runs dir (default: <cwd>/%s)", DefaultRunsDirname))
	windowH := fs.Float64("window-h", DefaultWindowH, "window in hours")
	asJSON := fs.Bool("json", false, "emit the machine-readable report")
	failOnLeak := fs.Int("fail-on-leak", 0, "exit 1 when leaked_unswept exceeds N (default: report-only)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	gateSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "fail-on-leak" {
			gateSet = true
		}
	})

	dir := *runsDir
	if dir == "" {
		wd, _ := os.Getwd()
		dir = filepath.Join(wd, DefaultRunsDirname)
	}
	since := now.Add(-time.Duration(*windowH * float64(time.Hour)))
	nowISO := now.UTC().Format("2006-01-02T15:04:05Z")

	units := CollectUnits(dir, since, aliveProvider())
	closes := WindowedCloses(filepath.Join(dir, "progress.jsonl"), since)
	holds := WindowedContractHolds(filepath.Join(dir, "contract-holds.jsonl"), since)
	report := FoldConservation(units, closes, holds, *windowH, nowISO)

	if *asJSON {
		b, _ := json.MarshalIndent(report, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintln(stdout, Render(report))
	}
	if gateSet && report.Units.LeakedUnswept > *failOnLeak {
		return 1
	}
	return 0
}

// --- small helpers ---

func sidecar(log, suffix string) string {
	return strings.TrimSuffix(log, ".log") + suffix
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func parseISOUTC(ts string) (time.Time, bool) {
	ts = strings.TrimSpace(ts)
	if ts == "" {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return time.Time{}, false
	}
	return t.UTC(), true
}

func readLines(path string) []string { return ReadTailLines(path, DefaultTailLines) }

func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func fmtPtrFloat(p *float64) string {
	if p == nil {
		return "None"
	}
	return strconv.FormatFloat(*p, 'g', -1, 64)
}

func fmtPtrInt(p *int) string {
	if p == nil {
		return "None"
	}
	return strconv.Itoa(*p)
}
