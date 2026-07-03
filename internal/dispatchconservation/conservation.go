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
//	                    + no_commit{self_modify|policy_block|auth_wall|
//	                                off_trunk|banner_noop|unknown}
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
var noCommitReasons = map[string]bool{
	"self_modify": true, "policy_block": true, "auth_wall": true,
	"off_trunk": true, "banner_noop": true, "unknown": true,
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

func readLines(path string) []string {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []string
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

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
