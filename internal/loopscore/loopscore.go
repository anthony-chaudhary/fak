// Package loopscore scores the AGENTIC BACKGROUND LOOPS themselves — the always-on
// processes (issue dispatch, resolve-progress, freshness cadences, smoke loops) that
// keep the fleet moving while no human is watching. The question it answers: "are
// these loops first-class, durable, self-reporting processes, or fire-and-forget
// scripts that vanish on the next reboot and report nothing?"
//
// It grades the loops on the three axes the loop program names, all re-derived from
// the on-disk loop ledger + job registry fak's own tooling writes — so the score
// cannot be moved by editing a JSON file, only by making the loops genuinely better:
//
//   - DURABILITY (auto-restart on system restart): does every loop that actually
//     fires survive a reboot? A firing loop with no registry cadence is re-launched
//     by nobody after the box restarts. Measured from the registry (registered,
//     armed, cron-emittable) joined against which loops the ledger shows firing.
//   - SELF-REPORT (status): does each loop surface its own liveness and outcome
//     without a human tailing logs? Measured from the health fold (no dark loop) and
//     the ledger (fires that record an end outcome, heartbeat/notify self-reports).
//   - DOGFOOD (use fak's own tooling): does the loop run THROUGH fak's surface —
//     `fak loop run` under `fak guard`, the canonical hash-chained ledger, the
//     witness contract — rather than raw cron + an ad-hoc log? Measured from the run
//     metrics (guard_enabled), the ledger's presence, and the witness verdicts.
//
// Every number is re-derived from disk via the same loopmgr folds `fak loop status`
// and `fak loop health` use — so this scorecard DOGFOODS the very ledger it grades.
// Driving the loopscore_debt down IS making the background loops durable, observable,
// and fak-native: register the firing loops, run them through `fak loop run`, and let
// them witness their own outcomes.
package loopscore

import (
	"math"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	Schema = "fak-loopscore-scorecard/1"

	// Metric keys the run wrapper / dispatchers record on loop events. argc is
	// stamped by `fak loop run` (the run went through fak's surface, not raw exec);
	// guard_enabled is the containment bit (1 under `fak guard`, 0 with --no-guard).
	metricArgc         = "argc"
	metricGuardEnabled = "guard_enabled"

	// Axis weights. Durability leads because a loop that does not survive a reboot is
	// not a background process at all — it is a one-shot script. Self-report is next:
	// a loop nobody can see the state of is operationally dark. Dogfooding is the
	// thinnest axis here and the one the loop program is built to grow.
	durabilityWeight = 0.40
	selfReportWeight = 0.35
	dogfoodWeight    = 0.25

	// passShare is the fraction of a population a SHARE-style KPI must reach to pass.
	passShare = 0.90
)

// KPIResult is one graded criterion. Identical shape to conceptusage.KPIResult /
// dogfoodscore.KPIResult so the three scorecards render and fold the same way.
type KPIResult struct {
	Key    string `json:"key"`
	Label  string `json:"label"`
	Hard   bool   `json:"hard"`
	Weight int    `json:"weight"`
	Axis   string `json:"axis"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail"`
}

type KPIPayload struct {
	KPI     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   int      `json:"score"`
	Value   float64  `json:"value"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

// Evidence is the raw, re-derived-from-disk corpus the KPIs read. Exported so a
// caller (the verb, a test) can inspect exactly what was counted. Every field is a
// fleet-wide tally over the loop ledger and the job registry.
type Evidence struct {
	ActiveLoops int `json:"active_loops"` // loops registered OR ledgered (the fleet)
	Registered  int `json:"registered"`   // loops with a registry cadence definition
	Ledgered    int `json:"ledgered"`     // loops with >=1 ledger event
	Fired       int `json:"fired"`        // loops with >=1 fire event

	FiredRegistered int `json:"fired_registered"` // firing loops that ARE registered (survive reboot)
	Armed           int `json:"armed"`            // registered loops in the armed state (re-fire at boot)
	CronEmittable   int `json:"cron_emittable"`   // registered loops with a cadence fak cron emit can project

	Dark int `json:"dark"` // active loops the health fold marks dark (silent / never observed)

	TotalFires int `json:"total_fires"` // sum of fire events across active loops
	TotalEnds  int `json:"total_ends"`  // sum of end events (capped per loop at its fires)

	HeartbeatOrNotify int `json:"heartbeat_or_notify"` // loops that emit heartbeat/notify self-reports

	RanViaFakLoop int `json:"ran_via_fak_loop"` // loops with an argc metric (went through `fak loop run`)
	GuardWrapped  int `json:"guard_wrapped"`    // ran-via-fak-loop loops with guard_enabled==1
	Witnessed     int `json:"witnessed"`        // loops with >=1 witnessed-done verdict

	LedgerPresent bool `json:"ledger_present"` // the canonical loop ledger exists and has events
}

type ScorecardPayload struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace"`
	Corpus     map[string]any `json:"corpus"`
	KPIs       []KPIPayload   `json:"kpis"`
	Durability []KPIResult    `json:"durability"`
	SelfReport []KPIResult    `json:"self_report"`
	Dogfood    []KPIResult    `json:"dogfood"`
	Evidence   Evidence       `json:"evidence"`
}

// Options pins the inputs and clock so the score is deterministic for tests.
type Options struct {
	Root         string
	LedgerPath   string
	RegistryPath string
	Now          time.Time

	// events/registry override the disk reads for tests; nil/empty means load from
	// LedgerPath / RegistryPath via loopmgr.
	events    []loopmgr.Event
	registry  *loopmgr.Registry
	useInputs bool
}

func (o Options) normalize() Options {
	if o.Root == "" {
		o.Root = "."
	}
	if o.Now.IsZero() {
		o.Now = time.Now().UTC()
	}
	if o.LedgerPath == "" {
		o.LedgerPath = filepath.Join(o.Root, ".fak", "loops.jsonl")
	}
	if o.RegistryPath == "" {
		o.RegistryPath = filepath.Join(o.Root, "tools", "loop-registry.json")
	}
	return o
}

// ---- evidence gathering (the impure shell, kept thin) -----------------------------

func gatherEvidence(opts Options) Evidence {
	var events []loopmgr.Event
	var reg loopmgr.Registry
	if opts.useInputs {
		events = opts.events
		if opts.registry != nil {
			reg = *opts.registry
		}
	} else {
		// LoadPrefix (not Load) tolerates a broken tail the way `fak loop health` does:
		// the loop ledger is a shared, concurrently-appended file, so a torn final line
		// must degrade to "score the valid prefix", never "go blind on every loop". A
		// missing ledger returns no events and no error.
		events, _, _ = loopmgr.LoadPrefix(opts.LedgerPath)
		reg, _ = loopmgr.LoadRegistry(opts.RegistryPath)
	}

	st := loopmgr.Summarize(events, opts.Now)
	health := loopmgr.FoldHealth(st, reg, opts.Now, loopmgr.HealthThresholds{})

	var ev Evidence
	ev.LedgerPresent = len(events) > 0

	// Per-loop run signals are read from RAW events, not the folded snapshot: the
	// snapshot does not carry heartbeat counts, and deriving the run metrics (argc =
	// "went through `fak loop run`", guard_enabled = "ran under `fak guard`") straight
	// from the events keeps this fold independent of any optional snapshot field.
	heartbeat := map[string]bool{}
	ranViaFakLoop := map[string]bool{}
	guardOn := map[string]bool{}
	for _, e := range events {
		if e.Kind == loopmgr.EventHeartbeat {
			heartbeat[e.LoopID] = true
		}
		if _, ok := e.Metrics[metricArgc]; ok {
			ranViaFakLoop[e.LoopID] = true
		}
		if e.Metrics[metricGuardEnabled] == 1 {
			guardOn[e.LoopID] = true
		}
	}

	registered := map[string]loopmgr.Job{}
	for _, j := range reg.List() {
		registered[j.JobID()] = j
		ev.Registered++
		if j.State.Armed() {
			ev.Armed++
		}
		if j.Schedule.IntervalSeconds > 0 {
			ev.CronEmittable++
		}
	}

	// The active fleet is the union of ledgered loops and registered jobs.
	active := map[string]struct{}{}
	for _, s := range st.Loops {
		active[s.LoopID] = struct{}{}
	}
	for id := range registered {
		active[id] = struct{}{}
	}
	ev.ActiveLoops = len(active)

	for _, s := range st.Loops {
		ev.Ledgered++
		if s.Fires > 0 {
			ev.Fired++
			if _, ok := registered[s.LoopID]; ok {
				ev.FiredRegistered++
			}
		}
		ev.TotalFires += int(s.Fires)
		// Cap ends at fires per loop so an append-only outcome ledger (ends without a
		// matching fire) cannot inflate the completion rate above 100%.
		ends := int(s.Ended)
		if ends > int(s.Fires) {
			ends = int(s.Fires)
		}
		ev.TotalEnds += ends
		if s.Notifications > 0 || heartbeat[s.LoopID] {
			ev.HeartbeatOrNotify++
		}
		if s.Witnessed > 0 {
			ev.Witnessed++
		}
	}

	// Dogfood run signals, folded from the raw-event maps over the loops that ran
	// through `fak loop run` (argc present); guard-wrapped is the share of those with
	// guard_enabled=1.
	for id := range ranViaFakLoop {
		ev.RanViaFakLoop++
		if guardOn[id] {
			ev.GuardWrapped++
		}
	}

	for _, row := range health.Rows {
		if row.Dark {
			ev.Dark++
		}
	}
	return ev
}

// ---- KPI definitions --------------------------------------------------------------

func shareOK(num, den int) bool {
	if den <= 0 {
		return true // vacuous: an empty population is not a failure (no slander on a fresh tree)
	}
	return float64(num)/float64(den) >= passShare
}

func pctStr(num, den int) string {
	if den <= 0 {
		return "n/a"
	}
	return itoa(int(math.Round(100*float64(num)/float64(den)))) + "%"
}

// durabilityResults grade auto-restart on system restart: would the fleet come back
// after a reboot without a human re-launching it?
func durabilityResults(ev Evidence) []KPIResult {
	const axis = "durability"
	return []KPIResult{
		result("firing_loops_registered", axis, true, 3,
			"every loop that actually fires is registered with a cadence, so the OS scheduler re-arms it after a reboot",
			shareOK(ev.FiredRegistered, ev.Fired),
			itoa(ev.FiredRegistered)+"/"+itoa(ev.Fired)+" ("+pctStr(ev.FiredRegistered, ev.Fired)+") firing loops are registered"),
		result("registered_armed", axis, true, 2,
			"registered jobs are armed (a stopped/disabled job will NOT re-fire after a restart)",
			shareOK(ev.Armed, ev.Registered),
			itoa(ev.Armed)+"/"+itoa(ev.Registered)+" ("+pctStr(ev.Armed, ev.Registered)+") registered jobs armed"),
		result("cron_emittable", axis, false, 1,
			"registered jobs carry a cadence `fak cron emit` can project to a launchd/systemd/task-scheduler unit",
			shareOK(ev.CronEmittable, ev.Registered),
			itoa(ev.CronEmittable)+"/"+itoa(ev.Registered)+" ("+pctStr(ev.CronEmittable, ev.Registered)+") cron-emittable"),
	}
}

// selfReportResults grade whether each loop surfaces its own liveness and outcome.
func selfReportResults(ev Evidence) []KPIResult {
	const axis = "self_report"
	return []KPIResult{
		result("no_dark_loop", axis, true, 3,
			"no active loop is dark — every registered/firing loop is observably ticking, not silent",
			ev.Dark == 0,
			itoa(ev.Dark)+" dark loop(s) across "+itoa(ev.ActiveLoops)+" active loop(s)"),
		result("outcome_recorded", axis, true, 2,
			"loops that fire also record an end outcome in the ledger (not fire-and-forget)",
			shareOK(ev.TotalEnds, ev.TotalFires),
			itoa(ev.TotalEnds)+"/"+itoa(ev.TotalFires)+" ("+pctStr(ev.TotalEnds, ev.TotalFires)+") fires recorded an end"),
		result("heartbeat_or_notify", axis, false, 2,
			"loops self-report liveness via heartbeat/notify events, not just fire/end",
			ev.ActiveLoops == 0 || ev.HeartbeatOrNotify > 0,
			itoa(ev.HeartbeatOrNotify)+"/"+itoa(ev.ActiveLoops)+" loop(s) emit a heartbeat or notify self-report"),
	}
}

// dogfoodResults grade whether the loop runs THROUGH fak's own tooling.
func dogfoodResults(ev Evidence) []KPIResult {
	const axis = "dogfood"
	return []KPIResult{
		result("guard_wrapped", axis, true, 3,
			"loop runs route through `fak guard` (guard_enabled=1) — the containment wrapper, not raw exec",
			shareOK(ev.GuardWrapped, ev.RanViaFakLoop),
			itoa(ev.GuardWrapped)+"/"+itoa(ev.RanViaFakLoop)+" ("+pctStr(ev.GuardWrapped, ev.RanViaFakLoop)+") runs guard-wrapped"),
		result("canonical_ledger", axis, true, 1,
			"the loops append to the canonical hash-chained loop ledger (fak loop), not an ad-hoc log",
			ev.LedgerPresent,
			boolStr(ev.LedgerPresent, "ledger present with events", "no loop ledger events found")),
		result("runs_witnessed", axis, false, 2,
			"loop runs reach a witnessed-done verdict — the loop dogfoods the witness contract",
			ev.ActiveLoops == 0 || ev.Witnessed > 0,
			itoa(ev.Witnessed)+"/"+itoa(ev.ActiveLoops)+" loop(s) reached a witnessed-done verdict"),
	}
}

// ---- fold -------------------------------------------------------------------------

// axisWeights names the top-level axis weight for each KPIResult.Axis, mirroring the
// durabilityWeight/selfReportWeight/dogfoodWeight constants so kpiWeights below can
// derive the shared kernel's per-KPI weight from exactly the same numbers this card
// has always graded with (no retune).
var axisWeights = map[string]float64{
	"durability":  durabilityWeight,
	"self_report": selfReportWeight,
	"dogfood":     dogfoodWeight,
}

// toKPI converts one KPIResult into a scorecard.KPI, preserving the HARD/SOFT split
// (Fold sums len(Defects) as debt; Soft entries never gate). Score is 100/0 per-row
// exactly as kpiPayloads has always rendered it.
func toKPI(r KPIResult) scorecard.KPI {
	k := scorecard.KPI{Key: r.Key, Group: r.Axis, Detail: r.Detail}
	if r.Passed {
		k.Score = 100
	} else if r.Hard {
		k.Defects = []string{r.Key + ": " + r.Detail}
	} else {
		k.Soft = []string{r.Key + ": " + r.Detail}
	}
	return k
}

// kpiWeights derives the shared kernel's per-KPI weight (keyed by KPI Key, since Fold
// tries Group before Key and every row's Group is its shared axis name) from each
// row's own within-axis Weight and that axis's top-level weight. Scaling each row's
// weight by axisWeight/axisWeightSum reproduces, bit-for-bit, the two-level weighted
// mean this card has always computed (per-axis weighted share of axisScore, then the
// three axes combined by durability/self-report/dogfood weight) as a SINGLE weighted
// mean over all rows -- which is exactly what scorecard.Fold's weightedMean takes.
func kpiWeights(all []KPIResult) map[string]float64 {
	axisWeightSum := map[string]float64{}
	for _, r := range all {
		axisWeightSum[r.Axis] += float64(r.Weight)
	}
	w := make(map[string]float64, len(all))
	for _, r := range all {
		sum := axisWeightSum[r.Axis]
		if sum == 0 {
			continue
		}
		w[r.Key] = axisWeights[r.Axis] * float64(r.Weight) / sum
	}
	return w
}

func Build(opts Options) ScorecardPayload {
	opts = opts.normalize()
	root, _ := filepath.Abs(opts.Root)
	if root == "" {
		root = opts.Root
	}
	ev := gatherEvidence(opts)

	durability := durabilityResults(ev)
	selfReport := selfReportResults(ev)
	dogfood := dogfoodResults(ev)
	all := append(append(append([]KPIResult{}, durability...), selfReport...), dogfood...)

	dScore := axisScore(durability)
	sScore := axisScore(selfReport)
	gScore := axisScore(dogfood)
	// composite is rounded to an int BEFORE grading, exactly as this card has always
	// done (GradeLetter graded the already-rounded display value, not the raw mean) --
	// preserved here so porting onto the shared kernel cannot shift a grade at a
	// boundary the old int-rounding-then-grade order would not have crossed.
	composite := int(math.Round(durabilityWeight*float64(dScore) + selfReportWeight*float64(sScore) + dogfoodWeight*float64(gScore)))
	grade := GradeLetter(composite)

	kpis := make([]scorecard.KPI, len(all))
	for i, r := range all {
		kpis[i] = toKPI(r)
	}

	var hardFail []KPIResult
	for _, r := range all {
		if r.Hard && !r.Passed {
			hardFail = append(hardFail, r)
		}
	}
	// loopscore_debt = one per HARD gap, unchanged: Fold sums len(Defects) across the
	// KPIs, and toKPI puts exactly one Defect on each failing HARD row.
	debt := len(hardFail)
	ok := debt == 0

	finding, next, reason := "loops_durable_observable_native", "hold the line; re-run after a loop session — keep firing loops registered and guard-wrapped", ""
	if ok {
		reason = "loop-score: durability value " + valueStr(dScore) + ", self-report value " + valueStr(sScore) +
			", dogfood value " + valueStr(gScore) + ", composite value " + valueStr(composite) + " (" + grade + ", legacy score " + itoa(composite) +
			"); the background loops are registered, observable, and fak-native; zero hard gaps"
	} else {
		finding = "loopscore_debt"
		keys := make([]string, len(hardFail))
		for i, r := range hardFail {
			keys[i] = r.Key
		}
		reason = "loop-score carries " + itoa(debt) + " debt (durability value " + valueStr(dScore) +
			", self-report value " + valueStr(sScore) + ", dogfood value " + valueStr(gScore) + ", composite value " +
			valueStr(composite) + " " + grade + ", legacy score " + itoa(composite) + "): " + strings.Join(keys, ", ")
		lead := hardFail[0]
		next = "retire worst-first: " + lead.Key + " — " + lead.Detail
	}

	// Grade ignores the kernel's own raw weighted-mean input and instead grades the
	// pre-rounded int composite computed above, so the letter grade is bit-identical
	// to the pre-port fold; ExtraCorpus likewise overrides the kernel-written score
	// (Round1 of the raw mean) with the same int composite this card has always
	// reported, and loopscore_debt/grade with the values just derived.
	p := scorecard.Fold(Schema, kpis, "loopscore_debt", kpiWeights(all), scorecard.Messages{
		Grade:           func(float64) string { return grade },
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		Reason:          reason,
		ExtraCorpus: map[string]any{
			"loopscore_debt":    debt,
			"score":             composite,
			"durability_value":  scorecard.Round3(scorecard.ValueFromScore(float64(dScore))),
			"grade":             grade,
			"durability_score":  dScore,
			"self_report_value": scorecard.Round3(scorecard.ValueFromScore(float64(sScore))),
			"self_report_score": sScore,
			"dogfood_value":     scorecard.Round3(scorecard.ValueFromScore(float64(gScore))),
			"dogfood_score":     gScore,
			"active_loops":      ev.ActiveLoops,
			"registered":        ev.Registered,
			"fired":             ev.Fired,
			"dark":              ev.Dark,
			"guard_wrapped":     ev.GuardWrapped,
			"witnessed":         ev.Witnessed,
		},
	})

	return ScorecardPayload{
		Schema:     p.Schema,
		OK:         p.OK,
		Verdict:    p.Verdict,
		Finding:    p.Finding,
		Reason:     p.Reason,
		NextAction: p.NextAction,
		Workspace:  root,
		Corpus:     p.Corpus,
		KPIs:       kpiPayloads(all),
		Durability: durability,
		SelfReport: selfReport,
		Dogfood:    dogfood,
		Evidence:   ev,
	}
}

// ---- render -----------------------------------------------------------------------

func Render(p ScorecardPayload) string {
	c := p.Corpus
	lines := []string{
		"loop-score — " + p.Verdict + " (" + p.Finding + ")",
		"  loopscore_debt: " + anyStr(c["loopscore_debt"]) + "   value " + anyStr(c["value"]) +
			" [" + anyStr(c["grade"]) + "]   (legacy score " + anyStr(c["score"]) + "; durability value " + anyStr(c["durability_value"]) +
			"; self-report value " + anyStr(c["self_report_value"]) + "; dogfood value " + anyStr(c["dogfood_value"]) + ")",
		"  evidence: " + anyStr(c["active_loops"]) + " active loop(s); " + anyStr(c["registered"]) +
			" registered; " + anyStr(c["fired"]) + " firing; " + anyStr(c["dark"]) + " dark; " +
			anyStr(c["guard_wrapped"]) + " guard-wrapped; " + anyStr(c["witnessed"]) + " witnessed",
		"",
		"  DURABILITY (auto-restart on system restart):",
	}
	for _, r := range p.Durability {
		lines = append(lines, "    "+passMark(r.Passed)+" "+r.Label+"  ["+r.Detail+"]")
	}
	lines = append(lines, "", "  SELF-REPORT (does each loop surface its own status?):")
	for _, r := range p.SelfReport {
		lines = append(lines, "    "+passMark(r.Passed)+" "+r.Label+"  ["+r.Detail+"]")
	}
	lines = append(lines, "", "  DOGFOOD (does the loop run THROUGH fak's tooling?):")
	for _, r := range p.Dogfood {
		lines = append(lines, "    "+passMark(r.Passed)+" "+r.Label+"  ["+r.Detail+"]")
	}
	lines = append(lines, "", "  NEXT: "+p.NextAction)
	return strings.Join(lines, "\n")
}

func Markdown(p ScorecardPayload) string {
	c := p.Corpus
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString(`title: "fak loop scorecard"` + "\n")
	b.WriteString(`description: "Whether fak's always-on agentic background loops are first-class durable processes — they survive a system restart (auto-restart), self-report their own status, and run through fak's own tooling — re-derived from the loop ledger + job registry fak writes, never a self-report."` + "\n")
	b.WriteString("---\n\n")
	b.WriteString("# fak loop scorecard\n\n")
	b.WriteString("**loopscore_debt: " + anyStr(c["loopscore_debt"]) + "**; value **" + anyStr(c["value"]) +
		" (" + anyStr(c["grade"]) + ")**; legacy score " + anyStr(c["score"]) +
		"; durability value " + anyStr(c["durability_value"]) + "; self-report value " +
		anyStr(c["self_report_value"]) + "; dogfood value " + anyStr(c["dogfood_value"]) + "\n\n")
	b.WriteString("> " + p.Reason + "\n\n")
	b.WriteString("The question: are fak's always-on background loops — the issue dispatchers, the resolve-progress tracker, the freshness cadences, the smoke loops — *first-class durable processes*, or fire-and-forget scripts that vanish on the next reboot and report nothing? Every number is re-derived from the loop ledger (`.fak/loops.jsonl`) and the job registry (`tools/loop-registry.json`) fak's own `fak loop` tooling writes, folded with the same `loopmgr` projection `fak loop status` / `fak loop health` use — so the score moves only when the loops actually become more durable, observable, and fak-native.\n\n")
	b.WriteString("## Durability — auto-restart on system restart\n\n| ok | criterion | detail |\n|---|---|---|\n")
	for _, r := range p.Durability {
		b.WriteString("| " + passMark(r.Passed) + " | " + r.Label + " | " + r.Detail + " |\n")
	}
	b.WriteString("\n## Self-report — does each loop surface its own status?\n\n| ok | criterion | detail |\n|---|---|---|\n")
	for _, r := range p.SelfReport {
		b.WriteString("| " + passMark(r.Passed) + " | " + r.Label + " | " + r.Detail + " |\n")
	}
	b.WriteString("\n## Dogfood — does the loop run THROUGH fak's tooling?\n\n| ok | criterion | detail |\n|---|---|---|\n")
	for _, r := range p.Dogfood {
		b.WriteString("| " + passMark(r.Passed) + " | " + r.Label + " | " + r.Detail + " |\n")
	}
	b.WriteString("\n## Run it\n\n```bash\ngo run ./cmd/fak loop-score             # score this box's background loops\ngo run ./cmd/fak loop-score --markdown  # regenerate this doc\ngo run ./cmd/fak loop-score --json      # control-pane payload (corpus.loopscore_debt)\ngo test ./internal/loopscore/...        # prove the fold over a fragile vs durable corpus\n```\n\n")
	b.WriteString("## The 3× program — make the loops durable, observable, and fak-native\n\n")
	b.WriteString("The debt is concentrated in one structural gap: the loops that actually fire are driven by **external schedulers + ad-hoc logging**, not by `fak loop run`. So they are unregistered (a reboot re-launches nobody), often dark (the registry's own jobs have never been observed ticking), and unguarded (no `guard_enabled=1` run). A 3× is NOT hand-appending events during the measurement window (that is the data-gaming pattern every fak scorecard refuses) — it is making the durability + observability a **byproduct of how the loop is driven**, so the score rises structurally:\n\n")
	b.WriteString("1. **Register every firing loop.** Add the issue-dispatch / resolve-progress / smoke loops to `tools/loop-registry.json` with a cadence (`fak loop` registry). A registered job re-arms at boot — that is the auto-restart on system restart. Then `fak cron emit --target launchd|systemd|taskscheduler` projects it to a real OS unit that survives the reboot.\n")
	b.WriteString("2. **Drive them through `fak loop run`.** Replacing the raw scheduler call with `fak loop run --loop ID --source cron -- <cmd>` records fire/admit/start/**end** around every run under `fak guard` (guard_enabled=1) and posts a witnessed dispatch-result card — closing the self-report and dogfood gaps in one move.\n")
	b.WriteString("3. **Let them witness.** A run that ends with a `fak loop append --kind witness --status witnessed_done` (the resolve-progress loop already does this once) makes the keep-rate real, so the loop dogfoods the witness contract instead of trusting its own exit code.\n\n")
	b.WriteString("Re-run after a loop session and `--compare` against a pinned `--json` baseline: the verdict reports the multiple on the composite (the lever), so a real 3× (composite ~33 → ~99, debt → 0) is provable, not asserted.\n\n")
	b.WriteString("**Next:** " + p.NextAction + "\n")
	return b.String()
}

func Compare(current ScorecardPayload, baseline map[string]any) string {
	bc, _ := baseline["corpus"].(map[string]any)
	if bc == nil {
		bc = baseline
	}
	bDebt := anyInt(bc["loopscore_debt"])
	cDebt := anyInt(current.Corpus["loopscore_debt"])
	bScore := anyInt(bc["score"])
	cScore := anyInt(current.Corpus["score"])
	lines := []string{
		"loop-score compare:",
		"  loopscore_debt: " + itoa(bDebt) + " -> " + itoa(cDebt) + "  (retired " + itoa(bDebt-cDebt) + ")",
		"  value: " + anyStr(bc["value"]) + " -> " + anyStr(current.Corpus["value"]) +
			"  legacy score " + itoa(bScore) + " -> " + itoa(cScore) +
			"  grade " + anyStr(bc["grade"]) + " -> " + anyStr(current.Corpus["grade"]),
	}
	// The loop program drives the composite up; report the multiple on the composite
	// (the lever) as well as the debt.
	switch {
	case bDebt > 0 && cDebt == 0:
		lines = append(lines, "  VERDICT: all loopscore debt retired")
	case bScore > 0 && cScore >= 3*bScore:
		lines = append(lines, "  VERDICT: >=3x composite lift ("+itoa(bScore)+" -> "+itoa(cScore)+")")
	case bScore > 0 && cScore >= 2*bScore:
		lines = append(lines, "  VERDICT: >=2x composite lift ("+itoa(bScore)+" -> "+itoa(cScore)+")")
	case cScore > bScore || cDebt < bDebt:
		lines = append(lines, "  VERDICT: improved ("+itoa(bDebt)+" -> "+itoa(cDebt)+" debt, composite "+itoa(bScore)+" -> "+itoa(cScore)+")")
	default:
		lines = append(lines, "  VERDICT: no improvement")
	}
	return strings.Join(lines, "\n")
}

// ---- small helpers (mirror conceptusage/dogfoodscore idiom) -----------------------

func axisScore(rows []KPIResult) int {
	total, got := 0, 0
	for _, r := range rows {
		total += r.Weight
		if r.Passed {
			got += r.Weight
		}
	}
	if total == 0 {
		return 0
	}
	return int(math.Round(100 * float64(got) / float64(total)))
}

func GradeLetter(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

// kpiPayloads renders the local KPIPayload JSON shape from the same per-row
// hard/soft judgment toKPI uses for the shared kernel, so the two can never drift.
func kpiPayloads(rows []KPIResult) []KPIPayload {
	out := make([]KPIPayload, 0, len(rows))
	for _, r := range rows {
		sk := toKPI(r)
		out = append(out, KPIPayload{KPI: sk.Key, Group: sk.Group, Score: int(sk.Score), Value: scorecard.Round3(scorecard.ValueFromScore(sk.Score)), Detail: sk.Detail, Defects: sk.Defects, Soft: sk.Soft})
	}
	return out
}

func result(key, axis string, hard bool, weight int, label string, passed bool, detail string) KPIResult {
	return KPIResult{Key: key, Label: label, Hard: hard, Weight: weight, Axis: axis, Passed: passed, Detail: detail}
}

func passMark(ok bool) string {
	if ok {
		return "yes"
	}
	return "no"
}

func boolStr(ok bool, yes, no string) string {
	if ok {
		return yes
	}
	return no
}

func itoa(n int) string { return strconv.Itoa(n) }

func valueStr(score int) string {
	return anyStr(scorecard.Round3(scorecard.ValueFromScore(float64(score))))
}

func anyStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case int:
		return itoa(x)
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	default:
		return itoa(anyInt(v))
	}
}

func anyInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}
