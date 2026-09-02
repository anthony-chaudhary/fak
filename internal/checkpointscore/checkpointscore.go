// Package checkpointscore is fak's deterministic WIP-checkpoint readiness scorecard.
//
// The law: a long-running process that does real work must be able to survive a mid-task
// crash -- it persists durable, resumable work-in-progress state (crash_recovery) and it
// exposes a witnessed status surface a peer can read without tailing its logs (status). A
// process that runs unbounded with neither is a silent liability: when it dies, its WIP
// evaporates and no one can tell it stopped.
//
// This card is the static, tree-reading complement to internal/loopscore. loopscore grades
// the *runtime* loop ledger (.fak/loops.jsonl) -- reboot survival and per-run fire/end
// outcomes. checkpointscore instead grades the *source* of the repo's long-running process
// subsystems for whether they IMPLEMENT the two affordances, by probing each subsystem's own
// package for the concrete signature of its durable store / status fold. Because every KPI is
// re-derived from the tree (not from a data file an operator can edit), the score cannot be
// gamed by editing a manifest: the affordance token must exist in real, non-test source.
//
// The roster below is grounded in the repo's actual recovery/status map (the six unentangled
// subsystems -- shadowgit, resume, loopmgr, watchdoghealth, sessionjournal, plus the
// gap-catching layer looprecover/treedoctor). Planned gaps name subsystems that SHOULD exist
// but do not yet (the unified worker-state checkpoint tracked by #2394/#3784); each is debt
// until its package lands with a real durable store.
package checkpointscore

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/walkfiles"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema tag for this card's payload.
const Schema = "fak-checkpoint-scorecard/1"

// DebtKey is the corpus key holding the folded, unbounded checkpoint-debt integer:
// the number of (subsystem, axis) gaps plus each still-open planned subsystem.
const DebtKey = "checkpoint_debt"

// Subsystem is one long-running process subsystem in the roster. RecoveryTokens and
// StatusTokens are the concrete, grounded signatures of the subsystem's durable-resume store
// and its witnessed-status fold; Scan reports the axis as satisfied when ANY of a set's tokens
// appears in the package's non-test .go source.
type Subsystem struct {
	Name           string
	Dir            string // repo-relative package dir, e.g. "internal/resume"
	Role           string // one line: what long-running work it does
	RecoveryTokens []string
	StatusTokens   []string
}

// PlannedGap names a process subsystem the repo is known to still lack -- the unified
// worker-state checkpoint that shadowgit's #2394 reference and epic #3784/C7 describe but that
// no package implements yet. It is debt until Dir lands with a real durable store.
type PlannedGap struct {
	Name string
	Dir  string // where it would live; open while this holds no real durable-store source
	Role string
	Why  string
}

// roster is the grounded set of the repo's long-running process subsystems. Tokens are the
// real affordance signatures each subsystem uses today (or, where a subsystem legitimately
// lacks an axis, a token it does not carry -- surfacing a defensible gap).
var roster = []Subsystem{
	{
		Name: "resume", Dir: "internal/resume",
		Role:           "resumes dead/stopped agent sessions across accounts via the watchdog plan",
		RecoveryTokens: []string{"resume_drivestate", "resume_plan", "DecideWatchdogRow", "drivestate"},
		StatusTokens:   []string{"watchdog_status", "WatchdogStatus", "FoldStatus"},
	},
	{
		Name: "loopmgr", Dir: "internal/loopmgr",
		Role:           "the durable, hash-chained long-running-loop run ledger (.fak/loops.jsonl)",
		RecoveryTokens: []string{"loops.jsonl", "prev_hash", "LoadPrefix", "Summarize"},
		StatusTokens:   []string{"Summarize", "heartbeat", "witnessed_done"},
	},
	{
		Name: "sessionjournal", Dir: "internal/sessionjournal",
		Role:           "the boot-epoch crash journal that decides LIVE/CRASHED/STALE/CLOSED",
		RecoveryTokens: []string{"session-journal", "BootID", "boot"},
		StatusTokens:   []string{"Classify", "FoldEvents", "CRASHED"},
	},
	{
		Name: "watchdoghealth", Dir: "internal/watchdoghealth",
		Role:           "default-monitor health, autoheal, and human-vs-fleet triage",
		RecoveryTokens: []string{"heal", "HealState", "autoheal"},
		StatusTokens:   []string{"HEALTHY", "GAVE_UP", "PartitionAttention"},
	},
	{
		Name: "looprecover", Dir: "internal/looprecover",
		Role:           "the cross-run recovery worklist over the loop ledger (orphan/unwitnessed)",
		RecoveryTokens: []string{"Probe", "DefaultStaleSeconds", "orphaned"},
		StatusTokens:   []string{"Classify", "unwitnessed", "Summarize"},
	},
	{
		Name: "treedoctor", Dir: "internal/treedoctor",
		Role:           "the untracked-WIP land-or-park inventory (build-poison / abandonment)",
		RecoveryTokens: []string{"wipaudit", "abandoned", "AbandonAfter"},
		StatusTokens:   []string{"Classify", "poison", "resident"},
	},
	{
		Name: "shadowgit", Dir: "internal/shadowgit",
		Role:           "the per-step write ledger binding step -> bytes-changed",
		RecoveryTokens: []string{"state_changelog", "Snapshot", "WriteChangelog"},
		// shadowgit is a witness/attribution layer with no status/summary fold today;
		// StatusTokens name the surface it is expected to grow (a defensible gap until then).
		StatusTokens: []string{"FoldStatus", "Summarize", "StatusReport"},
	},
	{
		Name: "doomloop", Dir: "internal/doomloop",
		Role:           "runaway-loop detection for long-running processes",
		RecoveryTokens: []string{"ledger", "jsonl", "persist", "Resume"},
		StatusTokens:   []string{"Classify", "Status", "Report"},
	},
}

// planned is the known-missing subsystem set: the unified worker-state checkpoint/restore that
// composes shadowgit's write-ledger, the loop ledger, and the session journal into a single
// "resume this worker exactly where it crashed" replay. No package builds it yet.
var planned = []PlannedGap{
	{
		Name: "unified-worker-checkpoint", Dir: "internal/wipcheckpoint",
		Role: "a unified worker-state checkpoint/restore that replays a crashed agent's in-flight work",
		Why:  "shadowgit records what changed but cannot resume; the #2394 checkpoint / epic #3784 unification is referenced but unbuilt, so a mid-task crash still loses in-flight WIP.",
	},
}

// Scanned is the per-subsystem probe result.
type Scanned struct {
	Subsystem
	Present     bool
	HasRecovery bool
	HasStatus   bool
	RecoveryHit string // the token that satisfied crash_recovery, "" if none
	StatusHit   string // the token that satisfied status, "" if none
}

// Gap is one dispatchable checkpoint-debt unit: a present subsystem missing an axis, or a
// planned subsystem not yet built. One Gap maps to one KPI defect and one backlog ticket.
type Gap struct {
	Subsystem string
	Dir       string
	Axis      string // "crash_recovery" | "status" | "planned"
	Role      string
	Detail    string
}

// pkgSource concatenates the non-test .go source of dir (under root). Missing dir -> ("", false).
func pkgSource(root, dir string) (string, bool) {
	full := filepath.Join(root, filepath.FromSlash(dir))
	info, err := os.Stat(full)
	if err != nil || !info.IsDir() {
		return "", false
	}
	var b strings.Builder
	found := false
	_ = walkfiles.Files(full, func(path string, d os.DirEntry) error {
		name := d.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		found = true
		b.Write(data)
		b.WriteByte('\n')
		return nil
	})
	return b.String(), found
}

// firstHit returns the first token present in src, or "".
func firstHit(src string, tokens []string) string {
	for _, t := range tokens {
		if t != "" && strings.Contains(src, t) {
			return t
		}
	}
	return ""
}

// Scan probes every rostered subsystem's source under root and returns the results in roster
// order. It is a pure tree read: same tree in, same results out.
func Scan(root string) []Scanned {
	out := make([]Scanned, 0, len(roster))
	for _, s := range roster {
		src, present := pkgSource(root, s.Dir)
		rec := firstHit(src, s.RecoveryTokens)
		st := firstHit(src, s.StatusTokens)
		out = append(out, Scanned{
			Subsystem:   s,
			Present:     present,
			HasRecovery: present && rec != "",
			HasStatus:   present && st != "",
			RecoveryHit: rec,
			StatusHit:   st,
		})
	}
	return out
}

// plannedOpen reports whether a planned subsystem is still an open gap: its dir is absent, or
// present but carries no durable-store signature (a stub that has not really landed).
func plannedOpen(root string, p PlannedGap) bool {
	src, present := pkgSource(root, p.Dir)
	if !present {
		return true
	}
	// Graduated only when it carries a real durable store (a journal/ledger write).
	return firstHit(src, []string{"jsonl", "journal", "ledger", "Checkpoint", "Restore"}) == ""
}

// Gaps returns every checkpoint-debt unit in a stable order: each present subsystem missing
// crash_recovery, then each missing status, then each still-open planned subsystem.
func Gaps(root string) []Gap {
	scanned := Scan(root)
	var recov, stat []Gap
	for _, s := range scanned {
		if !s.HasRecovery {
			recov = append(recov, Gap{
				Subsystem: s.Name, Dir: s.Dir, Axis: "crash_recovery", Role: s.Role,
				Detail: recoveryDetail(s),
			})
		}
		if !s.HasStatus {
			stat = append(stat, Gap{
				Subsystem: s.Name, Dir: s.Dir, Axis: "status", Role: s.Role,
				Detail: statusDetail(s),
			})
		}
	}
	var plan []Gap
	for _, p := range planned {
		if plannedOpen(root, p) {
			plan = append(plan, Gap{
				Subsystem: p.Name, Dir: p.Dir, Axis: "planned", Role: p.Role,
				Detail: p.Why,
			})
		}
	}
	sort.SliceStable(recov, func(i, j int) bool { return recov[i].Subsystem < recov[j].Subsystem })
	sort.SliceStable(stat, func(i, j int) bool { return stat[i].Subsystem < stat[j].Subsystem })
	out := append(recov, stat...)
	return append(out, plan...)
}

func recoveryDetail(s Scanned) string {
	return missingSurfaceDetail(s, "durable resumable-state signature", s.RecoveryTokens)
}

func statusDetail(s Scanned) string {
	return missingSurfaceDetail(s, "witnessed-status surface", s.StatusTokens)
}

// missingSurfaceDetail renders the "package absent" / "no <surface> found, expected
// one of ..." explanation the crash_recovery and status KPIs both emit. `surface` and
// `tokens` carry each caller's own wording and token list, so recoveryDetail's and
// statusDetail's strings stay exactly what they were.
func missingSurfaceDetail(s Scanned, surface string, tokens []string) string {
	if !s.Present {
		return "subsystem package " + s.Dir + " is absent from the tree"
	}
	return "no " + surface + " found in " + s.Dir +
		" (expected one of: " + strings.Join(tokens, ", ") + ")"
}

// Build folds the scan into the shared control-pane Payload. Three KPIs -- crash_recovery,
// status, and the planned unified checkpoint -- fold to checkpoint_debt = Σ defects.
func Build(root string) scorecard.Payload {
	scanned := Scan(root)

	var recovDefects, statusDefects, plannedDefects []string
	recovered, statused := 0, 0
	for _, s := range scanned {
		if s.HasRecovery {
			recovered++
		} else {
			recovDefects = append(recovDefects, s.Name+": "+recoveryDetail(s))
		}
		if s.HasStatus {
			statused++
		} else {
			statusDefects = append(statusDefects, s.Name+": "+statusDetail(s))
		}
	}
	openPlanned := 0
	for _, p := range planned {
		if plannedOpen(root, p) {
			openPlanned++
			plannedDefects = append(plannedDefects, p.Name+": "+p.Why)
		}
	}
	sort.Strings(recovDefects)
	sort.Strings(statusDefects)
	sort.Strings(plannedDefects)

	total := len(roster)
	kpis := []scorecard.KPI{
		{
			Key:     "crash_recovery",
			Group:   "crash_recovery",
			Score:   sharePct(recovered, total),
			Detail:  "process subsystems that persist durable, resumable WIP state",
			Defects: recovDefects,
		},
		{
			Key:     "status",
			Group:   "status",
			Score:   sharePct(statused, total),
			Detail:  "process subsystems that expose a witnessed status surface",
			Defects: statusDefects,
		},
		{
			Key:     "unified_checkpoint",
			Group:   "unified_checkpoint",
			Score:   boolPct(openPlanned == 0),
			Detail:  "the unified worker-state checkpoint/restore that resumes a crashed agent's in-flight work",
			Defects: plannedDefects,
		},
	}

	debt := len(recovDefects) + len(statusDefects) + len(plannedDefects)
	return scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Finding: "checkpoint-debt: long-running process subsystems missing crash-recovery or " +
			"witnessed status, plus the unbuilt unified worker checkpoint",
		FindingClean: "every rostered process subsystem persists resumable WIP state and exposes " +
			"witnessed status",
		NextAction: "run `fak checkpoint-debt-dispatch` to fan each gap out to a sink (stdout / " +
			"local-db ledger / GitHub issue)",
		NextActionClean: "no action -- re-run after adding or changing a long-running process",
		ExtraCorpus: map[string]any{
			"subsystems":     total,
			"recovery_ready": recovered,
			"status_ready":   statused,
			"planned_open":   openPlanned,
			"gaps":           debt,
		},
	})
}

func sharePct(n, total int) float64 {
	if total <= 0 {
		return 100
	}
	return 100 * float64(n) / float64(total)
}

func boolPct(ok bool) float64 {
	if ok {
		return 100
	}
	return 0
}
