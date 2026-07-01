// Package conceptusage scores the OVERALL dogfooding of fak's own concepts while
// fak itself is being developed by an agent fleet — the question "when we build fak,
// how much does that development route through fak's own primitives, versus generic
// agentic dev (raw git, unverified self-reports, no lane arbitration)?"
//
// It is the breadth-and-depth sibling of internal/dogfoodscore, which scores ONE
// narrow loop (a launched session's honesty over a Stop-hook error). This scorecard
// is wider: it folds the concrete, on-disk traces fak's own tooling leaves behind
// during real development into a single conceptusage_debt + an A–F grade, across two
// axes:
//
//   - USAGE breadth — does the development *output* carry the fak discipline? Measured
//     from `git log`: the (fak <leaf>) ship-stamp, the DCO sign-off, a conventional
//     type, and a verb the witness contract BINDS (add/fix/implement…), plus whether
//     concurrent dev arbitrated disjoint lanes (the lane-journal ACQUIRE/RELEASE rows).
//   - WITNESS depth — does the development *trust evidence over self-report*? Measured
//     from `.dos/verdict-journal.jsonl`: the share of decisions made via the verify /
//     improve syscalls (proactive, evidence-grounded) rather than passive recall, and
//     whether recalled memory was actually re-verified (RECALL_FRESH/STALE) instead of
//     left RECALL_UNVERIFIABLE.
//
// Every number is re-derived from disk (git + the journals fak's tools wrote). The
// score cannot be moved by editing a JSON file — only by actually using the concepts
// more while developing. That is the point: driving this debt down IS dogfooding more.
package conceptusage

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	Schema = "fak-conceptusage-scorecard/1"
	// DefaultCommitWindow is how many recent commits define "agentic dev now". Wide
	// enough to be stable across a single session, narrow enough that a long-ago lapse
	// does not keep the score red forever.
	DefaultCommitWindow = 200
	// Axis weights: usage breadth is the output discipline (well-dogfooded already),
	// witness depth is the thin concept the 3x program must grow. Weight depth higher
	// so the composite tracks the lever that actually moves.
	usageAxisWeight   = 0.4
	witnessAxisWeight = 0.6
	// witnessWindow scopes the witness-SHARE to the most-recent decision rows so the
	// metric measures whether the CURRENT dev loop witnesses, not whether all of repo
	// history did. Wide enough to span a working session, narrow enough that a recent
	// burst of real witnessing actually moves the number.
	witnessWindow = 50
)

// decClass labels a recent journal row by what KIND of dev decision it represents,
// for the windowed witness-share. decNoise (a passive RECALL_UNVERIFIABLE the kernel
// could not check) is excluded from the share denominator: the dev made no
// trust-vs-witness decision there, so counting it as a failed witness is wrong.
type decClass uint8

const (
	decNoise          decClass = iota // passive RECALL_UNVERIFIABLE — kernel background, not a decision
	decResolvedRecall                 // a recall the dev actually re-checked (FRESH/STALE/REVERT)
	decProactive                      // a proactive verify/improve witness syscall
)

// stampRe / convRe / bindRe mirror the commit contract the witness referee grades
// (AGENTS.md): a ship commit ends with a `(fak <leaf>)` trailer, uses a Conventional
// type, and leads after `type(scope):` with a verb that BINDS a witnessable effect.
var (
	stampRe = regexp.MustCompile(`\(fak [a-z0-9][a-z0-9-]*\)\s*$`)
	convRe  = regexp.MustCompile(`(?i)^(feat|fix|docs|test|refactor|perf|chore|build|ci|style|add)(\([^)]*\))?!?:\s`)
	bindRe  = regexp.MustCompile(`(?i)^\w+(\([^)]*\))?!?:\s*(add|fix|implement|wire|port|extract|split|map|gate|fail|kill|pin|hoist|declare|resolve|exclude|treat|close|fold|route|enforce|attest|witness|cancel)\b`)
)

// KPIResult is one graded criterion. Identical shape to dogfoodscore.KPIResult so the
// two scorecards render and fold the same way.
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

// Evidence is the raw, re-derived-from-disk corpus the KPIs read. Exported so a caller
// (the verb, a test) can inspect exactly what was counted.
type Evidence struct {
	Commits         int  `json:"commits"`
	Stamped         int  `json:"stamped"`
	Signed          int  `json:"signed"`
	Conventional    int  `json:"conventional"`
	BindingVerb     int  `json:"binding_verb"`
	VerdictRows     int  `json:"verdict_rows"`
	VerifySyscalls  int  `json:"verify_syscalls"`
	ImproveCalls    int  `json:"improve_syscalls"`
	RecallRows      int  `json:"recall_rows"`
	RecallResolved  int  `json:"recall_resolved"`  // RECALL_FRESH + RECALL_STALE + RECALL_REVERT (actually re-checked)
	WindowProactive int  `json:"window_proactive"` // verify+improve in the recent witnessWindow
	WindowRecall    int  `json:"window_recall"`    // RESOLVED recalls (re-checked) in the recent window
	WindowNoise     int  `json:"window_noise"`     // passive RECALL_UNVERIFIABLE in the window (excluded from the share)
	LaneRows        int  `json:"lane_rows"`
	LaneAcquires    int  `json:"lane_acquires"`
	DistinctLanes   int  `json:"distinct_lanes"`
	JournalPresent  bool `json:"journal_present"`

	// window is a scratch ring of decision classes, folded into the Window* counts
	// after the scan. Not serialized.
	window []decClass `json:"-"`
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
	Usage      []KPIResult    `json:"usage"`
	Witness    []KPIResult    `json:"witness"`
	Evidence   Evidence       `json:"evidence"`
}

// Options pins the clock and root so the score is deterministic for tests.
type Options struct {
	Root         string
	Now          time.Time
	CommitWindow int
	// gitLog overrides the git-log read for tests; nil means shell out to git.
	gitLog func(root string, window int) []commit
}

func (o Options) normalize() Options {
	if o.Root == "" {
		o.Root = "."
	}
	if o.Now.IsZero() {
		o.Now = time.Now().UTC()
	}
	if o.CommitWindow <= 0 {
		o.CommitWindow = DefaultCommitWindow
	}
	return o
}

type commit struct {
	subject string
	body    string
}

// ---- evidence gathering (the impure shell, kept thin) -----------------------------

// gitCommits reads recent commits via git, splitting subject/body on a record
// separator so multi-line bodies survive. Returns nil if git is unavailable, which
// degrades the usage axis to "no evidence" rather than a false pass.
func gitCommits(root string, window int) []commit {
	cmd := exec.Command("git", "-C", root, "log",
		"--format=%s%x1f%b%x1e", "-n", strconv.Itoa(window))
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var cs []commit
	for _, rec := range strings.Split(string(out), "\x1e") {
		rec = strings.TrimLeft(rec, "\n")
		if strings.TrimSpace(rec) == "" {
			continue
		}
		parts := strings.SplitN(rec, "\x1f", 2)
		c := commit{subject: strings.TrimSpace(parts[0])}
		if len(parts) > 1 {
			c.body = parts[1]
		}
		cs = append(cs, c)
	}
	return cs
}

func gatherEvidence(opts Options) Evidence {
	root, _ := filepath.Abs(opts.Root)
	if root == "" {
		root = opts.Root
	}
	var ev Evidence

	// --- USAGE: commit discipline ---
	logFn := opts.gitLog
	if logFn == nil {
		logFn = gitCommits
	}
	for _, c := range logFn(root, opts.CommitWindow) {
		ev.Commits++
		if stampRe.MatchString(c.subject) {
			ev.Stamped++
		}
		if convRe.MatchString(c.subject) {
			ev.Conventional++
		}
		if bindRe.MatchString(c.subject) {
			ev.BindingVerb++
		}
		if strings.Contains(c.body, "Signed-off-by:") {
			ev.Signed++
		}
	}

	// --- WITNESS: verdict journal (verify/improve/recall) ---
	scanVerdictJournal(filepath.Join(root, ".dos", "verdict-journal.jsonl"), &ev)

	// --- USAGE: lane arbitration journal ---
	scanLaneJournal(filepath.Join(root, ".dos", "lane-journal.jsonl"), &ev)

	return ev
}

// scanVerdictJournal folds the decision journal fak's verify/recall/improve syscalls
// write. Tolerates a malformed tail line (a journal being appended concurrently).
func scanVerdictJournal(path string, ev *Evidence) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	ev.JournalPresent = true
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row struct {
			Syscall string `json:"syscall"`
			Verdict string `json:"verdict"`
		}
		if json.Unmarshal([]byte(line), &row) != nil {
			continue
		}
		ev.VerdictRows++
		switch row.Syscall {
		case "verify":
			ev.VerifySyscalls++
		case "improve":
			ev.ImproveCalls++
		case "memory_recall":
			ev.RecallRows++
			switch row.Verdict {
			case "RECALL_FRESH", "RECALL_STALE", "RECALL_REVERT", "KEEP", "REVERT":
				ev.RecallResolved++
			}
		}
		// Window the decision-class counts to the most-recent rows so the witness
		// SHARE reflects whether the CURRENT dev loop witnesses — not whether all of
		// repo history did. Three classes: a proactive witness (verify/improve), a
		// recall the dev actually RE-CHECKED (resolved to FRESH/STALE/REVERT — a real
		// trust-vs-witness decision), and passive RECALL_UNVERIFIABLE NOISE — a memory
		// the kernel injected but could not check, where the dev made NO decision at
		// all. Counting that noise as a failed witness conflates kernel background
		// chatter with dev behavior, which is exactly what pinned the old ratio near
		// zero. We keep a ring of decision classes and fold the window after the scan.
		switch row.Syscall {
		case "verify", "improve":
			ev.window = append(ev.window, decProactive)
		case "memory_recall":
			switch row.Verdict {
			case "RECALL_FRESH", "RECALL_STALE", "RECALL_REVERT", "KEEP", "REVERT":
				ev.window = append(ev.window, decResolvedRecall)
			default:
				ev.window = append(ev.window, decNoise)
			}
		}
	}
	// Fold the recency window (tail of the ring) into the windowed counts. Passive
	// noise is excluded from the denominator — it is not a dev decision.
	w := ev.window
	if len(w) > witnessWindow {
		w = w[len(w)-witnessWindow:]
	}
	for _, d := range w {
		switch d {
		case decProactive:
			ev.WindowProactive++
		case decResolvedRecall:
			ev.WindowRecall++
		case decNoise:
			ev.WindowNoise++
		}
	}
}

// scanLaneJournal folds the lane-lease journal dos_arbitrate writes. ACQUIRE/RELEASE
// are the active arbitration acts; ENFORCE rows are the passive per-call gate and are
// not counted as proactive usage.
func scanLaneJournal(path string, ev *Evidence) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	lanes := map[string]struct{}{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row struct {
			Op   string `json:"op"`
			Lane string `json:"lane"`
		}
		if json.Unmarshal([]byte(line), &row) != nil {
			continue
		}
		ev.LaneRows++
		if row.Op == "ACQUIRE" {
			ev.LaneAcquires++
		}
		if row.Lane != "" {
			lanes[row.Lane] = struct{}{}
		}
	}
	ev.DistinctLanes = len(lanes)
}

// ---- KPI definitions --------------------------------------------------------------

func pct(num, den int) int {
	if den <= 0 {
		return 0
	}
	return int(math.Round(100 * float64(num) / float64(den)))
}

// usageResults grade the development OUTPUT discipline. These are mostly green in a
// well-run fak tree — the point is to keep them green, and to red if dev ever ships
// without the stamp/sign-off/binding-verb/lane discipline.
func usageResults(ev Evidence) []KPIResult {
	const axis = "usage"
	stampPct := pct(ev.Stamped, ev.Commits)
	signPct := pct(ev.Signed, ev.Commits)
	convPct := pct(ev.Conventional, ev.Commits)
	bindPct := pct(ev.BindingVerb, ev.Commits)
	return []KPIResult{
		result("ship_stamp", axis, true, 3,
			"recent commits carry the (fak <leaf>) ship-stamp the dos verify referee binds",
			ev.Commits > 0 && stampPct >= 90,
			itoa(ev.Stamped)+"/"+itoa(ev.Commits)+" ("+itoa(stampPct)+"%) carry the (fak <leaf>) trailer"),
		result("dco_signoff", axis, true, 2,
			"recent commits are DCO signed-off (git commit -s)",
			ev.Commits > 0 && signPct >= 90,
			itoa(ev.Signed)+"/"+itoa(ev.Commits)+" ("+itoa(signPct)+"%) signed-off"),
		result("conventional_type", axis, true, 2,
			"recent commits use a Conventional-Commits type",
			ev.Commits > 0 && convPct >= 90,
			itoa(ev.Conventional)+"/"+itoa(ev.Commits)+" ("+itoa(convPct)+"%) conventional"),
		result("binding_verb", axis, false, 2,
			"recent commit subjects lead with a verb the witness BINDS (not surface/print)",
			ev.Commits > 0 && bindPct >= 70,
			itoa(ev.BindingVerb)+"/"+itoa(ev.Commits)+" ("+itoa(bindPct)+"%) lead with a binding verb"),
		result("lane_arbitration", axis, false, 1,
			"concurrent dev arbitrated disjoint lanes (dos_arbitrate ACQUIRE/RELEASE rows)",
			ev.LaneAcquires > 0,
			itoa(ev.LaneAcquires)+" lane ACQUIRE(s) across "+itoa(ev.DistinctLanes)+" distinct lane(s)"),
	}
}

// witnessResults grade whether development TRUSTS EVIDENCE OVER SELF-REPORT. This is
// the thin axis the 3x program grows: the share of decisions made via the proactive
// verify/improve syscalls, and whether recalled memory was actually re-verified.
func witnessResults(ev Evidence) []KPIResult {
	const axis = "witness"
	// Proactive-witness share, measured over the RECENT window so the number reflects
	// whether the CURRENT dev loop witnesses — not whether all of repo history did.
	// Passive memory_recall rows accrue every session while proactive witnessing is
	// rare; an all-time ratio is structurally pinned low and never registers a recent
	// burst of genuine witnessing. The window fixes that without becoming gameable: the
	// only way to raise it is to actually witness more in the recent decisions.
	proactive := ev.VerifySyscalls + ev.ImproveCalls
	windowPoints := ev.WindowProactive + ev.WindowRecall
	witnessShare := pct(ev.WindowProactive, windowPoints)
	recallFreshPct := pct(ev.RecallResolved, ev.RecallRows)
	return []KPIResult{
		result("verify_syscall_used", axis, true, 3,
			"development proactively witnessed claims via the verify/improve syscall",
			proactive > 0,
			itoa(ev.VerifySyscalls)+" verify + "+itoa(ev.ImproveCalls)+" improve syscall(s) in the journal"),
		result("witness_share", axis, true, 3,
			"a healthy share of RECENT dev decisions are evidence-grounded (verify/improve), not recall-only",
			windowPoints > 0 && witnessShare >= 15,
			itoa(witnessShare)+"% of the last "+itoa(windowPoints)+" dev decision(s) used a proactive witness syscall (target >=15%; "+itoa(ev.WindowNoise)+" passive UNVERIFIABLE auto-recalls excluded as non-decisions)"),
		result("recall_reverified", axis, false, 2,
			"recalled memory was re-verified against ground truth, not left UNVERIFIABLE",
			ev.RecallRows == 0 || recallFreshPct >= 40,
			itoa(ev.RecallResolved)+"/"+itoa(ev.RecallRows)+" ("+itoa(recallFreshPct)+"%) recalls resolved to a checked verdict"),
		result("journal_present", axis, true, 1,
			"the verdict journal exists — development actually ran the witnessing syscalls",
			ev.JournalPresent && ev.VerdictRows > 0,
			itoa(ev.VerdictRows)+" verdict-journal row(s)"),
	}
}

// ---- fold -------------------------------------------------------------------------

// axisKPIs converts this card's KPIResult rows into shared-kernel scorecard.KPI rows for
// one axis, scaling each KPI's own int Weight by axisWeight/axisTotalWeight so that
// scorecard.Fold's overall weighted mean (Sigma(w*score)/Sigma(w) across usage+witness
// together) reproduces exactly usageAxisWeight*uScore + witnessAxisWeight*wScore -- the
// same two-axis composite this card has always reported, now computed by the shared
// fold instead of a bespoke axisScore. A HARD fail becomes exactly one Defect (so
// Fold's debt = Sigma(len(Defects)) equals the legacy hard-fail count); a SOFT fail
// becomes exactly one Soft entry, which Fold never counts as debt.
func axisKPIs(rows []KPIResult, axisWeight float64, weights map[string]float64) []scorecard.KPI {
	total := 0
	for _, r := range rows {
		total += r.Weight
	}
	out := make([]scorecard.KPI, 0, len(rows))
	for _, r := range rows {
		score := 0.0
		if r.Passed {
			score = 100.0
		}
		k := scorecard.KPI{Key: r.Key, Group: r.Axis, Score: score, Detail: r.Detail}
		if !r.Passed {
			if r.Hard {
				k.Defects = []string{r.Key + ": " + r.Detail}
			} else {
				k.Soft = []string{r.Key + ": " + r.Detail}
			}
		}
		out = append(out, k)
		w := float64(r.Weight)
		if total > 0 {
			w = axisWeight * float64(r.Weight) / float64(total)
		}
		weights[r.Key] = w
	}
	return out
}

func Build(opts Options) ScorecardPayload {
	opts = opts.normalize()
	root, _ := filepath.Abs(opts.Root)
	if root == "" {
		root = opts.Root
	}
	ev := gatherEvidence(opts)

	usage := usageResults(ev)
	witness := witnessResults(ev)
	all := append(append([]KPIResult{}, usage...), witness...)

	// weights is keyed by KPI Key (each key is unique across both axes) with a value
	// scaled so scorecard.Fold's weighted mean reproduces the usage/witness axis blend.
	weights := map[string]float64{}
	kpis := append(axisKPIs(usage, usageAxisWeight, weights), axisKPIs(witness, witnessAxisWeight, weights)...)

	uScore := axisScore(usage)
	wScore := axisScore(witness)

	var hardFail []KPIResult
	for _, r := range all {
		if r.Hard && !r.Passed {
			hardFail = append(hardFail, r)
		}
	}
	finding, next := "concept_usage_healthy", "hold the line; re-run after a dev session — keep witnessing claims via the verify syscall, not self-report"
	findingClean, nextClean := finding, next
	if len(hardFail) > 0 {
		finding = "conceptusage_debt"
		lead := hardFail[0]
		next = "retire worst-first: " + lead.Key + " — " + lead.Detail
	}

	p := scorecard.Fold(Schema, kpis, "conceptusage_debt", weights, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    findingClean,
		NextAction:      next,
		NextActionClean: nextClean,
		ExtraCorpus: map[string]any{
			"usage_score":     uScore,
			"usage_value":     scorecard.Round3(scorecard.ValueFromScore(float64(uScore))),
			"witness_score":   wScore,
			"witness_value":   scorecard.Round3(scorecard.ValueFromScore(float64(wScore))),
			"commits_scanned": ev.Commits,
			"verify_syscalls": ev.VerifySyscalls,
			"recall_rows":     ev.RecallRows,
			"lane_acquires":   ev.LaneAcquires,
		},
	})
	debt := len(hardFail)
	grade := anyStr(p.Corpus["grade"])
	if p.OK {
		p.Reason = "concept-usage: usage value " + anyStr(p.Corpus["usage_value"]) + ", witness value " + anyStr(p.Corpus["witness_value"]) +
			", composite value " + anyStr(p.Corpus["value"]) + " (" + grade + ", legacy score " + anyStr(p.Corpus["score"]) + "); dev routes through the fak concepts; zero hard gaps"
	} else {
		keys := make([]string, len(hardFail))
		for i, r := range hardFail {
			keys[i] = r.Key
		}
		p.Reason = "concept-usage carries " + itoa(debt) + " debt (usage value " + anyStr(p.Corpus["usage_value"]) +
			", witness value " + anyStr(p.Corpus["witness_value"]) + ", composite value " + anyStr(p.Corpus["value"]) +
			" " + grade + ", legacy score " + anyStr(p.Corpus["score"]) + "): " +
			strings.Join(keys, ", ")
	}
	p.Workspace = root

	return ScorecardPayload{
		Schema:     p.Schema,
		OK:         p.OK,
		Verdict:    p.Verdict,
		Finding:    p.Finding,
		Reason:     p.Reason,
		NextAction: p.NextAction,
		Workspace:  p.Workspace,
		Corpus:     p.Corpus,
		KPIs:       kpiPayloads(all),
		Usage:      usage,
		Witness:    witness,
		Evidence:   ev,
	}
}

// ---- render -----------------------------------------------------------------------

func Render(p ScorecardPayload) string {
	c := p.Corpus
	lines := []string{
		"concept-usage — " + p.Verdict + " (" + p.Finding + ")",
		"  conceptusage_debt: " + anyStr(c["conceptusage_debt"]) + "   value " + anyStr(c["value"]) +
			" [" + anyStr(c["grade"]) + "]   (legacy score " + anyStr(c["score"]) + "; usage value " + anyStr(c["usage_value"]) + "; witness value " + anyStr(c["witness_value"]) + ")",
		"  evidence: " + anyStr(c["commits_scanned"]) + " commit(s); " + anyStr(c["verify_syscalls"]) +
			" verify syscall(s); " + anyStr(c["recall_rows"]) + " recall(s); " + anyStr(c["lane_acquires"]) + " lane acquire(s)",
		"",
		"  USAGE (does the dev OUTPUT carry the fak discipline?):",
	}
	for _, r := range p.Usage {
		lines = append(lines, "    "+passMark(r.Passed)+" "+r.Label+"  ["+r.Detail+"]")
	}
	lines = append(lines, "", "  WITNESS (does dev TRUST EVIDENCE over self-report?):")
	for _, r := range p.Witness {
		lines = append(lines, "    "+passMark(r.Passed)+" "+r.Label+"  ["+r.Detail+"]")
	}
	lines = append(lines, "", "  NEXT: "+p.NextAction)
	return strings.Join(lines, "\n")
}

func Markdown(p ScorecardPayload) string {
	c := p.Corpus
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString(`title: "fak concept-usage scorecard"` + "\n")
	b.WriteString(`description: "How much the agentic DEVELOPMENT of fak routes through fak's own concepts — the ship-stamp/DCO/binding-verb commit discipline and lane arbitration (usage breadth), and the verify/improve witness syscalls over passive recall (witness depth), all re-derived from git and the .dos journals."` + "\n")
	b.WriteString("---\n\n")
	b.WriteString("# fak concept-usage scorecard\n\n")
	b.WriteString("**conceptusage_debt: " + anyStr(c["conceptusage_debt"]) + "**; value **" + anyStr(c["value"]) +
		" (" + anyStr(c["grade"]) + ")**; legacy score " + anyStr(c["score"]) +
		"; usage value " + anyStr(c["usage_value"]) + "; witness value " +
		anyStr(c["witness_value"]) + "\n\n")
	b.WriteString("> " + p.Reason + "\n\n")
	b.WriteString("The question: when an agent builds fak, how much does that development route through fak's *own* concepts — committing with the witness contract (ship-stamp, DCO, a binding verb), arbitrating disjoint lanes, and **witnessing its own claims via the verify syscall instead of trusting a self-report** — versus generic agentic dev? Every number is re-derived from `git log` and the `.dos` journals fak's tooling wrote; the score moves only when development actually uses the concepts more.\n\n")
	b.WriteString("## Usage — does the development OUTPUT carry the fak discipline?\n\n| ok | criterion | detail |\n|---|---|---|\n")
	for _, r := range p.Usage {
		b.WriteString("| " + passMark(r.Passed) + " | " + r.Label + " | " + r.Detail + " |\n")
	}
	b.WriteString("\n## Witness — does development TRUST EVIDENCE over self-report?\n\n| ok | criterion | detail |\n|---|---|---|\n")
	for _, r := range p.Witness {
		b.WriteString("| " + passMark(r.Passed) + " | " + r.Label + " | " + r.Detail + " |\n")
	}
	b.WriteString("\n## Run it\n\n```bash\ngo run ./cmd/fak concept-usage-score            # score this tree's concept dogfooding\ngo run ./cmd/fak concept-usage-score --markdown # regenerate this doc\ngo test ./internal/conceptusage/...             # prove the fold over a thin vs healthy corpus\n```\n\n")
	b.WriteString("## The 3× program — grow the witness axis honestly\n\n")
	b.WriteString("The usage axis is already saturated (commit discipline + lane arbitration are fully dogfooded); the witness axis is the lever. It is thin because **witnessing is manual and rare** — `dos verify` / `dos improve --observe` rows accrue only when someone runs them by hand, while passive `memory_recall` rows dominate the journal. So a 3× is NOT firing verify calls by hand during the measurement window (that is the data-gaming pattern every fak scorecard refuses) — it is making the witness syscall a **byproduct of real work** so the share rises structurally across sessions:\n\n")
	b.WriteString("1. **Witness every ship.** Run `dos verify <PLAN> <PHASE>` (or `dos improve --observe`) at ship time, not `dos commit-audit` alone — commit-audit is read-only and writes no row; `verify`/`improve` are the syscalls this axis counts.\n")
	b.WriteString("2. **Re-verify recalled memory.** When a memory is recalled, re-check it against ground truth (`dos recall <name>`) so it resolves to FRESH/STALE instead of sitting at the 76%-UNVERIFIABLE floor.\n")
	b.WriteString("3. **Wire it into the dev loop.** The durable fix is a ship-path step (a post-commit / Stop-hook auto-`dos verify`) so the witness share climbs without anyone remembering to — the same way the usage axis is green because the commit hooks make the stamp/DCO automatic.\n\n")
	b.WriteString("Re-run after a dev session and `--compare` against a pinned `--json` baseline: the verdict reports the multiple on the witness score (the lever), so a real 3× (witness 6% → 18% share) is provable, not asserted.\n\n")
	b.WriteString("**Next:** " + p.NextAction + "\n")
	return b.String()
}

func Compare(current ScorecardPayload, baseline map[string]any) string {
	bc, _ := baseline["corpus"].(map[string]any)
	if bc == nil {
		bc = baseline
	}
	bDebt := anyInt(bc["conceptusage_debt"])
	cDebt := anyInt(current.Corpus["conceptusage_debt"])
	bWit := anyInt(bc["witness_score"])
	cWit := anyInt(current.Corpus["witness_score"])
	lines := []string{
		"concept-usage compare:",
		"  conceptusage_debt: " + itoa(bDebt) + " -> " + itoa(cDebt) + "  (retired " + itoa(bDebt-cDebt) + ")",
		"  value: " + anyStr(bc["value"]) + " -> " + anyStr(current.Corpus["value"]) +
			"  legacy score " + anyStr(bc["score"]) + " -> " + anyStr(current.Corpus["score"]) +
			"  grade " + anyStr(bc["grade"]) + " -> " + anyStr(current.Corpus["grade"]),
		"  witness_score: " + itoa(bWit) + " -> " + itoa(cWit),
	}
	// The 3x program drives the witness axis up; report the multiple on the witness
	// score (the lever) as well as the debt.
	switch {
	case bDebt > 0 && cDebt == 0:
		lines = append(lines, "  VERDICT: all concept-usage debt retired")
	case bWit > 0 && cWit >= 3*bWit:
		lines = append(lines, "  VERDICT: >=3x witness-axis lift ("+itoa(bWit)+" -> "+itoa(cWit)+")")
	case bWit > 0 && cWit >= 2*bWit:
		lines = append(lines, "  VERDICT: >=2x witness-axis lift ("+itoa(bWit)+" -> "+itoa(cWit)+")")
	case cWit > bWit || cDebt < bDebt:
		lines = append(lines, "  VERDICT: improved ("+itoa(bDebt)+" -> "+itoa(cDebt)+" debt, witness "+itoa(bWit)+" -> "+itoa(cWit)+")")
	default:
		lines = append(lines, "  VERDICT: no improvement")
	}
	return strings.Join(lines, "\n")
}

// ---- small helpers (mirror dogfoodscore idiom) ------------------------------------

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

func kpiPayloads(rows []KPIResult) []KPIPayload {
	out := make([]KPIPayload, 0, len(rows))
	for _, r := range rows {
		k := KPIPayload{KPI: r.Key, Group: r.Axis, Detail: r.Detail}
		if r.Passed {
			k.Score = 100
			k.Value = 1
		} else if r.Hard {
			k.Defects = []string{r.Key + ": " + r.Detail}
		} else {
			k.Soft = []string{r.Key + ": " + r.Detail}
		}
		out = append(out, k)
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

func itoa(n int) string { return strconv.Itoa(n) }

func anyStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case int:
		return itoa(x)
	default:
		b, _ := json.Marshal(v)
		return string(b)
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
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	default:
		return 0
	}
}
