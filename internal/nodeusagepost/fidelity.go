package nodeusagepost

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// FidelitySchema tags the machine output of the node-usage report scorer.
const FidelitySchema = "fak.nodeusage.fidelity/v1"

// A node-usage card is a SELF-REPORT about the fleet: FromSnapshot folds a snapshot into
// prose, a grade, and a verdict, and #node-usage trusts that prose. The whole reason this
// package exists is that the naive fold LIES — it calls a silent box "down", paints a real
// outage green, hides a down-with-error behind "no visibility". FromSnapshot was rewritten
// to stop lying; nothing, until now, PROVED it still doesn't.
//
// Score is that proof. It is an INDEPENDENT witness: it never trusts the renderer, it
// re-derives every fact straight from the ground-truth snapshot (the exact JSON `fak lab
// status --json` emits) and checks the card against it — the same "evidence, not
// self-report" discipline the truth syscall applies to a commit, aimed here at the report.
// A card that faithfully represents its snapshot scores 100; every point lost names a
// specific lie or dropped fact. This makes the reports accurate (a regression that
// reintroduces a lie fails the gate), useful (an operator can grade any card against its
// snapshot), and — because the scorer is itself a pure function with a corpus — honest
// about its own coverage.

// Check is one verified invariant between a card and its snapshot. A check that does not
// apply to a given snapshot (no GPU stat, no down box) is omitted entirely, so Score's
// denominator only ever counts checks that had something to verify — a card is never
// credited for a check that could not fail.
type Check struct {
	Name   string `json:"name"`
	Weight int    `json:"weight"`
	Pass   bool   `json:"pass"`
	Detail string `json:"detail"` // what was verified, or how the card diverged from the snapshot
}

// Fidelity is the scored verdict for one node-usage card. Pass is the honest bit: the card
// is trustworthy only when EVERY applicable check held — a single lie is disqualifying, so
// Pass is not "Score is high enough". Score is the weighted pass fraction, a graded read
// for a card that is mostly-but-not-perfectly faithful, and Grade bands it the same A–F way
// the cards themselves are graded.
type Fidelity struct {
	Schema     string   `json:"schema"`
	Score      int      `json:"score"` // 0–100, weighted fraction of applicable checks that passed
	Grade      string   `json:"grade"` // A–F over Score, for a one-glyph read
	Pass       bool     `json:"pass"`  // true iff no applicable check failed (Score == 100)
	Checks     []Check  `json:"checks"`
	Violations []string `json:"violations,omitempty"` // "name: detail" for each failed check, for a quick scan
}

// weights: honesty checks (the ones that keep a lie off #node-usage) are worth more than
// completeness checks (a dropped state count is sloppy, not dishonest), so a single honesty
// failure dominates the score the way it should.
const (
	weightHonesty      = 2
	weightCompleteness = 1
)

// Score grades how faithfully a rendered node-usage card (up) represents its ground-truth
// snapshot (snap). It is pure: no clock, no I/O, snapshot in / verdict out, so the CLI, the
// CI gate, and the corpus test all read the same number. snap must be the SAME snapshot the
// card was folded from (`fak lab status --json`); scoring a card against a different
// snapshot is a category error the caller must not make.
func Score(up scoreboard.Update, snap fleet.Snapshot) Fidelity {
	sig := readSignals(snap) // the honest re-derivation off ByState — never off the card
	lines := up.Lines

	var checks []Check
	add := func(name string, weight int, applies bool, pass bool, detail string) {
		if !applies {
			return
		}
		checks = append(checks, Check{Name: name, Weight: weight, Pass: pass, Detail: detail})
	}

	// --- headline counts: the numbers an operator reads first ----------------------------
	wantReach := fmt.Sprintf("%d/%d reachable", snap.Reachable, snap.Total)
	add("reachable-headline", weightHonesty, true, up.Score == wantReach,
		fmt.Sprintf("score line %q, snapshot wants %q", up.Score, wantReach))

	wantReadiness := fmt.Sprintf("readiness: %d", snap.Score)
	add("readiness-line", weightCompleteness, true, hasLine(lines, wantReadiness),
		fmt.Sprintf("want a %q line", wantReadiness))

	// --- reporting / visibility: the load-bearing silent-vs-down honesty ------------------
	if snap.Total > 0 {
		want := fmt.Sprintf("reporting: %d/%d", snap.Total, snap.Total)
		if sig.silent > 0 {
			want = fmt.Sprintf("reporting: %d/%d (%d silent=unknown, not down)",
				snap.Total-sig.silent, snap.Total, sig.silent)
		}
		add("reporting-line", weightHonesty, true, hasLine(lines, want),
			fmt.Sprintf("want an honest reporting line %q", want))
	}

	// --- no phantom down: the card may name EXACTLY as many down boxes as the snapshot has.
	// The founding lie was calling a silent/errored box "down"; the mirror lie would be
	// hiding a real down. Both are caught by tying every "down" claim to ByState[down].
	if sig.down == 0 {
		blob := strings.ToLower(up.Detail + "\n" + strings.Join(lines, "\n"))
		bad := phantomDown(blob)
		add("no-phantom-down", weightHonesty, true, bad == "",
			fmt.Sprintf("snapshot has 0 down boxes; card must not assert an outage (offending text: %q)", bad))
	} else {
		named := strings.Contains(up.Detail, "reported down") || anyLineContains(lines, "down:")
		add("down-is-named", weightHonesty, true, named,
			fmt.Sprintf("snapshot has %d down box(es); card must name the outage", sig.down))
	}

	// --- red-paint honesty: the renderer picks its glyph from the grade PREFIX before the
	// verdict, so a real problem must never carry an A/B grade or it renders green/yellow. ---
	problem := sig.down > 0 || sig.hasCrit || sig.hasWarn
	add("problem-not-green", weightHonesty, problem, !gradeReadsHealthy(up.Grade),
		fmt.Sprintf("an observed problem must clamp the grade below B; grade=%q", up.Grade))

	add("down-forces-action", weightHonesty, sig.down > 0,
		up.Verdict == "ACTION" && up.Grade != "" && !gradeReadsHealthy(up.Grade),
		fmt.Sprintf("a reported-down fleet must read ACTION with a red grade; grade=%q verdict=%q", up.Grade, up.Verdict))

	// --- visibility gap is neutral, never an outage --------------------------------------
	noVis := snap.Total > 0 && sig.down == 0 && sig.silent == snap.Total
	add("no-visibility-neutral", weightHonesty, noVis,
		up.Grade == "" && up.Verdict == "" &&
			strings.Contains(up.Detail, "not down") &&
			strings.Contains(strings.ToLower(up.NextStep), "populate liveness"),
		fmt.Sprintf("an all-silent fleet must read neutral (no grade/verdict) with populate-liveness guidance; grade=%q verdict=%q detail=%q",
			up.Grade, up.Verdict, up.Detail))

	// --- completeness: no per-state or per-class count silently dropped -------------------
	add("state-counts-complete", weightCompleteness, snap.Total > 0,
		missingStateLines(lines, snap) == "",
		fmt.Sprintf("every non-zero per-state count must appear (missing: %s)", missingStateLines(lines, snap)))

	add("class-counts-complete", weightCompleteness, len(snap.ByClass) > 0,
		missingClassLines(lines, snap) == "",
		fmt.Sprintf("every per-class count must appear (missing: %s)", missingClassLines(lines, snap)))

	// --- GPU capacity line matches the aggregate exactly ---------------------------------
	if g := snap.GPUUtil; g != nil {
		want := fmt.Sprintf("gpu capacity: busy %d/%d, idle %d (%d%% util)",
			g.Busy, g.Total, idleGPUs(g), g.UtilPct)
		add("gpu-line-fidelity", weightCompleteness, true, hasLine(lines, want),
			fmt.Sprintf("want the GPU aggregate line %q", want))
	}

	// --- well-formedness: the renderer only understands this closed vocabulary -----------
	add("verdict-wellformed", weightCompleteness, true,
		wellformedVerdict(up.Verdict) && wellformedGrade(up.Grade),
		fmt.Sprintf("verdict %q must be one of \"\"|OK|ACTION and grade %q one of \"\"|N/A|A-F", up.Verdict, up.Grade))

	return finalize(checks)
}

// finalize folds the checks into the scored verdict: weighted pass fraction, A–F band, the
// all-passed bit, and a flat violation list for a one-glance read.
func finalize(checks []Check) Fidelity {
	var got, total int
	var violations []string
	for _, c := range checks {
		total += c.Weight
		if c.Pass {
			got += c.Weight
			continue
		}
		violations = append(violations, c.Name+": "+c.Detail)
	}
	score := 100
	if total > 0 {
		score = int(math.Round(100 * float64(got) / float64(total)))
	}
	return Fidelity{
		Schema:     FidelitySchema,
		Score:      score,
		Grade:      gradeOf(score),
		Pass:       len(violations) == 0,
		Checks:     checks,
		Violations: violations,
	}
}

// gradeReadsHealthy reports whether a grade would drive the card renderer's glyph green or
// yellow. gradeEmoji keys off HasPrefix(grade,"A")/"B" BEFORE it consults the verdict, so
// any grade starting A or B reads healthy regardless of an ACTION verdict — the exact
// mask-a-real-problem bug the clamp exists to prevent.
func gradeReadsHealthy(grade string) bool {
	return strings.HasPrefix(grade, "A") || strings.HasPrefix(grade, "B")
}

// phantomDown returns the offending fragment if the text asserts an outage over a fleet the
// snapshot says has zero down boxes, or "" if clean. Every legitimate mention of "down" in
// an honest zero-down card is the reassuring phrase "not down"; anything else — "reported
// down", a "down: N" state line, a "down N" capacity token — is a phantom outage.
func phantomDown(lowerBlob string) string {
	for _, line := range strings.Split(lowerBlob, "\n") {
		idx := 0
		for {
			at := strings.Index(line[idx:], "down")
			if at < 0 {
				break
			}
			pos := idx + at
			// "not down" is the honest, non-alarming phrasing — skip it.
			if pos >= 4 && line[pos-4:pos+4] == "not down" {
				idx = pos + 4
				continue
			}
			return strings.TrimSpace(line)
		}
	}
	return ""
}

func missingStateLines(lines []string, snap fleet.Snapshot) string {
	var missing []string
	// stateLines (fleet.go) is the single source of the wanted per-state lines, in the fixed
	// operational order. Reuse it here so "which per-state counts did the card drop?" can never
	// drift from what the card actually renders — and so this scorer holds no second copy of the
	// state-order iteration.
	for _, want := range stateLines(snap) {
		if !hasLine(lines, want) {
			missing = append(missing, want)
		}
	}
	return strings.Join(missing, ", ")
}

func missingClassLines(lines []string, snap fleet.Snapshot) string {
	var missing []string
	for _, c := range snap.ByClass {
		want := fmt.Sprintf("%s=%d", c.Key, c.Count)
		if !anyLineContains(lines, want) {
			missing = append(missing, want)
		}
	}
	sort.Strings(missing)
	return strings.Join(missing, ", ")
}

func wellformedVerdict(v string) bool {
	switch v {
	case "", "OK", "ACTION":
		return true
	}
	return false
}

func wellformedGrade(g string) bool {
	switch g {
	case "", "N/A", "A", "B", "C", "D", "F":
		return true
	}
	return false
}

func hasLine(lines []string, want string) bool {
	for _, l := range lines {
		if l == want {
			return true
		}
	}
	return false
}

func anyLineContains(lines []string, sub string) bool {
	for _, l := range lines {
		if strings.Contains(l, sub) {
			return true
		}
	}
	return false
}

// RenderFidelity is the human, one-screen read of a scored card — the header verdict, the
// score, then each check as a pass/fail line with its detail on a failure. It is the text
// `fak nodeusage score` prints; --json emits the Fidelity struct itself.
func RenderFidelity(f Fidelity) string {
	var b strings.Builder
	status := "PASS"
	if !f.Pass {
		status = "FAIL"
	}
	fmt.Fprintf(&b, "node-usage report fidelity: %s  score=%d grade=%s (%d checks)\n",
		status, f.Score, f.Grade, len(f.Checks))
	for _, c := range f.Checks {
		mark := "ok  "
		if !c.Pass {
			mark = "FAIL"
		}
		fmt.Fprintf(&b, "  [%s] %s\n", mark, c.Name)
		if !c.Pass {
			fmt.Fprintf(&b, "         %s\n", c.Detail)
		}
	}
	return b.String()
}
