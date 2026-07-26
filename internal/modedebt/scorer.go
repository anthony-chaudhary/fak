package modedebt

// scorer.go is the PRODUCER half of the mode-debt scorer/dispatcher pair (#4415,
// epic #4397, under harness-native #2387 / permission regimes #2389). dispatch.go
// (#4416) consumes what this file emits: the same Scorecard/Dial shape, so the
// scorer -> dispatcher contract is one struct, not two hand-synced copies.
//
// THE THESIS. Every behavior-reshaping dial fak exposes -- a FAK_* env var, a CLI
// flag, a boolean toggle -- is a MODE whether or not we call it one. Today they are
// scattered and implicit: read raw off the environment, silent about whether they
// can widen the safety floor, and invisible to the decision journal. #2405 wants
// them lifted into typed [regimes] entries and #2389 needs a prioritized worklist to
// sequence that lift. This file is that census: a pure scorer that enumerates the
// dials and grades each against the four regime criteria. It is a WORKLIST, NOT A
// FIX -- it builds no regime, lifts no dial, adds no clamp, journals no transition.
//
// THE TARGET IS IMPLICITNESS, NOT STATEFULNESS. A dial that is legitimately
// persistent, typed, floor-clamped and journaled scores CLEAN; a safety dial that is
// correctly harness-held and model-unreachable is NOT debt at all and is never
// ranked. Only the un-lifted remainder becomes the #2389/#2405 worklist.
//
// THE FOUR CRITERIA (Criteria):
//  1. Lifted       -- the dial is a typed [regimes] entry (#2405) rather than a raw read.
//  2. FloorClamped -- its profile passes through adjudicator.sanitizeProfile, which
//     clears every MANDATORY rung's elision bit, so the dial can only NARROW the
//     write/self-modify floor, never widen it.
//  3. Journaled    -- flipping it lands a durable chained row (journal.KindPolicyOp /
//     journal.KindConfigSwap), so an auditor can reconstruct the mode timeline.
//  4. Orthogonal   -- the dial names ONE axis instead of bundling several behind one
//     opaque flag.
//
// THE RANK. HARD/SOFT is the LIFT worklist rank, so it is reserved for un-lifted
// dials: HARD when the dial is also un-clamped (it can widen the mandatory floor --
// lift it first), SOFT when the floor already holds it (real debt, later in the
// queue). The HARD => un-lifted invariant is what makes dispatch.go's
// SelectHardUnlifted a safe filter, and it is pinned by TestModeDebtGradeInvariants.
// A dial that IS lifted but still misses a lower bar is graded SOFT rather than
// laundered into CLEAN -- CLEAN means all four, and an inconsistent fact set should
// stay visible. In practice a real #2405 lift satisfies (2) and (3) by construction
// (a typed regime entry is sanitized and its set_regime op is journaled), so that
// branch is a defensive honesty rung, not the expected steady state.
//
// DETERMINISTIC AND READ-ONLY. Scan walks the tree and mutates nothing; the same
// tree yields the same byte-identical scorecard. It CONSUMES internal/knobcensus
// (#2210) for dial enumeration rather than re-deriving a second flag/env inventory,
// the same "no second count" contract knobcensus itself honors toward #2199.

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/knobcensus"
)

// Grade is the closed four-token verdict vocabulary. Exactly one applies per dial.
const (
	// GradeClean marks a dial that meets all four regime criteria. Persistent is
	// fine -- CLEAN is about being typed, clamped and journaled, not about being
	// stateless.
	GradeClean = "CLEAN"
	// GradeHard marks an un-lifted dial that is ALSO un-clamped: nothing stops it
	// widening the mandatory floor, so it heads the #2405 lift queue. This is the
	// only grade dispatch.go fans out (SelectHardUnlifted).
	GradeHard = "HARD"
	// GradeSoft marks debt that cannot widen the floor: an un-lifted but already
	// floor-clamped dial, or a lifted dial still short of a lower bar.
	GradeSoft = "SOFT"
	// GradeNotDebt marks a correctly harness-held, model-unreachable safety dial.
	// It is excluded from the debt count and never ranked -- holding a safety floor
	// outside the model's reach is the design, not a gap.
	GradeNotDebt = "NOT-DEBT"
)

// Criteria is a dial's standing against the four regime bars (#2389/#2405).
type Criteria struct {
	Lifted       bool `json:"lifted"`
	FloorClamped bool `json:"floor_clamped"`
	Journaled    bool `json:"journaled"`
	Orthogonal   bool `json:"orthogonal"`
}

// AllMet reports whether every bar is met -- the CLEAN condition.
func (c Criteria) AllMet() bool {
	return c.Lifted && c.FloorClamped && c.Journaled && c.Orthogonal
}

// Met counts the bars cleared, the "how far from the bar" number the worklist sorts
// on within a grade.
func (c Criteria) Met() int {
	n := 0
	for _, ok := range []bool{c.Lifted, c.FloorClamped, c.Journaled, c.Orthogonal} {
		if ok {
			n++
		}
	}
	return n
}

// Unmet names the bars still open, in a fixed order so the emitted detail string is
// deterministic.
func (c Criteria) Unmet() []string {
	var out []string
	for _, bar := range []struct {
		ok   bool
		name string
	}{
		{c.Lifted, "not lifted into a typed [regimes] entry"},
		{c.FloorClamped, "not floor-clamped by sanitizeProfile"},
		{c.Journaled, "flip is not journaled"},
		{c.Orthogonal, "not decomposed into orthogonal axes"},
	} {
		if !bar.ok {
			out = append(out, bar.name)
		}
	}
	return out
}

// Facts is one dial as presented to the PURE grader: its identity and provenance,
// its standing on the four criteria, and the documented reason (if any) that it is a
// correctly harness-held safety dial rather than debt. Grade reads nothing else --
// no filesystem, no clock -- so a caller can grade a hypothetical dial in a test
// exactly as Scan grades a real one.
type Facts struct {
	Slug    string // stable identity; falls back to Name when empty
	Name    string // human name (the flag/env spelling)
	Surface string // "flag" | "env"
	File    string // repo-relative provenance, forward slashes
	Line    int
	Regime  string // the typed [regimes] entry it is lifted into ("" = un-lifted)

	Criteria Criteria

	// SafetyHold, when non-empty, is the reviewed reason this dial is harness-held
	// AND model-unreachable. It excludes the dial from debt entirely (GradeNotDebt).
	SafetyHold string
}

// Grade is the pure scorer: four criteria (plus the safety-hold exclusion) in, one
// closed-vocabulary verdict and its reason out. Total, deterministic, side-effect
// free. The order of the arms IS the doctrine -- exclusion first, then CLEAN, then
// the un-lifted rank.
func Grade(f Facts) (grade string, reason string) {
	if hold := strings.TrimSpace(f.SafetyHold); hold != "" {
		return GradeNotDebt, "harness-held and model-unreachable safety dial, not debt: " + hold
	}
	if f.Criteria.AllMet() {
		return GradeClean, "typed [regimes] lift, sanitizeProfile floor-clamp, journaled transition, orthogonal axis"
	}
	unmet := strings.Join(f.Criteria.Unmet(), "; ")
	if !f.Criteria.Lifted {
		if !f.Criteria.FloorClamped {
			return GradeHard, "un-lifted AND un-clamped, so it can widen the mandatory floor: " + unmet
		}
		return GradeSoft, "un-lifted but the floor already clamps it, so it cannot widen the floor: " + unmet
	}
	// Lifted, yet short of a lower bar. Never HARD: there is nothing left to lift,
	// so it can never join the lift worklist, but it is not CLEAN either.
	return GradeSoft, "lifted into a typed regime but still short of the bar: " + unmet
}

// GradeDial grades one dial and returns it in the Dial shape dispatch.go consumes.
func GradeDial(f Facts) Dial {
	grade, reason := Grade(f)
	id := strings.TrimSpace(f.Slug)
	if id == "" {
		id = f.Name
	}
	return Dial{
		Slug:     slug(id),
		Name:     strings.TrimSpace(f.Name),
		Grade:    grade,
		Lifted:   f.Criteria.Lifted,
		Regime:   strings.TrimSpace(f.Regime),
		Detail:   reason,
		Surface:  f.Surface,
		File:     f.File,
		Line:     f.Line,
		Criteria: f.Criteria,
		Excluded: strings.TrimSpace(f.SafetyHold),
	}
}

// Score grades a whole dial set into the scorecard dispatch.go loads. Dials sort by
// their stable key so a re-run is byte-identical, and the roll-up counts are derived
// here so no consumer has to re-fold the grades.
func Score(facts []Facts) Scorecard {
	sc := Scorecard{Schema: Schema, Dials: make([]Dial, 0, len(facts))}
	for _, f := range facts {
		sc.Dials = append(sc.Dials, GradeDial(f))
	}
	sort.SliceStable(sc.Dials, func(i, j int) bool { return sc.Dials[i].Key() < sc.Dials[j].Key() })
	for _, d := range sc.Dials {
		switch d.Grade {
		case GradeClean:
			sc.Clean++
		case GradeHard:
			sc.Hard++
		case GradeSoft:
			sc.Soft++
		case GradeNotDebt:
			sc.NotDebt++
		}
	}
	// The headline mode_debt integer: ranked debt only. A CLEAN dial and a
	// harness-held safety dial both contribute zero.
	sc.Debt = sc.Hard + sc.Soft
	return sc
}

// --- the live census ---

// HarnessHeldSafety is the reviewed exclusion list: dials that are genuinely
// harness-held AND model-unreachable, keyed by folded dial name (see foldDial) and
// paired with the reason. It is the same review chokepoint unwiredscore.AllowUnwired
// and architest's reg-off list apply.
//
// It is seeded EMPTY on purpose. This census's job is to SURFACE dials, and an
// unproven exclusion is the one error mode that silently shrinks the worklist --
// over-reporting debt is the safe direction, under-reporting it is not. A dial earns
// an entry only when "the model cannot reach it, and the harness holds it" is a
// reviewed fact, not an assumption. Every excluded dial must name why here.
var HarnessHeldSafety = map[string]string{}

// regimeSection matches the [regimes] / [regimes.<name>] TOML table headers that
// #2405 will introduce. Until that lift lands the tree declares none, so every dial
// scores un-lifted -- which is the honest census, and exactly the worklist #2405 is
// about to burn down.
var regimeSection = regexp.MustCompile(`^\s*\[regimes(?:\.([^\]]+))?\]\s*$`)

var (
	// reEnvName matches a FAK_* environment dial mentioned in an evidence root.
	reEnvName = regexp.MustCompile(`FAK_[A-Z0-9_]+`)
	// reQuoted matches a quoted flag-shaped token ("policy", "allow-widen"): a bare
	// lowercase word, so an evidence probe does not sweep in prose or paths.
	reQuoted = regexp.MustCompile(`"([a-z][a-z0-9-]{2,})"`)
	// reTOMLKey matches a bare TOML key on the left of an '='.
	reTOMLKey = regexp.MustCompile(`^\s*([A-Za-z0-9_.-]+)\s*=`)
	// reTOMLString matches every quoted string on a TOML line (a regime's dial list).
	reTOMLString = regexp.MustCompile(`"([^"]+)"`)
)

// floorRoots are the packages that own the sanitizeProfile-clamped capability floor.
// A dial named there reaches the floor through a policy, so the clamp holds it; a
// dial read raw in cmd/fak and never mentioned here does not.
var floorRoots = []string{
	filepath.Join("internal", "adjudicator"),
	filepath.Join("internal", "policy"),
}

// journalRoots own the durable decision journal. A dial named there has its
// transition recorded (the KindPolicyOp / KindConfigSwap rows); one that is not is a
// silent flip -- the "hidden state is a bug" case this census exists to rank.
var journalRoots = []string{
	filepath.Join("internal", "journal"),
}

// omnibusTokens name a dial that bundles several axes behind one opaque switch --
// the "one opaque flag" criterion (4) rejects. Name-only and deliberately narrow,
// the same philosophy as knobcensus.classify: a bundle word ("profile", "mode",
// "all", "yolo") denotes a preset, not an axis.
var omnibusTokens = []string{
	"all", "everything", "yolo", "dangerous", "unsafe", "bypass",
	"preset", "profile", "mode",
}

// Scan walks root and returns the graded, sorted mode-debt scorecard. It is
// deterministic and read-only: the same tree marshals to the same bytes, which is
// the census witness. A missing evidence root contributes nothing rather than
// erroring -- an installed tree without internal/journal simply proves no dial
// journaled, the conservative direction.
func Scan(root string) (Scorecard, error) {
	census, err := knobcensus.Scan(root)
	if err != nil {
		return Scorecard{}, err
	}
	regimes, err := RegimeIndex(root)
	if err != nil {
		return Scorecard{}, err
	}
	floor, err := mentions(root, floorRoots)
	if err != nil {
		return Scorecard{}, err
	}
	journaled, err := mentions(root, journalRoots)
	if err != nil {
		return Scorecard{}, err
	}

	facts := make([]Facts, 0, len(census.Knobs))
	for _, k := range census.Knobs {
		// Census the DIAL surfaces only. knobcensus also folds in #2199's skill
		// knobs; a skill is not a behavior dial, and per-invocation arguments are
		// out of census by the issue's own scope note.
		if k.Surface != knobcensus.SurfaceFlag && k.Surface != knobcensus.SurfaceEnv {
			continue
		}
		folded := foldDial(k.Name)
		regime := regimes[folded]
		facts = append(facts, Facts{
			Slug:    k.Name,
			Name:    k.Name,
			Surface: string(k.Surface),
			File:    k.File,
			Line:    k.Line,
			Regime:  regime,
			Criteria: Criteria{
				Lifted:       regime != "",
				FloorClamped: floor[folded],
				Journaled:    journaled[folded],
				Orthogonal:   isOrthogonal(k.Name),
			},
			SafetyHold: HarnessHeldSafety[folded],
		})
	}
	return Score(facts), nil
}

// RegimeIndex reads the workspace's typed [regimes] declarations and returns
// foldedDialName -> regime name. #2405 owns the final table schema, so the reader is
// deliberately liberal: under a [regimes.<name>] table it harvests both the bare
// keys and every quoted string (a `dials = [...]` list), and under a bare [regimes]
// table it harvests the keys. An absent dos.toml is not an error -- it simply means
// nothing is lifted yet.
func RegimeIndex(root string) (map[string]string, error) {
	out := map[string]string{}
	f, err := os.Open(filepath.Join(root, "dos.toml"))
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return nil, err
	}
	defer f.Close()

	regime := ""
	inRegimes := false
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if m := regimeSection.FindStringSubmatch(line); m != nil {
			inRegimes, regime = true, strings.TrimSpace(m[1])
			continue
		}
		// Any other table header ends the regimes block.
		if strings.HasPrefix(strings.TrimSpace(line), "[") {
			inRegimes, regime = false, ""
			continue
		}
		if !inRegimes {
			continue
		}
		name := regime
		if name == "" {
			name = "regimes"
		}
		if m := reTOMLKey.FindStringSubmatch(line); m != nil {
			out[foldDial(m[1])] = name
		}
		for _, m := range reTOMLString.FindAllStringSubmatch(line, -1) {
			out[foldDial(m[1])] = name
		}
	}
	return out, sc.Err()
}

// mentions collects every dial-shaped token named anywhere under the given roots,
// folded to the dial identity. It is the evidence probe behind criteria (2) and (3):
// a dial the floor packages never name cannot be reaching the clamp, and one the
// journal packages never name cannot be leaving a row. A missing root contributes
// nothing.
func mentions(root string, subdirs []string) (map[string]bool, error) {
	out := map[string]bool{}
	for _, sub := range subdirs {
		dir := filepath.Join(root, sub)
		err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				if os.IsNotExist(err) {
					return nil
				}
				return err
			}
			if d.IsDir() {
				if d.Name() == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") {
				return nil
			}
			return scanMentions(path, out)
		})
		if err != nil && !os.IsNotExist(err) {
			return nil, err
		}
	}
	return out, nil
}

func scanMentions(path string, out map[string]bool) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		for _, m := range reEnvName.FindAllString(line, -1) {
			out[foldDial(m)] = true
		}
		for _, m := range reQuoted.FindAllStringSubmatch(line, -1) {
			out[foldDial(m[1])] = true
		}
	}
	return sc.Err()
}

// isOrthogonal reports whether the dial names ONE axis rather than bundling several
// behind an opaque preset word (criterion 4).
func isOrthogonal(name string) bool {
	low := strings.ToLower(name)
	for _, t := range omnibusTokens {
		if strings.Contains(low, t) {
			return false
		}
	}
	return true
}

// foldDial folds a flag or env spelling to the logical dial identity, so the same
// control reached as --account and as FAK_ACCOUNT is one dial. Mirrors
// knobcensus's own route-coverage folding.
func foldDial(name string) string {
	low := strings.ToLower(strings.TrimSpace(name))
	low = strings.TrimPrefix(low, "fak_")
	low = strings.TrimPrefix(low, "fak-")
	low = strings.ReplaceAll(low, "_", "")
	low = strings.ReplaceAll(low, "-", "")
	return low
}
