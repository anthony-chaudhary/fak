// Package antipattern is the UNIFYING REGISTRY for the agentic-dev anti-patterns
// whose common shape is "work that did not convert into global, user-useful progress":
// work REDONE that was already done (repetition), and work LANDED but connected to
// nothing (lost / orphaned). The full 43-class taxonomy is research in
// docs/notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md; this package is the SPINE that
// note specced but never built -- a registry that folds several previously-siloed
// detectors into ONE anti-pattern debt integer and one worst-first work-list.
//
// THE DISCIPLINE THAT MAKES IT HONEST. The note describes a 5-rung detection ladder
// (R0 NAMED -> R4 RATCHETED). A registry is worthless if it fills up with R0 rows --
// classes named in prose with no detector behind them -- because then its coverage
// count lies. So this package refuses an aspirational row: a Class earns a Spec here
// ONLY once a real detector folds Findings for it. Every registered class is at least
// R2 SCORED (its findings count as debt). That is why v1 registers exactly three
// classes, not forty-three: those are the three with a wired detector today.
//
// WHAT IT UNIFIES (v1):
//
//   - REDUNDANT_REWORK (repetition) -- the genuinely-uncovered class. Write-time and
//     pre-spawn dedup already exist (internal/issuededup, internal/dispatchtick,
//     internal/guardrsi livelock), but nothing detected two LANDED commits that redid
//     the same unit of work after the fact. DetectRedundantRework is that detector.
//   - UNWIRED_PKG (lost work) -- a code-complete internal package imported by no .go
//     file. Folded from internal/unwiredscore (the existing package-granularity oracle).
//   - ORPHAN_FUNC (lost work) -- an unexported top-level func referenced nowhere in its
//     package. Folded from internal/orphanscan (the existing function-granularity oracle).
//
// The pure core is Fold([]Finding, universe) -> scorecard.Payload: data in, control-pane
// payload out, no disk or git. The impure adapters that PRODUCE findings from the tree
// and from git history live in detect.go, and Build wires them together -- the same
// pure-core / thin-adapter split internal/unwiredscore uses. This lets a future
// `fak superloop` intent walk one anti-pattern oracle instead of five siloed ones.
package antipattern

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id any consumer keys on.
const Schema = "fak-antipattern-scorecard/1"

// DebtKey is the headline integer the control-pane folds (corpus.antipattern_debt) --
// the total count of detected anti-pattern findings across every registered class.
const DebtKey = "antipattern_debt"

// Class is the closed anti-pattern vocabulary this registry has a WIRED detector for.
// A value outside the registry is a bug, not a lower-priority bucket -- the same
// closed-vocabulary discipline the kernel applies to a refusal reason or a maturity rung.
type Class string

const (
	// ClassRedundantRework: two or more LANDED commits that redid the same unit of work
	// (the churn-for-progress / post-hoc repetition class no other detector covers).
	ClassRedundantRework Class = "REDUNDANT_REWORK"
	// ClassUnwiredPkg: a code-complete internal package imported by no .go file (lost
	// work at package granularity; folded from internal/unwiredscore).
	ClassUnwiredPkg Class = "UNWIRED_PKG"
	// ClassOrphanFunc: an unexported top-level func referenced nowhere in its package
	// (lost work at function granularity; folded from internal/orphanscan).
	ClassOrphanFunc Class = "ORPHAN_FUNC"
)

// Group is the coarse anti-pattern family a class rolls up to.
type Group string

const (
	// GroupRepetition: work redone that was already done.
	GroupRepetition Group = "repetition"
	// GroupLostWork: work landed but wired into nothing.
	GroupLostWork     Group = "lost-work"
	GroupGraderGaming Group = "grader-gaming"
)

// Rung is the note's detection-maturity ladder. The registry records each class's rung
// so coverage is EXPLICIT: a reader sees not just "is this class listed" but "how far up
// the ladder is it actually enforced". v1 classes are all RungScored -- detected and
// folded into debt, but not yet dispatched to issues or held by a CI ratchet.
type Rung int

const (
	RungNamed      Rung = 0 // named in the taxonomy only; no detector (never registered here)
	RungDetected   Rung = 1 // a detector produces findings
	RungScored     Rung = 2 // findings fold into the debt integer (the v1 bar)
	RungDispatched Rung = 3 // debt fans out to tracked GitHub issues
	RungRatcheted  Rung = 4 // a CI floor holds the debt from regressing
)

// Label renders a rung as its ladder token, for a render line or a doc.
func (r Rung) Label() string {
	switch r {
	case RungNamed:
		return "R0-named"
	case RungDetected:
		return "R1-detected"
	case RungScored:
		return "R2-scored"
	case RungDispatched:
		return "R3-dispatched"
	case RungRatcheted:
		return "R4-ratcheted"
	default:
		return fmt.Sprintf("R?(%d)", int(r))
	}
}

// Spec is one registered anti-pattern class: its group, its ladder rung, a one-line
// title, the default mitigation phrase a finding of this class carries, and its Cure --
// the concrete, routed remediation (a real fak verb or an explicit manual action) that
// turns detection into mitigation. Edit this table (not a detector's logic) to add a
// class or move it up the ladder.
//
// THE CURE INVARIANT. Detection without a paired cure is half a loop: it names debt but
// never closes it. So every registered class MUST declare a non-empty Cure, enforced by
// TestEveryClassHasACure -- you cannot wire a detector here without also wiring how the
// meta-loop mitigates it. The Cure names a REAL surface (verified to exist): a fak verb
// that performs the remediation, or an explicit manual action where no verb exists yet.
// Never cite a cure verb that does not exist -- a fabricated cure is itself the lost-work
// anti-pattern this registry detects.
type Spec struct {
	Class      Class
	Group      Group
	Rung       Rung
	Title      string
	Mitigation string
	Cure       string
	// AutoCure reports whether Cure routes to a fak verb a mitigation loop may dispatch
	// UNATTENDED -- one that files an issue or a work order without rewriting code. A cure
	// that edits code (rename-concept) or needs human judgment (delete dead code) is false,
	// so the loop emits it as a work order instead of running it. This is the one bit that
	// separates "detect and auto-mitigate" from "detect and hand off".
	AutoCure bool
}

// registry is the canonical, ordered class table. The order is the render/KPI order:
// repetition first (the newest, most operator-visible loss), then lost-work coarse-to-fine.
var registry = []Spec{
	{
		Class:      CheckerGames,
		Group:      GroupGraderGaming,
		Title:      "solution games the checker",
		Mitigation: "remove the shortcut and make the real assertion exercise the produced behavior",
		Cure:       "manual: inspect the named artifact and replace the shortcut with real behavior",
		Rung:       RungScored,
	},
	{
		Class:      ClassRedundantRework,
		Group:      GroupRepetition,
		Rung:       RungScored,
		Title:      "commits that redundantly redid already-shipped work",
		Mitigation: "consolidate; confirm the later commits were not no-ops re-claiming shipped work",
		// Concept-disambiguation is the root-cause cure: two sessions redo one unit of work
		// when they name the same concept differently and so never see each other's landing.
		// `fak concept-usage-score` surfaces the conflated concept; `fak rename-concept`
		// unifies it tree-wide; `fak dup` locates the duplicated block to consolidate.
		Cure: "concept-disambiguation: `fak concept-usage-score` to surface the conflated concept, then `fak rename-concept` to unify it tree-wide, then `fak dup` to find the duplicate block",
	},
	{
		Class:      ClassUnwiredPkg,
		Group:      GroupLostWork,
		Rung:       RungScored,
		Title:      "code-complete internal package imported by nothing",
		Mitigation: "wire it into a default path (a verb, the request path, a benchmark) or retire it",
		Cure:       "`fak unwired-debt-dispatch` files one deduped issue to wire the package into a runnable surface or retire it",
		// Files a deduped issue -- reversible, no code rewrite -- so a loop may dispatch it unattended.
		AutoCure: true,
	},
	{
		Class:      ClassOrphanFunc,
		Group:      GroupLostWork,
		Rung:       RungScored,
		Title:      "unexported top-level func referenced nowhere in its package",
		Mitigation: "route/register it, or delete it",
		Cure:       "manual: reference the func from its intended call site, or delete it as dead code (no auto-dispatch verb yet)",
	},
}

// specByClass indexes the registry for O(1) lookup.
var specByClass = func() map[Class]Spec {
	m := make(map[Class]Spec, len(registry))
	for _, s := range registry {
		m[s.Class] = s
	}
	return m
}()

// Registry returns a copy of the ordered class table, so callers (a test, a doc
// generator) can read coverage without reaching into the unexported slice.
func Registry() []Spec { return append([]Spec(nil), registry...) }

// SpecOf returns the registered spec for a class and ok=false for an unregistered value.
func SpecOf(c Class) (Spec, bool) {
	s, ok := specByClass[c]
	return s, ok
}

// Finding is one detected anti-pattern instance. Ref locates it (a file:line, a package
// dir, or a sha cluster); Weight is the worst-first ordering key (bigger == louder:
// number of redundant commits, source lines stranded, etc.). Mitigation defaults to the
// class spec's phrase but a detector may override it with an instance-specific fix.
type Finding struct {
	Class      Class  `json:"class"`
	Ref        string `json:"ref"`
	Detail     string `json:"detail"`
	Mitigation string `json:"mitigation"`
	Weight     int    `json:"weight"`
}

// line renders a finding as a single closed-vocabulary work-list entry a gate or a human
// can grep: "<CLASS> <ref>: <detail> -- <mitigation>".
func (f Finding) line() string {
	mit := f.Mitigation
	if mit == "" {
		if s, ok := specByClass[f.Class]; ok {
			mit = s.Mitigation
		}
	}
	return fmt.Sprintf("%s %s: %s -- %s", f.Class, f.Ref, f.Detail, mit)
}

// SortFindings orders findings worst-first: by Weight desc, then registry class order,
// then Ref, so the fold is deterministic and the loudest debt renders first.
func SortFindings(fs []Finding) {
	order := map[Class]int{}
	for i, s := range registry {
		order[s.Class] = i
	}
	sort.SliceStable(fs, func(i, j int) bool {
		a, b := fs[i], fs[j]
		if a.Weight != b.Weight {
			return a.Weight > b.Weight
		}
		if order[a.Class] != order[b.Class] {
			return order[a.Class] < order[b.Class]
		}
		return a.Ref < b.Ref
	})
}

// CureAction is one routed remediation step: a detected finding paired with the exact cure
// that mitigates its class and whether a loop may run it unattended. It is the EXECUTABLE unit
// a mitigation loop consumes -- detection (Collect) answers "what work was lost?", the plan
// answers "run THIS to recover it", worst-first.
type CureAction struct {
	Class  Class  `json:"class"`
	Ref    string `json:"ref"`
	Detail string `json:"detail"`
	Weight int    `json:"weight"`
	Cure   string `json:"cure"`
	// Auto mirrors the class spec's AutoCure: true means a loop may dispatch Cure unattended
	// (it files an issue, no code rewrite); false means emit it as a work order for a human.
	Auto bool `json:"auto"`
}

// MitigationPlan turns detected findings into an ordered, worst-first list of routed cure
// actions -- the mitigation half of the loop made executable. A `fak loop`/cron tick calls
// Collect (detect) then MitigationPlan (mitigate): each action carries the exact cure command
// and whether it is auto-dispatchable, so the tick can run the safe ones and hand the rest off
// rather than re-deriving the remedy. It is PURE (findings in, plan out; no disk, no git, no
// clock) and re-sorts worst-first so a loop that affords one action per tick spends it on the
// heaviest loss. An unregistered class yields an empty Cure and Auto=false (fail-safe: never
// auto-run a cure we cannot name).
func MitigationPlan(findings []Finding) []CureAction {
	sorted := append([]Finding(nil), findings...)
	SortFindings(sorted)
	plan := make([]CureAction, 0, len(sorted))
	for _, f := range sorted {
		var cure string
		var auto bool
		if s, ok := SpecOf(f.Class); ok {
			cure, auto = s.Cure, s.AutoCure
		}
		plan = append(plan, CureAction{
			Class:  f.Class,
			Ref:    f.Ref,
			Detail: f.Detail,
			Weight: f.Weight,
			Cure:   cure,
			Auto:   auto,
		})
	}
	return plan
}

// Fold turns detected findings into the control-pane Payload via the shared scorecard
// kernel. It is PURE: findings + per-class universe counts in, payload out -- no disk, no
// git, no clock. universe[class] is the size of the scanned population for that class (so
// the legacy KPI score can be a real clean fraction); a class with an unknown or zero
// universe falls back to a monotone 100/(1+n) proxy, because the honest headline is the
// unbounded debt count, not the 0-100 score.
//
// One KPI is emitted per registered class (in registry order) even at zero findings, so
// the card always shows its full coverage surface rather than hiding a clean class.
func Fold(findings []Finding, universe map[Class]int) scorecard.Payload {
	SortFindings(findings)

	byClass := map[Class][]Finding{}
	for _, f := range findings {
		byClass[f.Class] = append(byClass[f.Class], f)
	}

	kpis := make([]scorecard.KPI, 0, len(registry))
	counts := map[Class]int{}
	for _, s := range registry {
		fs := byClass[s.Class]
		counts[s.Class] = len(fs)
		defects := make([]string, 0, len(fs))
		for _, f := range fs {
			defects = append(defects, f.line())
		}
		kpis = append(kpis, scorecard.KPI{
			Key:     kpiKey(s.Class),
			Group:   string(s.Group),
			Score:   classScore(len(fs), universe[s.Class]),
			Detail:  fmt.Sprintf("%s [%s]: %d finding(s)", s.Title, s.Rung.Label(), len(fs)),
			Defects: defects,
		})
	}

	total := len(findings)
	finding := "no anti-pattern findings: no redundant rework, no unwired package, no orphaned func in scope"
	next := "hold -- re-run after new commits land; a regression means work was redone or landed wired to nothing"
	if total > 0 {
		finding = fmt.Sprintf("%s across %s: %s",
			plural(total, "anti-pattern finding"), plural(coveredClasses(counts), "class"), topClassSummary(counts))
		next = "clear worst-first: route each finding to its class cure -- " + topCure(counts)
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"classes_registered": len(registry),
			"classes_with_debt":  coveredClasses(counts),
			"redundant_rework":   counts[ClassRedundantRework],
			"unwired_pkg":        counts[ClassUnwiredPkg],
			"orphan_func":        counts[ClassOrphanFunc],
			// The routed cure per registered class: the mitigation half of the loop. A
			// consumer (a `fak superloop` intent) reads this to turn each detected finding
			// into a concrete remediation action rather than re-deriving it.
			"cures": cureManifest(),
		},
	})
	return p
}

// Build reads the tree at root and folds the git-history window `commits` into the full
// anti-pattern payload. It is the one call the CLI makes. Filesystem reads (orphan + unwired
// scans) happen here, matching internal/unwiredscore.Build; git exec stays in the CLI, which
// passes the already-parsed commit window in as `commits` (keeping Build free of exec).
func Build(root string, commits []Commit) scorecard.Payload {
	findings, universe := Collect(root, commits)
	p := Fold(findings, universe)
	p.Workspace = root
	if p.Corpus == nil {
		p.Corpus = map[string]any{}
	}
	p.Corpus["commits_scanned"] = len(commits)
	return p
}

// --- pure helpers ---------------------------------------------------------------------------

// classScore maps a finding count to a legacy 0-100 KPI score. With a known universe it is
// the clean fraction (like unwiredscore's wired fraction); otherwise a monotone proxy that
// stays in (0,100] so more findings always read as a lower score. The debt count is the real
// signal -- this only feeds the legacy A-F grade.
func classScore(findings, universe int) float64 {
	if findings <= 0 {
		return 100.0
	}
	if universe > 0 {
		clean := universe - findings
		if clean < 0 {
			clean = 0
		}
		return 100.0 * float64(clean) / float64(universe)
	}
	return 100.0 / float64(1+findings)
}

func kpiKey(c Class) string {
	return "no_" + strings.ToLower(string(c))
}

// cureManifest returns the routed cure per registered class, in registry order, as the
// mitigation half of the loop a consumer reads. Keyed by class token so a finding maps
// straight to its remediation.
func cureManifest() map[string]string {
	m := make(map[string]string, len(registry))
	for _, s := range registry {
		m[string(s.Class)] = s.Cure
	}
	return m
}

// topCure names the cure for the worst class currently carrying debt (registry order among
// classes with findings), so the NextAction points at a concrete remediation, not just "fix
// it". Falls back to the first registered cure when nothing has debt (defensive; the total>0
// caller guarantees at least one).
func topCure(counts map[Class]int) string {
	for _, s := range registry {
		if counts[s.Class] > 0 {
			return string(s.Class) + " -> " + s.Cure
		}
	}
	if len(registry) > 0 {
		return string(registry[0].Class) + " -> " + registry[0].Cure
	}
	return ""
}

func coveredClasses(counts map[Class]int) int {
	n := 0
	for _, c := range counts {
		if c > 0 {
			n++
		}
	}
	return n
}

// topClassSummary renders the per-class counts of the classes that have debt, in registry
// order, e.g. "REDUNDANT_REWORK=2, ORPHAN_FUNC=5".
func topClassSummary(counts map[Class]int) string {
	var parts []string
	for _, s := range registry {
		if counts[s.Class] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", s.Class, counts[s.Class]))
		}
	}
	return strings.Join(parts, ", ")
}

func plural(n int, noun string) string {
	s := fmt.Sprintf("%d %s", n, noun)
	if n != 1 {
		s += "(s)"
	}
	return s
}
