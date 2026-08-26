// Package skilleffectiveness measures whether each Claude Code skill in
// `.claude/skills/*/SKILL.md` is BUILT to be effective -- discoverable (a
// frontmatter description), triggerable (an explicit "use when"/"use to"),
// affordable (per-load-tier word budgets), and reachable through the queried
// loader (a queryable, paging, in-sync capability catalog) -- and folds the gaps
// into the control-pane payload every fak scorecard emits.
//
// THE GAP THIS CLOSES
// -------------------
// A skill only helps if the model can FIND it and knows WHEN to load it. A skill
// with no frontmatter description is invisible to the catalog; one with no trigger
// phrase is discoverable but never fires; one whose always-resident metadata blows
// its word budget taxes every turn whether or not it is used; and a catalog whose
// cards do not page (the at-rest card carries the whole body) breaks the
// zero-cost-at-rest property the loader exists for. None of these fail loudly --
// they just quietly make the skill pack heavier and less useful, so they need a
// deterministic score rather than a periodic human read.
//
// It is a TREE-READING scorecard (no data dir, no clock, no network): every KPI is
// probed from the real `.claude/skills` tree and from the real capindex catalog the
// `fak skill` verbs drive, so a defect is retired only by fixing the skill or the
// loader -- never by editing a JSON file. The fold/grade/render/markdown/compare
// machinery lives in pkg/scorecard; this package holds only the probes and the KPIs.
//
// SHAPE (the propagation family convention, internal/propagationscore): the pure
// core lives here behind a thin cmd/fak shell (cmd/fak/skill_effectiveness.go),
// rides the shared pkg/scorecard kernel, and carries its own regression test.
//
// COMPATIBILITY PIN: corpus.skill_debt stays the AFFORDANCE dimension only
// (unreadable + missing description + missing trigger), exactly as the pre-kernel
// card published it and as .claude/skills/skill-score/SKILL.md tells an operator to
// record it. The kernel's Σ-defects total is published alongside it as
// corpus.total_debt, so no recorded baseline is silently re-based.
package skilleffectiveness

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	frontmatteryaml "github.com/anthony-chaudhary/fak/internal/frontmatter"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id.
const Schema = "fak-skill-effectiveness-scorecard/1"

// DebtKey is the headline HARD integer the control-pane folds (corpus.skill_debt).
const DebtKey = "skill_debt"

// TotalDebtKey is the kernel's Σ-defects integer, published alongside DebtKey so the
// affordance headline and the whole-card total are both legible.
const TotalDebtKey = "total_debt"

// Per-load-tier word budgets (#4056): the always-resident metadata tier (the
// frontmatter description) and the fault-on-demand body tier each carry a HARD
// word ceiling. Over-budget skill text is one debt unit per tier -- a KPI a skill
// cannot pass by carrying a well-formed trigger while bloating the window.
const (
	MetadataWordBudget = 100
	BodyWordBudget     = 5000
)

// Scanned is one skill's probed state. Readable=false means the SKILL.md could not be
// read at all; the affordance/budget fields are then meaningless and are not scored,
// matching the pre-kernel card (an unreadable file is one debt unit and nothing else).
type Scanned struct {
	Name           string // the skill directory name
	Path           string // absolute path to its SKILL.md
	Readable       bool
	HasDescription bool
	HasTrigger     bool
	MetaWords      int
	BodyWords      int
}

// OverMetadataBudget reports whether the always-resident metadata tier blows its ceiling.
func (s Scanned) OverMetadataBudget() bool { return s.Readable && s.MetaWords > MetadataWordBudget }

// OverBodyBudget reports whether the fault-on-demand body tier blows its ceiling.
func (s Scanned) OverBodyBudget() bool { return s.Readable && s.BodyWords > BodyWordBudget }

// Scan reads every `.claude/skills/*/SKILL.md` under root and probes its affordances.
// It is pure over the filesystem at root (no clock, no network) and returns the skills
// in glob order, which filepath.Glob already sorts -- so the fold is deterministic.
func Scan(root string) []Scanned {
	matches, _ := filepath.Glob(filepath.Join(root, ".claude", "skills", "*", "SKILL.md"))
	out := make([]Scanned, 0, len(matches))
	for _, path := range matches {
		s := Scanned{Name: filepath.Base(filepath.Dir(path)), Path: path}
		b, err := os.ReadFile(path)
		if err != nil {
			out = append(out, s)
			continue
		}
		text := string(b)
		s.Readable = true
		description, hasDescription := frontmatterDescription(text)
		s.HasDescription = hasDescription && strings.TrimSpace(description) != ""
		low := strings.ToLower(text)
		s.HasTrigger = hasTrigger(low)
		s.MetaWords, s.BodyWords = TierWordCounts(text)
		out = append(out, s)
	}
	return out
}

func hasTrigger(description string) bool {
	for _, phrase := range []string{
		"use when", "use to", "use for", "use after", "use before",
		"run when", "invoke when", "trigger when", "triggered when", "triggers when",
	} {
		if strings.Contains(description, phrase) {
			return true
		}
	}
	return false
}

// TierWordCounts splits a SKILL.md into its two load tiers and counts words in each: the
// frontmatter `description:` value (the always-resident metadata tier) and everything
// after the frontmatter fence (the fault-on-demand body tier). A file with no `---`
// frontmatter fence is treated as all body.
func frontmatterDescription(text string) (string, bool) {
	t := strings.ReplaceAll(text, "\r\n", "\n")
	if !strings.HasPrefix(t, "---\n") {
		return "", false
	}
	end := strings.Index(t[4:], "\n---")
	if end < 0 {
		return "", false
	}
	for _, line := range strings.Split(t[4:4+end], "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok || strings.TrimSpace(key) != "description" {
			continue
		}
		decoded, _ := frontmatteryaml.DecodeScalar(value)
		return decoded, true
	}
	return "", false
}

func TierWordCounts(text string) (metaWords, bodyWords int) {
	t := strings.ReplaceAll(text, "\r\n", "\n")
	body := t
	if strings.HasPrefix(t, "---\n") {
		if end := strings.Index(t[4:], "\n---"); end >= 0 {
			body = t[4+end+len("\n---"):]
			if description, ok := frontmatterDescription(t); ok {
				metaWords = len(strings.Fields(description))
			}
		}
	}
	bodyWords = len(strings.Fields(body))
	return
}

// Loader is the queried-loader dimension, scored against the real capindex catalog.
type Loader struct {
	Skills    int // skill dirs the catalog walked (the KPI denominator)
	Queryable int // dirs with no catalog card or an empty trigger -- the catalog cannot match them
	Pages     int // cards whose at-rest bytes meet/exceed the body -- the body leaked into the index
	InSync    int // digest mismatches + non-idempotent re-sync rows
}

// Debt is the loader dimension's debt total.
func (l Loader) Debt() int { return l.Queryable + l.Pages + l.InSync }

// ScanLoader scores the queried-loader dimension against .claude/skills:
//
//   - queryable: every skill directory has a catalog card with a non-empty trigger, so a
//     model-emitted intent can actually match it. A missing card or an empty trigger is one
//     debt unit (the catalog is un-queryable for it).
//   - pages: the at-rest card must not hold the full body -- the body faults lazily. A card
//     whose at-rest bytes meet or exceed the body is one debt unit (the body leaked into the
//     index, breaking 0-cost-at-rest).
//   - inSync: each card's digest matches a fresh hash of its disk content, and re-syncing an
//     unchanged catalog is idempotent (zero CRUD changes). A digest mismatch or a
//     non-idempotent re-sync is one debt unit per row.
//
// It builds the real capindex catalog (the C1 keystone), so the score reflects the same
// loader the `fak skill` verbs drive.
func ScanLoader(root string) Loader {
	dir := filepath.Join(root, ".claude", "skills")
	resolver := capindex.NewSkillResolver(dir)
	cards := resolver.Index()

	var l Loader
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		body, err := os.ReadFile(filepath.Join(dir, e.Name(), "SKILL.md"))
		if err != nil {
			continue // not a skill dir
		}
		l.Skills++
		wantDigest := capindex.Digest(body)
		var card capindex.CapCard
		found := false
		for _, c := range cards {
			if c.Digest == wantDigest {
				card, found = c, true
				break
			}
		}
		if !found || strings.TrimSpace(card.Trigger) == "" {
			l.Queryable++
		}
		if found && len(card.CardBytes) >= len(body) {
			l.Pages++
		}
		if found && card.Digest != wantDigest {
			l.InSync++
		}
	}

	// Index-level in-sync: re-syncing an unchanged catalog must be idempotent. The first
	// Sync seeds the index (all-added); the second must report zero changes -- non-
	// deterministic digests or a broken hash-diff would surface spurious rows here.
	cat := capindex.NewCatalog()
	cat.AddResolver(capindex.CapKindSkill, resolver)
	cat.Sync()
	if changes := cat.Sync(); len(changes) != 0 {
		l.InSync += len(changes)
	}
	return l
}

// countKPI builds a "N of M satisfied" KPI: score is the satisfied share, and each
// unsatisfied item contributes one HARD defect. An empty corpus scores 100 (nothing to
// fix), never 0, so an empty tree reads clean rather than catastrophically broken.
func countKPI(key, group string, satisfied, total int, detail string, defects []string) scorecard.KPI {
	if satisfied < 0 {
		satisfied = 0 // the index-level in-sync check can exceed the per-skill denominator
	}
	score := 100.0
	if total > 0 {
		score = 100.0 * float64(satisfied) / float64(total)
	}
	return scorecard.KPI{
		Key: key, Group: group, Score: score,
		Detail:  fmt.Sprintf("%d/%d %s", satisfied, total, detail),
		Defects: defects,
	}
}

// KPIs scores the eight dimensions from an already-probed tree. Splitting it out of Build
// keeps the fold testable against a synthetic scan with no filesystem at all.
func KPIs(skills []Scanned, l Loader) []scorecard.KPI {
	readable := 0
	var unreadable, noDesc, noTrigger, overMeta, overBody []string
	for _, s := range skills {
		if !s.Readable {
			unreadable = append(unreadable, fmt.Sprintf("skill %q: SKILL.md could not be read (%s) -- an unreadable skill is invisible to the loader", s.Name, s.Path))
			continue
		}
		readable++
		if !s.HasDescription {
			noDesc = append(noDesc, fmt.Sprintf("skill %q: no `description:` in the frontmatter -- the catalog has nothing to match an intent against", s.Name))
		}
		if !s.HasTrigger {
			noTrigger = append(noTrigger, fmt.Sprintf("skill %q: no \"use when\"/\"use to\" trigger phrase -- discoverable but it never fires", s.Name))
		}
		if s.OverMetadataBudget() {
			overMeta = append(overMeta, fmt.Sprintf("skill %q: %d-word description exceeds the %d-word always-resident metadata budget -- it taxes every turn whether or not the skill loads", s.Name, s.MetaWords, MetadataWordBudget))
		}
		if s.OverBodyBudget() {
			overBody = append(overBody, fmt.Sprintf("skill %q: %d-word body exceeds the %d-word fault-on-demand body budget -- loading it floods the window", s.Name, s.BodyWords, BodyWordBudget))
		}
	}

	return []scorecard.KPI{
		countKPI("skill_readable", "trust", len(skills)-len(unreadable), len(skills),
			"discovered SKILL.md files are readable", unreadable),
		countKPI("skill_description", "discover", readable-len(noDesc), readable,
			"skills carry a frontmatter description", noDesc),
		countKPI("skill_trigger", "discover", readable-len(noTrigger), readable,
			"skills carry an explicit use-when trigger", noTrigger),
		countKPI("metadata_budget", "economy", readable-len(overMeta), readable,
			"skills fit the always-resident metadata word budget", overMeta),
		countKPI("body_budget", "economy", readable-len(overBody), readable,
			"skills fit the fault-on-demand body word budget", overBody),
		countKPI("loader_queryable", "operate", l.Skills-l.Queryable, l.Skills,
			"catalog cards are queryable (present, non-empty trigger)", loaderDefects(l.Queryable,
				"the loader catalog has no card (or an empty trigger) for a skill on disk -- a model-emitted intent cannot reach it")),
		countKPI("loader_paging", "operate", l.Skills-l.Pages, l.Skills,
			"catalog cards page (the body faults lazily)", loaderDefects(l.Pages,
				"a catalog card's at-rest bytes meet or exceed its body -- the body leaked into the index, breaking 0-cost-at-rest")),
		countKPI("loader_in_sync", "trust", l.Skills-l.InSync, l.Skills,
			"catalog rows are in sync with disk", loaderDefects(l.InSync,
				"a catalog row's digest disagrees with a fresh hash of its disk content, or re-syncing an unchanged catalog was not idempotent")),
	}
}

// loaderDefects expands a counted loader gap into n identical, individually-countable
// defect lines. The capindex probe yields a COUNT (it matches cards by digest, so it
// cannot name the offending skill without re-deriving the match), and the kernel counts
// debt as len(Defects) -- so one line per unit keeps corpus debt exact.
func loaderDefects(n int, msg string) []string {
	if n <= 0 {
		return nil
	}
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, fmt.Sprintf("%s (%d of %d)", msg, i+1, n))
	}
	return out
}

// AffordanceDebt is the corpus.skill_debt headline: the affordance dimension only
// (unreadable + missing description + missing trigger), pinned to the value the
// pre-kernel card published so recorded baselines stay comparable.
func AffordanceDebt(skills []Scanned) int {
	n := 0
	for _, s := range skills {
		if !s.Readable {
			n++
			continue
		}
		if !s.HasDescription {
			n++
		}
		if !s.HasTrigger {
			n++
		}
	}
	return n
}

// BudgetDebt is the per-load-tier word-budget dimension total.
func BudgetDebt(skills []Scanned) (metadata, body int) {
	for _, s := range skills {
		if s.OverMetadataBudget() {
			metadata++
		}
		if s.OverBodyBudget() {
			body++
		}
	}
	return
}

// Build probes the skill tree and the loader catalog and folds the KPIs into the
// control-pane payload via the shared kernel. root is the repo root.
func Build(root string) scorecard.Payload {
	skills := Scan(root)
	loader := ScanLoader(root)
	return Fold(root, skills, loader)
}

// Fold turns an already-probed tree into the payload. It is separated from Build so a test
// can fold a synthetic corpus with no filesystem at all.
func Fold(root string, skills []Scanned, loader Loader) scorecard.Payload {
	kpis := KPIs(skills, loader)
	affordance := AffordanceDebt(skills)
	metaOver, bodyOver := BudgetDebt(skills)
	budget := metaOver + bodyOver
	loaderDebt := loader.Debt()

	total := 0
	for _, k := range kpis {
		total += len(k.Defects)
	}

	finding := "skills_effective"
	reason := "all discovered skills carry the minimal trigger affordances and the loader index is queryable, paged, and in sync"
	next := "rerun after changing .claude/skills"
	if total > 0 {
		finding = "skill_debt"
		reason = fmt.Sprintf("%d skill affordance + %d loader debt unit(s)", affordance, loaderDebt)
		if budget > 0 {
			reason += fmt.Sprintf(" + %d word-budget violation(s)", budget)
		}
		next = "add missing front-matter descriptions/triggers, re-sync the loader index, or trim over-budget skill text"
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		Reason:          reason,
		ExtraCorpus: map[string]any{
			// The published affordance headline, deliberately overriding the kernel's
			// Σ-defects value for DebtKey (see the package doc's COMPATIBILITY PIN).
			DebtKey:            affordance,
			TotalDebtKey:       total,
			"loader_debt":      loaderDebt,
			"loader_queryable": loader.Queryable,
			"loader_pages":     loader.Pages,
			"loader_in_sync":   loader.InSync,
			"metadata_budget":  metaOver,
			"body_budget":      bodyOver,
			"skills":           len(skills),
		},
	})
	p.Workspace = root
	return p
}

// MarkdownDoc is the committed-snapshot front matter for `--markdown`, kept beside the
// KPIs so the published page and the card never drift.
func MarkdownDoc(p scorecard.Payload) scorecard.MarkdownDoc {
	return scorecard.MarkdownDoc{
		Title: "fak skill-effectiveness scorecard - is each skill built to be effective",
		Description: "fak's deterministic skill-effectiveness scorecard: every .claude/skills SKILL.md probed for " +
			"the affordances that make it discoverable (a frontmatter description), triggerable (an explicit " +
			"use-when phrase), and affordable (per-load-tier word budgets), plus the queried loader's own " +
			"queryable/paging/in-sync health - folded into the skill_debt headline.",
		Heading: "Skill-effectiveness scorecard",
		AutoGen: "Auto-generated by `fak skill-effectiveness-scorecard --markdown`. Do not hand-edit; re-run the tool.",
		Law: "The law: a skill only helps if the model can FIND it and knows WHEN to load it. Every skill " +
			"carries a frontmatter description and an explicit trigger, fits its per-load-tier word budget, " +
			"and resolves to a queryable catalog card whose body faults lazily and whose digest matches disk.",
		DebtKey: DebtKey,
		HeaderExtra: fmt.Sprintf(" - %v skill(s) - %v loader debt - %v total debt",
			p.Corpus["skills"], p.Corpus["loader_debt"], p.Corpus[TotalDebtKey]),
	}
}
