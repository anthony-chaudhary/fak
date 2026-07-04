package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// scorecardCmdSetup runs the shared front-half of a scorecard subcommand: it parses
// the common --json/--markdown flags, collects the payload, and on the --json path
// emits it and signals the caller to stop. It returns the payload p, its corpus map
// c, whether --markdown was requested, and done=true when the --json branch already
// rendered (the caller returns immediately). On a flag parse error it exits(2), the
// same as the inline form it replaces.
func scorecardCmdSetup(name string, argv []string, collect func(string) map[string]any) (p, c map[string]any, asMarkdown, done bool) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit machine-readable scorecard JSON")
	md := fs.Bool("markdown", false, "emit markdown")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	p = collect(repoRoot())
	if *asJSON {
		_ = writeIndentedJSONNoEscape(os.Stdout, p)
		return p, nil, false, true
	}
	return p, p["corpus"].(map[string]any), *md, false
}

func cmdSkillEffectivenessScorecard(argv []string) {
	p, c, asMarkdown, done := scorecardCmdSetup("fak skill-effectiveness-scorecard", argv, collectSkillEffectivenessScorecard)
	if done {
		return
	}
	if asMarkdown {
		fmt.Printf("# fak skill-effectiveness scorecard\n\n**skill_debt: %v · loader_debt: %v** across **%v** skills.\n", c["skill_debt"], c["loader_debt"], c["skills"])
		return
	}
	fmt.Printf("skill-effectiveness-scorecard: %s (%s)\n  skill_debt: %v   loader_debt: %v   skills: %v\n",
		p["verdict"], p["finding"], c["skill_debt"], c["loader_debt"], c["skills"])
}

func collectSkillEffectivenessScorecard(root string) map[string]any {
	matches, _ := filepath.Glob(filepath.Join(root, ".claude", "skills", "*", "SKILL.md"))
	debt := 0
	for _, path := range matches {
		b, err := os.ReadFile(path)
		if err != nil {
			debt++
			continue
		}
		text := string(b)
		if !strings.Contains(text, "description:") {
			debt++
		}
		if !strings.Contains(strings.ToLower(text), "use when") && !strings.Contains(strings.ToLower(text), "use to") {
			debt++
		}
	}

	// Loader dimension (C7 / #1110): is the catalog queryable? does it page? is
	// the index in sync? Each gap is one debt unit, same RSI discipline as the
	// affordance checks above.
	loaderDebt, queryable, pagesNot, inSync := collectLoaderDebt(root)

	totalDebt := debt + loaderDebt
	score := 100
	grade := "A"
	ok, verdict, finding := true, "OK", "skills_effective"
	reason := "all discovered skills carry the minimal trigger affordances and the loader index is queryable, paged, and in sync"
	next := "rerun after changing .claude/skills"
	if totalDebt > 0 {
		ok, verdict, finding = false, "ACTION", "skill_debt"
		score, grade = 85, "B"
		reason = fmt.Sprintf("%d skill affordance + %d loader debt unit(s)", debt, loaderDebt)
		next = "add missing front-matter descriptions/triggers, or re-sync the loader index"
	}
	return map[string]any{
		"schema":      "fak-skill-effectiveness-scorecard/1",
		"ok":          ok,
		"verdict":     verdict,
		"finding":     finding,
		"reason":      reason,
		"next_action": next,
		"corpus": map[string]any{
			"skill_debt":           debt,
			"loader_debt":          loaderDebt,
			"loader_queryable":     queryable,
			"loader_pages":         pagesNot,
			"loader_in_sync":       inSync,
			"skills":               len(matches),
			"score":                score,
			"grade":                grade,
		},
	}
}

// collectLoaderDebt scores the queried-loader dimension against .claude/skills:
//
//   - queryable: every skill directory has a catalog card with a non-empty
//     trigger, so a model-emitted intent can actually match it. A missing card
//     or an empty trigger is one debt unit (the catalog is un-queryable for it).
//   - pages: the at-rest card must not hold the full body — the body faults
//     lazily. A card whose at-rest bytes meet or exceed the body is one debt
//     unit (the body leaked into the index, breaking 0-cost-at-rest).
//   - inSync: each card's digest matches a fresh hash of its disk content, and
//     re-syncing an unchanged catalog is idempotent (zero CRUD changes). A
//     digest mismatch or a non-idempotent re-sync is one debt unit per row.
//
// It builds the real capindex catalog (the C1 keystone), so the score reflects
// the same loader the `fak skill` verbs drive.
func collectLoaderDebt(root string) (loaderDebt, queryable, pages, inSync int) {
	dir := filepath.Join(root, ".claude", "skills")
	resolver := capindex.NewSkillResolver(dir)
	cards := resolver.Index()

	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		body, err := os.ReadFile(filepath.Join(dir, e.Name(), "SKILL.md"))
		if err != nil {
			continue // not a skill dir
		}
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
			queryable++
		}
		if found && len(card.CardBytes) >= len(body) {
			pages++
		}
		if found && card.Digest != wantDigest {
			inSync++
		}
	}

	// Index-level in-sync: re-syncing an unchanged catalog must be idempotent.
	// The first Sync seeds the index (all-added); the second must report zero
	// changes — non-deterministic digests or a broken hash-diff would surface
	// spurious rows here.
	cat := capindex.NewCatalog()
	cat.AddResolver(capindex.CapKindSkill, resolver)
	cat.Sync()
	if changes := cat.Sync(); len(changes) != 0 {
		inSync += len(changes)
	}

	loaderDebt = queryable + pages + inSync
	return
}
