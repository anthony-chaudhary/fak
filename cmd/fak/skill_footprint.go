package main

// fak skill footprint — the userland resident-floor scorecard for the
// .claude/skills index (issue #3234, epic #3229). The harness renders each
// SKILL.md's frontmatter `description` into the always-resident context, so the
// resident tax = the sum of those description fields and it grows linearly with
// skill count. This verb measures that floor deterministically and offline,
// reusing the shipped SkillResolver.Index() cards: it reports per-skill resident
// description bytes and at-rest card bytes, a floor total, and the top-N
// heaviest — the userland analog of #3230's systemic tool-schema scorecard.
//
// The `--profile` knob (issue #3612) models the two ways the index is shipped:
// `interactive` is #3234's name+description floor; `headless` is the name-only
// floor a single-issue `-p` dispatch worker needs (it rarely invokes a skill,
// so it pays for the names alone, not the descriptions). The name-only floor is
// strictly smaller whenever a description is non-empty, and the names survive —
// so a skill is still invocable by name from a headless worker. This measures
// the profile delta; wiring the harness to actually ship name-only for headless
// workers is a harness-side dependency (Claude Code renders the resident Skills
// slice itself and exposes no session `--settings` knob to trim it today).

import (
	"errors"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/skillfootprint"
)

// computeSkillFootprint folds the at-rest cards into a resident-floor scorecard.
//
// The fold itself lives in internal/skillfootprint, which also holds the #5444
// ratchet that gates the number. Keeping the two in one package is deliberate: the
// figure this verb prints and the figure the budget refuses on are then the same
// measurement by construction, and no second estimator can drift into existence.
func computeSkillFootprint(cards []capindex.CapCard) skillfootprint.Floor {
	return skillfootprint.Fold(cards)
}

// skillDescriptionBudgetStatus reports how the measured description floor sits
// against the committed #5444 ceiling: the refusal token when the ratchet would
// refuse, "ok" when it sits inside the band. The verb REPORTS the gate rather than
// enforcing it — enforcement is the package test over the real `.claude/skills`
// tree, so a scorecard run against an arbitrary tree (a fixture, another checkout)
// stays a measurement and never a refusal.
func skillDescriptionBudgetStatus(fp skillfootprint.Floor) string {
	err := skillfootprint.CheckDescriptions(fp)
	if err == nil {
		return "ok"
	}
	var sbe *skillfootprint.SkillDescBudgetError
	if errors.As(err, &sbe) {
		return sbe.Reason
	}
	return "refused"
}

func runSkillFootprint(out, errw io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak skill footprint", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	includeMCP := fs.Bool("mcp", false, "fold in the MCP-tool resolver")
	top := fs.Int("top", 5, "show the N heaviest skills (0 = all)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	profile := fs.String("profile", "interactive", "resident profile: 'interactive' (name+description floor, #3234) or 'headless' (name-only floor for a dispatch worker, #3612)")
	flagArgs, _ := partitionArgs(argv, map[string]bool{"top": true, "profile": true})
	if err := fs.Parse(flagArgs); err != nil {
		fmt.Fprintln(errw, err)
		return 2
	}

	// The resident floor the harness ships depends on the profile: interactive
	// ships name+description (#3234's DescFloor); a headless dispatch worker
	// ships name-only (#3612), so its floor is the sum of the skill names alone.
	var residentFloor int
	root := repoRoot()
	cards := catalogCards(root, *includeMCP)
	fp := computeSkillFootprint(cards)
	switch *profile {
	case "interactive", "":
		*profile = "interactive"
		residentFloor = fp.DescFloor
	case "headless":
		residentFloor = fp.NameFloor
	default:
		fmt.Fprintf(errw, "unknown --profile %q (want 'interactive' or 'headless')\n", *profile)
		return 2
	}

	limit := *top
	if limit <= 0 || limit > len(fp.Entries) {
		limit = len(fp.Entries)
	}
	heaviest := fp.Entries[:limit]

	if *asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":                  "fak-skill-footprint/1",
			"profile":                 *profile,
			"include_mcp":             *includeMCP,
			"skill_count":             fp.SkillCount,
			"description_floor_bytes": fp.DescFloor,
			"name_floor_bytes":        fp.NameFloor,
			"intent_floor_bytes":      fp.IntentFloor,
			"resident_floor_bytes":    residentFloor,
			"card_floor_bytes":        fp.CardFloor,
			"approx_tokens":           skillfootprint.ApproxTokens(residentFloor),
			// The #5444 ratchet, reported alongside the measurement it gates: the
			// committed ceiling and which way (if either) the floor has drifted out
			// of the band. Enforcement lives in the internal/skillfootprint test.
			"description_budget_bytes":  skillfootprint.SkillDescriptionBudgetBytes,
			"description_budget_status": skillDescriptionBudgetStatus(fp),
			"entries":                   fp.Entries,
			"heaviest":                  heaviest,
		})
		return 0
	}

	fmt.Fprintf(out, "skill footprint [%s]: %d skill(s); resident floor = %d bytes (~%d tokens); description floor = %d B; name-only floor = %d B; at-rest card floor = %d bytes; mcp=%v\n",
		*profile, fp.SkillCount, residentFloor, skillfootprint.ApproxTokens(residentFloor), fp.DescFloor, fp.NameFloor, fp.CardFloor, *includeMCP)
	fmt.Fprintf(out, "  description budget (#5444): %d B committed, measured %d B -> %s\n",
		skillfootprint.SkillDescriptionBudgetBytes, fp.DescFloor, skillDescriptionBudgetStatus(fp))
	// The #5560 residency split, reported next to the floor it moved: the at-rest card
	// now carries a capped one-line intent, not the full description prose.
	fmt.Fprintf(out, "  at-rest intent slice (#5560): %d B (~%d tokens) across %d skill(s); the full description stays the in-process ranking key and faults in with the body\n",
		fp.IntentFloor, skillfootprint.ApproxTokens(fp.IntentFloor), fp.SkillCount)
	if len(heaviest) == 0 {
		fmt.Fprintln(out, "  (no skills discovered under .claude/skills)")
		return 0
	}
	fmt.Fprintf(out, "  top %d heaviest (by resident description bytes):\n", len(heaviest))
	for i, e := range heaviest {
		fmt.Fprintf(out, "  %2d. %-28s desc=%5d B  card=%5d B  digest=%s\n",
			i+1, refLabel(capindex.CapRef{Name: e.Name, Version: e.Version}), e.DescBytes, e.CardBytes, shortDigest(e.Digest))
	}
	return 0
}
