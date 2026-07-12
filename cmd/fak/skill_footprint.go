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
	"flag"
	"fmt"
	"io"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// skillFootprintEntry is one skill's resident cost row.
type skillFootprintEntry struct {
	Kind      string `json:"kind"`
	Name      string `json:"name"`
	Version   string `json:"version"`
	Digest    string `json:"digest"`
	CardBytes int    `json:"card_bytes"`        // fak's at-rest serialized card
	DescBytes int    `json:"description_bytes"` // the harness-resident description slice
	NameBytes int    `json:"name_bytes"`        // the name-only resident slice (headless profile, #3612)
}

// skillFootprint is the whole-catalog resident floor: per-skill rows plus the
// two floor totals — the description floor (the harness `/context` Skills slice)
// and the card floor (fak's own at-rest index cost).
type skillFootprint struct {
	Entries    []skillFootprintEntry
	DescFloor  int // interactive resident floor: sum of description bytes (#3234)
	NameFloor  int // headless resident floor: sum of name bytes only (#3612)
	CardFloor  int
	SkillCount int
}

// computeSkillFootprint folds the at-rest cards into a resident-floor scorecard.
// It is pure (no filesystem) and deterministic: entries sort by resident
// description bytes descending, then name, then kind — so the heaviest resident
// skills lead and equal-weight rows have a stable order.
func computeSkillFootprint(cards []capindex.CapCard) skillFootprint {
	fp := skillFootprint{SkillCount: len(cards), Entries: make([]skillFootprintEntry, 0, len(cards))}
	for _, c := range cards {
		desc := len(c.Trigger)
		name := len(c.Ref.Name)
		cb := len(c.CardBytes)
		fp.DescFloor += desc
		fp.NameFloor += name
		fp.CardFloor += cb
		fp.Entries = append(fp.Entries, skillFootprintEntry{
			Kind:      string(c.Ref.Kind),
			Name:      c.Ref.Name,
			Version:   c.Ref.Version,
			Digest:    c.Digest,
			CardBytes: cb,
			DescBytes: desc,
			NameBytes: name,
		})
	}
	sort.Slice(fp.Entries, func(i, j int) bool {
		if fp.Entries[i].DescBytes != fp.Entries[j].DescBytes {
			return fp.Entries[i].DescBytes > fp.Entries[j].DescBytes
		}
		if fp.Entries[i].Name != fp.Entries[j].Name {
			return fp.Entries[i].Name < fp.Entries[j].Name
		}
		return fp.Entries[i].Kind < fp.Entries[j].Kind
	})
	return fp
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
			"resident_floor_bytes":    residentFloor,
			"card_floor_bytes":        fp.CardFloor,
			"approx_tokens":           residentFloor / 4, // ~4 bytes/token
			"entries":                 fp.Entries,
			"heaviest":                heaviest,
		})
		return 0
	}

	fmt.Fprintf(out, "skill footprint [%s]: %d skill(s); resident floor = %d bytes (~%d tokens); description floor = %d B; name-only floor = %d B; at-rest card floor = %d bytes; mcp=%v\n",
		*profile, fp.SkillCount, residentFloor, residentFloor/4, fp.DescFloor, fp.NameFloor, fp.CardFloor, *includeMCP)
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
