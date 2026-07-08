package main

// fak skill footprint — the userland resident-floor scorecard for the
// .claude/skills index (issue #3234, epic #3229). The harness renders each
// SKILL.md's frontmatter `description` into the always-resident context, so the
// resident tax = the sum of those description fields and it grows linearly with
// skill count. This verb measures that floor deterministically and offline,
// reusing the shipped SkillResolver.Index() cards: it reports per-skill resident
// description bytes and at-rest card bytes, a floor total, and the top-N
// heaviest — the userland analog of #3230's systemic tool-schema scorecard.

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
}

// skillFootprint is the whole-catalog resident floor: per-skill rows plus the
// two floor totals — the description floor (the harness `/context` Skills slice)
// and the card floor (fak's own at-rest index cost).
type skillFootprint struct {
	Entries    []skillFootprintEntry
	DescFloor  int
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
		cb := len(c.CardBytes)
		fp.DescFloor += desc
		fp.CardFloor += cb
		fp.Entries = append(fp.Entries, skillFootprintEntry{
			Kind:      string(c.Ref.Kind),
			Name:      c.Ref.Name,
			Version:   c.Ref.Version,
			Digest:    c.Digest,
			CardBytes: cb,
			DescBytes: desc,
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
	flagArgs, _ := partitionArgs(argv, map[string]bool{"top": true})
	if err := fs.Parse(flagArgs); err != nil {
		fmt.Fprintln(errw, err)
		return 2
	}

	root := repoRoot()
	cards := catalogCards(root, *includeMCP)
	fp := computeSkillFootprint(cards)

	limit := *top
	if limit <= 0 || limit > len(fp.Entries) {
		limit = len(fp.Entries)
	}
	heaviest := fp.Entries[:limit]

	if *asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":                  "fak-skill-footprint/1",
			"include_mcp":             *includeMCP,
			"skill_count":             fp.SkillCount,
			"description_floor_bytes": fp.DescFloor,
			"card_floor_bytes":        fp.CardFloor,
			"approx_tokens":           fp.DescFloor / 4, // ~4 bytes/token
			"entries":                 fp.Entries,
			"heaviest":                heaviest,
		})
		return 0
	}

	fmt.Fprintf(out, "skill footprint: %d skill(s); resident description floor = %d bytes (~%d tokens); at-rest card floor = %d bytes; mcp=%v\n",
		fp.SkillCount, fp.DescFloor, fp.DescFloor/4, fp.CardFloor, *includeMCP)
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
