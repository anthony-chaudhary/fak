// Package skillfootprint prices the resident `.claude/skills` description floor
// — the always-shipped USERLAND slice of the token tax epic #3229 is shrinking —
// and holds the one-way ratchet that keeps it from growing unopposed (#5444).
//
// WHY THIS PACKAGE EXISTS
// -----------------------
// #3234 landed the MEASUREMENT (`fak skill footprint`): each SKILL.md carries a
// frontmatter `description`, the skills index holds every one of them at rest, and
// the resident tax is therefore the SUM of those description fields — a number that
// grows linearly with skill count and that no single author ever sees grow. In the
// twenty days after #3234 closed, that measured floor grew 30.4% with nothing
// opposing it. A number that cannot REFUSE a change is taste, and taste lost.
//
// The MCP half of the same epic already has both halves: internal/mcpfootprint
// measures the always-sent tool schemas AND gates them (floorgate.go,
// descbudget.go). This package is the userland twin, and it deliberately holds the
// fold and the gate together so they can never become rival authorities on what
// "the resident floor" means:
//
//   - the FOLD (Fold / Measure) is what cmd/fak's `fak skill footprint` verb
//     renders, so the gated number is the same number the scorecard prints; and
//   - the GATE (descbudget.go) refuses growth and demands a banked win.
//
// PROVENANCE (Law A2 — every value carries its provenance)
// --------------------------------------------------------
// The floor is denominated in BYTES of frontmatter `description` text as parsed by
// internal/capindex's SkillResolver. That is fak's own MODEL of the resident index
// — the `interactive` profile #3234 defined — not a provider-billed count, and it
// must never be compared against one. Harness listings have been observed rendering
// project skills name-only while built-in skills carry their full descriptions, so
// the description floor is the tax fak's model says an interactive session holds,
// not a witnessed on-the-wire byte count. The ratchet is worth holding either way:
// it is fak's own committed scorecard, and pinning today's floor is what turns a
// later trim into a bankable win instead of headroom for the next skill.
package skillfootprint

import (
	"path/filepath"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// Entry is one skill's resident cost row. The JSON tags are the shipped
// `fak skill footprint --json` payload (schema fak-skill-footprint/1) — renaming
// one is a breaking change to that contract, not a refactor.
type Entry struct {
	Kind      string `json:"kind"`
	Name      string `json:"name"`
	Version   string `json:"version"`
	Digest    string `json:"digest"`
	CardBytes   int `json:"card_bytes"`        // fak's at-rest serialized card
	DescBytes   int `json:"description_bytes"` // the full description: the ranking key (#3234)
	NameBytes   int `json:"name_bytes"`        // the name-only resident slice (#3612)
	IntentBytes int `json:"intent_bytes"`      // the resident one-line intent slice (#5560)
}

// Floor is the whole-catalog resident floor: per-skill rows plus the totals — the
// description floor (the `interactive` profile, #3234), the name-only floor (the
// `headless` profile, #3612), the one-line intent floor (#5560), and fak's own
// at-rest card floor.
type Floor struct {
	Entries   []Entry
	DescFloor int // interactive resident floor: sum of description bytes (#3234)
	NameFloor int // headless resident floor: sum of name bytes only (#3612)
	// IntentFloor is the sum of the resident one-line INTENT slices (#5560): what a
	// card now carries at rest to decide whether to load a skill, after the residency
	// split moved the full description out of CardBytes and into the in-process
	// ranking key. DescFloor stays the gated number — see descbudget.go — because it
	// is the frontmatter prose the #5444 ratchet exists to refuse the growth of, and
	// pricing the ratchet on a DERIVED leading sentence would blind it to a
	// description that doubled in its tail.
	IntentFloor int
	CardFloor   int
	SkillCount  int
}

// SkillsDir is the skills directory under a repo root. It is the single place the
// `.claude/skills` layout is spelled, so the verb, the gate and the gate's real-tree
// test can never disagree about which tree is being priced.
func SkillsDir(root string) string { return filepath.Join(root, ".claude", "skills") }

// Fold folds at-rest capability cards into a resident-floor scorecard. It is pure
// (no filesystem, no clock, no network) and deterministic: entries sort by resident
// description bytes descending, then name, then kind — so the heaviest resident
// skills lead and equal-weight rows have a stable order.
//
// The three floors are exact BYTE sums, never per-row rounded quantities, so the
// total is a faithful partition of the rows: sum(Entries[i].DescBytes) == DescFloor.
func Fold(cards []capindex.CapCard) Floor {
	fp := Floor{SkillCount: len(cards), Entries: make([]Entry, 0, len(cards))}
	for _, c := range cards {
		desc := len(c.Trigger)
		name := len(c.Ref.Name)
		intent := len(c.Intent)
		cb := len(c.CardBytes)
		fp.DescFloor += desc
		fp.NameFloor += name
		fp.IntentFloor += intent
		fp.CardFloor += cb
		fp.Entries = append(fp.Entries, Entry{
			Kind:        string(c.Ref.Kind),
			Name:        c.Ref.Name,
			Version:     c.Ref.Version,
			Digest:      c.Digest,
			CardBytes:   cb,
			DescBytes:   desc,
			NameBytes:   name,
			IntentBytes: intent,
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

// Measure prices the REAL `.claude/skills` tree under root, through the same
// shipped SkillResolver.Index() cards `fak skill footprint` reads — so the gate is
// never measuring a private re-parse of the frontmatter.
//
// A root with no skills directory folds to a zero Floor rather than an error: the
// gate below treats that as a refusal (fail closed), which is the honest reading of
// "I measured nothing", and is strictly better than a gate that greens on it.
func Measure(root string) Floor {
	return Fold(capindex.NewSkillResolver(SkillsDir(root)).Index())
}
