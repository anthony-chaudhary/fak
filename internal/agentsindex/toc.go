package agentsindex

// toc.go — the resident-TOC rendering + marker-splice for #3535. RenderTOC produces the
// compact, deterministic block that lives in the repo CLAUDE.md between the markers
// below, so every session holds a <=TOCBudgetTokens map of AGENTS.md instead of the
// ~10.4k-token forced full Read. SpliceResident regenerates that block in place;
// `fak index agents --write-resident` is the one writer, and a live-tree drift test
// re-derives the block from the real AGENTS.md and fails closed if it drifts.

import (
	"bytes"
	"fmt"
	"strings"
)

const (
	// TOCBudgetTokens caps the ESTIMATED size of the resident TOC block. The whole point
	// of the lever is that this ceiling is a fraction of the full-doc price; the resident
	// budget test fails closed if RenderTOC ever exceeds it (or returns zero sections).
	TOCBudgetTokens = 2000

	// ResidentBegin/ResidentEnd bound the generated block inside CLAUDE.md. SpliceResident
	// rewrites only the span between them; everything else in CLAUDE.md is preserved.
	ResidentBegin = "<!-- fak:agents-toc:begin -->"
	ResidentEnd   = "<!-- fak:agents-toc:end -->"
)

// RenderTOC returns the resident block body: a one-line orientation, then one title-only
// row per level>=2 section with its slug and ESTIMATED per-section price. Deterministic
// (source order), no prose summaries, so the block stays small and the drift gate can
// compare it byte-for-byte against the committed CLAUDE.md region.
func (d *Doc) RenderTOC() string {
	var b strings.Builder
	b.WriteString("**AGENTS.md** (~")
	fmt.Fprintf(&b, "%d est. tok total)", d.EstTokens())
	b.WriteString(" — fetch one section instead of reading the whole file:\n")
	b.WriteString("`fak index agents --section <slug>` · `fak index agents <query>` to rank · `fak index agents --full` for the whole doc.\n\n")
	for _, s := range d.Sections {
		indent := ""
		if s.Level > 2 {
			indent = strings.Repeat("  ", s.Level-2)
		}
		fmt.Fprintf(&b, "%s- `%s` — %s (~%d tok)\n", indent, s.Slug, s.Title, s.EstTokens)
	}
	return b.String()
}

// ResidentBlock is RenderTOC wrapped in the begin/end markers — the exact bytes
// SpliceResident writes into CLAUDE.md and the drift gate expects to find there.
func (d *Doc) ResidentBlock() string {
	return ResidentBegin + "\n" + d.RenderTOC() + ResidentEnd
}

// SpliceResident replaces the marker-bounded region of claudeMD with block (which must
// be a full ResidentBegin…ResidentEnd block, e.g. from ResidentBlock). It errors rather
// than guess if either marker is missing or out of order, so a malformed CLAUDE.md is
// never silently rewritten.
func SpliceResident(claudeMD []byte, block string) ([]byte, error) {
	bi := bytes.Index(claudeMD, []byte(ResidentBegin))
	if bi < 0 {
		return nil, fmt.Errorf("agentsindex: begin marker %q not found in CLAUDE.md", ResidentBegin)
	}
	ei := bytes.Index(claudeMD, []byte(ResidentEnd))
	if ei < 0 {
		return nil, fmt.Errorf("agentsindex: end marker %q not found in CLAUDE.md", ResidentEnd)
	}
	if ei < bi {
		return nil, fmt.Errorf("agentsindex: end marker precedes begin marker in CLAUDE.md")
	}
	tail := ei + len(ResidentEnd)

	var out bytes.Buffer
	out.Write(claudeMD[:bi])
	out.WriteString(block)
	out.Write(claudeMD[tail:])
	return out.Bytes(), nil
}

// ExtractResident returns the marker-bounded block currently in claudeMD (inclusive of
// the markers), for the drift gate to compare against a freshly rendered ResidentBlock.
func ExtractResident(claudeMD []byte) (string, bool) {
	bi := bytes.Index(claudeMD, []byte(ResidentBegin))
	if bi < 0 {
		return "", false
	}
	ei := bytes.Index(claudeMD, []byte(ResidentEnd))
	if ei < 0 || ei < bi {
		return "", false
	}
	return string(claudeMD[bi : ei+len(ResidentEnd)]), true
}
