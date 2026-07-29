package main

// fak footprint --doc — the INSTRUCTION-PULLED floor scorecard (issue #5445,
// epic #3229).
//
// #3230's `fak footprint` and #3234's `fak skill footprint` both price a
// *resident* floor: bytes the harness places in context before turn 1, visible to
// `/context`. Neither can see the other half of the per-agent floor. `CLAUDE.md`
// is resident and lean (2.2 kB), but it *instructs* the agent to read `AGENTS.md`
// (66.9 kB); every agent that obeys pays that as a turn-1 `Read`. Because the
// bytes are pulled by an instruction rather than seated in the system prompt they
// appear in NEITHER `/context` NOR `fak footprint` — a floor in effect but not in
// form. This verb names and prices that category.
//
// The unit of the report is the markdown SECTION, because the lever the epic wants
// is "page this subsection out to a queryable store", and a section is the
// smallest thing you can relocate without breaking a link. Pricing is
// deterministic and offline: every token number goes through
// agent.RequestFootprint (the same char-walk and the same ~4-byte/token divisor as
// EstimateAnthropicTokens), so a doc floor can never drift from the estimator the
// gateway and #3230 already use.
//
// The committed baseline lives in docs/context-budget/agents-md-floor.md; the
// witness is TestDocFootprintPartitionsFile / TestDocFootprintAGENTSMD.

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// docSectionEntry is one heading's slice of an instruction-pulled doc.
//
// Bytes is INCLUSIVE (the heading line, its prose, and every nested subsection) —
// that is the quantity a "move this section out" lever actually removes. OwnBytes
// excludes nested subsections, so the OwnBytes of every section plus the preamble
// partition the file exactly; Bytes double-counts by design and must never be
// summed across levels.
type docSectionEntry struct {
	Level     int     `json:"level"` // 1 for `#`, 2 for `##`, …
	Title     string  `json:"title"`
	Line      int     `json:"line"` // 1-based line number of the heading
	Bytes     int     `json:"bytes"`
	Tokens    int     `json:"tokens"`
	OwnBytes  int     `json:"own_bytes"`
	OwnTokens int     `json:"own_tokens"`
	Pct       float64 `json:"pct_of_file"` // Bytes as a share of the whole file
}

// docFootprint is one instruction-pulled doc's per-section inventory.
type docFootprint struct {
	Path          string
	Bytes         int
	Tokens        int
	PreambleBytes int // bytes before the first heading (frontmatter, lede)
	Sections      []docSectionEntry
}

// docEstTokens prices a byte span through the house estimator. It routes the text
// through agent.RequestFootprint rather than dividing by a locally-copied constant
// so this verb can never drift from EstimateAnthropicTokens — the same discipline
// internal/mcpfootprint.Price applies to the tool-schema floor.
func docEstTokens(s string) int {
	return agent.RequestFootprint(&agent.AnthropicMessagesRequest{System: s}).System.Tokens
}

// docHeadingLevel returns the ATX heading level of a line, or 0 if it is not a
// heading. Setext underlines are deliberately not recognised: this repo's docs are
// ATX-only, and a `---` underline is ambiguous with YAML frontmatter.
func docHeadingLevel(line string) int {
	n := 0
	for n < len(line) && line[n] == '#' {
		n++
	}
	if n == 0 || n > 6 || n >= len(line) {
		return 0
	}
	if line[n] != ' ' && line[n] != '\t' {
		return 0
	}
	return n
}

// computeDocFootprint folds a markdown source into a per-section inventory. It is
// pure (no filesystem) and deterministic.
//
// Fenced code blocks are tracked so a shell comment at column 0 inside a ``` block
// is never mistaken for a heading — AGENTS.md is command-dense, so this is the
// difference between a real inventory and a garbage one.
func computeDocFootprint(path, src string) docFootprint {
	fp := docFootprint{Path: path, Bytes: len(src), Tokens: docEstTokens(src)}

	type mark struct {
		level  int
		title  string
		line   int
		offset int
	}
	var marks []mark

	offset, lineNo, inFence := 0, 0, false
	fenceMark := ""
	for _, line := range strings.SplitAfter(src, "\n") {
		if line == "" { // SplitAfter yields a trailing "" only when src ends in \n
			break
		}
		lineNo++
		body := strings.TrimRight(line, "\r\n")
		trimmed := strings.TrimLeft(body, " ")
		switch {
		case inFence:
			if strings.HasPrefix(trimmed, fenceMark) {
				inFence = false
			}
		case strings.HasPrefix(trimmed, "```"), strings.HasPrefix(trimmed, "~~~"):
			inFence, fenceMark = true, trimmed[:3]
		default:
			if lvl := docHeadingLevel(body); lvl > 0 {
				marks = append(marks, mark{
					level:  lvl,
					title:  strings.TrimSpace(strings.Trim(body[lvl:], "#")),
					line:   lineNo,
					offset: offset,
				})
			}
		}
		offset += len(line)
	}

	if len(marks) == 0 {
		fp.PreambleBytes = len(src)
		return fp
	}
	fp.PreambleBytes = marks[0].offset

	// OwnBytes is the gap to the NEXT heading of any level, so preamble + every
	// OwnBytes is an exact partition of the file.
	own := make([]int, len(marks))
	for i := range marks {
		end := len(src)
		if i+1 < len(marks) {
			end = marks[i+1].offset
		}
		own[i] = end - marks[i].offset
	}

	fp.Sections = make([]docSectionEntry, 0, len(marks))
	for i, m := range marks {
		// Inclusive extent: run forward while the following headings are deeper.
		end := len(src)
		for j := i + 1; j < len(marks); j++ {
			if marks[j].level <= m.level {
				end = marks[j].offset
				break
			}
		}
		inclusive := src[m.offset:end]
		e := docSectionEntry{
			Level:     m.level,
			Title:     m.title,
			Line:      m.line,
			Bytes:     len(inclusive),
			Tokens:    docEstTokens(inclusive),
			OwnBytes:  own[i],
			OwnTokens: docEstTokens(src[m.offset : m.offset+own[i]]),
		}
		if len(src) > 0 {
			e.Pct = float64(e.Bytes) * 100 / float64(len(src))
		}
		fp.Sections = append(fp.Sections, e)
	}
	return fp
}

// docSectionsByWeight ranks sections heaviest-first by inclusive bytes, ties broken
// by document order, so the "where do the bytes live" view is stable across runs.
func docSectionsByWeight(fp docFootprint) []docSectionEntry {
	out := append([]docSectionEntry(nil), fp.Sections...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Bytes != out[j].Bytes {
			return out[i].Bytes > out[j].Bytes
		}
		return out[i].Line < out[j].Line
	})
	return out
}

// resolveDocPath accepts a path relative to the caller's cwd or to the repo root,
// so the regeneration command in docs/context-budget/agents-md-floor.md works from
// anywhere in the tree.
func resolveDocPath(p string) string {
	if _, err := os.Stat(p); err == nil {
		return p
	}
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(repoRoot(), p)
}

func runFootprintDoc(out, errw io.Writer, path string, top int, asJSON bool) int {
	resolved := resolveDocPath(path)
	raw, err := os.ReadFile(resolved)
	if err != nil {
		fmt.Fprintf(errw, "footprint --doc: %v\n", err)
		return 1
	}
	fp := computeDocFootprint(filepath.ToSlash(path), string(raw))
	ranked := docSectionsByWeight(fp)

	limit := top
	if limit <= 0 || limit > len(ranked) {
		limit = len(ranked)
	}

	if asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":         "fak-doc-footprint/1",
			"provenance":     agent.FootprintProvenance,
			"kind":           "instruction-pulled",
			"path":           fp.Path,
			"bytes":          fp.Bytes,
			"tokens":         fp.Tokens,
			"preamble_bytes": fp.PreambleBytes,
			"section_count":  len(fp.Sections),
			"shown":          limit,
			"sections":       ranked[:limit],
		})
		return 0
	}

	fmt.Fprintf(out, "doc-footprint: %s · %d est. tokens (%d bytes, %s, instruction-pulled) · %d section(s)\n",
		fp.Path, fp.Tokens, fp.Bytes, agent.FootprintProvenance, len(fp.Sections))
	for _, s := range ranked[:limit] {
		fmt.Fprintf(out, "  %6d tok  %7d B  %5.1f%%  L%d %s%s\n",
			s.Tokens, s.Bytes, s.Pct, s.Level, strings.Repeat("  ", s.Level-1), s.Title)
	}
	if limit < len(ranked) {
		fmt.Fprintf(out, "  … and %d more section(s) not shown (--top %d)\n", len(ranked)-limit, limit)
	}
	fmt.Fprintf(out, "  %6d tok  %7d B          preamble (before the first heading)\n",
		docEstTokens(string(raw)[:fp.PreambleBytes]), fp.PreambleBytes)
	return 0
}
