package agentsindex

// bytefloor.go — the one-way byte ratchet on AGENTS.md itself (epic #3229; the
// instruction-pulled floor measured in docs/context-budget/agents-md-floor.md, #5445).
//
// CLAUDE.md tells every agent to read AGENTS.md first, so the whole file is a turn-1
// Read in every session. Its siblings in docs/context-budget are ratcheted
// (internal/mcpfootprint/floorgate.go for the MCP schema floor,
// internal/skillfootprint/descbudget.go for the skill-description floor, #5444).
// AGENTS.md was the largest slice and the only one without a ratchet: it was cut 3.9x
// on 2026-08-23 and regrew 2.4x in the following 34 days, one reasonable-looking
// paragraph at a time. This file is the refusal that makes that growth reviewable.
//
// It is the same two-sided ratchet as its siblings:
//
//   - GROWTH is refused (AGENTS_MD_FLOOR_EXCEEDED). The only way through is to
//     regenerate the baseline in the SAME commit, so the new per-session tax is a
//     reviewable diff line bound to the change that caused it.
//
//   - A BANKED WIN is required (AGENTS_MD_FLOOR_STALE). A trim larger than the slack
//     also reds the gate until the floor is re-pinned down; otherwise the recovered
//     bytes silently become headroom for the next paragraph.
//
// The committed floor is a counted baseline file (FloorBaselineFile), and it follows
// the counted-ratchet contract internal/promptlint/breath and internal/pythongate run:
//
//	(a) keys are KIND<TAB>path and stable under editing. A section is keyed by its
//	    heading slug, never its line number, so inserting a line renumbers nothing.
//	    Parse numbers repeated slugs by position (notes, notes-2), which WOULD shift
//	    keys, so MeasureFloor refuses two headings that share a slug;
//	(b) the baseline stores a COUNT (bytes) per key, so trimming a section and
//	    regenerating tightens the floor, while growth past the tightened count is
//	    still caught even though the key already exists;
//	(c) a row that does not parse is a HARD error naming its line. A lenient parser
//	    would read a mangled DOC_BYTES row as "no ceiling" (a silent pass) and a
//	    mangled SECTION_BYTES row as a false attribution in someone else's refusal;
//	(d) a green run claims only "AGENTS.md is not growing", never "small enough".
//
// Only DOC_BYTES gates. SECTION_BYTES rows record each heading's OWN bytes when the
// floor was pinned; they partition DOC_BYTES exactly (ParseFloorBaseline enforces it)
// and exist so a refusal can name where the bytes came back. Moving prose between
// sections is not growth and is not refused.
//
// DENOMINATION. Bytes of the LF-normalized file, which is the committed blob size
// (`git cat-file -s HEAD:AGENTS.md`) on every platform. Counting raw working-tree bytes
// would refuse a byte-identical commit on a CRLF checkout (one extra byte per line).

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// FloorBaselineFile is the committed, repo-relative counted floor. It sits beside the
// page that explains it so a reviewer reads the number and its rationale together.
const FloorBaselineFile = "docs/context-budget/agents-md-floor.tsv"

// FloorSlackBytes is how far AGENTS.md may sit BELOW its floor before the gate demands
// a re-pin. It absorbs a reworded sentence without nagging, and it is smaller than the
// smallest top-level Hard-rules bullet when the floor was pinned (229 B, 2026-09-28),
// so deleting any whole bullet cannot pass unbanked.
const FloorSlackBytes = 200

// RegenerateFloorCommand re-pins FloorBaselineFile from the live AGENTS.md. It is the
// one sanctioned writer; the refusal messages quote it verbatim.
const RegenerateFloorCommand = "go test ./internal/agentsindex -run TestAgentsMDByteFloorAtHEAD -update-agents-md-floor"

// Closed finding kinds of the counted baseline.
const (
	KindDocBytes     = "DOC_BYTES"     // the whole file; the only gated key
	KindSectionBytes = "SECTION_BYTES" // one heading's own bytes; attribution only
)

// Gate refusal reasons, in the same closed-vocabulary spirit as the sibling floors'
// FLOOR_BUDGET_* and SKILL_DESC_BUDGET_* tokens.
const (
	ReasonFloorExceeded = "AGENTS_MD_FLOOR_EXCEEDED"
	ReasonFloorStale    = "AGENTS_MD_FLOOR_STALE"
)

// preambleSlug keys the bytes before the first level>=2 heading (the title and the
// lede). Parentheses never survive slugify, so it cannot collide with a heading slug.
const preambleSlug = "(preamble)"

// maxRegrownShown caps the attribution list in a refusal; the rest are counted.
const maxRegrownShown = 5

// maxFloorCount bounds every baseline count (1 GiB), so a mangled row can neither
// overflow the partition sum nor pass as a plausible file size.
const maxFloorCount = 1 << 30

// FloorFinding is one counted measurement. Kind and Path form the key; Title and Line
// are display-only, so a heading that moves down the file keeps its key.
type FloorFinding struct {
	Kind  string
	Path  string // FileName for DOC_BYTES, FileName+"#"+slug for SECTION_BYTES
	Count int    // bytes
	Title string
	Line  int
}

// Key is the stable KIND<TAB>path identity the baseline counts against.
func (f FloorFinding) Key() string { return f.Kind + "\t" + f.Path }

// normalizeEOL folds CRLF to LF so the measurement equals the committed blob size.
func normalizeEOL(raw []byte) []byte {
	return bytes.ReplaceAll(raw, []byte("\r\n"), []byte("\n"))
}

// MeasureFloor folds AGENTS.md bytes into counted findings: DOC_BYTES first, then the
// preamble, then every level>=2 section in source order. It reuses Parse, so the gate's
// sections and slugs are exactly the ones `--section <slug>` would page, and the
// SECTION_BYTES counts partition DOC_BYTES exactly.
//
// It refuses two headings whose text slugs the same (see contract clause (a)). Their
// positional keys would shift when a third is added above them, and can collide
// outright, which would make a regenerated baseline fail its own parser.
func MeasureFloor(raw []byte) ([]FloorFinding, error) {
	norm := normalizeEOL(raw)
	doc := Parse(norm)
	offsets, _ := splitLines(norm)

	firstLine := map[string]int{}
	for _, s := range doc.Sections {
		base := slugify(slugHead(s.Title))
		if base == "" {
			base = "section"
		}
		if prev, dup := firstLine[base]; dup {
			return nil, fmt.Errorf("%s: headings on line %d and line %d both slug to %q; the byte floor keys sections by slug, "+
				"so give one a distinct heading (the text before any '(', ':' or dash)", FileName, prev, s.Line, base)
		}
		firstLine[base] = s.Line
	}

	out := []FloorFinding{{Kind: KindDocBytes, Path: FileName, Count: len(norm), Title: FileName}}
	starts := make([]int, len(doc.Sections))
	for i, s := range doc.Sections {
		starts[i] = offsets[s.Line-1]
	}
	preamble := len(norm)
	if len(starts) > 0 {
		preamble = starts[0]
	}
	out = append(out, FloorFinding{Kind: KindSectionBytes, Path: FileName + "#" + preambleSlug, Count: preamble, Title: "(preamble: title and lede)", Line: 1})
	for i, s := range doc.Sections {
		end := len(norm)
		if i+1 < len(starts) {
			end = starts[i+1]
		}
		out = append(out, FloorFinding{
			Kind:  KindSectionBytes,
			Path:  FileName + "#" + s.Slug,
			Count: end - starts[i],
			Title: s.Title,
			Line:  s.Line,
		})
	}
	return out, nil
}

// FloorBaseline is the committed floor: finding key -> pinned byte count.
type FloorBaseline map[string]int

// Ceiling is the gated DOC_BYTES floor.
func (b FloorBaseline) Ceiling() int { return b[KindDocBytes+"\t"+FileName] }

// floorBaselineHeader opens every regenerated baseline so the file says what it is.
const floorBaselineHeader = "# fak AGENTS.md byte floor (epic #3229). KIND<TAB>path<TAB>count, counts in bytes.\n" +
	"# Explained in docs/context-budget/agents-md-floor.md; gated by internal/agentsindex/bytefloor.go.\n" +
	"# DOC_BYTES is the ceiling: LF-normalized AGENTS.md bytes (== git cat-file -s HEAD:AGENTS.md).\n" +
	"# SECTION_BYTES rows are each heading's OWN bytes when the ceiling was pinned. They must sum\n" +
	"# to DOC_BYTES and only attribute a refusal to the sections that regrew.\n" +
	"# Regenerate in the same commit as the AGENTS.md change it justifies; never hand-edit one number:\n" +
	"#   " + RegenerateFloorCommand + "\n"

// FormatFloorBaseline renders findings as a regenerated baseline in measurement order
// (DOC_BYTES, preamble, then source order), so a re-pin diff reads like the file.
func FormatFloorBaseline(findings []FloorFinding) string {
	var b strings.Builder
	b.WriteString(floorBaselineHeader)
	for _, f := range findings {
		fmt.Fprintf(&b, "%s\t%s\t%d\n", f.Kind, f.Path, f.Count)
	}
	return b.String()
}

// ParseFloorBaseline reads a KIND<TAB>path<TAB>count baseline. Blank lines and `#`
// comments are skipped; every other line that does not parse is a hard error naming its
// 1-based line number (contract clause (c)). It also refuses a duplicate key, a missing
// DOC_BYTES row, and SECTION_BYTES rows that do not partition DOC_BYTES, because each of
// those would make the gate or its attribution quietly wrong.
func ParseFloorBaseline(r io.Reader) (FloorBaseline, error) {
	out := FloorBaseline{}
	firstLine := map[string]int{}
	docLine, sectionSum := 0, 0
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for n := 1; sc.Scan(); n++ {
		line := sc.Text()
		if t := strings.TrimSpace(line); t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		f := strings.Split(line, "\t")
		if len(f) != 3 {
			return nil, fmt.Errorf("agents-md floor baseline line %d: want 3 tab-separated fields "+
				"(KIND<TAB>path<TAB>count), got %d in %q", n, len(f), line)
		}
		kind, path, count := strings.TrimSpace(f[0]), strings.TrimSpace(f[1]), strings.TrimSpace(f[2])
		switch kind {
		case KindDocBytes:
			if path != FileName {
				return nil, fmt.Errorf("agents-md floor baseline line %d: %s path must be %q, got %q", n, kind, FileName, path)
			}
		case KindSectionBytes:
			slug, ok := strings.CutPrefix(path, FileName+"#")
			if !ok || slug == "" {
				return nil, fmt.Errorf("agents-md floor baseline line %d: %s path must be %q, got %q", n, kind, FileName+"#<slug>", path)
			}
			if slug != preambleSlug && slugify(slug) != slug {
				return nil, fmt.Errorf("agents-md floor baseline line %d: %q is not a heading slug MeasureFloor can produce "+
					"(want lowercase a-z0-9 and single dashes, e.g. %q)", n, slug, slugify(slug))
			}
		default:
			return nil, fmt.Errorf("agents-md floor baseline line %d: %q is not a floor finding kind (known: %s, %s) in %q",
				n, kind, KindDocBytes, KindSectionBytes, line)
		}
		c, err := strconv.Atoi(count)
		if err != nil || c < 0 {
			return nil, fmt.Errorf("agents-md floor baseline line %d: count %q is not a non-negative integer in %q", n, count, line)
		}
		if c > maxFloorCount {
			return nil, fmt.Errorf("agents-md floor baseline line %d: count %d exceeds %d bytes in %q", n, c, maxFloorCount, line)
		}
		key := kind + "\t" + path
		if prev, dup := firstLine[key]; dup {
			return nil, fmt.Errorf("agents-md floor baseline line %d: duplicate key %s %s (first on line %d)", n, kind, path, prev)
		}
		firstLine[key] = n
		out[key] = c
		if kind == KindDocBytes {
			docLine = n
		} else {
			sectionSum += c
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("agents-md floor baseline: read: %w", err)
	}
	if docLine == 0 {
		return nil, fmt.Errorf("agents-md floor baseline: no %s\t%s row; a baseline without a ceiling would admit any size",
			KindDocBytes, FileName)
	}
	if sectionSum != out.Ceiling() {
		return nil, fmt.Errorf("agents-md floor baseline line %d: %s is %d but the %s rows sum to %d; the rows must partition "+
			"the file, so regenerate rather than hand-editing one number: %s",
			docLine, KindDocBytes, out.Ceiling(), KindSectionBytes, sectionSum, RegenerateFloorCommand)
	}
	return out, nil
}

// SectionDelta is one section's movement against the pinned baseline.
type SectionDelta struct {
	Slug     string
	Title    string
	Line     int
	Pinned   int  // baseline own bytes (0 when New)
	Measured int  // current own bytes
	New      bool // heading absent from the baseline (added, or renamed from an old slug)
}

// Growth is the byte change against the pinned count.
func (d SectionDelta) Growth() int { return d.Measured - d.Pinned }

// FloorError is a structured AGENTS.md floor refusal. Callers branch on Reason.
type FloorError struct {
	Reason   string // ReasonFloorExceeded | ReasonFloorStale
	Measured int    // current DOC_BYTES
	Ceiling  int    // committed DOC_BYTES
	Slack    int
	Regrown  []SectionDelta // grown sections, largest growth first (EXCEEDED only)
}

func (e *FloorError) Error() string {
	var b strings.Builder
	switch e.Reason {
	case ReasonFloorExceeded:
		over := e.Measured - e.Ceiling
		fmt.Fprintf(&b, "%s: %s is %d bytes, %d bytes over its committed floor of %d. Every agent that obeys "+
			"CLAUDE.md reads this file whole on turn 1, so each byte is paid in every session "+
			"(`fak footprint --doc %s` prices it in est. tokens).\n",
			e.Reason, FileName, e.Measured, over, e.Ceiling, FileName)
		b.WriteString("  Sections most likely to have regrown (own bytes vs the pinned baseline):\n")
		shown := e.Regrown
		if len(shown) > maxRegrownShown {
			shown = shown[:maxRegrownShown]
		}
		for _, d := range shown {
			if d.New {
				fmt.Fprintf(&b, "    %+6d B  L%-4d %s  [new heading %q, not in the baseline; a rename shows here too]\n",
					d.Growth(), d.Line, d.Title, d.Slug)
				continue
			}
			fmt.Fprintf(&b, "    %+6d B  L%-4d %s  [%s: %d -> %d]\n", d.Growth(), d.Line, d.Title, d.Slug, d.Pinned, d.Measured)
		}
		if rest := len(e.Regrown) - len(shown); rest > 0 {
			fmt.Fprintf(&b, "    ... and %d more grown section(s)\n", rest)
		}
		fmt.Fprintf(&b, "  Fix, in order: (1) move the new prose one hop away (a linked doc, `dos man wedge <TOKEN>`, or a fak verb) "+
			"and leave a one-line pointer; (2) trim elsewhere in %s to pay for it; (3) only if every agent needs it on turn 1, "+
			"re-pin in the SAME commit with `%s` and update the figure in docs/context-budget/agents-md-floor.md.",
			FileName, RegenerateFloorCommand)
	case ReasonFloorStale:
		fmt.Fprintf(&b, "%s: %s fell to %d bytes, %d below its committed floor of %d (slack %d). A trim was won but never banked. "+
			"Re-pin in the same commit with `%s` and update the figure in docs/context-budget/agents-md-floor.md, so the ratchet "+
			"tightens and the recovered bytes cannot be silently refilled.",
			e.Reason, FileName, e.Measured, e.Ceiling-e.Measured, e.Ceiling, e.Slack, RegenerateFloorCommand)
	default:
		fmt.Fprintf(&b, "%s: %s is %d bytes vs floor %d", e.Reason, FileName, e.Measured, e.Ceiling)
	}
	return b.String()
}

// CheckFloor gates measured findings against the committed baseline. It returns nil
// when DOC_BYTES sits inside [Ceiling-FloorSlackBytes, Ceiling], and a *FloorError
// naming the direction otherwise. It fails closed: an empty or unreadable doc measures
// 0 bytes and refuses as STALE whatever the band says, rather than passing on "I
// measured nothing".
func CheckFloor(measured []FloorFinding, base FloorBaseline) error {
	return checkFloorAgainst(measured, base, FloorSlackBytes)
}

// checkFloorAgainst is CheckFloor with the slack injected so tests can pin the band
// edges at small, legible numbers.
func checkFloorAgainst(measured []FloorFinding, base FloorBaseline, slack int) error {
	total := 0
	for _, f := range measured {
		if f.Kind == KindDocBytes {
			total = f.Count
			break
		}
	}
	ceiling := base.Ceiling()
	switch {
	case total > ceiling:
		return &FloorError{Reason: ReasonFloorExceeded, Measured: total, Ceiling: ceiling, Slack: slack, Regrown: regrown(measured, base)}
	case total == 0 || total < ceiling-slack:
		return &FloorError{Reason: ReasonFloorStale, Measured: total, Ceiling: ceiling, Slack: slack}
	}
	return nil
}

// regrown lists the sections whose own bytes exceed their pinned count, largest growth
// first (ties in source order). Because the baseline partitions DOC_BYTES, the growths
// of every current and removed section sum to the total overage, so an EXCEEDED
// refusal always has at least one section to name.
func regrown(measured []FloorFinding, base FloorBaseline) []SectionDelta {
	var out []SectionDelta
	for _, f := range measured {
		if f.Kind != KindSectionBytes {
			continue
		}
		pinned, known := base[f.Key()]
		if f.Count <= pinned {
			continue
		}
		out = append(out, SectionDelta{
			Slug:     strings.TrimPrefix(f.Path, FileName+"#"),
			Title:    f.Title,
			Line:     f.Line,
			Pinned:   pinned,
			Measured: f.Count,
			New:      !known,
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Growth() > out[j].Growth() })
	return out
}
