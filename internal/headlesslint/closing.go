package headlesslint

// closing.go — the CLOSING-SHAPE dual of leftovers.go.
//
// leftovers.go asks whether a run FILED the follow-ups it narrated. This asks a
// different run-level question about the SAME final summary: does it CLOSE in a shape
// the operator can scan? AGENTS.md makes the rule explicit —
//
//	Close operator-facing turns with verdict-first bullets, one claim and inline
//	evidence per line; make the final line the next checkable step.
//
// A turn that ends on a trailing multi-sentence prose paragraph buries the verdict and
// the next step in text the operator has to re-read to act on. ScanClosing is the fold
// that catches it: it isolates the LAST non-empty block of a final summary and refuses
// when that block is a prose wall — a long, multi-sentence paragraph with no list
// marker — so the closing can be re-cast as bullets. A short single-line closer
// ("nothing left; pushed abc123") and a block that already leads with a bullet are the
// honest scannable shapes, so both pass clean.
//
// Pure and stdlib-only, like every fold in this package: (final summary, operator
// override) in, a typed ClosingReport out. Any layer — the Stop-hook guard rung,
// `fak headless-lint --closing`, a loop gate — binds an agent's closing to the doctrine
// through the same taxonomy.

import (
	"regexp"
	"strings"
)

// ClosingDoctrine is the AGENTS.md rule this fold enforces, quoted verbatim so code and
// doctrine stay coupled — TestClosingDoctrineBindsAgentsMd asserts AGENTS.md still
// carries this exact line, so a reworded rule reds the fold's binding instead of
// silently drifting. It is the closing-shape twin of leftovers.go's Doctrine.
const ClosingDoctrine = "Close operator-facing turns with verdict-first bullets, one claim and inline evidence per line;\nmake the final line the next checkable step."

// ClosingSchema is the versioned envelope tag for a ClosingReport.
const ClosingSchema = "fak-closing-fold/1"

// Closing verdicts — the closed top-level judgment of ScanClosing.
const (
	// ClosingClean: the final block already closes in a scannable shape — it leads
	// with a list marker, is a short single-line closer, or the summary is empty.
	ClosingClean = "clean"
	// ClosingProseWall: the final block is a long, multi-sentence prose paragraph with
	// no list marker — the verdict and next step are buried in text. The breach this
	// refuses.
	ClosingProseWall = "closing_prose_wall"
)

// closingProseWords is the length threshold: a trailing block with no list marker over
// this many words is a prose wall. Kept generous so an honest short completion never
// trips — the cited clean closers ("Done. Pushed abc123.", a three-sentence
// "Implemented… Pushed to main.") all sit well under it — while a real trailing
// paragraph, multi-sentence or single run-on, clears it. The gate ships OFF and soaks
// before it ever blocks, so a conservative single threshold is the right starting point.
const closingProseWords = 40

// listMarkerRE matches a line that opens a bullet / numbered / checkbox list item — the
// scannable closing shape the doctrine asks for. A block whose first line is a list item
// is a list, not a prose wall.
var listMarkerRE = regexp.MustCompile(`^\s*([-*+•‣◦]|\d+[.)]|[a-zA-Z][.)]|\[[ xX]\])\s+`)

// sentenceSplitRE counts sentence terminators (., !, ?) followed by whitespace or
// end-of-text — the "multi-sentence" arm of the prose-wall test.
var sentenceSplitRE = regexp.MustCompile(`[.!?]+(\s+|$)`)

// ClosingHit is the offending final block of a prose-wall closing.
type ClosingHit struct {
	Line      int    `json:"line"`
	Sentences int    `json:"sentences"`
	Words     int    `json:"words"`
	Excerpt   string `json:"excerpt"`
}

// ClosingReport is the fold over one final summary. Verdict is ClosingProseWall iff the
// last non-empty block is a prose wall and no operator escape was set; otherwise
// ClosingClean.
type ClosingReport struct {
	Schema     string      `json:"schema"`
	Verdict    string      `json:"verdict"`
	Doctrine   string      `json:"doctrine"`
	Overridden bool        `json:"overridden"`
	Hit        *ClosingHit `json:"hit,omitempty"`
	Resolve    string      `json:"resolve,omitempty"`
}

// Refused reports whether this summary closed on a prose wall — the arm a Stop-hook /
// guard gate blocks (or nudges) on. A clean report is never refused.
func (r ClosingReport) Refused() bool { return r.Verdict == ClosingProseWall }

// ScanClosing folds a final summary into a ClosingReport. It isolates the LAST non-empty
// block (the trailing paragraph the operator reads last) and refuses when that block is
// prose — over closingProseWords words with no leading list marker. override is the
// operator escape ("this prose closer is intentional"): it forces clean. A short
// single-line closer, a block that leads with a bullet, or an empty summary is clean.
func ScanClosing(summary string, override bool) ClosingReport {
	rep := ClosingReport{
		Schema:     ClosingSchema,
		Verdict:    ClosingClean,
		Doctrine:   ClosingDoctrine,
		Overridden: override,
	}
	block, startLine := lastBlock(summary)
	if len(block) == 0 {
		return rep
	}
	// A block that already leads with a list marker is the scannable shape — clean.
	if listMarkerRE.MatchString(block[0]) {
		return rep
	}
	text := strings.Join(block, " ")
	words := len(strings.Fields(text))
	if words > closingProseWords && !override {
		rep.Verdict = ClosingProseWall
		rep.Hit = &ClosingHit{
			Line:      startLine,
			Sentences: countSentences(text),
			Words:     words,
			Excerpt:   clip(text, 200),
		}
		rep.Resolve = "re-cast the closing as bullets, verdict first; put the next checkable step as the final bullet"
	}
	return rep
}

// lastBlock returns the final non-empty block of a summary (a run of consecutive
// non-blank lines, blank-line separated), each line trimmed, plus the 1-based line
// number the block starts at. The closing shape is a property of this trailing block,
// not the whole summary — so a summary with prose in the middle but a bulleted final
// block closes clean.
func lastBlock(summary string) (lines []string, startLine int) {
	all := splitLines(summary)
	end := len(all)
	for end > 0 && strings.TrimSpace(all[end-1]) == "" {
		end--
	}
	if end == 0 {
		return nil, 0
	}
	start := end
	for start > 0 && strings.TrimSpace(all[start-1]) != "" {
		start--
	}
	block := make([]string, 0, end-start)
	for _, l := range all[start:end] {
		block = append(block, strings.TrimSpace(l))
	}
	return block, start + 1
}

// countSentences counts sentence-terminated runs in a block. Any non-empty block holds
// at least one sentence even without a trailing period.
func countSentences(text string) int {
	if strings.TrimSpace(text) == "" {
		return 0
	}
	if n := len(sentenceSplitRE.FindAllString(text, -1)); n > 0 {
		return n
	}
	return 1
}
