// pacing.go — how long each line of the capture stays on screen.
//
// THE PROBLEM THIS REPLACES. The original timing walk had exactly three knobs:
// minFrame, maxGap and speed, applied uniformly to every chunk script(1)
// recorded. That maps the MACHINE's clock onto the screen, and the machine's
// clock is not a reader's. Measured on the shipped 117 s cut: 40.4% of the
// runtime was seven static cards, and the 35 proof steps inside the remaining
// 66.9 s got 1.91 s each — 0.94 s per command, counting its output and its
// return code. The two things the video exists to show are the two things it
// never held on, while it held 8 s on a card the viewer skimmed in 3 (#750).
//
// The knob could not be tuned out of that, because it is one-dimensional and
// the content is not: raising minFrame also slows the `why:` paragraph the
// capture prints three times verbatim.
//
// WHAT REPLACES IT. Every capture already marks its own structure in plain
// text — a step banner, a `$ ` command echo, a `[rc=n]` verdict — so the plan
// below reads that grammar and gives each class the dwell a reader actually
// needs for it:
//
//	step   a chapter boundary   held ALONE, before its body types
//	cmd    the command echo     a beat, so it is read before output floods
//	out    the command's output streams fast; batched, several lines per frame
//	rc     the verdict          held for the READING TIME of the output above it
//	emph   a money line         held longest; never skipped
//	prose  the author narrating held per block, not per line
//
// and skips what a reader gains nothing from: output a step has already shown
// verbatim, lines a skim rule names, and anything past a step's flow budget.
//
// ⛔ THE HONESTY CONSTRAINT. This re-times a capture; it never restates one.
// Every byte of the typescript is still written to the terminal in its recorded
// order — a "skipped" line is one that gets no frame OF ITS OWN, not one that
// is withheld; it is on screen in the next frame. Nothing is inserted, dropped,
// reordered or re-typed. What is no longer preserved is the wall-clock spacing,
// so the title card has to say so, and waitHint keeps a trace of it: a line the
// machine made you wait 5 s or more for still reads as a wait.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// ── config ───────────────────────────────────────────────────────────────────

// emphasisRule promotes any line matching Match to its own frame, held for
// Hold seconds and exempt from every skip rule. These are the lines the proof
// turns on: ADMITTED, REFUSED, a p50, a hash that had to match.
type emphasisRule struct {
	Match string  `json:"match"`
	Hold  float64 `json:"hold"`
	Note  string  `json:"note"`

	re *regexp.Regexp
}

type pacing struct {
	// Structure detection. Each is a regexp matched against one ANSI-stripped
	// line. Empty means "this capture has no such marker".
	StepPattern string `json:"stepPattern"`
	CmdPattern  string `json:"cmdPattern"`
	RCPattern   string `json:"rcPattern"`

	// Dwells, in seconds.
	StepHold     float64 `json:"stepHold"`     // a banner, held alone
	CmdHold      float64 `json:"cmdHold"`      // a `$ ` line, before its output
	RCHoldMin    float64 `json:"rcHoldMin"`    // floor on a verdict's hold
	RCHoldMax    float64 `json:"rcHoldMax"`    // ceiling, so a 40-line dump cannot stall
	LinesPerSec  float64 `json:"linesPerSec"`  // skim rate: output lines -> rc hold
	ProsePerLine float64 `json:"prosePerLine"` // narration, charged per non-blank line
	ProseMin     float64 `json:"proseMin"`
	ProseMax     float64 `json:"proseMax"`
	FlowFrame    float64 `json:"flowFrame"` // one batched frame of streaming output

	// StepMinSecs is a floor on a whole step, paid onto its last frame. A short
	// step is not a less important one — STEP 22 is three lines and is the
	// setup for the entire positive/negative arm — and a chapter you can land
	// in has to be worth landing in. The renderer ACHIEVES this floor; verify's
	// minStepSecs, set lower, is the independent assertion that it did.
	StepMinSecs float64 `json:"stepMinSecs"`

	// Batching and skipping.
	BulkLinesPerFrame int      `json:"bulkLinesPerFrame"` // output lines per streamed frame
	StepFlowBudget    float64  `json:"stepFlowBudget"`    // seconds of streaming per step, then jump
	DedupeRepeats     bool     `json:"dedupeRepeats"`     // a line this step already showed earns no frame
	Skim              []string `json:"skim"`              // output lines that never earn a frame

	// A trace of the machine's clock. A gap the capture recorded as >= WaitHintMin
	// gets WaitHintHold added, so `tb build`'s 49 s still reads as a wait.
	WaitHintMin  float64 `json:"waitHintMin"`
	WaitHintHold float64 `json:"waitHintHold"`

	// Cards reveal line by line rather than landing as a wall of text: the eye
	// gets an entry point and a direction. Blank lines divide concepts. The
	// active concept keeps its colour while each earlier one sits one step
	// further down the fade ramp, still legible as context;
	// CardConceptHold is the reading beat before focus moves on.
	CardReveal      float64 `json:"cardReveal"`
	CardConceptHold float64 `json:"cardConceptHold"`

	Emphasis []emphasisRule `json:"emphasis"`

	stepRE, cmdRE, rcRE *regexp.Regexp
	skimRE              []*regexp.Regexp
}

// defaults fills anything the config left at zero and compiles the patterns.
// A zero-valued pacing block therefore still produces a watchable render.
func (p *pacing) defaults() error {
	setf := func(dst *float64, v float64) {
		if *dst <= 0 {
			*dst = v
		}
	}
	setf(&p.StepHold, 1.6)
	setf(&p.CmdHold, 0.5)
	setf(&p.RCHoldMin, 0.55)
	setf(&p.RCHoldMax, 3.0)
	setf(&p.LinesPerSec, 11)
	setf(&p.ProsePerLine, 0.32)
	setf(&p.ProseMin, 0.5)
	setf(&p.ProseMax, 1.6)
	setf(&p.FlowFrame, 0.13)
	setf(&p.StepFlowBudget, 2.6)
	setf(&p.WaitHintMin, 5.0)
	setf(&p.WaitHintHold, 1.2)
	setf(&p.CardReveal, 0.32)
	setf(&p.CardConceptHold, 1.0)
	setf(&p.StepMinSecs, 2.6)
	if p.BulkLinesPerFrame <= 0 {
		p.BulkLinesPerFrame = 3
	}
	if p.StepPattern == "" {
		p.StepPattern = `^-- `
	}
	if p.CmdPattern == "" {
		p.CmdPattern = `^\$ `
	}
	if p.RCPattern == "" {
		p.RCPattern = `^\[rc=`
	}
	var err error
	if p.stepRE, err = regexp.Compile(p.StepPattern); err != nil {
		return fmt.Errorf("stepPattern: %w", err)
	}
	if p.cmdRE, err = regexp.Compile(p.CmdPattern); err != nil {
		return fmt.Errorf("cmdPattern: %w", err)
	}
	if p.rcRE, err = regexp.Compile(p.RCPattern); err != nil {
		return fmt.Errorf("rcPattern: %w", err)
	}
	p.skimRE = p.skimRE[:0]
	for _, s := range p.Skim {
		re, err := regexp.Compile(s)
		if err != nil {
			return fmt.Errorf("skim %q: %w", s, err)
		}
		p.skimRE = append(p.skimRE, re)
	}
	for i := range p.Emphasis {
		re, err := regexp.Compile(p.Emphasis[i].Match)
		if err != nil {
			return fmt.Errorf("emphasis %q: %w", p.Emphasis[i].Match, err)
		}
		p.Emphasis[i].re = re
		if p.Emphasis[i].Hold <= 0 {
			p.Emphasis[i].Hold = 2.4
		}
	}
	return nil
}

// ── the plan ─────────────────────────────────────────────────────────────────

// A unit is one atom of the render: bytes to write to the terminal, then either
// a frame held for Dwell seconds, or nothing at all (Emit false). Planning the
// whole segment before rendering a pixel is what lets a prose BLOCK be charged
// once instead of per line, and lets a step's flow budget be spent knowingly.
type unit struct {
	Raw   []byte
	Dwell float64
	Emit  bool
	Class string // step | cmd | out | rc | emph | prose | skip
	Line  string // the line that decided the class, ANSI-stripped and trimmed
}

// ansiRE matches the escape sequences the terminal emulator consumes. Stripping
// them is for CLASSIFICATION only — the raw bytes, colour included, are what
// gets written.
var ansiRE = regexp.MustCompile("\x1b\\[[0-9;?]*[ -/]*[@-~]")

func stripANSI(s string) string { return ansiRE.ReplaceAllString(s, "") }

// planSegment turns one script(1) capture into the unit list that renders it.
//
// body is the typescript with its "Script started on …" header already removed;
// chunks are the timing rows, whose byte counts index into body and whose
// delays are used only for waitHint.
func planSegment(body []byte, chunks [][2]float64, p *pacing) []unit {
	// Where each recorded chunk starts, so a line can find the largest wait that
	// happened anywhere inside it. The delay PRECEDES its chunk in this format.
	type mark struct {
		off   int
		delay float64
	}
	marks := make([]mark, 0, len(chunks))
	off := 0
	for _, c := range chunks {
		n := int(c[1])
		if off+n > len(body) {
			n = len(body) - off
		}
		if n <= 0 {
			continue
		}
		marks = append(marks, mark{off: off, delay: c[0]})
		off += n
	}
	mi := 0
	gapIn := func(lo, hi int) float64 {
		g := 0.0
		for mi < len(marks) && marks[mi].off < lo {
			mi++
		}
		for j := mi; j < len(marks) && marks[j].off < hi; j++ {
			if marks[j].delay > g {
				g = marks[j].delay
			}
		}
		return g
	}

	var (
		out []unit

		// pending group — consecutive lines that share one frame
		pend      []byte
		pendClass string
		pendLines int    // non-blank lines in the group, for prose reading time
		pendLine  string // the group's FIRST non-blank line, so a grouped frame
		//                  can name what put it on screen (#750 DoD B1)

		inOutput  bool                // between a `$ ` line and its `[rc=]`
		outLines  int                 // output lines since the last `$ `
		flowSpent float64             // streaming seconds spent in this step
		seen      = map[string]bool{} // output already shown verbatim in this step
	)

	clamp := func(v, lo, hi float64) float64 {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}

	flush := func() {
		if len(pend) == 0 {
			return
		}
		u := unit{Raw: pend, Class: pendClass, Emit: true, Line: pendLine}
		switch pendClass {
		case "prose":
			u.Dwell = clamp(float64(pendLines)*p.ProsePerLine, p.ProseMin, p.ProseMax)
		case "out":
			u.Dwell = p.FlowFrame
			// Once a step has spent its streaming budget the rest of its output
			// lands in one jump: written, never dwelt on. This is the skip.
			if flowSpent+u.Dwell > p.StepFlowBudget {
				u.Dwell, u.Emit, u.Class = 0, false, "skip"
			} else {
				flowSpent += u.Dwell
			}
		case "skip":
			u.Dwell, u.Emit = 0, false
		default:
			u.Dwell = p.FlowFrame
		}
		out = append(out, u)
		pend, pendClass, pendLines, pendLine = nil, "", 0, ""
	}

	// add appends a line to the pending group, flushing first when the group's
	// class changes or a bulk batch is full. line is the ANSI-stripped, trimmed
	// text: the group keeps its first non-blank one so the frame it becomes can
	// say which line put it on screen.
	add := func(raw []byte, class, line string, blank bool) {
		if pendClass != "" && pendClass != class {
			flush()
		}
		pendClass = class
		pend = append(pend, raw...)
		if !blank {
			pendLines++
			if pendLine == "" {
				pendLine = line
			}
		}
		if class == "out" && pendLines >= p.BulkLinesPerFrame {
			flush()
		}
	}

	// solo emits one line as its own frame, held for dwell.
	solo := func(raw []byte, class, line string, dwell float64) {
		flush()
		out = append(out, unit{Raw: raw, Dwell: dwell, Emit: true, Class: class, Line: line})
	}

	pos := 0
	for pos < len(body) {
		end := pos
		for end < len(body) && body[end] != '\n' {
			end++
		}
		if end < len(body) {
			end++ // keep the newline with its line
		}
		raw := body[pos:end]
		gap := gapIn(pos, end)
		pos = end

		line := strings.TrimRight(stripANSI(string(raw)), "\r\n")
		trimmed := strings.TrimSpace(line)
		hint := 0.0
		if gap >= p.WaitHintMin {
			hint = p.WaitHintHold
		}

		switch {
		case p.stepRE.MatchString(line):
			// A new chapter: the banner gets the screen to itself, and the
			// step-scoped state (flow budget, duplicate memory) starts over.
			solo(raw, "step", trimmed, p.StepHold+hint)
			inOutput, outLines, flowSpent = false, 0, 0
			seen = map[string]bool{}

		case p.cmdRE.MatchString(line):
			solo(raw, "cmd", trimmed, p.CmdHold+hint)
			inOutput, outLines = true, 0

		case p.rcRE.MatchString(line):
			// The verdict is where a reader stops, so it carries the reading
			// time for everything the command printed above it.
			solo(raw, "rc", trimmed, clamp(float64(outLines)/p.LinesPerSec, p.RCHoldMin, p.RCHoldMax)+hint)
			inOutput = false

		default:
			if r := p.matchEmphasis(line); r != nil {
				solo(raw, "emph", trimmed, r.Hold+hint)
				if inOutput {
					outLines++
				}
				continue
			}
			if inOutput {
				outLines++
				if p.skimmed(line) || (p.DedupeRepeats && trimmed != "" && seen[trimmed]) {
					add(raw, "skip", trimmed, trimmed == "")
					continue
				}
				if trimmed != "" {
					seen[trimmed] = true
				}
				add(raw, "out", trimmed, trimmed == "")
				continue
			}
			add(raw, "prose", trimmed, trimmed == "")
		}
	}
	flush()
	padSteps(out, p.StepMinSecs)
	return out
}

// padSteps pays any step that came in under the floor onto its LAST emitted
// frame — so the complete step, command and output and verdict together, sits
// on screen a moment longer before the next banner wipes the reader's context.
// Padding the banner instead would hold an empty screen.
func padSteps(out []unit, min float64) {
	if min <= 0 {
		return
	}
	pad := func(seg []unit) {
		if len(seg) == 0 || seg[0].Class != "step" {
			return
		}
		total, last := 0.0, -1
		for i := range seg {
			total += seg[i].Dwell
			if seg[i].Emit {
				last = i
			}
		}
		if last >= 0 && total < min {
			seg[last].Dwell += min - total
		}
	}
	start := -1
	for i := 0; i <= len(out); i++ {
		if i == len(out) || out[i].Class == "step" {
			if start >= 0 {
				pad(out[start:i])
			}
			start = i
		}
	}
}

func (p *pacing) matchEmphasis(line string) *emphasisRule {
	for i := range p.Emphasis {
		if p.Emphasis[i].re.MatchString(line) {
			return &p.Emphasis[i]
		}
	}
	return nil
}

func (p *pacing) skimmed(line string) bool {
	for _, re := range p.skimRE {
		if re.MatchString(line) {
			return true
		}
	}
	return false
}

// ── the timeline ─────────────────────────────────────────────────────────────

// A timeline is the render's own account of itself: where every frame started,
// what decided its dwell, and where each chapter begins. It is written out as
// timeline.json so the pacing claim is auditable rather than a matter of taste,
// and -verify reads it back to assert the floors this was built to hold.
type timeline struct {
	Now      float64            `json:"-"`
	TotalSec float64            `json:"totalSecs"`
	Frames   int                `json:"frames"`
	Skipped  int                `json:"skippedUnits"`
	Bars     int                `json:"barMarks"` // non-text marks DRAWN; see bar.go
	Log      []tlFrame          `json:"frameLog"`
	Segments []tlSeg            `json:"segments"`
	Steps    []tlStep           `json:"steps"`
	Chapters []tlChap           `json:"chapters"`
	Holds    []tlHold           `json:"notableHolds"`
	Counts   map[string]int     `json:"unitsByClass"`
	Secs     map[string]float64 `json:"secsByClass"`
	MinDwell map[string]float64 `json:"minDwellByClass"`
}

type tlSeg struct {
	Index int     `json:"index"`
	Kind  string  `json:"kind"`
	Start float64 `json:"startSecs"`
	Secs  float64 `json:"secs"`
	Title string  `json:"title,omitempty"`
}

type tlStep struct {
	Title  string  `json:"title"`
	Start  float64 `json:"startSecs"`
	Secs   float64 `json:"secs"`
	Frames int     `json:"frames"`
}

type tlChap struct {
	Start float64 `json:"startSecs"`
	End   float64 `json:"endSecs"`
	Title string  `json:"title"`
}

// tlFrame is one emitted frame, in the order it appears on screen: when it
// starts, how long it holds, which class decided that, and the line that put it
// there. #750's DoD asks for this per FRAME and the first cut shipped a count
// plus a filtered highlight list (notableHolds, gated on class and dwell >= 1s),
// which covered 25 of 511 frames — 4.9%. An audit artifact that can only account
// for a twentieth of the runtime is a sample, and the 95% it omits is exactly
// where a pacing regression would hide: nothing in `out`, `prose`, `card` or
// `step` was representable at all.
//
// ⭐ The point is not the bigger file, it is that Σ Secs must equal totalSecs,
// which is the number the ffmpeg encode is checked against (encode.go). So this
// log is cross-checkable against the shipped video rather than against itself.
type tlFrame struct {
	N     int     `json:"n"` // 1-based, in screen order
	Start float64 `json:"startSecs"`
	Secs  float64 `json:"secs"`
	Class string  `json:"class"`
	Step  int     `json:"step"`           // 1-based step this frame belongs to; 0 = before the first
	Line  string  `json:"line,omitempty"` // the trigger line, ANSI-stripped and trimmed
}

type tlHold struct {
	At    float64 `json:"atSecs"`
	Secs  float64 `json:"secs"`
	Class string  `json:"class"`
	Line  string  `json:"line"`
}

func newTimeline() *timeline {
	return &timeline{
		Counts:   map[string]int{},
		Secs:     map[string]float64{},
		MinDwell: map[string]float64{},
	}
}

// mark records one unit. dwell of 0 with emit=false is a skipped unit: it moves
// no clock and adds no frame, but it is counted so the report can say how much
// was skipped rather than quietly dropping it.
func (tl *timeline) mark(class, line string, dwell float64, emit bool) {
	tl.Counts[class]++
	if !emit {
		tl.Skipped++
		return
	}
	tl.Frames++
	tl.Secs[class] += dwell
	if d, ok := tl.MinDwell[class]; !ok || dwell < d {
		tl.MinDwell[class] = dwell
	}
	switch class {
	case "step":
		// Close the running step — but only if closeSegment has not already
		// closed it. Overwriting here is what made STEP 32 read as 35.8s: it is
		// the last step of the consumer capture, so the next banner it sees is
		// the first of Part D, four title cards later.
		if n := len(tl.Steps); n > 0 && tl.Steps[n-1].Secs == 0 {
			tl.Steps[n-1].Secs = tl.Now - tl.Steps[n-1].Start
		}
		tl.Steps = append(tl.Steps, tlStep{Title: line, Start: tl.Now})
	case "emph", "rc", "cmd":
		if dwell >= 1.0 {
			tl.Holds = append(tl.Holds, tlHold{At: tl.Now, Secs: dwell, Class: class, Line: line})
		}
	}
	if n := len(tl.Steps); n > 0 {
		tl.Steps[n-1].Frames++
	}
	// Every emitted frame, unconditionally — no class filter and no dwell
	// threshold. A row that is only written when it is interesting cannot show
	// that the boring ones were paced at all (#750 B1). Recorded AFTER the step
	// bookkeeping above so Step names the chapter this frame is inside.
	tl.Log = append(tl.Log, tlFrame{
		N: tl.Frames, Start: tl.Now, Secs: dwell, Class: class,
		Step: len(tl.Steps), Line: trunc(line, 120),
	})
	tl.Now += dwell
}

func (tl *timeline) openSegment(kind, title string) int {
	tl.Segments = append(tl.Segments, tlSeg{Index: len(tl.Segments), Kind: kind, Start: tl.Now, Title: title})
	return len(tl.Segments) - 1
}

// closeSegment also closes whatever step was still running. Without this the
// last step of a capture swallows every card that follows it before the next
// capture starts — which read as a 36.5s STEP 32 that is really 4.4s of step
// and 32s of title cards, and would have made the per-step floor meaningless.
func (tl *timeline) closeSegment(i int) {
	tl.Segments[i].Secs = tl.Now - tl.Segments[i].Start
	if n := len(tl.Steps); n > 0 && tl.Steps[n-1].Secs == 0 {
		tl.Steps[n-1].Secs = tl.Now - tl.Steps[n-1].Start
	}
}

// chapter opens a chapter at the current instant; the previous one is closed
// here. ffmpeg wants contiguous, non-overlapping chapters, so they are only
// ever closed by the next one or by finish().
func (tl *timeline) chapter(title string) {
	if n := len(tl.Chapters); n > 0 {
		tl.Chapters[n-1].End = tl.Now
	}
	tl.Chapters = append(tl.Chapters, tlChap{Start: tl.Now, Title: title})
}

func (tl *timeline) finish(total float64) {
	tl.TotalSec = total
	if n := len(tl.Steps); n > 0 && tl.Steps[n-1].Secs == 0 {
		tl.Steps[n-1].Secs = total - tl.Steps[n-1].Start
	}
	if n := len(tl.Chapters); n > 0 {
		tl.Chapters[n-1].End = total
	}
	for i := range tl.Chapters {
		if tl.Chapters[i].End <= tl.Chapters[i].Start {
			tl.Chapters[i].End = tl.Chapters[i].Start + 0.1
		}
	}
}

func (tl *timeline) writeJSON(path string) error {
	b, err := json.MarshalIndent(tl, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// writeChapters emits an ffmetadata file. Chapters are what turn a four-minute
// proof into something a viewer can enter in the middle: "STEP 26 . VERDICT"
// becomes a click in QuickTime or VLC instead of a hunt with the scrub bar.
func (tl *timeline) writeChapters(path string) error {
	var b strings.Builder
	b.WriteString(";FFMETADATA1\n")
	esc := func(s string) string {
		r := strings.NewReplacer("=", `\=`, ";", `\;`, "#", `\#`, `\`, `\\`, "\n", " ")
		return r.Replace(s)
	}
	for _, c := range tl.Chapters {
		fmt.Fprintf(&b, "\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=%d\nEND=%d\ntitle=%s\n",
			int64(c.Start*1000+0.5), int64(c.End*1000+0.5), esc(c.Title))
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}
