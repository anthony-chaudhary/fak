package main

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Interactive tabbed views + mouse + a click-a-term glossary for the live `fak info` overlay.
//
// The overview block (info_visual.go / info_panels.go) shows every subsystem at once, degrading
// each to fit the pane. When an operator wants to DWELL on one subsystem — read every agent row,
// see every deny reason, inspect each seat — the height-degradation works against them. This
// layer adds a TAB BAR and per-subsystem focused VIEWS: press 1-5 (or click a tab, or arrow
// between them) to swap the body to Overview / Agents / Accounts+Nodes / Cache / Safety, each
// rendered in full without the overview's shrink. A '?' (or click on the tab bar's glossary
// toggle) opens a GLOSSARY: a row of clickable term chips over plain-words definitions, so a
// watcher who does not know what "prefix" or "set aside" means can click the word and read it.
//
// Everything here is PURE and TTY-free: the byte-stream input scanner, the view-state reducer,
// the click hit-test, and the renderers all take values and return values, so the whole
// interaction is unit-testable without a terminal. The thin plumbing (raw stdin, the DECSET
// mouse/focus toggles, the select loop) lives in runGuardInfoOverlay (info.go) and is gated to
// an interactive visual-mode TTY, so the non-TTY / piped / --once / --style line paths never see
// a tab bar and stay byte-for-byte unchanged. The rendered rows carry only the same payload-free
// /debug/vars projection the overview does — no prompt/result text ever crosses.

// infoView is the pane's active view. Overview is the stacked-panels default (the existing
// composed block); the rest are single-subsystem focused views that trade the overview's
// height-degradation for room to show one subsystem in full.
type infoView int

const (
	viewOverview infoView = iota
	viewAgents
	viewEndpoints
	viewCache
	viewSafety
	numInfoViews // sentinel: the count of real views
)

// viewNone is the region sentinel for a clickable that is NOT a view tab (the glossary toggle).
const viewNone infoView = -1

func infoViewCount() int { return int(numInfoViews) }

// infoViewName is the short tab label for a view (the "overview"/"agents"/… word after the
// number). Unknown values degrade to "view" rather than panicking, so a future view added to
// the enum without a name still renders a stable, if generic, tab.
func infoViewName(v infoView) string {
	switch v {
	case viewOverview:
		return "overview"
	case viewAgents:
		return "agents"
	case viewEndpoints:
		return "accounts"
	case viewCache:
		return "cache"
	case viewSafety:
		return "safety"
	default:
		return "view"
	}
}

// ── input scanning ──

// infoInputKind is the class of a decoded input event. infoInputNone means the byte advanced
// (or did not advance) the state machine without completing a recognized event.
type infoInputKind int

const (
	infoInputNone        infoInputKind = iota
	infoInputFocusIn                   // terminal focus report ESC [ I
	infoInputFocusOut                  // ESC [ O
	infoInputQuit                      // a quit key/byte (Ctrl-C/D/\, or 'q')
	infoInputTabNext                   // Tab / right-arrow — cycle to the next view
	infoInputTabPrev                   // left-arrow — cycle to the previous view
	infoInputTabSelect                 // a digit 1-9 — jump to view Index (1-based)
	infoInputToggleGloss               // '?' / 'g' — toggle the glossary overlay
	infoInputMouseClick                // an SGR left-button press at (X,Y), 1-based cells
)

// infoInput is one decoded input event. Index is meaningful only for infoInputTabSelect; X/Y
// only for infoInputMouseClick (the 1-based cell coordinates the terminal reported, ABSOLUTE
// on screen — the loop translates them to block-relative rows before hit-testing).
type infoInput struct {
	Kind  infoInputKind
	Index int
	X, Y  int
}

// info input scanner states. A focus report / arrow is ESC [ <final>; an SGR mouse event is
// ESC [ < params <M|m>; any other CSI is swallowed to its final byte so its terminator is never
// mistaken for a key.
const (
	inScanGround   = iota // no pending escape
	inScanESC             // saw 0x1b
	inScanCSI             // saw ESC [ — the next byte decides
	inScanCSIParam        // inside a parameterized CSI (mouse or an ignored sequence)
)

// infoInputScanner is a resumable byte-at-a-time decoder for the interactive overlay's raw
// stdin. It is a superset of the focus scanner (info_focus.go): it recognizes focus reports AND
// tab/arrow keys, digit view-selects, the glossary toggle, quit keys, and SGR (1006) mouse
// presses. Resumable because a raw Read can split an escape sequence across calls; its only
// state is the parser state + a bounded param buffer, so a value is cheap to copy and reset.
type infoInputScanner struct {
	state  int
	priv   byte   // the private marker after ESC[ ('<' for SGR mouse), else 0
	params []byte // accumulated param/intermediate bytes of the current CSI
}

func (s *infoInputScanner) reset() {
	s.state = inScanGround
	s.priv = 0
	s.params = s.params[:0]
}

// step feeds one byte and returns the event it completes (Kind infoInputNone for an in-progress
// or inert byte). Quit control bytes win from any state (raw mode disables ISIG, so Ctrl-C
// arrives as 0x03 rather than a signal); a lone ESC always (re)starts a sequence so a truncated
// one re-syncs on the next ESC rather than wedging the parser.
func (s *infoInputScanner) step(b byte) infoInput {
	switch b {
	case 0x03, 0x04, 0x1c: // Ctrl-C / Ctrl-D / Ctrl-\ — quit (raw mode swallowed the signal)
		s.reset()
		return infoInput{Kind: infoInputQuit}
	case 0x1b: // ESC always (re)starts a sequence — the re-sync point
		s.state = inScanESC
		s.priv = 0
		s.params = s.params[:0]
		return infoInput{}
	}
	switch s.state {
	case inScanGround:
		return s.stepGround(b)
	case inScanESC:
		if b == '[' {
			s.state = inScanCSI
		} else {
			s.reset() // ESC + non-'[' is not a CSI we track
		}
		return infoInput{}
	case inScanCSI:
		return s.stepCSIStart(b)
	case inScanCSIParam:
		return s.stepCSIParam(b)
	default:
		s.reset()
		return infoInput{}
	}
}

// stepGround decodes a plain (non-escape) key.
func (s *infoInputScanner) stepGround(b byte) infoInput {
	switch {
	case b == 0x09: // Tab — cycle forward
		return infoInput{Kind: infoInputTabNext}
	case b == 'q' || b == 'Q':
		return infoInput{Kind: infoInputQuit}
	case b == '?' || b == 'g' || b == 'G':
		return infoInput{Kind: infoInputToggleGloss}
	case b >= '1' && b <= '9':
		return infoInput{Kind: infoInputTabSelect, Index: int(b - '0')}
	}
	return infoInput{}
}

// stepCSIStart decodes the byte immediately after ESC [.
func (s *infoInputScanner) stepCSIStart(b byte) infoInput {
	switch b {
	case 'I':
		s.reset()
		return infoInput{Kind: infoInputFocusIn}
	case 'O':
		s.reset()
		return infoInput{Kind: infoInputFocusOut}
	case 'C': // right arrow
		s.reset()
		return infoInput{Kind: infoInputTabNext}
	case 'D': // left arrow
		s.reset()
		return infoInput{Kind: infoInputTabPrev}
	case 'A', 'B': // up/down arrow — consumed but inert (no vertical navigation yet)
		s.reset()
		return infoInput{}
	case '<': // SGR (1006) mouse private marker — accumulate the Cb;Cx;Cy params
		s.priv = '<'
		s.params = s.params[:0]
		s.state = inScanCSIParam
		return infoInput{}
	}
	if b >= 0x40 && b <= 0x7e { // an immediate final byte we do not handle — done
		s.reset()
		return infoInput{}
	}
	s.params = append(s.params, b) // a param/intermediate byte of some other CSI — swallow it
	s.state = inScanCSIParam
	return infoInput{}
}

// stepCSIParam accumulates param/intermediate bytes until the final byte ends the sequence. A
// bounded buffer guards a malformed stream from growing without limit.
func (s *infoInputScanner) stepCSIParam(b byte) infoInput {
	if b >= 0x40 && b <= 0x7e { // final byte
		final := b
		priv := s.priv
		params := string(s.params)
		s.reset()
		if priv == '<' && (final == 'M' || final == 'm') {
			return parseSGRMouse(params, final)
		}
		return infoInput{} // some other CSI (cursor report, mode reply) — inert
	}
	s.params = append(s.params, b)
	if len(s.params) > 32 { // a malformed sequence: abandon it and re-sync on the next ESC
		s.reset()
	}
	return infoInput{}
}

// parseSGRMouse turns an SGR (1006) mouse sequence's "Cb;Cx;Cy" params + final byte into a
// click event. It reports ONLY a left-button PRESS ('M', button 0, no motion/wheel bits) as a
// click; a release ('m'), a drag (motion bit 0x20), or a wheel (code >= 64) is inert — the pane
// reacts to a deliberate click, not to hovering or scrolling.
func parseSGRMouse(params string, final byte) infoInput {
	if final != 'M' {
		return infoInput{}
	}
	fields := strings.Split(params, ";")
	if len(fields) != 3 {
		return infoInput{}
	}
	cb, err1 := strconv.Atoi(fields[0])
	x, err2 := strconv.Atoi(fields[1])
	y, err3 := strconv.Atoi(fields[2])
	if err1 != nil || err2 != nil || err3 != nil {
		return infoInput{}
	}
	if cb&0x20 != 0 || cb >= 64 { // motion/drag or wheel — not a click
		return infoInput{}
	}
	if cb&0x03 != 0 { // not the left button (low two bits select the button; 0 = left)
		return infoInput{}
	}
	if x < 1 || y < 1 {
		return infoInput{}
	}
	return infoInput{Kind: infoInputMouseClick, X: x, Y: y}
}

// ── view state + reducer ──

// infoViewState is the interactive overlay's whole UI state: which view is active, whether the
// glossary overlay is open, and (when open) which term's definition is expanded ("" = show the
// term list). It is the only cross-tick UI state the loop carries beyond the trend ring; a value
// is cheap to copy, and every transition is a pure fold via applyInfoInput.
type infoViewState struct {
	active       infoView
	glossaryOpen bool
	glossaryTerm string // the expanded term key, or "" for the whole list
}

// applyInfoInput folds one decoded event into the UI state. Focus/quit/none are handled by the
// loop (cadence + teardown), not the UI state, so they pass through unchanged. A mouse click is
// resolved against the deterministic tab-bar / glossary layout by applyInfoClick.
func applyInfoInput(s infoViewState, ev infoInput) infoViewState {
	switch ev.Kind {
	case infoInputTabNext:
		s.active = infoView((int(s.active) + 1) % infoViewCount())
	case infoInputTabPrev:
		s.active = infoView((int(s.active) + infoViewCount() - 1) % infoViewCount())
	case infoInputTabSelect:
		if ev.Index >= 1 && ev.Index <= infoViewCount() {
			s.active = infoView(ev.Index - 1)
		}
	case infoInputToggleGloss:
		s.glossaryOpen = !s.glossaryOpen
		s.glossaryTerm = ""
	case infoInputMouseClick:
		return applyInfoClick(s, ev.X, ev.Y)
	}
	return s
}

// applyInfoClick resolves a block-relative click (x,y are 1-based cells, y is the row WITHIN the
// rendered block) against the same deterministic layout the renderer draws: row 1 is the tab bar
// (a tab region selects its view; the glossary toggle region toggles the overlay), and — when
// the glossary is open — row 2 is the term-chip bar (a chip expands that term, or collapses back
// to the list when the already-expanded term is clicked again). A click on any other cell is
// inert.
func applyInfoClick(s infoViewState, x, y int) infoViewState {
	if y == 1 {
		bar := buildInfoTabBar(s.active, s.glossaryOpen)
		for _, r := range bar.regions {
			if x >= r.start && x <= r.end {
				if r.view == viewNone { // the glossary toggle
					s.glossaryOpen = !s.glossaryOpen
					s.glossaryTerm = ""
					return s
				}
				s.active = r.view
				return s
			}
		}
		return s
	}
	if s.glossaryOpen && y == 2 {
		chips := buildInfoGlossChips()
		for _, r := range chips.regions {
			if x >= r.start && x <= r.end {
				if s.glossaryTerm == r.term {
					s.glossaryTerm = "" // clicking the open term again collapses to the list
				} else {
					s.glossaryTerm = r.term
				}
				return s
			}
		}
	}
	return s
}

// ── tab bar + glossary chip layout (the single source of truth for render AND hit-test) ──

// infoTabRegion is one clickable span on a bar row: the view it selects (or viewNone for the
// glossary toggle), the glossary term key it expands (chips only), and its inclusive 1-based
// cell range so a mouse click can be resolved to it.
type infoTabRegion struct {
	view  infoView
	term  string
	start int
	end   int
}

// infoBar is a rendered bar row plus its clickable regions — one object so the renderer and the
// hit-test can never disagree about where a chip sits.
type infoBar struct {
	text    string
	regions []infoTabRegion
}

// barBuilder accumulates a bar row cell-by-cell, recording the [start,end] span of each labeled
// segment so the rendered text and the click regions are computed in lockstep.
type barBuilder struct {
	b   strings.Builder
	col int // cells written so far
}

func (bb *barBuilder) sep(s string) {
	bb.b.WriteString(s)
	bb.col += dispWidthTUI(s)
}

// segment writes a labeled span and returns its inclusive 1-based cell range.
func (bb *barBuilder) segment(s string) (start, end int) {
	start = bb.col + 1
	bb.b.WriteString(s)
	bb.col += dispWidthTUI(s)
	return start, bb.col
}

// buildInfoTabBar renders the tab bar and its click regions for the given active view and
// glossary state. The active view's chip is wrapped in guillemets («…»), the rest in brackets
// ([…]), so the selection reads at a glance without relying on color; a trailing glossary toggle
// («? close» when open, "[? glossary]" when closed) rounds out the row.
func buildInfoTabBar(active infoView, glossaryOpen bool) infoBar {
	var bb barBuilder
	var regions []infoTabRegion
	for i := 0; i < infoViewCount(); i++ {
		v := infoView(i)
		if i > 0 {
			bb.sep(" ")
		}
		label := fmt.Sprintf("%d %s", i+1, infoViewName(v))
		chip := "[" + label + "]"
		if v == active {
			chip = "«" + label + "»"
		}
		start, end := bb.segment(chip)
		regions = append(regions, infoTabRegion{view: v, start: start, end: end})
	}
	bb.sep("   ")
	glossChip := "[? glossary]"
	if glossaryOpen {
		glossChip = "«? close»"
	}
	start, end := bb.segment(glossChip)
	regions = append(regions, infoTabRegion{view: viewNone, start: start, end: end})
	return infoBar{text: bb.b.String(), regions: regions}
}

// buildInfoGlossChips renders the glossary term-chip bar and its click regions. The chip order
// is the fixed glossaryTerms order so the layout is deterministic across ticks.
func buildInfoGlossChips() infoBar {
	var bb barBuilder
	var regions []infoTabRegion
	for i, t := range glossaryTerms {
		if i > 0 {
			bb.sep(" ")
		}
		start, end := bb.segment("[" + t.key + "]")
		regions = append(regions, infoTabRegion{view: viewNone, term: t.key, start: start, end: end})
	}
	return infoBar{text: bb.b.String(), regions: regions}
}

// ── the glossary ──

// glossaryTerm is one plain-words definition. key is the clickable chip label (and the word that
// appears in the pane); short is a one-line gloss for the chip list; long is the full paragraph
// shown when the term is expanded.
type glossaryTerm struct {
	key   string
	short string
	long  string
}

// glossaryTerms is the ordered glossary: the same plain-words explanations the one-time legend
// prints (guardInfoLegend), turned into individually-clickable entries so a watcher can look up a
// single word on demand instead of re-reading the whole guide.
var glossaryTerms = []glossaryTerm{
	{"cache", "re-using text to cost less",
		"fak re-uses text it already sent so the model costs less. \"saving money\" means the re-use has paid back the small setup cost; \"reused %\" is how much was re-used; \"×N cheaper\" is how much cheaper it made those tokens."},
	{"safety", "what fak blocked/fixed/set aside",
		"what fak did to keep you safe: BLOCKED an unsafe tool call, FIXED a risky one before it ran, or SET ASIDE a suspicious result instead of feeding it to the model. A clean session reads \"nothing blocked\"."},
	{"why", "the reason behind the blocks",
		"the reason code(s) behind the safety blocks — the same \"blocked: reason ×N\" breakdown fak prints when the session ends, now live — plus anything HELD FOR A WITNESS (a call paused pending approval) or DEFERRED (a result let through while raising the taint watermark)."},
	{"saved", "engine calls fak avoided",
		"\"turns saved\": engine calls fak spared the agent this session (served from its own cache or handled in-kernel) so the agent never had to make them. Counted in CALLS, not tokens, so it never reads as a provider-cache token saving."},
	{"agents", "live sessions through this fak",
		"the live sessions running through this fak — the main agent plus any SUB-AGENTS it spawned (shown with parent lineage and spawn depth) — each with its remaining budget, wall-clock, and what it is doing right now (last tool, in-flight/idle)."},
	{"accounts", "which Claude seats this session uses",
		"the Claude subscription SEATS in the on-box roster this session can serve from: ● the active seat, ⊘ a seat a failover walled this session, ○ an idle seat, with each seat's login readiness."},
	{"nodes", "where the session runs",
		"where the session runs: every guarded session has at least two — the KERNEL node (this host, where fak + the agent + adjudication run) and a SERVING node (where inference runs: a provider proxy, a remote fak serve box, a local server, or in-kernel)."},
	{"prefix", "did the cacheable prefix stay stable",
		"whether the stable/cacheable front of the prompt (system + tools + any protected span) stayed byte-identical across the last turn boundary. \"stable\" keeps the provider cache warm; \"mutated\" means it diverged (and names where), which bursts the cache."},
	{"assumptions", "facts the session relies on",
		"the active facts the session is relying on, each with its source class (user-stated / inferred / queried / witnessed / stale), confidence, expiry, and origin reference — read straight from public session state, never from hidden transcript text."},
	{"replies", "answers, work in flight, uptime",
		"the liveness summary: \"replies\" is how many answers the model has given, \"busy with\" is work happening right now, and \"running\" is how long this fak has been up."},
}

// ── TTY plumbing (the thin, gated, non-unit-tested layer, mirroring info_focus.go) ──

// The xterm mouse-reporting toggles: 1000 (button-press/release tracking) + 1006 (SGR extended
// coordinates, so a click past column 223 still reports accurately — the X10 encoding caps at
// 223). Enabling makes a supporting terminal (Windows Terminal, xterm, iTerm, tmux) emit the
// ESC[<b;x;yM/m sequences infoInputScanner decodes; disabling on exit is mandatory so the pane is
// not left reporting mouse events to whatever runs next.
const (
	mouseEnable  = "\033[?1000h\033[?1006h"
	mouseDisable = "\033[?1006l\033[?1000l"
)

func writeMouseEnable(w io.Writer)  { _, _ = io.WriteString(w, mouseEnable) }
func writeMouseDisable(w io.Writer) { _, _ = io.WriteString(w, mouseDisable) }

// startGuardInfoInputReader is the interactive overlay's raw-stdin reader goroutine — the
// superset of startGuardInfoFocusReader (info_focus.go): it decodes focus reports AND tab/arrow
// keys, digit view-selects, the glossary toggle, quit keys, and SGR mouse presses, forwarding
// every completed event on a buffered channel the loop selects on. A quit event also fires onQuit
// so the loop tears down even if the channel is momentarily full. The goroutine exits when r
// returns an error/EOF. It is only started behind the overlay's interactive gate, so it never
// touches a non-TTY or piped stdin.
func startGuardInfoInputReader(r io.Reader, onQuit func()) <-chan infoInput {
	ch := make(chan infoInput, 16)
	go func() {
		var sc infoInputScanner
		buf := make([]byte, 1)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				if ev := sc.step(buf[0]); ev.Kind != infoInputNone {
					if ev.Kind == infoInputQuit && onQuit != nil {
						onQuit()
					}
					select {
					case ch <- ev:
					default: // a slow consumer must never block the reader; drop the surplus event
					}
				}
			}
			if err != nil {
				return
			}
		}
	}()
	return ch
}

// blockRelativeRow translates an ABSOLUTE terminal row (as a mouse report carries) into a row
// WITHIN the in-place-redrawn block, which is pinned to the bottom prevRows rows of a pane of the
// given height. It returns 0 — an inert, un-hittable row — when the geometry is unknown (height
// or prevRows unset) or the click landed outside the block, so a click above the block or on an
// unmeasured pane simply does nothing rather than mis-selecting a chip.
func blockRelativeRow(absY, height, prevRows int) int {
	if height <= 0 || prevRows <= 0 {
		return 0
	}
	top := height - prevRows // rows scrolled above the parked block
	row := absY - top
	if row < 1 || row > prevRows {
		return 0
	}
	return row
}

// glossaryTermByKey returns the definition for a term key, or false when unknown.
func glossaryTermByKey(key string) (glossaryTerm, bool) {
	for _, t := range glossaryTerms {
		if t.key == key {
			return t, true
		}
	}
	return glossaryTerm{}, false
}

// renderInfoGlossaryBody renders the glossary overlay's body (everything below the tab bar): the
// clickable chip row, then either the expanded definition of the selected term or the full
// one-line list when no term is selected. Rows are width-capped by the caller.
func renderInfoGlossaryBody(term string, width int) []string {
	rows := []string{" " + buildInfoGlossChips().text}
	if t, ok := glossaryTermByKey(term); ok {
		rows = append(rows, "")
		rows = append(rows, " "+t.key+" — "+t.short)
		rows = append(rows, wrapInfoText(t.long, " ", width)...)
		return rows
	}
	rows = append(rows, " click a term above, or read the one-liners:")
	for _, t := range glossaryTerms {
		rows = append(rows, fmt.Sprintf("  %-12s %s", t.key, t.short))
	}
	return rows
}

// wrapInfoText word-wraps s into rows of at most width cells, each prefixed with indent. width<=0
// (unknown pane) leaves it a single prefixed line. It is a plain greedy wrapper — enough for the
// short definition paragraphs, and rune-safe because it measures with dispWidthTUI.
func wrapInfoText(s, indent string, width int) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	budget := width - dispWidthTUI(indent)
	if width <= 0 || budget <= 0 {
		return []string{indent + s}
	}
	var rows []string
	var line strings.Builder
	for _, word := range strings.Fields(s) {
		if line.Len() == 0 {
			line.WriteString(word)
			continue
		}
		if dispWidthTUI(line.String())+1+dispWidthTUI(word) > budget {
			rows = append(rows, indent+line.String())
			line.Reset()
			line.WriteString(word)
			continue
		}
		line.WriteByte(' ')
		line.WriteString(word)
	}
	if line.Len() > 0 {
		rows = append(rows, indent+line.String())
	}
	return rows
}
