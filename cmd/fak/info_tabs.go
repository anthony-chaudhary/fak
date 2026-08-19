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
	infoInputNone             infoInputKind = iota
	infoInputFocusIn                        // terminal focus report ESC [ I
	infoInputFocusOut                       // ESC [ O
	infoInputQuit                           // a quit key/byte (Ctrl-C/D/\, or 'q')
	infoInputTabNext                        // Tab / right-arrow — cycle to the next view
	infoInputTabPrev                        // left-arrow — cycle to the previous view
	infoInputTabSelect                      // a digit 1-9 — jump to view Index (1-based)
	infoInputToggleGloss                    // '?' / 'g' — toggle the glossary overlay
	infoInputMouseClick                     // an SGR left-button press at (X,Y), 1-based cells
	infoInputScrollUp                       // up-arrow / wheel-up — scroll the active view up a line
	infoInputScrollDown                     // down-arrow / wheel-down — scroll the active view down a line
	infoInputPageUp                         // PageUp — scroll the active view up a page
	infoInputPageDown                       // PageDown — scroll the active view down a page
	infoInputScrollHome                     // Home — jump the active view to the top
	infoInputScrollEnd                      // End — jump the active view to the bottom
	infoInputCopyMode                       // 'c' — toggle copy/freeze mode (hand selection back to the terminal)
	infoInputLaunchHarnessWeb               // 'w' — launch the shipped loopback web gateway
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
	case b == 'c' || b == 'C':
		return infoInput{Kind: infoInputCopyMode}
	case b == 'w' || b == 'W':
		return infoInput{Kind: infoInputLaunchHarnessWeb}
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
	case 'C': // right arrow — cycle to the next view
		s.reset()
		return infoInput{Kind: infoInputTabNext}
	case 'D': // left arrow — cycle to the previous view
		s.reset()
		return infoInput{Kind: infoInputTabPrev}
	case 'A': // up arrow — scroll the active view up a line
		s.reset()
		return infoInput{Kind: infoInputScrollUp}
	case 'B': // down arrow — scroll the active view down a line
		s.reset()
		return infoInput{Kind: infoInputScrollDown}
	case 'H': // Home — jump to the top
		s.reset()
		return infoInput{Kind: infoInputScrollHome}
	case 'F': // End — jump to the bottom
		s.reset()
		return infoInput{Kind: infoInputScrollEnd}
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
		if priv == 0 && final == '~' {
			return decodeTildeKey(params)
		}
		return infoInput{} // some other CSI (cursor report, mode reply) — inert
	}
	s.params = append(s.params, b)
	if len(s.params) > 32 { // a malformed sequence: abandon it and re-sync on the next ESC
		s.reset()
	}
	return infoInput{}
}

// decodeTildeKey maps a VT-style keypad sequence "ESC [ N ~" (N in params, an optional
// ";mods" modifier suffix ignored) to a scroll event: 5=PageUp, 6=PageDown, 1/7=Home,
// 4/8=End. An N the pane does not navigate on is inert.
func decodeTildeKey(params string) infoInput {
	if i := strings.IndexByte(params, ';'); i >= 0 {
		params = params[:i] // drop a "N;mods" modifier suffix — navigate on the base key
	}
	switch params {
	case "5":
		return infoInput{Kind: infoInputPageUp}
	case "6":
		return infoInput{Kind: infoInputPageDown}
	case "1", "7":
		return infoInput{Kind: infoInputScrollHome}
	case "4", "8":
		return infoInput{Kind: infoInputScrollEnd}
	}
	return infoInput{}
}

// parseSGRMouse turns an SGR (1006) mouse sequence's "Cb;Cx;Cy" params + final byte into an
// event. A wheel notch (button bit 6 set) scrolls the active view a line — up on the even code
// (64), down on the odd (65). Otherwise ONLY a left-button PRESS ('M', button 0, no motion bit)
// is a click; a release ('m'), a drag (motion bit 0x20), or a non-left button is inert — the
// pane reacts to a deliberate click, not to hovering.
func parseSGRMouse(params string, final byte) infoInput {
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
	if cb&0xC0 == 0x40 { // wheel notch (bit 6 set, bit 7 clear); modifiers ride the middle bits
		if final != 'M' { // a notch is a press; ignore the (rare) release form
			return infoInput{}
		}
		if cb&0x01 == 0 {
			return infoInput{Kind: infoInputScrollUp}
		}
		return infoInput{Kind: infoInputScrollDown}
	}
	if final != 'M' { // a release — not a click
		return infoInput{}
	}
	if cb&0x20 != 0 { // motion/drag — not a click
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
	// scroll is the per-view scrollable offset (rows hidden above the window), indexed by
	// view. It is per-view so paging through Agents and coming back to Safety keeps each
	// view's place. Raw here (only floored at 0 / capped at the End sentinel); the renderer
	// and the loop's clampInfoScrollToSample pin it to the view's real content each frame.
	scroll [numInfoViews]int
	// cacheMech is the 1-based Cache-tab ablation mechanism whose detail sub-panel is expanded
	// (0 = none). Toggled by clicking a mechanism bar row; see applyInfoCacheMechClick. Kept on
	// the state (not per-view scroll) because it survives paging away and back to the Cache tab.
	cacheMech int
	// copyMode freezes the pane and hands text selection back to the terminal so a watcher can
	// select + copy the frame without the next tick's in-place redraw erasing it; toggled by 'c'.
	// The loop (info.go) owns the side effects — disabling mouse reporting, suppressing the redraw,
	// and making Ctrl-C forgiving here — while the renderer only swaps the tab bar for the banner.
	copyMode     bool
	launchWeb    bool   // one-shot request consumed by the interactive loop
	launchNotice string // observable start/open/failure result shown beneath the action bar
}

// infoScrollPageStep is how many rows a PageUp/PageDown moves the active view. Fixed (not
// pane-relative) so the reducer stays pure and TTY-free; the render/clamp step pins the result
// to the real content, so an over-long page can never scroll past the ends.
const infoScrollPageStep = 10

// infoScrollToEnd is the End-key sentinel: a large offset the render/clamp step pulls back to
// the last page. Storing a sentinel (rather than the true max, which the reducer cannot know
// without the content) keeps End working from any pane height.
const infoScrollToEnd = 1 << 30

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
	case infoInputLaunchHarnessWeb:
		s.launchWeb = true
	case infoInputToggleGloss:
		s.glossaryOpen = !s.glossaryOpen
		s.glossaryTerm = ""
	case infoInputScrollUp:
		s.scroll[s.active] = maxInt(0, s.scroll[s.active]-1)
	case infoInputScrollDown:
		s.scroll[s.active] = maxInt(0, s.scroll[s.active]+1)
	case infoInputPageUp:
		s.scroll[s.active] = maxInt(0, s.scroll[s.active]-infoScrollPageStep)
	case infoInputPageDown:
		s.scroll[s.active] = maxInt(0, s.scroll[s.active]+infoScrollPageStep)
	case infoInputScrollHome:
		s.scroll[s.active] = 0
	case infoInputScrollEnd:
		s.scroll[s.active] = infoScrollToEnd
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
				if r.term == "web-gateway" {
					s.launchWeb = true
					return s
				}
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
// ([…]), so the selection reads at a glance without relying on color. The dedicated web-gateway
// button leads the row so it remains visible when narrow terminals clip the right edge; the glossary
// toggle («? close» when open, "[? glossary]" when closed) follows the view tabs.
func buildInfoTabBar(active infoView, glossaryOpen bool) infoBar {
	var bb barBuilder
	var regions []infoTabRegion
	start, end := bb.segment("[w web gateway]")
	regions = append(regions, infoTabRegion{view: viewNone, term: "web-gateway", start: start, end: end})
	bb.sep("   ")
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
	start, end = bb.segment(glossChip)
	regions = append(regions, infoTabRegion{view: viewNone, start: start, end: end})
	// Keyboard-only hint for copy/freeze mode. No click region: entering copy mode disables mouse
	// reporting (a loop side effect the pure click-fold cannot do), so 'c' is the only way in.
	bb.sep("   c copy")
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
	{"witness", "a call paused for approval",
		"\"held for witness\": fak did not block or allow the call outright — it PAUSED it pending an approving witness (a human ok, or a corroborating check). The call is neither done nor refused; it waits. This is fak refusing to guess on a call it is not sure about, rather than failing open (allow) or failing shut (block)."},
	{"deferred", "a result let through, taint raised",
		"\"deferred\": fak let a suspicious RESULT reach the model instead of setting it aside, but raised the session's taint watermark so anything that result influences is watched more closely from here on. It is the softer sibling of \"set aside\": the data flows, but the session remembers it was doubtful."},
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
