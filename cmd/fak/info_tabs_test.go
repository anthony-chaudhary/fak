package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// feedInfoInput runs a byte string through a fresh scanner and returns every non-none event, so
// a test can assert an escape sequence split arbitrarily still decodes to the same events.
func feedInfoInput(s string) []infoInput {
	var sc infoInputScanner
	var out []infoInput
	for i := 0; i < len(s); i++ {
		if ev := sc.step(s[i]); ev.Kind != infoInputNone {
			out = append(out, ev)
		}
	}
	return out
}

// TestInfoInputScannerKeys pins the plain-key decoding: Tab/arrows cycle, digits select, '?'/'g'
// toggle the glossary, and the quit keys (Ctrl-C and 'q') both quit.
func TestInfoInputScannerKeys(t *testing.T) {
	cases := []struct {
		in   string
		want infoInputKind
	}{
		{"\t", infoInputTabNext},
		{"\x1b[C", infoInputTabNext},  // right arrow
		{"\x1b[D", infoInputTabPrev},  // left arrow
		{"?", infoInputToggleGloss},   //
		{"g", infoInputToggleGloss},   //
		{"q", infoInputQuit},          //
		{"\x03", infoInputQuit},       // Ctrl-C (raw mode delivers the byte)
		{"\x1b[I", infoInputFocusIn},  //
		{"\x1b[O", infoInputFocusOut}, //
	}
	for _, tc := range cases {
		evs := feedInfoInput(tc.in)
		if len(evs) != 1 || evs[0].Kind != tc.want {
			t.Errorf("feed %q = %+v, want single kind %v", tc.in, evs, tc.want)
		}
	}
	// A digit selects the 1-based view index.
	evs := feedInfoInput("3")
	if len(evs) != 1 || evs[0].Kind != infoInputTabSelect || evs[0].Index != 3 {
		t.Errorf("digit 3 = %+v, want TabSelect index 3", evs)
	}
}

// TestInfoInputScannerSplitSequence proves resumability: an escape sequence delivered one byte at
// a time (as a raw Read can split it) still decodes to exactly one event, and an unrelated CSI
// (a cursor-position report) is swallowed without emitting a spurious key.
func TestInfoInputScannerSplitSequence(t *testing.T) {
	var sc infoInputScanner
	if ev := sc.step(0x1b); ev.Kind != infoInputNone {
		t.Fatalf("ESC alone must be inert, got %v", ev)
	}
	if ev := sc.step('['); ev.Kind != infoInputNone {
		t.Fatalf("ESC[ must be inert, got %v", ev)
	}
	if ev := sc.step('I'); ev.Kind != infoInputFocusIn {
		t.Fatalf("split ESC[I must decode focus-in, got %v", ev)
	}
	// A cursor-position report ESC[12;34R must NOT be read as a focus/key event.
	if evs := feedInfoInput("\x1b[12;34R"); len(evs) != 0 {
		t.Fatalf("cursor report must be swallowed, got %+v", evs)
	}
}

// TestInfoInputScannerMouse pins SGR (1006) mouse decoding: a left-button press is a click at
// (x,y); a release, a drag (motion bit), a wheel (code>=64), and a right/middle button are all
// inert (the pane reacts only to a deliberate left click).
func TestInfoInputScannerMouse(t *testing.T) {
	evs := feedInfoInput("\x1b[<0;14;1M") // left press at col 14, row 1
	if len(evs) != 1 || evs[0].Kind != infoInputMouseClick || evs[0].X != 14 || evs[0].Y != 1 {
		t.Fatalf("left press = %+v, want click at (14,1)", evs)
	}
	for _, in := range []string{
		"\x1b[<0;14;1m",  // release
		"\x1b[<35;14;1M", // motion/drag (bit 0x20 set)
		"\x1b[<64;14;1M", // wheel up (code >= 64)
		"\x1b[<2;14;1M",  // right button (low bits != 0)
	} {
		if evs := feedInfoInput(in); len(evs) != 0 {
			t.Errorf("non-click mouse %q must be inert, got %+v", in, evs)
		}
	}
}

// TestApplyInfoInputTabs pins the view reducer: next/prev wrap around, a digit selects, an
// out-of-range digit is ignored, and the glossary toggle flips open/closed.
func TestApplyInfoInputTabs(t *testing.T) {
	s := infoViewState{active: viewOverview}
	s = applyInfoInput(s, infoInput{Kind: infoInputTabNext})
	if s.active != viewAgents {
		t.Fatalf("next from overview = %d, want agents", s.active)
	}
	s = applyInfoInput(s, infoInput{Kind: infoInputTabPrev})
	if s.active != viewOverview {
		t.Fatalf("prev back to overview = %d", s.active)
	}
	// Prev from the first view wraps to the last.
	s = applyInfoInput(infoViewState{active: viewOverview}, infoInput{Kind: infoInputTabPrev})
	if int(s.active) != infoViewCount()-1 {
		t.Fatalf("prev from first view = %d, want last (%d)", s.active, infoViewCount()-1)
	}
	// A digit jumps directly; an out-of-range digit is ignored.
	s = applyInfoInput(infoViewState{}, infoInput{Kind: infoInputTabSelect, Index: 4})
	if s.active != viewCache {
		t.Fatalf("select 4 = %d, want cache", s.active)
	}
	s = applyInfoInput(infoViewState{active: viewCache}, infoInput{Kind: infoInputTabSelect, Index: 9})
	if s.active != viewCache {
		t.Fatalf("out-of-range select must be a no-op, got %d", s.active)
	}
	// Glossary toggles.
	s = applyInfoInput(infoViewState{}, infoInput{Kind: infoInputToggleGloss})
	if !s.glossaryOpen {
		t.Fatalf("toggle must open the glossary")
	}
	s = applyInfoInput(s, infoInput{Kind: infoInputToggleGloss})
	if s.glossaryOpen {
		t.Fatalf("second toggle must close the glossary")
	}
}

// TestApplyInfoClickTabBar proves a click on a tab-bar region selects that view, and a click on
// the glossary toggle region opens the glossary — resolved against the SAME layout the renderer
// draws, so the click always lands where the chip is.
func TestApplyInfoClickTabBar(t *testing.T) {
	bar := buildInfoTabBar(viewOverview, false)
	// Find the "safety" tab region and click its first cell.
	var safety infoTabRegion
	var gloss infoTabRegion
	for _, r := range bar.regions {
		if r.view == viewSafety {
			safety = r
		}
		if r.view == viewNone {
			gloss = r
		}
	}
	if safety.end == 0 || gloss.end == 0 {
		t.Fatalf("tab bar missing safety/glossary regions: %+v", bar.regions)
	}
	s := applyInfoInput(infoViewState{active: viewOverview}, infoInput{Kind: infoInputMouseClick, X: safety.start, Y: 1})
	if s.active != viewSafety {
		t.Fatalf("click on the safety tab = %d, want safety", s.active)
	}
	s = applyInfoInput(infoViewState{}, infoInput{Kind: infoInputMouseClick, X: gloss.start, Y: 1})
	if !s.glossaryOpen {
		t.Fatalf("click on the glossary toggle must open it")
	}
	// A click off any region (far right) is inert.
	s = applyInfoInput(infoViewState{active: viewAgents}, infoInput{Kind: infoInputMouseClick, X: 9999, Y: 1})
	if s.active != viewAgents || s.glossaryOpen {
		t.Fatalf("off-region click must be a no-op, got %+v", s)
	}
}

// TestApplyInfoClickGlossaryChip proves that with the glossary open, a click on a term chip (row
// 2) expands that term, and clicking the same chip again collapses back to the list.
func TestApplyInfoClickGlossaryChip(t *testing.T) {
	chips := buildInfoGlossChips()
	var why infoTabRegion
	for _, r := range chips.regions {
		if r.term == "why" {
			why = r
		}
	}
	if why.end == 0 {
		t.Fatalf("glossary chips missing the why term: %+v", chips.regions)
	}
	s := infoViewState{glossaryOpen: true}
	s = applyInfoInput(s, infoInput{Kind: infoInputMouseClick, X: why.start, Y: 2})
	if s.glossaryTerm != "why" {
		t.Fatalf("click on the why chip = %q, want why expanded", s.glossaryTerm)
	}
	s = applyInfoInput(s, infoInput{Kind: infoInputMouseClick, X: why.start, Y: 2})
	if s.glossaryTerm != "" {
		t.Fatalf("re-click must collapse to the list, got %q", s.glossaryTerm)
	}
	// With the glossary CLOSED, a row-2 click is inert (no chip bar there).
	closed := applyInfoInput(infoViewState{}, infoInput{Kind: infoInputMouseClick, X: why.start, Y: 2})
	if closed.glossaryTerm != "" || closed.glossaryOpen {
		t.Fatalf("row-2 click with glossary closed must be a no-op, got %+v", closed)
	}
}

// TestBuildInfoTabBarActiveMarker proves the tab bar marks the active view distinctly (guillemets
// vs brackets) and that its regions cover contiguous, non-overlapping cell spans.
func TestBuildInfoTabBarActiveMarker(t *testing.T) {
	bar := buildInfoTabBar(viewCache, false)
	if !strings.Contains(bar.text, "«4 cache»") {
		t.Fatalf("active cache tab must be guillemet-marked: %q", bar.text)
	}
	if !strings.Contains(bar.text, "[1 overview]") {
		t.Fatalf("inactive overview tab must be bracketed: %q", bar.text)
	}
	prevEnd := 0
	for _, r := range bar.regions {
		if r.start <= prevEnd {
			t.Fatalf("region %+v overlaps previous end %d", r, prevEnd)
		}
		if r.end < r.start {
			t.Fatalf("region %+v has end<start", r)
		}
		prevEnd = r.end
	}
}

// TestRenderInteractiveBlockViews proves the top-level interactive block draws the tab bar and the
// active view's focused body: the agents view lists every session (past the overview's 4-row cap),
// and the safety view shows the full uncapped deny-reason breakdown.
func TestRenderInteractiveBlockViews(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := richVisualVars()
	// Seven sessions — more than the overview's guardInfoAgentsMaxRows cap.
	v.Sessions = nil
	for i := 0; i < 7; i++ {
		v.Sessions = append(v.Sessions, guardInfoSession{TraceID: "trace" + string(rune('a'+i)), Run: "running"})
	}
	tr.push(v)

	agents := renderGuardInfoInteractiveBlock(infoViewState{active: viewAgents}, v, tr, 120, 0 /*roomy*/)
	if !strings.Contains(agents, "«2 agents»") {
		t.Fatalf("agents view must mark the agents tab active:\n%s", agents)
	}
	if strings.Count(agents, "running") < 7 {
		t.Fatalf("agents view must list every session (7), got:\n%s", agents)
	}

	// Safety view: full breakdown of many reasons, uncapped.
	v2 := provenVisualVars()
	v2.Adjudication = &gateway.AdjudicationSummary{
		Denied: 5, Escalated: 1,
		ByReason: map[string]uint64{"dangerous_command": 3, "out_of_tree_write": 1, "secret_in_arg": 1, "unknown_tool": 1, "path_escape": 1},
	}
	safety := renderGuardInfoInteractiveBlock(infoViewState{active: viewSafety}, v2, tr, 120, 0)
	for _, want := range []string{"«5 safety»", "blocked: dangerous_command", "blocked: path_escape", "held for witness: 1"} {
		if !strings.Contains(safety, want) {
			t.Fatalf("safety view missing %q:\n%s", want, safety)
		}
	}
}

// TestRenderInteractiveBlockGlossary proves the glossary overlay renders its chip row and, once a
// term is expanded, that term's plain-words definition.
func TestRenderInteractiveBlockGlossary(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := provenVisualVars()
	tr.push(v)
	block := renderGuardInfoInteractiveBlock(infoViewState{glossaryOpen: true, glossaryTerm: "why"}, v, tr, 120, 0)
	if !strings.Contains(block, "«? close»") {
		t.Fatalf("open glossary must show the close toggle:\n%s", block)
	}
	if !strings.Contains(block, "[why]") || !strings.Contains(block, "[cache]") {
		t.Fatalf("glossary chip row must list the term chips:\n%s", block)
	}
	if !strings.Contains(block, "reason code") {
		t.Fatalf("expanded why term must show its definition:\n%s", block)
	}
}

// TestRenderInfoEndpointsViewSeatDetail proves the accounts view expands each seat into a detail
// row with its posture + login identity, and falls back to a plain note when no roster is present.
func TestRenderInfoEndpointsViewSeatDetail(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{Endpoints: endpointsFixture()}, width: 120}
	rows := strings.Join(renderInfoEndpointsView(ctx), "\n")
	if !strings.Contains(rows, "july2") || !strings.Contains(rows, "active") || !strings.Contains(rows, "walled") {
		t.Fatalf("endpoints view must show per-seat posture:\n%s", rows)
	}
	if !strings.Contains(rows, "july2@x") {
		t.Fatalf("endpoints view must show the seat login identity:\n%s", rows)
	}
	empty := strings.Join(renderInfoEndpointsView(guardInfoPanelCtx{v: guardInfoVars{}, width: 80}), "\n")
	if !strings.Contains(empty, "none reported") {
		t.Fatalf("empty endpoints view must note the absence:\n%s", empty)
	}
}

// TestGlossaryTermByKey pins the glossary lookup: a known key resolves, an unknown one reports
// false, and every chip in the layout has a resolvable definition (no dangling chip).
func TestGlossaryTermByKey(t *testing.T) {
	if _, ok := glossaryTermByKey("safety"); !ok {
		t.Fatalf("safety must resolve")
	}
	if _, ok := glossaryTermByKey("nope"); ok {
		t.Fatalf("unknown term must not resolve")
	}
	for _, r := range buildInfoGlossChips().regions {
		if _, ok := glossaryTermByKey(r.term); !ok {
			t.Fatalf("glossary chip %q has no definition", r.term)
		}
	}
}

// TestSafetyViewTermsHaveGlossary proves the two forensic labels the Safety focused view surfaces
// as standalone words — "held for witness" and "deferred" — each have a clickable glossary
// definition, so a watcher who lands on the Safety tab and does not know the term can look it up.
// It is the coherence witness for the Phase-B "why" detail promoted into the focused view: the
// view speaks the words, the glossary defines them.
func TestSafetyViewTermsHaveGlossary(t *testing.T) {
	v := provenVisualVars()
	v.Adjudication = &gateway.AdjudicationSummary{
		Denied: 2, Escalated: 1, Deferred: 1,
		ByReason: map[string]uint64{"dangerous_command": 2},
	}
	safety := strings.Join(renderInfoSafetyView(v), "\n")
	// The view must speak both words...
	if !strings.Contains(safety, "held for witness") || !strings.Contains(safety, "deferred") {
		t.Fatalf("safety view must surface both forensic labels:\n%s", safety)
	}
	// ...and the glossary must define each as a clickable term.
	for _, key := range []string{"witness", "deferred"} {
		term, ok := glossaryTermByKey(key)
		if !ok {
			t.Fatalf("glossary must define %q for the safety view", key)
		}
		if strings.TrimSpace(term.short) == "" || strings.TrimSpace(term.long) == "" {
			t.Fatalf("glossary term %q must carry both a short and long definition", key)
		}
	}
}
