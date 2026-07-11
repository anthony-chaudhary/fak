package main

import (
	"strings"
	"testing"
)

// TestInfoInputScannerCopyModeKey pins the 'c'/'C' decode: both toggle copy/freeze mode. This is
// the entry point for the whole feature — the loop's mouse-reporting handoff is only reachable
// once this byte decodes to infoInputCopyMode.
func TestInfoInputScannerCopyModeKey(t *testing.T) {
	for _, in := range []string{"c", "C"} {
		evs := feedInfoInput(in)
		if len(evs) != 1 || evs[0].Kind != infoInputCopyMode {
			t.Errorf("feed %q = %+v, want single infoInputCopyMode", in, evs)
		}
	}
}

// TestInfoTabBarShowsCopyHint proves the tab bar advertises the keyboard-only entry point, so a
// watcher can discover copy mode without reading the docs. Entering copy mode disables mouse
// reporting (a loop side effect the pure click-fold cannot do), so the hint is deliberately not a
// click region.
func TestInfoTabBarShowsCopyHint(t *testing.T) {
	bar := buildInfoTabBar(viewOverview, false)
	if !strings.Contains(bar.text, "c copy") {
		t.Errorf("tab bar must advertise the copy hint, got: %q", bar.text)
	}
}

// TestApplyInfoInputIgnoresCopyMode locks in the ownership boundary: copyMode is driven by the
// loop (which owns the mouse-reporting + freeze side effects), so the pure fold must never flip
// it — infoInputCopyMode is a no-op in applyInfoInput, and an ordinary action leaves it clear.
func TestApplyInfoInputIgnoresCopyMode(t *testing.T) {
	frozen := infoViewState{active: viewCache, copyMode: true}
	if got := applyInfoInput(frozen, infoInput{Kind: infoInputCopyMode}); got != frozen {
		t.Errorf("applyInfoInput must leave copyMode state untouched, got %+v want %+v", got, frozen)
	}
	if got := applyInfoInput(infoViewState{}, infoInput{Kind: infoInputTabSelect, Index: 2}); got.copyMode {
		t.Errorf("an ordinary fold must not set copyMode: %+v", got)
	}
}

// TestBuildInfoCopyBanner proves the banner states the how-to + the resume keys and never exceeds
// the pane width (a wider-than-pane banner would wrap and corrupt the in-place redraw the whole
// overlay depends on).
func TestBuildInfoCopyBanner(t *testing.T) {
	for _, w := range []int{20, 80, 200} {
		if got := dispWidthTUI(buildInfoCopyBanner(w)); got > w {
			t.Errorf("banner width %d exceeds pane width %d: %q", got, w, buildInfoCopyBanner(w))
		}
	}
	full := buildInfoCopyBanner(120)
	for _, want := range []string{"COPY MODE", "select", "c or Ctrl-C to resume"} {
		if !strings.Contains(full, want) {
			t.Errorf("banner missing %q: %q", want, full)
		}
	}
}

// TestRenderInteractiveBlockCopyBanner proves the render swaps the tab bar for the copy banner
// while frozen (and only then): normal mode shows the tab chips + the discoverability hint and no
// banner; copy mode shows the banner + how-to on the top row and no tab chips.
func TestRenderInteractiveBlockCopyBanner(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := richVisualVars()
	tr.push(v)

	normal := renderGuardInfoInteractiveBlock(infoViewState{active: viewOverview}, v, tr, 120, 0)
	copyOn := renderGuardInfoInteractiveBlock(infoViewState{active: viewOverview, copyMode: true}, v, tr, 120, 0)

	normalTop := strings.SplitN(normal, "\n", 2)[0]
	copyTop := strings.SplitN(copyOn, "\n", 2)[0]

	if !strings.Contains(normalTop, "c copy") {
		t.Errorf("normal tab bar must advertise the copy hint, got:\n%s", normalTop)
	}
	if strings.Contains(normal, "COPY MODE") {
		t.Errorf("normal mode must not show the copy banner:\n%s", normal)
	}
	for _, want := range []string{"COPY MODE", "select", "resume"} {
		if !strings.Contains(copyTop, want) {
			t.Errorf("copy banner missing %q, got:\n%s", want, copyTop)
		}
	}
	if strings.Contains(copyTop, "«") {
		t.Errorf("copy mode must replace the tab bar (no tab chips) on the top row, got:\n%s", copyTop)
	}
}
