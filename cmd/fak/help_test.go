package main

import (
	"strings"
	"testing"
)

// The compact-overview taste gate. `fak --help` regressed into a ~650-line wall
// once; these tests hold the new shape: the overview stays under a screen, every
// verb it advertises is real, the per-verb carver finds the deep documentation,
// and a mistyped verb gets a suggestion instead of the wall.

// TestUsageCompactStaysCompact is the anti-wall ratchet: the curated overview
// must fit comfortably on one screen. If this reds, cut overview entries — do
// not widen the budget; depth belongs in `fak help <verb>` and `fak help --all`.
func TestUsageCompactStaysCompact(t *testing.T) {
	var b strings.Builder
	usageCompact(&b)
	lines := strings.Split(strings.TrimRight(b.String(), "\n"), "\n")
	if len(lines) > 45 {
		t.Fatalf("compact overview is %d lines; the budget is 45 — trim overviewGroups, don't grow the wall back", len(lines))
	}
	for _, ln := range lines {
		if n := len([]rune(ln)); n > 100 {
			t.Errorf("overview line %d runes wide (max 100): %q", n, ln)
		}
	}
}

// TestOverviewVerbsAreLive pins every curated overview verb to the live verb
// catalog (dispatch-derived when the repo is readable), so the overview can
// never advertise a verb the binary does not route.
func TestOverviewVerbsAreLive(t *testing.T) {
	cat := helpCatalog()
	if cat == nil {
		t.Skip("devindex catalog unavailable (no repo root); overview membership is only checkable in-repo")
	}
	live := map[string]bool{}
	for _, v := range cat.Verbs() {
		for _, sp := range v.Spellings() {
			live[strings.ToLower(sp)] = true
		}
	}
	for _, g := range overviewGroups {
		for _, e := range g.entries {
			if !live[e.name] {
				t.Errorf("overview advertises %q under %q but no dispatched verb has that spelling", e.name, g.title)
			}
		}
	}
}

// TestVerbWallSectionsCarvesDepth proves the per-verb carver recovers the deep
// documentation from the wall constants — the depth `fak --help` used to dump
// must remain reachable, verb by verb.
func TestVerbWallSectionsCarvesDepth(t *testing.T) {
	sections := verbWallSections([]string{"commit"})
	if len(sections) == 0 {
		t.Fatal("verbWallSections found no wall block for 'commit'")
	}
	joined := strings.Join(sections, "")
	if !strings.Contains(joined, "SAFE SHARED-TRUNK COMMIT") {
		t.Errorf("commit wall block lost its paragraph; got:\n%s", joined)
	}
	if strings.Contains(joined, "fak edit-tx") {
		t.Errorf("commit wall block leaked the next verb's section:\n%s", joined)
	}
	// A verb documented mid-wall with a one-line entry still carves cleanly.
	if s := verbWallSections([]string{"version"}); len(s) == 0 {
		t.Error("verbWallSections found no wall block for 'version'")
	}
	if s := verbWallSections([]string{"no-such-verb-ever"}); len(s) != 0 {
		t.Errorf("carver invented a section for an unknown verb: %q", s)
	}
}

// TestSuggestVerb pins did-you-mean: a near-miss typo maps to the real verb, and
// garbage maps to nothing rather than a random suggestion.
func TestSuggestVerb(t *testing.T) {
	if got := suggestVerb("comit"); got != "commit" {
		t.Errorf("suggestVerb(comit) = %q, want commit", got)
	}
	if got := suggestVerb("swep"); got != "sweep" {
		t.Errorf("suggestVerb(swep) = %q, want sweep", got)
	}
	if got := suggestVerb("zzqx"); got != "" {
		t.Errorf("suggestVerb(zzqx) = %q, want no suggestion", got)
	}
}
