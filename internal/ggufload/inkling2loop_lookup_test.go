package ggufload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// lookupOf syncs d to history and returns its raw lookup, before the cost gate.
func lookupOf(d *inkling2LookupDrafter, history []int, k int) []int {
	d.sync(history)
	got, _ := d.lookup(k)
	return got
}

func TestInkling2LookupDrafterMissDraftsNothing(t *testing.T) {
	d := newInkling2LookupDrafter([]int{1, 2, 3, 4, 5})
	if got := d.Draft([]int{6, 7}, 4); got != nil {
		t.Fatalf("no repeated suffix drafted %v", got)
	}
	if got := d.Draft([]int{6, 7, 8}, 0); got != nil {
		t.Fatalf("depth 0 drafted %v", got)
	}
	// a two-token repeat is below the minimum match
	if got := newInkling2LookupDrafter([]int{1, 2, 9, 4}).Draft([]int{1, 2}, 4); got != nil {
		t.Fatalf("a two-token match drafted %v", got)
	}
}

func TestInkling2LookupDrafterCopiesPromptContinuation(t *testing.T) {
	prompt := []int{10, 11, 12, 13, 14, 15, 16, 17, 18}
	d := newInkling2LookupDrafter(prompt)
	// history ends on 11 12 13, which the prompt continues with 14 15 16 ...
	if got, want := lookupOf(d, []int{11, 12, 13}, 4), []int{14, 15, 16, 17}; !slices.Equal(got, want) {
		t.Fatalf("drafted %v, want %v", got, want)
	}
	if got, want := lookupOf(d, []int{11, 12, 13, 14}, 2), []int{15, 16}; !slices.Equal(got, want) {
		t.Fatalf("depth 2 drafted %v, want %v", got, want)
	}
	// near the end of the prompt the continuation is shorter than k
	if got, want := lookupOf(d, []int{11, 12, 13, 14, 15, 16, 17}, 8), []int{18, 11, 12, 13, 14, 15, 16, 17}; !slices.Equal(got, want) {
		t.Fatalf("tail drafted %v, want %v", got, want)
	}
}

func TestInkling2LookupDrafterPrefersMostRecentMatch(t *testing.T) {
	d := newInkling2LookupDrafter([]int{4, 5, 6, 1, 9, 9, 4, 5, 6, 2})
	if got, want := lookupOf(d, []int{4, 5, 6}, 1), []int{2}; !slices.Equal(got, want) {
		t.Fatalf("drafted %v, want the most recent continuation %v", got, want)
	}
}

// The incremental lookup must draft exactly what a fresh scan of
// prompt+history drafts, including after the history is replaced.
func TestInkling2LookupDrafterIncrementalMatchesFullScan(t *testing.T) {
	prompt := make([]int, 0, 200)
	for i := range 200 {
		prompt = append(prompt, (i*7+i/13)%23)
	}
	ref := model.NgramDrafter{Enabled: true, MinMatch: inkling2LookupMinMatch, MaxMatch: inkling2LookupMaxMatch}
	d := newInkling2LookupDrafter(prompt)
	var history []int
	check := func(k int) {
		t.Helper()
		r := ref
		r.MaxDraft = k
		want := r.Draft(append(slices.Clone(prompt), history...))
		if got := lookupOf(d, history, k); !slices.Equal(got, want) {
			t.Fatalf("history %d tokens, k %d: drafted %v, full scan %v", len(history), k, got, want)
		}
	}
	for i := range 150 {
		history = append(history, (i*5+3)%23)
		if i%3 == 0 {
			history = append(history, (i*11)%23, i%23)
		}
		check(1 + i%8)
	}
	history = []int{4, 4, 4}
	check(4)
	history = []int{3, 1, 4, 1, 5}
	check(8)
}

// repeatPrompt is a prompt whose second half repeats its first, so a history
// that copies it always has a long match.
func repeatPrompt() []int {
	p := make([]int, 0, 80)
	for i := range 40 {
		p = append(p, 100+i)
	}
	return append(p, p...)
}

func TestInkling2LookupDrafterCapsDepthByMatchLength(t *testing.T) {
	d := newInkling2LookupDrafter(repeatPrompt())
	if got := d.Draft([]int{7, 100, 101, 102}, 8); len(got) > 2 {
		t.Fatalf("a three-token match drafted %d tokens, want at most 2", len(got))
	}
	h := make([]int, 0, 20)
	for i := range 20 {
		h = append(h, 100+i)
	}
	// long matches that keep their whole draft earn deeper drafts
	for range 6 {
		props := d.Draft(h, 8)
		if len(props) == 0 {
			t.Fatal("a long match drafted nothing")
		}
		d.Observe(nil, make([]int, len(props)+1), len(props)+1)
	}
	// with the M3 Pro prior, four rows per verify are the most tokens per time
	if got := d.Draft(h, 8); len(got) < 4 {
		t.Fatalf("a long match after full accepts drafted %d tokens, want at least 4", len(got))
	}
}

func TestInkling2LookupDrafterBacksOffAfterMisses(t *testing.T) {
	d := newInkling2LookupDrafter(repeatPrompt())
	h := []int{100, 101, 102, 103, 104, 105}
	miss := func() {
		t.Helper()
		props := d.Draft(h, 4)
		if len(props) == 0 {
			t.Fatal("a long match drafted nothing")
		}
		d.Observe(nil, append([]int{0}, props...), 1)
	}
	plain := func(n int) {
		t.Helper()
		for i := range n {
			if got := d.Draft(h, 4); got != nil {
				t.Fatalf("plain step %d of the back-off drafted %v", i, got)
			}
		}
	}
	miss()
	plain(1)
	miss()
	plain(2)
	// enough misses and the expected gain no longer pays for a verify
	for range 12 {
		if props := d.Draft(h, 4); len(props) > 0 {
			d.Observe(nil, append([]int{0}, props...), 1)
		}
	}
	if p := d.acc[0] / (d.acc[0] + d.rej[0]); p > 0.01 {
		t.Fatalf("accept estimate %.2f after only misses", p)
	}
}

func TestInkling2LookupDrafterGatesOnMeasuredCost(t *testing.T) {
	h := []int{100, 101, 102, 103, 104, 105}
	d := newInkling2LookupDrafter(repeatPrompt())
	d.ObserveCost(1, 40*time.Millisecond)
	for w := 2; w <= inkling2SpecMaxDraft+1; w++ {
		d.ObserveCost(w, time.Duration(w)*40*time.Millisecond)
	}
	// a verify costing a plain step per row can never beat plain decode
	if got := d.Draft(h, 4); got != nil {
		t.Fatalf("drafted %v when every row costs a plain step", got)
	}
	d = newInkling2LookupDrafter(repeatPrompt())
	d.ObserveCost(1, 40*time.Millisecond)
	for w := 2; w <= inkling2SpecMaxDraft+1; w++ {
		d.ObserveCost(w, 42*time.Millisecond)
	}
	if got := d.Draft(h, 4); len(got) != 4 {
		t.Fatalf("drafted %d tokens when a verify costs one plain step, want 4", len(got))
	}
}

func TestInkling2SpecDraftKind(t *testing.T) {
	for _, c := range []struct{ env, want string }{
		{"", inkling2DraftAuto},
		{"lookup", inkling2DraftLookup},
		{" DSpark ", inkling2DraftDSpark},
		{"dflash2", inkling2DraftDFlash},
		{"auto", inkling2DraftAuto},
	} {
		t.Setenv("FAK_INKLING2_SPEC_DRAFT", c.env)
		got, err := inkling2SpecDraftKind()
		if err != nil || got != c.want {
			t.Errorf("FAK_INKLING2_SPEC_DRAFT=%q: %q, %v; want %q", c.env, got, err, c.want)
		}
	}
	t.Setenv("FAK_INKLING2_SPEC_DRAFT", "eagle")
	if _, err := inkling2SpecDraftKind(); err == nil {
		t.Error("an unknown draft source was accepted")
	}
}

func TestInkling2ResolveSpecDraftAutoNeedsWeights(t *testing.T) {
	empty := t.TempDir()
	withWeights := t.TempDir()
	if err := os.WriteFile(filepath.Join(withWeights, inkling2DSparkWeightsFile), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	asDir := t.TempDir()
	if err := os.Mkdir(filepath.Join(asDir, inkling2DSparkWeightsFile), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ kind, dir, want string }{
		{inkling2DraftAuto, "", inkling2DraftLookup},
		{inkling2DraftAuto, empty, inkling2DraftLookup},
		{inkling2DraftAuto, asDir, inkling2DraftLookup},
		{inkling2DraftAuto, withWeights, inkling2DraftDSpark},
		{inkling2DraftLookup, withWeights, inkling2DraftLookup},
		{inkling2DraftDSpark, "", inkling2DraftDSpark},
		{inkling2DraftDFlash, "", inkling2DraftDFlash},
	} {
		if got := inkling2ResolveSpecDraft(c.kind, c.dir); got != c.want {
			t.Errorf("resolve(%q, %q) = %q, want %q", c.kind, c.dir, got, c.want)
		}
	}
}
