package wipattr

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// The measured incident this file fences (2026-07-26 on the fak trunk ledger):
//
//   - internal/agent/loop.go ranked WAIT with 8 blocked admissions while its INDEX
//     still held the pre-#5235 blob and its worktree already equalled HEAD. Landing it
//     would have reverted commit 66e132fbf.
//   - internal/gateway/role_alternation.go and its test were staged DELETED while
//     byte-identical 14580/10650-byte files sat on disk. Landing them would have
//     removed 632 lines the trunk still has.
//
// Under path-and-mtime alone both read as ordinary WIP. These tests pin that Content
// makes them RESIDUE instead, and — just as importantly — that a caller who cannot
// probe still gets the old ranking rather than an emptied queue.

// TestResiduePreemptsEveryOtherVerdict is the core safety property: no combination of
// staleness and blocked admissions can promote a path that carries no new work.
func TestResiduePreemptsEveryOtherVerdict(t *testing.T) {
	cases := []struct {
		name    string
		content Content
		age     float64
	}{
		// Stale + blocking is the exact shape that would otherwise be LAND.
		{"stale index, abandoned", ContentMatchesHEAD, 9.0},
		{"phantom delete, abandoned", ContentPhantomDelete, 9.0},
		{"landed upstream, abandoned", ContentMatchesUpstream, 9.0},
		// Fresh + blocking would otherwise be WAIT. Age must not change the verdict:
		// a fresh stale-index entry is exactly as destructive to commit as an old one.
		{"stale index, minutes old", ContentMatchesHEAD, 0.01},
		{"phantom delete, minutes old", ContentPhantomDelete, 0.01},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rows := Rank(
				[]Blocker{{Path: "p.go", Set: "s", AgeDays: c.age, Content: c.content}},
				map[string]int{"p.go": 8}, DefaultStaleAfterDays)
			if len(rows) != 1 {
				t.Fatalf("want 1 row, got %d", len(rows))
			}
			if rows[0].State != BlockResidue {
				t.Fatalf("%s content = state %q, want %q (reason: %s)",
					c.content, rows[0].State, BlockResidue, rows[0].Reason)
			}
			// The reason must name the remedy, not merely the refusal: an operator
			// reading it has to know that CLEARING is the action, not committing.
			if !strings.Contains(rows[0].Reason, "instead") {
				t.Errorf("%s reason gives no remedy: %q", c.content, rows[0].Reason)
			}
			// Blocks stay visible — the admissions are real and recoverable, which is
			// the whole reason residue is worth ranking rather than hiding.
			if rows[0].Blocks != 8 {
				t.Errorf("blocks = %d, want 8 (residue still reports its cost)", rows[0].Blocks)
			}
			if rows[0].Content != c.content {
				t.Errorf("content = %s, want %s (carried through for the consumer)", rows[0].Content, c.content)
			}
		})
	}
}

// TestUnprobedContentPreservesPreContentRanking is the compatibility fence: a caller
// whose git probe failed leaves Content at its zero value and must get exactly the
// ranking this fold produced before Content existed. Degrading to an EMPTY queue would
// hide real work every time a git read hiccupped.
func TestUnprobedContentPreservesPreContentRanking(t *testing.T) {
	dirty := []Blocker{
		{Path: "stale/blocking.go", Set: "a", AgeDays: 9.0},
		{Path: "fresh/blocking.go", Set: "b", AgeDays: 0.01},
		{Path: "stale/quiet.go", Set: "c", AgeDays: 9.0},
		{Path: "fresh/quiet.go", Set: "d", AgeDays: 0.01},
	}
	blocks := map[string]int{"stale/blocking.go": 40, "fresh/blocking.go": 7}
	rows := Rank(dirty, blocks, DefaultStaleAfterDays)

	want := map[string]BlockState{
		"stale/blocking.go": BlockLand,
		"fresh/blocking.go": BlockWait,
		"stale/quiet.go":    BlockIdle,
		"fresh/quiet.go":    BlockActive,
	}
	for _, r := range rows {
		if r.State != want[r.Path] {
			t.Errorf("%s: state = %q, want %q — unprobed must rank as before", r.Path, r.State, want[r.Path])
		}
		if r.Content != ContentUnprobed {
			t.Errorf("%s: content = %s, want unprobed", r.Path, r.Content)
		}
	}
	if got := Residue(rows); len(got) != 0 {
		t.Errorf("unprobed rows must never be residue, got %d", len(got))
	}
	// ContentDiverged must agree with ContentUnprobed on every verdict — the probe
	// changes the ranking ONLY for the residue shapes.
	probed := make([]Blocker, len(dirty))
	for i, b := range dirty {
		b.Content = ContentDiverged
		probed[i] = b
	}
	for _, r := range Rank(probed, blocks, DefaultStaleAfterDays) {
		if r.State != want[r.Path] {
			t.Errorf("%s: diverged state = %q, want %q — probing real work must not change its rank",
				r.Path, r.State, want[r.Path])
		}
	}
}

// TestResidueRanksBelowLandAboveWait pins the queue order. Residue is actionable now
// (clear the entry, commit nothing), so it outranks WAIT — which cannot be acted on at
// all — but never outranks a genuine LAND row, which recovers admissions AND lands work.
func TestResidueRanksBelowLandAboveWait(t *testing.T) {
	rows := Rank([]Blocker{
		{Path: "z/wait.go", Set: "w", AgeDays: 0.01, Content: ContentDiverged},
		{Path: "z/residue-small.go", Set: "r1", AgeDays: 9.0, Content: ContentMatchesHEAD},
		{Path: "z/land.go", Set: "l", AgeDays: 9.0, Content: ContentDiverged},
		{Path: "z/idle.go", Set: "i", AgeDays: 9.0, Content: ContentDiverged},
		{Path: "z/residue-big.go", Set: "r2", AgeDays: 9.0, Content: ContentPhantomDelete},
		{Path: "z/active.go", Set: "a", AgeDays: 0.01, Content: ContentDiverged},
	}, map[string]int{
		"z/wait.go":          500, // most blocks on the board, still unactionable
		"z/land.go":          10,
		"z/residue-small.go": 2,
		"z/residue-big.go":   60,
	}, DefaultStaleAfterDays)

	got := make([]string, len(rows))
	for i, r := range rows {
		got[i] = r.Path
	}
	want := []string{
		"z/land.go",          // LAND
		"z/residue-big.go",   // RESIDUE, 60 blocks
		"z/residue-small.go", // RESIDUE, 2 blocks
		"z/wait.go",          // WAIT — 500 blocks, unactionable
		"z/idle.go",          // IDLE
		"z/active.go",        // ACTIVE
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("queue order = %v, want %v", got, want)
	}
}

// TestResidueBlocksStaySeparateFromBlocksRecovered is the arithmetic fence. The two
// totals must never be summed by a consumer, so they are never merged here: adding them
// would advertise admissions recoverable by a commit that reverts a peer.
func TestResidueBlocksStaySeparateFromBlocksRecovered(t *testing.T) {
	rows := Rank([]Blocker{
		{Path: "a/land.go", Set: "a", AgeDays: 9.0, Content: ContentDiverged},
		{Path: "b/stale-index.go", Set: "b", AgeDays: 9.0, Content: ContentMatchesHEAD},
		{Path: "c/phantom.go", Set: "c", AgeDays: 9.0, Content: ContentPhantomDelete},
	}, map[string]int{"a/land.go": 11, "b/stale-index.go": 8, "c/phantom.go": 21}, DefaultStaleAfterDays)

	if got := BlocksRecovered(rows); got != 11 {
		t.Errorf("BlocksRecovered = %d, want 11 (LAND only — residue must not inflate it)", got)
	}
	if got := ResidueBlocks(rows); got != 8+21 {
		t.Errorf("ResidueBlocks = %d, want %d", got, 8+21)
	}
	if got := len(Landable(rows)); got != 1 {
		t.Errorf("landable rows = %d, want 1", got)
	}
	paths := make([]string, 0, 2)
	for _, r := range Residue(rows) {
		paths = append(paths, r.Path)
	}
	if want := []string{"c/phantom.go", "b/stale-index.go"}; !reflect.DeepEqual(paths, want) {
		t.Errorf("residue queue = %v, want %v (most blocks first)", paths, want)
	}
}

// TestContentJSONIsANameNotAnOrdinal keeps the wire contract stable: a consumer must
// not have to track integer values that shift the moment a residue shape is added.
func TestContentJSONIsANameNotAnOrdinal(t *testing.T) {
	want := map[Content]string{
		ContentUnprobed:        "unprobed",
		ContentDiverged:        "diverged",
		ContentMatchesHEAD:     "stale-index",
		ContentPhantomDelete:   "phantom-delete",
		ContentMatchesUpstream: "landed-upstream",
	}
	for c, name := range want {
		if c.String() != name {
			t.Errorf("Content(%d).String() = %q, want %q", int(c), c.String(), name)
		}
		b, err := json.Marshal(c)
		if err != nil {
			t.Fatalf("marshal %s: %v", name, err)
		}
		if string(b) != `"`+name+`"` {
			t.Errorf("json(%s) = %s, want %q", name, b, name)
		}
	}
	// An unrecognised ordinal degrades to "unprobed" — the value that keeps the old
	// ranking — rather than to a residue shape that would suppress a row.
	if got := Content(99).String(); got != "unprobed" {
		t.Errorf("Content(99).String() = %q, want %q", got, "unprobed")
	}
}

func TestResidueEmptiesAreNonNil(t *testing.T) {
	if got := Residue(nil); got == nil || len(got) != 0 {
		t.Errorf("Residue(nil) = %v, want empty non-nil", got)
	}
	if got := ResidueBlocks(nil); got != 0 {
		t.Errorf("ResidueBlocks(nil) = %d, want 0", got)
	}
}
