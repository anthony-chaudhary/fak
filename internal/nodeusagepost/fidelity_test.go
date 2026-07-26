package nodeusagepost

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// The fidelity scorer's contract has two halves, proven separately below:
//
//  1. SOUNDNESS — every card the HONEST renderer produces, across every bucket, scores a
//     perfect 100. If the scorer flagged a faithful card it would cry wolf and the gate
//     would be worthless. This corpus folds real snapshots through fleet.Fold +
//     FromSnapshot (never a hand-rolled Update) so it tracks the renderer exactly.
//
//  2. SENSITIVITY — for each lie the package was built to prevent, corrupting an honest
//     card to reintroduce that lie drops Pass to false and trips the SPECIFIC named check.
//     A scorer that always returned 100 would pass half (1) and is caught here.

// --- (1) soundness: honest cards score 100 ----------------------------------------------

// corpusCase is one labelled snapshot; scoreHonest folds it and scores the resulting card.
type corpusCase struct {
	name    string
	boxes   []fleet.Box
	reports []fleet.Report
	// hand is an optional pre-folded snapshot for shapes fleet.Fold cannot produce from a
	// roster (e.g. down-with-error, where Fold derives Reachable from the report itself).
	hand *fleet.Snapshot
}

func (c corpusCase) snap() fleet.Snapshot {
	if c.hand != nil {
		return *c.hand
	}
	return foldRoster(c.boxes, c.reports)
}

func honestCorpus() []corpusCase {
	live := func(v string) fleet.Report { return fleet.Report{State: fleet.StateLive, Version: v} }
	nb := func(ids ...string) []fleet.Box {
		out := make([]fleet.Box, len(ids))
		for i, id := range ids {
			out[i] = fleet.Box{ID: id}
		}
		return out
	}
	mostlyHealthy := func(down int) corpusCase {
		var boxes []fleet.Box
		var reports []fleet.Report
		for i := 0; i < 9; i++ {
			boxes = append(boxes, fleet.Box{ID: string(rune('a' + i))})
			reports = append(reports, live("v1"))
		}
		for i := 0; i < down; i++ {
			boxes = append(boxes, fleet.Box{ID: "d" + string(rune('0'+i))})
			reports = append(reports, fleet.Report{State: fleet.StateDown})
		}
		return corpusCase{name: "mostly-healthy-one-down", boxes: boxes, reports: reports}
	}

	return []corpusCase{
		{name: "empty-roster"},
		{
			name:    "all-healthy-one-version",
			boxes:   []fleet.Box{{ID: "a1", Class: "a100x8"}, {ID: "a2", Class: "a100x8"}},
			reports: []fleet.Report{live("v1"), live("v1")},
		},
		{
			name:    "all-silent-visibility-gap",
			boxes:   []fleet.Box{{ID: "a1", Class: "a100x8"}, {ID: "a2", Class: "metal"}},
			reports: nil,
		},
		{
			name:    "partial-visibility",
			boxes:   nb("a1", "a2", "a3"),
			reports: []fleet.Report{live("v1")},
		},
		{
			name:    "one-reported-down",
			boxes:   nb("a1", "a2"),
			reports: []fleet.Report{live("v1"), {State: fleet.StateDown}},
		},
		mostlyHealthy(1),
		{
			name:    "version-skew",
			boxes:   append(nb("h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"), fleet.Box{ID: "skew"}),
			reports: []fleet.Report{live("v1"), live("v1"), live("v1"), live("v1"), live("v1"), live("v1"), live("v1"), live("v1"), live("v1"), live("v2")},
		},
		{
			name:  "gpu-waste",
			boxes: []fleet.Box{{ID: "a1", Class: "a100x8"}, {ID: "a2", Class: "a100x8"}},
			reports: []fleet.Report{
				{State: fleet.StateLive, Version: "v1", GPU: &fleet.GPUStats{Total: 8, Busy: 1, UtilPct: 5}},
				{State: fleet.StateLive, Version: "v1", GPU: &fleet.GPUStats{Total: 8, Busy: 8, UtilPct: 95}},
			},
		},
		{
			name: "down-with-error", // Reachable==0 yet a real outage — the down-hidden-as-silence shape
			hand: &fleet.Snapshot{
				Schema:  fleet.SnapshotSchema,
				Total:   2,
				ByState: map[fleet.State]int{fleet.StateDown: 2},
			},
		},
	}
}

func TestFidelityHonestCardsAlwaysScore100(t *testing.T) {
	for _, c := range honestCorpus() {
		t.Run(c.name, func(t *testing.T) {
			snap := c.snap()
			up := FromSnapshot(snap, "test")
			f := Score(up, snap)
			if !f.Pass || f.Score != 100 {
				t.Fatalf("honest card scored %d (pass=%v); an honest renderer must be perfect.\nviolations: %s",
					f.Score, f.Pass, strings.Join(f.Violations, " | "))
			}
			if f.Grade != "A" {
				t.Fatalf("score 100 must grade A, got %q", f.Grade)
			}
			if len(f.Checks) == 0 {
				t.Fatal("no checks ran — the scorer must always verify at least the headline counts")
			}
		})
	}
}

// TestFidelitySchemaAndDenominator guards the invariant that a non-applicable check is
// OMITTED (never a free pass): the empty roster runs strictly fewer checks than a
// fully-populated fleet, and both still score 100.
func TestFidelitySchemaAndDenominator(t *testing.T) {
	empty := Score(FromSnapshot(foldRoster(nil, nil), "t"), foldRoster(nil, nil))
	full := honestCorpus()[7] // gpu-waste: exercises GPU + state + class + problem checks
	rich := Score(FromSnapshot(full.snap(), "t"), full.snap())

	if empty.Schema != FidelitySchema {
		t.Fatalf("schema = %q, want %q", empty.Schema, FidelitySchema)
	}
	if len(empty.Checks) >= len(rich.Checks) {
		t.Fatalf("empty roster ran %d checks, a populated fleet ran %d — non-applicable checks must be omitted, not passed for free",
			len(empty.Checks), len(rich.Checks))
	}
}

// --- (2) sensitivity: each lie trips its own check --------------------------------------

// checkFailed reports whether the named check is present AND failed.
func checkFailed(f Fidelity, name string) bool {
	for _, c := range f.Checks {
		if c.Name == name {
			return !c.Pass
		}
	}
	return false
}

// honestBase returns a real down-fleet card + snapshot to corrupt. A down fleet exercises
// the most honesty checks (down-forces-action, problem-not-green, down-is-named), so it is
// the richest substrate for the tamper tests.
func honestBase(t *testing.T) (scoreboard.Update, fleet.Snapshot) {
	t.Helper()
	snap := foldRoster(
		[]fleet.Box{{ID: "a1", Class: "a100x8"}, {ID: "a2", Class: "a100x8"}},
		[]fleet.Report{{State: fleet.StateLive, Version: "v1"}, {State: fleet.StateDown}},
	)
	up := FromSnapshot(snap, "test")
	if f := Score(up, snap); !f.Pass {
		t.Fatalf("base card is not honest: %s", strings.Join(f.Violations, " | "))
	}
	return up, snap
}

func TestFidelityCatchesFakedReachableHeadline(t *testing.T) {
	up, snap := honestBase(t)
	up.Score = "99/99 reachable" // inflate the headline the snapshot does not support
	f := Score(up, snap)
	if f.Pass || !checkFailed(f, "reachable-headline") {
		t.Fatalf("a faked reachable headline must trip reachable-headline; got pass=%v violations=%v", f.Pass, f.Violations)
	}
}

func TestFidelityCatchesGreenGradeOverRealDown(t *testing.T) {
	up, snap := honestBase(t)
	up.Grade = "A" // the founding lie: paint a real outage green
	f := Score(up, snap)
	// The renderer keys its glyph off an A/B prefix before the verdict, so BOTH the
	// problem-not-green and down-forces-action honesty checks must catch this.
	if f.Pass {
		t.Fatal("an A grade over a reported-down fleet must not pass")
	}
	if !checkFailed(f, "problem-not-green") {
		t.Fatalf("green-over-down must trip problem-not-green; violations=%v", f.Violations)
	}
	if !checkFailed(f, "down-forces-action") {
		t.Fatalf("green-over-down must trip down-forces-action; violations=%v", f.Violations)
	}
}

func TestFidelityCatchesSilentRelabeledAsDown(t *testing.T) {
	// The all-silent visibility gap is the original bug: a card that calls silent boxes
	// "down" or drops the neutral read must fail no-visibility-neutral.
	snap := foldRoster([]fleet.Box{{ID: "a1"}, {ID: "a2"}}, nil)
	up := FromSnapshot(snap, "test")
	if f := Score(up, snap); !f.Pass {
		t.Fatalf("honest visibility-gap card must pass first: %v", f.Violations)
	}
	up.Grade = "F"
	up.Verdict = "ACTION"
	up.Detail = "2 box(es) down or unreachable"
	f := Score(up, snap)
	if f.Pass || !checkFailed(f, "no-visibility-neutral") {
		t.Fatalf("relabeling silent boxes as a down outage must trip no-visibility-neutral; pass=%v violations=%v", f.Pass, f.Violations)
	}
	if !checkFailed(f, "no-phantom-down") {
		t.Fatalf("asserting 'down' over a zero-down snapshot must also trip no-phantom-down; violations=%v", f.Violations)
	}
}

func TestFidelityCatchesDroppedStateCount(t *testing.T) {
	up, snap := honestBase(t)
	// Drop the "down: 1" state line — a silently omitted count.
	var kept []string
	for _, l := range up.Lines {
		if !strings.HasPrefix(l, "down:") {
			kept = append(kept, l)
		}
	}
	up.Lines = kept
	f := Score(up, snap)
	if f.Pass || !checkFailed(f, "state-counts-complete") {
		t.Fatalf("a dropped state count must trip state-counts-complete; pass=%v violations=%v", f.Pass, f.Violations)
	}
}

func TestFidelityCatchesWrongGPUNumbers(t *testing.T) {
	full := honestCorpus()[7] // gpu-waste
	snap := full.snap()
	up := FromSnapshot(snap, "test")
	// Rewrite the gpu line with numbers the aggregate does not support.
	for i, l := range up.Lines {
		if strings.HasPrefix(l, "gpu capacity:") {
			up.Lines[i] = "gpu capacity: busy 16/16, idle 0 (100% util)"
		}
	}
	f := Score(up, snap)
	if f.Pass || !checkFailed(f, "gpu-line-fidelity") {
		t.Fatalf("a doctored GPU line must trip gpu-line-fidelity; pass=%v violations=%v", f.Pass, f.Violations)
	}
}

func TestFidelityCatchesMalformedVerdict(t *testing.T) {
	up, snap := honestBase(t)
	up.Verdict = "MELTDOWN" // outside the renderer's closed OK|ACTION vocabulary
	f := Score(up, snap)
	if f.Pass || !checkFailed(f, "verdict-wellformed") {
		t.Fatalf("an unknown verdict must trip verdict-wellformed; pass=%v violations=%v", f.Pass, f.Violations)
	}
}

// TestFidelityScoreDegradesWithMoreLies proves Score is a real gradient, not a boolean:
// each additional independent lie strictly lowers the weighted score.
func TestFidelityScoreDegradesWithMoreLies(t *testing.T) {
	up, snap := honestBase(t)
	clean := Score(up, snap).Score

	up.Score = "99/99 reachable"
	oneLie := Score(up, snap).Score

	up.Verdict = "MELTDOWN"
	twoLies := Score(up, snap).Score

	if !(clean == 100 && clean > oneLie && oneLie > twoLies) {
		t.Fatalf("score must degrade monotonically with lies: clean=%d oneLie=%d twoLies=%d", clean, oneLie, twoLies)
	}
}

func TestRenderFidelityShowsFailuresProminently(t *testing.T) {
	up, snap := honestBase(t)
	up.Grade = "A"
	out := RenderFidelity(Score(up, snap))
	if !strings.Contains(out, "FAIL") {
		t.Fatalf("render must surface FAIL for a corrupted card:\n%s", out)
	}
	if !strings.Contains(out, "problem-not-green") {
		t.Fatalf("render must name the failed check:\n%s", out)
	}
	// An honest card renders PASS with no FAIL rows.
	okOut := RenderFidelity(Score(FromSnapshot(snap, "t"), snap))
	if !strings.Contains(okOut, "PASS") || strings.Contains(okOut, "[FAIL]") {
		t.Fatalf("honest card must render a clean PASS:\n%s", okOut)
	}
}
