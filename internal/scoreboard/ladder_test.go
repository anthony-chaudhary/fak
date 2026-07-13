package scoreboard

import (
	"strings"
	"testing"
)

// goodCase is a provenance-complete, freshly-passing PR-tier case. Tests mutate a
// copy of it to plant a defect or strip a required field, so each assertion isolates
// exactly one deviation from a known-green row.
func goodCase(id string) LadderCase {
	return LadderCase{
		ID:        id,
		Model:     "qwen3-0.6b",
		Tokenizer: "qwen3-bpe",
		Engine:    "fak-cpu",
		Oracle:    "greedy-token-diff",
		Revision:  "abc123",
		Tolerance: "exact",
		Baseline:  "golden@abc123",
		Tier:      TierPR,
		Cost:      "0.4s / 1 CPU",
		Status:    StatusPass,
	}
}

// TestLadderWitnessPlantedDefect is the issue #4585 witness: a captured proof that
// the same case FAILS with a localized, replayable defect and PASSES after the fix,
// rendered by a pure fold that replays identically in this clean test process.
func TestLadderWitnessPlantedDefect(t *testing.T) {
	// --- planted representative defect: the engine emitted "decreased" where the
	// reference emitted "increased" at token 7 — a fluent-but-wrong decode. ---
	defect := goodCase("exec-summary-grounding")
	defect.Status = StatusFail
	defect.FirstDivergence = &LadderDivergence{Index: 7, Reference: "increased", Engine: "decreased"}
	defect.Replay = ".dispatch-runs/quality/exec-summary-grounding.bundle.json"

	before := LadderInput{
		Title: "quality ladder",
		Cases: []LadderCase{goodCase("greedy-decode"), defect},
	}.Render()

	if before.Verdict != "ACTION" {
		t.Fatalf("planted defect must force ACTION, got %q\n%s", before.Verdict, before.Summary())
	}
	if before.Counts[StatusFail] != 1 {
		t.Fatalf("expected exactly 1 fail, got %d\n%s", before.Counts[StatusFail], before.Summary())
	}
	v, ok := before.FirstActionable()
	if !ok {
		t.Fatal("failing board must surface a first actionable divergence")
	}
	if v.ID != "exec-summary-grounding" || v.FirstDivergence == nil || v.FirstDivergence.Index != 7 {
		t.Fatalf("first actionable must localize the defect, got %+v", v)
	}
	if strings.TrimSpace(v.Replay) == "" {
		t.Fatal("a failure must emit a scrubbed replay artifact")
	}
	summary := before.Summary()
	if !strings.Contains(summary, "increased") || !strings.Contains(summary, "replay:") {
		t.Fatalf("summary must show the divergence and replay artifact:\n%s", summary)
	}

	// --- after the fix: the engine now agrees; the case is a provenance-complete
	// pass. Nothing else about the board changed. ---
	fixed := goodCase("exec-summary-grounding")
	after := LadderInput{
		Title: "quality ladder",
		Cases: []LadderCase{goodCase("greedy-decode"), fixed},
	}.Render()

	if after.Verdict != "OK" {
		t.Fatalf("after the fix the board must be OK, got %q\n%s", after.Verdict, after.Summary())
	}
	if after.Counts[StatusPass] != 2 || after.Counts[StatusFail] != 0 {
		t.Fatalf("after the fix expected 2 pass / 0 fail, got %d/%d", after.Counts[StatusPass], after.Counts[StatusFail])
	}
	if _, ok := after.FirstActionable(); ok {
		t.Fatal("a green board must have no actionable divergence")
	}

	t.Logf("WITNESS before-fix board:\n%s", before.Summary())
	t.Logf("WITNESS after-fix board:\n%s", after.Summary())
}

// TestEffectiveHonestyGate proves the load-bearing invariant of acceptance
// criterion 3: missing or inconclusive evidence is NEVER pass. Each row is a case
// that a naive board might count green; Effective must refuse every one.
func TestEffectiveHonestyGate(t *testing.T) {
	withDivergence := func(c LadderCase) LadderCase {
		c.FirstDivergence = &LadderDivergence{Index: 1, Reference: "a", Engine: "b"}
		c.Replay = "bundle.json"
		return c
	}

	cases := []struct {
		name   string
		mutate func(LadderCase) LadderCase
		stale  int64
		want   LadderStatus
	}{
		{"clean pass", func(c LadderCase) LadderCase { return c }, 0, StatusPass},
		{"no status is no-data", func(c LadderCase) LadderCase { c.Status = ""; return c }, 0, StatusNoData},
		{"pass missing model is inconclusive", func(c LadderCase) LadderCase { c.Model = ""; return c }, 0, StatusInconclusive},
		{"pass missing tokenizer is inconclusive", func(c LadderCase) LadderCase { c.Tokenizer = ""; return c }, 0, StatusInconclusive},
		{"pass missing engine is inconclusive", func(c LadderCase) LadderCase { c.Engine = ""; return c }, 0, StatusInconclusive},
		{"pass missing revision is inconclusive", func(c LadderCase) LadderCase { c.Revision = ""; return c }, 0, StatusInconclusive},
		{"pass with neither seed nor oracle is inconclusive", func(c LadderCase) LadderCase { c.Oracle = ""; c.Seed = 0; return c }, 0, StatusInconclusive},
		{"pass with a seed and no oracle is fine", func(c LadderCase) LadderCase { c.Oracle = ""; c.Seed = 42; return c }, 0, StatusPass},
		{"pass missing tolerance and baseline is inconclusive", func(c LadderCase) LadderCase { c.Tolerance = ""; c.Baseline = ""; return c }, 0, StatusInconclusive},
		{"pass missing tier is inconclusive", func(c LadderCase) LadderCase { c.Tier = ""; return c }, 0, StatusInconclusive},
		{"fail without divergence is inconclusive", func(c LadderCase) LadderCase { c.Status = StatusFail; c.Replay = "b.json"; return c }, 0, StatusInconclusive},
		{"fail without replay is inconclusive", func(c LadderCase) LadderCase {
			c.Status = StatusFail
			c.FirstDivergence = &LadderDivergence{Index: 0, Reference: "a", Engine: "b"}
			return c
		}, 0, StatusInconclusive},
		{"localized fail with replay is fail", func(c LadderCase) LadderCase { c.Status = StatusFail; return withDivergence(c) }, 0, StatusFail},
		{"fresh pass under window", func(c LadderCase) LadderCase { c.AgeSeconds = 10; return c }, 100, StatusPass},
		{"aged pass past window is stale", func(c LadderCase) LadderCase { c.AgeSeconds = 200; return c }, 100, StatusStale},
		{"skipped passes through", func(c LadderCase) LadderCase { c.Status = StatusSkipped; return c }, 0, StatusSkipped},
		{"declared inconclusive stays inconclusive", func(c LadderCase) LadderCase { c.Status = StatusInconclusive; return c }, 0, StatusInconclusive},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.mutate(goodCase("c")).Effective(tc.stale)
			if got != tc.want {
				t.Fatalf("Effective = %q, want %q", got, tc.want)
			}
			// The universal invariant: anything that is not a clean, backed pass
			// must not be reported as pass.
			if tc.want != StatusPass && got == StatusPass {
				t.Fatalf("honesty gate breached: unbacked evidence rendered pass")
			}
		})
	}
}

// TestRenderCoversAllStates checks acceptance criterion 1: the render reports a
// count for each of the six states (including zero-lines), tracks covered
// revisions, and diffs prior status into regressions and improvements.
func TestRenderCoversAllStates(t *testing.T) {
	pass := goodCase("pass-1")

	failCase := goodCase("fail-1")
	failCase.Status = StatusFail
	failCase.FirstDivergence = &LadderDivergence{Index: 2, Reference: "x", Engine: "y"}
	failCase.Replay = "f.json"
	failCase.Revision = "def456"
	failCase.Prev = StatusPass // a regression: was green, now red

	stale := goodCase("stale-1")
	stale.AgeSeconds = 999

	skipped := goodCase("skip-1")
	skipped.Status = StatusSkipped

	inconclusive := goodCase("inc-1")
	inconclusive.Status = StatusPass
	inconclusive.Model = "" // strips provenance -> inconclusive

	noData := goodCase("nodata-1")
	noData.Status = ""

	recovered := goodCase("rec-1")
	recovered.Prev = StatusFail // an improvement: was red, now green

	d := LadderInput{
		Cases:      []LadderCase{pass, failCase, stale, skipped, inconclusive, noData, recovered},
		StaleAfter: 100,
	}.Render()

	want := map[LadderStatus]int{
		StatusPass:         2, // pass-1 + rec-1
		StatusFail:         1,
		StatusStale:        1,
		StatusSkipped:      1,
		StatusInconclusive: 1,
		StatusNoData:       1,
	}
	for s, n := range want {
		if d.Counts[s] != n {
			t.Errorf("Counts[%s] = %d, want %d", s, d.Counts[s], n)
		}
	}
	// Every state must be a present key, even a zero one — the board is never
	// allowed to silently omit a state.
	for _, s := range allStatuses {
		if _, ok := d.Counts[s]; !ok {
			t.Errorf("Counts missing state %s (must render a zero-line)", s)
		}
	}
	if d.Verdict != "ACTION" {
		t.Errorf("mixed board must be ACTION, got %q", d.Verdict)
	}
	if len(d.Regressions) != 1 || d.Regressions[0] != "fail-1" {
		t.Errorf("regressions = %v, want [fail-1]", d.Regressions)
	}
	if len(d.Improvements) != 1 || d.Improvements[0] != "rec-1" {
		t.Errorf("improvements = %v, want [rec-1]", d.Improvements)
	}
	// Covered revisions are deduped and sorted; abc123 appears on several cases.
	if strings.Join(d.Revisions, ",") != "abc123,def456" {
		t.Errorf("revisions = %v, want [abc123 def456]", d.Revisions)
	}
	// Slices bucket by tier and engine.
	if d.Slices["tier:pr"][StatusPass] != 2 {
		t.Errorf("tier:pr pass slice = %d, want 2", d.Slices["tier:pr"][StatusPass])
	}
	if d.Slices["engine:fak-cpu"][StatusFail] != 1 {
		t.Errorf("engine:fak-cpu fail slice = %d, want 1", d.Slices["engine:fak-cpu"][StatusFail])
	}
}

// TestEmptyBoardIsNeverGreen guards the ladder-level restatement of "no evidence
// is never pass": a board with no cases is ACTION, not OK.
func TestEmptyBoardIsNeverGreen(t *testing.T) {
	d := LadderInput{Title: "empty"}.Render()
	if d.Verdict != "ACTION" {
		t.Fatalf("empty board must be ACTION, got %q", d.Verdict)
	}
	if !strings.Contains(d.detailLine(), "never green") {
		t.Fatalf("empty board detail should explain itself, got %q", d.detailLine())
	}
}

// TestToUpdatePublishSurface checks the board folds into the shared Update
// publish surface: the verdict drives the channel glyph and the first actionable
// divergence + replay ride the next-step line.
func TestToUpdatePublishSurface(t *testing.T) {
	failCase := goodCase("fail-1")
	failCase.Status = StatusFail
	failCase.FirstDivergence = &LadderDivergence{Index: 3, Reference: "up", Engine: "down"}
	failCase.Replay = "f.json"

	u := LadderInput{Title: "quality ladder", Cases: []LadderCase{goodCase("ok-1"), failCase}}.ToUpdate("ci")
	if u.Verdict != "ACTION" {
		t.Fatalf("update verdict = %q, want ACTION", u.Verdict)
	}
	txt := u.Text()
	if !strings.Contains(txt, ":red_circle:") {
		t.Errorf("ACTION board must show the action glyph:\n%s", txt)
	}
	if !strings.Contains(u.NextStep, "fail-1") || !strings.Contains(u.NextStep, "f.json") {
		t.Errorf("next-step must carry the actionable case and its replay: %q", u.NextStep)
	}
	if !strings.Contains(txt, "pass: 1") || !strings.Contains(txt, "fail: 1") {
		t.Errorf("update must carry per-status counts:\n%s", txt)
	}

	// A clean board publishes green.
	ok := LadderInput{Title: "quality ladder", Cases: []LadderCase{goodCase("ok-1")}}.ToUpdate("ci")
	if ok.Verdict != "OK" || !strings.Contains(ok.Text(), ":white_check_mark:") {
		t.Errorf("clean board must publish green, got verdict %q:\n%s", ok.Verdict, ok.Text())
	}
}
