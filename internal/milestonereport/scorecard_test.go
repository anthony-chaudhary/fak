package milestonereport

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// fixtureMaturity builds the CLIMB dimension from an explicit cell list so the
// debt is deterministic (the four cells land at M0/M1/M3/M4).
func fixtureMaturity() Maturity {
	return InterpretMaturity([]covmatrix.Cell{
		{Family: "a", Backend: "cpu", Support: covmatrix.Supported},       // M4 (matured)
		{Family: "a", Backend: "metal", Support: covmatrix.ProofPathOnly}, // M3 -> owes 1 rung
		{Family: "b", Backend: "cpu", Support: covmatrix.Fenced},          // M1 -> owes 3 rungs
		{Family: "b", Backend: "metal", Support: covmatrix.Undefined},     // M0 -> owes 4 rungs
	})
}

// fixtureEpics builds the ROADMAP dimension with two DISCRETE epics carrying open
// children. #1315 and #1178 are NOT in worktype's declared ongoing map, so both
// classify as discrete (#1010/#1301 are the ongoing programs — deliberately not
// used here, since a program owes no roadmap debt). A label source keeps the fold
// offline.
func fixtureEpics() Epics {
	specs := []EpicSpec{
		{Number: 1315, Title: "native agent harness"},
		{Number: 1178, Title: "first-class time horizons"},
	}
	counts := []EpicCounts{
		{Number: 1315, Closed: 2, Total: 5, Source: "label"}, // 3 open
		{Number: 1178, Closed: 4, Total: 5, Source: "label"}, // 1 open
	}
	return InterpretEpics(specs, counts, "")
}

// TestMilestoneDebtIsDeterministic pins the headline integer over a fixed
// grid+roadmap fixture: climb debt = (M4-M3)+(M4-M1)+(M4-M0) = 1+3+4 = 8 rung-steps
// over the three below-matured cells; roadmap debt = 3+1 = 4 open discrete children.
func TestMilestoneDebtIsDeterministic(t *testing.T) {
	m := fixtureMaturity()
	e := fixtureEpics()

	// Guard the fixture so a future worktype reclassification can't silently turn an
	// epic ongoing and quietly drop the roadmap-debt expectation.
	if e.Discrete != 2 {
		t.Fatalf("fixture expects 2 discrete epics, got %d (a worktype reclassification?)", e.Discrete)
	}

	p := BuildScorecard(m, e)

	wantClimb := (int(supportmaturity.M4Correct) - int(supportmaturity.M3Runs)) +
		(int(supportmaturity.M4Correct) - int(supportmaturity.M1Fenced)) +
		(int(supportmaturity.M4Correct) - int(supportmaturity.M0None))
	if wantClimb != 8 {
		t.Fatalf("fixture sanity: climb shortfall should be 8, computed %d", wantClimb)
	}
	const wantRoad = 4
	wantDebt := wantClimb + wantRoad

	if got := corpusInt(t, p, DebtKey); got != wantDebt {
		t.Fatalf("milestone_debt = %d, want %d (climb %d + roadmap %d)", got, wantDebt, wantClimb, wantRoad)
	}
	if got := corpusInt(t, p, "climb_debt"); got != wantClimb {
		t.Fatalf("climb_debt = %d, want %d", got, wantClimb)
	}
	if got := corpusInt(t, p, "roadmap_debt"); got != wantRoad {
		t.Fatalf("roadmap_debt = %d, want %d", got, wantRoad)
	}

	// Determinism: a second fold of the same inputs is bit-identical.
	p2 := BuildScorecard(fixtureMaturity(), fixtureEpics())
	if jsonOf(t, p) != jsonOf(t, p2) {
		t.Fatalf("the scorecard must be deterministic over a fixed grid+roadmap")
	}

	if p.Schema != ScorecardSchema {
		t.Fatalf("schema = %q, want %q", p.Schema, ScorecardSchema)
	}
	if p.OK {
		t.Fatalf("a fixture with debt must report OK=false")
	}
	if p.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION (debt present)", p.Verdict)
	}
}

// TestWorklistOrdersWorstFirst proves the retire worklist is climb-then-roadmap,
// each by descending severity. The worst climb item is the M0 bucket (4 rungs); the
// worst roadmap item is #1315 (3 open) ahead of #1010 (1 open).
func TestWorklistOrdersWorstFirst(t *testing.T) {
	p := BuildScorecard(fixtureMaturity(), fixtureEpics())
	wl := worklistOf(t, p)
	if len(wl) != 5 { // 3 climb buckets (M0/M1/M3) + 2 roadmap epics
		t.Fatalf("worklist len = %d, want 5: %+v", len(wl), wl)
	}

	// All climb items precede all roadmap items.
	var sawRoadmap bool
	for _, it := range wl {
		if it.Kind == "roadmap" {
			sawRoadmap = true
		} else if sawRoadmap {
			t.Fatalf("a climb item appeared after a roadmap item — climb must retire first: %+v", wl)
		}
	}

	// The very first item is the deepest climb debt: the M0 bucket (severity 4).
	if wl[0].Kind != "climb" || wl[0].Rung != "M0" || wl[0].Severity != 4 {
		t.Fatalf("worst item should be the M0 climb bucket (severity 4), got %+v", wl[0])
	}

	// Severity is non-increasing within each kind block.
	for i := 1; i < len(wl); i++ {
		if wl[i].Kind == wl[i-1].Kind && wl[i].Severity > wl[i-1].Severity {
			t.Fatalf("severity rose within a kind block at %d: %+v", i, wl)
		}
	}

	// The worst roadmap item is the most-open discrete epic (#1315, 3 open).
	var firstRoad *WorklistItem
	for i := range wl {
		if wl[i].Kind == "roadmap" {
			firstRoad = &wl[i]
			break
		}
	}
	if firstRoad == nil || firstRoad.Severity != 3 {
		t.Fatalf("worst roadmap item should have 3 open children (#1315), got %+v", firstRoad)
	}
}

// TestScorecardCleanWhenMatured proves a fully-matured grid with closed discrete
// epics is zero debt, OK, and carries an empty worklist — the --check 0 case.
func TestScorecardCleanWhenMatured(t *testing.T) {
	m := InterpretMaturity([]covmatrix.Cell{
		{Family: "a", Backend: "cpu", Support: covmatrix.Supported},
		{Family: "b", Backend: "cpu", Support: covmatrix.Supported},
	})
	e := InterpretEpics(
		[]EpicSpec{{Number: 1315, Title: "native agent harness"}},
		[]EpicCounts{{Number: 1315, Closed: 4, Total: 4, Source: "label"}},
		"",
	)
	p := BuildScorecard(m, e)
	if got := corpusInt(t, p, DebtKey); got != 0 {
		t.Fatalf("a matured grid + complete epic must be zero debt, got %d", got)
	}
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("zero debt must report OK/OK, got ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if wl := worklistOf(t, p); len(wl) != 0 {
		t.Fatalf("a clean scorecard must carry an empty worklist, got %+v", wl)
	}
}

// TestUnreadableEpicIsNotRoadmapDebt proves an epic whose gh read failed is NOT
// counted as retire-able debt (you cannot retire an unmeasured gap) — the report's
// ACTION verdict owns the unmeasured case, not the scorecard's debt count.
func TestUnreadableEpicIsNotRoadmapDebt(t *testing.T) {
	m := InterpretMaturity([]covmatrix.Cell{{Family: "a", Backend: "cpu", Support: covmatrix.Supported}})
	e := InterpretEpics(
		[]EpicSpec{
			{Number: 1315, Title: "readable"},
			{Number: 1178, Title: "unreadable"}, // discrete, but gh read failed
		},
		[]EpicCounts{
			{Number: 1315, Closed: 1, Total: 3, Source: "label"}, // 2 open
			{Number: 1178, Err: "gh: not found"},                 // excluded (unmeasured, not debt)
		},
		"",
	)
	p := BuildScorecard(m, e)
	if got := corpusInt(t, p, "roadmap_debt"); got != 2 {
		t.Fatalf("roadmap_debt = %d, want 2 (only the readable epic's open children)", got)
	}
}

// TestScorecardMarkdownSnapshotCarriesTheRealNumbers pins the `--markdown` committed
// snapshot surface (the propagate_markdown convention every sibling card already carries).
// The point of the gate is NOT that a markdown-shaped string comes out — a canned body
// would satisfy that — so this asserts the body is DERIVED from the same payload --json
// emits: the header cites the real milestone_debt, the header tail cites the real climb /
// roadmap / cell counts, both KPI rows are tabulated, and the HARD work list carries one
// bullet per owed rung-step and open child (len == milestone_debt, the kernel's own fold
// contract). Change the fixture's debt and every one of these must move with it.
func TestScorecardMarkdownSnapshotCarriesTheRealNumbers(t *testing.T) {
	p := BuildScorecard(fixtureMaturity(), fixtureEpics())
	debt := corpusInt(t, p, DebtKey) // 8 climb rung-steps + 4 open children = 12
	if debt != 12 {
		t.Fatalf("fixture sanity: milestone_debt should be 12, got %d", debt)
	}
	body := scorecard.Markdown(p, ScorecardMarkdownDoc(p))

	// Front matter + heading: the published page needs a <title>/description (the SEO card
	// reads them) and the auto-generated banner that forbids hand-editing.
	for _, want := range []string{
		"---\ntitle: \"fak milestone scorecard",
		"description: \"fak's deterministic milestone scorecard",
		"# Milestone scorecard",
		"_Auto-generated by `fak milestone-scorecard --markdown`",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("markdown snapshot missing %q:\n%s", want, body)
		}
	}

	// The header line and its tail cite the payload's REAL integers, not constants.
	for _, want := range []string{
		fmt.Sprintf("%s **%d**", DebtKey, debt),
		fmt.Sprintf("%d/%d cell(s) matured", corpusInt(t, p, "matured"), corpusInt(t, p, "cells")),
		fmt.Sprintf("%d climb rung(s)", corpusInt(t, p, "climb_debt")),
		fmt.Sprintf("%d open roadmap child(ren)", corpusInt(t, p, "roadmap_debt")),
		fmt.Sprintf("over %d discrete epic(s)", corpusInt(t, p, "discrete_epics")),
		"(" + maturedRung.String() + ")",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("markdown snapshot missing %q:\n%s", want, body)
		}
	}

	// Both folded KPIs are tabulated with their own detail line.
	for _, k := range p.KPIs {
		if !strings.Contains(body, "| "+k.Key+" |") {
			t.Fatalf("markdown snapshot missing a table row for KPI %q:\n%s", k.Key, body)
		}
		if !strings.Contains(body, k.Detail) {
			t.Fatalf("markdown snapshot missing KPI %q detail %q:\n%s", k.Key, k.Detail, body)
		}
	}

	// The HARD work list is the retire list: one bullet per owed rung-step / open child.
	_, after, found := strings.Cut(body, "## Work list (HARD)")
	if !found {
		t.Fatalf("markdown snapshot has debt %d but no HARD work list:\n%s", debt, body)
	}
	bullets := 0
	for _, line := range strings.Split(after, "\n") {
		if strings.HasPrefix(line, "- ") {
			bullets++
		}
	}
	if bullets != debt {
		t.Fatalf("work list has %d bullet(s), want %d (one per unit of milestone_debt):\n%s", bullets, debt, body)
	}
}

// TestScorecardMarkdownSnapshotCleanHasNoWorkList proves the snapshot's work list is
// data-driven, not boilerplate: a fully-matured grid with complete epics renders the same
// front matter and table with zero debt and NO work list at all.
func TestScorecardMarkdownSnapshotCleanHasNoWorkList(t *testing.T) {
	m := InterpretMaturity([]covmatrix.Cell{{Family: "a", Backend: "cpu", Support: covmatrix.Supported}})
	e := InterpretEpics(
		[]EpicSpec{{Number: 1315, Title: "native agent harness"}},
		[]EpicCounts{{Number: 1315, Closed: 4, Total: 4, Source: "label"}},
		"",
	)
	p := BuildScorecard(m, e)
	body := scorecard.Markdown(p, ScorecardMarkdownDoc(p))
	if !strings.Contains(body, DebtKey+" **0**") {
		t.Fatalf("a clean fold must render %s **0**:\n%s", DebtKey, body)
	}
	if strings.Contains(body, "## Work list (HARD)") {
		t.Fatalf("a clean fold must render no HARD work list:\n%s", body)
	}
	if !strings.Contains(body, "| climb_distance_to_matured |") {
		t.Fatalf("a clean fold must still tabulate its KPIs:\n%s", body)
	}
}

// --- test helpers -----------------------------------------------------------

// corpusInt reads an integer corpus key. The kernel writes the debt counts as int
// (the headline) and BuildScorecard writes its own counts as int, so a direct
// type assertion suffices; a float fallback covers any value that arrived via JSON.
func corpusInt(t *testing.T, p scorecard.Payload, key string) int {
	t.Helper()
	v, ok := p.Corpus[key]
	if !ok {
		t.Fatalf("corpus missing key %q (have %v)", key, keysOf(p.Corpus))
	}
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	default:
		t.Fatalf("corpus[%q] = %v (%T), want an integer", key, v, v)
		return 0
	}
}

// worklistOf reads the milestone_worklist corpus key back as the typed slice.
func worklistOf(t *testing.T, p scorecard.Payload) []WorklistItem {
	t.Helper()
	v, ok := p.Corpus["milestone_worklist"]
	if !ok {
		t.Fatalf("corpus missing milestone_worklist")
	}
	wl, ok := v.([]WorklistItem)
	if !ok {
		t.Fatalf("milestone_worklist is %T, want []WorklistItem", v)
	}
	return wl
}

func keysOf(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func jsonOf(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}
