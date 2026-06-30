package scorecardpane

import (
	"strings"
	"testing"
)

func TestKPIBloatHardAndSoft(t *testing.T) {
	k := KPIBloat([]BloatDoc{
		{Path: "docs/huge.md", NLines: 1500}, // > hard
		{Path: "docs/long.md", NLines: 700},  // > soft, <= hard
		{Path: "docs/ok.md", NLines: 100},
	})
	if len(k.Defects) != 1 {
		t.Fatalf("one oversized doc want 1 defect, got %d", len(k.Defects))
	}
	if len(k.Soft) != 1 {
		t.Fatalf("one long doc want 1 soft, got %d", len(k.Soft))
	}
	if !strings.Contains(k.Defects[0], "docs/huge.md") {
		t.Fatalf("defect must name the oversized doc: %q", k.Defects[0])
	}
}

func TestKPIRootHygieneFlagsStrayDocAndClutter(t *testing.T) {
	k := KPIRootHygiene([]string{"README.md", "RANDOM-NOTE.md"}, []string{"go.mod", "scratch.bin"})
	// RANDOM-NOTE.md is not on the (default) allowlist; scratch.bin is not allowed.
	if len(k.Defects) != 2 {
		t.Fatalf("stray doc + clutter want 2 defects, got %d: %v", len(k.Defects), k.Defects)
	}
}

func TestKPIIndexPresenceMissing(t *testing.T) {
	k := KPIIndexPresence(map[string]bool{"INDEX.md": true, "llms.txt": false, "docs/index.md": true})
	if len(k.Defects) != 1 || !strings.Contains(k.Defects[0], "llms.txt") {
		t.Fatalf("missing llms.txt want 1 defect naming it, got %v", k.Defects)
	}
}

func TestKPIIndexIntegrityDeadEntry(t *testing.T) {
	k := KPIIndexIntegrity(map[string][]string{"INDEX.md": {"docs/gone.md", "docs/also-gone.md"}})
	if len(k.Defects) != 2 {
		t.Fatalf("two dead entries want 2 defects, got %d", len(k.Defects))
	}
	// sorted: also-gone before gone
	if !strings.Contains(k.Defects[0], "also-gone") {
		t.Fatalf("dead entries must be sorted: %v", k.Defects)
	}
}

func TestKPIOrphansPercent(t *testing.T) {
	k := KPIOrphans([]string{"docs/lost.md"}, 4)
	if len(k.Defects) != 1 {
		t.Fatalf("one orphan want 1 defect, got %d", len(k.Defects))
	}
	// 3/4 indexed = 75%
	if k.Score != 75 {
		t.Fatalf("3/4 reachable want score 75, got %d", k.Score)
	}
}

func TestKPIDirDisciplineNearDuplicate(t *testing.T) {
	k := KPIDirDiscipline([]string{"docs/benchmark", "docs/benchmarks", "docs/guides"})
	if len(k.Defects) != 1 {
		t.Fatalf("benchmark vs benchmarks want 1 defect, got %d: %v", len(k.Defects), k.Defects)
	}
}

func TestKPIAITellsCapAndDefects(t *testing.T) {
	k := KPIAITells([]AITellDoc{
		{Path: "docs/a.md", Hits: []string{"delve", "seamless"}, EmdashOver: 3},
	})
	if len(k.Defects) != 2 {
		t.Fatalf("two tell hits want 2 defects, got %d", len(k.Defects))
	}
	if len(k.Soft) != 1 || !strings.Contains(k.Soft[0], "em-dash") {
		t.Fatalf("em-dash overage want 1 soft, got %v", k.Soft)
	}
}

func TestKPIAITellsPerDocCap(t *testing.T) {
	var hits []string
	for i := 0; i < 12; i++ {
		hits = append(hits, "delve")
	}
	k := KPIAITells([]AITellDoc{{Path: "docs/a.md", Hits: hits}})
	if len(k.Defects) != aitellPerDocCap {
		t.Fatalf("hits must cap at %d, got %d", aitellPerDocCap, len(k.Defects))
	}
	if len(k.Soft) != 1 || !strings.Contains(k.Soft[0], "more AI-tells") {
		t.Fatalf("overflow must surface as soft, got %v", k.Soft)
	}
}

func TestBuildHygienePayloadCleanTree(t *testing.T) {
	var clean []HygieneKPI
	for kpi, grp := range kpiGroupFor() {
		clean = append(clean, kpiResult(kpi, grp, 100, "ok", nil, nil))
	}
	p := BuildHygienePayload("/repo", clean, nil)
	if !p.OK || p.Verdict != "OK" || p.Finding != "repo_clean" {
		t.Fatalf("zero-defect tree must be clean, got ok=%v verdict=%s finding=%s", p.OK, p.Verdict, p.Finding)
	}
	if p.Corpus.HygieneDebt != 0 || p.Corpus.Grade != "A" {
		t.Fatalf("clean tree want debt 0 grade A, got debt=%d grade=%s", p.Corpus.HygieneDebt, p.Corpus.Grade)
	}
}

func TestBuildHygienePayloadDebtRanksWorstFirst(t *testing.T) {
	kpis := []HygieneKPI{
		kpiResult("redundancy", "verbosity", 80, "d", []string{"x"}, nil),
		kpiResult("orphans", "indexing", 50, "d", []string{"a", "b", "c"}, nil),
		kpiResult("bloat", "verbosity", 90, "d", nil, nil),
	}
	p := BuildHygienePayload("/repo", kpis, nil)
	if p.Corpus.HygieneDebt != 4 {
		t.Fatalf("hygiene_debt want 4, got %d", p.Corpus.HygieneDebt)
	}
	if p.Corpus.Breakdown[0].KPI != "orphans" {
		t.Fatalf("worst-first must put orphans (3 defects) first, got %q", p.Corpus.Breakdown[0].KPI)
	}
	if !p.OK == false && p.Finding != "hygiene_debt" {
		t.Fatalf("debt tree must find hygiene_debt, got %s", p.Finding)
	}
}

func TestHygieneA11yDebtIsAccessibilitySlice(t *testing.T) {
	kpis := []HygieneKPI{
		kpiResult("alt_text", "accessibility", 80, "d", []string{"img1"}, nil),
		kpiResult("ai_tells", "accessibility", 80, "d", []string{"delve", "seamless"}, nil),
		kpiResult("bloat", "verbosity", 90, "d", []string{"big"}, nil),
	}
	p := BuildHygienePayload("/repo", kpis, nil)
	if p.Corpus.A11yDebt != 3 {
		t.Fatalf("a11y_debt is the accessibility-group HARD slice, want 3, got %d", p.Corpus.A11yDebt)
	}
	if p.Corpus.HygieneDebt != 4 {
		t.Fatalf("total hygiene_debt want 4, got %d", p.Corpus.HygieneDebt)
	}
}

// kpiGroupFor returns the canonical kpi -> group map for synthesizing a clean tree.
func kpiGroupFor() map[string]string {
	return map[string]string{
		"redundancy": "verbosity", "bloat": "verbosity",
		"root_hygiene": "organization", "placement": "organization", "dir_discipline": "organization",
		"index_presence": "indexing", "index_integrity": "indexing", "orphans": "indexing",
		"alt_text": "accessibility", "ai_tells": "accessibility",
		"jargon": "accessibility", "plain_language": "accessibility",
	}
}
