package main

import (
	"strings"
	"testing"
)

// TestSkillEffectivenessBodyBudget pins the body-tier HARD word budget: a skill
// whose faulted body stays under skillBodyWordBudget scores zero, and one that
// blows past it scores a body_budget debt unit that flips the verdict to ACTION.
func TestSkillEffectivenessBodyBudget(t *testing.T) {
	// Under-budget: a clean "Use when" skill so ONLY body_budget could fire
	// (zero affordance + zero loader debt per TestLoaderDebtZeroForCleanCatalog).
	under := chdirRepo(t)
	body := "---\nname: lean\ndescription: Use when you need a lean skill\n---\n\n# lean\n\n" + strings.Repeat("word ", 100)
	writeSkill(t, under, "lean", body)
	p := collectSkillEffectivenessScorecard(under)
	corpus := p["corpus"].(map[string]any)
	if corpus["body_budget"] != 0 {
		t.Fatalf("under-budget body_budget=%v, want 0", corpus["body_budget"])
	}

	// Over-budget: 5001 body words > 5000-word ceiling.
	over := chdirRepo(t)
	big := "---\nname: fat\ndescription: Use when you need a fat skill\n---\n\n# fat\n\n" + strings.Repeat("word ", 5001)
	writeSkill(t, over, "fat", big)
	q := collectSkillEffectivenessScorecard(over)
	qc := q["corpus"].(map[string]any)
	if qc["body_budget"] != 1 {
		t.Fatalf("over-budget body_budget=%v, want 1", qc["body_budget"])
	}
	if q["verdict"] != "ACTION" {
		t.Fatalf("over-budget verdict=%v, want ACTION", q["verdict"])
	}
}

// TestSkillEffectivenessMetadataBudget pins the metadata-tier HARD word budget:
// a frontmatter description over skillMetadataWordBudget words scores a
// metadata_budget debt unit; a short description scores zero.
func TestSkillEffectivenessMetadataBudget(t *testing.T) {
	over := chdirRepo(t)
	desc := strings.TrimSpace(strings.Repeat("w ", 101)) // 101 words on the description: line
	body := "---\nname: verbose\ndescription: " + desc + "\n---\n\n# verbose\n\nUse when you need a verbose skill\n"
	writeSkill(t, over, "verbose", body)
	p := collectSkillEffectivenessScorecard(over)
	corpus := p["corpus"].(map[string]any)
	if corpus["metadata_budget"] != 1 {
		t.Fatalf("over-budget metadata_budget=%v, want 1", corpus["metadata_budget"])
	}

	under := chdirRepo(t)
	writeSkill(t, under, "terse", "---\nname: terse\ndescription: Use when you need a terse skill\n---\n\n# terse\n\nUse when you need a terse skill\n")
	q := collectSkillEffectivenessScorecard(under)
	qc := q["corpus"].(map[string]any)
	if qc["metadata_budget"] != 0 {
		t.Fatalf("short-description metadata_budget=%v, want 0", qc["metadata_budget"])
	}
}
