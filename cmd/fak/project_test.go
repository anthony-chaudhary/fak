package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/projectreport"
)

func TestProjectCardMeasuredCarriesVerdictAndLines(t *testing.T) {
	r := projectreport.Fold([]projectreport.Item{
		{Issue: 4030, Status: "Todo", Generation: "now", Priority: "P1"},
		{Issue: 9999}, // unclassified
	}, projectreport.FoldOpts{})
	card := projectCard(r, "agent")
	if card.Title != "project board" || card.Verdict != "ACTION" {
		t.Fatalf("card title/verdict = %q/%q, want 'project board'/'ACTION'", card.Title, card.Verdict)
	}
	if card.Debt != "1" || card.DebtKey != "unclassified" {
		t.Fatalf("card debt = %q/%q, want unclassified/1", card.DebtKey, card.Debt)
	}
	text := card.Text()
	for _, want := range []string{"project board", "status", "horizon"} {
		if !strings.Contains(text, want) {
			t.Fatalf("card text missing %q in:\n%s", want, text)
		}
	}
}

func TestProjectCardUnmeasuredIsSparse(t *testing.T) {
	card := projectCard(projectreport.Unmeasured("", projectreport.FoldOpts{}), "agent")
	if card.Verdict != "UNMEASURED" {
		t.Fatalf("verdict = %q, want UNMEASURED", card.Verdict)
	}
	if len(card.Lines) != 0 {
		t.Fatalf("unmeasured card should carry no stat lines, got %v", card.Lines)
	}
}

// TestParseProjectReportItems proves the board GraphQL payload folds into report items
// with Status / Generation / Priority mapped from their single-select field names, so
// the live path stays covered without a gh call.
func TestParseProjectReportItems(t *testing.T) {
	raw := []byte(`{"data":{"repositoryOwner":{"projectV2":{"items":{"nodes":[
		{"content":{"number":4030},"fieldValues":{"nodes":[
			{"name":"In progress","field":{"name":"Status"}},
			{"name":"now","field":{"name":"Generation"}},
			{"name":"P1","field":{"name":"Priority"}}]}},
		{"content":{"number":9999},"fieldValues":{"nodes":[]}},
		{"content":{},"fieldValues":{"nodes":[]}}
	]}}}}}`)
	items, ok := parseProjectReportItems(raw)
	if !ok {
		t.Fatalf("parse returned ok=false on valid payload")
	}
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2 (the contentless node is skipped): %+v", len(items), items)
	}
	first := items[0]
	if first.Issue != 4030 || first.Status != "In progress" || first.Generation != "now" || first.Priority != "P1" {
		t.Fatalf("first item mismapped: %+v", first)
	}
	if items[1].Issue != 9999 || items[1].Status != "" || items[1].Generation != "" {
		t.Fatalf("second (unclassified) item mismapped: %+v", items[1])
	}
}

func TestParseProjectReportItemsRejectsGarbage(t *testing.T) {
	if _, ok := parseProjectReportItems([]byte("not json")); ok {
		t.Fatalf("garbage payload should return ok=false")
	}
}
