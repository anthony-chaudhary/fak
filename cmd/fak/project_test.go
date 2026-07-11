package main

import "testing"

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
