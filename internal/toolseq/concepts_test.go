package toolseq

import "testing"

func TestDiscoverBuildsExplainableWorkflowConcepts(t *testing.T) {
	sessions := []Session{
		{ID: "s-search-a", Calls: []Call{{Tool: "search"}, {Tool: "read"}, {Tool: "read"}}},
		{ID: "s-search-b", Calls: []Call{{Tool: "search"}, {Tool: "read", Error: true}}},
		{ID: "s-edit-a", Calls: []Call{{Tool: "read"}, {Tool: "edit"}, {Tool: "test"}}},
		{ID: "s-edit-b", Calls: []Call{{Tool: "read"}, {Tool: "edit"}, {Tool: "test"}}},
	}
	got := Discover(sessions, 0.55)
	if len(got) != 2 {
		t.Fatalf("concepts=%d want 2: %#v", len(got), got)
	}
	for _, c := range got {
		if c.Sessions != 2 || c.Share != 0.5 {
			t.Errorf("bad prevalence: %#v", c)
		}
		if len(c.Signature) == 0 || len(c.Exemplars) == 0 || len(c.Members) != 2 {
			t.Errorf("concept is not drillable: %#v", c)
		}
		if c.ID[:3] != "wf-" {
			t.Errorf("unstable selector shape: %q", c.ID)
		}
	}
	var search Concept
	for _, c := range got {
		if c.Signature[0] == "search" {
			search = c
		}
	}
	if search.ErrorRate != 0.2 {
		t.Errorf("search error rate=%v want .2", search.ErrorRate)
	}
}

func TestDiscoverIsInputOrderStable(t *testing.T) {
	a := Session{ID: "a", Calls: []Call{{Tool: "search"}, {Tool: "read"}}}
	b := Session{ID: "b", Calls: []Call{{Tool: "search"}, {Tool: "read"}}}
	x := Discover([]Session{a, b}, .55)
	y := Discover([]Session{b, a}, .55)
	if x[0].ID != y[0].ID || x[0].Exemplars[0] != y[0].Exemplars[0] {
		t.Fatalf("not stable: %#v vs %#v", x, y)
	}
}
