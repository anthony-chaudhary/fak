package orientation

import (
	"strings"
	"testing"
	"time"
)

func TestCurrentValidAndPerformanceHarnessTransitionsExplicit(t *testing.T) {
	s, err := Current()
	if err != nil {
		t.Fatal(err)
	}
	items := map[string]Item{}
	for _, item := range s.Items {
		items[item.ID] = item
	}
	for _, id := range []string{"performance-cache", "harness-bindings"} {
		item, ok := items[id]
		if !ok {
			t.Fatalf("missing %s", id)
		}
		if item.RetainedContract == "" || item.DecreaseWhen == "" || item.FutureState == "" {
			t.Fatalf("%s lacks transition contract: %#v", id, item)
		}
	}
}

func TestAssessFreshnessStates(t *testing.T) {
	s := Snapshot{ReviewBy: "2026-10-15"}
	cases := []struct{ now, want string }{{"2026-09-01", "current"}, {"2026-10-05", "due-soon"}, {"2026-10-16", "stale"}}
	for _, tc := range cases {
		now, _ := time.Parse(time.DateOnly, tc.now)
		if got := Assess(s, now).Freshness; got != tc.want {
			t.Errorf("%s: got %s want %s", tc.now, got, tc.want)
		}
	}
}

func TestValidateRejectsIncompleteTransitionsAndDuplicateIDs(t *testing.T) {
	s, err := Current()
	if err != nil {
		t.Fatal(err)
	}
	s.Items[0].DecreaseWhen = ""
	s.Items[1].ID = s.Items[0].ID
	err = Validate(s)
	if err == nil || !strings.Contains(err.Error(), "decrease_when is required") || !strings.Contains(err.Error(), "id is duplicate") {
		t.Fatalf("error = %v", err)
	}
}

func TestTextNamesInvariantAndTemporalState(t *testing.T) {
	s, err := Current()
	if err != nil {
		t.Fatal(err)
	}
	v := Assess(s, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	got := v.Text()
	for _, want := range []string{"FAK ORIENTATION — CURRENT", "Enduring promise:", "Performance, cache reuse", "decrease when:"} {
		if !strings.Contains(got, want) {
			t.Fatalf("output missing %q:\n%s", want, got)
		}
	}
}
