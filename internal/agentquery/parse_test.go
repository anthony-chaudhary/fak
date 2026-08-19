package agentquery

import (
	"reflect"
	"testing"
	"time"
)

func TestParseQueryMatchesGroupedPlan(t *testing.T) {
	q := `SELECT lane, state, count(*) AS agents, max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC`
	got, err := ParseQuery(q)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := GroupedPlan(7 * 24 * time.Hour)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
}
func TestParseQueryRejectsEscapeAndUnboundedShapes(t *testing.T) {
	bad := []string{"DELETE FROM agents", "SELECT * FROM agents", `SELECT lane, state, count(*) AS agents, max(elapsed_ms) AS max_elapsed_ms FROM agents GROUP BY lane,state ORDER BY max_elapsed_ms DESC`, `SELECT lane, state, count(*) AS agents, max(cost) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC`, `SELECT lane, state, count(*) AS agents, max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC; SELECT 1`}
	for _, q := range bad {
		if _, err := ParseQuery(q); err == nil {
			t.Errorf("accepted %q", q)
		}
	}
}
func TestParseQueryAcceptsDocumentedObservedAtAlias(t *testing.T) {
	q := `SELECT lane,state,count(*) AS agents,max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE observed_at >= now()-interval '7 days' GROUP BY lane,state ORDER BY max_elapsed_ms DESC`
	p, err := ParseQuery(q)
	if err != nil || p.TimeColumn != "observed_at" {
		t.Fatalf("plan=%+v err=%v", p, err)
	}
}
