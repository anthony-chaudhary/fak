package agentquery

import (
	"reflect"
	"sort"
	"testing"
	"time"
)

func TestSchemaDescriptorMatchesRowJSONKeys(t *testing.T) {
	want := DescriptorFieldNames()
	sort.Strings(want)
	got, err := MarshalRowKeys()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("json=%v descriptor=%v", got, want)
	}
	d := SchemaDescriptor()
	if d.Schema != DescriptorSchema || d.MaxRows != 10000 || len(d.Filters) < 10 || len(d.Sorts) != 12 {
		t.Fatalf("descriptor=%+v", d)
	}
}
func TestValidateResult(t *testing.T) {
	now := time.Now().UTC().Format(time.RFC3339)
	r := Result{Metadata: Metadata{Schema: Schema, Source: "history", ObservedAt: now, Limit: 10}, Rows: []Row{{AgentID: "a", LogicalSessionID: "a", Source: "history", ObservedAt: now}}}
	if err := ValidateResult(r); err != nil {
		t.Fatal(err)
	}
	r.Rows[0].AgentID = ""
	if err := ValidateResult(r); err == nil {
		t.Fatal("accepted missing identity")
	}
}

func TestValidateGroupResult(t *testing.T) {
	observed := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	lane := "cmd"
	max := int64(25)
	plan, err := GroupedPlan(time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	r := GroupResult{Metadata: GroupMetadata{Schema: GroupSchema, GroupBy: plan.GroupBy, Source: "history", Since: observed.Add(-time.Hour).Format(time.RFC3339), ObservedAt: observed.Format(time.RFC3339), InputRows: 2, MatchedRows: 2, Plan: plan}, Rows: []GroupRow{{Lane: &lane, State: "closed", Count: 2, MaxElapsedMS: &max}}}
	if err := ValidateGroupResult(r); err != nil {
		t.Fatal(err)
	}
	r.Metadata.MatchedRows = 1
	if err := ValidateGroupResult(r); err == nil {
		t.Fatal("accepted mismatched group counts")
	}
}
