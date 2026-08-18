package disambiguation

import (
	"errors"
	"testing"
)

func TestRuntimeCanonicalRequiresScope(t *testing.T) {
	if _, err := Query("runtime"); !errors.Is(err, ErrScopeRequired) {
		t.Fatalf("error=%v", err)
	}
	search := Search("runtime")
	if search.Verdict != SearchVerdictAmbiguous || len(search.Groups.Exact) != 5 {
		t.Fatalf("search=%#v", search)
	}
}

func TestRunRuntimeSourceSelfTestResolvesFivePublicSurfaces(t *testing.T) {
	report, err := RunRuntimeSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != RuntimeSourceSelfTestSchemaVersion || !report.UnscopedAmbiguous || len(report.Choices) != 5 {
		t.Fatalf("report=%#v", report)
	}
	owners := map[string]bool{}
	for _, choice := range report.Choices {
		if choice.CanonicalTerm != "runtime" || choice.Scope.Kind != "runtime" || choice.Alias == "" || choice.SourcePath == "" {
			t.Errorf("choice=%#v", choice)
		}
		owners[choice.OwnerLeaf] = true
	}
	if len(owners) != 5 {
		t.Fatalf("owners=%v", owners)
	}
}
