package disambiguation

import (
	"reflect"
	"testing"
	"testing/fstest"
)

func TestIncidentalClassificationIsCoveredButNeverQueryable(t *testing.T) {
	classification := TermClassification{Term: "LocalRetryToken", Classification: ClassificationIncidental, Reason: ClassificationReasonLocalImplementation}
	index, err := NewClassifiedIndex(publicEntries, []TermClassification{classification})
	if err != nil {
		t.Fatal(err)
	}
	if got, found, ambiguous := index.queryCanonical("LocalRetryToken"); found || ambiguous {
		t.Fatalf("incidental query = %#v found=%t ambiguous=%t, want no canonical row", got, found, ambiguous)
	}
	fixture := fstest.MapFS{"api/terms.go": {Data: []byte("package api\n\ntype LocalRetryToken struct{}\n")}}
	report, err := InventoryCoverage(fixture, []PublicTerminologySurface{{Locator: "api", Kind: "go_package"}}, index, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !report.OK || report.Incidental != 1 || report.Canonical != 0 || len(report.Findings) != 0 {
		t.Fatalf("coverage = %#v", report)
	}
	if !reflect.DeepEqual(report.Classifications, []TermClassification{{SchemaVersion: ClassificationSchemaVersion, Term: "LocalRetryToken", Classification: ClassificationIncidental, Reason: ClassificationReasonLocalImplementation}}) {
		t.Fatalf("classifications = %#v", report.Classifications)
	}
}

func TestValidateClassificationsIsDeterministicAndStrict(t *testing.T) {
	input := []TermClassification{
		{Term: "ZetaToken", Classification: ClassificationIncidental, Reason: ClassificationReasonLocalImplementation},
		{Term: "AlphaToken", Classification: ClassificationIncidental, Reason: ClassificationReasonLocalImplementation},
	}
	one, err := ValidateClassifications(input)
	if err != nil {
		t.Fatal(err)
	}
	two, err := ValidateClassifications([]TermClassification{input[1], input[0]})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one, two) || one[0].Term != "AlphaToken" || one[0].SchemaVersion != ClassificationSchemaVersion {
		t.Fatalf("non-deterministic validation: %#v != %#v", one, two)
	}
	bad := []TermClassification{
		{Term: "localToken", Classification: ClassificationIncidental, Reason: ClassificationReasonLocalImplementation},
		{Term: "LocalToken", Classification: "canonical", Reason: ClassificationReasonLocalImplementation},
		{Term: "LocalToken", Classification: ClassificationIncidental, Reason: "free text"},
		{SchemaVersion: "future/2", Term: "LocalToken", Classification: ClassificationIncidental, Reason: ClassificationReasonLocalImplementation},
	}
	for _, item := range bad {
		if _, err := ValidateClassifications([]TermClassification{item}); err == nil {
			t.Fatalf("accepted invalid classification %#v", item)
		}
	}
	if _, err := ValidateClassifications([]TermClassification{input[0], input[0]}); err == nil {
		t.Fatal("accepted duplicate classification")
	}
}
