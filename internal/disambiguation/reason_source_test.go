package disambiguation

import (
	"errors"
	"testing"
)

func TestValidateVocabularyRejectsIncompatibleDuplicate(t *testing.T) {
	terms := []VocabularyTerm{
		{Code: "DENY", Kind: VocabularyReason, Package: "a", Symbol: "ReasonDeny", CanonicalMeaning: "policy-refusal", SourcePath: "a/a.go"},
		{Code: "DENY", Kind: VocabularyVerdict, Package: "b", Symbol: "VerdictDeny", CanonicalMeaning: "decision-denied", SourcePath: "b/b.go"},
	}
	if err := ValidateVocabulary(terms); !errors.Is(err, ErrVocabularyCollision) {
		t.Fatalf("error=%v", err)
	}
}

func TestValidateVocabularyPermitsDeclaredCrossPackageAlias(t *testing.T) {
	terms := []VocabularyTerm{
		{Code: "COLLISION_RISK", Kind: VocabularyReason, Package: "a", Symbol: "One", CanonicalMeaning: "unsafe-overlap", SourcePath: "a/a.go"},
		{Code: "COLLISION_RISK", Kind: VocabularyReason, Package: "b", Symbol: "Two", CanonicalMeaning: "unsafe-overlap", SourcePath: "b/b.go"},
	}
	if err := ValidateVocabulary(terms); err != nil {
		t.Fatal(err)
	}
}

func TestRunReasonSourceSelfTest(t *testing.T) {
	report, err := RunReasonSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != ReasonSourceSelfTestSchemaVersion || len(report.Terms) != 6 || !report.IncompatibleRejected || !report.CrossPackageAliasAllowed {
		t.Fatalf("report=%#v", report)
	}
	for _, term := range report.Terms {
		if term.SourcePath == "" || term.Symbol == "" {
			t.Errorf("incomplete term=%#v", term)
		}
	}
}
