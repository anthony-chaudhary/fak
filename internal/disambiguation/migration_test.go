package disambiguation

import (
	"errors"
	"testing"
)

func TestValidateMigrationsRejectsSilentRemoval(t *testing.T) {
	changes := DiffReport{Changes: []IndexChange{{Kind: ChangeRemoval, CanonicalTerm: "old term", QueryImpact: ImpactBreaking}}}
	if _, err := ValidateMigrations(changes, nil); !errors.Is(err, ErrSilentRemoval) {
		t.Fatalf("err=%v", err)
	}
}

func TestValidateMigrationsAcceptsVersionedAliasReplacement(t *testing.T) {
	changes := DiffReport{Changes: []IndexChange{{Kind: ChangeAliasMove, CanonicalTerm: "old alias", QueryImpact: ImpactBreaking}}}
	record := MigrationRecord{RemovedTerm: "old alias", ReplacementTerm: "new canonical", DeprecatedIn: "public-seed/1", RemovedIn: "public-seed/2"}
	report, err := ValidateMigrations(changes, []MigrationRecord{record})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Accepted) != 1 || report.Accepted[0].ReplacementTerm != "new canonical" {
		t.Fatalf("report=%#v", report)
	}
}
