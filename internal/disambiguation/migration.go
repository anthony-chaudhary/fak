package disambiguation

import (
	"errors"
	"fmt"
	"strings"
)

const MigrationSchemaVersion = "fak-disambiguation-migration/1"

var ErrSilentRemoval = errors.New("silent identity removal")
var ErrMigrationRecordInvalid = errors.New("migration record invalid")

type MigrationRecord struct {
	RemovedTerm     string `json:"removed_term"`
	ReplacementTerm string `json:"replacement_term"`
	DeprecatedIn    string `json:"deprecated_in"`
	RemovedIn       string `json:"removed_in"`
	Breaking        bool   `json:"breaking"`
}

type MigrationReport struct {
	Schema   string            `json:"schema"`
	Accepted []MigrationRecord `json:"accepted"`
}

func ValidateMigrations(changes DiffReport, records []MigrationRecord) (MigrationReport, error) {
	byRemoved := map[string]MigrationRecord{}
	for _, record := range records {
		if strings.TrimSpace(record.RemovedTerm) == "" || strings.TrimSpace(record.ReplacementTerm) == "" || strings.TrimSpace(record.DeprecatedIn) == "" || strings.TrimSpace(record.RemovedIn) == "" || record.DeprecatedIn == record.RemovedIn {
			return MigrationReport{}, fmt.Errorf("%w: %#v", ErrMigrationRecordInvalid, record)
		}
		byRemoved[record.RemovedTerm] = record
	}
	report := MigrationReport{Schema: MigrationSchemaVersion, Accepted: []MigrationRecord{}}
	for _, change := range changes.Changes {
		if change.Kind != ChangeRemoval && !(change.Kind == ChangeAliasMove && change.QueryImpact == ImpactBreaking) {
			continue
		}
		record, ok := byRemoved[change.CanonicalTerm]
		if !ok {
			return report, fmt.Errorf("%w: %s", ErrSilentRemoval, change.CanonicalTerm)
		}
		report.Accepted = append(report.Accepted, record)
	}
	return report, nil
}

type MigrationSelfTestReport struct {
	Schema                 string `json:"schema"`
	SilentRemovalRejected  bool   `json:"silent_removal_rejected"`
	VersionedAliasAccepted bool   `json:"versioned_alias_accepted"`
	ReplacementTarget      string `json:"replacement_target"`
}

func RunMigrationSelfTest() (MigrationSelfTestReport, error) {
	report := MigrationSelfTestReport{Schema: MigrationSchemaVersion}
	removal := DiffReport{Changes: []IndexChange{{Kind: ChangeRemoval, CanonicalTerm: "retired alias", QueryImpact: ImpactBreaking}}}
	_, err := ValidateMigrations(removal, nil)
	report.SilentRemovalRejected = errors.Is(err, ErrSilentRemoval)
	record := MigrationRecord{RemovedTerm: "retired alias", ReplacementTerm: "canonical replacement", DeprecatedIn: "public-seed/1", RemovedIn: "public-seed/2"}
	accepted, err := ValidateMigrations(removal, []MigrationRecord{record})
	if err != nil {
		return report, err
	}
	report.VersionedAliasAccepted = len(accepted.Accepted) == 1
	report.ReplacementTarget = accepted.Accepted[0].ReplacementTerm
	if !report.SilentRemovalRejected || !report.VersionedAliasAccepted {
		return report, errors.New("migration self-test failed")
	}
	return report, nil
}
