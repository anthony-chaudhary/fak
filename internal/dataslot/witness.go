package dataslot

import (
	"context"
	"fmt"
	"strings"
)

const (
	VerdictOK      = "CLAIM_WITNESSED"
	VerdictMissing = "CLAIM_UNWITNESSED"
)

// MigrationReceipt records the verifiable ground-truth evidence of a database migration.
type MigrationReceipt struct {
	Verdict         string          `json:"verdict"`
	Engine          MigrationEngine `json:"engine"`
	ExpectedVersion string          `json:"expected_version"`
	FoundVersion    string          `json:"found_version,omitempty"`
	IntegrityClean  bool            `json:"integrity_clean"`
	AppliedVersions []string        `json:"applied_versions,omitempty"`
	Reason          string          `json:"reason,omitempty"`
	DatabasePath    string          `json:"database_path"`
}

// VerifySchemaIntegrity runs database integrity checks and asserts that the file is non-corrupt.
func VerifySchemaIntegrity(ctx context.Context, dbPath string) (bool, error) {
	driver, err := NewSQLiteDriver()
	if err != nil {
		return false, fmt.Errorf("dataslot: failed to initialize SQLite driver: %w", err)
	}

	res, err := driver.ExecuteQuery(ctx, dbPath, "PRAGMA integrity_check;", &QueryOptions{MaxRows: 10})
	if err != nil {
		return false, fmt.Errorf("dataslot: integrity check query failed: %w", err)
	}

	if res.RowCount == 0 || len(res.Rows) == 0 || len(res.Rows[0]) == 0 {
		return false, nil
	}

	status, ok := res.Rows[0][0].(string)
	if !ok {
		return false, nil
	}

	return strings.EqualFold(strings.TrimSpace(status), "ok"), nil
}

// VerifyMigration queries the framework-specific migration table to assert that expectedVersion is recorded as applied.
func VerifyMigration(ctx context.Context, dbPath string, engine MigrationEngine, expectedVersion string) (*MigrationReceipt, error) {
	receipt := &MigrationReceipt{
		Verdict:         VerdictMissing,
		Engine:          engine,
		ExpectedVersion: expectedVersion,
		IntegrityClean:  false,
		DatabasePath:    dbPath,
	}

	// 1. Verify schema integrity first
	clean, err := VerifySchemaIntegrity(ctx, dbPath)
	if err != nil || !clean {
		receipt.Reason = "database file failed integrity check or is corrupt"
		return receipt, nil
	}
	receipt.IntegrityClean = true

	driver, err := NewSQLiteDriver()
	if err != nil {
		return nil, fmt.Errorf("dataslot: SQLite driver: %w", err)
	}

	// 2. Query engine-specific migration ledger
	var query string
	var colName string

	switch engine {
	case MigrationPrisma:
		query = "SELECT migration_name FROM _prisma_migrations WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL ORDER BY finished_at DESC;"
		colName = "migration_name"
	case MigrationAlembic:
		query = "SELECT version_num FROM alembic_version;"
		colName = "version_num"
	case MigrationGoose:
		query = "SELECT version_id FROM goose_db_version WHERE is_applied = 1 ORDER BY version_id DESC;"
		colName = "version_id"
	case MigrationFlyway:
		query = "SELECT version FROM flyway_schema_history WHERE success = 1 ORDER BY installed_rank DESC;"
		colName = "version"
	case MigrationDrizzle:
		query = "SELECT hash FROM __drizzle_migrations ORDER BY created_at DESC;"
		colName = "hash"
	default:
		receipt.Reason = fmt.Sprintf("unsupported migration engine: %s", engine)
		return receipt, nil
	}

	res, err := driver.ExecuteQuery(ctx, dbPath, query, &QueryOptions{MaxRows: 500})
	if err != nil {
		receipt.Reason = fmt.Sprintf("migration table query failed: %v", err)
		return receipt, nil
	}

	var applied []string
	for i := 0; i < res.RowCount; i++ {
		row := res.RowMap(i)
		val := row[colName]
		if val == nil && len(res.Rows[i]) > 0 {
			val = res.Rows[i][0]
		}
		if s, ok := val.(string); ok && s != "" {
			applied = append(applied, s)
		} else if num, ok := val.(float64); ok {
			applied = append(applied, fmt.Sprintf("%.0f", num))
		}
	}
	receipt.AppliedVersions = applied

	if len(applied) == 0 {
		receipt.Reason = "migration ledger table is empty; no applied migrations recorded"
		return receipt, nil
	}

	// 3. Match expectedVersion
	receipt.FoundVersion = applied[0]
	if expectedVersion == "" {
		// If empty, any applied migration is sufficient
		receipt.Verdict = VerdictOK
		return receipt, nil
	}

	for _, v := range applied {
		if strings.EqualFold(v, expectedVersion) || strings.HasPrefix(strings.ToLower(v), strings.ToLower(expectedVersion)) {
			receipt.Verdict = VerdictOK
			receipt.FoundVersion = v
			return receipt, nil
		}
	}

	receipt.Reason = fmt.Sprintf("expected migration %q was not found in applied versions: %v", expectedVersion, applied)
	return receipt, nil
}
