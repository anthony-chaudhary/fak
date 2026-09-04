package dataslot

import (
	"errors"
	"fmt"
	"strings"
)

// DatabaseFamily identifies the underlying database engine/family.
type DatabaseFamily string

const (
	// FamilySQLite identifies embedded SQLite database files.
	FamilySQLite DatabaseFamily = "sqlite"
	// FamilyDuckDB identifies columnar DuckDB embedded database files.
	FamilyDuckDB DatabaseFamily = "duckdb"
	// FamilyPostgres identifies PostgreSQL network service targets.
	FamilyPostgres DatabaseFamily = "postgres"
	// FamilyMySQL identifies MySQL and MariaDB service targets.
	FamilyMySQL DatabaseFamily = "mysql"
	// FamilyRedis identifies Redis in-memory cache service targets.
	FamilyRedis DatabaseFamily = "redis"
	// FamilyDBT identifies dbt data modeling project roots.
	FamilyDBT DatabaseFamily = "dbt"
)

// ValidFamilies enumerates recognized database families.
var ValidFamilies = map[DatabaseFamily]bool{
	FamilySQLite:   true,
	FamilyDuckDB:   true,
	FamilyPostgres: true,
	FamilyMySQL:    true,
	FamilyRedis:    true,
	FamilyDBT:      true,
}

// MigrationEngine identifies the migration framework in use.
type MigrationEngine string

const (
	// MigrationNone indicates no detected migration framework.
	MigrationNone MigrationEngine = "none"
	// MigrationPrisma identifies Prisma ORM schema migrations.
	MigrationPrisma MigrationEngine = "prisma"
	// MigrationAlembic identifies Python Alembic database migrations.
	MigrationAlembic MigrationEngine = "alembic"
	// MigrationGoose identifies Goose database migration scripts.
	MigrationGoose MigrationEngine = "goose"
	// MigrationFlyway identifies Java Flyway versioned SQL migrations.
	MigrationFlyway MigrationEngine = "flyway"
	// MigrationDrizzle identifies Drizzle TypeScript ORM schema migrations.
	MigrationDrizzle MigrationEngine = "drizzle"
)

// SlotStatus represents the progressive lifecycle state of a data capability slot.
type SlotStatus string

const (
	// StatusAbsent indicates a declared data capability is absent from the workspace.
	StatusAbsent SlotStatus = "absent"
	// StatusUnmaterialized indicates migration artifacts exist without a live database file.
	StatusUnmaterialized SlotStatus = "dormant:unmaterialized"
	// StatusReady indicates the database file or service is ready for read-only binding.
	StatusReady SlotStatus = "dormant:ready"
	// StatusActive indicates the database capability is currently bound and active.
	StatusActive SlotStatus = "active"
)

// ValidStatuses enumerates valid progressive lifecycle states.
var ValidStatuses = map[SlotStatus]bool{
	StatusAbsent:         true,
	StatusUnmaterialized: true,
	StatusReady:          true,
	StatusActive:         true,
}

// DataSlotDescriptor describes a detected dormant database capability in a repository workspace.
type DataSlotDescriptor struct {
	ID              string          `json:"id"`
	Family          DatabaseFamily  `json:"family"`
	Status          SlotStatus      `json:"status"`
	SourceArtifact  string          `json:"source_artifact"`
	LocalPath       string          `json:"local_path,omitempty"`
	MigrationEngine MigrationEngine `json:"migration_engine,omitempty"`
	MigrationPath   string          `json:"migration_path,omitempty"`
	ServiceImage    string          `json:"service_image,omitempty"`
	Port            int             `json:"port,omitempty"`
	ReadOnly        bool            `json:"read_only"`
	Metadata        map[string]any  `json:"metadata,omitempty"`
}

// IsDormant returns true if the slot is in a dormant state (unmaterialized or ready).
func (d DataSlotDescriptor) IsDormant() bool {
	return d.Status == StatusUnmaterialized || d.Status == StatusReady
}

// IsReady returns true if the database capability is immediately ready without migration.
func (d DataSlotDescriptor) IsReady() bool {
	return d.Status == StatusReady
}

// IsActive returns true if the database capability is actively bound.
func (d DataSlotDescriptor) IsActive() bool {
	return d.Status == StatusActive
}

// Validate checks that required descriptor fields are present and valid,
// rejecting missing identifiers, unmapped database families, or invalid
// lifecycle statuses before runtime admission.
func (d DataSlotDescriptor) Validate() error {
	if strings.TrimSpace(d.ID) == "" {
		return errors.New("dataslot: descriptor ID is required")
	}
	if !ValidFamilies[d.Family] {
		return fmt.Errorf("dataslot: unknown database family %q", d.Family)
	}
	if !ValidStatuses[d.Status] {
		return fmt.Errorf("dataslot: unknown status %q", d.Status)
	}
	return nil
}
