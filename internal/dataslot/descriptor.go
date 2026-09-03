package dataslot

import (
	"errors"
	"fmt"
	"strings"
)

// DatabaseFamily identifies the underlying database engine/family.
type DatabaseFamily string

const (
	FamilySQLite   DatabaseFamily = "sqlite"
	FamilyDuckDB   DatabaseFamily = "duckdb"
	FamilyPostgres DatabaseFamily = "postgres"
	FamilyMySQL    DatabaseFamily = "mysql"
	FamilyRedis    DatabaseFamily = "redis"
)

// ValidFamilies enumerates recognized database families.
var ValidFamilies = map[DatabaseFamily]bool{
	FamilySQLite:   true,
	FamilyDuckDB:   true,
	FamilyPostgres: true,
	FamilyMySQL:    true,
	FamilyRedis:    true,
}

// MigrationEngine identifies the migration framework in use.
type MigrationEngine string

const (
	MigrationNone    MigrationEngine = "none"
	MigrationPrisma  MigrationEngine = "prisma"
	MigrationAlembic MigrationEngine = "alembic"
	MigrationGoose   MigrationEngine = "goose"
	MigrationFlyway  MigrationEngine = "flyway"
	MigrationDrizzle MigrationEngine = "drizzle"
)

// SlotStatus represents the progressive lifecycle state of a data capability slot.
type SlotStatus string

const (
	StatusAbsent         SlotStatus = "absent"
	StatusUnmaterialized SlotStatus = "dormant:unmaterialized"
	StatusReady          SlotStatus = "dormant:ready"
	StatusActive         SlotStatus = "active"
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

// Validate checks that required descriptor fields are present and valid.
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
