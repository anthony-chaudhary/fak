package dataslot

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// ColumnSchema describes a single column in a database table.
type ColumnSchema struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	Nullable   bool   `json:"nullable"`
	PrimaryKey bool   `json:"primary_key"`
}

// ForeignKeySchema describes a foreign key relation between tables.
type ForeignKeySchema struct {
	Column           string `json:"column"`
	ReferencedTable  string `json:"referenced_table"`
	ReferencedColumn string `json:"referenced_column"`
}

// TableSchema describes a single database table and its structure.
type TableSchema struct {
	Name        string             `json:"name"`
	Columns     []ColumnSchema     `json:"columns"`
	PrimaryKeys []string           `json:"primary_keys,omitempty"`
	ForeignKeys []ForeignKeySchema `json:"foreign_keys,omitempty"`
	RowCount    int64              `json:"row_count,omitempty"`
}

// DatabaseSchema carries the complete reflected schema summary for a database.
type DatabaseSchema struct {
	Family   DatabaseFamily `json:"family"`
	Database string         `json:"database"`
	Tables   []TableSchema  `json:"tables"`
}

// SchemaReflector reflects table and column structures from a database target.
type SchemaReflector interface {
	ReflectSchema(ctx context.Context, target string) (*DatabaseSchema, error)
}

// Table returns the TableSchema matching tableName if present.
func (s *DatabaseSchema) Table(tableName string) (*TableSchema, bool) {
	for i := range s.Tables {
		if strings.EqualFold(s.Tables[i].Name, tableName) {
			return &s.Tables[i], true
		}
	}
	return nil, false
}

// Validate checks that a reflected schema contains valid table and column definitions.
func (s *DatabaseSchema) Validate() error {
	if s.Database == "" {
		return errors.New("dataslot: database target is required")
	}
	for _, t := range s.Tables {
		if strings.TrimSpace(t.Name) == "" {
			return fmt.Errorf("dataslot: table in %q has empty name", s.Database)
		}
		if len(t.Columns) == 0 {
			return fmt.Errorf("dataslot: table %q has no columns", t.Name)
		}
	}
	return nil
}
