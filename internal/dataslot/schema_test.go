package dataslot

import (
	"context"
	"os/exec"
	"path/filepath"
	"testing"
)

func createTestDBWithSchema(t *testing.T, dbPath string) {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		t.Skip("python not found, skipping sqlite integration test")
	}

	script := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, active INT);")
conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, FOREIGN KEY(user_id) REFERENCES users(id));")
conn.commit()
`
	cmd := exec.Command(python, "-c", script, dbPath)
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed to create test SQLite DB: %v", err)
	}
}

func TestReflectSchema(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "store.db")
	createTestDBWithSchema(t, dbPath)

	driver, err := NewSQLiteDriver()
	if err != nil {
		t.Skipf("no SQLite runner available: %v", err)
	}

	schema, err := driver.ReflectSchema(context.Background(), dbPath)
	if err != nil {
		t.Fatalf("ReflectSchema failed: %v", err)
	}

	if err := schema.Validate(); err != nil {
		t.Fatalf("schema validation failed: %v", err)
	}

	if len(schema.Tables) != 2 {
		t.Fatalf("expected 2 tables, got %d", len(schema.Tables))
	}

	users, ok := schema.Table("users")
	if !ok {
		t.Fatal("table 'users' not found in reflected schema")
	}
	if len(users.Columns) != 3 {
		t.Errorf("expected 3 columns for 'users', got %d", len(users.Columns))
	}
	if len(users.PrimaryKeys) != 1 || users.PrimaryKeys[0] != "id" {
		t.Errorf("expected primary key 'id', got %+v", users.PrimaryKeys)
	}

	orders, ok := schema.Table("orders")
	if !ok {
		t.Fatal("table 'orders' not found in reflected schema")
	}
	if len(orders.ForeignKeys) != 1 {
		t.Fatalf("expected 1 foreign key for 'orders', got %d", len(orders.ForeignKeys))
	}
	fk := orders.ForeignKeys[0]
	if fk.Column != "user_id" || fk.ReferencedTable != "users" || fk.ReferencedColumn != "id" {
		t.Errorf("unexpected foreign key: %+v", fk)
	}
}

func TestDatabaseSchema_TableNotFound(t *testing.T) {
	s := &DatabaseSchema{
		Database: "test.db",
		Tables: []TableSchema{
			{Name: "users", Columns: []ColumnSchema{{Name: "id", Type: "int"}}},
		},
	}
	if _, ok := s.Table("nonexistent"); ok {
		t.Errorf("expected false for nonexistent table")
	}
}

func TestDatabaseSchema_ValidateErrors(t *testing.T) {
	emptyDB := &DatabaseSchema{}
	if err := emptyDB.Validate(); err == nil {
		t.Errorf("expected error for empty database target")
	}

	emptyTable := &DatabaseSchema{
		Database: "app.db",
		Tables: []TableSchema{
			{Name: "  ", Columns: []ColumnSchema{{Name: "id"}}},
		},
	}
	if err := emptyTable.Validate(); err == nil {
		t.Errorf("expected error for table with empty name")
	}

	noCols := &DatabaseSchema{
		Database: "app.db",
		Tables: []TableSchema{
			{Name: "users", Columns: nil},
		},
	}
	if err := noCols.Validate(); err == nil {
		t.Errorf("expected error for table with no columns")
	}
}
