package dataslot

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkDataSlot measures descriptor validation and lifecycle predicate evaluation.
func BenchmarkDataSlot(b *testing.B) {
	descriptors := []DataSlotDescriptor{
		{
			ID:             "sqlite:app.db",
			Family:         FamilySQLite,
			Status:         StatusReady,
			SourceArtifact: "app.db",
			LocalPath:      "app.db",
			ReadOnly:       true,
		},
		{
			ID:              "prisma:schema.prisma",
			Family:          FamilyPostgres,
			Status:          StatusUnmaterialized,
			SourceArtifact:  "schema.prisma",
			MigrationEngine: MigrationPrisma,
			MigrationPath:   "migrations",
			ReadOnly:        true,
		},
		{
			ID:             "dbt:dbt_project.yml",
			Family:         FamilyDBT,
			Status:         StatusReady,
			SourceArtifact: "dbt_project.yml",
			ReadOnly:       true,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		desc := descriptors[i%len(descriptors)]
		if err := desc.Validate(); err != nil {
			b.Fatalf("validation failed: %v", err)
		}
		_ = desc.IsDormant()
		_ = desc.IsReady()
		_ = desc.IsActive()
	}
}

// BenchmarkDataSlotDetect measures dormant database discovery across directory structures.
func BenchmarkDataSlotDetect(b *testing.B) {
	dir := b.TempDir()
	dbPath := filepath.Join(dir, "production.sqlite")
	content := append([]byte(sqliteMagic), make([]byte, 256)...)
	if err := os.WriteFile(dbPath, content, 0644); err != nil {
		b.Fatal(err)
	}

	prismaDir := filepath.Join(dir, "prisma")
	if err := os.MkdirAll(prismaDir, 0755); err != nil {
		b.Fatal(err)
	}
	prismaPath := filepath.Join(prismaDir, "schema.prisma")
	prismaContent := []byte(`datasource db { provider = "sqlite" url = "file:./dev.db" }`)
	if err := os.WriteFile(prismaPath, prismaContent, 0644); err != nil {
		b.Fatal(err)
	}

	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slots, err := DetectWorkspace(ctx, dir)
		if err != nil {
			b.Fatalf("detection failed: %v", err)
		}
		if len(slots) < 2 {
			b.Fatalf("expected at least 2 slots, got %d", len(slots))
		}
	}
}

// BenchmarkValidateReadOnlySQL measures parser validation for read-only SQL queries.
func BenchmarkValidateReadOnlySQL(b *testing.B) {
	queries := []string{
		"SELECT id, name, email FROM users WHERE active = 1 ORDER BY id DESC LIMIT 50;",
		"EXPLAIN QUERY PLAN SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id;",
		"-- fetch table schema\nPRAGMA table_info(customers);",
		"/* audit check */ PRAGMA foreign_key_list(orders);",
		"WITH active_users AS (SELECT id FROM users WHERE active = 1) SELECT count(*) FROM active_users;",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		if err := ValidateReadOnlySQL(q); err != nil {
			b.Fatalf("read-only SQL check failed for query %q: %v", q, err)
		}
	}
}

// BenchmarkReadDBTSemantics measures parsing and lineage graph construction for dbt projects.
func BenchmarkReadDBTSemantics(b *testing.B) {
	dir := b.TempDir()
	manifestPath := filepath.Join(dir, "manifest.json")

	manifest := map[string]any{
		"nodes": map[string]any{
			"model.jaffle_shop.customers": map[string]any{
				"name":          "customers",
				"package_name":  "jaffle_shop",
				"description":   "Customers dimensional model",
				"resource_type": "model",
				"depends_on": map[string]any{
					"nodes": []string{"model.jaffle_shop.stg_customers", "model.jaffle_shop.stg_orders"},
				},
				"columns": map[string]any{
					"customer_id": map[string]any{"name": "customer_id"},
					"first_name":  map[string]any{"name": "first_name"},
				},
				"tags": []string{"core", "nightly"},
			},
			"model.jaffle_shop.stg_customers": map[string]any{
				"name":          "stg_customers",
				"package_name":  "jaffle_shop",
				"description":   "Staging customers",
				"resource_type": "model",
				"depends_on": map[string]any{
					"nodes": []string{},
				},
				"columns": map[string]any{
					"customer_id": map[string]any{"name": "customer_id"},
				},
				"tags": []string{"staging"},
			},
			"model.jaffle_shop.stg_orders": map[string]any{
				"name":          "stg_orders",
				"package_name":  "jaffle_shop",
				"description":   "Staging orders",
				"resource_type": "model",
				"depends_on": map[string]any{
					"nodes": []string{},
				},
				"columns": map[string]any{
					"order_id":    map[string]any{"name": "order_id"},
					"customer_id": map[string]any{"name": "customer_id"},
				},
				"tags": []string{"staging"},
			},
		},
	}

	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, manifestBytes, 0644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		receipt, err := ReadDBTSemantics(manifestPath)
		if err != nil {
			b.Fatalf("read dbt semantics failed: %v", err)
		}
		if receipt.ModelCount != 3 {
			b.Fatalf("expected 3 models, got %d", receipt.ModelCount)
		}
		_, _, ok := receipt.Lineage("customers")
		if !ok {
			b.Fatal("customers lineage missing")
		}
	}
}

// BenchmarkDatabaseSchemaValidate measures schema definition validation and table lookup.
func BenchmarkDatabaseSchemaValidate(b *testing.B) {
	schema := &DatabaseSchema{
		Family:   FamilySQLite,
		Database: "analytics.db",
		Tables: []TableSchema{
			{
				Name: "events",
				Columns: []ColumnSchema{
					{Name: "id", Type: "INTEGER", PrimaryKey: true},
					{Name: "session_id", Type: "TEXT", Nullable: false},
					{Name: "payload", Type: "TEXT", Nullable: true},
					{Name: "created_at", Type: "DATETIME", Nullable: false},
				},
				PrimaryKeys: []string{"id"},
			},
			{
				Name: "sessions",
				Columns: []ColumnSchema{
					{Name: "id", Type: "TEXT", PrimaryKey: true},
					{Name: "user_id", Type: "TEXT", Nullable: false},
					{Name: "started_at", Type: "DATETIME", Nullable: false},
				},
				PrimaryKeys: []string{"id"},
			},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := schema.Validate(); err != nil {
			b.Fatalf("schema validate failed: %v", err)
		}
		if _, ok := schema.Table("events"); !ok {
			b.Fatal("events table not found")
		}
	}
}

// BenchmarkSnapshotAndRollback measures database file snapshotting and rollback restoration.
func BenchmarkSnapshotAndRollback(b *testing.B) {
	dir := b.TempDir()
	dbPath := filepath.Join(dir, "bench.db")
	initialData := []byte("database snapshot benchmark test payload bytes")
	if err := os.WriteFile(dbPath, initialData, 0644); err != nil {
		b.Fatal(err)
	}

	mgr := NewSnapshotManager(filepath.Join(dir, "snaps"))
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap, err := mgr.Snapshot(ctx, dbPath, "bench-scope", i)
		if err != nil {
			b.Fatalf("snapshot failed: %v", err)
		}

		// Mutate database file
		if err := os.WriteFile(dbPath, []byte("mutated database state"), 0644); err != nil {
			b.Fatal(err)
		}

		// Rollback and verify restoration
		if err := mgr.Rollback(ctx, snap); err != nil {
			b.Fatalf("rollback failed: %v", err)
		}
	}
}
