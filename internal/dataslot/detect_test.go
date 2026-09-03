package dataslot

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectEmptyRepo(t *testing.T) {
	dir := t.TempDir()
	descs, err := Detect(dir)
	if err != nil {
		t.Fatalf("unexpected error on empty repo: %v", err)
	}
	if len(descs) != 0 {
		t.Fatalf("expected 0 descriptors for empty repo, got %d", len(descs))
	}
}

func TestDetectSQLiteFile(t *testing.T) {
	dir := t.TempDir()

	// Write valid SQLite magic header file
	dbPath := filepath.Join(dir, "app.sqlite")
	content := append([]byte(sqliteMagic), make([]byte, 100)...)
	if err := os.WriteFile(dbPath, content, 0644); err != nil {
		t.Fatal(err)
	}

	descs, err := Detect(dir)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(descs) != 1 {
		t.Fatalf("expected 1 descriptor, got %d", len(descs))
	}

	d := descs[0]
	if d.Family != FamilySQLite {
		t.Errorf("expected family SQLite, got %s", d.Family)
	}
	if d.Status != StatusReady {
		t.Errorf("expected status %s, got %s", StatusReady, d.Status)
	}
	if d.SourceArtifact != "app.sqlite" {
		t.Errorf("expected source artifact app.sqlite, got %s", d.SourceArtifact)
	}
	if !d.ReadOnly {
		t.Errorf("expected ReadOnly=true")
	}
}

func TestDetectDuckDBFile(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "analytics.duckdb")
	if err := os.WriteFile(dbPath, []byte("DUCKdummydata"), 0644); err != nil {
		t.Fatal(err)
	}

	descs, err := Detect(dir)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(descs) != 1 {
		t.Fatalf("expected 1 descriptor, got %d", len(descs))
	}
	if descs[0].Family != FamilyDuckDB {
		t.Errorf("expected family duckdb, got %s", descs[0].Family)
	}
}

func TestDetectPrismaSchema(t *testing.T) {
	dir := t.TempDir()
	prismaDir := filepath.Join(dir, "prisma")
	if err := os.MkdirAll(prismaDir, 0755); err != nil {
		t.Fatal(err)
	}

	schemaContent := `
datasource db {
  provider = "sqlite"
  url      = "file:./dev.db"
}

generator client {
  provider = "prisma-client-js"
}
`
	if err := os.WriteFile(filepath.Join(prismaDir, "schema.prisma"), []byte(schemaContent), 0644); err != nil {
		t.Fatal(err)
	}

	// 1. Without dev.db -> dormant:unmaterialized
	descs, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) != 1 {
		t.Fatalf("expected 1 descriptor, got %d", len(descs))
	}
	if descs[0].Status != StatusUnmaterialized {
		t.Errorf("expected unmaterialized status, got %s", descs[0].Status)
	}
	if descs[0].MigrationEngine != MigrationPrisma {
		t.Errorf("expected prisma engine, got %s", descs[0].MigrationEngine)
	}

	// 2. Now touch dev.db -> dormant:ready
	devDB := filepath.Join(prismaDir, "dev.db")
	if err := os.WriteFile(devDB, append([]byte(sqliteMagic), 0), 0644); err != nil {
		t.Fatal(err)
	}

	descs2, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}
	// dev.db is detected as both prisma materialized and raw db file
	foundMaterializedPrisma := false
	for _, d := range descs2 {
		if d.MigrationEngine == MigrationPrisma && d.Status == StatusReady {
			foundMaterializedPrisma = true
		}
	}
	if !foundMaterializedPrisma {
		t.Errorf("expected materialized prisma descriptor with StatusReady")
	}
}

func TestDetectAlembicAndGoose(t *testing.T) {
	dir := t.TempDir()

	// Alembic ini
	if err := os.WriteFile(filepath.Join(dir, "alembic.ini"), []byte("[alembic]\nscript_location = alembic\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Goose migrations
	migDir := filepath.Join(dir, "migrations")
	if err := os.MkdirAll(migDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(migDir, "001_init.sql"), []byte("-- +goose Up\nCREATE TABLE users (id int);\n"), 0644); err != nil {
		t.Fatal(err)
	}

	descs, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) < 2 {
		t.Fatalf("expected at least 2 descriptors, got %d", len(descs))
	}

	engines := make(map[MigrationEngine]bool)
	for _, d := range descs {
		engines[d.MigrationEngine] = true
	}
	if !engines[MigrationAlembic] {
		t.Errorf("expected Alembic migration engine detected")
	}
	if !engines[MigrationGoose] {
		t.Errorf("expected Goose migration engine detected")
	}
}

func TestDetectDockerCompose(t *testing.T) {
	dir := t.TempDir()

	composeContent := `
version: '3.8'
services:
  postgres-db:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
  cache:
    image: redis:7.0-alpine
    ports:
      - "6379:6379"
`
	if err := os.WriteFile(filepath.Join(dir, "docker-compose.yml"), []byte(composeContent), 0644); err != nil {
		t.Fatal(err)
	}

	descs, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) != 2 {
		t.Fatalf("expected 2 compose services, got %d", len(descs))
	}

	families := make(map[DatabaseFamily]int)
	for _, d := range descs {
		families[d.Family]++
		if d.Status != StatusReady {
			t.Errorf("expected compose service status %s, got %s", StatusReady, d.Status)
		}
	}
	if families[FamilyPostgres] != 1 {
		t.Errorf("expected 1 postgres service, got %d", families[FamilyPostgres])
	}
	if families[FamilyRedis] != 1 {
		t.Errorf("expected 1 redis service, got %d", families[FamilyRedis])
	}
}

func TestDetectIgnoresNodeModulesAndGit(t *testing.T) {
	dir := t.TempDir()

	// Put sqlite inside node_modules and .git
	nmDir := filepath.Join(dir, "node_modules", "some-pkg")
	gitDir := filepath.Join(dir, ".git", "objects")
	_ = os.MkdirAll(nmDir, 0755)
	_ = os.MkdirAll(gitDir, 0755)

	_ = os.WriteFile(filepath.Join(nmDir, "leak.db"), []byte(sqliteMagic), 0644)
	_ = os.WriteFile(filepath.Join(gitDir, "leak.db"), []byte(sqliteMagic), 0644)

	descs, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) != 0 {
		t.Fatalf("expected ignored dirs to produce 0 descriptors, got %d", len(descs))
	}
}

func TestDetectDeterministicJSON(t *testing.T) {
	dir := t.TempDir()

	_ = os.WriteFile(filepath.Join(dir, "b.sqlite"), []byte(sqliteMagic), 0644)
	_ = os.WriteFile(filepath.Join(dir, "a.sqlite"), []byte(sqliteMagic), 0644)

	descs1, err1 := Detect(dir)
	descs2, err2 := Detect(dir)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}

	bytes1, _ := json.Marshal(descs1)
	bytes2, _ := json.Marshal(descs2)

	if string(bytes1) != string(bytes2) {
		t.Fatalf("JSON output not deterministic:\n%s\nvs\n%s", string(bytes1), string(bytes2))
	}
}

func TestDetectWithContextCancellation(t *testing.T) {
	dir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := DetectWorkspace(ctx, dir)
	if err != context.Canceled {
		t.Errorf("expected context.Canceled error, got %v", err)
	}
}

func TestDetectDockerComposeNamespacedImages(t *testing.T) {
	dir := t.TempDir()

	composeContent := `
version: '3.8'
services:
  custom-pg:
    image: bitnami/postgresql:16
    ports:
      - "5432:5432"
  vk:
    image: docker.io/valkey/valkey:7.2
    ports:
      - "6379:6379"
  maria:
    image: library/mariadb:10.11
    ports:
      - "3306:3306"
`
	if err := os.WriteFile(filepath.Join(dir, "compose.yaml"), []byte(composeContent), 0644); err != nil {
		t.Fatal(err)
	}

	descs, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) != 3 {
		t.Fatalf("expected 3 compose services, got %d", len(descs))
	}

	families := make(map[DatabaseFamily]bool)
	for _, d := range descs {
		families[d.Family] = true
	}
	if !families[FamilyPostgres] {
		t.Errorf("expected postgres family from bitnami/postgresql")
	}
	if !families[FamilyRedis] {
		t.Errorf("expected redis family from valkey")
	}
	if !families[FamilyMySQL] {
		t.Errorf("expected mysql family from library/mariadb")
	}
}

func TestDetectDrizzleConfig(t *testing.T) {
	dir := t.TempDir()

	if err := os.WriteFile(filepath.Join(dir, "drizzle.config.ts"), []byte("export default {};"), 0644); err != nil {
		t.Fatal(err)
	}

	descs, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) != 1 {
		t.Fatalf("expected 1 descriptor, got %d", len(descs))
	}
	if descs[0].MigrationEngine != MigrationDrizzle {
		t.Errorf("expected drizzle migration engine, got %s", descs[0].MigrationEngine)
	}
}

func TestDetectNonDirectoryRoot(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "plain.txt")
	if err := os.WriteFile(filePath, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := Detect(filePath)
	if err == nil || !strings.Contains(err.Error(), "is not a directory") {
		t.Fatalf("expected not a directory error, got: %v", err)
	}
}
