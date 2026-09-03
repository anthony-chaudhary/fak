package dataslot

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func createMigrationDB(t *testing.T, dbPath string, script string) {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		t.Skip("python not found, skipping migration verification test")
	}

	cmd := exec.Command(python, "-c", script, dbPath)
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed setting up migration test DB: %v", err)
	}
}

func TestVerifyMigrationStatus(t *testing.T) {
	dir := t.TempDir()

	// 1. Goose migration fixture
	gooseDB := filepath.Join(dir, "goose.db")
	gooseScript := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE goose_db_version (id INTEGER PRIMARY KEY, version_id INTEGER, is_applied INTEGER, tstamp TIMESTAMP);")
conn.execute("INSERT INTO goose_db_version (version_id, is_applied) VALUES (1001, 1);")
conn.execute("INSERT INTO goose_db_version (version_id, is_applied) VALUES (1002, 1);")
conn.commit()
conn.close()
`
	createMigrationDB(t, gooseDB, gooseScript)

	receiptGoose, err := VerifyMigration(context.Background(), gooseDB, MigrationGoose, "1002")
	if err != nil {
		t.Fatalf("VerifyMigration goose failed: %v", err)
	}
	if receiptGoose.Verdict != VerdictOK {
		t.Fatalf("expected goose migration 1002 to be witnessed, got %s (reason=%s)", receiptGoose.Verdict, receiptGoose.Reason)
	}
	if !receiptGoose.IntegrityClean {
		t.Errorf("expected clean integrity check")
	}

	// Missing version in Goose
	receiptGooseMissing, err := VerifyMigration(context.Background(), gooseDB, MigrationGoose, "9999")
	if err != nil {
		t.Fatal(err)
	}
	if receiptGooseMissing.Verdict != VerdictMissing {
		t.Fatalf("expected missing goose migration to be unwitnessed, got %s", receiptGooseMissing.Verdict)
	}

	// 2. Alembic migration fixture
	alembicDB := filepath.Join(dir, "alembic.db")
	alembicScript := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);")
conn.execute("INSERT INTO alembic_version VALUES ('abc123rev');")
conn.commit()
conn.close()
`
	createMigrationDB(t, alembicDB, alembicScript)

	receiptAlembic, err := VerifyMigration(context.Background(), alembicDB, MigrationAlembic, "abc123rev")
	if err != nil {
		t.Fatal(err)
	}
	if receiptAlembic.Verdict != VerdictOK {
		t.Fatalf("expected alembic migration to be witnessed, got %s (reason=%s)", receiptAlembic.Verdict, receiptAlembic.Reason)
	}

	// 3. Prisma migration fixture
	prismaDB := filepath.Join(dir, "prisma.db")
	prismaScript := `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE _prisma_migrations (id VARCHAR(36) PRIMARY KEY, migration_name VARCHAR(255) NOT NULL, finished_at TIMESTAMP, rolled_back_at TIMESTAMP);")
conn.execute("INSERT INTO _prisma_migrations (id, migration_name, finished_at, rolled_back_at) VALUES ('1', '20260903_add_users', '2026-09-03 12:00:00', NULL);")
conn.commit()
conn.close()
`
	createMigrationDB(t, prismaDB, prismaScript)

	receiptPrisma, err := VerifyMigration(context.Background(), prismaDB, MigrationPrisma, "20260903_add_users")
	if err != nil {
		t.Fatal(err)
	}
	if receiptPrisma.Verdict != VerdictOK {
		t.Fatalf("expected prisma migration to be witnessed, got %s (reason=%s)", receiptPrisma.Verdict, receiptPrisma.Reason)
	}

	// 4. Corrupt / non-sqlite file fails integrity check
	corruptDB := filepath.Join(dir, "corrupt.db")
	if err := os.WriteFile(corruptDB, []byte("NOT_A_SQLITE_DATABASE_GARBAGE_BYTES"), 0644); err != nil {
		t.Fatal(err)
	}

	receiptCorrupt, err := VerifyMigration(context.Background(), corruptDB, MigrationGoose, "1001")
	if err != nil {
		t.Fatal(err)
	}
	if receiptCorrupt.Verdict != VerdictMissing {
		t.Fatalf("expected corrupt db to yield unwitnessed verdict, got %s", receiptCorrupt.Verdict)
	}
	if receiptCorrupt.IntegrityClean {
		t.Errorf("expected IntegrityClean=false for corrupt file")
	}

	// 5. Unsupported engine
	receiptBadEngine, err := VerifyMigration(context.Background(), gooseDB, MigrationNone, "")
	if err != nil {
		t.Fatal(err)
	}
	if receiptBadEngine.Verdict != VerdictMissing {
		t.Fatalf("expected unsupported engine to yield unwitnessed verdict, got %s", receiptBadEngine.Verdict)
	}
}

func TestVerifySchemaIntegrityClean(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "clean.db")
	createMigrationDB(t, dbPath, `import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE test (id int);")
conn.commit()
conn.close()
`)

	clean, err := VerifySchemaIntegrity(context.Background(), dbPath)
	if err != nil {
		t.Fatalf("VerifySchemaIntegrity failed: %v", err)
	}
	if !clean {
		t.Errorf("expected clean schema integrity to be true")
	}
}
