package accounts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeCred(t *testing.T, dir, file, body string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, file), []byte(body), 0o600); err != nil {
		t.Fatalf("write %s/%s: %v", dir, file, err)
	}
}

func readCred(t *testing.T, dir, file string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, file))
	if err != nil {
		t.Fatalf("read %s/%s: %v", dir, file, err)
	}
	return string(b)
}

// The load-bearing acceptance test: after an op OVERWRITES a seat's credential, the pre-image is
// recoverable from the backup taken before the overwrite.
func TestBackupThenOverwrite_PreImageRecoverable(t *testing.T) {
	home := t.TempDir()
	root := BackupRoot(home)
	dir := filepath.Join(home, ".claude-july4-netra")
	writeCred(t, dir, ".credentials.json", `{"claudeAiOauth":{"accessToken":"july4-token"}}`)
	writeCred(t, dir, ".claude.json", `{"oauthAccount":{"emailAddress":"july4@x"}}`)

	t0 := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	taken, err := SnapshotBeforeOverwrite(root, "july4-netra", dir, t0)
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if len(taken) != 2 {
		t.Fatalf("want 2 blobs snapshotted (.credentials.json + .claude.json), got %d: %+v", len(taken), taken)
	}

	// Simulate the wrong-dir /login: july4's dir is re-logged into july17.
	writeCred(t, dir, ".credentials.json", `{"claudeAiOauth":{"accessToken":"july17-token"}}`)

	// The prior july4 credential must still be recoverable.
	restored, err := RestoreCredential(root, "july4-netra", dir, ".credentials.json", "", t0.Add(time.Hour))
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if !strings.Contains(readCred(t, dir, ".credentials.json"), "july4-token") {
		t.Fatalf("restore did not bring back the july4 credential; got %q", readCred(t, dir, ".credentials.json"))
	}
	if restored.File != ".credentials.json" {
		t.Errorf("restored.File = %q, want .credentials.json", restored.File)
	}
}

// restore-credential round-trips a seat's credentials from a snapshot, and the restore is itself
// reversible (the overwritten blob is backed up before it is replaced).
func TestRestoreCredential_RoundTripAndReversible(t *testing.T) {
	home := t.TempDir()
	root := BackupRoot(home)
	dir := filepath.Join(home, ".claude-seat")
	writeCred(t, dir, ".credentials.json", `A`)

	t0 := time.Date(2026, 7, 10, 1, 0, 0, 0, time.UTC)
	if _, err := BackupSeatCredentials(root, "seat", dir, t0); err != nil {
		t.Fatalf("backup A: %v", err)
	}
	// Overwrite with B and back that up too.
	writeCred(t, dir, ".credentials.json", `B`)
	t1 := t0.Add(time.Minute)
	if _, err := BackupSeatCredentials(root, "seat", dir, t1); err != nil {
		t.Fatalf("backup B: %v", err)
	}
	// Now the live blob is C (never backed up); restore the newest snapshot (B).
	writeCred(t, dir, ".credentials.json", `C`)
	if _, err := RestoreCredential(root, "seat", dir, ".credentials.json", "", t1.Add(time.Minute)); err != nil {
		t.Fatalf("restore newest: %v", err)
	}
	if got := readCred(t, dir, ".credentials.json"); got != "B" {
		t.Fatalf("restore newest = %q, want B", got)
	}
	// The pre-restore live blob (C) must have been captured, so the restore was reversible.
	all, _ := ListCredentialBackups(root, "seat")
	var haveC bool
	for _, b := range all {
		body, _ := os.ReadFile(b.Path)
		if string(body) == "C" {
			haveC = true
		}
	}
	if !haveC {
		t.Fatalf("restore did not back up the live blob (C) it overwrote; store: %+v", all)
	}
}

// Selecting a specific snapshot by timestamp prefix restores THAT blob, not the newest.
func TestRestoreCredential_SelectByStamp(t *testing.T) {
	home := t.TempDir()
	root := BackupRoot(home)
	dir := filepath.Join(home, ".claude-seat")

	writeCred(t, dir, ".credentials.json", `oldest`)
	tA := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	BackupSeatCredentials(root, "seat", dir, tA)

	writeCred(t, dir, ".credentials.json", `newest`)
	tB := time.Date(2026, 7, 9, 0, 0, 0, 0, time.UTC)
	BackupSeatCredentials(root, "seat", dir, tB)

	// Ask for the July 1 snapshot explicitly.
	if _, err := RestoreCredential(root, "seat", dir, ".credentials.json", "20260701", tB.Add(time.Hour)); err != nil {
		t.Fatalf("restore by stamp: %v", err)
	}
	if got := readCred(t, dir, ".credentials.json"); got != "oldest" {
		t.Fatalf("restore by stamp = %q, want oldest", got)
	}
}

// Content addressing: the same bytes snapshotted twice produce ONE store entry (dedup), while
// changed bytes produce a new one.
func TestBackupDedupByContent(t *testing.T) {
	home := t.TempDir()
	root := BackupRoot(home)
	dir := filepath.Join(home, ".claude-seat")
	writeCred(t, dir, ".credentials.json", `same`)

	t0 := time.Date(2026, 7, 10, 0, 0, 0, 0, time.UTC)
	if got, _ := BackupSeatCredentials(root, "seat", dir, t0); len(got) != 1 {
		t.Fatalf("first backup: want 1, got %d", len(got))
	}
	// Same content, later stamp: no new entry.
	if got, _ := BackupSeatCredentials(root, "seat", dir, t0.Add(time.Hour)); len(got) != 0 {
		t.Fatalf("identical-content backup: want 0 (dedup), got %d", len(got))
	}
	writeCred(t, dir, ".credentials.json", `changed`)
	if got, _ := BackupSeatCredentials(root, "seat", dir, t0.Add(2*time.Hour)); len(got) != 1 {
		t.Fatalf("changed-content backup: want 1, got %d", len(got))
	}
	all, _ := ListCredentialBackups(root, "seat")
	if len(all) != 2 {
		t.Fatalf("store should hold 2 distinct blobs, got %d: %+v", len(all), all)
	}
}

// Prune keeps the newest N per file and never drops the most recent recoverable blob.
func TestPruneKeepsNewestPerFile(t *testing.T) {
	home := t.TempDir()
	root := BackupRoot(home)
	dir := filepath.Join(home, ".claude-seat")
	base := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < 5; i++ {
		writeCred(t, dir, ".credentials.json", strings.Repeat("x", i+1))
		BackupSeatCredentials(root, "seat", dir, base.Add(time.Duration(i)*time.Hour))
	}
	removed, err := PruneCredentialBackups(root, "seat", 2)
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if removed != 3 {
		t.Fatalf("prune removed %d, want 3", removed)
	}
	all, _ := ListCredentialBackups(root, "seat")
	if len(all) != 2 {
		t.Fatalf("after prune want 2, got %d", len(all))
	}
	// The survivors must be the two newest (5 x's and 4 x's).
	if all[0].Size != 5 || all[1].Size != 4 {
		t.Fatalf("prune kept the wrong blobs: %+v", all)
	}
}

// Empty / missing credential dirs are a valid no-op, never an error.
func TestBackupEmptyDirNoOp(t *testing.T) {
	home := t.TempDir()
	root := BackupRoot(home)
	dir := filepath.Join(home, ".claude-empty")
	os.MkdirAll(dir, 0o755)
	got, err := BackupSeatCredentials(root, "empty", dir, time.Unix(0, 0).UTC())
	if err != nil {
		t.Fatalf("empty backup errored: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("empty dir should snapshot nothing, got %d", len(got))
	}
	// An empty (0-byte) credential file is also skipped.
	writeCred(t, dir, ".credentials.json", "")
	if got, _ := BackupSeatCredentials(root, "empty", dir, time.Unix(0, 0).UTC()); len(got) != 0 {
		t.Fatalf("empty file should be skipped, got %d", len(got))
	}
}

// The store lives under the gitignored home tree (~/.claude/backups), never a repo path, and never
// as a ~/.claude-* sibling that discovery would scan as a seat.
func TestBackupRootUnderHomeClaude(t *testing.T) {
	home := filepath.FromSlash("/home/op")
	got := BackupRoot(home)
	want := filepath.Join(home, ".claude", "backups")
	if got != want {
		t.Fatalf("BackupRoot = %q, want %q", got, want)
	}
	if strings.Contains(got, ".claude-") {
		t.Fatalf("backup root must not be a ~/.claude-* sibling (scanned as a seat): %q", got)
	}
}

// A corrupted store file (content no longer matches its address) is refused, not silently restored.
func TestRestoreRefusesCorruptSnapshot(t *testing.T) {
	home := t.TempDir()
	root := BackupRoot(home)
	dir := filepath.Join(home, ".claude-seat")
	writeCred(t, dir, ".credentials.json", `good`)
	BackupSeatCredentials(root, "seat", dir, time.Date(2026, 7, 10, 0, 0, 0, 0, time.UTC))

	all, _ := ListCredentialBackups(root, "seat")
	if len(all) != 1 {
		t.Fatalf("setup: want 1 snapshot, got %d", len(all))
	}
	// Tamper with the stored blob without fixing its content-address name.
	if err := os.WriteFile(all[0].Path, []byte("tampered"), 0o600); err != nil {
		t.Fatalf("tamper: %v", err)
	}
	if _, err := RestoreCredential(root, "seat", dir, ".credentials.json", "", time.Now()); err == nil {
		t.Fatalf("restore of a corrupt snapshot should error")
	}
}
