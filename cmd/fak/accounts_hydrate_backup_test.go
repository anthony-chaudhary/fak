package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// Growth-boundary regression for #3505: every hydrate used to write a fresh
// "<file>.before-hydrate-<stamp>.bak" into <seat>/backups/ with no cap, so the dir grew
// strictly monotonically for the seat's lifetime and retained every superseded plaintext
// credential/OAuth token indefinitely. These tests pin the bound at the write site.

// writeHydrateSeed writes a credential file with known content into dir.
func writeHydrateSeed(t *testing.T, dir, file, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, file), []byte(body), 0o600); err != nil {
		t.Fatalf("seed %s: %v", file, err)
	}
}

// hydrateBackupNames lists the surviving before-hydrate backups of file, oldest first.
func hydrateBackupNames(t *testing.T, backupDir, file string) []string {
	t.Helper()
	ents, err := os.ReadDir(backupDir)
	if err != nil {
		t.Fatalf("read backups dir: %v", err)
	}
	prefix := hydrateBackupPrefix(file)
	var out []string
	for _, e := range ents {
		if n := e.Name(); strings.HasPrefix(n, prefix) && strings.HasSuffix(n, ".bak") {
			out = append(out, n)
		}
	}
	return out
}

// TestBackupIfExistsCapsBeforeHydrateChain is the core bound: many more hydrates than the cap
// leave exactly hydrateBackupKeep backups, and the survivors are the NEWEST ones — a cap that
// kept the oldest would defeat the point of a pre-image.
func TestBackupIfExistsCapsBeforeHydrateChain(t *testing.T) {
	dir := t.TempDir()
	backupDir := filepath.Join(dir, "backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("mkdir backups: %v", err)
	}
	writeHydrateSeed(t, dir, ".credentials.json", `{"token":"live"}`)

	stamps := []string{
		"20260101T000001Z", "20260101T000002Z", "20260101T000003Z", "20260101T000004Z",
		"20260101T000005Z", "20260101T000006Z", "20260101T000007Z", "20260101T000008Z",
	}
	for _, s := range stamps {
		if err := backupIfExists(dir, backupDir, ".credentials.json", s); err != nil {
			t.Fatalf("backupIfExists(%s): %v", s, err)
		}
	}

	got := hydrateBackupNames(t, backupDir, ".credentials.json")
	if len(got) != hydrateBackupKeep {
		t.Fatalf("after %d hydrates want %d retained backups, got %d: %v",
			len(stamps), hydrateBackupKeep, len(got), got)
	}
	// The survivors must be the last hydrateBackupKeep stamps, newest included.
	for i, name := range got {
		want := hydrateBackupPrefix(".credentials.json") + stamps[len(stamps)-hydrateBackupKeep+i] + ".bak"
		if name != want {
			t.Errorf("survivor %d = %q, want %q (oldest must be evicted, newest kept)", i, name, want)
		}
	}
}

// TestBackupIfExistsCapsPerFile pins that the cap is per credential file: a long
// .credentials.json chain must never evict the .oauth-token pre-image, which is a different
// secret with its own recovery value.
func TestBackupIfExistsCapsPerFile(t *testing.T) {
	dir := t.TempDir()
	backupDir := filepath.Join(dir, "backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("mkdir backups: %v", err)
	}
	writeHydrateSeed(t, dir, ".credentials.json", `{"token":"live"}`)
	writeHydrateSeed(t, dir, ".oauth-token", "oauth-live")

	for _, s := range []string{"20260101T000001Z", "20260101T000002Z"} {
		if err := backupIfExists(dir, backupDir, ".oauth-token", s); err != nil {
			t.Fatalf("backupIfExists(.oauth-token, %s): %v", s, err)
		}
	}
	for i := 0; i < hydrateBackupKeep*3; i++ {
		s := "20260202T0000" + string(rune('0'+i/10)) + string(rune('0'+i%10)) + "Z"
		if err := backupIfExists(dir, backupDir, ".credentials.json", s); err != nil {
			t.Fatalf("backupIfExists(.credentials.json, %s): %v", s, err)
		}
	}

	if got := len(hydrateBackupNames(t, backupDir, ".credentials.json")); got != hydrateBackupKeep {
		t.Errorf(".credentials.json chain = %d backups, want capped at %d", got, hydrateBackupKeep)
	}
	if got := len(hydrateBackupNames(t, backupDir, ".oauth-token")); got != 2 {
		t.Errorf(".oauth-token chain = %d backups, want its own 2 untouched by the cred chain", got)
	}
}

// TestBackupIfExistsKeepsContentAndSkipsAbsent pins the behaviour the cap must not regress:
// a present credential is copied byte-for-byte, and an absent one is a silent no-op.
func TestBackupIfExistsKeepsContentAndSkipsAbsent(t *testing.T) {
	dir := t.TempDir()
	backupDir := filepath.Join(dir, "backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("mkdir backups: %v", err)
	}
	writeHydrateSeed(t, dir, ".credentials.json", `{"token":"pre-image"}`)

	if err := backupIfExists(dir, backupDir, ".credentials.json", "20260101T000001Z"); err != nil {
		t.Fatalf("backupIfExists: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(backupDir, hydrateBackupPrefix(".credentials.json")+"20260101T000001Z.bak"))
	if err != nil {
		t.Fatalf("read backup: %v", err)
	}
	if string(b) != `{"token":"pre-image"}` {
		t.Errorf("backup content = %q, want the pre-image bytes", string(b))
	}

	// Absent file: no error, no backup written.
	if err := backupIfExists(dir, backupDir, ".oauth-token", "20260101T000002Z"); err != nil {
		t.Fatalf("absent credential should be a no-op, got %v", err)
	}
	if got := hydrateBackupNames(t, backupDir, ".oauth-token"); len(got) != 0 {
		t.Errorf("absent credential wrote backups: %v", got)
	}
}

// TestPruneHydrateBackupsRefusesToEmpty pins the safety floor: keep <= 0 must never be read as
// "delete everything", and a backups dir that does not exist yet is a valid state, not an error.
func TestPruneHydrateBackupsRefusesToEmpty(t *testing.T) {
	dir := t.TempDir()
	backupDir := filepath.Join(dir, "backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("mkdir backups: %v", err)
	}
	writeHydrateSeed(t, dir, ".credentials.json", `{"token":"live"}`)
	if err := backupIfExists(dir, backupDir, ".credentials.json", "20260101T000001Z"); err != nil {
		t.Fatalf("backupIfExists: %v", err)
	}

	for _, keep := range []int{0, -1} {
		if err := pruneHydrateBackups(backupDir, ".credentials.json", keep); err != nil {
			t.Fatalf("prune keep=%d: %v", keep, err)
		}
		if got := len(hydrateBackupNames(t, backupDir, ".credentials.json")); got != 1 {
			t.Errorf("keep=%d pruned to %d backups, want the pre-image retained", keep, got)
		}
	}

	if err := pruneHydrateBackups(filepath.Join(dir, "no-such-backups"), ".credentials.json", hydrateBackupKeep); err != nil {
		t.Errorf("missing backups dir should be a no-op, got %v", err)
	}
}

// TestApplyAccountHydrateBoundsSeatBackupDir is the END-TO-END bound at the verb #3505 names.
// The tests above pin backupIfExists in isolation, but the dir an operator actually watches is
// grown by applyAccountHydrate, which writes TWO chains in one run: a .credentials.json hydrate
// also retires the target's stale .oauth-token. So a helper-level cap alone does not witness
// that the production path is bounded — this does.
//
// It also pins the MIGRATION case, which is the one a long-lived seat is actually in: the seat
// is seeded with a pile that accumulated BEFORE the cap existed, and one real hydrate must reap
// it back to the cap. A fix that only bounded newly-created seats would leave every existing
// seat's historical plaintext credentials on disk forever and still pass the helper tests.
func TestApplyAccountHydrateBoundsSeatBackupDir(t *testing.T) {
	root := t.TempDir()
	sourceDir := filepath.Join(root, "source")
	targetDir := filepath.Join(root, "target")
	backupDir := filepath.Join(targetDir, "backups")
	for _, d := range []string{sourceDir, targetDir, backupDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", d, err)
		}
	}

	// The source holds the live credential; the target holds the stale pair a hydrate replaces,
	// so a single run backs up both credential shapes.
	writeHydrateSeed(t, sourceDir, ".credentials.json", `{"token":"fresh"}`)
	writeHydrateSeed(t, targetDir, ".credentials.json", `{"token":"stale"}`)
	writeHydrateSeed(t, targetDir, ".oauth-token", "stale-oauth")

	// Pre-cap accumulation. The stamps are older than any real hydrate's, so the run's own
	// pre-image is the newest and must be the one that survives.
	const historical = 20
	for _, file := range []string{".credentials.json", ".oauth-token"} {
		for i := 0; i < historical; i++ {
			name := fmt.Sprintf("%s20250101T0000%02dZ.bak", hydrateBackupPrefix(file), i)
			if err := os.WriteFile(filepath.Join(backupDir, name), []byte("superseded"), 0o600); err != nil {
				t.Fatalf("seed historical backup %s: %v", name, err)
			}
		}
	}

	reg := accounts.Registry{Homes: []accounts.Home{
		{Name: "target", Dir: targetDir, Identity: accounts.Identity{AccountUUID: "acct-1"}},
		{Name: "source", Dir: sourceDir, Identity: accounts.Identity{AccountUUID: "acct-1"}},
	}}
	if _, err := applyAccountHydrate(reg, "target", "source"); err != nil {
		t.Fatalf("applyAccountHydrate: %v", err)
	}

	for _, file := range []string{".credentials.json", ".oauth-token"} {
		got := hydrateBackupNames(t, backupDir, file)
		if len(got) != hydrateBackupKeep {
			t.Errorf("%s chain holds %d backups after one hydrate, want the cap %d (seeded %d pre-cap)",
				file, len(got), hydrateBackupKeep, historical)
		}
	}

	// The bound must not have cost the hydrate its actual job.
	b, err := os.ReadFile(filepath.Join(targetDir, ".credentials.json"))
	if err != nil {
		t.Fatalf("read hydrated credential: %v", err)
	}
	if string(b) != `{"token":"fresh"}` {
		t.Errorf("hydrated credential = %q, want the source's", string(b))
	}
}
