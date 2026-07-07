package logvault

// sync_test.go — the #2454 acceptance gate: a local-disk sync with post-sync
// verify on the receiving side, and a scrub witness proving a deliberately
// planted token-shaped string in a source file is ABSENT from the synced copy.
// The planted token is built at runtime by concatenation so no token-shaped
// literal is ever committed (the public-repo scrub laws apply to this test
// file too).

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// plantedToken is a GitHub-PAT-shaped string (ghp_ + 36 alphanumerics), the
// canonical "must never leave the box unredacted" shape the reference
// redactor's github_token pattern matches.
func plantedToken() string { return "ghp_" + strings.Repeat("Ab1", 12) }

// syncFixture captures one source file into a fresh vault and returns the
// vault plus a sibling destination directory (a "second disk").
func syncFixture(t *testing.T, content string) (*Vault, string) {
	t.Helper()
	srcRoot := t.TempDir()
	writeFile(t, filepath.Join(srcRoot, "session.jsonl"), content)
	v := testVault(t, Source{ID: "s", Root: srcRoot})
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	return v, filepath.Join(t.TempDir(), "replica")
}

func TestSyncScrubsPlantedTokenAndVerifiesOnArrival(t *testing.T) {
	token := plantedToken()
	v, dst := syncFixture(t, "turn 1 ok\nauth header used "+token+" for the call\nturn 2 ok\n")

	stats, problems, err := v.SyncTo(dst, 0)
	if err != nil {
		t.Fatalf("sync: %v", err)
	}
	if len(problems) != 0 {
		t.Fatalf("receiving-side verify problems: %+v", problems)
	}
	if stats.Copied != 1 || stats.Errors != 0 {
		t.Fatalf("stats = %+v, want 1 copy, 0 errors", stats)
	}
	if stats.Redacted == 0 {
		t.Fatalf("stats.Redacted = 0, want at least the planted token span")
	}

	// Scrub witness: the planted token-shaped string is provably absent from
	// the synced copy; the surrounding bytes survived.
	synced := readFile(t, filepath.Join(dst, "by-source", "s", "session.jsonl"))
	if strings.Contains(synced, token) {
		t.Fatalf("planted token LEAKED into the synced copy:\n%s", synced)
	}
	if !strings.Contains(synced, "[REDACTED:github_token]") {
		t.Fatalf("synced copy lacks the redaction placeholder:\n%s", synced)
	}
	if !strings.Contains(synced, "turn 1 ok") || !strings.Contains(synced, "turn 2 ok") {
		t.Fatalf("scrub damaged surrounding content:\n%s", synced)
	}
	// The scrub witness must also hold for the receiving MANIFEST — every
	// outbound byte, not just mirror content.
	man := readFile(t, filepath.Join(dst, ManifestName))
	if strings.Contains(man, token) {
		t.Fatalf("planted token LEAKED into the receiving manifest:\n%s", man)
	}

	// The receiving side re-runs verify independently (the arrival witness):
	// its own chain + the scrubbed mirror hashes are green.
	recv := &Vault{Dir: dst}
	rows, checked, recvProblems, err := recv.Verify(0)
	if err != nil {
		t.Fatalf("receiving-side verify: %v", err)
	}
	if len(recvProblems) != 0 {
		t.Fatalf("receiving-side verify problems: %+v", recvProblems)
	}
	if rows == 0 || checked != 1 {
		t.Fatalf("receiving-side verify rows=%d checked=%d, want a chained manifest + 1 mirror", rows, checked)
	}
}

func TestSyncIsIncrementalAcrossPasses(t *testing.T) {
	v, dst := syncFixture(t, "plain content, nothing to redact\n")
	if _, _, err := v.SyncTo(dst, 0); err != nil {
		t.Fatal(err)
	}
	stats, problems, err := v.SyncTo(dst, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(problems) != 0 {
		t.Fatalf("second pass verify problems: %+v", problems)
	}
	if stats.Copied != 0 || stats.Unchanged != 1 {
		t.Fatalf("second pass stats = %+v, want 0 copied / 1 unchanged", stats)
	}
}

func TestSyncRefusesCorruptSourceMirror(t *testing.T) {
	v, dst := syncFixture(t, "honest content\n")
	// Corrupt the SOURCE mirror behind the manifest's back: the sync must
	// refuse to ship bytes the chain does not attest.
	mirror := filepath.Join(v.Dir, "by-source", "s", "session.jsonl")
	if err := os.WriteFile(mirror, []byte("tampered bytes\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	stats, problems, err := v.SyncTo(dst, 0)
	if err != nil {
		t.Fatalf("sync: %v", err)
	}
	if stats.Errors != 1 || stats.Copied != 0 {
		t.Fatalf("stats = %+v, want the tampered mirror refused (1 error, 0 copies)", stats)
	}
	if _, statErr := os.Stat(filepath.Join(dst, "by-source", "s", "session.jsonl")); !os.IsNotExist(statErr) {
		t.Fatalf("tampered bytes travelled: %v", statErr)
	}
	// The refusal is still a verifiable receiving-side chain (skip rows chain too).
	if len(problems) != 0 {
		t.Fatalf("receiving-side verify problems: %+v", problems)
	}
}

func TestSyncRefusesBrokenSourceChain(t *testing.T) {
	v, dst := syncFixture(t, "content\n")
	manPath := filepath.Join(v.Dir, ManifestName)
	data := readFile(t, manPath)
	if err := os.WriteFile(manPath, []byte(strings.Replace(data, `"op":"capture-full"`, `"op":"capture-touch"`, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := v.SyncTo(dst, 0); err == nil {
		t.Fatal("sync of a vault with a broken chain must refuse before any byte copies")
	}
	if _, statErr := os.Stat(filepath.Join(dst, "by-source")); !os.IsNotExist(statErr) {
		t.Fatalf("bytes copied despite a broken source chain: %v", statErr)
	}
}

func TestSyncRefusesOverlappingDestination(t *testing.T) {
	v, _ := syncFixture(t, "content\n")
	if _, _, err := v.SyncTo(filepath.Join(v.Dir, "replica"), 0); err == nil {
		t.Fatal("sync into the source vault must refuse")
	}
	if _, _, err := v.SyncTo("", 0); err == nil {
		t.Fatal("sync with an empty destination must refuse")
	}
}
