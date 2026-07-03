package logvault

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func testVault(t *testing.T, src Source) *Vault {
	t.Helper()
	return &Vault{Dir: t.TempDir(), Sources: []Source{src}}
}

func TestManifestChainAppendsAndVerifies(t *testing.T) {
	dir := t.TempDir()
	m, err := OpenManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if _, err := m.Append(ManifestRow{TSUnixNano: int64(i), Op: OpFull, Source: "s", RelPath: "a.jsonl", Bytes: 1, SizeAfter: 1, SHA256: "x"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	// Reopen continues the same chain rather than forking it.
	m2, err := OpenManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	row, err := m2.Append(ManifestRow{Op: OpAppend, Source: "s", RelPath: "a.jsonl"})
	if err != nil {
		t.Fatal(err)
	}
	if row.Seq != 4 {
		t.Fatalf("reopened chain seq = %d, want 4", row.Seq)
	}
	m2.Close()
	n, err := VerifyManifest(filepath.Join(dir, ManifestName))
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if n != 4 {
		t.Fatalf("verified %d rows, want 4", n)
	}
}

func TestManifestVerifyNamesTamperedRow(t *testing.T) {
	dir := t.TempDir()
	m, _ := OpenManifest(dir)
	m.Append(ManifestRow{Op: OpFull, Source: "s", RelPath: "a", SHA256: "x"})
	m.Append(ManifestRow{Op: OpFull, Source: "s", RelPath: "b", SHA256: "y"})
	m.Close()
	path := filepath.Join(dir, ManifestName)
	data := readFile(t, path)
	if err := os.WriteFile(path, []byte(strings.Replace(data, `"rel_path":"a"`, `"rel_path":"Z"`, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyManifest(path); err == nil {
		t.Fatal("verify accepted a tampered row")
	}
}

func TestCaptureFullThenAppendThenRewrite(t *testing.T) {
	srcDir := t.TempDir()
	log := filepath.Join(srcDir, "loops.jsonl")
	writeFile(t, log, "row1\n")
	v := testVault(t, Source{ID: "s", Root: srcDir})

	stats, err := v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Full != 1 || stats[0].CopyBytes != 5 {
		t.Fatalf("first capture: %+v, want 1 full copy of 5 bytes", stats[0])
	}

	// Append-only growth captures only the delta.
	f, _ := os.OpenFile(log, os.O_APPEND|os.O_WRONLY, 0o644)
	f.WriteString("row2\n")
	f.Close()
	future := time.Now().Add(2 * time.Second)
	os.Chtimes(log, future, future) // ensure mtime differs across fast test steps
	stats, err = v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Append != 1 || stats[0].CopyBytes != 5 {
		t.Fatalf("append capture: %+v, want 1 append of 5 bytes", stats[0])
	}
	mirror := v.mirrorPath("s", "loops.jsonl")
	if got := readFile(t, mirror); got != "row1\nrow2\n" {
		t.Fatalf("mirror = %q, want both rows", got)
	}

	// In-place rewrite retires the old mirror to history.
	writeFile(t, log, "rewritten\n")
	future = future.Add(2 * time.Second)
	os.Chtimes(log, future, future)
	stats, err = v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Rewrite != 1 {
		t.Fatalf("rewrite capture: %+v, want 1 rewrite", stats[0])
	}
	if got := readFile(t, mirror); got != "rewritten\n" {
		t.Fatalf("mirror = %q, want rewritten content", got)
	}
	histDir := filepath.Join(v.Dir, "by-source", "s", ".history")
	ents, err := os.ReadDir(histDir)
	if err != nil || len(ents) != 1 {
		t.Fatalf("history = %v (err %v), want 1 retired version", ents, err)
	}

	// Steady state: nothing to do.
	stats, err = v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Unchanged != 1 || stats[0].Full+stats[0].Append+stats[0].Rewrite != 0 {
		t.Fatalf("steady state: %+v, want 1 unchanged", stats[0])
	}
}

func TestCaptureExcludesAndMissingRoot(t *testing.T) {
	srcDir := t.TempDir()
	writeFile(t, filepath.Join(srcDir, "keep.jsonl"), "k\n")
	writeFile(t, filepath.Join(srcDir, "tool.exe"), "bin")
	writeFile(t, filepath.Join(srcDir, ".oauth-token"), "secret")
	writeFile(t, filepath.Join(srcDir, "tmp", "scratch.txt"), "big")
	v := testVault(t, Source{ID: "s", Root: srcDir, Excludes: []string{"tmp/"}})
	stats, err := v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Files != 1 || stats[0].Full != 1 {
		t.Fatalf("capture with excludes: %+v, want exactly keep.jsonl", stats[0])
	}
	if _, err := os.Stat(v.mirrorPath("s", ".oauth-token")); !os.IsNotExist(err) {
		t.Fatal("credential file must never be captured")
	}

	missing := &Vault{Dir: v.Dir, Sources: []Source{{ID: "gone", Root: filepath.Join(srcDir, "no-such-dir")}}}
	stats, err = missing.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if !stats[0].Missing || stats[0].Files != 0 {
		t.Fatalf("missing root: %+v, want valid-empty", stats[0])
	}
}

func TestPlanMatchesCaptureAndVerifyCatchesBitrot(t *testing.T) {
	srcDir := t.TempDir()
	writeFile(t, filepath.Join(srcDir, "a.jsonl"), "aaaa\n")
	writeFile(t, filepath.Join(srcDir, "sub", "b.log"), "bb\n")
	v := testVault(t, Source{ID: "s", Root: srcDir})

	plan, err := v.Plan()
	if err != nil {
		t.Fatal(err)
	}
	if plan[0].Full != 2 || plan[0].CopyBytes != 8 {
		t.Fatalf("plan: %+v, want 2 full copies / 8 bytes", plan[0])
	}
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	plan, err = v.Plan()
	if err != nil {
		t.Fatal(err)
	}
	if plan[0].Unchanged != 2 || plan[0].CopyBytes != 0 {
		t.Fatalf("post-capture plan: %+v, want all unchanged", plan[0])
	}

	chainRows, checked, problems, err := v.Verify(0)
	if err != nil {
		t.Fatal(err)
	}
	if chainRows != 2 || checked != 2 || len(problems) != 0 {
		t.Fatalf("verify clean vault: rows=%d checked=%d problems=%v", chainRows, checked, problems)
	}

	// Flip a byte in a mirror: verify must name it.
	writeFile(t, v.mirrorPath("s", "a.jsonl"), "AAAA\n")
	_, _, problems, err = v.Verify(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(problems) != 1 || problems[0].RelPath != "a.jsonl" {
		t.Fatalf("verify after bitrot: %v, want a.jsonl named", problems)
	}
}

func TestTornManifestTailIsRepairedOnReopen(t *testing.T) {
	dir := t.TempDir()
	m, _ := OpenManifest(dir)
	m.Append(ManifestRow{Op: OpFull, Source: "s", RelPath: "a", SHA256: "x"})
	m.Append(ManifestRow{Op: OpFull, Source: "s", RelPath: "b", SHA256: "y"})
	m.Close()
	path := filepath.Join(dir, ManifestName)
	// Simulate a crash mid-append: partial row bytes, no trailing newline.
	f, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
	f.WriteString(`{"seq":3,"ts`)
	f.Close()

	m2, err := OpenManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	row, err := m2.Append(ManifestRow{Op: OpFull, Source: "s", RelPath: "c", SHA256: "z"})
	if err != nil {
		t.Fatal(err)
	}
	m2.Close()
	if row.Seq != 3 {
		t.Fatalf("post-repair seq = %d, want 3 (torn tail dropped, chain continued)", row.Seq)
	}
	n, err := VerifyManifest(path)
	if err != nil || n != 3 {
		t.Fatalf("verify after torn-tail repair: n=%d err=%v, want 3 rows clean", n, err)
	}
}

func TestInterruptedAppendHealsViaRewrite(t *testing.T) {
	srcDir := t.TempDir()
	log := filepath.Join(srcDir, "hot.jsonl")
	writeFile(t, log, "row1\n")
	v := testVault(t, Source{ID: "s", Root: srcDir})
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	// Corrupt the mirror the way an interrupted append would: extra bytes past
	// the recorded state.
	mirror := v.mirrorPath("s", "hot.jsonl")
	mf, _ := os.OpenFile(mirror, os.O_WRONLY|os.O_APPEND, 0o644)
	mf.WriteString("GARBAGE")
	mf.Close()
	// Source grows normally.
	sf, _ := os.OpenFile(log, os.O_APPEND|os.O_WRONLY, 0o644)
	sf.WriteString("row2\n")
	sf.Close()
	future := time.Now().Add(2 * time.Second)
	os.Chtimes(log, future, future)

	stats, err := v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Append != 0 || stats[0].Rewrite != 1 {
		t.Fatalf("capture over diverged mirror: %+v, want a rewrite (append would duplicate bytes)", stats[0])
	}
	if got := readFile(t, mirror); got != "row1\nrow2\n" {
		t.Fatalf("healed mirror = %q, want clean source content", got)
	}
	if _, _, problems, err := v.Verify(0); err != nil || len(problems) != 0 {
		t.Fatalf("verify after heal: problems=%v err=%v", problems, err)
	}
}

func TestCaptureRefusedWhileLockHeld(t *testing.T) {
	srcDir := t.TempDir()
	writeFile(t, filepath.Join(srcDir, "a.jsonl"), "a\n")
	v := testVault(t, Source{ID: "s", Root: srcDir})
	if err := os.MkdirAll(v.Dir, 0o755); err != nil {
		t.Fatal(err)
	}
	lock, err := os.OpenFile(filepath.Join(v.Dir, LockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		t.Fatal(err)
	}
	defer flock.Unlock(lock)
	if _, err := v.Capture(); err == nil {
		t.Fatal("concurrent capture must be refused while the vault lock is held")
	}
}

func TestVerifyCatchesManifestLossAndTruncation(t *testing.T) {
	srcDir := t.TempDir()
	writeFile(t, filepath.Join(srcDir, "a.jsonl"), "a\n")
	writeFile(t, filepath.Join(srcDir, "b.jsonl"), "b\n")
	v := testVault(t, Source{ID: "s", Root: srcDir})
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}

	// Tail truncation: drop the last manifest row; the anchor must flag it.
	manPath := filepath.Join(v.Dir, ManifestName)
	lines := strings.Split(strings.TrimSuffix(readFile(t, manPath), "\n"), "\n")
	if err := os.WriteFile(manPath, []byte(strings.Join(lines[:len(lines)-1], "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, problems, err := v.Verify(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(problems) == 0 {
		t.Fatal("tail-truncated manifest must fail verify via the head anchor")
	}

	// Whole-manifest loss over a populated vault must not verify clean.
	if err := os.Remove(manPath); err != nil {
		t.Fatal(err)
	}
	_, _, problems, err = v.Verify(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(problems) == 0 {
		t.Fatal("deleted manifest over populated by-source/ must fail verify")
	}
}

func TestMTimeTouchWritesRowThenGoesQuiet(t *testing.T) {
	srcDir := t.TempDir()
	log := filepath.Join(srcDir, "a.jsonl")
	writeFile(t, log, "same\n")
	v := testVault(t, Source{ID: "s", Root: srcDir})
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(3 * time.Second)
	os.Chtimes(log, future, future) // touch: same content, new mtime
	stats, err := v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Unchanged != 1 {
		t.Fatalf("touch capture: %+v, want verified-unchanged", stats[0])
	}
	rows, _ := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	last := rows[len(rows)-1]
	if last.Op != OpTouch {
		t.Fatalf("touch must write a state-advancing row, got op %q", last.Op)
	}
	// Third capture: the recorded mtime now matches — cheap path, no new row.
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	rows2, _ := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if len(rows2) != len(rows) {
		t.Fatalf("steady state after touch grew the manifest: %d -> %d rows", len(rows), len(rows2))
	}
}

func TestSourceOverlappingVaultIsRefused(t *testing.T) {
	dir := t.TempDir()
	v := &Vault{Dir: dir, Sources: []Source{{ID: "self", Root: dir}}}
	writeFile(t, filepath.Join(dir, "by-source", "x", "f.jsonl"), "x\n")
	if _, err := v.Capture(); err == nil {
		t.Fatal("a source rooted at the vault itself must be refused")
	}
}

func TestIncludeListRestrictsCapture(t *testing.T) {
	srcDir := t.TempDir()
	writeFile(t, filepath.Join(srcDir, "fak-shadow-ledger.jsonl"), "s\n")
	writeFile(t, filepath.Join(srcDir, "telemetry", "t.json"), "t\n")
	writeFile(t, filepath.Join(srcDir, "projects", "other", "big.jsonl"), "nope\n")
	writeFile(t, filepath.Join(srcDir, "settings.json"), "nope\n")
	v := testVault(t, Source{
		ID:       "s",
		Root:     srcDir,
		Includes: []string{"fak-*.jsonl", "telemetry/"},
	})
	stats, err := v.Capture()
	if err != nil {
		t.Fatal(err)
	}
	if stats[0].Full != 2 {
		t.Fatalf("include-listed capture: %+v, want exactly the ledger + telemetry file", stats[0])
	}
	if _, err := os.Stat(v.mirrorPath("s", "settings.json")); !os.IsNotExist(err) {
		t.Fatal("non-included file must not be captured")
	}
}

func TestHarnessProjectSlug(t *testing.T) {
	if got := harnessProjectSlug(`C:\work\fak`); got != "C--work-fak" {
		t.Fatalf("slug = %q, want C--work-fak", got)
	}
	if got := harnessProjectSlug("/home/u/fak"); got != "-home-u-fak" {
		t.Fatalf("slug = %q, want -home-u-fak", got)
	}
}
