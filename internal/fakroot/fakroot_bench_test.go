package fakroot

import (
	"os"
	"path/filepath"
	"testing"
)

// Benchmarks time the discovery ladder's hot paths without touching the process
// environment: every Ladder injects noEnv, and every fixture is built once in
// b.TempDir() outside the timed loop. Path assertions use the same symlink-tolerant
// resolution as samePath in fakroot_test.go, because Discover hands back either form
// depending on which rung answered.

// writeRepoRootMainB is the *testing.B twin of writeRepoRootMain: builds dir/cmd/fak/main.go
// as a REGULAR file — the structural proof IsRepoRoot requires.
func writeRepoRootMainB(b *testing.B, dir string) {
	b.Helper()
	if err := os.MkdirAll(filepath.Join(dir, "cmd", "fak"), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cmd", "fak", "main.go"), []byte("package main\n"), 0o644); err != nil {
		b.Fatal(err)
	}
}

// writeFileAtB is the *testing.B twin of writeFileAt: writes data at rel under dir,
// creating parents.
func writeFileAtB(b *testing.B, dir, rel, data string) string {
	b.Helper()
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
		b.Fatal(err)
	}
	return p
}

// benchSamePath compares two paths after resolving symlinks, mirroring samePath in
// fakroot_test.go but reporting through *testing.B.
func benchSamePath(b *testing.B, got, want string) bool {
	b.Helper()
	resolve := func(p string) string {
		if r, err := filepath.EvalSymlinks(p); err == nil {
			return r
		}
		return filepath.Clean(p)
	}
	return resolve(got) == resolve(want)
}

// BenchmarkDiscoverGitRootHit times the rung-1 fast path: a probed GitRoot that IS a public
// checkout answers immediately, so the whole ladder collapses to a single IsRepoRoot stat.
func BenchmarkDiscoverGitRootHit(b *testing.B) {
	b.ReportAllocs()
	pub := b.TempDir()
	writeRepoRootMainB(b, pub)
	cwd := b.TempDir() // distinct empty dir: rung 1 must answer before any cwd work matters
	if got := (Ladder{GitRoot: pub, Cwd: cwd, Getenv: noEnv}).Discover(); got == "" || !benchSamePath(b, got, pub) {
		b.Fatalf("pre-loop: Discover() = %q, want the GitRoot checkout %q", got, pub)
	}
	for i := 0; i < b.N; i++ {
		Ladder{GitRoot: pub, Cwd: cwd, Getenv: noEnv}.Discover()
	}
}

// BenchmarkDiscoverSiblingFakWalk times the FULL ladder walk that ends at rung 5: the cwd is
// a private-companion-shaped dir (notes only, no go.work) and only the sibling ../fak passes
// the proof, so every earlier rung must run and fail before the hit.
func BenchmarkDiscoverSiblingFakWalk(b *testing.B) {
	b.ReportAllocs()
	parent := b.TempDir()
	priv := filepath.Join(parent, "priv")
	sibling := filepath.Join(parent, "fak")
	if err := os.MkdirAll(priv, 0o755); err != nil {
		b.Fatal(err)
	}
	writeFileAtB(b, priv, "notes.md", "private companion, no go.work\n")
	writeRepoRootMainB(b, sibling)
	// Pin the process cwd into an empty dir so the process-relative candidates of rungs 4/5
	// ("fak", "../fak") are deterministic instead of depending on where the bench was launched.
	b.Chdir(b.TempDir())
	if got := (Ladder{Cwd: priv, Getenv: noEnv}).Discover(); got == "" || !benchSamePath(b, got, sibling) {
		b.Fatalf("pre-loop: Discover() = %q, want the sibling checkout %q", got, sibling)
	}
	for i := 0; i < b.N; i++ {
		Ladder{Cwd: priv, Getenv: noEnv}.Discover()
	}
}

// BenchmarkDiscoverNoMatch times the negative walk: no rung proves a checkout, so Discover
// runs the entire ladder and returns "". The pre-loop assertion pins that empty answer.
func BenchmarkDiscoverNoMatch(b *testing.B) {
	b.ReportAllocs()
	parent := b.TempDir()
	priv := filepath.Join(parent, "priv")
	if err := os.MkdirAll(priv, 0o755); err != nil {
		b.Fatal(err)
	}
	writeFileAtB(b, priv, "notes.md", "no companion here\n")
	b.Chdir(b.TempDir())
	if got := (Ladder{Cwd: priv, Getenv: noEnv}).Discover(); got != "" {
		b.Fatalf("pre-loop: Discover() = %q, want \"\" when no rung proves a checkout", got)
	}
	for i := 0; i < b.N; i++ {
		Ladder{Cwd: priv, Getenv: noEnv}.Discover()
	}
}

// BenchmarkIsRepoRootPositive times the stat proof on a real fixture: one Stat of
// cmd/fak/main.go plus the whitespace guard, on the happy path.
func BenchmarkIsRepoRootPositive(b *testing.B) {
	b.ReportAllocs()
	root := b.TempDir()
	writeRepoRootMainB(b, root)
	if !IsRepoRoot(root) {
		b.Fatal("pre-loop: IsRepoRoot(fixture) = false, want true")
	}
	for i := 0; i < b.N; i++ {
		IsRepoRoot(root)
	}
}

// BenchmarkIsRepoRootNegative times the same stat on an empty dir, where the Stat fails and
// the proof must stay closed.
func BenchmarkIsRepoRootNegative(b *testing.B) {
	b.ReportAllocs()
	empty := b.TempDir()
	if IsRepoRoot(empty) {
		b.Fatal("pre-loop: IsRepoRoot(empty dir) = true, want false")
	}
	for i := 0; i < b.N; i++ {
		IsRepoRoot(empty)
	}
}

// BenchmarkParseGoWorkUseDirsRealistic times the go.work reader on the realistic private
// shape: a go directive, comment lines, and a use block whose entries carry trailing
// comments. Each call reads the file, splits lines, and cleans three paths.
func BenchmarkParseGoWorkUseDirsRealistic(b *testing.B) {
	b.ReportAllocs()
	dir := b.TempDir()
	work := filepath.Join(dir, "go.work")
	body := "go 1.26.0\n\n" +
		"// this workspace joins the private tree with the public one\n" +
		"use (\n" +
		"\t.\n" +
		"\t../fak\n" +
		"\t./aux // auxiliary tooling module\n" +
		")\n"
	if err := os.WriteFile(work, []byte(body), 0o644); err != nil {
		b.Fatal(err)
	}
	if got := ParseGoWorkUseDirs(work); len(got) != 3 {
		b.Fatalf("pre-loop: ParseGoWorkUseDirs = %v, want 3 entries", got)
	}
	for i := 0; i < b.N; i++ {
		ParseGoWorkUseDirs(work)
	}
}
