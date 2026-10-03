package fakroot

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// samePath compares two directory paths after resolving symlinks. t.TempDir() hands back
// the unresolved form (/var/folders/... on macOS, where /var -> /private/var) while the
// ladder may hand back either form (Discover runs filepath.Abs on some rungs), so a
// lexical string compare is not a valid assertion here. Mirrors the convention already
// used by internal/selfupdate/cmd's TestDiscoverRepoRoot_ChildFakInCWD.
func samePath(t *testing.T, got, want string) bool {
	t.Helper()
	resolve := func(p string) string {
		if r, err := filepath.EvalSymlinks(p); err == nil {
			return r
		}
		return filepath.Clean(p)
	}
	return resolve(got) == resolve(want)
}

// writeRepoRootMain builds dir/cmd/fak/main.go as a REGULAR file — the structural proof
// IsRepoRoot requires.
func writeRepoRootMain(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(dir, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cmd", "fak", "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeFileAt writes data at rel under dir, creating parents.
func writeFileAt(t *testing.T, dir, rel, data string) string {
	t.Helper()
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// noEnv is the injected Getenv for a ladder that must not see a developer's real FAK_ROOT:
// rung 2 reads the process environment otherwise, which would make these tests machine-
// dependent (and would let an unrelated FAK_ROOT answer "is the companion discoverable?").
func noEnv(string) string { return "" }

// TestIsRepoRoot pins the ONE proof every rung of the ladder depends on. If IsRepoRoot ever
// stops requiring a regular cmd/fak/main.go, the read-root widening downstream becomes a
// default-open: any sibling directory would be admitted as a read root.
func TestIsRepoRoot(t *testing.T) {
	t.Run("public checkout with cmd/fak/main.go is a repo root", func(t *testing.T) {
		root := t.TempDir()
		writeRepoRootMain(t, root)
		if !IsRepoRoot(root) {
			t.Fatalf("IsRepoRoot(%q) = false, want true: a dir with a regular cmd/fak/main.go IS a public checkout", root)
		}
	})

	t.Run("cmd/fak/main.go as a DIRECTORY is not a repo root", func(t *testing.T) {
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "cmd", "fak", "main.go"), 0o755); err != nil {
			t.Fatal(err)
		}
		if IsRepoRoot(root) {
			t.Fatalf("IsRepoRoot(%q) = true, want false: the proof requires a REGULAR file, a directory named main.go must not pass", root)
		}
	})

	// THE ASYMMETRY INVARIANT: fak-private is the private companion checkout and has NO
	// cmd/fak tree. It therefore never satisfies the proof, which is the only reason a
	// public working tree can never reach a private one.
	t.Run("private companion shape is NOT a repo root", func(t *testing.T) {
		root := t.TempDir()
		writeFileAt(t, root, "platform/testruntime/doc.go", "package testruntime\n")
		writeFileAt(t, root, "cmd/fak-server/main.go", "package main\n")
		writeFileAt(t, root, "ops/iso-builder/main.go", "package main\n")
		if err := os.MkdirAll(filepath.Join(root, "cmd"), 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(filepath.Join(root, "cmd", "fak")); !os.IsNotExist(err) {
			t.Fatalf("fixture precondition broken: private shape must have no cmd/fak (stat err = %v)", err)
		}
		if IsRepoRoot(root) {
			t.Fatalf("IsRepoRoot(%q) = true, want false: a private-companion checkout (no cmd/fak) must never pass the proof", root)
		}
	})

	t.Run("empty dir is not a repo root", func(t *testing.T) {
		if IsRepoRoot("") {
			t.Fatal(`IsRepoRoot("") = true, want false`)
		}
		if IsRepoRoot("   ") {
			t.Fatal(`IsRepoRoot("   ") = true, want false: whitespace is not a path`)
		}
	})

	t.Run("nonexistent dir is not a repo root", func(t *testing.T) {
		if IsRepoRoot(filepath.Join(t.TempDir(), "nope")) {
			t.Fatal("IsRepoRoot on a nonexistent dir = true, want false")
		}
	})
}

// TestParseGoWorkUseDirs pins the go.work reader. The shape asserted first is the REAL
// fak-private/go.work, because that file is the whole reason the read-root widening exists:
// it names the sibling public checkout with `use`.
func TestParseGoWorkUseDirs(t *testing.T) {
	t.Run("real private go.work resolves . and ../fak against the go.work dir", func(t *testing.T) {
		dir := t.TempDir()
		work := filepath.Join(dir, "go.work")
		body := "go 1.26.0\n\nuse (\n\t.\n\t../fak\n)\n"
		if err := os.WriteFile(work, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		want := []string{filepath.Clean(dir), filepath.Join(filepath.Dir(dir), "fak")}
		got := ParseGoWorkUseDirs(work)
		if len(got) != len(want) {
			t.Fatalf("ParseGoWorkUseDirs = %v, want %v", got, want)
		}
		for i := range want {
			if !samePath(t, got[i], want[i]) {
				t.Fatalf("use[%d] = %q, want %q (dirs = %v)", i, got[i], want[i], got)
			}
		}
	})

	t.Run("single-line use directive", func(t *testing.T) {
		dir := t.TempDir()
		work := filepath.Join(dir, "go.work")
		if err := os.WriteFile(work, []byte("go 1.26.0\n\nuse ../fak\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		got := ParseGoWorkUseDirs(work)
		want := filepath.Join(filepath.Dir(dir), "fak")
		if len(got) != 1 || !samePath(t, got[0], want) {
			t.Fatalf("ParseGoWorkUseDirs = %v, want [%s]", got, want)
		}
	})

	t.Run("comment lines are ignored", func(t *testing.T) {
		dir := t.TempDir()
		work := filepath.Join(dir, "go.work")
		body := "go 1.26.0\n\n" +
			"// this workspace joins the private tree with the public one\n" +
			"use (\n" +
			"\t. // the private tree\n" +
			"\t../fak // the public tree\n" +
			")\n" +
			"\n" +
			"// a second block is a second block, not a continuation\n" +
			"use (\n" +
			"\t./extra\n" +
			")\n"
		if err := os.WriteFile(work, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		got := ParseGoWorkUseDirs(work)
		for _, entry := range got {
			if strings.Contains(entry, "public one") || strings.Contains(entry, "public tree") || strings.Contains(entry, "private tree") || strings.Contains(entry, "continuation") {
				t.Fatalf("comment text leaked into a use entry: %v", got)
			}
		}
		want := []string{filepath.Clean(dir), filepath.Join(filepath.Dir(dir), "fak"), filepath.Join(dir, "extra")}
		if len(got) != len(want) {
			t.Fatalf("ParseGoWorkUseDirs = %v, want %v", got, want)
		}
		for i := range want {
			if !samePath(t, got[i], want[i]) {
				t.Fatalf("use[%d] = %q, want %q (dirs = %v)", i, got[i], want[i], got)
			}
		}
	})

	t.Run("commented-out use directive is not a use entry", func(t *testing.T) {
		dir := t.TempDir()
		work := filepath.Join(dir, "go.work")
		if err := os.WriteFile(work, []byte("go 1.26.0\n\n// use ../retired\n\nuse (\n\t../fak\n)\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		got := ParseGoWorkUseDirs(work)
		want := filepath.Join(filepath.Dir(dir), "fak")
		if len(got) != 1 || !samePath(t, got[0], want) {
			t.Fatalf("ParseGoWorkUseDirs = %v, want exactly [%s] (a commented-out use must be inert)", got, want)
		}
	})

	t.Run("missing file yields nil", func(t *testing.T) {
		if got := ParseGoWorkUseDirs(filepath.Join(t.TempDir(), "go.work")); got != nil {
			t.Fatalf("ParseGoWorkUseDirs(missing) = %v, want nil", got)
		}
	})

	t.Run("directory instead of a file yields nil", func(t *testing.T) {
		work := filepath.Join(t.TempDir(), "go.work")
		if err := os.MkdirAll(work, 0o755); err != nil {
			t.Fatal(err)
		}
		if got := ParseGoWorkUseDirs(work); got != nil {
			t.Fatalf("ParseGoWorkUseDirs(directory) = %v, want nil", got)
		}
	})
}

// TestLadderDiscover pins the ladder rung by rung. Every case names the rung it exercises,
// because "Discover returns the sibling" is the property and "which rung found it" is what
// regresses silently when a rung is reordered or dropped.
func TestLadderDiscover(t *testing.T) {
	t.Run("rung 3: go.work use entry naming a public sibling", func(t *testing.T) {
		parent := t.TempDir()
		cwd := filepath.Join(parent, "priv")
		pub := filepath.Join(parent, "pub")
		if err := os.MkdirAll(cwd, 0o755); err != nil {
			t.Fatal(err)
		}
		writeRepoRootMain(t, pub)
		// `../pub` — deliberately NOT named `fak`, so rung 3 (go.work) is what answers and
		// not the sibling `../fak` rung.
		writeFileAt(t, cwd, "go.work", "go 1.26.0\n\nuse (\n\t.\n\t../pub\n)\n")

		got := Ladder{Cwd: cwd, Getenv: noEnv}.Discover()
		if got == "" {
			t.Fatalf("Discover() = \"\", want the go.work-declared public checkout %q", pub)
		}
		if !samePath(t, got, pub) {
			t.Fatalf("Discover() = %q, want %q", got, pub)
		}
	})

	t.Run("rung 3: use entry WITHOUT cmd/fak/main.go is never returned", func(t *testing.T) {
		parent := t.TempDir()
		cwd := filepath.Join(parent, "pub")
		priv := filepath.Join(parent, "priv")
		if err := os.MkdirAll(cwd, 0o755); err != nil {
			t.Fatal(err)
		}
		writeRepoRootMain(t, cwd) // the cwd IS public; its go.work names the private sibling
		writeFileAt(t, priv, "cmd/fak-server/main.go", "package main\n")
		writeFileAt(t, cwd, "go.work", "go 1.26.0\n\nuse ../priv\n")

		got := Ladder{Cwd: cwd, Getenv: noEnv}.Discover()
		if got != "" {
			t.Fatalf("Discover() = %q, want \"\": the private companion checkout fails the cmd/fak/main.go proof and must never be returned", got)
		}
		if IsRepoRoot(priv) {
			t.Fatalf("fixture precondition broken: %q must not pass IsRepoRoot", priv)
		}
	})

	t.Run("rung 2: FAK_ROOT through the injected Getenv", func(t *testing.T) {
		cwd := t.TempDir()
		envRoot := t.TempDir()
		writeRepoRootMain(t, envRoot)
		ladder := Ladder{
			Cwd: cwd,
			Getenv: func(k string) string {
				if k == "FAK_ROOT" {
					return envRoot
				}
				return ""
			},
		}
		got := ladder.Discover()
		if got == "" {
			t.Fatal(`Discover() = "", want the FAK_ROOT checkout`)
		}
		if !samePath(t, got, envRoot) {
			t.Fatalf("Discover() = %q, want %q", got, envRoot)
		}
	})

	t.Run("rung 2: FAK_ROOT that fails the proof is never returned", func(t *testing.T) {
		cwd := t.TempDir()
		notARoot := t.TempDir()
		writeFileAt(t, notARoot, "cmd/fak-server/main.go", "package main\n")
		ladder := Ladder{
			Cwd: cwd,
			Getenv: func(k string) string {
				if k == "FAK_ROOT" {
					return notARoot
				}
				return ""
			},
		}
		if got := ladder.Discover(); got != "" {
			t.Fatalf("Discover() = %q, want \"\": FAK_ROOT is proved like every other candidate", got)
		}
	})

	t.Run("rung 5: sibling ../fak resolves when the cwd has no go.work", func(t *testing.T) {
		parent := t.TempDir()
		cwd := filepath.Join(parent, "priv")
		if err := os.MkdirAll(cwd, 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(filepath.Join(cwd, "go.work")); !os.IsNotExist(err) {
			t.Fatalf("fixture precondition broken: cwd must have no go.work (stat err = %v)", err)
		}
		sibling := filepath.Join(parent, "fak")
		writeRepoRootMain(t, sibling)

		got := Ladder{Cwd: cwd, Getenv: noEnv}.Discover()
		if got == "" {
			t.Fatal(`Discover() = "", want the sibling ../fak checkout`)
		}
		if !samePath(t, got, sibling) {
			t.Fatalf("Discover() = %q, want %q", got, sibling)
		}
	})

	t.Run("rung 1: a probed GitRoot that IS a checkout wins", func(t *testing.T) {
		root := t.TempDir()
		writeRepoRootMain(t, root)
		ladder := Ladder{GitRoot: root, Cwd: t.TempDir(), Getenv: noEnv}
		got := ladder.Discover()
		if got == "" || !samePath(t, got, root) {
			t.Fatalf("Discover() = %q, want the GitRoot checkout %q", got, root)
		}
	})

	t.Run("nothing discoverable returns empty, never a guess", func(t *testing.T) {
		parent := t.TempDir()
		cwd := filepath.Join(parent, "priv")
		if err := os.MkdirAll(cwd, 0o755); err != nil {
			t.Fatal(err)
		}
		writeFileAt(t, cwd, "notes.md", "no companion here\n")
		if got := (Ladder{Cwd: cwd, Getenv: noEnv}).Discover(); got != "" {
			t.Fatalf("Discover() = %q, want \"\" when no rung proves a checkout", got)
		}
	})

	// The empty-input case: the MCP read engine calls Discover without a git probe (the
	// ladder spawns nothing), so Cwd=="" && GitRoot=="" is a live shape, not a theoretical
	// one. It must return "" instead of panicking or inventing a path.
	t.Run("empty Cwd and empty GitRoot return empty instead of panicking", func(t *testing.T) {
		// Pin the process cwd inside a temp dir with no `fak` child and no `fak` sibling, so
		// the only remaining candidates are the process-relative ones and the answer is
		// deterministic on any machine.
		t.Chdir(t.TempDir())
		if got := (Ladder{Getenv: noEnv}).Discover(); got != "" {
			t.Fatalf("Discover() with empty Cwd and GitRoot = %q, want \"\"", got)
		}
		if got := (Ladder{Cwd: "", GitRoot: "", Getenv: noEnv}).Discover(); got != "" {
			t.Fatalf("Discover() with explicit empty inputs = %q, want \"\"", got)
		}
	})
}
