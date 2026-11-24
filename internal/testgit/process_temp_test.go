package testgit

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

func isolateTempDir(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("TMP", tmp)
	t.Setenv("TEMP", tmp)
	t.Setenv("TMPDIR", tmp)
	if os.TempDir() != tmp {
		t.Skipf("os.TempDir() = %q, not overridable to %q on this host", os.TempDir(), tmp)
	}
	return tmp
}

func deadPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	if err := cmd.Run(); err != nil {
		t.Fatalf("exited child: %v", err)
	}
	pid := cmd.Process.Pid
	if processalive.Check(pid) {
		t.Skipf("pid %d reused before the probe", pid)
	}
	return pid
}

func TestProcessTempDirReapsDeadOwnersAndRemovesOwn(t *testing.T) {
	tmp := isolateTempDir(t)
	mk := func(name string, mtime time.Time) string {
		p := filepath.Join(tmp, name)
		if err := os.MkdirAll(filepath.Join(p, "sub"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, mtime, mtime); err != nil {
			t.Fatal(err)
		}
		return p
	}
	dead := mk("ptd-x-pid"+strconv.Itoa(deadPID(t))+"-1", time.Now())
	live := mk("ptd-x-pid"+strconv.Itoa(os.Getppid())+"-2", time.Now())
	legacyOld := mk("ptd-x-123", time.Now().Add(-3*time.Hour))
	legacyFresh := mk("ptd-x-456", time.Now())
	other := mk("other-pid1-1", time.Now().Add(-3*time.Hour))

	dir, err := ProcessTempDir("ptd-x-")
	if err != nil {
		t.Fatal(err)
	}
	if pid, ok := processalive.OwnedTempOwner(filepath.Base(dir), "ptd-x-"); !ok || pid != os.Getpid() {
		t.Fatalf("owner of %q = %d,%v; want self", dir, pid, ok)
	}
	for _, p := range []string{dead, legacyOld} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s survived; want reaped", p)
		}
	}
	for _, p := range []string{live, legacyFresh, other, dir} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s removed; want kept: %v", p, err)
		}
	}
	RemoveProcessTempDirs()
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("own dir %s survived RemoveProcessTempDirs", dir)
	}
	if _, err := os.Stat(live); err != nil {
		t.Errorf("live owner's dir removed: %v", err)
	}
}

func TestParentProcessTempDirSkipsInReexecChild(t *testing.T) {
	tmp := isolateTempDir(t)
	t.Setenv(processTempParentEnv("ptd-parent-"), strconv.Itoa(os.Getppid()))
	dir, err := ParentProcessTempDir("ptd-parent-")
	if err != nil || dir != "" {
		t.Fatalf("child ParentProcessTempDir = %q, %v; want \"\", nil", dir, err)
	}
	if entries, _ := os.ReadDir(tmp); len(entries) != 0 {
		t.Fatalf("child created %d entries; want none", len(entries))
	}

	t.Setenv(processTempParentEnv("ptd-parent-"), strconv.Itoa(deadPID(t)))
	dir, err = ParentProcessTempDir("ptd-parent-")
	if err != nil || dir == "" {
		t.Fatalf("parent ParentProcessTempDir = %q, %v; want a dir", dir, err)
	}
	if got := os.Getenv(processTempParentEnv("ptd-parent-")); got != strconv.Itoa(os.Getpid()) {
		t.Fatalf("owner env = %q; want self", got)
	}
	RemoveProcessTempDirs()
}

func TestProcessTempOwnerParsesOnlyMarkedNames(t *testing.T) {
	for name, want := range map[string]int{
		"p-pid42-123": 42, "p-pid42": 0, "p-123": 0, "p-pid-1": 0, "p-pidx-1": 0, "p-pid0-1": 0,
	} {
		got, ok := processalive.OwnedTempOwner(name, "p-")
		if (want != 0) != ok || got != want {
			t.Errorf("processalive.OwnedTempOwner(%q) = %d,%v; want %d", name, got, ok, want)
		}
	}
}

// TestNoRawMkdirTempInTestMain is the gate: a TestMain temp dir made with
// os.MkdirTemp leaks whenever os.Exit skips its cleanup or a re-executed child
// exits inside its test. Use ProcessTempDir and ExitProcessTemp instead.
func TestNoRawMkdirTempInTestMain(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Skipf("repo root not found at %s", root)
	}
	var offenders []string
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			n := d.Name()
			if p != root && (strings.HasPrefix(n, ".") || strings.HasPrefix(n, "_") || n == "testdata" || n == "vendor" || n == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, "_test.go") {
			return nil
		}
		src, err := os.ReadFile(p)
		if err != nil || !strings.Contains(string(src), "func TestMain(") || !strings.Contains(string(src), "MkdirTemp") {
			return nil
		}
		offenders = append(offenders, rawMkdirTempInTestMain(p, src)...)
		return nil
	})
	if len(offenders) > 0 {
		t.Fatalf("TestMain calls os.MkdirTemp (leaks on os.Exit / re-exec); use testgit.ProcessTempDir + ExitProcessTemp:\n  %s", strings.Join(offenders, "\n  "))
	}
}

func rawMkdirTempInTestMain(path string, src []byte) []string {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, src, 0)
	if err != nil {
		return nil
	}
	var out []string
	for _, decl := range f.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Recv != nil || fd.Name.Name != "TestMain" || fd.Body == nil {
			continue
		}
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || sel.Sel.Name != "MkdirTemp" {
				return true
			}
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "os" {
				out = append(out, fset.Position(call.Pos()).String())
			}
			return true
		})
	}
	return out
}

func TestRawMkdirTempInTestMainDetector(t *testing.T) {
	bad := []byte("package p\nimport (\"os\";\"testing\")\nfunc TestMain(m *testing.M) { d, _ := os.MkdirTemp(\"\", \"x\"); _ = d; func() { os.MkdirTemp(\"\", \"y\") }(); os.Exit(m.Run()) }\n")
	if got := rawMkdirTempInTestMain("bad_test.go", bad); len(got) != 2 {
		t.Fatalf("detector found %d calls; want 2: %v", len(got), got)
	}
	good := []byte("package p\nimport (\"os\";\"testing\")\nfunc TestMain(m *testing.M) { os.Exit(m.Run()) }\nfunc helper() { os.MkdirTemp(\"\", \"x\") }\n")
	if got := rawMkdirTempInTestMain("good_test.go", good); len(got) != 0 {
		t.Fatalf("detector flagged %v outside TestMain", got)
	}
}
