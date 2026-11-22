package abi_test

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// A seam is one package-level Register* setter. Getters named Registered* only
// read a registry, so they are not attach points.
var seamFuncRE = regexp.MustCompile(`^func (Register[A-Za-z0-9_]*)\(`)

const modulePath = "github.com/anthony-chaudhary/fak"

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("repo root not found from %s: %v", root, err)
	}
	return root
}

func productionGoFiles(t *testing.T, dir string) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == "testdata" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

// internalSeams returns every seam under internal/ keyed "import/path.Func".
func internalSeams(t *testing.T, root string) map[string]bool {
	t.Helper()
	seams := map[string]bool{}
	for _, path := range productionGoFiles(t, filepath.Join(root, "internal")) {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		rel, _ := filepath.Rel(root, filepath.Dir(path))
		pkg := modulePath + "/" + filepath.ToSlash(rel)
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for sc.Scan() {
			m := seamFuncRE.FindStringSubmatch(sc.Text())
			if m == nil || strings.HasPrefix(m[1], "Registered") {
				continue
			}
			seams[pkg+"."+m[1]] = true
		}
		f.Close()
		if err := sc.Err(); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
	}
	return seams
}

// exportedSeams returns every internal symbol that pkg/ re-exports by
// package-level alias, keyed like internalSeams, with the exporting "pkg/x.Name"
// as the value. Callers intersect it with internalSeams.
func exportedSeams(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	fset := token.NewFileSet()
	for _, path := range productionGoFiles(t, filepath.Join(root, "pkg")) {
		file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		imports := map[string]string{}
		for _, imp := range file.Imports {
			ip, _ := strconv.Unquote(imp.Path.Value)
			name := ip[strings.LastIndex(ip, "/")+1:]
			if imp.Name != nil {
				name = imp.Name.Name
			}
			imports[name] = ip
		}
		rel, _ := filepath.Rel(root, filepath.Dir(path))
		for _, decl := range file.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.VAR {
				continue
			}
			for _, spec := range gen.Specs {
				vs := spec.(*ast.ValueSpec)
				for i, val := range vs.Values {
					sel, ok := val.(*ast.SelectorExpr)
					if !ok || i >= len(vs.Names) || !vs.Names[i].IsExported() {
						continue
					}
					x, ok := sel.X.(*ast.Ident)
					if !ok {
						continue
					}
					ip := imports[x.Name]
					if !strings.HasPrefix(ip, modulePath+"/internal/") {
						continue
					}
					out[ip+"."+sel.Sel.Name] = filepath.ToSlash(rel) + "." + vs.Names[i].Name
				}
			}
		}
	}
	return out
}

// TestSeamReachability reports how many registration seams a module outside this
// one can reach, and fails if a kernel-ABI seam loses its pkg/abi export: an
// out-of-tree driver has no other way to attach.
func TestSeamReachability(t *testing.T) {
	root := repoRoot(t)
	seams := internalSeams(t, root)
	exported := exportedSeams(t, root)

	if len(seams) == 0 {
		t.Fatal("found no Register* seams under internal/; the scan is broken")
	}

	reachable := 0
	var sealed []string
	for key := range seams {
		if _, ok := exported[key]; ok {
			reachable++
			continue
		}
		sealed = append(sealed, strings.TrimPrefix(key, modulePath+"/"))
	}
	sort.Strings(sealed)

	const abiPkg = modulePath + "/internal/abi."
	abiSeams := 0
	for key := range seams {
		if !strings.HasPrefix(key, abiPkg) {
			continue
		}
		abiSeams++
		if _, ok := exported[key]; !ok {
			t.Errorf("internal/abi.%s has no export in pkg/abi; out-of-tree drivers cannot reach it",
				strings.TrimPrefix(key, abiPkg))
		}
	}
	if abiSeams == 0 {
		t.Fatal("found no Register* seams in internal/abi; the scan is broken")
	}

	t.Logf("seam reachability: %d/%d Register* seams reachable from outside the module", reachable, len(seams))
	t.Logf("sealed under internal/ (%d): %s", len(sealed), strings.Join(sealed, ", "))
}

// TestSeamProbeFromForeignModule builds and runs the probe under a module path
// outside the fak prefix. It fails when an exported seam stops being importable
// from outside, and when pkg/abi exports a seam the probe does not exercise.
func TestSeamProbeFromForeignModule(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns the go toolchain")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH")
	}
	root := repoRoot(t)

	want := 0
	seams := internalSeams(t, root)
	for key, exportedAs := range exportedSeams(t, root) {
		if seams[key] && strings.HasPrefix(exportedAs, "pkg/abi.") {
			want++
		}
	}

	// The probe is copied out so the build never writes into the source tree.
	dir := t.TempDir()
	src, err := os.ReadFile(filepath.Join("testdata", "seamprobe", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), src, 0o644); err != nil {
		t.Fatal(err)
	}
	gomod := "module example.com/fakseamprobe\n\ngo 1.26.0\n\n" +
		"require " + modulePath + " v0.0.0\n\n" +
		"replace " + modulePath + " => " + filepath.ToSlash(root) + "\n"
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(gomod), 0o644); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(goBin, "run", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("foreign-module probe failed: %v\n%s", err, out)
	}
	got := fmt.Sprintf("seamprobe: %d registration seams", want)
	if !strings.Contains(string(out), got) {
		t.Fatalf("probe exercises a different seam count than pkg/abi exports; want %q in:\n%s", got, out)
	}
}
