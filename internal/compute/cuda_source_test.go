package compute

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

var cudaQuotedIncludeRE = regexp.MustCompile(`^\s*#\s*include\s*"([^"\r\n]+)"`)

// cudaProductionFiles lists the package's CUDA production sources. Contract
// checks read these rather than one filename, because the god-file split moves
// declarations between siblings.
func cudaProductionFiles(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package directory: %v", err)
	}
	var names []string
	for _, e := range entries {
		name := e.Name()
		if !e.IsDir() && strings.HasPrefix(name, "cuda") && strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
			names = append(names, name)
		}
	}
	if len(names) == 0 {
		t.Fatal("no CUDA production sources found")
	}
	return names
}

// cudaPackageSyntax merges the top-level declarations of every CUDA production
// source into one file, so a declaration is found whichever sibling holds it.
func cudaPackageSyntax(t *testing.T) (*token.FileSet, *ast.File) {
	t.Helper()
	fset := token.NewFileSet()
	merged := &ast.File{Name: ast.NewIdent("compute")}
	for _, name := range cudaProductionFiles(t) {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		merged.Decls = append(merged.Decls, f.Decls...)
	}
	return fset, merged
}

// cudaSourceText returns the exact source text of a node parsed by
// cudaPackageSyntax, read from the sibling file that declares it.
func cudaSourceText(t *testing.T, fset *token.FileSet, n ast.Node) string {
	t.Helper()
	start, end := fset.Position(n.Pos()), fset.Position(n.End())
	data, err := os.ReadFile(start.Filename)
	if err != nil {
		t.Fatalf("read %s: %v", start.Filename, err)
	}
	return string(data[start.Offset:end.Offset])
}

// cudaPackageText concatenates the CUDA production sources for marker checks
// that may land in any sibling file.
func cudaPackageText(t *testing.T) string {
	t.Helper()
	var out strings.Builder
	for _, name := range cudaProductionFiles(t) {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		out.Write(data)
		out.WriteByte('\n')
	}
	return out.String()
}

// cudaSourceClosure reads each CUDA source root and its repository-local quoted
// includes exactly once. It makes source-contract tests follow focused fragments
// such as cuda/*.cuh without requiring every assertion to know their filenames.
func cudaSourceClosure(t *testing.T, roots ...string) string {
	t.Helper()
	base, err := os.Getwd()
	if err != nil {
		t.Fatalf("resolve CUDA source base: %v", err)
	}
	base, err = filepath.Abs(base)
	if err != nil {
		t.Fatalf("resolve CUDA source base: %v", err)
	}

	const (
		visiting = 1
		visited  = 2
	)
	state := make(map[string]int)
	var out strings.Builder
	var visit func(string)
	visit = func(name string) {
		t.Helper()
		path := name
		if !filepath.IsAbs(path) {
			path = filepath.Join(base, path)
		}
		path, err = filepath.Abs(filepath.Clean(path))
		if err != nil {
			t.Fatalf("resolve CUDA source %q: %v", name, err)
		}
		rel, err := filepath.Rel(base, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			t.Fatalf("CUDA source %q escapes package directory", name)
		}
		switch state[path] {
		case visiting:
			t.Fatalf("CUDA quoted-include cycle at %s", filepath.ToSlash(rel))
		case visited:
			return
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read CUDA source %s: %v", filepath.ToSlash(rel), err)
		}
		state[path] = visiting
		out.Write(data)
		if len(data) == 0 || data[len(data)-1] != '\n' {
			out.WriteByte('\n')
		}
		scanner := bufio.NewScanner(strings.NewReader(string(data)))
		for scanner.Scan() {
			match := cudaQuotedIncludeRE.FindStringSubmatch(scanner.Text())
			if match != nil {
				visit(filepath.Join(filepath.Dir(path), match[1]))
			}
		}
		if err := scanner.Err(); err != nil {
			t.Fatalf("scan CUDA source %s: %v", filepath.ToSlash(rel), err)
		}
		state[path] = visited
	}
	for _, root := range roots {
		visit(root)
	}
	return out.String()
}
