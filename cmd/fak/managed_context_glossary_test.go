package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

func TestManagedContextGlossaryCitationsResolve(t *testing.T) {
	root := filepath.Join("..", "..")
	docs := readManagedContextDocs(t, root)
	if issues := lintManagedContextDocs(root, docs); len(issues) > 0 {
		t.Fatalf("managed-context citations drifted:\n%s", strings.Join(issues, "\n"))
	}
}

func TestManagedContextGlossaryGuardCatchesBrokenCitation(t *testing.T) {
	root := filepath.Join("..", "..")
	docs := readManagedContextDocs(t, root)
	docs["docs/managed-context-glossary.md"] = strings.Replace(docs["docs/managed-context-glossary.md"],
		"ctxplan.Assumption", "ctxplan.NoSuchAssumption", 1)
	docs["docs/managed-context-continuous-usage.md"] = strings.Replace(docs["docs/managed-context-continuous-usage.md"],
		"../internal/ctxplan/pagefault.go", "../internal/ctxplan/fault.go", 1)

	issues := lintManagedContextDocs(root, docs)
	joined := strings.Join(issues, "\n")
	for _, want := range []string{"ctxplan.NoSuchAssumption", "ctxplan.PageFaultOutcome", "fault.go"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("mutated docs did not trip %q; issues:\n%s", want, joined)
		}
	}
}

func readManagedContextDocs(t *testing.T, root string) map[string]string {
	t.Helper()
	paths := []string{
		"docs/managed-context-glossary.md",
		"docs/managed-context-continuous-usage.md",
	}
	out := make(map[string]string, len(paths))
	for _, p := range paths {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(p)))
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		out[p] = string(b)
	}
	return out
}

var (
	managedContextSymbolRe = regexp.MustCompile("`?\\b([a-z][a-z0-9]*)\\.([A-Z][A-Za-z0-9_]*)\\b`?")
	managedContextFileRe   = regexp.MustCompile("(?:\\.\\./)?internal/[A-Za-z0-9_./-]+\\.go")
	managedContextLinkRe   = regexp.MustCompile("\\[\\s*`?([a-z][a-z0-9]*)\\.([A-Z][A-Za-z0-9_]*)`?\\s*\\]\\(([^)]*(?:\\.\\./)?internal/[^)]*\\.go)\\)")
	managedContextParenRe  = regexp.MustCompile("`([a-z][a-z0-9]*)\\.([A-Z][A-Za-z0-9_]*)`\\s*\\(([^)]*(?:\\.\\./)?internal/[^)]*\\.go)[^)]*\\)")
)

type managedContextRef struct {
	Doc    string
	Pkg    string
	Symbol string
	File   string
}

func lintManagedContextDocs(root string, docs map[string]string) []string {
	var issues []string
	allSymbols := map[string]managedContextRef{}
	var linked []managedContextRef
	files := map[string]string{}
	for doc, body := range docs {
		for _, m := range managedContextSymbolRe.FindAllStringSubmatch(body, -1) {
			ref := managedContextRef{Doc: doc, Pkg: m[1], Symbol: m[2]}
			allSymbols[ref.Pkg+"."+ref.Symbol] = ref
		}
		for _, m := range managedContextFileRe.FindAllString(body, -1) {
			files[normalizeManagedContextPath(m)] = doc
		}
		for _, m := range managedContextLinkRe.FindAllStringSubmatch(body, -1) {
			linked = append(linked, managedContextRef{Doc: doc, Pkg: m[1], Symbol: m[2], File: normalizeManagedContextPath(m[3])})
		}
		for _, m := range managedContextParenRe.FindAllStringSubmatch(body, -1) {
			linked = append(linked, managedContextRef{Doc: doc, Pkg: m[1], Symbol: m[2], File: normalizeManagedContextPath(m[3])})
		}
	}

	pkgCache := map[string]map[string]bool{}
	for key, ref := range allSymbols {
		symbols, ok := pkgCache[ref.Pkg]
		if !ok {
			var err error
			symbols, err = packageSymbols(root, ref.Pkg)
			if err != nil {
				issues = append(issues, fmt.Sprintf("%s: %s cannot resolve package internal/%s: %v", ref.Doc, key, ref.Pkg, err))
				continue
			}
			pkgCache[ref.Pkg] = symbols
		}
		if !symbols[ref.Symbol] {
			issues = append(issues, fmt.Sprintf("%s: %s not defined in internal/%s", ref.Doc, key, ref.Pkg))
		}
	}

	for file, doc := range files {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(file))); err != nil {
			issues = append(issues, fmt.Sprintf("%s: linked file %s does not exist: %v", doc, file, err))
		}
	}
	for _, ref := range linked {
		symbols, err := fileSymbols(root, ref.File)
		if err != nil {
			issues = append(issues, fmt.Sprintf("%s: linked file %s for %s.%s cannot be parsed: %v", ref.Doc, ref.File, ref.Pkg, ref.Symbol, err))
			continue
		}
		if !symbols[ref.Symbol] {
			issues = append(issues, fmt.Sprintf("%s: %s.%s links to %s, but that file does not define %s", ref.Doc, ref.Pkg, ref.Symbol, ref.File, ref.Symbol))
		}
	}
	sort.Strings(issues)
	return issues
}

func normalizeManagedContextPath(p string) string {
	p = strings.TrimSpace(strings.Trim(p, "`"))
	p = strings.TrimPrefix(p, "../")
	return filepath.ToSlash(filepath.Clean(filepath.FromSlash(p)))
}

func packageSymbols(root, pkg string) (map[string]bool, error) {
	dir := filepath.Join(root, "internal", pkg)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	out := map[string]bool{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		symbols, err := symbolsInGoFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		for s := range symbols {
			out[s] = true
		}
	}
	return out, nil
}

func fileSymbols(root, path string) (map[string]bool, error) {
	return symbolsInGoFile(filepath.Join(root, filepath.FromSlash(path)))
}

func symbolsInGoFile(path string) (map[string]bool, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return nil, err
	}
	out := map[string]bool{}
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			out[d.Name.Name] = true
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					out[s.Name.Name] = true
				case *ast.ValueSpec:
					for _, n := range s.Names {
						out[n.Name] = true
					}
				}
			}
		}
	}
	return out, nil
}
