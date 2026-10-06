package witness

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"sort"
	"strconv"
	"strings"
)

// fixOnlyTestReferences predicts, from Git blobs alone, which changed test
// files cannot compile at the parent: a file that names a package-level
// identifier the fix introduced. It maps each such file to the names it uses.
// The prediction only ever selects files the compiler would also reject, so a
// miss (methods, fields, signature changes) falls through to the build.
func fixOnlyTestReferences(ctx context.Context, git Runner, repoDir, commit, parent string, tests []string) map[string][]string {
	if git == nil {
		git = gitRunner
	}
	byDir := map[string][]string{}
	for _, rel := range tests {
		rel = strings.ReplaceAll(strings.TrimSpace(rel), "\\", "/")
		if strings.HasSuffix(rel, "_test.go") {
			byDir[path.Dir(rel)] = append(byDir[path.Dir(rel)], rel)
		}
	}
	out := map[string][]string{}
	for dir, files := range byDir {
		added := fixIntroducedNames(ctx, git, repoDir, commit, parent, dir)
		if len(added) == 0 {
			continue
		}
		for _, rel := range files {
			source, code, err := git(ctx, repoDir, "show", commit+":"+rel)
			if err != nil || code != 0 {
				continue
			}
			if names := testFileFixOnlyNames(source, dir, added); len(names) > 0 {
				out[rel] = names
			}
		}
	}
	return out
}

// fixIntroducedNames is the set of top-level non-method names declared by the
// candidate's changed non-test files in dir and absent from their parent
// versions. A name cannot move in from an unchanged file without a duplicate
// declaration, so the changed files alone decide it.
func fixIntroducedNames(ctx context.Context, git Runner, repoDir, commit, parent, dir string) map[string]bool {
	diff, code, err := git(ctx, repoDir, "diff", "--name-only", "--no-renames", parent, commit, "--", dir+"/")
	if err != nil || code != 0 {
		return nil
	}
	before, after := map[string]bool{}, map[string]bool{}
	for _, rel := range strings.Split(diff, "\n") {
		rel = strings.TrimSpace(rel)
		if path.Dir(rel) != dir || !strings.HasSuffix(rel, ".go") || strings.HasSuffix(rel, "_test.go") {
			continue
		}
		for rev, into := range map[string]map[string]bool{parent: before, commit: after} {
			if source, c, e := git(ctx, repoDir, "show", rev+":"+rel); e == nil && c == 0 {
				topLevelDeclNames(source, into)
			}
		}
	}
	added := map[string]bool{}
	for name := range after {
		if !before[name] && name != "_" && name != "init" {
			added[name] = true
		}
	}
	return added
}

func topLevelDeclNames(source string, into map[string]bool) {
	file, err := parser.ParseFile(token.NewFileSet(), "src.go", source, parser.SkipObjectResolution)
	if err != nil {
		return
	}
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Recv == nil {
				into[d.Name.Name] = true
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					into[s.Name.Name] = true
				case *ast.ValueSpec:
					for _, n := range s.Names {
						into[n.Name] = true
					}
				}
			}
		}
	}
}

// testFileFixOnlyNames returns the fix-introduced names a test file uses: a
// package-scope identifier for an in-package test, or pkg.Name through the
// import of dir for an external _test package.
func testFileFixOnlyNames(source, dir string, added map[string]bool) []string {
	file, err := parser.ParseFile(token.NewFileSet(), "src_test.go", source, 0)
	if err != nil {
		return nil
	}
	used := map[string]bool{}
	if strings.HasSuffix(file.Name.Name, "_test") {
		local := ""
		for _, imp := range file.Imports {
			p, err := strconv.Unquote(imp.Path.Value)
			if err != nil || !strings.HasSuffix(p, "/"+dir) {
				continue
			}
			local = path.Base(p)
			if imp.Name != nil {
				local = imp.Name.Name
			}
		}
		if local == "" || local == "_" || local == "." {
			return nil
		}
		ast.Inspect(file, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if x, ok := sel.X.(*ast.Ident); ok && x.Name == local && added[sel.Sel.Name] {
					used[sel.Sel.Name] = true
				}
			}
			return true
		})
	} else {
		for _, ident := range file.Unresolved {
			if added[ident.Name] {
				used[ident.Name] = true
			}
		}
	}
	names := make([]string, 0, len(used))
	for name := range used {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
