// Package dependencyquarantine enforces the repository dependency budget and ensures
// external dependencies remain quarantined in isolated tools submodules.
//
// Invariant: dependency quarantine enforcement is fail-closed and bounded. Any unreviewed
// root go.mod require entry, root go.sum checksum line, or nested tool facade import of
// non-stdlib packages immediately yields structured violations and blocks repository verification.
package dependencyquarantine

import (
	"bufio"
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

var allowedRootRequires = map[string]string{
	"golang.org/x/sys":  "v0.46.0",
	"golang.org/x/term": "v0.44.0",
}

var allowedRootSum = map[string]bool{
	"golang.org/x/sys v0.46.0 h1:noSf2Fq6F8DBgS+LysIkx7rIExoNHJsxOAtPp4rthXw=":         true,
	"golang.org/x/sys v0.46.0/go.mod h1:4GL1E5IUh+htKOUEOaiffhrAeqysfVGipDYzABqnCmw=":  true,
	"golang.org/x/term v0.44.0 h1:0rLvDRCtNj0gZkyIXhCyOb2OAzEhLVqc4B+hrsBhrmc=":        true,
	"golang.org/x/term v0.44.0/go.mod h1:7ze4MdzUzLXpSAoFP1H0bOI9aXDqveSvatT5vKcFh2Y=": true,
}

// Violation describes an unreviewed dependency or facade policy violation.
type Violation struct{ Path, Reason string }

func (v Violation) Error() string { return v.Path + ": " + v.Reason }

// Check scans the repository root for root dependency drift and unreviewed nested tool facades.
//
// Contract: Check executes fail-closed; if any file cannot be read or parsed, it returns
// a non-nil error, and if any dependency policy invariant is broken, it returns non-empty violations.
func Check(root string) ([]Violation, error) {
	var out []Violation
	req, err := rootRequires(filepath.Join(root, "go.mod"))
	if err != nil {
		return nil, err
	}
	if len(req) != len(allowedRootRequires) {
		out = append(out, Violation{"go.mod", fmt.Sprintf("root require set changed: got %v; explicit dependency-budget review required", req)})
	}
	for mod, want := range allowedRootRequires {
		if req[mod] != want {
			out = append(out, Violation{"go.mod", fmt.Sprintf("require %s = %q, want reviewed %q", mod, req[mod], want)})
		}
	}
	sum, err := lines(filepath.Join(root, "go.sum"))
	if err != nil {
		return nil, err
	}
	if len(sum) != len(allowedRootSum) {
		out = append(out, Violation{"go.sum", "root checksum set changed; explicit dependency-budget review required"})
	}
	for _, line := range sum {
		if !allowedRootSum[line] {
			out = append(out, Violation{"go.sum", "unreviewed checksum: " + line})
		}
	}

	modules, err := nestedModules(root)
	if err != nil {
		return nil, err
	}
	for _, mod := range modules {
		rel := filepath.ToSlash(mustRel(root, mod))
		if !strings.HasPrefix(rel, "tools/") {
			continue
		}
		facade := filepath.Dir(mod)
		if err := stdlibOnly(facade); err != nil {
			out = append(out, Violation{filepath.ToSlash(mustRel(root, facade)), err.Error()})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path == out[j].Path {
			return out[i].Reason < out[j].Reason
		}
		return out[i].Path < out[j].Path
	})
	return out, nil
}

// NestedModules discovers all nested modules under root by searching for nested go.mod files.
//
// Key invariant: NestedModules ignores hidden directories, scratch spaces, and repo root itself,
// guaranteeing bounded traversal across legitimate nested modules.
func NestedModules(root string) ([]string, error) { return nestedModules(root) }

func nestedModules(root string) ([]string, error) {
	var mods []string
	skip := map[string]bool{".git": true, "_scratch": true, "_hc5898": true}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && path != root && (skip[d.Name()] || strings.HasPrefix(d.Name(), ".")) {
			return filepath.SkipDir
		}
		if !d.IsDir() && d.Name() == "go.mod" && filepath.Dir(path) != root {
			mods = append(mods, filepath.Dir(path))
		}
		return nil
	})
	sort.Strings(mods)
	return mods, err
}

func stdlibOnly(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	found := false
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		found = true
		f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(dir, e.Name()), nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, imp := range f.Imports {
			p, _ := strconv.Unquote(imp.Path.Value)
			if strings.Contains(strings.Split(p, "/")[0], ".") {
				return fmt.Errorf("facade imports non-stdlib package %q", p)
			}
		}
	}
	if !found {
		return errors.New("nested tools module has no stdlib-only facade .go file in its parent")
	}
	return nil
}

func rootRequires(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	got := map[string]string{}
	in := false
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(strings.SplitN(s.Text(), "//", 2)[0])
		if line == "require (" {
			in = true
			continue
		}
		if in && line == ")" {
			in = false
			continue
		}
		if strings.HasPrefix(line, "require ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "require "))
		}
		if !in && !strings.Contains(s.Text(), "require ") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			got[fields[0]] = fields[1]
		}
	}
	return got, s.Err()
}
func lines(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if x := strings.TrimSpace(l); x != "" {
			out = append(out, x)
		}
	}
	return out, nil
}
func mustRel(root, path string) string { r, _ := filepath.Rel(root, path); return r }
