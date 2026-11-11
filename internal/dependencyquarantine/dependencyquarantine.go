// Package dependencyquarantine enforces the repository dependency budget and ensures
// external dependencies remain quarantined in isolated tools submodules.
//
// Enforcement is fail-closed and bounded: any unreviewed root go.mod require entry,
// root go.sum checksum line, or nested tool facade import of non-stdlib packages
// immediately yields structured violations and blocks repository verification.
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

// allowedRootRequires is the reviewed dependency budget (2026-11-08 review of the
// dual-repo workspace alignment + carried videogen fix): x/term v0.47.0 and
// x/sys v0.49.0 align the public module graph with the private x/tools v0.52.0
// pair (a3034f8c484, 2aab0c0567f); x/image v0.47.0 is the carried fix for
// tools/videogen's existing import (b3c4d0a1629); x/text v0.43.0 rides the same
// module-graph resolution as an indirect requirement. All four are
// golang.org/x extended-stdlib modules under the Go project's BSD-3-Clause.
var allowedRootRequires = map[string]string{
	"golang.org/x/image": "v0.47.0",
	"golang.org/x/sys":   "v0.49.0",
	"golang.org/x/term":  "v0.47.0",
	"golang.org/x/text":  "v0.43.0",
}

var allowedRootSum = map[string]bool{
	"golang.org/x/image v0.47.0 h1:za/8gVz++PPZqmZJBNSISICN5Rrj5zixP/8K++fgmqo=":        true,
	"golang.org/x/image v0.47.0/go.mod h1:IkN5o4A7vq55vXN8FWjjnhg9Rs2GOtffqsSi9mWAKYE=": true,
	"golang.org/x/sys v0.49.0 h1:XbzkgYJHdqh/8m2Uu0W/dQv8nktxx4BFHp1M0gROTXA=":          true,
	"golang.org/x/sys v0.49.0/go.mod h1:hNLxWAXmnKAxqDtdwIYC4bM9oQPEecfsnNMuSxOs3og=":   true,
	"golang.org/x/term v0.47.0 h1:cpyOwQkr2Nzxe2p6KnrTG7Ei2wtzjsuSjhpXXepZ/BM=":         true,
	"golang.org/x/term v0.47.0/go.mod h1:efAfAuG6RKABvg1ArGebRkvX6LXul14SM67o2e4mUCY=":  true,
	"golang.org/x/text v0.43.0 h1:1QivrlhwAsnMOcqgOdl9my7s9OyZl8O5/UGUTJVXoWc=":         true,
	"golang.org/x/text v0.43.0/go.mod h1:hbSIYA/amXcRXTFSZqh8tWHEoqRvvbKPJ2uz/KU+a+s=":  true,
}

// Violation describes an unreviewed dependency or facade policy violation.
type Violation struct{ Path, Reason string }

func (v Violation) Error() string { return v.Path + ": " + v.Reason }

// Check scans the repository root for root dependency drift and unreviewed nested tool facades.
// Unreadable or unparsable files return a non-nil error, and broken dependency policy rules
// return non-empty violations, so a partial scan never passes silently.
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
// Hidden directories, scratch spaces, and the repo root itself are skipped so traversal
// stays bounded to legitimate nested modules.
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
