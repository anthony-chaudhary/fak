// Package boundarylint is a small, extensible policy engine for "boundary tells":
// source patterns where the code makes a claim about the outside world (the OS, the
// network, the clock, a peer process) without the check that would make the claim
// true. It is the DOS witness idea turned on the codebase itself — "the kernel is the
// part that doesn't believe the agents," here applied to the author: a green build is a
// self-report, and these tells are where a green build hides a latent boundary failure.
//
// pathlint (UNEXPANDED_USER_PATH) and urllint (UNVERIFIED_EXTERNAL_URL) were the first
// two such witnesses, each in its own package. This package is the registry they
// generalize into: each tell is a Rule with a closed-vocabulary Code, the scanner runs
// them in a single pass, and TestBoundaryPolicy enforces the whole family at once. See
// catalog.go for the full set of tells — enforced and proposed.
//
// A finding can be suppressed in place with a line comment:
//
//	resp, err := http.Get(url) //boundarylint:ignore MISSING_HTTP_TIMEOUT one-shot localhost probe
//
// as a trailing comment on the offending line. Suppressions are deliberately visible
// and greppable, so an exception is a recorded decision, not a silent gap.
package boundarylint

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Finding is one boundary tell at one source location.
type Finding struct {
	Code   string // closed-vocabulary reason, e.g. "MISSING_HTTP_TIMEOUT"
	File   string // repo-relative, slash-separated
	Line   int
	Detail string // what was found and how to resolve it
}

// String renders the finding as "file:line: detail (CODE)".
func (f Finding) String() string {
	return fmt.Sprintf("%s:%d: %s (%s)", f.File, f.Line, f.Detail, f.Code)
}

// Rule is one boundary tell. Check inspects a single parsed file and returns findings;
// the scanner fills in File and applies suppression comments.
type Rule interface {
	Code() string
	Check(fset *token.FileSet, file *ast.File, relPath string) []Finding
}

// DefaultRules is the policy enforced by TestBoundaryPolicy. Add a tell by appending a
// Rule here and a catalog.go entry.
func DefaultRules() []Rule {
	return []Rule{
		MissingHTTPTimeout{},
	}
}

var ignoreRe = regexp.MustCompile(`//\s*boundarylint:ignore\s+([A-Z0-9_,\s]+)`)

// Scan walks each root, parses every non-test Go file once, runs rules, and drops any
// finding suppressed by a //boundarylint:ignore comment on its line or the line above.
func Scan(roots []string, rules []Rule) ([]Finding, error) {
	var findings []Finding
	for _, root := range roots {
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if name := d.Name(); name == ".git" || name == "vendor" || name == "node_modules" || name == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			fset := token.NewFileSet()
			file, perr := parser.ParseFile(fset, path, nil, parser.ParseComments)
			if perr != nil {
				return nil // not parseable as a standalone file; the compiler owns that error
			}
			rel := filepath.ToSlash(path)
			ignores := collectIgnores(fset, file)
			for _, r := range rules {
				for _, f := range r.Check(fset, file, rel) {
					if suppressed(ignores, f.Line, f.Code) {
						continue
					}
					findings = append(findings, f)
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].File != findings[j].File {
			return findings[i].File < findings[j].File
		}
		if findings[i].Line != findings[j].Line {
			return findings[i].Line < findings[j].Line
		}
		return findings[i].Code < findings[j].Code
	})
	return findings, nil
}

// collectIgnores maps each source line carrying a //boundarylint:ignore to the set of
// codes it suppresses.
func collectIgnores(fset *token.FileSet, file *ast.File) map[int]map[string]bool {
	out := map[int]map[string]bool{}
	for _, cg := range file.Comments {
		for _, c := range cg.List {
			m := ignoreRe.FindStringSubmatch(c.Text)
			if m == nil {
				continue
			}
			line := fset.Position(c.Pos()).Line
			if out[line] == nil {
				out[line] = map[string]bool{}
			}
			for _, code := range strings.FieldsFunc(m[1], func(r rune) bool { return r == ',' || r == ' ' || r == '\t' }) {
				out[line][code] = true
			}
		}
	}
	return out
}

// suppressed reports whether a trailing ignore for code sits on the finding's line.
// Same-line only: a trailing comment is unambiguous, whereas a standalone comment line
// would also suppress the statement below it and silently mask an adjacent finding.
func suppressed(ignores map[int]map[string]bool, line int, code string) bool {
	return ignores[line][code]
}
