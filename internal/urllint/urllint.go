// Package urllint is a static witness for the network external-boundary claim's twin
// of pathlint: that no Go source hardcodes a model/tokenizer DOWNLOAD url outside the
// one audited chokepoint that derives and (via a network-gated test) verifies them.
//
// The original bug shipped a download url that 404'd because the filename convention
// was wrong (dash-vs-dot) and the repo was hardcoded. The fix routed every url through
// cmd/simpledemo's modelDownload(), which the reachability test HEAD-checks. A second
// command that pastes its own `huggingface.co/.../resolve/...` literal would silently
// escape that verification. This witness refuses such a literal with the closed
// vocabulary reason UNVERIFIED_EXTERNAL_URL, forcing all download urls through the
// chokepoint.
package urllint

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// ReasonUnverifiedExternalURL is the closed-vocabulary refusal code for a hardcoded
// download url found outside the audited chokepoint.
const ReasonUnverifiedExternalURL = "UNVERIFIED_EXTERNAL_URL"

// downloadURLRe matches a model/tokenizer download url (a HuggingFace or mirror host
// with a /resolve/ path). It deliberately ignores plain browse/model-page links
// (huggingface.co/models?search=...) which carry no /resolve/ and download nothing.
var downloadURLRe = regexp.MustCompile(`(huggingface\.co|hf-mirror\.com)/[^"]*resolve/`)

// Offense is one hardcoded download url literal outside the allowlist.
type Offense struct {
	File    string // repo-relative, slash-separated
	Line    int
	Snippet string
}

// String renders the offense as a "file:line: hardcoded download url ... must go
// through the audited builder (UNVERIFIED_EXTERNAL_URL)" diagnostic line.
func (o Offense) String() string {
	return fmt.Sprintf("%s:%d: hardcoded download url %q must go through the audited builder (%s)",
		o.File, o.Line, o.Snippet, ReasonUnverifiedExternalURL)
}

// ScanForDownloadURLs walks repoRoot's non-test Go sources and returns every download
// url string literal whose file is not in allow (repo-relative, slash-separated paths).
func ScanForDownloadURLs(repoRoot string, allow map[string]bool) ([]Offense, error) {
	var offenses []Offense
	err := filepath.WalkDir(repoRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip VCS, vendored, and dependency trees.
			if name := d.Name(); name == ".git" || name == "vendor" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		rel, rerr := filepath.Rel(repoRoot, path)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if allow[rel] {
			return nil
		}
		off, perr := scanFile(path, rel)
		if perr != nil {
			return perr
		}
		offenses = append(offenses, off...)
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(offenses, func(i, j int) bool {
		if offenses[i].File != offenses[j].File {
			return offenses[i].File < offenses[j].File
		}
		return offenses[i].Line < offenses[j].Line
	})
	return offenses, nil
}

func scanFile(path, rel string) ([]Offense, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return nil, nil // not parseable as standalone Go (e.g. a //go:build cuda cgo file); skip
	}
	var offenses []Offense
	ast.Inspect(f, func(n ast.Node) bool {
		lit, ok := n.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return true
		}
		val, uerr := strconv.Unquote(lit.Value)
		if uerr != nil {
			return true
		}
		if downloadURLRe.MatchString(val) {
			offenses = append(offenses, Offense{File: rel, Line: fset.Position(lit.Pos()).Line, Snippet: val})
		}
		return true
	})
	return offenses, nil
}
