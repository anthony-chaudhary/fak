package boundarylint

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/pathlint"
	"github.com/anthony-chaudhary/fak/internal/urllint"
)

// checkSrc parses src and runs one rule over it (no suppression).
func checkSrc(t *testing.T, rule Rule, src string) []Finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "synthetic.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return rule.Check(fset, f, "synthetic.go")
}

// TestMissingHTTPTimeoutRule verifies the witness (verify the verifier): the no-timeout
// helpers and DefaultClient are flagged; a configured client is not.
func TestMissingHTTPTimeoutRule(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want int
	}{
		{"http.Get flagged", `package p
import "net/http"
func f() { http.Get("http://x") }`, 1},
		{"http.DefaultClient flagged", `package p
import "net/http"
func f() { _ = http.DefaultClient }`, 1},
		{"http.Post and Head", `package p
import "net/http"
func f() { http.Post("u","",nil); http.Head("u") }`, 2},
		{"configured client clean", `package p
import "net/http"
func f() { c := &http.Client{}; c.Get("u") }`, 0},
		{"method named Get on other type clean", `package p
type T struct{}
func (T) Get(s string) {}
func f() { var t T; t.Get("u") }`, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := checkSrc(t, MissingHTTPTimeout{}, tc.src)
			if len(got) != tc.want {
				t.Fatalf("got %d findings %v, want %d", len(got), got, tc.want)
			}
		})
	}
}

// TestIgnoreComment verifies that a //boundarylint:ignore on the line (or directly
// above) suppresses exactly that code.
func TestIgnoreComment(t *testing.T) {
	fset := token.NewFileSet()
	src := `package p
import "net/http"
func f() {
	http.Get("u") //boundarylint:ignore MISSING_HTTP_TIMEOUT probe
	http.Get("v")
}`
	f, err := parser.ParseFile(fset, "s.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	ignores := collectIgnores(fset, f)
	var kept []Finding
	for _, fd := range (MissingHTTPTimeout{}).Check(fset, f, "s.go") {
		if !suppressed(ignores, fd.Line, fd.Code) {
			kept = append(kept, fd)
		}
	}
	if len(kept) != 1 {
		t.Fatalf("want 1 unsuppressed finding (the un-annotated Get), got %d: %v", len(kept), kept)
	}
}

// TestCatalogCoversEnforcedRules keeps the documented policy honest: every rule in
// DefaultRules must have a catalog entry marked enforced.
func TestCatalogCoversEnforcedRules(t *testing.T) {
	byCode := map[string]CatalogEntry{}
	for _, e := range Catalog {
		byCode[e.Code] = e
	}
	for _, r := range DefaultRules() {
		e, ok := byCode[r.Code()]
		if !ok {
			t.Errorf("rule %s has no catalog entry", r.Code())
			continue
		}
		if e.Status != StatusEnforced {
			t.Errorf("rule %s is in DefaultRules but cataloged as %q, want enforced", r.Code(), e.Status)
		}
	}
}

// TestBoundaryPolicy is the single live enforcement entrypoint for the whole family:
// the registry rules here PLUS the two sibling witnesses (pathlint, urllint). Any tell
// anywhere in cmd/ or internal/ fails this one test with the aggregated list.
func TestBoundaryPolicy(t *testing.T) {
	root := repoRoot(t)

	findings, err := Scan([]string{filepath.Join(root, "cmd"), filepath.Join(root, "internal")}, DefaultRules())
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	for _, f := range findings {
		t.Errorf("boundary tell: %s", f)
	}

	pathOff, err := pathlint.ScanCmdTree(root)
	if err != nil {
		t.Fatalf("pathlint: %v", err)
	}
	for _, o := range pathOff {
		t.Errorf("boundary tell: %s", o)
	}

	urlOff, err := urllint.ScanForDownloadURLs(root, map[string]bool{"cmd/simpledemo/main.go": true})
	if err != nil {
		t.Fatalf("urllint: %v", err)
	}
	for _, o := range urlOff {
		t.Errorf("boundary tell: %s", o)
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
