package astquery

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const fixtureSource = `package fixture

import (
	"errors"
	"fmt"
)

// return nil, err (in single line comment - should be ignored)
// return nil, $ERR (metavar in comment - should be ignored)
/*
   return nil, err
   return nil, customErr
   multiline comment mentioning pattern - should be ignored
*/

func Foo() (any, error) {
	_ = "return nil, err"  // string literal mentioning code - ignored
	_ = "return nil, $ERR" // string literal mentioning metavar - ignored

	err := errors.New("boom")
	if err != nil {
		return nil, err // line 22: true match 1
	}
	return "ok", nil // non-matching return
}

func Bar() (any, error) {
	customErr := errors.New("custom")
	return nil, customErr // line 29: true match 2
}

func Baz() (any, error) {
	// Comment inside function: return nil, err
	msg := fmt.Sprintf("cannot return nil, %s", "err")
	_ = msg
	return nil, fmt.Errorf("wrapped: %s", "failed") // line 36: true match 3
}
`

func TestSearchToolReturnNilErr(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "fixture.go")
	if err := os.WriteFile(filePath, []byte(fixtureSource), 0644); err != nil {
		t.Fatalf("failed to write fixture: %v", err)
	}

	result, err := SearchTool(dir, "return nil, $ERR", nil, 0)
	if err != nil {
		t.Fatalf("SearchTool failed: %v", err)
	}

	if result.Count != 3 {
		t.Fatalf("expected 3 matches, got %d", result.Count)
	}
	if len(result.Matches) != 3 {
		t.Fatalf("expected 3 match items, got %d", len(result.Matches))
	}

	// Verify Match 1: return nil, err
	m1 := result.Matches[0]
	if m1.File != "fixture.go" {
		t.Errorf("expected m1.File 'fixture.go', got %q", m1.File)
	}
	if m1.Line != 22 {
		t.Errorf("expected m1.Line 22, got %d", m1.Line)
	}
	if m1.Column != 3 {
		t.Errorf("expected m1.Column 3, got %d", m1.Column)
	}
	if m1.Offset <= 0 {
		t.Errorf("expected positive m1.Offset, got %d", m1.Offset)
	}
	if m1.Pos.Line != 22 || m1.Pos.Column != 3 {
		t.Errorf("expected m1.Pos line 22 col 3, got %v", m1.Pos)
	}
	if m1.Text != "return nil, err" {
		t.Errorf("expected m1.Text 'return nil, err', got %q", m1.Text)
	}
	if !strings.Contains(m1.SourceLine, "return nil, err") {
		t.Errorf("expected m1.SourceLine to contain 'return nil, err', got %q", m1.SourceLine)
	}
	if got := m1.Bindings["$ERR"]; got != "err" {
		t.Errorf("expected m1.Bindings[\"$ERR\"] = \"err\", got %q", got)
	}
	if got := m1.Bindings["ERR"]; got != "err" {
		t.Errorf("expected m1.Bindings[\"ERR\"] = \"err\", got %q", got)
	}

	// Verify Match 2: return nil, customErr
	m2 := result.Matches[1]
	if m2.Line != 29 {
		t.Errorf("expected m2.Line 29, got %d", m2.Line)
	}
	if m2.Text != "return nil, customErr" {
		t.Errorf("expected m2.Text 'return nil, customErr', got %q", m2.Text)
	}
	if got := m2.Bindings["$ERR"]; got != "customErr" {
		t.Errorf("expected m2.Bindings[\"$ERR\"] = \"customErr\", got %q", got)
	}
	if !strings.Contains(m2.SourceLine, "return nil, customErr") {
		t.Errorf("expected m2.SourceLine to contain 'return nil, customErr', got %q", m2.SourceLine)
	}

	// Verify Match 3: return nil, fmt.Errorf(...)
	m3 := result.Matches[2]
	if m3.Line != 36 {
		t.Errorf("expected m3.Line 36, got %d", m3.Line)
	}
	wantBinding := `fmt.Errorf("wrapped: %s", "failed")`
	if got := m3.Bindings["$ERR"]; got != wantBinding {
		t.Errorf("expected m3.Bindings[\"$ERR\"] = %q, got %q", wantBinding, got)
	}

	// Verify comments and string literals were ignored
	for _, m := range result.Matches {
		if m.Line < 20 && m.Line > 0 {
			t.Errorf("match found on comment/string line %d: %+v", m.Line, m)
		}
	}
}

func TestSearchToolSourceCommentsAndStringsIgnored(t *testing.T) {
	result, err := SearchToolSource(fixtureSource, "return nil, $ERR")
	if err != nil {
		t.Fatalf("SearchToolSource failed: %v", err)
	}

	if result.Count != 3 {
		t.Fatalf("expected 3 matches, got %d", result.Count)
	}

	// Ensure bindings correctly map metavariable
	if result.Matches[0].Bindings["$ERR"] != "err" {
		t.Errorf("expected $ERR = err, got %q", result.Matches[0].Bindings["$ERR"])
	}
	if result.Matches[1].Bindings["$ERR"] != "customErr" {
		t.Errorf("expected $ERR = customErr, got %q", result.Matches[1].Bindings["$ERR"])
	}

	// Ensure no comments or string literal lines are matched
	for _, m := range result.Matches {
		if strings.HasPrefix(strings.TrimSpace(m.SourceLine), "//") ||
			strings.HasPrefix(strings.TrimSpace(m.SourceLine), "/*") ||
			strings.Contains(m.SourceLine, `_ = "return nil`) {
			t.Errorf("comment or string literal was matched: line %d: %q", m.Line, m.SourceLine)
		}
	}
}

func TestSearchToolPathsAndMultiFile(t *testing.T) {
	dir := t.TempDir()

	pkgA := filepath.Join(dir, "pkg", "a")
	pkgB := filepath.Join(dir, "pkg", "b")
	if err := os.MkdirAll(pkgA, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(pkgB, 0755); err != nil {
		t.Fatal(err)
	}

	srcA := "package a\nfunc A() (any, error) { return nil, errA }\n"
	srcB := "package b\nfunc B() (any, error) { return nil, errB }\n"
	srcRoot := "package root\nfunc R() (any, error) { return nil, errRoot }\n"

	if err := os.WriteFile(filepath.Join(pkgA, "a.go"), []byte(srcA), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgB, "b.go"), []byte(srcB), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "root.go"), []byte(srcRoot), 0644); err != nil {
		t.Fatal(err)
	}

	// 1. Search entire workspace
	allRes, err := SearchTool(dir, "return nil, $ERR", nil, 0)
	if err != nil {
		t.Fatalf("all workspace search: %v", err)
	}
	if allRes.Count != 3 {
		t.Fatalf("expected 3 matches across workspace, got %d", allRes.Count)
	}

	// 2. Search specific sub-path (pkg/a)
	subRes, err := SearchTool(dir, "return nil, $ERR", []string{"pkg/a"}, 0)
	if err != nil {
		t.Fatalf("pkg/a search: %v", err)
	}
	if subRes.Count != 1 {
		t.Fatalf("expected 1 match in pkg/a, got %d", subRes.Count)
	}
	if subRes.Matches[0].File != "pkg/a/a.go" {
		t.Errorf("expected file 'pkg/a/a.go', got %q", subRes.Matches[0].File)
	}
	if subRes.Matches[0].Bindings["$ERR"] != "errA" {
		t.Errorf("expected $ERR = errA, got %q", subRes.Matches[0].Bindings["$ERR"])
	}

	// 3. Search specific file (root.go)
	rootRes, err := SearchTool(dir, "return nil, $ERR", []string{"root.go"}, 0)
	if err != nil {
		t.Fatalf("root.go search: %v", err)
	}
	if rootRes.Count != 1 {
		t.Fatalf("expected 1 match in root.go, got %d", rootRes.Count)
	}
	if rootRes.Matches[0].File != "root.go" {
		t.Errorf("expected file 'root.go', got %q", rootRes.Matches[0].File)
	}
	if rootRes.Matches[0].Bindings["$ERR"] != "errRoot" {
		t.Errorf("expected $ERR = errRoot, got %q", rootRes.Matches[0].Bindings["$ERR"])
	}
}

func TestSearchToolMaxMatchesTruncation(t *testing.T) {
	dir := t.TempDir()
	src := `package p
func f() (any, error) {
	if true { return nil, e1 }
	if true { return nil, e2 }
	if true { return nil, e3 }
	if true { return nil, e4 }
	return nil, e5
}
`
	if err := os.WriteFile(filepath.Join(dir, "test.go"), []byte(src), 0644); err != nil {
		t.Fatal(err)
	}

	// With maxMatches = 2
	limited, err := SearchTool(dir, "return nil, $ERR", nil, 2)
	if err != nil {
		t.Fatalf("limited search: %v", err)
	}
	if len(limited.Matches) != 2 || limited.Count != 2 {
		t.Fatalf("expected 2 matches, got %d", len(limited.Matches))
	}
	if !limited.Truncated {
		t.Errorf("expected Truncated to be true")
	}

	// With maxMatches = 0 (unlimited)
	unlimited, err := SearchTool(dir, "return nil, $ERR", nil, 0)
	if err != nil {
		t.Fatalf("unlimited search: %v", err)
	}
	if len(unlimited.Matches) != 5 || unlimited.Count != 5 {
		t.Fatalf("expected 5 matches, got %d", len(unlimited.Matches))
	}
	if unlimited.Truncated {
		t.Errorf("expected Truncated to be false")
	}
}

func TestSearchToolParamsAndFormatting(t *testing.T) {
	dir := t.TempDir()
	src := "package p\nfunc f() (any, error) { return nil, err }\n"
	if err := os.WriteFile(filepath.Join(dir, "f.go"), []byte(src), 0644); err != nil {
		t.Fatal(err)
	}

	params := SearchToolParams{
		Workspace:  dir,
		Pattern:    "return nil, $ERR",
		MaxMatches: 10,
	}
	res, err := params.Execute()
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if res.Count != 1 {
		t.Fatalf("expected 1 match, got %d", res.Count)
	}
	str := res.String()
	if !strings.Contains(str, "return nil, err") || !strings.Contains(str, "f.go:2:25") {
		t.Errorf("unexpected String output: %q", str)
	}
}

func TestSearchToolInvalidInputs(t *testing.T) {
	// Empty pattern
	if _, err := SearchTool(".", "", nil, 0); err == nil {
		t.Error("expected error for empty pattern")
	}

	// Syntax error pattern
	if _, err := SearchTool(".", "func(", nil, 0); err == nil {
		t.Error("expected error for unparseable pattern")
	}

	// Non-existent path
	if _, err := SearchTool(".", "return nil, $ERR", []string{"nonexistent_path_xyz.go"}, 0); err == nil {
		t.Error("expected error for nonexistent file path")
	}
}

func TestSearchToolMetavariableUnification(t *testing.T) {
	src := `package p

func testUnify() {
	check(a, a) // match 1: X = a
	check(a, b) // rejected: X = a != b
	_ = "check(c, c)" // rejected: string literal
	// check(d, d)   // rejected: comment
	check(foo.Bar(), foo.Bar()) // match 2: X = foo.Bar()
}
`
	res, err := SearchToolSource(src, `check($X, $X)`)
	if err != nil {
		t.Fatalf("SearchToolSource failed: %v", err)
	}
	if res.Count != 2 {
		t.Fatalf("expected 2 matches, got %d", res.Count)
	}
	if len(res.Matches) != 2 {
		t.Fatalf("expected 2 match objects, got %d", len(res.Matches))
	}

	m1 := res.Matches[0]
	if m1.Bindings["$X"] != "a" || m1.Bindings["X"] != "a" {
		t.Errorf("expected m1 binding X=a, got %+v", m1.Bindings)
	}
	if m1.Text != "check(a, a)" {
		t.Errorf("expected m1 text 'check(a, a)', got %q", m1.Text)
	}
	if m1.Line != 4 {
		t.Errorf("expected m1 line 4, got %d", m1.Line)
	}

	m2 := res.Matches[1]
	if m2.Bindings["$X"] != "foo.Bar()" || m2.Bindings["X"] != "foo.Bar()" {
		t.Errorf("expected m2 binding X=foo.Bar(), got %+v", m2.Bindings)
	}
	if m2.Text != "check(foo.Bar(), foo.Bar())" {
		t.Errorf("expected m2 text 'check(foo.Bar(), foo.Bar())', got %q", m2.Text)
	}
	if m2.Line != 8 {
		t.Errorf("expected m2 line 8, got %d", m2.Line)
	}
}
