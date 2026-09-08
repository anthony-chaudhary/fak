package astquery

import (
	"os"
	"path/filepath"
	"testing"
)

const searchTestFixture = `package testpkg

import (
	"errors"
	"fmt"
)

// return nil, $ERR (comment mentioning return pattern)
// if $X != nil { return $X } (comment mentioning if pattern)
/*
   return nil, err
   if err != nil { return err }
   multiline block comments must be ignored
*/

func SampleFunc() (any, error) {
	_ = "return nil, err"              // string literal decoy
	_ = "if err != nil { return err }"  // string literal decoy

	err := errors.New("sample error")
	if err != nil {
		return nil, err // line 23: return match 1
	}

	customErr := fmt.Errorf("custom error")
	if customErr != nil {
		return customErr // line 28: if match 1
	}

	wrappedErr := fmt.Errorf("wrapped")
	if wrappedErr != nil {
		return nil, wrappedErr // line 33: return match 2
	}

	// Inconsistent condition vs return: should NOT match if $X != nil { return $X }
	if err != nil {
		return customErr
	}

	return "success", nil
}
`

func TestSearchReturnNilErrPattern(t *testing.T) {
	res, err := StructuralSearch(searchTestFixture, "return nil, $ERR")
	if err != nil {
		t.Fatalf("StructuralSearch failed: %v", err)
	}

	if res.Count != 2 {
		t.Fatalf("expected 2 matches, got %d", res.Count)
	}

	// Match 1: line 22 "return nil, err"
	m1 := res.Matches[0]
	if m1.Line != 22 {
		t.Errorf("match 1 line = %d, want 22", m1.Line)
	}
	if m1.Text != "return nil, err" {
		t.Errorf("match 1 text = %q, want 'return nil, err'", m1.Text)
	}
	if got := m1.Bindings["$ERR"]; got != "err" {
		t.Errorf("match 1 bindings[\"$ERR\"] = %q, want 'err'", got)
	}
	if got := m1.Bindings["ERR"]; got != "err" {
		t.Errorf("match 1 bindings[\"ERR\"] = %q, want 'err'", got)
	}

	// Match 2: line 32 "return nil, wrappedErr"
	m2 := res.Matches[1]
	if m2.Line != 32 {
		t.Errorf("match 2 line = %d, want 32", m2.Line)
	}
	if m2.Text != "return nil, wrappedErr" {
		t.Errorf("match 2 text = %q, want 'return nil, wrappedErr'", m2.Text)
	}
	if got := m2.Bindings["$ERR"]; got != "wrappedErr" {
		t.Errorf("match 2 bindings[\"$ERR\"] = %q, want 'wrappedErr'", got)
	}

	// Ensure comment and string literal lines were not matched
	for _, m := range res.Matches {
		if m.Line < 20 {
			t.Errorf("unexpected match on line %d (comments/strings region): %+v", m.Line, m)
		}
	}
}

func TestSearchIfErrReturnErrPattern(t *testing.T) {
	res, err := StructuralSearch(searchTestFixture, "if $X != nil { return $X }")
	if err != nil {
		t.Fatalf("StructuralSearch failed: %v", err)
	}

	// Should match only line 27-29: "if customErr != nil { return customErr }"
	// Should NOT match "if err != nil { return customErr }" because $X != $X
	if res.Count != 1 {
		t.Fatalf("expected 1 match, got %d", res.Count)
	}

	m := res.Matches[0]
	if m.Line != 26 {
		t.Errorf("match line = %d, want 26", m.Line)
	}
	if got := m.Bindings["$X"]; got != "customErr" {
		t.Errorf("bindings[\"$X\"] = %q, want 'customErr'", got)
	}
	if got := m.Bindings["X"]; got != "customErr" {
		t.Errorf("bindings[\"X\"] = %q, want 'customErr'", got)
	}
}

func TestSearchCommentsAndStringsImmunity(t *testing.T) {
	src := `package p
// return nil, $ERR
// return nil, err
/*
return nil, err
*/
func F() (any, error) {
	_ = "return nil, err"
	_ = "return nil, $ERR"
	return "ok", nil
}
`
	res, err := StructuralSearch(src, "return nil, $ERR")
	if err != nil {
		t.Fatalf("StructuralSearch failed: %v", err)
	}
	if res.Count != 0 {
		t.Fatalf("expected 0 matches for comments/strings, got %d: %+v", res.Count, res.Matches)
	}
}

func TestSearchPatternHelper(t *testing.T) {
	matches, err := SearchPattern(searchTestFixture, "return nil, $ERR")
	if err != nil {
		t.Fatalf("SearchPattern failed: %v", err)
	}
	if len(matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(matches))
	}
	if matches[0].Bindings["$ERR"] != "err" {
		t.Errorf("expected $ERR = err, got %q", matches[0].Bindings["$ERR"])
	}
}

func TestSearchWorkspaceMultiFile(t *testing.T) {
	dir := t.TempDir()
	f1 := filepath.Join(dir, "f1.go")
	f2 := filepath.Join(dir, "f2.go")

	src1 := "package p\nfunc A() (any, error) { return nil, errOne }\n"
	src2 := "package p\nfunc B() (any, error) { return nil, errTwo }\n"

	if err := os.WriteFile(f1, []byte(src1), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f2, []byte(src2), 0644); err != nil {
		t.Fatal(err)
	}

	res, err := SearchWorkspace(dir, "return nil, $ERR", nil, 0)
	if err != nil {
		t.Fatalf("SearchWorkspace failed: %v", err)
	}
	if res.Count != 2 {
		t.Fatalf("expected 2 matches in workspace, got %d", res.Count)
	}

	files := make(map[string]string)
	for _, m := range res.Matches {
		files[m.File] = m.Bindings["$ERR"]
	}
	if files["f1.go"] != "errOne" {
		t.Errorf("expected f1.go to bind errOne, got %q", files["f1.go"])
	}
	if files["f2.go"] != "errTwo" {
		t.Errorf("expected f2.go to bind errTwo, got %q", files["f2.go"])
	}
}
