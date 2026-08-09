package testquality

import (
	"strings"
	"testing"
)

func TestAnalyzeMinimumFindingFamilies(t *testing.T) {
	src := []byte(`package p
import "testing"
func TestEmpty(t *testing.T) { x := 1; _ = x }
func TestSelf(t *testing.T) { got := 1; if got != got { t.Fatal("bad") } }
func TestErr(t *testing.T) { _, err := f(); t.Log("ran") }
func TestTable(t *testing.T) {
 tests := []struct{name string; want int}{{"x", 1}}
 for _, tc := range tests { t.Run(tc.name, func(t *testing.T) { if 1 != 1 { t.Fatal("bad") } }) }
}
func f()(int,error){ return 0,nil }
`)
	got, err := Analyze("p/x_test.go", src)
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for _, f := range got {
		have[f.Code] = true
	}
	for _, code := range Codes {
		if !have[code] {
			t.Errorf("missing finding %s; got %#v", code, got)
		}
	}
}

func TestFindingKeyStableUnderInsertionAndCounts(t *testing.T) {
	one := Finding{Code: CodeSelfComparison, File: "p/x_test.go", Func: "TestX", Line: 3}
	two := one
	two.Line = 99
	if one.Key() != two.Key() {
		t.Fatalf("line leaked into key: %q != %q", one.Key(), two.Key())
	}
	fresh, slack := NewFindings([]Finding{one, two}, Baseline{one.Key(): 1})
	if len(fresh) != 1 {
		t.Fatalf("fresh=%d want 1", len(fresh))
	}
	if len(slack) != 0 {
		t.Fatalf("unexpected slack=%v", slack)
	}
}

func TestParseBaselineRejectsMalformedRowWithLine(t *testing.T) {
	_, err := ParseBaseline([]byte("# ok\n\nTESTQ_NO_ASSERTION\tp/x_test.go\tTestX\t1\nbroken\n"))
	if err == nil || !strings.Contains(err.Error(), "line 4") {
		t.Fatalf("err=%v; want line 4 hard error", err)
	}
}

func TestFormatBaselineTightensAfterFix(t *testing.T) {
	f := Finding{Code: CodeNoAssertion, File: "p/x_test.go", Func: "TestX"}
	before := string(FormatBaseline([]Finding{f, f}))
	after := string(FormatBaseline([]Finding{f}))
	if !strings.Contains(before, "\t2\n") || !strings.Contains(after, "\t1\n") {
		t.Fatalf("counts not tightened:\n%s\n%s", before, after)
	}
}
