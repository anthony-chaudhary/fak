package astquery

import (
	"os"
	"path/filepath"
	"testing"
)

const src = `package p

func f() {
	_ = a == a
	_ = c == d
	g(x)
	h(y, y)
	h(m, n)
	w.Close()
}
`

func TestMetavarBackReference(t *testing.T) {
	// $X == $X binds both sides to the same hole, so it matches a == a but NOT
	// c == d — the property regex and trigram search cannot express.
	hits, err := Search(src, `$X == $X`)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("$X == $X matched %d exprs, want 1 (a == a only): %+v", len(hits), hits)
	}
	if got := hits[0].Bindings["X"]; got != "a" {
		t.Errorf("binding X = %q, want a", got)
	}
	if hits[0].Text != "a == a" {
		t.Errorf("matched text = %q, want %q", hits[0].Text, "a == a")
	}
}

func TestCallShapeBinding(t *testing.T) {
	hits, err := Search(src, `g($X)`)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].Bindings["X"] != "x" {
		t.Fatalf("g($X) = %+v, want one match binding X=x", hits)
	}

	// Type conversion call with ArrayType: []byte(s)
	byteSrc := `package p
func f() {
	_ = []byte(s)
}
`
	byteHits, err := Search(byteSrc, `[]byte($S)`)
	if err != nil {
		t.Fatalf("Search []byte: %v", err)
	}
	if len(byteHits) != 1 || byteHits[0].Bindings["S"] != "s" {
		t.Fatalf("[]byte($S) = %+v, want one match binding S=s", byteHits)
	}
}

func TestRepeatedArgConsistency(t *testing.T) {
	// h($X, $X) matches h(y, y) but not h(m, n).
	hits, err := Search(src, `h($X, $X)`)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("h($X,$X) matched %d, want 1 (y,y only): %+v", len(hits), hits)
	}
	if hits[0].Bindings["X"] != "y" {
		t.Errorf("binding X = %q, want y", hits[0].Bindings["X"])
	}
}

func TestWildcardSelector(t *testing.T) {
	// $_ is the anonymous wildcard: match any receiver .Close().
	hits, err := Search(src, `$_.Close()`)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("$_.Close() matched %d, want 1: %+v", len(hits), hits)
	}
	if len(hits[0].Bindings) != 0 {
		t.Errorf("wildcard bound something: %+v", hits[0].Bindings)
	}

	// A named receiver hole binds it.
	named, err := Search(src, `$R.Close()`)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(named) != 1 || named[0].Bindings["R"] != "w" {
		t.Fatalf("$R.Close() = %+v, want one match binding R=w", named)
	}

	// Metavariable in method/selector position:
	methodHits, err := Search(src, `$R.$M()`)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(methodHits) != 1 || methodHits[0].Bindings["M"] != "Close" {
		t.Fatalf("$R.$M() = %+v, want one match binding M=Close", methodHits)
	}
}

func TestBadPattern(t *testing.T) {
	if _, err := Search(src, `func(`); err == nil {
		t.Error("expected error for unparseable pattern")
	}
}

func TestStatementPattern(t *testing.T) {
	stmtSrc := `package p
func f() {
	if err != nil {
		return err
	}
	x := 1
	return nil
}
`
	hits, err := Search(stmtSrc, `if $X != nil { return $Y }`)
	if err != nil {
		t.Fatalf("Search statement: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected 1 match, got %d: %+v", len(hits), hits)
	}
	if hits[0].Bindings["X"] != "err" || hits[0].Bindings["Y"] != "err" {
		t.Errorf("unexpected bindings: %+v", hits[0].Bindings)
	}
}

func TestDeclarationPattern(t *testing.T) {
	declSrc := `package p

var globalVar int = 42

type MyInt int
`
	hits, err := Search(declSrc, `var $X int = 42`)
	if err != nil {
		t.Fatalf("Search decl: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected 1 match for top-level var, got %d: %+v", len(hits), hits)
	}
	hitsType, err := Search(declSrc, `type $T int`)
	if err != nil {
		t.Fatalf("Search type: %v", err)
	}
	if len(hitsType) != 1 {
		t.Fatalf("expected 1 match for type, got %d: %+v", len(hitsType), hitsType)
	}
	if hitsType[0].Bindings["T"] != "MyInt" {
		t.Errorf("unexpected bindings: %+v", hitsType[0].Bindings)
	}

	funcSrc := `package p

func Hello() string {
	return "world"
}
`
	funcHits, err := Search(funcSrc, `func $F() string`)
	if err != nil {
		t.Fatalf("Search func: %v", err)
	}
	if len(funcHits) != 1 || funcHits[0].Bindings["F"] != "Hello" {
		t.Fatalf("expected 1 match for func, got %d: %+v", len(funcHits), funcHits)
	}
}

func TestSearchFileAndDir(t *testing.T) {
	dir := t.TempDir()
	f1 := filepath.Join(dir, "f1.go")
	f2 := filepath.Join(dir, "f2.go")

	src1 := "package p\nfunc a() { w.Close() }\n"
	src2 := "package p\nfunc b() { w.Close() }\n"

	if err := os.WriteFile(f1, []byte(src1), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f2, []byte(src2), 0644); err != nil {
		t.Fatal(err)
	}

	fMatches, err := SearchFile(f1, `$_.Close()`)
	if err != nil {
		t.Fatalf("SearchFile: %v", err)
	}
	if len(fMatches) != 1 {
		t.Fatalf("SearchFile len=%d, want 1", len(fMatches))
	}

	dirMatches, err := SearchDir(dir, `$_.Close()`, 0)
	if err != nil {
		t.Fatalf("SearchDir: %v", err)
	}
	if len(dirMatches) != 2 {
		t.Fatalf("SearchDir len=%d, want 2", len(dirMatches))
	}

	limitedMatches, err := SearchDir(dir, `$_.Close()`, 1)
	if err != nil {
		t.Fatalf("SearchDir limit: %v", err)
	}
	if len(limitedMatches) != 1 {
		t.Fatalf("SearchDir limited len=%d, want 1", len(limitedMatches))
	}
}
