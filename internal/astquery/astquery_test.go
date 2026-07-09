package astquery

import "testing"

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
}

func TestBadPattern(t *testing.T) {
	if _, err := Search(src, `func(`); err == nil {
		t.Error("expected error for unparseable pattern")
	}
}
