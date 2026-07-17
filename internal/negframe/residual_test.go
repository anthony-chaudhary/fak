package negframe

import "testing"

func TestResidual(t *testing.T) {
	residual, ok := ResolveResidual("Do not forget to push after the commit.")
	if !ok {
		t.Fatal("mechanical negation did not produce a residual")
	}
	if residual.Positive != "remember to push after the commit." || residual.Applied != 1 {
		t.Fatalf("residual=%+v", residual)
	}

	for _, input := range []string{
		"Push after the commit.",
		"Do not deploy unless the migration is complete.",
	} {
		if residual, ok := ResolveResidual(input); ok {
			t.Fatalf("unsafe residual for %q: %+v", input, residual)
		}
	}
}
