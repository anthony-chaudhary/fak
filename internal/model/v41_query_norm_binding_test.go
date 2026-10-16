package model

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestV41QueryNormReplacementStripsAuthority(t *testing.T) {
	var absent *v41QueryNormBinding
	if absent.normFunc() != nil {
		t.Fatal("nil callback")
	}
	called := false
	b := &v41QueryNormBinding{ordinary: func(int, []float32) ([]float32, error) { called = true; return nil, nil }}
	if b.owner != nil || b.backend != nil || b.publication != nil {
		t.Fatal("replacement retained pool authority")
	}
	b.normFunc()(0, nil)
	if !called {
		t.Fatal("ordinary callback lost")
	}
	if handled, err := b.panel(nil, nil, 0, nil, 64, 1e-5, nil); handled || err != nil {
		t.Fatal("ordinary replacement selected")
	}
}
