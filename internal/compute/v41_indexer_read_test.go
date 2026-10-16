package compute

import "testing"

type indexReadLegacy struct {
	Backend
	values  []float32
	calls   int
	failure any
}

func (b *indexReadLegacy) Read(Tensor) []float32 {
	b.calls++
	if b.failure != nil {
		panic(b.failure)
	}
	return b.values
}

type indexReadOwned struct{ indexReadLegacy }

func (b *indexReadOwned) v41ReadIndexerOwned(original Backend, t Tensor) ([]float32, bool) {
	if original != b {
		return original.Read(t), false
	}
	return b.Read(t), true
}

type indexReadWrapper struct {
	*indexReadOwned
	replacement   []float32
	overrideCalls int
}

func (b *indexReadWrapper) Read(Tensor) []float32 { b.overrideCalls++; return b.replacement }

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexerReadOwnership(t *testing.T) {
	legacy := &indexReadLegacy{values: []float32{1, 2}}
	got, ok := ReadV41IndexerScores(legacy, Tensor{}, 2)
	if !ok || legacy.calls != 1 || &got[0] == &legacy.values[0] {
		t.Fatal("legacy ownership")
	}
	legacy.values[0] = 9
	if got[0] != 1 {
		t.Fatal("legacy alias")
	}
	owned := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: []float32{3, 4}}}
	got, ok = ReadV41IndexerScores(owned, Tensor{}, 2)
	if !ok || owned.calls != 1 || &got[0] != &owned.values[0] {
		t.Fatal("owned transfer")
	}
	wrapped := &indexReadWrapper{indexReadOwned: owned, replacement: []float32{5, 6}}
	got, ok = ReadV41IndexerScores(wrapped, Tensor{}, 2)
	if !ok || wrapped.overrideCalls != 1 || owned.calls != 1 || &got[0] == &wrapped.replacement[0] || got[0] != 5 {
		t.Fatal("promoted method bypass")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexerReadMalformedAndPanic(t *testing.T) {
	b := &indexReadLegacy{values: []float32{1, 2}}
	if got, ok := ReadV41IndexerScores(b, Tensor{}, 1); ok || got != nil || b.calls != 1 {
		t.Fatal("shape")
	}
	sentinel := new(int)
	b.failure = sentinel
	defer func() {
		if recover() != sentinel || b.calls != 2 {
			t.Fatal("panic identity/calls")
		}
	}()
	ReadV41IndexerScores(b, Tensor{}, 2)
}
