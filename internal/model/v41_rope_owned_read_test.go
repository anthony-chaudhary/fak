package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41RoPEPairedReadNoPartialPublication(t *testing.T) {
	for _, fault := range []string{"short-query", "short-kv", "query-and-kv-nan", "kv-nan", "row-count", "verdict-count"} {
		b := newV41PairedRoPERecorder()
		b.groupFault = fault
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		q, k, e := s.v41TailRoPEFunc()(0, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 2, 3, 4}, []float32{1}, []float32{0}, 2, 4, 2)
		if e == nil || q != nil || k != nil {
			t.Fatalf("%s published partial rows", fault)
		}
		var closed *BackendForwardOperationError
		if !errors.As(e, &closed) {
			t.Fatal(e)
		}
		want := "paired readback"
		if fault == "short-query" || fault == "query-and-kv-nan" {
			want = "query readback"
		}
		if fault == "short-kv" || fault == "kv-nan" {
			want = "kv readback"
		}
		if closed.Stage != want {
			t.Fatalf("%s: %s", fault, closed.Stage)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41RoPEPairedLegacyRowsRemainIndependent(t *testing.T) {
	b := newV41PairedRoPERecorder()
	s := &Session{M: &Model{}, Backend: b}
	t.Cleanup(s.Close)
	f := s.v41TailRoPEFunc()
	q, k, e := f(0, []float32{math.Float32frombits(0x80000000), 2, 3, 4, 5, 6, 7, 8}, []float32{1, 2, 3, 4}, []float32{1}, []float32{0}, 2, 4, 2)
	if e != nil {
		t.Fatal(e)
	}
	wantQ, wantK := append([]float32(nil), q...), append([]float32(nil), k...)
	_, _, e = f(0, []float32{8, 7, 6, 5, 4, 3, 2, 1}, []float32{4, 3, 2, 1}, []float32{1}, []float32{0}, 2, 4, 2)
	if e != nil {
		t.Fatal(e)
	}
	v41TailRoPECheckBits(t, "earlier Q", q, wantQ)
	v41TailRoPECheckBits(t, "earlier K", k, wantK)
	if b.paired != 2 || b.readCount != 0 {
		t.Fatal("read route changed")
	}
}
