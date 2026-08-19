package model

import (
	"bytes"
	"errors"
	"sync"
	"testing"
)

func TestLazyQ4KMaterializesExactRangeWithoutResidentCopy(t *testing.T) {
	cfg := Config{HiddenSize: 256}
	b := NewQuantBuilder(cfg, false)
	payload := make([]byte, q4kBlockBytes)
	for i := range payload {
		payload[i] = byte(i*17 + 3)
	}
	blob := append([]byte("prefix"), payload...)
	name := "model.layers.0.mlp.down_proj.weight"
	if err := b.AddLazyQ4K(name, []int{1, 256}, LazyQ4KRange{Reader: bytes.NewReader(blob), Offset: 6, Bytes: len(payload)}); err != nil {
		t.Fatal(err)
	}
	qt := b.m.q4kw[name]
	if qt == nil || len(qt.raw) != 0 || qt.lazy == nil {
		t.Fatalf("lazy tensor = %+v", qt)
	}
	got, err := qt.materializeRaw()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("materialized payload mismatch")
	}
	if len(qt.raw) != 0 {
		t.Fatalf("materialization retained %d bytes, want ephemeral", len(qt.raw))
	}
}

type closeWitness struct{ closes int }

func (c *closeWitness) Close() error { c.closes++; return nil }

func TestCloseWeightsOwnsLazyCheckpointLifetime(t *testing.T) {
	m := &Model{}
	w := &closeWitness{}
	m.SetWeightCloser(w)
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if w.closes != 1 {
		t.Fatalf("checkpoint close count = %d, want 1", w.closes)
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if w.closes != 1 {
		t.Fatalf("checkpoint close count after double close = %d, want 1", w.closes)
	}
}

type failingReaderAt struct{}

func (failingReaderAt) ReadAt([]byte, int64) (int, error) { return 0, errors.New("read failed") }

func TestCloseWeightsConcurrentCallsCloseOnce(t *testing.T) {
	m := &Model{}
	w := &closeWitness{}
	m.SetWeightCloser(w)
	var wg sync.WaitGroup
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.CloseWeights(); err != nil {
				t.Errorf("CloseWeights: %v", err)
			}
		}()
	}
	wg.Wait()
	if w.closes != 1 {
		t.Fatalf("concurrent checkpoint close count = %d, want 1", w.closes)
	}
}

func TestLazyQ4KMaterializationFailsClosed(t *testing.T) {
	qt := &q4kTensor{out: 1, in: 256, nblk: 1, lazy: &LazyQ4KRange{Reader: failingReaderAt{}, Bytes: q4kBlockBytes}}
	if _, err := qt.materializeRaw(); err == nil {
		t.Fatal("materialize succeeded after checkpoint read failure")
	}
}

func TestLazyQ4KCPUFallbackFailsClosed(t *testing.T) {
	qt := &q4kTensor{
		out:  1,
		in:   qkK,
		nblk: 1,
		lazy: &LazyQ4KRange{Reader: bytes.NewReader(make([]byte, q4kBlockBytes)), Bytes: q4kBlockBytes},
	}
	defer func() {
		got := recover()
		if got == nil {
			t.Fatal("CPU GEMV accepted a lazy Q4_K tensor")
		}
		if msg, ok := got.(string); !ok || msg == "" {
			t.Fatalf("CPU fallback panic = %#v, want a clear diagnostic", got)
		}
	}()
	q4kMatRows(qt, make([]float32, qkK))
}

func TestLazyQ4KMaterializationRejectsShortRead(t *testing.T) {
	qt := &q4kTensor{
		out:  1,
		in:   qkK,
		nblk: 1,
		lazy: &LazyQ4KRange{Reader: bytes.NewReader(make([]byte, q4kBlockBytes-1)), Bytes: q4kBlockBytes},
	}
	if _, err := qt.materializeRaw(); err == nil {
		t.Fatal("materialize accepted a truncated checkpoint range")
	}
}
