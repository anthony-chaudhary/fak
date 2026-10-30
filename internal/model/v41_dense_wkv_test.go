package model

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type denseWKVTestWeights struct {
	calls int
	fail  bool
}

func (w *denseWKVTestWeights) Release() error {
	w.calls++
	if w.fail {
		return errors.New("retirement refused")
	}
	return nil
}

type denseWKVTestBackend struct {
	compute.Backend
	w                 denseWKVTestWeights
	uploads, projects int
	qualified         bool
}

func (b *denseWKVTestBackend) V41FP8WKVSupported() bool { return true }
func (b *denseWKVTestBackend) V41FP8WKVQualified() bool { return b.qualified }
func (b *denseWKVTestBackend) UploadV41FP8WKV(w, s []byte) (compute.V41FP8WKVWeights, error) {
	b.uploads++
	return &b.w, nil
}
func (b *denseWKVTestBackend) V41FP8WKV(w compute.V41FP8WKVWeights, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.projects++
	return compute.Tensor{Dtype: compute.BF16, Shape: []int{rows, 512}}, nil
}

func TestV41DenseWKVBoundAndLifetime(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	// Fixture entries carry the exact on-disk names of the pinned checkpoint
	// shard, which is what ReadV41DenseWKV references through the inventory.
	name, scale := v41DenseWKVNames(0)
	data := make([]byte, V41DenseWKVPayloadBytes)
	for i := 512 * 5120; i < len(data); i++ {
		data[i] = 127
	}
	path := filepath.Join(t.TempDir(), "wkv.safetensors")
	v41TestWriteEntries(t, path, map[string]stEntry{
		name:  {Dtype: "F8_E4M3", Shape: []int{512, 5120}, DataOffsets: []int{0, 512 * 5120}},
		scale: {Dtype: "F8_E8M0", Shape: []int{16, 160}, DataOffsets: []int{512 * 5120, len(data)}},
	}, data)
	s, err := OpenV41PackedShard(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, bound := range []int64{0, -1, V41DenseWKVPayloadBytes - 1} {
		if _, err := s.ReadV41DenseWKV(0, bound); err == nil {
			t.Fatalf("accepted bound %d", bound)
		}
	}
	w, err := s.ReadV41DenseWKV(0, V41DenseWKVPayloadBytes)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if len(w.codes) != 512*5120 || w.scales[0] != 127 {
		t.Fatal("owned pair lost after close")
	}
	b := &denseWKVTestBackend{}
	if _, err := w.Bind(b); err == nil || b.uploads != 0 {
		t.Fatal("unqualified production bind admitted")
	}
	p, err := w.BindForWitness(b)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if _, err = p.Project(compute.Tensor{}, 128); err != nil {
			t.Fatal(err)
		}
	}
	if b.uploads != 1 || b.projects != 2 {
		t.Fatal("projection reuploaded or failed to invoke backend")
	}
	b.w.fail = true
	if err := p.Release(); err == nil {
		t.Fatal("release failure hidden")
	}
	b.w.fail = false
	if err := p.Release(); err != nil {
		t.Fatal(err)
	}
	if err := p.Release(); err != nil || b.w.calls != 2 {
		t.Fatal("release not idempotent")
	}
	if _, err := p.Project(compute.Tensor{}, 128); err == nil {
		t.Fatal("released handle resurrected")
	}
}

func TestV41DenseWKVFinitePair(t *testing.T) {
	w := make([]byte, 512*5120)
	s := make([]byte, 16*160)
	for i := range s {
		s[i] = 127
	}
	if err := validateV41DenseWKV(w, s); err != nil {
		t.Fatal(err)
	}
	w[0] = 0x7f
	if err := validateV41DenseWKV(w, s); err == nil {
		t.Fatal("NaN code admitted")
	}
	w[0] = 0
	s[0] = 255
	if err := validateV41DenseWKV(w, s); err == nil {
		t.Fatal("NaN scale admitted")
	}
	s[0] = 254
	w[0] = 0x7e
	if err := validateV41DenseWKV(w, s); err == nil {
		t.Fatal("F32 overflow admitted")
	}
}
