package model

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type lazyQ4KUploadBackend struct {
	compute.Backend
	got []byte
}

func (b *lazyQ4KUploadBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (b *lazyQ4KUploadBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	if as != compute.Q4_K {
		panic("unexpected upload dtype")
	}
	b.got = append([]byte(nil), int8Bytes(t.Buf().(compute.HostBuffer).I8())...)
	return t
}

func int8Bytes(in []int8) []byte {
	out := make([]byte, len(in))
	for i, v := range in {
		out[i] = byte(v)
	}
	return out
}

func TestWeightHALQ4KStagesLazyCheckpointPayload(t *testing.T) {
	raw := make([]byte, q4kBlockBytes)
	for i := range raw {
		raw[i] = byte(i)
	}
	qt := &q4kTensor{out: 1, in: qkK, nblk: 1, lazy: &LazyQ4KRange{Reader: bytes.NewReader(raw), Bytes: len(raw)}}
	be := &lazyQ4KUploadBackend{Backend: compute.Default()}
	s := &Session{Backend: be, halW: map[string]compute.Tensor{}}
	s.weightHALQ4K("fixture", qt)
	if !bytes.Equal(be.got, raw) {
		t.Fatalf("uploaded bytes differ: got %d want %d", len(be.got), len(raw))
	}
	if got := q4kResidentBytes(qt); got != int64(len(raw)) {
		t.Fatalf("resident bytes = %d, want %d", got, len(raw))
	}
}
