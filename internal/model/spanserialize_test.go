package model

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func denseSpanCfg() Config {
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		RopeThetaPerLayer: []float64{10000, 1000000},
		BlockTopology:     SandwichNorm,
	}
}

func spanF32Equal(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestSerializeSpanRoundTripsBitExact proves the byte-source #1472 stages: a middle
// span serialized off a real dense KV cache decodes back to exactly the cache's
// Kraw/V/pos for that span (max|delta|=0) — the pre-RoPE rows a durable L3 tier
// needs to restore the span bit-exact at a new position.
func TestSerializeSpanRoundTripsBitExact(t *testing.T) {
	cfg := denseSpanCfg()
	s := NewSynthetic(cfg).NewSession()
	s.Prefill([]int{3, 17, 5, 23, 41, 2, 19})

	from, n := 2, 3
	b, err := s.Cache.SerializeSpan(from, n)
	if err != nil {
		t.Fatalf("SerializeSpan: %v", err)
	}

	w := s.Cache.kvStride()
	r := bytes.NewReader(b)
	var hdr [4]uint32
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		t.Fatalf("read header: %v", err)
	}
	if hdr[0] != spanSerMagic {
		t.Fatalf("magic = %#x, want %#x", hdr[0], spanSerMagic)
	}
	if int(hdr[1]) != cfg.NumLayers || int(hdr[2]) != w || int(hdr[3]) != n {
		t.Fatalf("header = %v, want layers=%d w=%d n=%d", hdr, cfg.NumLayers, w, n)
	}
	gotPos := make([]uint32, n)
	if err := binary.Read(r, binary.LittleEndian, &gotPos); err != nil {
		t.Fatalf("read pos: %v", err)
	}
	for i := 0; i < n; i++ {
		if int(gotPos[i]) != s.Cache.pos[from+i] {
			t.Fatalf("pos[%d] = %d, want %d", i, gotPos[i], s.Cache.pos[from+i])
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		gotK := make([]float32, n*w)
		if err := binary.Read(r, binary.LittleEndian, &gotK); err != nil {
			t.Fatalf("read Kraw L%d: %v", l, err)
		}
		if !spanF32Equal(gotK, s.Cache.Kraw[l][from*w:(from+n)*w]) {
			t.Fatalf("Kraw L%d not bit-exact", l)
		}
		gotV := make([]float32, n*w)
		if err := binary.Read(r, binary.LittleEndian, &gotV); err != nil {
			t.Fatalf("read V L%d: %v", l, err)
		}
		if !spanF32Equal(gotV, s.Cache.V[l][from*w:(from+n)*w]) {
			t.Fatalf("V L%d not bit-exact", l)
		}
	}
	if r.Len() != 0 {
		t.Fatalf("trailing bytes after decode: %d", r.Len())
	}
}

// TestKVBackendExposesStageSpanBytes proves the in-process KV backend now exposes
// the SpanStager capability (StageSpanBytes) — the byte-source that lets the durable
// L3 backend (internal/l3kv) stage a real span instead of reporting a FAULT.
func TestKVBackendExposesStageSpanBytes(t *testing.T) {
	s := NewSynthetic(denseSpanCfg()).NewSession()
	s.Prefill([]int{3, 17, 5, 23, 41})

	kb, ok := KVBackend(s)
	if !ok {
		t.Fatal("KVBackend ok=false")
	}
	st, ok := kb.(interface {
		StageSpanBytes(from, n int) ([]byte, error)
	})
	if !ok {
		t.Fatal("in-process KV backend does not expose StageSpanBytes (SpanStager) — l3kv cannot stage a real span")
	}
	b, err := st.StageSpanBytes(1, 2)
	if err != nil {
		t.Fatalf("StageSpanBytes: %v", err)
	}
	if len(b) == 0 {
		t.Fatal("StageSpanBytes returned empty serialization")
	}
	b2, err := s.Cache.SerializeSpan(1, 2)
	if err != nil {
		t.Fatalf("SerializeSpan: %v", err)
	}
	if !bytes.Equal(b, b2) {
		t.Fatal("StageSpanBytes bytes differ from SerializeSpan")
	}
}

// TestSerializeSpanRejectsBadRange proves an out-of-range span is a typed error
// (which the L3 wrapper surfaces as a FAULT, retaining the live span), never a panic
// or a truncated blob.
func TestSerializeSpanRejectsBadRange(t *testing.T) {
	s := NewSynthetic(denseSpanCfg()).NewSession()
	s.Prefill([]int{1, 2, 3})
	if _, err := s.Cache.SerializeSpan(2, 5); err == nil {
		t.Fatal("SerializeSpan(2,5) over a 3-position cache: want out-of-range error")
	}
	if _, err := s.Cache.SerializeSpan(-1, 1); err == nil {
		t.Fatal("SerializeSpan(-1,1): want error")
	}
	if _, err := s.Cache.SerializeSpan(0, 0); err == nil {
		t.Fatal("SerializeSpan(0,0): want error")
	}
}
