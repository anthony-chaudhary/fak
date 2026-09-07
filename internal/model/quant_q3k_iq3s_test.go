package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Goldens are SHA-256 of 1024 little-endian f32 values emitted by the actual
// dequantize_row_q3_K / dequantize_row_iq3_s C functions and IQ3_S codebook from
// ggml-org/llama.cpp@6fe74980162af0ed5e559870d5deccafaa034e7c, ggml/src/ggml-quants.c
// and ggml-common.h. Compiled with cc -O0 -ffp-contract=off. Input bytes are
// (i*37+11)%256 for four 110-byte blocks, with each fp16 scale replaced by 0x3c00.
// The oracle was run before extraction; it does not call these Go decoders.
func TestQ3KIQ3SNativeReference(t *testing.T) {
	const rows, cols = 2, 512
	for _, tc := range []struct {
		name   string
		kind   kQuantKind
		scale  int
		decode func([]float32, []byte)
		digest string
	}{
		{"Q3_K", kindQ3K, 108, DequantQ3K, "0a00e55eace95e14b27bd3699aff2378beeee5338838f2ccd1a3e8b578f210b6"},
		{"IQ3_S", kindIQ3S, 0, DequantIQ3S, "1e2334ae68cc0bf1c4e5bc55da93df406fffed8f07490ecb3787498395532bb6"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, 4*110)
			for i := range raw {
				raw[i] = byte(i*37 + 11)
			}
			for b := 0; b < 4; b++ {
				binary.LittleEndian.PutUint16(raw[b*110+tc.scale:], 0x3c00)
			}
			decoded := make([]float32, rows*cols)
			tc.decode(decoded, raw)
			bits := make([]byte, len(decoded)*4)
			for i, v := range decoded {
				binary.LittleEndian.PutUint32(bits[4*i:], math.Float32bits(v))
			}
			if got := fmt.Sprintf("%x", sha256.Sum256(bits)); got != tc.digest {
				t.Fatalf("independent upstream decoder digest=%s, want %s", got, tc.digest)
			}
			if tc.kind.String() != tc.name || tc.kind.blockBytes() != 110 || tc.kind.blockWeights() != 256 {
				t.Fatal("wrong format descriptor")
			}
			qt := quantizeKQuantFromRaw(raw, rows, cols, tc.kind)
			name := "model.layers.0.mlp.down_proj.weight"
			m := &Model{kqw: map[string]*kQuantTensor{name: qt}}
			s := &Session{M: m, Backend: compute.Default(), halW: map[string]compute.Tensor{}}
			x := make([]float32, 2*cols)
			for i := range x {
				x[i] = float32((i*7)%23-11) / 32
			}
			want := make([]float32, 2*rows)
			// Independent fixed-order dot of the complete f32 oracle-checked matrix.
			for token := 0; token < 2; token++ {
				for row := 0; row < rows; row++ {
					for b := 0; b < 2; b++ {
						var a0, a1, a2, a3 float32
						for j := 0; j < 256; j += 4 {
							w := decoded[row*cols+b*256+j:]
							v := x[token*cols+b*256+j:]
							a0 += w[0] * v[0]
							a1 += w[1] * v[1]
							a2 += w[2] * v[2]
							a3 += w[3] * v[3]
						}
						want[token*rows+row] += (a0 + a1) + (a2 + a3)
					}
				}
			}
			gemv := make([]float32, rows)
			s.kQuantMatRowsIntoDispatch(name, qt, x[:cols], gemv)
			assertSameF32(t, "native GEMV", gemv, want[:rows])
			assertSameF32(t, "native GEMM", s.kQuantGemmDispatch(name, qt, x, 2), want)
			assertSameF32(t, "backend native host path", (backendKernel{s: s}).mul(name, x[:cols], rows, cols), want[:rows])
			if len(s.halW) != 0 {
				t.Fatal("unsupported packed type was staged to device")
			}
			// A backend advertising routed K-quants must decline formats whose packed
			// device staging is unsupported, leaving actual execution on the native host.
			be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
			s.Backend = be
			if _, used := s.expertSwiGLUHAL(name, name, name, x[:cols]); used {
				t.Fatal("unsupported packed expert format admitted to device")
			}
			if len(be.uploads) != 0 || len(s.halW) != 0 {
				t.Fatal("declined expert uploaded weights")
			}
			assertSameF32(t, "capable backend host fallback", (backendKernel{s: s}).mul(name, x[:cols], rows, cols), want[:rows])
			got, ok := m.KQuantRaw(name)
			if !ok || !bytes.Equal(got, raw) {
				t.Fatal("resident payload differs from source")
			}
		})
	}
}
