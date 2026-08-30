package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

// These IQ1/IQ2 oracle digests and matvec result bits were produced by the scalar
// dequantize_row_iq* routines in the pinned llama.cpp revision documented beside the production
// decoder (6fe74980162af0ed5e559870d5deccafaa034e7c). The fixture bytes are generated here rather
// than copied from a model, so the witness is deterministic and exercises codebook indexes, sign
// selectors, sub-block scales, and row/block strides without depending on an external artifact.
var iq12ResidentCases = []struct {
	name       string
	kind       kQuantKind
	blockBytes int
	add        func(*QuantBuilder, string, []int, []byte) error
	oracleSHA  string
	matvecBits [3]uint32
}{
	{
		name:       "IQ2_XXS",
		kind:       kindIQ2XXS,
		blockBytes: 66,
		add:        (*QuantBuilder).AddResidentIQ2XXS,
		oracleSHA:  "a2e7d58b4bb58222e7274404028084bdd6bf0b69dadcbcff98abe95fdf1be4c0",
		matvecBits: [3]uint32{0x4676f780, 0x45936100, 0xc5d04d00},
	},
	{
		name:       "IQ2_XS",
		kind:       kindIQ2XS,
		blockBytes: 74,
		add:        (*QuantBuilder).AddResidentIQ2XS,
		oracleSHA:  "02cfdbe97860a584bbba443b2eae3f0df660ac6d138847968b656436b188df1e",
		matvecBits: [3]uint32{0xc5a32c00, 0x469c7180, 0x45213a00},
	},
	{
		name:       "IQ1_S",
		kind:       kindIQ1S,
		blockBytes: 50,
		add:        (*QuantBuilder).AddResidentIQ1S,
		oracleSHA:  "5db31749f88a458217f7b903c369f03d8203bd8c2676a7631cc005a5b19cee69",
		matvecBits: [3]uint32{0xc47da000, 0xc3c36000, 0x44718000},
	},
	{
		name:       "IQ2_S",
		kind:       kindIQ2S,
		blockBytes: 82,
		add:        (*QuantBuilder).AddResidentIQ2S,
		oracleSHA:  "1563356ad4a6765ecff973aa884e938676251dac8aa63322741b62f5e01b17e9",
		matvecBits: [3]uint32{0xc45df800, 0xc4c8e400, 0x447a5800},
	},
	{
		name:       "IQ1_M",
		kind:       kindIQ1M,
		blockBytes: 56,
		add:        (*QuantBuilder).AddResidentIQ1M,
		oracleSHA:  "22b952a3fb8808769869b7d13bfc4a2c606efca978494d3e8511fe8a744f42eb",
		matvecBits: [3]uint32{0x44269000, 0x4424e000, 0xc37a4000},
	},
}

func TestIQ12ResidentPinnedCodebookDigests(t *testing.T) {
	tests := []struct {
		name   string
		values []uint64
		digest string
	}{
		{"iq2xxs_grid", iq2XXSGrid[:], "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"},
		{"iq2xs_grid", iq2XSGrid[:], "06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de"},
		{"iq2s_grid", iq2SGrid[:], "e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88"},
		{"iq1s_grid", iq1SGrid[:], "07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, 8*len(tc.values))
			for i, value := range tc.values {
				binary.LittleEndian.PutUint64(raw[8*i:], value)
			}
			if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != tc.digest {
				t.Fatalf("table digest = %s, want pinned llama.cpp digest %s", got, tc.digest)
			}
		})
	}
}

func pinIQ12Scale(blk []byte, kind kQuantKind) {
	if kind != kindIQ1M {
		binary.LittleEndian.PutUint16(blk, f16One)
		return
	}
	// IQ1_M distributes the four nibbles of its shared f16 scale over the high nibble of
	// four uint16 scale words. Preserve each word's low 12 block-scale bits while pinning
	// the assembled f16 to 1.0 (0x3c00).
	scales := blk[iq1mBlockBytes-qkK/32:]
	high := [...]uint16{0x0000, 0x0000, 0xc000, 0x3000}
	for i := range high {
		v := binary.LittleEndian.Uint16(scales[2*i:])
		binary.LittleEndian.PutUint16(scales[2*i:], v&0x0fff|high[i])
	}
}

func makeIQ12Raw(kind kQuantKind, out, in int, seed uint64) []byte {
	nblk := in / kind.blockWeights()
	raw := make([]byte, out*nblk*kind.blockBytes())
	lcgBytes(raw, seed)
	for b := 0; b < out*nblk; b++ {
		pinIQ12Scale(raw[b*kind.blockBytes():], kind)
	}
	return raw
}

func TestIQ12PinnedLlamaDequantOracle(t *testing.T) {
	const seed = 0x6a09e667f3bcc909
	for _, tc := range iq12ResidentCases {
		t.Run(tc.name, func(t *testing.T) {
			blk := makeIQ12Raw(tc.kind, 1, qkK, seed)
			got := make([]float32, qkK)
			kQuantDequantSuperBlock(got, blk, tc.kind)
			encoded := make([]byte, 4*len(got))
			for i, v := range got {
				binary.LittleEndian.PutUint32(encoded[4*i:], math.Float32bits(v))
			}
			sum := sha256.Sum256(encoded)
			if gotSHA := fmt.Sprintf("%x", sum); gotSHA != tc.oracleSHA {
				t.Fatalf("pinned llama.cpp dequant digest=%s, want %s", gotSHA, tc.oracleSHA)
			}
		})
	}
}

func TestIQ12ResidentGeometryAndBuilderValidation(t *testing.T) {
	const (
		out = 3
		in  = 512
	)
	name := "model.layers.7.mlp.experts.2.down_proj.weight"
	for _, tc := range iq12ResidentCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.kind.String(); got != tc.name {
				t.Fatalf("String()=%q, want %q", got, tc.name)
			}
			if got := tc.kind.blockWeights(); got != qkK {
				t.Fatalf("blockWeights()=%d, want %d", got, qkK)
			}
			if got := tc.kind.blockBytes(); got != tc.blockBytes {
				t.Fatalf("blockBytes()=%d, want %d", got, tc.blockBytes)
			}

			raw := makeIQ12Raw(tc.kind, out, in, 0x3c6ef372fe94f82b)
			b := NewQuantBuilder(Config{}, false)
			if err := tc.add(b, name, []int{out, in}, raw); err != nil {
				t.Fatalf("AddResident%s: %v", tc.name, err)
			}
			qt := b.m.kqw[name]
			if qt == nil || qt.kind != tc.kind || qt.out != out || qt.in != in || qt.nblk != in/qkK {
				t.Fatalf("stored tensor=%+v, want kind=%s shape=[%d,%d] nblk=%d",
					qt, tc.kind, out, in, in/qkK)
			}
			if !bytes.Equal(qt.raw, raw) {
				t.Fatal("resident payload changed instead of retaining GGUF bytes verbatim")
			}

			t.Run("non-2D-shape-skips", func(t *testing.T) {
				skip := NewQuantBuilder(Config{}, false)
				if err := tc.add(skip, name, []int{in}, raw); err != nil {
					t.Fatalf("non-2D shape returned error: %v", err)
				}
				if skip.m.kqw[name] != nil {
					t.Fatal("non-2D shape was stored")
				}
			})
			t.Run("reduction-block-mismatch", func(t *testing.T) {
				assertIQ12PanicContains(t, "reduction dim not a multiple of block size", func() {
					_ = tc.add(NewQuantBuilder(Config{}, false), name, []int{out, in - 1}, nil)
				})
			})
			t.Run("raw-size-mismatch", func(t *testing.T) {
				assertIQ12PanicContains(t, "payload size mismatch", func() {
					_ = tc.add(NewQuantBuilder(Config{}, false), name, []int{out, in}, raw[:len(raw)-1])
				})
			})
		})
	}
}

func assertIQ12PanicContains(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		got := recover()
		if got == nil {
			t.Fatalf("did not panic; want panic containing %q", want)
		}
		if text := fmt.Sprint(got); !strings.Contains(text, want) {
			t.Fatalf("panic=%q, want substring %q", text, want)
		}
	}()
	f()
}

func TestIQ12ResidentNativeMatvecPinnedOracle(t *testing.T) {
	const (
		out  = 3
		in   = 512
		seed = 0xbb67ae8584caa73b
	)
	name := "model.layers.7.mlp.experts.2.gate_proj.weight"
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*17)%29) - 14
	}
	for _, tc := range iq12ResidentCases {
		t.Run(tc.name, func(t *testing.T) {
			raw := makeIQ12Raw(tc.kind, out, in, seed)
			b := NewQuantBuilder(Config{}, false)
			if err := tc.add(b, name, []int{out, in}, raw); err != nil {
				t.Fatalf("AddResident%s: %v", tc.name, err)
			}
			got := b.m.residentMatRows(name, x, out, in)
			for row, wantBits := range tc.matvecBits {
				if gotBits := math.Float32bits(got[row]); gotBits != wantBits {
					t.Fatalf("row %d result=%v bits=%08x, want pinned llama.cpp matvec bits=%08x",
						row, got[row], gotBits, wantBits)
				}
			}
		})
	}
}
