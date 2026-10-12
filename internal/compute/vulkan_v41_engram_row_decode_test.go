//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"
)

// v41EngramRowHostRef is the host Engram row decode: E4M3 times a binary32
// power of two then the v41BF16 RNE (packed), the scalar Q2_K super-block
// dequantization, or the stored word (f32), with the host's first refusal as a
// status word (1 + 2*j + kind).
func v41EngramRowHostRef(row []byte, encoding V41EngramRowEncoding, dim int) ([]uint32, uint32) {
	out := make([]uint32, dim)
	word := uint32(0)
	note := func(j int, kind uint32) {
		if word == 0 {
			word = 1 + uint32(2*j) + kind
		}
	}
	nonFinite := func(v float32) bool { return math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) }
	var q2k [256]float32
	if encoding == V41EngramRowQ2K {
		dequantQ2K(q2k[:], row)
	}
	for j := 0; j < dim; j++ {
		var v float32
		switch encoding {
		case V41EngramRowPacked:
			code, scale := row[j], row[256+j/32]
			if code&0x7f == 0x7f || scale == 0xff {
				note(j, 0)
				continue
			}
			e, m := int(code>>3)&15, float64(code&7)
			mag := math.Ldexp(m/8, -6)
			if e != 0 {
				mag = math.Ldexp(1+m/8, e-7)
			}
			if code&0x80 != 0 {
				mag = -mag
			}
			v = v41EngramRefBF16(float32(mag) * float32(math.Ldexp(1, int(scale)-127)))
		case V41EngramRowQ2K:
			v = q2k[j]
		default:
			v = math.Float32frombits(binary.LittleEndian.Uint32(row[4*j:]))
		}
		if nonFinite(v) {
			note(j, 1)
		}
		out[j] = math.Float32bits(v)
	}
	return out, word
}

// fak-test:runtime integration est=5s lane=optin
// Ref #13668. Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no
// physical witness. Every decoded lane is bit-exact against the host decode:
// all 65,536 E4M3 code x E8M0 scale pairs, random and edge Q2_K blocks, and F32
// words including NaN and Inf; every row status names the host's first refusal.
// The poison marks its target only after a faulting decode.
func TestVulkanV41EngramRowDecode(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41EngramRowDecode() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 Engram row decode module or binary32 float controls")
		}
		t.Skip("V4.1 Engram row decode module not loaded")
	}
	rng := rand.New(rand.NewSource(13668))
	f16s := []uint16{0, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00}
	type tc struct {
		name     string
		encoding V41EngramRowEncoding
		rows     int
		dim      int
		fill     func(row []byte, r int)
	}
	cases := []tc{
		{"packed-every-code-and-scale", V41EngramRowPacked, 256, 256, func(row []byte, r int) {
			for j := 0; j < 256; j++ {
				row[j] = byte(j)
			}
			for s := 0; s < 8; s++ {
				row[256+s] = byte(r)
			}
		}},
		{"packed-random-dim64", V41EngramRowPacked, 97, 64, func(row []byte, r int) {
			rng.Read(row)
			for j := range row[:256] {
				if row[j]&0x7f == 0x7f && r%5 != 0 {
					row[j] ^= 1
				}
			}
			for s := 256; s < 264; s++ {
				if row[s] == 0xff && r%7 != 0 {
					row[s] = 0xfe
				}
			}
		}},
		{"q2k", V41EngramRowQ2K, 2048, 256, func(row []byte, r int) {
			rng.Read(row)
			if r%3 == 0 {
				binary.LittleEndian.PutUint16(row[80:], f16s[rng.Intn(len(f16s))])
				binary.LittleEndian.PutUint16(row[82:], f16s[rng.Intn(len(f16s))])
			} else {
				binary.LittleEndian.PutUint16(row[80:], uint16(rng.Intn(0x7c00)))
				binary.LittleEndian.PutUint16(row[82:], uint16(rng.Intn(0x7c00))|uint16(rng.Intn(2))<<15)
			}
		}},
		{"q2k-dim24", V41EngramRowQ2K, 24, 24, func(row []byte, r int) {
			rng.Read(row)
			binary.LittleEndian.PutUint16(row[80:], uint16(rng.Intn(0x7c00)))
			binary.LittleEndian.PutUint16(row[82:], uint16(rng.Intn(0x7c00)))
		}},
		{"f32", V41EngramRowF32, 64, 256, func(row []byte, r int) {
			for j := 0; j < 256; j++ {
				w := rng.Uint32()
				if r%4 != 0 && w&0x7f800000 == 0x7f800000 {
					w &^= 0x00800000
				}
				binary.LittleEndian.PutUint32(row[4*j:], w)
			}
		}},
	}
	lanes, faults := 0, 0
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			stride := V41EngramRowStride(c.encoding)
			encoded := make([]byte, c.rows*stride)
			for r := 0; r < c.rows; r++ {
				c.fill(encoded[r*stride:(r+1)*stride], r)
			}
			panel, status, err := v.V41EngramRowDecode(encoded, c.encoding, c.rows, c.dim)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { v.Free(panel); v.Free(status) })
			got := append([]float32(nil), v.Read(panel)...)
			words := append([]float32(nil), v.Read(status)...)
			if len(got) != c.rows*c.dim || len(words) != c.rows {
				t.Fatalf("panel %d status %d, want %d and %d", len(got), len(words), c.rows*c.dim, c.rows)
			}
			for r := 0; r < c.rows; r++ {
				want, word := v41EngramRowHostRef(encoded[r*stride:(r+1)*stride], c.encoding, c.dim)
				if gotWord := math.Float32bits(words[r]); gotWord != word {
					t.Fatalf("row %d status %#x, want %#x", r, gotWord, word)
				}
				if word != 0 {
					faults++
				}
				for j := 0; j < c.dim; j++ {
					g := math.Float32bits(got[r*c.dim+j])
					w := want[j]
					if c.encoding == V41EngramRowPacked && (encoded[r*stride+j]&0x7f == 0x7f || encoded[r*stride+256+j/32] == 0xff) {
						continue // a NaN byte has no host value
					}
					if math.IsNaN(float64(math.Float32frombits(w))) && math.IsNaN(float64(math.Float32frombits(g))) {
						lanes++
						continue // both refuse; a NaN payload carries no value
					}
					if g != w {
						t.Fatalf("row %d element %d: device %08x host %08x", r, j, g, w)
					}
					lanes++
				}
			}
		})
	}
	t.Logf("V41_ENGRAM_ROW_DECODE lanes=%d bit_exact=%d faulting_rows=%d", lanes, lanes, faults)

	t.Run("poison", func(t *testing.T) {
		target := v.UploadClass(NewF32(Default(), []int{10}, make([]float32, 10)), F32, MemoryActivation, "V4.1 Engram poison witness")
		t.Cleanup(func() { v.Free(target) })
		clean := make([]byte, 3*84)
		for r := 0; r < 3; r++ {
			binary.LittleEndian.PutUint16(clean[r*84+80:], 0x3c00)
		}
		faulty := append([]byte(nil), clean...)
		binary.LittleEndian.PutUint16(faulty[2*84+82:], 0x7c00) // min = +Inf
		for _, fc := range []struct {
			name    string
			encoded []byte
			poison  bool
		}{{"clean", clean, false}, {"fault", faulty, true}} {
			_, status, err := v.V41EngramRowDecode(fc.encoded, V41EngramRowQ2K, 3, 16)
			if err != nil {
				t.Fatal(err)
			}
			if err := v.V41EngramRowDecodePoison(target, status, 3, 7); err != nil {
				t.Fatal(err)
			}
			v.Free(status)
			got := v.Read(target)
			if math.IsNaN(float64(got[7])) != fc.poison {
				t.Fatalf("%s: target[7]=%08x, poison=%t", fc.name, math.Float32bits(got[7]), fc.poison)
			}
			for i, x := range got {
				if i != 7 && math.Float32bits(x) != 0 {
					t.Fatalf("%s: poison wrote target[%d]=%08x", fc.name, i, math.Float32bits(x))
				}
			}
		}
	})

	t.Run("rejects", func(t *testing.T) {
		packed := make([]byte, 2*264)
		target := v.UploadClass(NewF32(Default(), []int{4}, make([]float32, 4)), F32, MemoryActivation, "V4.1 Engram poison witness")
		t.Cleanup(func() { v.Free(target) })
		_, status, err := v.V41EngramRowDecode(packed, V41EngramRowPacked, 2, 8)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { v.Free(status) })
		for name, call := range map[string]func() error{
			"unknown encoding": func() error { _, _, err := v.V41EngramRowDecode(packed, 7, 2, 8); return err },
			"dim zero":         func() error { _, _, err := v.V41EngramRowDecode(packed, V41EngramRowPacked, 2, 0); return err },
			"dim 257":          func() error { _, _, err := v.V41EngramRowDecode(packed, V41EngramRowPacked, 2, 257); return err },
			"short rows":       func() error { _, _, err := v.V41EngramRowDecode(packed[:264], V41EngramRowPacked, 2, 8); return err },
			"poison index":     func() error { return v.V41EngramRowDecodePoison(target, status, 2, 4) },
			"poison rows":      func() error { return v.V41EngramRowDecodePoison(target, status, 3, 0) },
			"host target": func() error {
				return v.V41EngramRowDecodePoison(NewF32(Default(), []int{4}, make([]float32, 4)), status, 2, 0)
			},
		} {
			if err := call(); err == nil {
				t.Fatalf("%s accepted", name)
			}
		}
	})
}
