//go:build cuda && cgo && fak_cuda_v41_embed_gather && fak_cuda_v41_engram_iq4nl

package compute

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// One real-device known-answer/negative fixture. The raw rows are the retained
// TestCUDAV41IQ4NLRowFiniteStatus fixture; the scalar oracle is existing compute
// IQ4_NL code, not another decoder. Missing hardware is a failure, not a pass.
// fak-test:runtime integration est=1s lane=optin
func TestCUDAV41EngramIQ4NLRows(t *testing.T) {
	be := Pick("cuda")
	c, ok := be.(*cudaBackend)
	if !ok || c == nil || c != cudaDev {
		t.Fatal("registered CUDA backend required for IQ4_NL row witness")
	}
	if c.SupportsV41EngramRowDecode() || c.SupportsV41EngramRowEncoding(V41EngramRowIQ4NL) ||
		V41EngramRowDecodeSupportsEncoding(c, V41EngramRowIQ4NL) {
		t.Fatal("unqualified IQ4_NL adapter advertised a serving capability")
	}
	const dim = 97
	raw := make([]byte, 2*144)
	for row := 0; row < 2; row++ {
		for block := 0; block < 8; block++ {
			p := raw[row*144+block*18:]
			p[1] = 0x3c
			for j := 0; j < 16; j++ {
				p[2+j] = byte(j | (15-j)<<4)
			}
		}
	}
	raw[7*18], raw[7*18+1] = 1, 0x7e // outside the clean row's requested prefix
	raw[144+2*18+1] = 0x7c
	raw[144+3*18], raw[144+3*18+1] = 1, 0x7e
	ids := []int{1, 0, 1}
	stored := make([]byte, 0, len(ids)*144)
	for _, id := range ids {
		stored = append(stored, raw[id*144:(id+1)*144]...)
	}
	before := c.CUDADebugPoolStats()
	if panel, status, err := c.V41EngramRowDecode(stored, V41EngramRowIQ4NL, len(ids), dim); err == nil || panel.Buf() != nil || status.Buf() != nil {
		t.Fatal("public method did not refuse the unqualified route with empty outputs")
	}
	if panel, status, err := c.cudaV41EngramRowDecode(nil, V41EngramRowEncoding(99), 1, dim, true); err == nil ||
		!strings.Contains(err.Error(), "unknown encoding") || panel.Buf() != nil || status.Buf() != nil {
		t.Fatal("unknown dialect crossed the adapter admission boundary")
	}
	if after := c.CUDADebugPoolStats(); after != before {
		t.Fatal("refused adapter call changed allocator pools")
	}
	panel, status, err := c.cudaV41EngramRowDecode(stored, V41EngramRowIQ4NL, len(ids), dim, true)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(panel)
	defer be.Free(status)
	if panel.Buf() == nil || status.Buf() == nil || panel.Buf() == status.Buf() || !panel.Ready() || !status.Ready() {
		t.Fatal("decoded panel/status were not fresh, distinct and complete")
	}
	got, words := be.Read(panel), be.Read(status)
	if len(got) != len(ids)*dim || len(words) != len(ids) {
		t.Fatal("decoded tensor extents differ from the gathered rows")
	}
	for r, id := range ids {
		wantStatus := uint32(0)
		if id == 1 {
			wantStatus = 130 // first nonfinite is j=64, kind=1
		}
		if bits := math.Float32bits(words[r]); bits != wantStatus {
			t.Fatalf("row %d status=%d, want %d", r, bits, wantStatus)
		}
		var want [256]float32
		iq4nlDequantSuperBlock(want[:], raw[id*144:(id+1)*144])
		for j, value := range want[:dim] {
			actual := got[r*dim+j]
			if math.IsNaN(float64(value)) {
				if !math.IsNaN(float64(actual)) {
					t.Fatalf("row %d element %d lost NaN", r, j)
				}
			} else if math.Float32bits(actual) != math.Float32bits(value) {
				t.Fatalf("row %d element %d bits=%08x, want %08x", r, j, math.Float32bits(actual), math.Float32bits(value))
			}
		}
	}
	// Alias refusal must not mark the still-valid status tensor as poisoned.
	if err := c.cudaV41EngramRowDecodePoison(status, status, len(ids), 1, true); err == nil || !status.Ready() {
		t.Fatal("poison alias was accepted or invalidated a pre-dispatch input")
	}
	for r, value := range be.Read(status) {
		if math.Float32bits(value) != math.Float32bits(words[r]) {
			t.Fatal("refused poison call changed status")
		}
	}
	target := be.Upload(NewF32(Default(), []int{3}, []float32{1, math.Float32frombits(0x80000000), -1}), F32)
	defer be.Free(target)
	if err := c.cudaV41EngramRowDecodePoison(target, status, len(ids), 1, true); err != nil {
		t.Fatal(err)
	}
	poisoned := be.Read(target)
	if math.Float32bits(poisoned[0]) != 0x3f800000 || math.Float32bits(poisoned[1]) != 0x7fc00000 || math.Float32bits(poisoned[2]) != 0xbf800000 {
		t.Fatal("first-fault status did not poison exactly the selected target element")
	}
}

// The scalar oracle follows the existing host/Vulkan fixture, not the CUDA
// integer packed decoder. In particular it multiplies two binary32 values
// before BF16 RNE. Q2_K uses the existing shared scalar super-block routine.
func cudaEngramRowsReference(row []byte, encoding V41EngramRowEncoding, dim int) ([]uint32, uint32) {
	out := make([]uint32, dim)
	var word uint32
	var q2 [256]float32
	if encoding == V41EngramRowQ2K {
		q2kDequantSuperBlock(q2[:], row)
	}
	for j := 0; j < dim; j++ {
		var v float32
		switch encoding {
		case V41EngramRowPacked:
			code, scale := row[j], row[256+j/32]
			if code&0x7f == 0x7f || scale == 0xff {
				if word == 0 {
					word = 1 + uint32(2*j)
				}
				continue
			}
			v = DecodeE4M3(code) * float32(math.Ldexp(1, int(scale)-127))
			bits := math.Float32bits(v)
			if bits&0x7f800000 != 0x7f800000 {
				bits += 0x7fff + ((bits >> 16) & 1)
			}
			v = math.Float32frombits(bits & 0xffff0000)
		case V41EngramRowQ2K:
			v = q2[j]
		default:
			v = math.Float32frombits(binary.LittleEndian.Uint32(row[4*j:]))
		}
		if word == 0 && (math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)) {
			word = 2 + uint32(2*j)
		}
		out[j] = math.Float32bits(v)
	}
	return out, word
}

// One source-bound actual CUDA family witness. The explicit build tags select
// this test; absence of the registered CUDA backend/ABI is fatal, never a skip.
// Native archive must define FAK_CUDA_V41_EMBED_GATHER and
// FAK_CUDA_V41_EMBED_IQ4NL. This does not qualify serving or a model request.
// fak-test:runtime integration est=5s lane=optin
func TestCUDAV41EngramRows(t *testing.T) {
	be := Pick("cuda")
	c, ok := be.(*cudaBackend)
	if !ok || c == nil || c != cudaDev {
		t.Fatal("registered CUDA backend required for checked Engram family witness")
	}
	for _, encoding := range []V41EngramRowEncoding{V41EngramRowPacked, V41EngramRowQ2K, V41EngramRowF32, V41EngramRowIQ4NL} {
		if c.SupportsV41EngramRowEncoding(encoding) || V41EngramRowDecodeSupportsEncoding(c, encoding) {
			t.Fatal("unqualified row family advertised a serving capability")
		}
	}
	if c.SupportsV41EngramRowDecode() {
		t.Fatal("unqualified row family advertised broad capability")
	}
	// Keep the original nonlinear decoder's repeated-row, out-of-prefix fault,
	// fresh status, exact poison and pre-mutation alias witness in the family.
	t.Run("iq4nl-existing-contract", TestCUDAV41EngramIQ4NLRows)
	type fixture struct {
		name      string
		encoding  V41EngramRowEncoding
		rows, dim int
		fill      func([]byte, int)
	}
	cases := []fixture{
		{"packed-every-code-and-scale", V41EngramRowPacked, 256, 256, func(row []byte, r int) {
			for j := 0; j < 256; j++ {
				row[j] = byte(j)
			}
			for j := 256; j < 264; j++ {
				row[j] = byte(r)
			}
		}},
		{"packed-prefix-and-first-fault", V41EngramRowPacked, 3, 97, func(row []byte, r int) {
			for j := 0; j < 256; j++ {
				row[j] = 0x38
			}
			for j := 256; j < 264; j++ {
				row[j] = 127
			}
			row[250] = 0x7f // all rows: outside requested prefix, not a status fault
			if r == 1 {
				row[2] = 0x7f
				row[256+2] = 0xff
			}
			if r == 2 {
				row[256+2] = 0xff
			}
		}},
		{"q2k-shared-scalar-edges", V41EngramRowQ2K, 8, 256, func(row []byte, r int) {
			for j := range row {
				row[j] = byte(j*37 + r*11)
			}
			halves := []uint16{0, 0x8000, 1, 0x3c00, 0x7bff, 0x7c00, 0x7e00, 0xfc00}
			binary.LittleEndian.PutUint16(row[80:], halves[r])
			binary.LittleEndian.PutUint16(row[82:], halves[(r+3)%len(halves)])
		}},
		{"f32-bits-prefix-and-status", V41EngramRowF32, 3, 97, func(row []byte, r int) {
			values := []uint32{0, 0x80000000, 1, 0x007fffff, 0x3f800000, 0x7f7fffff}
			for j := 0; j < 256; j++ {
				binary.LittleEndian.PutUint32(row[4*j:], values[j%len(values)])
			}
			binary.LittleEndian.PutUint32(row[250*4:], 0x7fc01234)
			if r == 1 {
				binary.LittleEndian.PutUint32(row[64*4:], 0x7fc01234)
			}
			if r == 2 {
				binary.LittleEndian.PutUint32(row[32*4:], 0xff800000)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stride := V41EngramRowStride(tc.encoding)
			raw := make([]byte, tc.rows*stride)
			for r := 0; r < tc.rows; r++ {
				tc.fill(raw[r*stride:(r+1)*stride], r)
			}
			before := append([]byte(nil), raw...)
			poolBefore := c.CUDADebugPoolStats()
			if p, s, err := c.V41EngramRowDecode(raw, tc.encoding, tc.rows, tc.dim); err == nil || p.Buf() != nil || s.Buf() != nil {
				t.Fatal("public method did not refuse before allocation")
			}
			if p, s, err := c.cudaV41EngramRowDecode(raw[:len(raw)-1], tc.encoding, tc.rows, tc.dim, true); err == nil || p.Buf() != nil || s.Buf() != nil {
				t.Fatal("short encoded panel did not refuse with empty outputs")
			}
			if poolAfter := c.CUDADebugPoolStats(); poolAfter != poolBefore {
				t.Fatal("pre-mutation refusal changed pools")
			}
			panel, status, err := c.cudaV41EngramRowDecode(raw, tc.encoding, tc.rows, tc.dim, true)
			if err != nil {
				t.Fatal(err)
			}
			defer be.Free(panel)
			defer be.Free(status)
			if panel.Buf() == nil || status.Buf() == nil || panel.Buf() == status.Buf() || !panel.Ready() || !status.Ready() {
				t.Fatal("row family did not publish distinct fresh completed outputs")
			}
			got, words := be.Read(panel), be.Read(status)
			if len(got) != tc.rows*tc.dim || len(words) != tc.rows {
				t.Fatal("row family output extent mismatch")
			}
			for r := 0; r < tc.rows; r++ {
				want, wantWord := cudaEngramRowsReference(raw[r*stride:(r+1)*stride], tc.encoding, tc.dim)
				if word := math.Float32bits(words[r]); word != wantWord {
					t.Fatalf("row %d status=%d want=%d", r, word, wantWord)
				}
				for j, bits := range want {
					actual := got[r*tc.dim+j]
					// Quantized NaN payloads are not the scalar contract. F32 is a raw
					// word copy, so its NaN payload/sign bits must remain exact too.
					if tc.encoding != V41EngramRowF32 && math.IsNaN(float64(math.Float32frombits(bits))) {
						if !math.IsNaN(float64(actual)) {
							t.Fatalf("row %d element %d lost NaN", r, j)
						}
					} else if math.Float32bits(actual) != bits {
						t.Fatalf("row %d element %d bits=%08x want=%08x", r, j, math.Float32bits(actual), bits)
					}
				}
			}
			if !bytes.Equal(raw, before) {
				t.Fatal("borrowed encoded rows were mutated")
			}
		})
	}
}
