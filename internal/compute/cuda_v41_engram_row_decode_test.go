//go:build cuda && cgo && fak_cuda_v41_embed_gather && fak_cuda_v41_engram_iq4nl

package compute

import (
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
	if panel, status, err := c.cudaV41EngramRowDecode(make([]byte, 1024), V41EngramRowF32, 1, dim, true); err == nil ||
		!strings.Contains(err.Error(), "only IQ4_NL") || panel.Buf() != nil || status.Buf() != nil {
		t.Fatal("non-IQ4_NL dialect crossed the adapter admission boundary")
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
