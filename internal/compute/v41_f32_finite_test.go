package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestV41F32FiniteEnvelope(t *testing.T) {
	t.Parallel()
	record := func(magic, site, index, count uint32) []byte {
		raw := make([]byte, 16)
		for i, word := range []uint32{magic, site, index, count} {
			binary.LittleEndian.PutUint32(raw[4*i:], word)
		}
		return raw
	}
	for _, site := range []uint32{V41F32FiniteRoutedSum, V41F32FiniteFinalNormRaw} {
		for _, index := range []uint32{math.MaxUint32, 0, 64} {
			raw := record(v41F32FiniteMagic, site, index, 65)
			err := v41DeferredVerdict("fixture", V41DeferredStatus{Kind: V41DeferredF32Finite, Count: 65, Site: site}, raw)
			if index == math.MaxUint32 {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				var numeric *V41F32FiniteArithmeticError
				if !errors.As(err, &numeric) || numeric.Site != site || numeric.Index != int(index) {
					t.Fatalf("raw fault identity lost: %v", err)
				}
			}
		}
	}
	for _, raw := range [][]byte{
		nil,
		record(math.MaxUint32, math.MaxUint32, math.MaxUint32, math.MaxUint32),
		record(v41BF16CheckedStatusMagic, 1, math.MaxUint32, 65),
		record(v41F32FiniteMagic, 2, math.MaxUint32, 65),
		record(v41F32FiniteMagic, 1, math.MaxUint32, 64),
		record(v41F32FiniteMagic, 1, 65, 65),
	} {
		var protocol *V41F32FiniteProtocolError
		if !errors.As(decodeV41F32FiniteStatus(raw, 65, 1), &protocol) {
			t.Fatal("malformed status was accepted")
		}
	}
	if decodeV41F32FiniteStatus(record(v41F32FiniteMagic, 3, math.MaxUint32, 65), 65, 3) == nil {
		t.Fatal("unknown raw-F32 site was accepted")
	}
	// The source-bound bit classifier is independent of BF16 publication.
	// This is not a GLSL execution or hardware parity witness.
	raw, err := os.ReadFile("shaders/v41_f32_finite.comp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	for _, term := range []string{
		"readonly buffer Input { uint inputBits[]; };",
		"if (index == 0u)",
		"status[0] = 0x46333243u;",
		"status[3] = pc.count;",
		"for (uint i = index; i < pc.count; i += stride)",
		"(inputBits[i] & 0x7f800000u) == 0x7f800000u",
		"atomicMin(status[2], i);",
	} {
		if !strings.Contains(source, term) {
			t.Fatalf("raw-F32 source contract changed: %s", term)
		}
	}
	for _, bits := range []uint32{0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001} {
		got := bits&0x7f800000 == 0x7f800000
		value := float64(math.Float32frombits(bits))
		if got != (math.IsInf(value, 0) || math.IsNaN(value)) {
			t.Fatalf("raw classifier differs for %08x", bits)
		}
	}
	if float32(math.MaxFloat32) == float32(math.Inf(1)) {
		t.Fatal("finite-extreme control is invalid")
	}
}
