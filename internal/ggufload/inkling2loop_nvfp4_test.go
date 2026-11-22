package ggufload

import (
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func randomInkling2NVFP4(r *rand.Rand, rows, cols int) []byte {
	raw := make([]byte, rows*cols/qkNVFP4*blockNVFP4Bytes)
	r.Read(raw)
	for b := 0; b < len(raw); b += blockNVFP4Bytes {
		for s := 0; s < qkNVFP4/qkNVFP4Sub; s++ {
			if raw[b+s] == 0xFF {
				raw[b+s] = 0x7F
			}
		}
	}
	return raw
}

func TestInkling2NVFP4RepackMatchesGGMLDequant(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, shape := range [][2]int{{1, 64}, {3, 1152}, {5, 2304}} {
		rows, cols := shape[0], shape[1]
		raw := randomInkling2NVFP4(r, rows, cols)
		// pin the special scale bytes so every run exercises them
		raw[0], raw[1], raw[2] = 0x7F, 0x80, 0xBE
		want := make([]float32, rows*cols)
		dequantNVFP4(want, raw)
		packed, scales, err := repackInkling2NVFP4(raw, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		got, err := compute.DecodeNVFP4Reference(rows, cols, packed, scales, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("[%d %d] value %d: fak %v, ggml %v", rows, cols, i, got[i], want[i])
			}
		}
		if _, err := compute.NewNVFP4(compute.Default(), []int{rows, cols}, packed, scales, 1); err != nil {
			t.Fatalf("NewNVFP4 refused the repack: %v", err)
		}
	}
}

func TestInkling2NVFP4RepackRefusesUnrepresentableScale(t *testing.T) {
	raw := make([]byte, blockNVFP4Bytes)
	raw[3] = 0xFF
	if _, _, err := repackInkling2NVFP4(raw, 1, qkNVFP4); err == nil {
		t.Fatal("UE4M3 0xFF (480) repacked without error")
	}
	if _, _, err := repackInkling2NVFP4(raw[:10], 1, qkNVFP4); err == nil {
		t.Fatal("short payload repacked without error")
	}
}
