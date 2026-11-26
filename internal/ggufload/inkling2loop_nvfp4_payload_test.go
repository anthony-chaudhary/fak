package ggufload

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// splitInkling2NVFP4Repack is the former load path: separate packed and scale
// buffers that compute.NewNVFP4 then copies into its own payload.
func splitInkling2NVFP4Repack(t *testing.T, raw []byte, rows, cols int) []byte {
	t.Helper()
	blocks := rows * (cols / qkNVFP4)
	const subs = qkNVFP4 / qkNVFP4Sub
	packed := make([]byte, blocks*qkNVFP4/2)
	scales := make([]byte, blocks*subs)
	if err := repackInkling2NVFP4Blocks(raw, packed, scales, 0, blocks); err != nil {
		t.Fatal(err)
	}
	w, err := compute.NewNVFP4(compute.Default(), []int{rows, cols}, packed, scales, 1)
	if err != nil {
		t.Fatal(err)
	}
	return hostNVFP4Bytes(w)
}

func hostNVFP4Bytes(w compute.Tensor) []byte {
	i8 := w.Buf().(compute.HostBuffer).I8()
	out := make([]byte, len(i8))
	for i, v := range i8 {
		out[i] = byte(v)
	}
	return out
}

func TestInkling2NVFP4PayloadMatchesSplitRepack(t *testing.T) {
	r := rand.New(rand.NewSource(19))
	for _, shape := range [][2]int{{1, 64}, {3, 1152}, {257, 2304}} {
		rows, cols := shape[0], shape[1]
		raw := randomInkling2NVFP4(r, rows, cols)
		raw[0], raw[1], raw[2] = 0x7F, 0x80, 0xBE
		want := splitInkling2NVFP4Repack(t, raw, rows, cols)
		payload, packedBytes, err := repackInkling2NVFP4Payload(raw, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		if packedBytes != rows*cols/2 {
			t.Fatalf("[%d %d] packed bytes %d", rows, cols, packedBytes)
		}
		w, err := compute.NewNVFP4Owned(compute.Default(), []int{rows, cols}, payload, 1)
		if err != nil {
			t.Fatal(err)
		}
		if got := hostNVFP4Bytes(w); !bytes.Equal(got, want) {
			t.Fatalf("[%d %d] owned payload differs from the split repack", rows, cols)
		}
	}
}

func TestInkling2NVFP4PayloadRefusesLikeSplitRepack(t *testing.T) {
	raw := make([]byte, blockNVFP4Bytes)
	raw[3] = 0xFF
	if _, _, err := repackInkling2NVFP4Payload(raw, 1, qkNVFP4); err == nil {
		t.Fatal("UE4M3 0xFF (480) repacked without error")
	}
	if _, _, err := repackInkling2NVFP4Payload(raw[:10], 1, qkNVFP4); err == nil {
		t.Fatal("short payload repacked without error")
	}
}
