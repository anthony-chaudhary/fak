//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

// The one-token MatMul routes F32 and Q6_K to the decode layouts of matmul.comp
// and q6k_matmul.comp; BatchedMatMul with two tokens keeps the per-row layouts.
// Both must produce the same bits for every row, because the decode layouts only
// move where each serial sum runs, never its operation order.
// fak-test:runtime fast est=2s lane=default
func TestVulkanDecodeLayoutsBitIdenticalToPanelRows(t *testing.T) {
	v := vk(t)
	full := os.Getenv("FAK_VULKAN_DECODE_SPLIT_FULL") == "1"
	type shape struct {
		name    string
		dtype   Dtype
		out, in int
	}
	shapes := []shape{
		{"f32_mhc_mix_4096", F32, 24, 16384},
		{"f32_mhc_mix_5120", F32, 24, 20480},
		{"f32_query_prefix", F32, 1024, 4096},
		{"f32_partial_chunk", F32, 131, 5000},
		{"f32_one_chunk_wide", F32, 3000, 576},
		{"f32_single", F32, 1, 7},
		{"q6k_tail_rows", Q6_K, 129, 4096},
		{"q6k_one_block", Q6_K, 1, 256},
	}
	if full {
		shapes = append(shapes, shape{"q6k_head_4096", Q6_K, 129280, 4096})
	}
	for _, sh := range shapes {
		t.Run(sh.name, func(t *testing.T) {
			if sh.dtype == Q6_K && !v.SupportsQ6KMatMul() {
				t.Fatal("registered Vulkan backend lacks q6k_matmul")
			}
			var weight Tensor
			if sh.dtype == Q6_K {
				weight = v.Upload(NewQ6K(Default(), []int{sh.out, sh.in}, q6KStrixFixture(sh.out, sh.in)), Q6_K)
			} else {
				w := make([]float32, sh.out*sh.in)
				for i := range w {
					w[i] = decodeSplitValue(i, 7919)
				}
				weight = v.Upload(NewF32(Default(), []int{sh.out, sh.in}, w), F32)
			}
			defer v.Free(weight)
			x := make([]float32, sh.in)
			for i := range x {
				x[i] = decodeSplitValue(i, 104729)
			}
			panelX := append(append(make([]float32, 0, 2*sh.in), x...), x...)
			dx := v.UploadClass(NewF32(Default(), []int{sh.in}, x), F32, MemoryActivation, "decode layout input")
			defer v.Free(dx)
			dp := v.UploadClass(NewF32(Default(), []int{2, sh.in}, panelX), F32, MemoryActivation, "panel layout input")
			defer v.Free(dp)
			decodeOut := v.MatMul(weight, dx)
			defer v.Free(decodeOut)
			panelOut := v.BatchedMatMul(weight, dp, 2)
			defer v.Free(panelOut)
			got, panel := v.Read(decodeOut), v.Read(panelOut)
			t.Logf("DECODE_DIGEST %s %s", sh.name, decodeSplitDigest(got))
			if len(got) != sh.out || len(panel) != 2*sh.out {
				t.Fatalf("lengths decode=%d panel=%d, want %d and %d", len(got), len(panel), sh.out, 2*sh.out)
			}
			for row := 0; row < 2; row++ {
				for o := 0; o < sh.out; o++ {
					if math.Float32bits(got[o]) != math.Float32bits(panel[row*sh.out+o]) {
						t.Fatalf("output %d: decode %v (%#08x) != panel row %d %v (%#08x)", o, got[o], math.Float32bits(got[o]), row, panel[row*sh.out+o], math.Float32bits(panel[row*sh.out+o]))
					}
				}
			}
		})
	}
}

// decodeSplitValue is a deterministic mixed-sign value in (-1, 1) with a
// nontrivial mantissa, so reordered sums would differ in their low bits.
func decodeSplitValue(i, salt int) float32 {
	h := uint32(i)*2654435761 + uint32(salt)*40503
	h ^= h >> 15
	h *= 2246822519
	h ^= h >> 13
	return float32(int32(h>>8)-(1<<23)) / float32(1<<23)
}

func decodeSplitDigest(values []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, v := range values {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		h.Write(b[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:8])
}

// The rms-split digest is compared across shader builds: a reduction that only
// changes which lanes load the squares must leave every coefficient bit equal.
// fak-test:runtime fast est=20ms lane=default
func TestVulkanV41MHCRMSSplitDigest(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41MHC() {
		t.Fatal("registered Vulkan backend lacks v41_mhc")
	}
	for _, geom := range [][2]int{{1, 4096}, {1, 5120}, {3, 4096}, {2, 700}} {
		rows, hidden := geom[0], geom[1]
		raw := make([]float32, rows*V41MHCCoeffWidth)
		for i := range raw {
			raw[i] = decodeSplitValue(i, 31) * 40
		}
		streams := make([]float32, rows*4*hidden)
		for i := range streams {
			streams[i] = decodeSplitValue(i, 977) * 3
		}
		params := make([]float32, V41MHCParamWidth)
		for i := range params {
			params[i] = decodeSplitValue(i, 5) * 0.5
		}
		dr := v.UploadClass(NewF32(Default(), []int{rows, V41MHCCoeffWidth}, raw), F32, MemoryActivation, "rms-split raw")
		ds := v.UploadClass(NewF32(Default(), []int{rows, 4, hidden}, streams), F32, MemoryActivation, "rms-split streams")
		dp := v.UploadClass(NewF32(Default(), []int{V41MHCParamWidth}, params), F32, MemoryActivation, "rms-split params")
		out, err := v.V41MHCRMSSplit(dr, ds, dp, rows, hidden, 20, 1e-6, 1e-6)
		if err != nil {
			t.Fatal(err)
		}
		got := v.Read(out)
		t.Logf("RMS_SPLIT_DIGEST rows%d_hidden%d %s", rows, hidden, decodeSplitDigest(got))
		for _, tensor := range []Tensor{dr, ds, dp, out} {
			v.Free(tensor)
		}
	}
}

// Opt-in microbenchmark: run with FAK_VULKAN_DECODE_SPLIT_BENCH=1 and
// FAK_VULKAN_TIMESTAMP_PROFILE=1 FAK_VULKAN_TIMESTAMP_PROFILE_EVERY=1 set before
// process start; the per-kernel device time is the native ts-stage report.
// fak-test:runtime fast est=3s lane=optin
func TestVulkanDecodeLayoutsTimestampBench(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DECODE_SPLIT_BENCH") != "1" {
		t.Skip("decode layout timestamp bench is opt-in")
	}
	v := vk(t)
	const calls = 16
	matmul := func(name string, w Tensor, in int) {
		x := v.UploadClass(NewF32(Default(), []int{in}, make([]float32, in)), F32, MemoryActivation, name+" input")
		defer v.Free(x)
		v.BeginBatch()
		outs := make([]Tensor, 0, calls)
		for i := 0; i < calls; i++ {
			outs = append(outs, v.MatMul(w, x))
		}
		v.FlushBatch()
		for _, out := range outs {
			v.Free(out)
		}
	}
	for _, sh := range [][2]int{{24, 16384}, {24, 20480}, {1024, 4096}} {
		w := v.Upload(NewF32(Default(), []int{sh[0], sh[1]}, make([]float32, sh[0]*sh[1])), F32)
		matmul(fmt.Sprintf("f32_%dx%d", sh[0], sh[1]), w, sh[1])
		v.Free(w)
	}
	head := v.Upload(NewQ6K(Default(), []int{129280, 4096}, q6KStrixFixture(129280, 4096)), Q6_K)
	matmul("q6k_head", head, 4096)
	v.Free(head)
	for _, hidden := range []int{4096, 5120} {
		dr := v.UploadClass(NewF32(Default(), []int{1, V41MHCCoeffWidth}, make([]float32, V41MHCCoeffWidth)), F32, MemoryActivation, "bench raw")
		streams := make([]float32, 4*hidden)
		for i := range streams {
			streams[i] = decodeSplitValue(i, 3)
		}
		ds := v.UploadClass(NewF32(Default(), []int{1, 4, hidden}, streams), F32, MemoryActivation, "bench streams")
		dp := v.UploadClass(NewF32(Default(), []int{V41MHCParamWidth}, make([]float32, V41MHCParamWidth)), F32, MemoryActivation, "bench params")
		v.BeginBatch()
		outs := make([]Tensor, 0, calls)
		for i := 0; i < calls; i++ {
			out, err := v.V41MHCRMSSplit(dr, ds, dp, 1, hidden, 20, 1e-6, 1e-6)
			if err != nil {
				t.Fatal(err)
			}
			outs = append(outs, out)
		}
		v.FlushBatch()
		for _, tensor := range append(outs, dr, ds, dp) {
			v.Free(tensor)
		}
	}
}
