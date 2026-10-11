//go:build darwin && arm64 && cgo

// fak-test:runtime slow est=20s

package metalgemm

import (
	"math/rand"
	"os"
	"testing"
)

// decodeGEMVBandwidth encodes reps independent P=1 projections into one decode graph and returns
// the GPU milliseconds and the weight bytes streamed. Distinct weights defeat the SLC so each
// dispatch reads its bytes from DRAM, as the resident decode does.
func decodeGEMVBandwidth(t *testing.T, in int, encode func(g *ProjectionGraph, x *GraphResult, q *QuantizedGraphResult) (*GraphResult, error)) float64 {
	t.Helper()
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(i%17) * 0.01
	}
	best := 0.0
	for trial := 0; trial < 5; trial++ {
		g, err := BeginProjectionGraph(x, nil, nil, 1, in)
		if err != nil {
			t.Fatal(err)
		}
		if !g.SetGEMVDecode() {
			t.Fatal("GEMV decode route unavailable")
		}
		input, err := g.Input(in)
		if err != nil {
			t.Fatal(err)
		}
		q, err := g.QuantizeQ8(input)
		if err != nil {
			t.Fatal(err)
		}
		last, err := encode(g, input, q)
		if err != nil {
			t.Fatal(err)
		}
		_, receipt, err := g.FinishRead(last)
		g.Free()
		if err != nil || !receipt.TimingAvailable {
			t.Fatalf("finish: %v timing=%v", err, receipt.TimingAvailable)
		}
		if trial == 0 || receipt.GPUMilliseconds < best {
			best = receipt.GPUMilliseconds
		}
	}
	return best
}

// TestDecodeGEMVBandwidthQwen35Shapes reports streamed GB/s for the P=1 Q8, Q4_K and Q6_K decode
// GEMVs at Qwen3.5-4B linear-attention/MLP shapes. It is a measurement, not a gate.
func TestDecodeGEMVBandwidthQwen35Shapes(t *testing.T) {
	if os.Getenv("FAK_DECODE_GEMV_BANDWIDTH") == "" {
		t.Skip("set FAK_DECODE_GEMV_BANDWIDTH=1 to measure decode GEMV bandwidth")
	}
	if !Available() {
		t.Skip("no Metal device")
	}
	const reps = 24
	rng := rand.New(rand.NewSource(1))
	shapes := []struct{ out, in int }{{8192, 2560}, {4096, 2560}, {2560, 4096}, {9216, 2560}, {2560, 9216}}
	for _, s := range shapes {
		nblk := s.in / 32
		var q8 []*Q8Weight
		for r := 0; r < reps; r++ {
			codes := make([]int8, s.out*s.in)
			for i := range codes {
				codes[i] = int8(rng.Intn(255) - 127)
			}
			scales := make([]float32, s.out*nblk)
			for i := range scales {
				scales[i] = 0.001
			}
			w := UploadQ8(codes, scales, s.out, s.in)
			if w == nil {
				t.Fatal("UploadQ8 failed")
			}
			q8 = append(q8, w)
		}
		ms := decodeGEMVBandwidth(t, s.in, func(g *ProjectionGraph, x *GraphResult, q *QuantizedGraphResult) (*GraphResult, error) {
			var last *GraphResult
			for _, w := range q8 {
				r, err := g.EncodeQ8From(w, q)
				if err != nil {
					return nil, err
				}
				last = r
			}
			return last, nil
		})
		bytes := float64(reps) * float64(s.out) * (float64(s.in) + float64(nblk)*4)
		t.Logf("Q8   %5dx%-5d reps=%d gpu=%.3fms  %.1f GB/s", s.out, s.in, reps, ms, bytes/ms/1e6)
		for _, w := range q8 {
			w.Release()
		}

		var q4 []*Q4KWeight
		rowBytes := s.in / 256 * 144
		for r := 0; r < reps; r++ {
			raw := make([]byte, s.out*rowBytes)
			rng.Read(raw)
			for b := 0; b < len(raw); b += 144 {
				raw[b], raw[b+1], raw[b+2], raw[b+3] = 0x00, 0x14, 0x00, 0x14 // finite half d/dmin
			}
			w := UploadQ4K(raw, s.out, s.in)
			if w == nil {
				t.Fatal("UploadQ4K failed")
			}
			q4 = append(q4, w)
		}
		ms = decodeGEMVBandwidth(t, s.in, func(g *ProjectionGraph, x *GraphResult, _ *QuantizedGraphResult) (*GraphResult, error) {
			var last *GraphResult
			for _, w := range q4 {
				r, err := g.EncodeQ4KFrom(w, x)
				if err != nil {
					return nil, err
				}
				last = r
			}
			return last, nil
		})
		bytes = float64(reps) * float64(s.out*rowBytes)
		t.Logf("Q4_K %5dx%-5d reps=%d gpu=%.3fms  %.1f GB/s", s.out, s.in, reps, ms, bytes/ms/1e6)
		for _, w := range q4 {
			w.Release()
		}
	}
}
