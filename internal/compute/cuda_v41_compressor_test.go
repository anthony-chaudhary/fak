//go:build cuda && cgo && fak_cuda_v41_compressor && !fak_cuda_v41_compressor_pool

package compute

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// Opt-in physical witness for the CUDA compressor ring/pool (ported ds4
// kernels). Build the library with -DFAK_CUDA_V41_COMPRESSOR and run with
// -tags 'cuda fak_cuda_v41_compressor'. The pooled row is pinned to the binary64
// oracle; the ring's emitted row is bit-identical to the pool on this backend.

func cudaV41CompressorUpload(t *testing.T, be *cudaBackend, shape []int, x []float32) Tensor {
	t.Helper()
	d := mkResident(be, shape, x)
	t.Cleanup(func() { be.Free(d) })
	return d
}

func cudaV41CompressorGroup(seed int64, ratio, width int, scoreScale float64) (kv, scores []float32) {
	rng := rand.New(rand.NewSource(seed))
	kv, scores = make([]float32, ratio*width), make([]float32, ratio*width)
	for i := range kv {
		kv[i] = float32(rng.NormFloat64())
		scores[i] = float32(rng.NormFloat64() * scoreScale)
	}
	return kv, scores
}

// cudaV41CompressorWithinOracle bounds each pooled value by the serial
// accumulation error of a convex combination: (ratio+8) binary32 ulps of the
// dimension's largest |kv|.
func cudaV41CompressorWithinOracle(t *testing.T, label string, got, kv, scores []float32, ratio, width int) float64 {
	t.Helper()
	want := v41CompressorPoolFloat64Oracle(kv, scores, ratio, width)
	worst := 0.0
	for d := 0; d < width; d++ {
		maxAbs := 0.0
		for r := 0; r < ratio; r++ {
			maxAbs = math.Max(maxAbs, math.Abs(float64(kv[r*width+d])))
		}
		diff := math.Abs(float64(got[d]) - want[d])
		bound := float64(ratio+8) * 0x1p-23 * maxAbs
		if diff > bound {
			t.Fatalf("%s: dim %d got %g want %g |diff| %g > bound %g", label, d, got[d], want[d], diff, bound)
		}
		if maxAbs > 0 {
			worst = math.Max(worst, diff/(0x1p-23*maxAbs))
		}
	}
	return worst
}

func cudaV41CompressorPool(t *testing.T, be *cudaBackend, kv, scores []float32, ratio, width int) []float32 {
	t.Helper()
	before := cudaV41CompressorCompleted()
	out, err := be.V41CompressorPool(cudaV41CompressorUpload(t, be, []int{ratio, width}, kv),
		cudaV41CompressorUpload(t, be, []int{ratio, width}, scores), ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(out)
	if cudaV41CompressorCompleted() != before+1 {
		t.Fatal("no completed native compressor witness")
	}
	return append([]float32(nil), be.Read(out)...)
}

// cudaV41CompressorFill stores every group row in order, emitting on the last.
func cudaV41CompressorFill(t *testing.T, be *cudaBackend, ring [2]Tensor, kv, scores []float32, ratio, width int, order []int) ([]float32, error) {
	t.Helper()
	var out Tensor
	for i, slot := range order {
		emit := i == len(order)-1
		res, err := be.V41CompressorRingStore(ring[0], ring[1],
			cudaV41CompressorUpload(t, be, []int{width}, kv[slot*width:(slot+1)*width]),
			cudaV41CompressorUpload(t, be, []int{width}, scores[slot*width:(slot+1)*width]), slot, ratio, width, emit)
		if err != nil {
			t.Fatalf("store slot %d: %v", slot, err)
		}
		if !emit && res.Buf() != nil {
			t.Fatalf("non-emitting store slot %d returned a tensor", slot)
		}
		out = res
	}
	if out.Buf() == nil || len(out.Shape) != 2 || out.Shape[0] != 2 || out.Shape[1] != width {
		t.Fatalf("emit returned %v", out.Shape)
	}
	defer be.Free(out)
	return DecodeV41CompressorRing(append([]float32(nil), be.Read(out)...), width)
}

func cudaV41CompressorBitsEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: element %d bits %#08x, want %#08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// fak-test:runtime integration est=3s lane=optin
func TestCUDAV41CompressorPoolMatchesOracle(t *testing.T) {
	be := cudaGDNBackend(t)
	if !be.SupportsV41CompressorPool() || !be.SupportsV41CompressorRing() {
		t.Fatal("candidate built but native admission refused")
	}
	for _, geom := range [][2]int{{2, 7}, {4, 512}, {4, 1024}, {128, 512}, {300, 65}, {V41CompressorPoolMaxRatio, 33}} {
		for _, scoreScale := range []float64{1, 40} {
			ratio, width := geom[0], geom[1]
			t.Run(fmt.Sprintf("ratio%d_width%d_scores%g", ratio, width, scoreScale), func(t *testing.T) {
				kv, scores := cudaV41CompressorGroup(int64(ratio*1009+width), ratio, width, scoreScale)
				got := cudaV41CompressorPool(t, be, kv, scores, ratio, width)
				ulps := cudaV41CompressorWithinOracle(t, "pool", got, kv, scores, ratio, width)
				t.Logf("V41C_PARITY ratio=%d width=%d scores=%g worst_ulps_of_max_kv=%.2f", ratio, width, scoreScale, ulps)
			})
		}
	}
}

// fak-test:runtime integration est=3s lane=optin
// Emit and no-emit slots: only the group-closing store returns a row, the ring
// holds the stored bits, any store order and ring reuse pool identically.
func TestCUDAV41CompressorRingMatchesPool(t *testing.T) {
	be := cudaGDNBackend(t)
	for _, geom := range [][2]int{{2, 7}, {4, 512}, {128, 512}} {
		for _, scoreScale := range []float64{1, 40} {
			ratio, width := geom[0], geom[1]
			t.Run(fmt.Sprintf("ratio%d_width%d_scores%g", ratio, width, scoreScale), func(t *testing.T) {
				kvRing, scoreRing, err := be.V41CompressorRingAlloc(ratio, width)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { be.Free(kvRing); be.Free(scoreRing) })
				inOrder, reversed := make([]int, ratio), make([]int, ratio)
				for i := range inOrder {
					inOrder[i], reversed[i] = i, ratio-1-i
				}
				for group, order := range [][]int{inOrder, reversed, inOrder} {
					kv, scores := cudaV41CompressorGroup(int64(group*7919+ratio*31+width), ratio, width, scoreScale)
					got, err := cudaV41CompressorFill(t, be, [2]Tensor{kvRing, scoreRing}, kv, scores, ratio, width, order)
					if err != nil {
						t.Fatalf("group %d: %v", group, err)
					}
					cudaV41CompressorWithinOracle(t, fmt.Sprintf("group %d ring", group), got, kv, scores, ratio, width)
					cudaV41CompressorBitsEqual(t, fmt.Sprintf("group %d pooled", group), got, cudaV41CompressorPool(t, be, kv, scores, ratio, width))
					cudaV41CompressorBitsEqual(t, fmt.Sprintf("group %d kv ring", group), append([]float32(nil), be.Read(kvRing)...), kv)
					cudaV41CompressorBitsEqual(t, fmt.Sprintf("group %d score ring", group), append([]float32(nil), be.Read(scoreRing)...), scores)
				}
			})
		}
	}
}

// fak-test:runtime integration est=2s lane=optin
// A non-finite kv or score anywhere in the group refuses the pooled row with
// the typed sentinel, including a -Inf score whose softmax weight is zero.
func TestCUDAV41CompressorRefusesNonFinite(t *testing.T) {
	be := cudaGDNBackend(t)
	const ratio, width = 2, 70
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	for _, tc := range []struct {
		name   string
		slot   int
		dim    int
		kv     bool
		value  float32
		status int
	}{
		{"kv +inf first row", 0, 3, true, inf, V41CompressorRingKVNonFinite},
		{"score -inf emitting row", 1, 69, false, -inf, V41CompressorRingScoreNonFinite},
		{"score nan first row", 0, 64, false, nan, V41CompressorRingScoreNonFinite},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kv, scores := make([]float32, ratio*width), make([]float32, ratio*width)
			for i := range kv {
				kv[i], scores[i] = float32(i%5)-2, float32(i%3)
			}
			if tc.kv {
				kv[tc.slot*width+tc.dim] = tc.value
			} else {
				scores[tc.slot*width+tc.dim] = tc.value
			}
			kvRing, scoreRing, err := be.V41CompressorRingAlloc(ratio, width)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { be.Free(kvRing); be.Free(scoreRing) }()
			_, err = cudaV41CompressorFill(t, be, [2]Tensor{kvRing, scoreRing}, kv, scores, ratio, width, []int{0, 1})
			var typed *V41CompressorRingNonFiniteError
			if !errors.Is(err, ErrV41CompressorRingNonFinite) || !errors.As(err, &typed) || typed.Dim != tc.dim || typed.Status != tc.status {
				t.Fatalf("ring: got %v, want dim %d status %d", err, tc.dim, tc.status)
			}
			_, err = be.V41CompressorPool(cudaV41CompressorUpload(t, be, []int{ratio, width}, kv),
				cudaV41CompressorUpload(t, be, []int{ratio, width}, scores), ratio, width)
			typed = nil
			if !errors.Is(err, ErrV41CompressorRingNonFinite) || !errors.As(err, &typed) || typed.Dim != tc.dim || typed.Status != tc.status {
				t.Fatalf("pool: got %v, want dim %d status %d", err, tc.dim, tc.status)
			}
			if !be.SupportsV41CompressorRing() {
				t.Fatal("a refused group must not close admission")
			}
		})
	}
}

// fak-test:runtime integration est=1s lane=optin
func TestCUDAV41CompressorRejects(t *testing.T) {
	be := cudaGDNBackend(t)
	kvRing, scoreRing, err := be.V41CompressorRingAlloc(2, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { be.Free(kvRing); be.Free(scoreRing) }()
	row := cudaV41CompressorUpload(t, be, []int{4}, make([]float32, 4))
	for name, call := range map[string]func() error{
		"ratio one alloc": func() error { _, _, err := be.V41CompressorRingAlloc(1, 4); return err },
		"slot range": func() error {
			_, err := be.V41CompressorRingStore(kvRing, scoreRing, row, row, 2, 2, 4, false)
			return err
		},
		"aliased rings": func() error {
			_, err := be.V41CompressorRingStore(kvRing, kvRing, row, row, 0, 2, 4, false)
			return err
		},
		"host operand": func() error {
			_, err := be.V41CompressorRingStore(kvRing, scoreRing, NewF32(Default(), []int{4}, make([]float32, 4)), row, 0, 2, 4, false)
			return err
		},
		"row aliases ring": func() error {
			_, err := be.V41CompressorRingStore(kvRing, scoreRing, Tensor{be: be, buf: kvRing.buf, Dtype: F32, Layout: RowMajor, Shape: []int{4}}, row, 0, 2, 4, false)
			return err
		},
		"pool ratio one": func() error { _, err := be.V41CompressorPool(row, row, 1, 4); return err },
	} {
		if err := call(); !errors.Is(err, ErrV41CompressorRingGeometry) {
			t.Errorf("%s: got %v, want ErrV41CompressorRingGeometry", name, err)
		}
	}
	if !be.SupportsV41CompressorRing() {
		t.Fatal("refusals must not close admission")
	}
}

// Per-group timing at V4.1 compressor geometry: wall ms of ratio stores plus
// the emitting store, and of one pool over an uploaded group.
// fak-test:runtime integration est=5s lane=optin
func TestCUDAV41CompressorTiming(t *testing.T) {
	be := cudaGDNBackend(t)
	for _, geom := range [][2]int{{4, 1024}, {128, 512}} {
		const iters = 20
		ratio, width := geom[0], geom[1]
		kv, scores := cudaV41CompressorGroup(5, ratio, width, 1)
		kvRing, scoreRing, err := be.V41CompressorRingAlloc(ratio, width)
		if err != nil {
			t.Fatal(err)
		}
		rows := make([]Tensor, 2*ratio)
		for i := 0; i < ratio; i++ {
			rows[2*i] = cudaV41CompressorUpload(t, be, []int{width}, kv[i*width:(i+1)*width])
			rows[2*i+1] = cudaV41CompressorUpload(t, be, []int{width}, scores[i*width:(i+1)*width])
		}
		kt := cudaV41CompressorUpload(t, be, []int{ratio, width}, kv)
		st := cudaV41CompressorUpload(t, be, []int{ratio, width}, scores)
		var ringWall, poolWall time.Duration
		for i := 0; i < iters+3; i++ {
			start := time.Now()
			var out Tensor
			for slot := 0; slot < ratio; slot++ {
				if out, err = be.V41CompressorRingStore(kvRing, scoreRing, rows[2*slot], rows[2*slot+1], slot, ratio, width, slot == ratio-1); err != nil {
					t.Fatal(err)
				}
			}
			mid := time.Now()
			pooled, err := be.V41CompressorPool(kt, st, ratio, width)
			if err != nil {
				t.Fatal(err)
			}
			end := time.Now()
			be.Free(out)
			be.Free(pooled)
			if i >= 3 {
				ringWall += mid.Sub(start)
				poolWall += end.Sub(mid)
			}
		}
		hostStart := time.Now()
		_ = v41CompressorPoolFloat64Oracle(kv, scores, ratio, width)
		host := time.Since(hostStart)
		t.Logf("V41C_RECEIPT ratio=%d width=%d ring_group_wall_ms=%.4f pool_wall_ms=%.4f host_oracle_ms=%.3f",
			ratio, width, float64(ringWall.Microseconds())/1000/iters, float64(poolWall.Microseconds())/1000/iters,
			float64(host.Microseconds())/1000)
		be.Free(kvRing)
		be.Free(scoreRing)
	}
}
