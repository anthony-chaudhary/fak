//go:build darwin && arm64 && cgo && metal

package metalgemm

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// fak-test:runtime medium est=5s
func TestAttentionMLXDirectOracle(t *testing.T) {
	t.Setenv("FAK_QWEN35_ATTN_SPLIT", "0")
	lease, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if !Available() {
		t.Fatal("physical Metal device required")
	}
	owners, buffers := graphLiveOwnerCount(), graphLiveBufferCount()
	for _, hd := range []int{64, 128, 256} {
		for _, c := range []struct {
			rows, base, heads, kv int
			extreme               bool
		}{{1, 0, 2, 2, false}, {1, 127, 4, 1, false}, {4, 7, 4, 2, false}, {3, 29, 2, 1, true}, {1, 4095, 2, 1, false}, {4, 4092, 2, 1, false}} {
			t.Run(fmt.Sprintf("d%d_p%d_b%d_g%d_extreme%v", hd, c.rows, c.base, c.heads/c.kv, c.extreme), func(t *testing.T) {
				f := newMLXAttentionFixture(t, hd, c.rows, c.base, c.heads, c.kv, c.extreme)
				defer f.close()
				want := f.oracle()
				for _, device := range []bool{false, true} {
					for _, selector := range []string{"0", "1", ""} {
						t.Setenv("FAK_QWEN35_ATTN_MLX", selector)
						got, receipt := f.run(t, device, 1)
						mlxAttentionAssert(t, got, want)
						if !receipt.Committed || !receipt.CompletedWait {
							t.Fatalf("missing physical completion: %+v", receipt)
						}
					}
				}
				if c.rows > 1 {
					// Mutate only the final panel row's K/V. Earlier causal rows must be invariant.
					first, _ := f.run(t, false, 1)
					last := (f.rows - 1) * f.in
					for i := 2 * f.qw; i < 2*f.qw+2*f.kw; i++ {
						f.codes[last+i] = int8(100 - i%7)
					}
					changed, _ := f.run(t, false, 1)
					mlxAttentionAssert(t, changed[:(f.rows-1)*f.qw], first[:(f.rows-1)*f.qw])
				}
			})
		}
	}

	for _, hd := range []int{32, 96} {
		t.Run(fmt.Sprintf("default_fallback_hd%d", hd), func(t *testing.T) {
			f := newMLXAttentionFixture(t, hd, 4, 7, 4, 2, false)
			defer f.close()
			for _, selector := range []string{"0", "1", ""} {
				t.Setenv("FAK_QWEN35_ATTN_MLX", selector)
				for _, device := range []bool{false, true} {
					got, _ := f.run(t, device, 1)
					mlxAttentionAssert(t, got, f.oracle())
				}
			}
		})
	}
	if graphLiveOwnerCount() != owners || graphLiveBufferCount() != buffers {
		t.Fatal("attention fixtures leaked native graph resources")
	}
}

// fak-test:runtime medium est=10s
func TestAttentionMLXDirectTiming(t *testing.T) {
	lease, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if !Available() {
		t.Fatal("physical Metal device required")
	}
	for _, hd := range []int{64, 128, 256} {
		for _, rows := range []int{1, 16} {
			f := newMLXAttentionFixture(t, hd, rows, 512, 8, 2, false)
			var timing [2]float64
			for s, selector := range []string{"0", "1"} {
				t.Setenv("FAK_QWEN35_ATTN_MLX", selector)
				f.run(t, true, 2) // Warm the pipeline and caches outside the measurement.
				for sample := 0; sample < 5; sample++ {
					_, r := f.run(t, true, 32)
					if !r.TimingAvailable || r.GPUMilliseconds <= 0 {
						t.Fatalf("no GPU timestamp: %+v", r)
					}
					if r.RetainedBufferBytes > 64<<20 {
						t.Fatalf("unbounded retained memory: %+v", r)
					}
					timing[s] += r.GPUMilliseconds / 32 / 5
				}
			}
			t.Logf("physical GPU p=%d hd=%d gqa=4 base=512 samples=5 attention_calls=32 baseline_ms=%.6f mlx_ms=%.6f speedup=%.3f", rows, hd, timing[0], timing[1], timing[0]/timing[1])
			f.close()
		}
	}
}

type mlxAttentionFixture struct {
	rows, base, heads, kv, hd, qw, kw, in int
	codes                                 []int8
	scales, pk, pv                        []float32
	weights                               [4]*Q8Weight
}

func newMLXAttentionFixture(t *testing.T, hd, rows, base, heads, kv int, extreme bool) *mlxAttentionFixture {
	t.Helper()
	f := &mlxAttentionFixture{rows: rows, base: base, heads: heads, kv: kv, hd: hd, qw: heads * hd, kw: kv * hd}
	f.in = 2*f.qw + 2*f.kw
	f.codes = make([]int8, rows*f.in)
	f.scales = make([]float32, rows*f.in/32)
	unit := float32(.125)
	if extreme {
		unit = 8
	}
	for i := range f.scales {
		f.scales[i] = unit
	}
	for i := range f.codes {
		f.codes[i] = int8((i*13+i/f.in*7)%23 - 11)
	}
	offsets := []int{0, f.qw, 2 * f.qw, 2*f.qw + f.kw}
	widths := []int{f.qw, f.qw, f.kw, f.kw}
	for j, width := range widths {
		codes := make([]int8, width*f.in)
		scales := make([]float32, width*f.in/32)
		for i := range scales {
			scales[i] = 1
		}
		for i := 0; i < width; i++ {
			codes[i*f.in+offsets[j]+i] = 1
		}
		f.weights[j] = UploadQ8(codes, scales, width, f.in)
		if f.weights[j] == nil {
			f.close()
			t.Fatal("selector weight upload failed")
		}
	}
	f.pk = make([]float32, base*f.kw)
	f.pv = make([]float32, base*f.kw)
	for i := range f.pk {
		f.pk[i] = float32((i*5)%19-9) * unit
		f.pv[i] = float32((i*7)%17-8) * .0625
	}
	return f
}

func (f *mlxAttentionFixture) close() {
	for _, w := range f.weights {
		if w != nil {
			w.Release()
		}
	}
}

func (f *mlxAttentionFixture) run(t *testing.T, device bool, repeats int) ([]float32, GraphReceipt) {
	t.Helper()
	before := AttentionMLXDispatchCount()
	g, err := BeginProjectionGraph(nil, f.codes, f.scales, f.rows, f.in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	result := make([]*GraphResult, 4)
	for i, w := range f.weights {
		result[i], err = g.EncodeQ8(w)
		if err != nil {
			t.Fatal(err)
		}
	}
	norm := make([]float32, f.hd)
	for i := range norm {
		norm[i] = 1
	}
	cos := make([]float32, f.rows)
	sin := make([]float32, f.rows)
	for i := range cos {
		cos[i] = 1
	}
	var kv *DeviceKV
	if device {
		kv = NewDeviceKV(1, f.base+f.rows, f.kw)
		if kv == nil {
			t.Fatal("KV allocation failed")
		}
		defer kv.Close()
		if err = kv.Upload(1, f.pk); err != nil {
			t.Fatal(err)
		}
		if err = kv.Upload(2, f.pv); err != nil {
			t.Fatal(err)
		}
	}
	var a Qwen35GraphAttentionResult
	for i := 0; i < repeats; i++ {
		if device {
			a, err = g.FullAttentionDevice(result[0], result[2], result[3], result[1], kv, 0, norm, norm, cos, sin, f.base, f.heads, f.kv, f.hd, 2, 1/float32(math.Sqrt(float64(f.hd))), 1e-6, false, false)
		} else {
			a, err = g.FullAttention(result[0], result[2], result[3], result[1], norm, norm, cos, sin, f.pk, f.pv, f.base, f.heads, f.kv, f.hd, 2, 1/float32(math.Sqrt(float64(f.hd))), 1e-6, false, false)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	out, r, err := g.FinishRead(a.Output)
	if err != nil {
		t.Fatal(err)
	}
	after := AttentionMLXDispatchCount()
	wantDispatches := uint64(0)
	selector := os.Getenv("FAK_QWEN35_ATTN_MLX")
	if selector == "1" || selector == "" && (f.hd == 64 || f.hd == 128 || f.hd == 256) {
		wantDispatches = uint64(repeats)
	}
	if after-before != wantDispatches {
		t.Fatalf("physical candidate dispatches=%d want=%d", after-before, wantDispatches)
	}
	return out[0], r
}

// Independent float64 stable causal scaled-softmax oracle, without GPU readbacks.
func (f *mlxAttentionFixture) oracle() []float32 {
	out := make([]float32, f.rows*f.qw)
	current := func(row, off int) float64 {
		return float64(f.codes[row*f.in+off]) * float64(f.scales[row*f.in/32+off/32])
	}
	for row := 0; row < f.rows; row++ {
		for h := 0; h < f.heads; h++ {
			kh := h / (f.heads / f.kv)
			logits := make([]float64, f.base+row+1)
			max := math.Inf(-1)
			for token := range logits {
				sum := 0.0
				for d := 0; d < f.hd; d++ {
					var key float64
					if token < f.base {
						key = float64(f.pk[token*f.kw+kh*f.hd+d])
					} else {
						key = current(token-f.base, 2*f.qw+kh*f.hd+d)
					}
					sum += current(row, h*f.hd+d) * key
				}
				logits[token] = sum * float64(1/float32(math.Sqrt(float64(f.hd))))
				max = math.Max(max, logits[token])
			}
			denom := 0.0
			for i := range logits {
				logits[i] = math.Exp(logits[i] - max)
				denom += logits[i]
			}
			for d := 0; d < f.hd; d++ {
				value := 0.0
				for token, p := range logits {
					var v float64
					if token < f.base {
						v = float64(f.pv[token*f.kw+kh*f.hd+d])
					} else {
						v = current(token-f.base, 2*f.qw+f.kw+kh*f.hd+d)
					}
					value += p / denom * v
				}
				gate := current(row, f.qw+h*f.hd+d)
				out[row*f.qw+h*f.hd+d] = float32(value / (1 + math.Exp(-gate)))
			}
		}
	}
	return out
}

func mlxAttentionAssert(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape got=%d want=%d", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) || math.Abs(float64(g-w)) > 0.002+0.0005*math.Abs(float64(w)) {
			t.Fatalf("element %d got=%g want=%g", i, g, w)
		}
	}
}

// fak-test:runtime medium est=5s
func TestAttentionMLXDirectExistingGraphRegression(t *testing.T) {
	lease, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if !Available() {
		t.Fatal("physical Metal device required")
	}
	t.Setenv("FAK_QWEN35_ATTN_MLX", "")
	t.Run("ordered_panel_norm", TestProjectionGraphQwenOrderedPanelAttentionAndFinalNorm)
	t.Run("device_kv_parity", TestProjectionGraphQwenDeviceKVAttentionParity)
	t.Run("ordered_long_context", TestProjectionGraphQwenOrderedLongContextAttention)
	t.Run("split_kv_parity", TestProjectionGraphAttentionSplitKVParity)
	t.Run("batch_independent_kv", TestQwen35FullAttentionDecodeBatchIndependentKVSingleFence)
	t.Run("batch_long_context", TestQwen35FullAttentionDecodeBatchLongContextParity)
}
