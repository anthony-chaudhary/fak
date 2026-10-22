//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

func v41RingRequire(t *testing.T) *vulkanBackend {
	t.Helper()
	v := vk(t)
	if !v.SupportsV41CompressorRing() || !v.SupportsV41CompressorPool() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 compressor ring/pool modules or binary32 float controls")
		}
		t.Skip("V4.1 compressor ring module not loaded")
	}
	return v
}

func v41RingUpload(t *testing.T, v *vulkanBackend, shape []int, x []float32) Tensor {
	t.Helper()
	d := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "V4.1 compressor ring witness input")
	t.Cleanup(func() { v.Free(d) })
	return d
}

func v41RingAlloc(t *testing.T, v *vulkanBackend, ratio, width int) (Tensor, Tensor) {
	t.Helper()
	kv, scores, err := v.V41CompressorRingAlloc(ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { v.Free(kv); v.Free(scores) })
	return kv, scores
}

// v41RingFill stores every group row in slot order, emitting at emitSlot, and
// returns the decoded pooled row.
func v41RingFill(t *testing.T, v *vulkanBackend, ring [2]Tensor, kv, scores []float32, ratio, width int, order []int) ([]float32, error) {
	t.Helper()
	var out Tensor
	for i, slot := range order {
		emit := i == len(order)-1
		res, err := v.V41CompressorRingStore(ring[0], ring[1],
			v41RingUpload(t, v, []int{width}, kv[slot*width:(slot+1)*width]),
			v41RingUpload(t, v, []int{width}, scores[slot*width:(slot+1)*width]), slot, ratio, width, emit)
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
	defer v.Free(out)
	return DecodeV41CompressorRing(append([]float32(nil), v.Read(out)...), width)
}

func v41RingPool(t *testing.T, v *vulkanBackend, kv, scores []float32, ratio, width int) []float32 {
	t.Helper()
	out, err := v.V41CompressorPool(v41RingUpload(t, v, []int{ratio, width}, kv), v41RingUpload(t, v, []int{ratio, width}, scores), ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Free(out)
	return append([]float32(nil), v.Read(out)...)
}

func v41RingBitsEqual(t *testing.T, label string, got, want []float32) {
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
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no physical
// witness. The ring's emitted row must be bit-identical to V41CompressorPool on
// the same group, for any store order and after the ring is reused for a later
// group; the ring rows themselves must hold the stored bits.
func TestVulkanV41CompressorRingMatchesPool(t *testing.T) {
	v := v41RingRequire(t)
	rng := rand.New(rand.NewSource(20261022))
	for _, geom := range [][2]int{{2, 7}, {2, 512}, {4, 512}, {128, 512}} {
		for _, scoreScale := range []float64{1, 40} {
			ratio, width := geom[0], geom[1]
			t.Run(fmt.Sprintf("ratio%d_width%d_scores%g", ratio, width, scoreScale), func(t *testing.T) {
				kvRing, scoreRing := v41RingAlloc(t, v, ratio, width)
				ring := [2]Tensor{kvRing, scoreRing}
				inOrder := make([]int, ratio)
				for i := range inOrder {
					inOrder[i] = i
				}
				reversed := make([]int, ratio)
				for i := range reversed {
					reversed[i] = ratio - 1 - i
				}
				for group, order := range [][]int{inOrder, reversed, inOrder} {
					kv, scores := make([]float32, ratio*width), make([]float32, ratio*width)
					for i := range kv {
						kv[i] = float32(rng.NormFloat64())
						scores[i] = float32(rng.NormFloat64() * scoreScale)
					}
					got, err := v41RingFill(t, v, ring, kv, scores, ratio, width, order)
					if err != nil {
						t.Fatalf("group %d: %v", group, err)
					}
					v41RingBitsEqual(t, fmt.Sprintf("group %d pooled", group), got, v41RingPool(t, v, kv, scores, ratio, width))
					v41RingBitsEqual(t, fmt.Sprintf("group %d kv ring", group), append([]float32(nil), v.Read(kvRing)...), kv)
					v41RingBitsEqual(t, fmt.Sprintf("group %d score ring", group), append([]float32(nil), v.Read(scoreRing)...), scores)
				}
			})
		}
	}
}

// fak-test:runtime integration est=2s lane=optin
// A non-finite kv or score anywhere in the group refuses the pooled row with
// the typed sentinel, including a -Inf score whose softmax weight is zero and
// whose pooled value would otherwise be finite.
func TestVulkanV41CompressorRingRefusesNonFinite(t *testing.T) {
	v := v41RingRequire(t)
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
			kvRing, scoreRing := v41RingAlloc(t, v, ratio, width)
			kv, scores := make([]float32, ratio*width), make([]float32, ratio*width)
			for i := range kv {
				kv[i], scores[i] = float32(i%5)-2, float32(i%3)
			}
			if tc.kv {
				kv[tc.slot*width+tc.dim] = tc.value
			} else {
				scores[tc.slot*width+tc.dim] = tc.value
			}
			_, err := v41RingFill(t, v, [2]Tensor{kvRing, scoreRing}, kv, scores, ratio, width, []int{0, 1})
			var typed *V41CompressorRingNonFiniteError
			if !errors.Is(err, ErrV41CompressorRingNonFinite) || !errors.As(err, &typed) || typed.Dim != tc.dim || typed.Status != tc.status {
				t.Fatalf("got %v, want dim %d status %d", err, tc.dim, tc.status)
			}
			if code := v41RingSubmissionStatus(); code != 0 {
				t.Fatalf("a refused group left sticky submission status %d", code)
			}
		})
	}
}

// fak-test:runtime integration est=2s lane=optin
// Inside a V4.1 step batch every store records only: the whole group costs
// exactly one queue submission, the checked read of the emitted row, where the
// replaced pool alone costs two.
func TestVulkanV41CompressorRingStepBatchSubmits(t *testing.T) {
	v := v41RingRequire(t)
	if !VulkanDispatchProfileEnabled() {
		t.Skip("set FAK_VULKAN_DISPATCH_PROFILE=1 to count submissions")
	}
	const ratio, width = 4, 512
	kvRing, scoreRing := v41RingAlloc(t, v, ratio, width)
	rows := make([]Tensor, 2*ratio)
	rng := rand.New(rand.NewSource(7))
	for i := range rows {
		x := make([]float32, width)
		for j := range x {
			x[j] = float32(rng.NormFloat64())
		}
		rows[i] = v41RingUpload(t, v, []int{width}, x)
	}
	if !v.BeginV41StepBatch() {
		t.Fatal("step batch did not open")
	}
	v.VulkanDebugResetDispatchProfile()
	var out Tensor
	submits := func() uint64 {
		p := v.VulkanDebugDispatchProfileSnapshot()
		return p.BatchSubmits + p.OneShotSubmits
	}
	for slot := 0; slot < ratio; slot++ {
		res, err := v.V41CompressorRingStore(kvRing, scoreRing, rows[2*slot], rows[2*slot+1], slot, ratio, width, slot == ratio-1)
		if err != nil {
			t.Fatal(err)
		}
		if slot < ratio-1 && submits() != 0 {
			t.Fatalf("store %d submitted %d times before the group closed", slot, submits())
		}
		out = res
	}
	afterEmit := submits()
	got := append([]float32(nil), v.Read(out)...)
	afterRead := submits()
	v.Free(out)
	// The replaced path's pool alone: its checked flush plus its readback.
	v.VulkanDebugResetDispatchProfile()
	pooled, err := v.V41CompressorPool(kvRing, scoreRing, ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	want := append([]float32(nil), v.Read(pooled)...)
	poolSubmits := submits()
	v.Free(pooled)
	if err := v.EndV41StepBatch(); err != nil {
		t.Fatal(err)
	}
	t.Logf("step-batch submits per group: ring %d (stores+emit+read), pool alone %d (flush+read)", afterRead, poolSubmits)
	if afterEmit != 0 || afterRead != 1 {
		t.Fatalf("group submits: emit %d, after read %d; want 0 and 1", afterEmit, afterRead)
	}
	if poolSubmits != 2 {
		t.Fatalf("pool path submits %d, want 2", poolSubmits)
	}
	decoded, err := DecodeV41CompressorRing(got, width)
	if err != nil {
		t.Fatal(err)
	}
	v41RingBitsEqual(t, "step-batch pooled", decoded, want)
}

// fak-test:runtime integration est=1s lane=optin
func TestVulkanV41CompressorRingRejects(t *testing.T) {
	v := v41RingRequire(t)
	kvRing, scoreRing := v41RingAlloc(t, v, 2, 4)
	row := v41RingUpload(t, v, []int{4}, make([]float32, 4))
	for name, call := range map[string]func() error{
		"ratio one alloc": func() error { _, _, err := v.V41CompressorRingAlloc(1, 4); return err },
		"slot range": func() error {
			_, err := v.V41CompressorRingStore(kvRing, scoreRing, row, row, 2, 2, 4, false)
			return err
		},
		"aliased rings": func() error {
			_, err := v.V41CompressorRingStore(kvRing, kvRing, row, row, 0, 2, 4, false)
			return err
		},
		"host operand": func() error {
			_, err := v.V41CompressorRingStore(kvRing, scoreRing, NewF32(Default(), []int{4}, make([]float32, 4)), row, 0, 2, 4, false)
			return err
		},
		"row aliases ring": func() error {
			_, err := v.V41CompressorRingStore(kvRing, scoreRing, Tensor{be: v, buf: kvRing.buf, Dtype: F32, Layout: RowMajor, Shape: []int{4}}, row, 0, 2, 4, false)
			return err
		},
	} {
		if err := call(); !errors.Is(err, ErrVulkanInvalidGeometry) {
			t.Errorf("%s: got %v, want ErrVulkanInvalidGeometry", name, err)
		}
	}
	if code := v41RingSubmissionStatus(); code != 0 {
		t.Fatalf("refusals left sticky submission status %d", code)
	}
}
