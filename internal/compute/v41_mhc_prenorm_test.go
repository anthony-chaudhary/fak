package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=5ms lane=default
func TestV41MHCPrenormValidationClosedContract(t *testing.T) {
	const rows, hidden = 2, 64
	width := V41MHCStreams * hidden
	f32 := NewF32(Default(), []int{V41MHCCoeffWidth, width}, make([]float32, V41MHCCoeffWidth*width))
	q2k := NewQ2K(Default(), []int{V41MHCCoeffWidth, width}, make([]byte, V41MHCCoeffWidth*(width/q2kSuper)*q2kSuperBlock))
	streams := NewF32(Default(), []int{rows, V41MHCStreams, hidden}, make([]float32, rows*width))
	params := NewF32(Default(), []int{V41MHCParamWidth}, make([]float32, V41MHCParamWidth))
	for _, w := range []Tensor{f32, q2k} {
		if err := validateV41MHCPrenorm(w, streams, params, rows, hidden, 20, 1e-6, 1e-6); err != nil {
			t.Fatalf("%v weight refused: %v", w.Dtype, err)
		}
	}
	wide := f32
	wide.Shape = []int{V41MHCCoeffWidth, V41MHCStreams * 16641} // 66 chunks of 1024
	edge := f32
	edge.Shape = []int{V41MHCCoeffWidth, 64 * 1024} // exactly 64 chunks
	unaligned := q2k
	unaligned.Shape = []int{V41MHCCoeffWidth, V41MHCStreams * 65}
	q8 := f32
	q8.Dtype = Q8_0
	transposed := f32
	transposed.Shape = []int{width, V41MHCCoeffWidth}
	if !v41MHCPrenormWeightServed(edge, 16*1024) || v41MHCPrenormWeightServed(wide, 16641) {
		t.Fatal("F32 chunk bound is not exactly 64 chunks of 1024")
	}
	flat := streams
	flat.Shape = []int{rows * width}
	for name, err := range map[string]error{
		"f32 chunk bound": validateV41MHCPrenorm(wide, streams, params, rows, 16641, 20, 1e-6, 1e-6),
		"q2k superblock":  validateV41MHCPrenorm(unaligned, streams, params, rows, 65, 20, 1e-6, 1e-6),
		"other dtype":     validateV41MHCPrenorm(q8, streams, params, rows, hidden, 20, 1e-6, 1e-6),
		"transposed":      validateV41MHCPrenorm(transposed, streams, params, rows, hidden, 20, 1e-6, 1e-6),
		"flat streams":    validateV41MHCPrenorm(f32, flat, params, rows, hidden, 20, 1e-6, 1e-6),
		"short params":    validateV41MHCPrenorm(f32, streams, streams, rows, hidden, 20, 1e-6, 1e-6),
		"zero rows":       validateV41MHCPrenorm(f32, streams, params, 0, hidden, 20, 1e-6, 1e-6),
		"zero iters":      validateV41MHCPrenorm(f32, streams, params, rows, hidden, 0, 1e-6, 1e-6),
		"zero eps":        validateV41MHCPrenorm(f32, streams, params, rows, hidden, 20, 0, 1e-6),
		"zero rms eps":    validateV41MHCPrenorm(f32, streams, params, rows, hidden, 20, 1e-6, 0),
	} {
		if !errors.Is(err, ErrV41MHCPrenormGeometry) {
			t.Fatalf("%s: err=%v, want ErrV41MHCPrenormGeometry", name, err)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
// The self-check probe must reach every order the prenorm reproduces: an F32
// partial last chunk, a partly filled second projection workgroup, a row and a
// panel; a Q2_K row over several superblocks. Streams are BF16 like the
// model's panel, and the probe is deterministic.
func TestV41MHCPrenormProbeGeometry(t *testing.T) {
	f32 := newV41MHCPrenormProbe(F32)
	width := V41MHCStreams * f32.hidden
	chunks := (width + 1023) / 1024
	perGroup := v41MHCPrenormMaxF32Chunks / chunks
	if width%1024 == 0 || chunks < 2 || V41MHCCoeffWidth%perGroup == 0 || (V41MHCCoeffWidth+perGroup-1)/perGroup < 2 {
		t.Fatalf("F32 probe hidden=%d: chunks=%d partial=%t outputs/group=%d", f32.hidden, chunks, width%1024 != 0, perGroup)
	}
	if !reflect.DeepEqual(f32.rows, []int{1, 2}) || len(f32.weight) != 4*V41MHCCoeffWidth*width || len(f32.streams) != 2*width {
		t.Fatalf("F32 probe rows=%v weight=%d streams=%d", f32.rows, len(f32.weight), len(f32.streams))
	}
	// A device that contracts acc += w*x in only one of the two modules must
	// change probe bits: fused and unfused sums of each output already differ.
	contracted := 0
	for o := 0; o < V41MHCCoeffWidth; o++ {
		var fused, plain float32
		for i := 0; i < width; i++ {
			w := math.Float32frombits(binary.LittleEndian.Uint32(f32.weight[4*(o*width+i):]))
			fused = float32(math.FMA(float64(w), float64(f32.streams[i]), float64(fused)))
			plain += float32(w * f32.streams[i])
		}
		if math.Float32bits(fused) != math.Float32bits(plain) {
			contracted++
		}
	}
	if contracted < V41MHCCoeffWidth/2 {
		t.Fatalf("F32 probe separates fused from unfused accumulation in only %d of %d outputs", contracted, V41MHCCoeffWidth)
	}
	q2k := newV41MHCPrenormProbe(Q2_K)
	qWidth := V41MHCStreams * q2k.hidden
	if qWidth%q2kSuper != 0 || qWidth/q2kSuper < 2 || !reflect.DeepEqual(q2k.rows, []int{1}) || len(q2k.weight) != V41MHCCoeffWidth*(qWidth/q2kSuper)*q2kSuperBlock || len(q2k.streams) != qWidth {
		t.Fatalf("Q2_K probe hidden=%d rows=%v weight=%d streams=%d", q2k.hidden, q2k.rows, len(q2k.weight), len(q2k.streams))
	}
	for _, p := range []v41MHCPrenormProbe{f32, q2k} {
		w := Tensor{Dtype: p.dtype, Layout: RowMajor, Shape: []int{V41MHCCoeffWidth, V41MHCStreams * p.hidden}}
		if !v41MHCPrenormWeightServed(w, p.hidden) || len(p.params) != V41MHCParamWidth {
			t.Fatalf("%v probe is not a served prenorm operand set", p.dtype)
		}
		nonzero := 0
		for _, s := range p.streams {
			if math.Float32bits(s)&0xffff != 0 || math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
				t.Fatalf("%v probe stream %g is not a finite BF16 value", p.dtype, s)
			}
			if s != 0 {
				nonzero++
			}
		}
		if nonzero < len(p.streams)/2 || !reflect.DeepEqual(p, newV41MHCPrenormProbe(p.dtype)) {
			t.Fatalf("%v probe is vacuous (%d nonzero streams) or not deterministic", p.dtype, nonzero)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
// Admission compares every coefficient and every raw mix, never the sum of
// squares slot the reference does not expose.
func TestV41MHCPrenormSelfCheckComparatorClosed(t *testing.T) {
	probe := newV41MHCPrenormProbe(F32)
	var image [][]byte
	for _, rows := range probe.rows {
		raw := make([]byte, 4*rows*V41MHCCoeffWidth)
		out := make([]byte, 4*rows*V41MHCCoeffWidth)
		scratch := make([]byte, 4*rows*v41MHCPrenormStride)
		for i := range raw {
			raw[i], out[i] = byte(i*7+rows), byte(i*13+rows)
		}
		for r := 0; r < rows; r++ {
			copy(scratch[r*4*v41MHCPrenormStride:], raw[r*4*V41MHCCoeffWidth:(r+1)*4*V41MHCCoeffWidth])
		}
		image = append(image, raw, append([]byte(nil), out...), scratch, out)
	}
	clone := func() [][]byte {
		c := make([][]byte, len(image))
		for i := range image {
			c[i] = append([]byte(nil), image[i]...)
		}
		return c
	}
	if !v41MHCPrenormSelfCheckEqual(probe, image) {
		t.Fatal("equal image refused")
	}
	sumsq := clone()
	sumsq[6][4*(2*v41MHCPrenormStride-1)] ^= 1 // panel row 1 sum of squares
	if !v41MHCPrenormSelfCheckEqual(probe, sumsq) {
		t.Fatal("the unexposed sum of squares slot decided admission")
	}
	for name, mutate := range map[string]func([][]byte) [][]byte{
		"row coeff":   func(c [][]byte) [][]byte { c[3][5] ^= 1; return c },
		"panel coeff": func(c [][]byte) [][]byte { c[7][4*V41MHCCoeffWidth+3] ^= 0x80; return c },
		"row raw":     func(c [][]byte) [][]byte { c[2][0] ^= 1; return c },
		"panel raw":   func(c [][]byte) [][]byte { c[6][4*v41MHCPrenormStride+4*V41MHCCoeffWidth-1] ^= 1; return c },
		"region lost": func(c [][]byte) [][]byte { return c[:len(c)-1] },
	} {
		if v41MHCPrenormSelfCheckEqual(probe, mutate(clone())) {
			t.Fatalf("%s: differing image admitted", name)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
// The prenorm keeps the replaced MatMul's computetrace record: operation,
// phase, timer domain and [weight, input] shapes, under its own kernel id.
func TestV41MHCPrenormKernelEventKeepsMatMulRecord(t *testing.T) {
	const hidden = 64
	width := V41MHCStreams * hidden
	f32 := NewF32(Default(), []int{V41MHCCoeffWidth, width}, make([]float32, V41MHCCoeffWidth*width))
	q2k := NewQ2K(Default(), []int{V41MHCCoeffWidth, width}, make([]byte, V41MHCCoeffWidth*(width/q2kSuper)*q2kSuperBlock))
	started := time.Unix(1700000000, 0)
	for _, tc := range []struct {
		weight Tensor
		rows   int
		kernel string
		input  []int
	}{
		{f32, 1, "f32_v41_mhc_prenorm", []int{width}},
		{f32, 3, "f32_v41_mhc_prenorm", []int{3, width}},
		{q2k, 1, "q2_k_v41_mhc_prenorm", []int{width}},
	} {
		got := v41MHCPrenormKernelEvent("vulkan", tc.weight, tc.rows, hidden, started, vulkanKernelTiming{})
		matmul := vulkanKernelEvent("vulkan", tc.weight, NewF32(Default(), tc.input, make([]float32, tc.rows*width)), started, vulkanKernelTiming{})
		matmul.Kernel = tc.kernel
		if !reflect.DeepEqual(got, matmul) || got.Operation != "matmul" || got.TimerDomain != VulkanTimerDomainUnavailable {
			t.Fatalf("%s rows=%d: event %+v, want %+v", tc.kernel, tc.rows, got, matmul)
		}
	}
	var seen []computetrace.Event
	computetrace.SetObserver(nil)
	recordV41MHCPrenormKernel("vulkan", f32, 1, hidden, started, vulkanKernelTiming{})
	computetrace.SetObserver(func(e computetrace.Event) { seen = append(seen, e) })
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	recordV41MHCPrenormKernel("vulkan", f32, 1, hidden, started, vulkanKernelTiming{})
	if len(seen) != 1 || seen[0].Kernel != "f32_v41_mhc_prenorm" {
		t.Fatalf("recorded %d events %+v, want one f32_v41_mhc_prenorm record and none while nothing observed", len(seen), seen)
	}
}
