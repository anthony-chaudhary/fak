package model

import (
	"math"
	"math/rand"
	"testing"
)

// kvquant_test.go — the behavioural witness for the 4-bit KV-cache codec and its byte
// accounting (kvquant.go, #4874).
//
// The codec is a MIXTURE of two stores, not one: a 4-bit code lives in the LOW nibble for
// an even element index and the HIGH nibble for an odd one, with a separate write path
// (`=` vs `|= <<4`) and a separate read path (`&0x0f` vs `>>4`). A round-trip test over
// constant or symmetric data passes with either store broken, so the fixtures below
// deliberately drive the even and odd streams to opposite ends of the group range and
// decode each nibble by hand — a parity mix-up then shows up as a ~15-step error rather
// than as silence. The other regime splits covered here: scale>0 vs the constant-group
// scale==0 path, the bits==4 metadata branch in the byte accounting vs every other width,
// hybrid vs non-hybrid layer selection, and the VHeadDim-vs-HeadDim fallback.

func kvquantAbs(x float32) float32 {
	return float32(math.Abs(float64(x)))
}

// kvquantNibble reads element i's code straight out of the packed byte using the layout
// KVQuant4 documents, independently of Dequantize.
func kvquantNibble(q KVQuant4, i int) byte {
	b := q.Codes[i/2]
	if i%2 == 0 {
		return b & 0x0f
	}
	return b >> 4
}

// kvquantGroupBounds returns the [lo,hi) element span of group g in an N-element row-set.
func kvquantGroupBounds(g, n int) (int, int) {
	lo := g * KVQuant4GroupSize
	hi := lo + KVQuant4GroupSize
	if hi > n {
		hi = n
	}
	return lo, hi
}

// TestQuantizeKV4PacksBothNibbleStores drives the even-index and odd-index elements to
// opposite ends of one group's dynamic range, then decodes every element from its own raw
// nibble. Reading or writing a single store would collapse the two streams onto each
// other and blow the error bound by roughly the full group range.
func TestQuantizeKV4PacksBothNibbleStores(t *testing.T) {
	const n = KVQuant4GroupSize
	src := make([]float32, n)
	for i := 0; i < n; i++ {
		k := float32(i/2) / 16 // 0 .. 0.9375, spread within each stream
		if i%2 == 0 {
			src[i] = -4 + k // even stream: bottom of the range
		} else {
			src[i] = 3 + k // odd stream: top of the range
		}
	}

	q := QuantizeKV4(src)
	if q.N != n {
		t.Fatalf("N = %d, want %d", q.N, n)
	}
	if len(q.Codes) != n/2 {
		t.Fatalf("len(Codes) = %d, want %d (two codes per byte)", len(q.Codes), n/2)
	}
	if len(q.Scale) != 1 || len(q.Min) != 1 {
		t.Fatalf("len(Scale),len(Min) = %d,%d, want 1,1 for a single group", len(q.Scale), len(q.Min))
	}

	bound := q.ErrorBound()
	if bound <= 0 {
		t.Fatalf("ErrorBound = %v, want > 0 for a non-constant group", bound)
	}
	tol := bound * 1.001 // float32 rounding slack on a provable ceiling

	// Layout witness: decode each element from ITS OWN nibble, by hand.
	for i := 0; i < n; i++ {
		got := q.Min[0] + float32(kvquantNibble(q, i))*q.Scale[0]
		if d := kvquantAbs(got - src[i]); d > tol {
			t.Fatalf("element %d decoded from its own nibble = %v, want %v (err %v > bound %v): nibble store mismatched",
				i, got, src[i], d, bound)
		}
	}

	// Both stores must carry distinct code values, not one repeated code.
	evenCodes := map[byte]bool{}
	oddCodes := map[byte]bool{}
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			evenCodes[kvquantNibble(q, i)] = true
		} else {
			oddCodes[kvquantNibble(q, i)] = true
		}
	}
	if len(evenCodes) < 2 || len(oddCodes) < 2 {
		t.Fatalf("distinct codes: even=%d odd=%d, want >=2 in each store (fixture must exercise both)", len(evenCodes), len(oddCodes))
	}

	// Dequantize must keep the streams separated: every even reconstruction strictly
	// below every odd one.
	out := q.Dequantize()
	if len(out) != n {
		t.Fatalf("len(Dequantize()) = %d, want %d", len(out), n)
	}
	evenMax := float32(-math.MaxFloat32)
	oddMin := float32(math.MaxFloat32)
	for i := 0; i < n; i++ {
		if d := kvquantAbs(out[i] - src[i]); d > tol {
			t.Fatalf("Dequantize element %d = %v, want %v (err %v > bound %v)", i, out[i], src[i], d, bound)
		}
		if i%2 == 0 {
			if out[i] > evenMax {
				evenMax = out[i]
			}
		} else if out[i] < oddMin {
			oddMin = out[i]
		}
	}
	if evenMax >= oddMin {
		t.Fatalf("stream separation lost: max even reconstruction %v >= min odd reconstruction %v", evenMax, oddMin)
	}
}

// TestQuantizeKV4RoundTripRespectsErrorBound checks the provable ceiling over lengths that
// cover a sub-group row, an exact group, a partial trailing group, and odd N (where the
// final byte's high nibble is unused).
func TestQuantizeKV4RoundTripRespectsErrorBound(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	for _, n := range []int{1, 2, 3, 7, 31, 32, 33, 64, 65, 127, 256} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 3)
		}
		q := QuantizeKV4(src)

		wantGroups := (n + KVQuant4GroupSize - 1) / KVQuant4GroupSize
		if len(q.Scale) != wantGroups || len(q.Min) != wantGroups {
			t.Fatalf("n=%d: groups = %d/%d, want %d", n, len(q.Scale), len(q.Min), wantGroups)
		}
		if want := (n + 1) / 2; len(q.Codes) != want {
			t.Fatalf("n=%d: len(Codes) = %d, want %d", n, len(q.Codes), want)
		}

		// ErrorBound is defined as half the WIDEST group step.
		var widest float32
		for _, s := range q.Scale {
			if s > widest {
				widest = s
			}
		}
		if got := q.ErrorBound(); got != widest/2 {
			t.Fatalf("n=%d: ErrorBound = %v, want %v (half the widest group step)", n, got, widest/2)
		}

		out := q.Dequantize()
		if len(out) != n {
			t.Fatalf("n=%d: len(Dequantize()) = %d", n, len(out))
		}
		// Per-group: each element must respect ITS OWN group's step, which is the
		// stronger statement and the one a single global scale would fail.
		for g := 0; g < wantGroups; g++ {
			lo, hi := kvquantGroupBounds(g, n)
			groupBound := q.Scale[g] / 2
			for i := lo; i < hi; i++ {
				d := kvquantAbs(out[i] - src[i])
				if d > groupBound*1.001+1e-12 {
					t.Fatalf("n=%d element %d: err %v exceeds its group step/2 %v", n, i, d, groupBound)
				}
				if d > q.ErrorBound()*1.001+1e-12 {
					t.Fatalf("n=%d element %d: err %v exceeds ErrorBound %v", n, i, d, q.ErrorBound())
				}
			}
		}
	}
}

// TestQuantizeKV4UsesPerGroupScale pairs a narrow group with a group a thousand times
// wider. A single shared scale would drag the narrow group's error up to the global
// bound; group-wise scales keep it three orders of magnitude tighter.
func TestQuantizeKV4UsesPerGroupScale(t *testing.T) {
	n := 2 * KVQuant4GroupSize
	src := make([]float32, n)
	for i := 0; i < KVQuant4GroupSize; i++ {
		src[i] = float32(i) / float32(KVQuant4GroupSize) // group 0: range ~1
		src[KVQuant4GroupSize+i] = float32(i) * 1000     // group 1: range ~31000
	}
	q := QuantizeKV4(src)
	if len(q.Scale) != 2 {
		t.Fatalf("len(Scale) = %d, want 2", len(q.Scale))
	}
	if !(q.Scale[1] > 100*q.Scale[0]) {
		t.Fatalf("scales = %v, want group 1 far wider than group 0 (per-group, not global)", q.Scale)
	}
	if q.Min[0] != 0 || q.Min[1] != 0 {
		t.Fatalf("mins = %v, want both 0 for these fixtures", q.Min)
	}
	bound := q.ErrorBound()
	if bound != q.Scale[1]/2 {
		t.Fatalf("ErrorBound = %v, want %v (the wider group governs)", bound, q.Scale[1]/2)
	}

	out := q.Dequantize()
	var narrowWorst float32
	for i := 0; i < KVQuant4GroupSize; i++ {
		if d := kvquantAbs(out[i] - src[i]); d > narrowWorst {
			narrowWorst = d
		}
	}
	if narrowWorst > q.Scale[0]/2*1.001 {
		t.Fatalf("narrow-group worst error %v exceeds its own step/2 %v", narrowWorst, q.Scale[0]/2)
	}
	if narrowWorst >= bound/100 {
		t.Fatalf("narrow-group worst error %v is not far below the global bound %v: scale looks shared, not per-group", narrowWorst, bound)
	}
}

// TestQuantizeKV4ConstantGroupRoundTripsExactly covers the scale==0 branch: a group whose
// min equals its max stores the value in Min and dequantizes back bit-exactly, with no
// division by zero and no contribution to the error bound.
func TestQuantizeKV4ConstantGroupRoundTripsExactly(t *testing.T) {
	for _, v := range []float32{0, 2.5, -7.25} {
		n := 2 * KVQuant4GroupSize
		src := make([]float32, n)
		for i := range src {
			src[i] = v
		}
		q := QuantizeKV4(src)
		for g, s := range q.Scale {
			if s != 0 {
				t.Fatalf("v=%v: Scale[%d] = %v, want 0 for a constant group", v, g, s)
			}
			if q.Min[g] != v {
				t.Fatalf("v=%v: Min[%d] = %v, want %v", v, g, q.Min[g], v)
			}
		}
		if b := q.ErrorBound(); b != 0 {
			t.Fatalf("v=%v: ErrorBound = %v, want exactly 0", v, b)
		}
		out := q.Dequantize()
		for i := range out {
			if out[i] != v {
				t.Fatalf("v=%v: Dequantize[%d] = %v, want the exact input", v, i, out[i])
			}
		}
	}

	// Mixed: a constant group beside a varying one. The constant side stays exact even
	// though the global bound is now positive.
	n := 2 * KVQuant4GroupSize
	src := make([]float32, n)
	for i := 0; i < KVQuant4GroupSize; i++ {
		src[i] = 2.5
		src[KVQuant4GroupSize+i] = float32(i) - 16
	}
	q := QuantizeKV4(src)
	if q.Scale[0] != 0 || q.Scale[1] <= 0 {
		t.Fatalf("mixed scales = %v, want 0 then >0", q.Scale)
	}
	if got, want := q.ErrorBound(), q.Scale[1]/2; got != want {
		t.Fatalf("mixed ErrorBound = %v, want %v", got, want)
	}
	out := q.Dequantize()
	for i := 0; i < KVQuant4GroupSize; i++ {
		if out[i] != 2.5 {
			t.Fatalf("mixed constant-group element %d = %v, want exactly 2.5", i, out[i])
		}
	}
	for i := KVQuant4GroupSize; i < n; i++ {
		if d := kvquantAbs(out[i] - src[i]); d > q.ErrorBound()*1.001 {
			t.Fatalf("mixed varying-group element %d: err %v > bound %v", i, d, q.ErrorBound())
		}
	}
}

// TestQuantizeKV4EndpointsMapToCodeRange pins the affine mapping's boundary behaviour: the
// group minimum lands on code 0 and reconstructs EXACTLY, the maximum lands on code 15,
// and codes are monotone in the input.
func TestQuantizeKV4EndpointsMapToCodeRange(t *testing.T) {
	const n = KVQuant4GroupSize
	src := make([]float32, n)
	// A deliberately off-centre, non-zero-centred range: post-RoPE K rows are not
	// symmetric, which is why the codec is affine rather than symmetric.
	for i := range src {
		src[i] = 3 + float32(i)*0.7
	}
	src[5] = 3            // the minimum, off the sorted position
	src[n-3] = 3 + 31*0.7 // the maximum, also off-position

	q := QuantizeKV4(src)
	mn, mx := src[0], src[0]
	for _, v := range src {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if q.Min[0] != mn {
		t.Fatalf("Min = %v, want the group minimum %v", q.Min[0], mn)
	}
	if want := (mx - mn) / 15; q.Scale[0] != want {
		t.Fatalf("Scale = %v, want (max-min)/15 = %v (15 intervals across 16 code points)", q.Scale[0], want)
	}

	out := q.Dequantize()
	for i := range src {
		code := kvquantNibble(q, i)
		if code > 15 {
			t.Fatalf("element %d: code %d out of 0..15", i, code)
		}
		if src[i] == mn {
			if code != 0 {
				t.Fatalf("minimum element %d got code %d, want 0", i, code)
			}
			if out[i] != mn {
				t.Fatalf("minimum element %d reconstructed as %v, want the exact minimum %v", i, out[i], mn)
			}
		}
		if src[i] == mx && code != 15 {
			t.Fatalf("maximum element %d got code %d, want 15", i, code)
		}
	}

	// Monotone within the group: a larger input never gets a smaller code.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if src[i] < src[j] && kvquantNibble(q, i) > kvquantNibble(q, j) {
				t.Fatalf("non-monotone codes: src[%d]=%v code %d vs src[%d]=%v code %d",
					i, src[i], kvquantNibble(q, i), j, src[j], kvquantNibble(q, j))
			}
		}
	}
}

// TestQuantizeKV4HandlesEmptyAndPartialGroups covers the degenerate and ragged shapes: an
// empty row-set, a single element, and a trailing group that holds one element (which is
// therefore constant and exact).
func TestQuantizeKV4HandlesEmptyAndPartialGroups(t *testing.T) {
	empty := QuantizeKV4(nil)
	if empty.N != 0 || len(empty.Codes) != 0 || len(empty.Scale) != 0 || len(empty.Min) != 0 {
		t.Fatalf("QuantizeKV4(nil) = %+v, want a zero row-set", empty)
	}
	if empty.ErrorBound() != 0 || empty.Bytes() != 0 {
		t.Fatalf("empty bound/bytes = %v/%d, want 0/0", empty.ErrorBound(), empty.Bytes())
	}
	if got := empty.Dequantize(); len(got) != 0 {
		t.Fatalf("empty Dequantize len = %d, want 0", len(got))
	}

	one := QuantizeKV4([]float32{-1.5})
	if len(one.Codes) != 1 || len(one.Scale) != 1 {
		t.Fatalf("single element: codes=%d scale=%d, want 1/1", len(one.Codes), len(one.Scale))
	}
	if one.Scale[0] != 0 || one.Min[0] != -1.5 {
		t.Fatalf("single element: scale/min = %v/%v, want 0/-1.5", one.Scale[0], one.Min[0])
	}
	if got := one.Dequantize(); len(got) != 1 || got[0] != -1.5 {
		t.Fatalf("single element Dequantize = %v, want [-1.5]", got)
	}

	// 33 elements: a full group plus a one-element trailing group.
	n := KVQuant4GroupSize + 1
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 0.5
	}
	src[n-1] = 100 // the lone trailing element, far from group 0's range
	q := QuantizeKV4(src)
	if len(q.Scale) != 2 {
		t.Fatalf("n=%d: groups = %d, want 2", n, len(q.Scale))
	}
	if q.Scale[1] != 0 || q.Min[1] != 100 {
		t.Fatalf("trailing group scale/min = %v/%v, want 0/100 (a one-element group is constant)", q.Scale[1], q.Min[1])
	}
	if len(q.Codes) != (n+1)/2 {
		t.Fatalf("n=%d: len(Codes) = %d, want %d", n, len(q.Codes), (n+1)/2)
	}
	out := q.Dequantize()
	if out[n-1] != 100 {
		t.Fatalf("trailing element = %v, want exactly 100", out[n-1])
	}
	// The trailing element sits at index 32 — an EVEN index, i.e. the LOW nibble of the
	// last byte, whose high nibble is unused.
	if hi := q.Codes[(n-1)/2] >> 4; hi != 0 {
		t.Fatalf("unused high nibble of the final byte = %d, want 0", hi)
	}
}

// TestKVQuant4BytesCountsGroupMetadata checks the honest rate: 4 bits of payload plus a
// per-group f32 scale and min is 6 bits/element at group size 32, not 4. It also ties the
// codec's realized footprint to the accounting helper the cache sizer uses.
func TestKVQuant4BytesCountsGroupMetadata(t *testing.T) {
	for _, n := range []int{32, 64, 128, 4096} {
		q := QuantizeKV4(make([]float32, n))
		if got, want := q.Bytes()*8, 6*n; got != want {
			t.Fatalf("n=%d: %d bits total, want %d (6 bits/element incl. group metadata)", n, got, want)
		}
		if naive := n / 2; q.Bytes() <= naive {
			t.Fatalf("n=%d: Bytes %d does not exceed the bare nibble count %d — metadata not counted", n, q.Bytes(), naive)
		}
	}

	// The accounting helper must agree with the codec's actual packed size at every
	// length, including ragged ones.
	for _, n := range []int{1, 2, 3, 5, 7, 31, 32, 33, 63, 64, 65, 129} {
		q := QuantizeKV4(make([]float32, n))
		groups := (n + KVQuant4GroupSize - 1) / KVQuant4GroupSize
		if want := (n+1)/2 + 8*groups; q.Bytes() != want {
			t.Fatalf("n=%d: Bytes = %d, want %d (codes + 4B scale + 4B min per group)", n, q.Bytes(), want)
		}
		if want := int64(q.Bytes()); kvBitsToBytes(n, 4) != want {
			t.Fatalf("n=%d: kvBitsToBytes(_,4) = %d, want the codec's own %d", n, kvBitsToBytes(n, 4), want)
		}
	}

	// 16-bit and 32-bit widths are the plain rate; 4-bit and 8-bit carry group metadata.
	for _, tc := range []struct {
		bits int
		want int64
	}{{16, 128}, {32, 256}} {
		if got := kvBitsToBytes(64, tc.bits); got != tc.want {
			t.Fatalf("kvBitsToBytes(64,%d) = %d, want %d (no group metadata off the quantized path)", tc.bits, got, tc.want)
		}
	}
	if got, want := kvBitsToBytes(64, 4), int64(32+2*8); got != want {
		t.Fatalf("kvBitsToBytes(64,4) = %d, want %d", got, want)
	}
	if got, want := kvBitsToBytes(64, 8), int64(64+2*8); got != want {
		t.Fatalf("kvBitsToBytes(64,8) = %d, want %d (8-bit carries group metadata)", got, want)
	}
	if kvBitsToBytes(0, 4) != 0 || kvBitsToBytes(64, 0) != 0 || kvBitsToBytes(-1, 4) != 0 {
		t.Fatalf("kvBitsToBytes guards: got %d/%d/%d, want 0/0/0",
			kvBitsToBytes(0, 4), kvBitsToBytes(64, 0), kvBitsToBytes(-1, 4))
	}
}

// TestKVQuantLayersExcludesLinearAttentionLayers pins the line between layers that hold a
// softmax KV cache and Gated-DeltaNet layers that hold a recurrent state instead.
func TestKVQuantLayersExcludesLinearAttentionLayers(t *testing.T) {
	if got := (Config{NumLayers: 0}).KVQuantLayers(); got != nil {
		t.Fatalf("NumLayers=0 layers = %v, want nil", got)
	}
	if got := (Config{NumLayers: -3}).KVQuantLayers(); got != nil {
		t.Fatalf("NumLayers<0 layers = %v, want nil", got)
	}

	// Non-hybrid: every layer holds KV.
	dense := Config{NumLayers: 4}
	if got := dense.KVQuantLayers(); len(got) != 4 || got[0] != 0 || got[3] != 3 {
		t.Fatalf("dense layers = %v, want the full range 0..3", got)
	}

	// Qwen3.5-style hybrid: 3 linear layers per full-attention layer.
	hybrid := Config{NumLayers: 12, LayerTypes: make([]string, 12)}
	for l := range hybrid.LayerTypes {
		if (l+1)%4 == 0 {
			hybrid.LayerTypes[l] = "full_attention"
		} else {
			hybrid.LayerTypes[l] = "linear_attention"
		}
	}
	got := hybrid.KVQuantLayers()
	want := []int{3, 7, 11}
	if len(got) != len(want) {
		t.Fatalf("hybrid layers = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("hybrid layers = %v, want %v", got, want)
		}
	}

	// Sliding-window attention still holds a KV cache — only linear attention does not.
	gemma := Config{NumLayers: 4, LayerTypes: []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}}
	if got := gemma.KVQuantLayers(); len(got) != 4 {
		t.Fatalf("sliding-window layers = %v, want all 4 (sliding attention still caches K/V)", got)
	}

	// A LayerTypes list shorter than NumLayers leaves the tail untyped; untyped layers
	// are treated as attention layers and stay admitted.
	short := Config{NumLayers: 4, LayerTypes: []string{"linear_attention", "linear_attention"}}
	if got := short.KVQuantLayers(); len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Fatalf("short LayerTypes layers = %v, want [2 3]", got)
	}
}

// TestKVCacheBytesAtBitsAccountsPrecisionAndKVLayers exercises the sizing branches: the
// guards, the per-bit rate (including the 4-bit metadata that stops it flattering itself),
// the hybrid layer count, and the VHeadDim fallback.
func TestKVCacheBytesAtBitsAccountsPrecisionAndKVLayers(t *testing.T) {
	dense := Config{NumLayers: 2, NumKVHeads: 4, HeadDim: 64}

	// Guards.
	if got := dense.KVCacheBytesAtBits(0, 32); got != 0 {
		t.Fatalf("positions=0 -> %d, want 0", got)
	}
	if got := dense.KVCacheBytesAtBits(-1, 32); got != 0 {
		t.Fatalf("positions<0 -> %d, want 0", got)
	}
	if got := dense.KVCacheBytesAtBits(128, 0); got != 0 {
		t.Fatalf("bits=0 -> %d, want 0", got)
	}
	allLinear := Config{NumLayers: 2, NumKVHeads: 4, HeadDim: 64,
		LayerTypes: []string{"linear_attention", "linear_attention"}}
	if got := allLinear.KVCacheBytesAtBits(128, 32); got != 0 {
		t.Fatalf("all-linear backbone -> %d, want 0 (no layer holds a KV cache)", got)
	}

	// f32 is exactly twice f16.
	f32 := dense.KVCacheBytesAtBits(1024, 32)
	f16 := dense.KVCacheBytesAtBits(1024, 16)
	if f32 != 2*f16 {
		t.Fatalf("f32 %d != 2 * f16 %d", f32, f16)
	}
	// K+V, both 4*64 elements, over 2 layers and 1024 positions at 4 bytes each.
	if want := int64(2 * 4 * 64 * 4 * 2 * 1024); f32 != want {
		t.Fatalf("f32 bytes = %d, want %d", f32, want)
	}

	// 4-bit is 16/3 smaller than f32, NOT the naive 8x: the per-group scale/min pushes
	// the honest rate to 6 bits/element.
	q4 := dense.KVCacheBytesAtBits(1024, 4)
	if 3*f32 != 16*q4 {
		t.Fatalf("4-bit accounting: 3*f32 = %d, 16*q4 = %d, want equal (6 bits/element)", 3*f32, 16*q4)
	}
	if q4 <= f32/8 {
		t.Fatalf("4-bit bytes %d <= f32/8 %d: group metadata is not being counted", q4, f32/8)
	}

	// Linear in positions.
	if got, want := dense.KVCacheBytesAtBits(2048, 4), 2*q4; got != want {
		t.Fatalf("doubling positions -> %d, want %d", got, want)
	}

	// Hybrid: 3 of every 4 layers hold no KV, so the cache is exactly a quarter of the
	// same backbone sized as if every layer cached.
	hybrid := Config{NumLayers: 12, NumKVHeads: 4, HeadDim: 64, LayerTypes: make([]string, 12)}
	for l := range hybrid.LayerTypes {
		if (l+1)%4 == 0 {
			hybrid.LayerTypes[l] = "full_attention"
		} else {
			hybrid.LayerTypes[l] = "linear_attention"
		}
	}
	naive := Config{NumLayers: 12, NumKVHeads: 4, HeadDim: 64}
	if got, want := hybrid.KVCacheBytesAtBits(4096, 4), naive.KVCacheBytesAtBits(4096, 4)/4; got != want {
		t.Fatalf("hybrid bytes = %d, want %d (a quarter of the all-layers number)", got, want)
	}

	// VHeadDim fallback: zero means "same as HeadDim"; an MLA-style asymmetric V head
	// shrinks only the V half.
	sym := Config{NumLayers: 1, NumKVHeads: 4, HeadDim: 64}
	explicit := Config{NumLayers: 1, NumKVHeads: 4, HeadDim: 64, VHeadDim: 64}
	if a, b := sym.KVCacheBytesAtBits(512, 32), explicit.KVCacheBytesAtBits(512, 32); a != b {
		t.Fatalf("VHeadDim=0 -> %d, VHeadDim=HeadDim -> %d, want equal (fallback)", a, b)
	}
	halfV := Config{NumLayers: 1, NumKVHeads: 4, HeadDim: 64, VHeadDim: 32}
	if got, want := 4*halfV.KVCacheBytesAtBits(512, 32), 3*sym.KVCacheBytesAtBits(512, 32); got != want {
		t.Fatalf("half-width V: 4*asym = %d, 3*sym = %d, want equal (K unchanged, V halved)", got, want)
	}

	// The sizer and the codec must agree byte for byte on one position of one layer.
	oneLayer := Config{NumLayers: 1, NumKVHeads: 4, HeadDim: 64}
	row := make([]float32, 4*64)
	codec := int64(QuantizeKV4(row).Bytes() * 2) // K row + V row
	if got := oneLayer.KVCacheBytesAtBits(1, 4); got != codec {
		t.Fatalf("KVCacheBytesAtBits(1,4) = %d, want the codec's own K+V footprint %d", got, codec)
	}
}

// TestQuantizeKV8PacksCodesAndReconstructs tests the 8-bit affine group-wise quantization
// codec across ragged and regular lengths, checking code bounds, monotonicity, endpoints,
// and round-trip error ceilings.
func TestQuantizeKV8PacksCodesAndReconstructs(t *testing.T) {
	rng := rand.New(rand.NewSource(20260902))
	for _, n := range []int{1, 2, 3, 7, 31, 32, 33, 64, 65, 127, 256} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 3)
		}
		q := QuantizeKV8(src)

		wantGroups := (n + KVQuant8GroupSize - 1) / KVQuant8GroupSize
		if len(q.Scale) != wantGroups || len(q.Min) != wantGroups {
			t.Fatalf("n=%d: groups = %d/%d, want %d", n, len(q.Scale), len(q.Min), wantGroups)
		}
		if len(q.Codes) != n {
			t.Fatalf("n=%d: len(Codes) = %d, want %d (one byte per element)", n, len(q.Codes), n)
		}

		var widest float32
		for _, s := range q.Scale {
			if s > widest {
				widest = s
			}
		}
		if got := q.ErrorBound(); got != widest/2 {
			t.Fatalf("n=%d: ErrorBound = %v, want %v (half widest group step)", n, got, widest/2)
		}

		out := q.Dequantize()
		if len(out) != n {
			t.Fatalf("n=%d: len(Dequantize()) = %d", n, len(out))
		}
		for g := 0; g < wantGroups; g++ {
			lo := g * KVQuant8GroupSize
			hi := lo + KVQuant8GroupSize
			if hi > n {
				hi = n
			}
			groupBound := q.Scale[g] / 2
			for i := lo; i < hi; i++ {
				d := kvquantAbs(out[i] - src[i])
				if d > groupBound*1.001+1e-6 {
					t.Fatalf("n=%d element %d: err %v exceeds group step/2 %v", n, i, d, groupBound)
				}
				if d > q.ErrorBound()*1.001+1e-6 {
					t.Fatalf("n=%d element %d: err %v exceeds ErrorBound %v", n, i, d, q.ErrorBound())
				}
			}
		}
	}

	// Boundary mapping: minimum maps to 0 and reconstructs exactly, maximum maps to 255.
	const n = KVQuant8GroupSize
	src := make([]float32, n)
	for i := range src {
		src[i] = 10 + float32(i)*0.4
	}
	src[3] = 10           // minimum
	src[20] = 10 + 31*0.4 // maximum
	q := QuantizeKV8(src)
	mn, mx := src[0], src[0]
	for _, v := range src {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if q.Min[0] != mn {
		t.Fatalf("Min = %v, want %v", q.Min[0], mn)
	}
	if want := (mx - mn) / 255; q.Scale[0] != want {
		t.Fatalf("Scale = %v, want (mx-mn)/255 = %v", q.Scale[0], want)
	}
	out := q.Dequantize()
	for i := range src {
		code := q.Codes[i]
		if src[i] == mn {
			if code != 0 {
				t.Fatalf("minimum element %d got code %d, want 0", i, code)
			}
			if out[i] != mn {
				t.Fatalf("minimum element %d reconstructed as %v, want exact %v", i, out[i], mn)
			}
		}
		if src[i] == mx && code != 255 {
			t.Fatalf("maximum element %d got code %d, want 255", i, code)
		}
	}

	// Monotonicity within group.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if src[i] < src[j] && q.Codes[i] > q.Codes[j] {
				t.Fatalf("non-monotone codes: src[%d]=%v code %d vs src[%d]=%v code %d",
					i, src[i], q.Codes[i], j, src[j], q.Codes[j])
			}
		}
	}

	// Constant group round-trips bit-exactly with zero error bound.
	for _, v := range []float32{0, -3.1415, 42.5} {
		cSrc := make([]float32, 2*KVQuant8GroupSize)
		for i := range cSrc {
			cSrc[i] = v
		}
		cq := QuantizeKV8(cSrc)
		if cq.ErrorBound() != 0 {
			t.Fatalf("constant group ErrorBound = %v, want 0", cq.ErrorBound())
		}
		cOut := cq.Dequantize()
		for i, val := range cOut {
			if val != v {
				t.Fatalf("constant group element %d = %v, want exact %v", i, val, v)
			}
		}
	}

	// Degenerate empty input.
	empty := QuantizeKV8(nil)
	if empty.N != 0 || len(empty.Codes) != 0 || len(empty.Scale) != 0 || len(empty.Min) != 0 {
		t.Fatalf("QuantizeKV8(nil) = %+v, want zero", empty)
	}
	if empty.ErrorBound() != 0 || empty.Bytes() != 0 || len(empty.Dequantize()) != 0 {
		t.Fatalf("empty bound/bytes/dequant = %v/%d/%d, want 0/0/0",
			empty.ErrorBound(), empty.Bytes(), len(empty.Dequantize()))
	}
}

// TestKVQuant8BytesCountsGroupMetadata verifies the honest byte rate for 8-bit quantization:
// 1 byte per element payload + 8 bytes per 32-element group metadata = 10 bits/element.
func TestKVQuant8BytesCountsGroupMetadata(t *testing.T) {
	for _, n := range []int{32, 64, 128, 4096} {
		q := QuantizeKV8(make([]float32, n))
		// 10 bits/element: 8 bits payload + 2 bits metadata (8B/32 elems)
		if got, want := q.Bytes()*8, 10*n; got != want {
			t.Fatalf("n=%d: %d bits total, want %d (10 bits/element incl. group metadata)", n, got, want)
		}
		if naive := n; q.Bytes() <= naive {
			t.Fatalf("n=%d: Bytes %d does not exceed bare payload %d", n, q.Bytes(), naive)
		}
	}

	for _, n := range []int{1, 2, 3, 5, 7, 31, 32, 33, 63, 64, 65, 129} {
		q := QuantizeKV8(make([]float32, n))
		groups := (n + KVQuant8GroupSize - 1) / KVQuant8GroupSize
		if want := n + 8*groups; q.Bytes() != want {
			t.Fatalf("n=%d: Bytes = %d, want %d", n, q.Bytes(), want)
		}
		if want := int64(q.Bytes()); kvBitsToBytes(n, 8) != want {
			t.Fatalf("n=%d: kvBitsToBytes(_,8) = %d, want %d", n, kvBitsToBytes(n, 8), want)
		}
	}
}

// TestQuantizeKVAsymmetricRoundTripAndCosineFloors tests asymmetric K (8-bit) and V (4-bit)
// quantization pairs across diverse dimensions, verifying packed byte accounting, individual
// error bounds, and cosine similarity floors (>= 0.999 for K, >= 0.995 for V).
func TestQuantizeKVAsymmetricRoundTripAndCosineFloors(t *testing.T) {
	rng := rand.New(rand.NewSource(20260902))

	testCases := []struct {
		name string
		kDim int
		vDim int
	}{
		{"SingleGroup", 32, 32},
		{"StandardHead64", 64, 64},
		{"StandardHead128", 128, 128},
		{"AsymmetricMLAHead", 128, 64},
		{"MultiHeadRow1024", 1024, 1024},
		{"RaggedAsymmetric", 133, 97},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kSrc := make([]float32, tc.kDim)
			vSrc := make([]float32, tc.vDim)
			for i := range kSrc {
				kSrc[i] = float32(rng.NormFloat64() * 2.5)
			}
			for i := range vSrc {
				vSrc[i] = float32(rng.NormFloat64() * 1.8)
			}

			asym := QuantizeKVAsymmetric(kSrc, vSrc)

			// Bytes must equal the sum of K and V components.
			if want := asym.K.Bytes() + asym.V.Bytes(); asym.Bytes() != want {
				t.Fatalf("asym.Bytes() = %d, want %d", asym.Bytes(), want)
			}

			kOut, vOut := DequantizeKVAsymmetric(asym)
			if len(kOut) != tc.kDim || len(vOut) != tc.vDim {
				t.Fatalf("dequantized lens = %d, %d; want %d, %d", len(kOut), len(vOut), tc.kDim, tc.vDim)
			}

			// Per-element error bounds.
			kBound := asym.K.ErrorBound()
			for i, v := range kSrc {
				if d := kvquantAbs(kOut[i] - v); d > kBound*1.001+1e-6 {
					t.Fatalf("K element %d error %v > bound %v", i, d, kBound)
				}
			}
			vBound := asym.V.ErrorBound()
			for i, v := range vSrc {
				if d := kvquantAbs(vOut[i] - v); d > vBound*1.001+1e-6 {
					t.Fatalf("V element %d error %v > bound %v", i, d, vBound)
				}
			}

			// Cosine similarity floors: Keys require higher precision for exponential
			// routing (>= 0.999), Values tolerate lower precision for linear combination (>= 0.995).
			cosK := cosine(kSrc, kOut)
			cosV := cosine(vSrc, vOut)

			if cosK < 0.999 {
				t.Fatalf("Key cosine similarity %f < floor 0.999", cosK)
			}
			if cosV < 0.995 {
				t.Fatalf("Value cosine similarity %f < floor 0.995", cosV)
			}

			t.Logf("[%s] K(8-bit) cosine: %.6f (bound %g), V(4-bit) cosine: %.6f (bound %g)",
				tc.name, cosK, kBound, cosV, vBound)
		})
	}

	// Constant group handling in asymmetric pairs.
	t.Run("ConstantPair", func(t *testing.T) {
		kSrc := make([]float32, 64)
		vSrc := make([]float32, 64)
		for i := range kSrc {
			kSrc[i] = -2.75
			vSrc[i] = 1.5
		}
		asym := QuantizeKVAsymmetric(kSrc, vSrc)
		if asym.K.ErrorBound() != 0 || asym.V.ErrorBound() != 0 {
			t.Fatalf("constant pair bounds = %v, %v; want 0, 0", asym.K.ErrorBound(), asym.V.ErrorBound())
		}
		kOut, vOut := DequantizeKVAsymmetric(asym)
		for i := range kSrc {
			if kOut[i] != -2.75 {
				t.Fatalf("K[%d] = %v, want -2.75", i, kOut[i])
			}
			if vOut[i] != 1.5 {
				t.Fatalf("V[%d] = %v, want 1.5", i, vOut[i])
			}
		}
	})
}

// TestKVCacheBytesAsymmetricAccounting verifies the Config.KVCacheBytesAsymmetric sizer:
// parameter validation guards, parity with KVCacheBytesAtBits on symmetric precision,
// exact byte agreement with codec Bytes(), and support for asymmetric head dimensions.
func TestKVCacheBytesAsymmetricAccounting(t *testing.T) {
	dense := Config{NumLayers: 4, NumKVHeads: 8, HeadDim: 128}

	// Input guards: non-positive inputs must return 0.
	if got := dense.KVCacheBytesAsymmetric(0, 8, 4); got != 0 {
		t.Fatalf("positions=0 -> %d, want 0", got)
	}
	if got := dense.KVCacheBytesAsymmetric(-5, 8, 4); got != 0 {
		t.Fatalf("positions<0 -> %d, want 0", got)
	}
	if got := dense.KVCacheBytesAsymmetric(1024, 0, 4); got != 0 {
		t.Fatalf("kBits=0 -> %d, want 0", got)
	}
	if got := dense.KVCacheBytesAsymmetric(1024, 8, 0); got != 0 {
		t.Fatalf("vBits=0 -> %d, want 0", got)
	}
	if got := dense.KVCacheBytesAsymmetric(1024, -1, 4); got != 0 {
		t.Fatalf("kBits<0 -> %d, want 0", got)
	}

	// All-linear attention architecture holds no KV cache.
	allLinear := Config{NumLayers: 4, NumKVHeads: 8, HeadDim: 128,
		LayerTypes: []string{"linear_attention", "linear_attention", "linear_attention", "linear_attention"}}
	if got := allLinear.KVCacheBytesAsymmetric(1024, 8, 4); got != 0 {
		t.Fatalf("all-linear backbone -> %d, want 0", got)
	}

	// Parity with symmetric KVCacheBytesAtBits when kBits == vBits.
	for _, bits := range []int{4, 8, 16, 32} {
		asym := dense.KVCacheBytesAsymmetric(1024, bits, bits)
		sym := dense.KVCacheBytesAtBits(1024, bits)
		if asym != sym {
			t.Fatalf("bits=%d: asymmetric(%d,%d) = %d != symmetric(%d) = %d", bits, bits, bits, asym, bits, sym)
		}
	}

	// Linearity in positions.
	p1024 := dense.KVCacheBytesAsymmetric(1024, 8, 4)
	p2048 := dense.KVCacheBytesAsymmetric(2048, 8, 4)
	if p2048 != 2*p1024 {
		t.Fatalf("doubling positions = %d, want 2 * %d = %d", p2048, p1024, 2*p1024)
	}

	// Byte-exact calculation for K=8, V=4:
	// NumKVHeads*HeadDim = 8*128 = 1024 elements per row.
	// Groups = 1024/32 = 32 groups.
	// K(8-bit) per pos = 1024 + 32*8 = 1280 bytes.
	// V(4-bit) per pos = 512 + 32*8 = 768 bytes.
	// Total per pos = 1280 + 768 = 2048 bytes per layer.
	// Total for 4 layers, 1024 positions = 2048 * 4 * 1024 = 8,388,608 bytes.
	if want := int64(2048 * 4 * 1024); p1024 != want {
		t.Fatalf("K=8,V=4 bytes = %d, want %d", p1024, want)
	}

	// Agreement with codec instance for asymmetric head dimensions (MLA style).
	mlaCfg := Config{NumLayers: 1, NumKVHeads: 4, HeadDim: 128, VHeadDim: 64}
	kRow := make([]float32, 4*128)
	vRow := make([]float32, 4*64)
	codecFootprint := int64(QuantizeKVAsymmetric(kRow, vRow).Bytes())
	sizerFootprint := mlaCfg.KVCacheBytesAsymmetric(1, 8, 4)
	if sizerFootprint != codecFootprint {
		t.Fatalf("sizer footprint %d != codec footprint %d", sizerFootprint, codecFootprint)
	}
}

// TestKVCacheBytesAsymmetric262KReduction verifies byte accounting of asymmetric allocation
// at a quarter-million tokens context (262,144 positions), demonstrating how K=8, V=4
// reduces memory footprint from ~61.4 GB down to ~20.08 GB.
func TestKVCacheBytesAsymmetric262KReduction(t *testing.T) {
	const pos = 262144

	// 26 layers, 23 KV heads, HeadDim 64 -> 2944 B/pos -> 20,065,550,336 B (~20.08 GB).
	cfg20GB := Config{NumLayers: 26, NumKVHeads: 23, HeadDim: 64}
	asym20GB := cfg20GB.KVCacheBytesAsymmetric(pos, 8, 4)
	want20GB := int64(26) * 2944 * pos
	if asym20GB != want20GB {
		t.Fatalf("asym20GB = %d, want %d", asym20GB, want20GB)
	}
	asymGigaBytes := float64(asym20GB) / 1e9
	if math.Abs(asymGigaBytes-20.07) > 0.1 {
		t.Fatalf("asym20GB in GB = %.2f, want ~20.08 GB", asymGigaBytes)
	}

	// 29 layers, 8 KV heads, HeadDim 128 -> 62.28 GB (f32) -> 15.57 GB (K=8, V=4, 4x reduction).
	cfg61GB := Config{NumLayers: 29, NumKVHeads: 8, HeadDim: 128}
	f32_61GB := cfg61GB.KVCacheBytesAtBits(pos, 32)
	asym_61GB := cfg61GB.KVCacheBytesAsymmetric(pos, 8, 4)
	if f32_61GB != 4*asym_61GB {
		t.Fatalf("f32 (%d) != 4 * asym (%d)", f32_61GB, asym_61GB)
	}

	t.Logf("262K context footprint reduction:")
	t.Logf("  Config 1 (20.08 GB target): %d bytes (%.2f GB / %.2f GiB)",
		asym20GB, float64(asym20GB)/1e9, float64(asym20GB)/(1<<30))
	t.Logf("  Config 2 (~61.4 GB f32 baseline): f32 = %.2f GB -> K=8,V=4 = %.2f GB (4x reduction)",
		float64(f32_61GB)/1e9, float64(asym_61GB)/1e9)
}

// Benchmarks

func BenchmarkQuantizeKV8(b *testing.B) {
	const n = 1024
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i % 100)
	}
	b.ResetTimer()
	b.SetBytes(n * 4)
	for i := 0; i < b.N; i++ {
		_ = QuantizeKV8(src)
	}
}

func BenchmarkDequantizeKV8(b *testing.B) {
	const n = 1024
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i % 100)
	}
	q := QuantizeKV8(src)
	b.ResetTimer()
	b.SetBytes(n * 4)
	for i := 0; i < b.N; i++ {
		_ = q.Dequantize()
	}
}

func BenchmarkQuantizeKVAsymmetric(b *testing.B) {
	const n = 1024
	k := make([]float32, n)
	v := make([]float32, n)
	for i := range k {
		k[i] = float32(i % 100)
		v[i] = float32(i % 50)
	}
	b.ResetTimer()
	b.SetBytes(2 * n * 4)
	for i := 0; i < b.N; i++ {
		_ = QuantizeKVAsymmetric(k, v)
	}
}

func BenchmarkDequantizeKVAsymmetric(b *testing.B) {
	const n = 1024
	k := make([]float32, n)
	v := make([]float32, n)
	for i := range k {
		k[i] = float32(i % 100)
		v[i] = float32(i % 50)
	}
	asym := QuantizeKVAsymmetric(k, v)
	b.ResetTimer()
	b.SetBytes(2 * n * 4)
	for i := 0; i < b.N; i++ {
		_, _ = DequantizeKVAsymmetric(asym)
	}
}
