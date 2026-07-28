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

	// Only the 4-bit path carries metadata; the wider widths are the plain rate.
	for _, tc := range []struct {
		bits int
		want int64
	}{{8, 64}, {16, 128}, {32, 256}} {
		if got := kvBitsToBytes(64, tc.bits); got != tc.want {
			t.Fatalf("kvBitsToBytes(64,%d) = %d, want %d (no group metadata off the 4-bit path)", tc.bits, got, tc.want)
		}
	}
	if got, want := kvBitsToBytes(64, 4), int64(32+2*8); got != want {
		t.Fatalf("kvBitsToBytes(64,4) = %d, want %d", got, want)
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
