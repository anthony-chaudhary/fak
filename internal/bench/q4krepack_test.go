package bench

import (
	"math"
	"math/rand"
	"testing"
)

// q4krepack_test.go — the correctness half of the #5285 spike. The A/B in q4krepack.go
// reports a TIMING ratio, and a timing ratio is only meaningful if the two layouts compute
// the same thing; these tests are what make the number interpretable rather than decorative.
//
// The two properties are deliberately different in kind:
//
//   - the repack is a pure byte PERMUTATION (nothing derived, nothing dropped) — proven by
//     round-trip AND by a byte histogram, because a round-trip alone can be satisfied by two
//     inverses that agree on the same offset bug;
//   - the interleaved GEMV is BIT-identical to the row-major one — proven with
//     math.Float32bits equality, not a tolerance, because the arithmetic is unchanged by
//     construction and any drift means the layout indices are wrong.
//
// Shapes below deliberately include out values that are NOT a multiple of the width, since
// the short tail group (w < width) is the one place the offset math differs.

// q4kTestShapes are the [out,in] shapes every layout property is checked across. 2048x6144
// is the report default; the rest are small shapes chosen so out%width != 0 for width 4
// and/or 8, exercising the short tail group.
var q4kTestShapes = []struct{ out, in int }{
	{1, 256},   // single row: every group is a tail group
	{3, 256},   // out < width for both 4 and 8
	{7, 512},   // prime out: tail under 4 and under 8
	{8, 256},   // exactly one full 8-group
	{9, 768},   // one full 8-group + a 1-row tail
	{16, 1024}, // clean multiple of both widths
}

var q4kTestWidths = []int{1, 4, 8}

// TestQ4KRepackIsPurePermutation is the property q4kInterleave's shared-body design exists
// to support: repack then unrepack must return the original bytes exactly, at every width
// and every shape including the short tail group. Width 1 must additionally be the identity
// on the FIRST hop, so a scalar host can call the repack unconditionally.
func TestQ4KRepackIsPurePermutation(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5285))
	for _, sh := range q4kTestShapes {
		raw := randQ4KWeight(rng, sh.out, sh.in)
		for _, w := range q4kTestWidths {
			packed, err := RepackQ4KInterleaved(raw, sh.out, sh.in, w)
			if err != nil {
				t.Fatalf("shape [%d,%d] width %d: repack: %v", sh.out, sh.in, w, err)
			}
			if len(packed) != len(raw) {
				t.Fatalf("shape [%d,%d] width %d: packed %d bytes, want %d — a permutation cannot change length",
					sh.out, sh.in, w, len(packed), len(raw))
			}
			if w == 1 && string(packed) != string(raw) {
				t.Fatalf("shape [%d,%d]: width 1 must be the identity repack, but the bytes moved", sh.out, sh.in)
			}
			back, err := UnrepackQ4KInterleaved(packed, sh.out, sh.in, w)
			if err != nil {
				t.Fatalf("shape [%d,%d] width %d: unrepack: %v", sh.out, sh.in, w, err)
			}
			if string(back) != string(raw) {
				t.Fatalf("shape [%d,%d] width %d: round trip did not restore the original bytes", sh.out, sh.in, w)
			}
		}
	}
}

// TestQ4KRepackPreservesByteHistogram is the independent check the round-trip cannot make.
// A shared-body inverse round-trips even if the layout maps two source bytes onto one
// destination (the second write wins and the read-back re-reads that same slot). Comparing
// the multiset of bytes catches exactly that: a real permutation moves every byte, so the
// 256-bin histogram is invariant.
func TestQ4KRepackPreservesByteHistogram(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for _, sh := range q4kTestShapes {
		raw := randQ4KWeight(rng, sh.out, sh.in)
		var want [256]int
		for _, b := range raw {
			want[b]++
		}
		for _, w := range q4kTestWidths {
			packed, err := RepackQ4KInterleaved(raw, sh.out, sh.in, w)
			if err != nil {
				t.Fatalf("shape [%d,%d] width %d: repack: %v", sh.out, sh.in, w, err)
			}
			var got [256]int
			for _, b := range packed {
				got[b]++
			}
			if got != want {
				t.Fatalf("shape [%d,%d] width %d: byte histogram changed — the repack is dropping or duplicating bytes, not permuting them",
					sh.out, sh.in, w)
			}
		}
	}
}

// TestQ4KInterleavedGemvIsBitIdentical is the claim the whole A/B rests on: the two kernels
// differ ONLY in memory order, so every output element must match bit for bit. A tolerance
// comparison would hide a lane mix-up that happens to land close; math.Float32bits will not.
func TestQ4KInterleavedGemvIsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(4623))
	for _, sh := range q4kTestShapes {
		raw := randQ4KWeight(rng, sh.out, sh.in)
		x := make([]float32, sh.in)
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * float32(1+(i%9))
		}
		yA := make([]float32, sh.out)
		q4kGemvRowMajor(raw, sh.out, sh.in, x, yA)

		for _, w := range q4kTestWidths {
			packed, err := RepackQ4KInterleaved(raw, sh.out, sh.in, w)
			if err != nil {
				t.Fatalf("shape [%d,%d] width %d: repack: %v", sh.out, sh.in, w, err)
			}
			yB := make([]float32, sh.out)
			q4kGemvInterleaved(packed, sh.out, sh.in, w, x, yB)
			for o := 0; o < sh.out; o++ {
				if math.Float32bits(yA[o]) != math.Float32bits(yB[o]) {
					t.Fatalf("shape [%d,%d] width %d row %d: row-major %v != interleaved %v (Δ=%g) — layouts must agree bit for bit",
						sh.out, sh.in, w, o, yA[o], yB[o], math.Abs(float64(yA[o]-yB[o])))
				}
			}
			// A finite, non-degenerate result: an all-zero y would make the equality above
			// vacuously true and hide a kernel that never wrote anything.
			nonZero := false
			for _, v := range yA {
				if v != 0 && !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
					nonZero = true
					break
				}
			}
			if !nonZero {
				t.Fatalf("shape [%d,%d]: row-major GEMV produced no finite non-zero output — the bit-identity check would be vacuous", sh.out, sh.in)
			}
		}
	}
}

// TestQ4KWidthForKernel pins the runtime feature -> layout mapping #5285 asks for, including
// the documented arm64 narrowing (no i8mm tier in fak, so both NEON tiers take 4) and the
// no-vector-unit fallback to the width-1 identity.
func TestQ4KWidthForKernel(t *testing.T) {
	cases := []struct {
		kernel string
		want   int
	}{
		{"avx512", 8},
		{"avx2", 8},
		{"neon", 4},
		{"neon-amort", 4},
		{"scalar", 1},
		{"", 1},
		{"some-future-tier", 1},
	}
	for _, c := range cases {
		got, why := q4kWidthForKernel(c.kernel)
		if got != c.want {
			t.Errorf("kernel %q: width = %d, want %d", c.kernel, got, c.want)
		}
		if why == "" {
			t.Errorf("kernel %q: width selection must carry a reason", c.kernel)
		}
	}
}

// TestQ4KInterleaveWidthIsRuntimeDetected checks the exported selector agrees with the
// tier fak actually resolved on THIS host — the "binary shipped to an unknown CPU picks its
// own layout" property, which a build tag could not provide.
func TestQ4KInterleaveWidthIsRuntimeDetected(t *testing.T) {
	width, why := Q4KInterleaveWidth()
	if width != 1 && width != 4 && width != 8 {
		t.Fatalf("runtime-detected width = %d, want one of 1/4/8", width)
	}
	if why == "" {
		t.Fatal("runtime width selection must carry a reason")
	}
	// Repacking at the host's own width must still be a pure permutation — the path a real
	// load-time repack would take.
	rng := rand.New(rand.NewSource(1))
	raw := randQ4KWeight(rng, 5, 256)
	packed, err := RepackQ4KInterleaved(raw, 5, 256, width)
	if err != nil {
		t.Fatalf("repack at runtime width %d: %v", width, err)
	}
	back, err := UnrepackQ4KInterleaved(packed, 5, 256, width)
	if err != nil {
		t.Fatalf("unrepack at runtime width %d: %v", width, err)
	}
	if string(back) != string(raw) {
		t.Fatalf("round trip failed at the runtime-detected width %d", width)
	}
}

// TestQ4KRepackRejectsMalformedShapes keeps the repack from silently mis-slicing a payload
// whose geometry does not match the declared shape — the failure mode that would otherwise
// surface as garbage weights rather than an error.
func TestQ4KRepackRejectsMalformedShapes(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	good := randQ4KWeight(rng, 4, 512)
	cases := []struct {
		name           string
		b              []byte
		out, in, width int
	}{
		{"width below 1", good, 4, 512, 0},
		{"negative width", good, 4, 512, -4},
		{"in not a multiple of 256", good, 4, 300, 4},
		{"in zero", good, 4, 0, 4},
		{"negative out", good, -1, 512, 4},
		{"payload too short", good[:len(good)-1], 4, 512, 4},
		{"payload too long", append(append([]byte{}, good...), 0), 4, 512, 4},
	}
	for _, c := range cases {
		if _, err := RepackQ4KInterleaved(c.b, c.out, c.in, c.width); err == nil {
			t.Errorf("%s: repack accepted a malformed input, want an error", c.name)
		}
		if _, err := UnrepackQ4KInterleaved(c.b, c.out, c.in, c.width); err == nil {
			t.Errorf("%s: unrepack accepted a malformed input, want an error", c.name)
		}
	}
}

// TestRunQ4KRepackAB is the end-to-end artifact witness on a small, fast shape. It asserts
// the report's SELF-VERIFYING parts — round trip, bit identity, a recomputable digest, and a
// verdict — but deliberately asserts NOTHING about the direction of the speedup: per the
// file's fence, a slower interleave is a RESULT, not a test failure. Asserting a win here
// would make the arm a gate on shared-host timing noise, which is exactly what the fence
// forbids.
func TestRunQ4KRepackAB(t *testing.T) {
	rep, err := RunQ4KRepackAB(Q4KRepackConfig{Out: 64, In: 512, Iters: 3})
	if err != nil {
		t.Fatalf("RunQ4KRepackAB: %v", err)
	}
	if rep.Schema != Q4KRepackSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, Q4KRepackSchema)
	}
	if rep.Issue != "#5285" {
		t.Errorf("issue = %q, want #5285", rep.Issue)
	}
	if rep.Fence == "" {
		t.Error("report must carry the OBSERVED fence")
	}
	if !rep.RepackRoundTripOK {
		t.Error("round trip must hold in the report")
	}
	if !rep.BitIdentical || rep.MaxAbsDelta != 0 {
		t.Errorf("layouts must be bit-identical: bit_identical=%v max|Δ|=%g", rep.BitIdentical, rep.MaxAbsDelta)
	}
	if rep.Width != 1 && rep.Width != 4 && rep.Width != 8 {
		t.Errorf("width = %d, want one of 1/4/8", rep.Width)
	}
	if rep.DecodeKernel == "" {
		t.Error("report must name the runtime-detected decode kernel the width came from")
	}
	if rep.Verdict == "" {
		t.Error("report must carry a verdict")
	}
	if !rep.VerifyDigest() {
		t.Error("report digest must recompute over its own canonical JSON")
	}
	// A tampered field must break the digest — otherwise the digest is decoration.
	tampered := rep
	tampered.InterleavedNS = rep.InterleavedNS + 1
	if tampered.VerifyDigest() {
		t.Error("digest still verified after a timing field was edited — the witness is not tamper-evident")
	}
}

// TestQ4KRepackABPinnedWidth checks the config override path: a pinned width must be used
// verbatim while the reason still records what runtime detection WOULD have chosen, so a
// hand-pinned run can never be mistaken for a detected one when the artifact is read later.
func TestQ4KRepackABPinnedWidth(t *testing.T) {
	rep, err := RunQ4KRepackAB(Q4KRepackConfig{Out: 32, In: 256, Iters: 2, Width: 4})
	if err != nil {
		t.Fatalf("RunQ4KRepackAB: %v", err)
	}
	if rep.Width != 4 {
		t.Fatalf("pinned width = %d, want 4", rep.Width)
	}
	if !rep.BitIdentical {
		t.Error("a pinned width must still be bit-identical")
	}
	if rep.WidthWhy == "" || !contains(rep.WidthWhy, "pinned") {
		t.Errorf("pinned run must say so in width_why, got %q", rep.WidthWhy)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
