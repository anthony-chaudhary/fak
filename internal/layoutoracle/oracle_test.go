package layoutoracle

import (
	"errors"
	"math"
	"testing"
)

// V4.1 attention/mHC axes as carried by the ggufload deepseek41 metadata-only
// preflight fixture (embedding_length 5120, head_count 128, qk_nope 448 +
// qk_rope 64, output_group_count 16, output_lora_rank 256). DTypeSize is the
// logical f32 element size: quantized artifacts dequantize before the forward
// consumes the tensor, so the oracle compares at the consumed element type.
const (
	v41Hidden      = 5120
	v41Heads       = 128
	v41HeadDim     = 512 // qk_nope 448 + qk_rope 64
	v41OutGroups   = 16
	v41OutLoraRank = 256
	v41MHCMixWidth = 24
	f32            = 4
)

func mhcFlat() Descriptor {
	return Descriptor{Shape: []int{4 * v41Hidden, v41MHCMixWidth}, DTypeSize: f32, Groups: 1}
}

// mhcGrouped keeps the stored [4H,24] shape but declares the leading axis as
// four stream blocks of H - the grouped reading of the same byte span.
func mhcGrouped() Descriptor {
	return Descriptor{Shape: []int{4 * v41Hidden, v41MHCMixWidth}, DTypeSize: f32, Groups: 4}
}

func woAFlat() Descriptor {
	return Descriptor{Shape: []int{v41OutLoraRank, v41Heads * v41HeadDim}, DTypeSize: f32, Groups: 1}
}

// woAGrouped is the artifact's group-major declaration: [OGroups*OLoraRank,
// HeadsPerGroup*HeadDim]. Same element count as the flat form, different axis
// numbers.
func woAGrouped() Descriptor {
	hpg := v41Heads / v41OutGroups
	return Descriptor{Shape: []int{v41OutGroups * v41OutLoraRank, hpg * v41HeadDim}, DTypeSize: f32, Groups: v41OutGroups}
}

// naiveBytes is the pre-oracle admission heuristic: total bytes only, ignoring
// strides and grouping. It is the comparator the LML-02 red run proves
// insufficient.
func naiveBytes(d Descriptor) int64 {
	var n int64 = 1
	for _, s := range d.Shape {
		n *= int64(s)
	}
	return n * int64(d.DTypeSize)
}

// TestOracleCatchesFlattenedVsGroupedMHC_LML02 pins ledger row LML-02
// (model-bringup lesson MBL-027): mHC [4H,24] flattened vs grouped. Identical
// shape, identical total bytes, different leading-axis partitioning ->
// grouping refusal, named on the structured fields, not prose.
func TestOracleCatchesFlattenedVsGroupedMHC_LML02(t *testing.T) {
	t.Parallel()
	declared, actual := mhcFlat(), mhcGrouped()
	if b := naiveBytes(declared); b != naiveBytes(actual) {
		t.Fatalf("fixture sanity: naive bytes must agree, got %d vs %d", b, naiveBytes(actual))
	}
	err := CheckGeometry(declared, actual)
	if !errors.Is(err, ErrGroupingMismatch) {
		t.Fatalf("CheckGeometry(flat, grouped) = %v, want ErrGroupingMismatch", err)
	}
	m, ok := err.(*Mismatch)
	if !ok || m.Kind != "grouping" || m.Dim != 0 || m.Declared != 1 || m.Actual != 4 {
		t.Fatalf("structured fields = %#v, want grouping dim 0 declared=1 actual=4", m)
	}
	if err := CheckGeometry(actual, declared); !errors.Is(err, ErrGroupingMismatch) {
		t.Fatalf("reversed pair = %v, want ErrGroupingMismatch", err)
	}
}

// TestOracleCatchesGroupedAttnOutputA_LML02 pins the second LML-02 defect
// shape: grouped attn_output_a admitted against the flat published inventory
// declaration. The two declarations carry the same element count but assign
// different numbers to both axes, so the refusal is a per-axis geometry
// mismatch naming the first disagreeing axis.
func TestOracleCatchesGroupedAttnOutputA_LML02(t *testing.T) {
	t.Parallel()
	declared, actual := woAFlat(), woAGrouped()
	if b := naiveBytes(declared); b != naiveBytes(actual) {
		t.Fatalf("fixture sanity: naive bytes must agree, got %d vs %d", b, naiveBytes(actual))
	}
	err := CheckGeometry(declared, actual)
	if !errors.Is(err, ErrGeometryMismatch) {
		t.Fatalf("CheckGeometry(flat, grouped) = %v, want ErrGeometryMismatch", err)
	}
	m, ok := err.(*Mismatch)
	if !ok || m.Kind != "shape" || m.Dim != 0 || m.Declared != int64(v41OutLoraRank) || m.Actual != int64(v41OutGroups*v41OutLoraRank) {
		t.Fatalf("structured fields = %#v, want shape dim 0 declared=%d actual=%d", m, v41OutLoraRank, v41OutGroups*v41OutLoraRank)
	}
	// Same physical shape re-declared under a different partitioning is the
	// pure same-bytes case: geometry agrees, grouping does not.
	sameShapeFlat := Descriptor{Shape: append([]int(nil), actual.Shape...), DTypeSize: f32, Groups: 1}
	if err := CheckGeometry(sameShapeFlat, actual); !errors.Is(err, ErrGroupingMismatch) {
		t.Fatalf("same-shape regroup = %v, want ErrGroupingMismatch", err)
	}
}

// TestOracleCatchesNoncontiguousStridedView covers the ParityCommand doctrine's
// noncontiguous-tensor case: equal shape and even equal byte span do not excuse
// a stride disagreement.
func TestOracleCatchesNoncontiguousStridedView(t *testing.T) {
	t.Parallel()
	rowMajor := Descriptor{Shape: []int{2, 3}, Strides: []int{3, 1}, DTypeSize: f32, Groups: 1}
	colMajor := Descriptor{Shape: []int{2, 3}, Strides: []int{1, 2}, DTypeSize: f32, Groups: 1}
	err := CheckGeometry(rowMajor, colMajor)
	if !errors.Is(err, ErrGeometryMismatch) {
		t.Fatalf("col-major vs row-major = %v, want ErrGeometryMismatch", err)
	}
	m, _ := err.(*Mismatch)
	if m == nil || m.Kind != "stride" || m.Dim != 0 || m.Declared != 3 || m.Actual != 1 {
		t.Fatalf("structured fields = %#v, want stride dim 0 declared=3 actual=1", m)
	}

	// A strided slice of a wider buffer spans more bytes than the tight
	// contiguous box: offsets 0..(3*8+3)=27 -> 28 elements vs 16 contiguous.
	sliced := Descriptor{Shape: []int{4, 4}, Strides: []int{8, 1}, DTypeSize: f32, Groups: 1}
	contig := Descriptor{Shape: []int{4, 4}, DTypeSize: f32, Groups: 1}
	sb, err := ExpectedBytes(sliced)
	if err != nil || sb != 28*int64(f32) {
		t.Fatalf("ExpectedBytes(sliced) = %d, %v; want %d", sb, err, 28*int64(f32))
	}
	cb, err := ExpectedBytes(contig)
	if err != nil || cb != 16*int64(f32) {
		t.Fatalf("ExpectedBytes(contig) = %d, %v; want %d", cb, err, 16*int64(f32))
	}
	if err := CheckGeometry(contig, sliced); !errors.Is(err, ErrGeometryMismatch) {
		t.Fatalf("sliced vs contiguous = %v, want ErrGeometryMismatch", err)
	}
}

// TestOracleCatchesRaggedPaddedTile covers a padded row where the declared
// span is smaller than the actual span: the declared descriptor forgot the
// inter-row padding.
func TestOracleCatchesRaggedPaddedTile(t *testing.T) {
	t.Parallel()
	declared := Descriptor{Shape: []int{4, 8}, DTypeSize: f32, Groups: 1}                     // tight: 32 elements
	actual := Descriptor{Shape: []int{4, 8}, Strides: []int{9, 1}, DTypeSize: f32, Groups: 1} // padded: 3*9+8 = 35 elements
	db, err := ExpectedBytes(declared)
	if err != nil || db != 32*int64(f32) {
		t.Fatalf("ExpectedBytes(declared) = %d, %v; want %d", db, err, 32*int64(f32))
	}
	ab, err := ExpectedBytes(actual)
	if err != nil || ab != 35*int64(f32) {
		t.Fatalf("ExpectedBytes(actual) = %d, %v; want %d", ab, err, 35*int64(f32))
	}
	if !(db < ab) {
		t.Fatalf("fixture sanity: declared span %d must be < actual span %d", db, ab)
	}
	if err := CheckGeometry(declared, actual); !errors.Is(err, ErrGeometryMismatch) {
		t.Fatalf("padded tile = %v, want ErrGeometryMismatch", err)
	}
}

// TestOraclePositiveControlIdenticalDescriptorsPass is the positive control:
// identical descriptors pass, and nil strides are normalized to the explicit
// row-major strides rather than refused.
func TestOraclePositiveControlIdenticalDescriptorsPass(t *testing.T) {
	t.Parallel()
	d := Descriptor{Shape: []int{4 * v41Hidden, v41MHCMixWidth}, DTypeSize: f32, Groups: 4}
	if err := CheckGeometry(d, d); err != nil {
		t.Fatalf("identical descriptors = %v, want nil", err)
	}
	nilStrides := Descriptor{Shape: []int{4, 8}, DTypeSize: f32, Groups: 1}
	explicit := Descriptor{Shape: []int{4, 8}, Strides: []int{8, 1}, DTypeSize: f32, Groups: 1}
	if err := CheckGeometry(nilStrides, explicit); err != nil {
		t.Fatalf("nil vs explicit row-major = %v, want nil", err)
	}
	if err := CheckGeometry(explicit, nilStrides); err != nil {
		t.Fatalf("explicit vs nil row-major = %v, want nil", err)
	}
}

// TestNaiveByteComparisonAcceptsLML02Pairs_RedRun documents the red run this
// oracle kills: the naive bytes = prod(shape)*dtype comparison ACCEPTS both
// LML-02 defect pairs, because their total bytes agree. Without the oracle,
// admission of the converted GGUF under the published safetensors inventory
// would have passed these tensors; with it, each pair is refused with a typed
// sentinel. The naive acceptance below is asserted, not assumed, so a future
// fixture edit that breaks the same-bytes property fails loudly instead of
// silently vacating the test.
func TestNaiveByteComparisonAcceptsLML02Pairs_RedRun(t *testing.T) {
	t.Parallel()
	pairs := []struct {
		name string
		a, b Descriptor
	}{
		{"mHC [4H,24] flattened vs grouped", mhcFlat(), mhcGrouped()},
		{"attn_output_a flat vs grouped", woAFlat(), woAGrouped()},
	}
	for _, p := range pairs {
		naiveA, naiveB := naiveBytes(p.a), naiveBytes(p.b)
		if naiveA != naiveB {
			t.Fatalf("%s: naive bytes disagree (%d vs %d); the red-run premise (naive comparison accepts the pair) no longer holds", p.name, naiveA, naiveB)
		}
		// Red: the pre-oracle comparator accepts the pair.
		_ = naiveA == naiveB // accepted pre-oracle: true by construction above
		// Green: the oracle refuses it, pinned by sentinel.
		for _, dir := range []struct{ d, a Descriptor }{{p.a, p.b}, {p.b, p.a}} {
			err := CheckGeometry(dir.d, dir.a)
			if err == nil {
				t.Fatalf("%s: CheckGeometry accepted a same-bytes mismatched pair", p.name)
			}
			if !errors.Is(err, ErrGeometryMismatch) && !errors.Is(err, ErrGroupingMismatch) {
				t.Fatalf("%s: CheckGeometry = %v, want ErrGeometryMismatch or ErrGroupingMismatch", p.name, err)
			}
		}
	}
}

// TestExpectedBytesOverflowRefused proves huge dims error instead of wrapping.
func TestExpectedBytesOverflowRefused(t *testing.T) {
	t.Parallel()
	cases := []Descriptor{
		{Shape: []int{1 << 40, 1 << 40}, DTypeSize: f32, Groups: 1}, // 2^80 elements
		{Shape: []int{math.MaxInt64 / 2, 3}, DTypeSize: f32, Groups: 1},
		{Shape: []int{1 << 30, 1 << 30}, DTypeSize: 1 << 4, Groups: 1}, // fits in elements, overflows bytes
	}
	for i, d := range cases {
		b, err := ExpectedBytes(d)
		if !errors.Is(err, ErrOverflow) {
			t.Fatalf("case %d: ExpectedBytes = %d, %v; want ErrOverflow", i, b, err)
		}
	}
	// Boundary that fits: 2^60 elements x 4 B = 2^62 bytes.
	ok := Descriptor{Shape: []int{1 << 30, 1 << 30}, DTypeSize: f32, Groups: 1}
	b, err := ExpectedBytes(ok)
	if err != nil || b != int64(1)<<62 {
		t.Fatalf("ExpectedBytes(boundary) = %d, %v; want 2^62", b, err)
	}
}

// TestExpectedBytesZeroSizeDefined pins the closed policy: zero-extent tensors
// are defined and occupy 0 bytes, and CheckGeometry passes between them.
func TestExpectedBytesZeroSizeDefined(t *testing.T) {
	t.Parallel()
	for _, d := range []Descriptor{
		{Shape: []int{0, 5}, DTypeSize: f32, Groups: 1},
		{Shape: []int{5, 0, 3}, DTypeSize: f32, Groups: 1},
		{Shape: []int{0}, DTypeSize: f32, Groups: 1},
	} {
		b, err := ExpectedBytes(d)
		if err != nil || b != 0 {
			t.Fatalf("ExpectedBytes(%#v) = %d, %v; want 0, nil", d.Shape, b, err)
		}
	}
	z := Descriptor{Shape: []int{0, 5}, DTypeSize: f32, Groups: 1}
	if err := CheckGeometry(z, z); err != nil {
		t.Fatalf("zero-size identical = %v, want nil", err)
	}
	// Rank-0 scalar is one element.
	scalar := Descriptor{DTypeSize: f32, Groups: 1}
	b, err := ExpectedBytes(scalar)
	if err != nil || b != f32 {
		t.Fatalf("ExpectedBytes(rank-0) = %d, %v; want %d", b, err, f32)
	}
}

// TestInvalidDescriptorsRefused pins the closed-policy refusals: every invalid
// side is named (Which) and field-named, and errors.Is resolves to
// ErrInvalidDescriptor.
func TestInvalidDescriptorsRefused(t *testing.T) {
	t.Parallel()
	base := Descriptor{Shape: []int{4, 8}, DTypeSize: f32, Groups: 1}
	cases := []struct {
		name  string
		which string
		desc  Descriptor
		field string
		index int
		value int64
	}{
		{"negative dim", "declared", Descriptor{Shape: []int{4, -8}, DTypeSize: f32, Groups: 1}, "shape", 1, -8},
		{"negative stride", "actual", Descriptor{Shape: []int{4, 8}, Strides: []int{8, -1}, DTypeSize: f32, Groups: 1}, "strides", 1, -1},
		{"zero dtype", "declared", Descriptor{Shape: []int{4, 8}, DTypeSize: 0, Groups: 1}, "dtype_size", -1, 0},
		{"zero groups", "declared", Descriptor{Shape: []int{4, 8}, DTypeSize: f32, Groups: 0}, "groups", -1, 0},
		{"nondividing groups", "actual", Descriptor{Shape: []int{4, 8}, DTypeSize: f32, Groups: 3}, "groups", 0, 3},
		{"stride rank mismatch", "declared", Descriptor{Shape: []int{4, 8}, Strides: []int{8}, DTypeSize: f32, Groups: 1}, "strides", -1, 1},
	}
	for _, c := range cases {
		var err error
		if c.which == "declared" {
			err = CheckGeometry(c.desc, base)
		} else {
			err = CheckGeometry(base, c.desc)
		}
		if !errors.Is(err, ErrInvalidDescriptor) {
			t.Fatalf("%s: CheckGeometry = %v, want ErrInvalidDescriptor", c.name, err)
		}
		id, ok := err.(*InvalidDescriptor)
		if !ok || id.Which != c.which || id.Field != c.field || id.Index != c.index || id.Value != c.value {
			t.Fatalf("%s: structured fields = %#v, want which=%s field=%s index=%d value=%d", c.name, id, c.which, c.field, c.index, c.value)
		}
	}
}
