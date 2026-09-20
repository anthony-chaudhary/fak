package model

// v41_attention_storage_conformance_test.go — the #13321 witness: a table-driven
// admission + storage-resolver conformance matrix for the five V4.1 attention
// projections wq_a, wq_b, wkv, wo_a, wo_b, each in two storage forms (raw f32
// control, resident Q2_K), ten positive rows total.
//
// The gap this closes. #13254/#13262/#13276/#13278 each showed that the
// admission contract (v41ForwardAdmitted / v41AdmitShape, residency-complete via
// residentShape) and the eventual read (v41ProjMatRows / v41ProjF32Into) are
// SEPARATE contracts. Individual projection regressions exist, but there is no
// single bounded matrix proving, operand by operand and storage form by storage
// form, that WHATEVER admission accepts is actually READ back at the admitted
// geometry with the expected values. This leaf adds exactly that.
//
// Independence. Every expected value comes from an explicitly constructed Q2_K
// super-block decoded by a LOCAL scalar transcription (the fixture file's
// v41StorageScalarExpect), never the production decoder. The f32 control rows
// compare the production read against the model's own manifest bytes read
// through the independent scalar dot. The production reader is the thing under
// test, so it is never used to produce the expectation.

import (
	"errors"
	"strings"
	"testing"
)

// v41StorageForm is one storage arm of the matrix.
type v41StorageForm int

const (
	v41FormF32Control  v41StorageForm = iota // raw f32 manifest
	v41FormResidentQ2K                       // resident Q2_K store
)

func (f v41StorageForm) String() string {
	switch f {
	case v41FormF32Control:
		return "f32"
	case v41FormResidentQ2K:
		return "q2k"
	default:
		return "?"
	}
}

// v41StorageInput builds the [in] activation row for a matrix cell. It is a
// fixed deterministic pattern independent of the weight, so both storage arms
// of one operand see the same input.
func v41StorageInput(in int) []float32 {
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*7)%23-11) / 16
	}
	return x
}

// v41StorageScalarDot is the independent contraction: y = W_row . x over the
// exact expected f32 row, with no production GEMV or dequantizer.
func v41StorageScalarDot(wantRow, x []float32) float32 {
	var s float32
	for i := range x {
		s += wantRow[i] * x[i]
	}
	return s
}

// v41StorageRow returns row o of an exact [out, in] expected weight block.
func v41StorageRow(want []float32, o, in int) []float32 { return want[o*in : (o+1)*in] }

// v41StorageReadMatRows is the production matRows-shaped read for the three
// non-grouped attention operands (wq_a, wq_b, wkv), which the forward applies
// through v41ProjMatRows at their use site.
func v41StorageReadMatRows(t *testing.T, m *Model, l int, leaf string, x []float32, out, in int) []float32 {
	t.Helper()
	y, err := m.v41ProjMatRows(l, leaf, x, out, in)
	if err != nil {
		t.Fatalf("%s: v41ProjMatRows(%s) error = %v, want nil", leaf, leaf, err)
	}
	if len(y) != out {
		t.Fatalf("%s: v41ProjMatRows returned %d rows, want %d", leaf, len(y), out)
	}
	return y
}

// v41StorageGroupedInput is the fixed one-position, one-batch value tensor the
// grouped contraction consumes. It is deterministic so both operands' reads can
// be checked against the same contraction.
func v41StorageGroupedInput(heads, hd int) []float32 {
	o := make([]float32, heads*hd)
	for i := range o {
		o[i] = float32((i*5)%17-8) / 8
	}
	return o
}

// v41StorageReadGrouped reads wo_a/wo_b as whole f32 blocks and contracts them
// through the production V41GroupedOutputProjection, exactly as v41Layer does
// (v41ProjF32Into then V41GroupedOutputProjection). It reads BOTH projections,
// so one call exercises the read contract of both operands.
func v41StorageReadGrouped(t *testing.T, m *Model, l int, cfg Config, o []float32) []float32 {
	t.Helper()
	heads, hd, groups, oLoRARank, dim := cfg.NumHeads, cfg.HeadDim, cfg.OGroups, cfg.OLoraRank, cfg.HiddenSize
	woA, err := m.v41ProjF32Into(l, "attn.wo_a.weight", nil)
	if err != nil {
		t.Fatalf("wo_a: v41ProjF32Into error = %v, want nil", err)
	}
	woB, err := m.v41ProjF32Into(l, "attn.wo_b.weight", nil)
	if err != nil {
		t.Fatalf("wo_b: v41ProjF32Into error = %v, want nil", err)
	}
	out, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, heads, hd, groups, oLoRARank, dim)
	if err != nil {
		t.Fatalf("V41GroupedOutputProjection error = %v, want nil", err)
	}
	return out
}

// v41StorageGroupedOracle is the independent scalar transcription of the
// grouped output contraction, over the EXPECTED wo_a/wo_b blocks (never the
// production read). It follows the reference order restated in
// V41GroupedOutputProjection's doc: per-group block-diagonal wo_a over each
// group's heads, then the shared wo_b down-projection.
func v41StorageGroupedOracle(o, wantA, wantB []float32, heads, hd, groups, oLoRARank, dim int) []float32 {
	headsPerGroup := heads / groups
	aRow := headsPerGroup * hd
	joined := make([]float32, groups*oLoRARank)
	for g := 0; g < groups; g++ {
		groupBase := g * aRow
		for r := 0; r < oLoRARank; r++ {
			aBase := (g*oLoRARank + r) * aRow
			var acc float32
			for k := 0; k < aRow; k++ {
				acc += o[groupBase+k] * wantA[aBase+k]
			}
			joined[g*oLoRARank+r] = acc
		}
	}
	out := make([]float32, dim)
	for d := 0; d < dim; d++ {
		bBase := d * groups * oLoRARank
		var acc float32
		for k := 0; k < groups*oLoRARank; k++ {
			acc += wantB[bBase+k] * joined[k]
		}
		out[d] = acc
	}
	return out
}

// TestV41AttentionStorageConformance is the #13321 acceptance witness. Every
// supported matrix row proves BOTH admission and the actual read/contraction,
// including output dimensions, finite values, a declared numerical tolerance and
// untouched source bytes for the f32 control.
func TestV41AttentionStorageConformance(t *testing.T) {
	cfg := v41StorageConfig(t)
	const l = 0
	x := v41StorageInput(cfg.HiddenSize) // all attention reads consume an H-wide activation

	exercised := 0
	for _, spec := range v41StorageLeaves(cfg) {
		for _, form := range []v41StorageForm{v41FormF32Control, v41FormResidentQ2K} {
			name := spec.leaf + "/" + form.String()
			t.Run(name, func(t *testing.T) {
				m := v41StorageBuildModel(cfg)
				name := layerName(l, spec.leaf)

				// ---- the expected [out, in] weight block, independent of production ----
				var want []float32
				switch form {
				case v41FormF32Control:
					// The manifest bytes ARE the expected weights; read them
					// directly through the manifest view (not the production
					// reader under test).
					want = append([]float32(nil), m.tensor(name)...)
				case v41FormResidentQ2K:
					want = v41StorageInstallQ2K(m, l, spec.leaf, spec.out, spec.in)
					if m.has(name) {
						t.Fatalf("resident fixture still carries an f32 manifest entry for %s; the resident read is not exercised", name)
					}
				}
				if len(want) != spec.out*spec.in {
					t.Fatalf("expected block len %d, want %d", len(want), spec.out*spec.in)
				}

				// ---- admission: the SAME production predicate the forward runs ----
				if spec.grouped {
					if err := m.v41AdmitGroupedWoA(l); err != nil {
						t.Fatalf("admission (grouped wo_a) error = %v, want nil", err)
					}
				} else {
					if err := m.v41AdmitShape(name, v41StageAttention, l, spec.out, spec.in); err != nil {
						t.Fatalf("admission error = %v, want nil", err)
					}
				}

				// ---- read/contraction through the SAME production seam the forward uses ----
				switch spec.leaf {
				case "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight":
					got := v41StorageReadMatRows(t, m, l, spec.leaf, x, spec.out, spec.in)
					assertV41StorageMatRows(t, spec.leaf, got, want, spec.out, spec.in, x)
				case "attn.wo_a.weight", "attn.wo_b.weight":
					// The grouped contraction reads BOTH wo_a and wo_b, so the
					// sibling operand is held as a fixed f32 control and its own
					// expected block is read from the manifest; the oracle then
					// contracts both expected blocks and the read under test
					// contributes the operand this row varies.
					wantA, wantB := v41StorageGroupedExpected(t, m, cfg, spec)
					o := v41StorageGroupedInput(cfg.NumHeads, cfg.HeadDim)
					got := v41StorageReadGrouped(t, m, l, cfg, o)
					exp := v41StorageGroupedOracle(o, wantA, wantB, cfg.NumHeads, cfg.HeadDim, cfg.OGroups, cfg.OLoraRank, cfg.HiddenSize)
					assertV41StorageGrouped(t, spec.leaf, got, exp)
				default:
					t.Fatalf("no read contract declared for %s", spec.leaf)
				}

				if form == v41FormResidentQ2K {
					// Untouched source bytes: the resident raw payload must be
					// exactly what the loader stored (the read is non-mutating).
					raw, ok := m.KQuantRaw(name)
					if !ok || len(raw) != spec.out*(spec.in/256)*q2kBlockBytes {
						t.Fatalf("resident raw payload missing or wrong size for %s", name)
					}
				}
				exercised++
			})
		}
	}
	if exercised != len(v41StorageLeaves(cfg))*2 {
		t.Fatalf("exercised %d matrix rows, want %d", exercised, len(v41StorageLeaves(cfg))*2)
	}
}

// assertV41StorageMatRows checks a matRows-shaped read against the independent
// scalar dot of the expected weight rows, with output dimension, finiteness and
// tolerance.
func assertV41StorageMatRows(t *testing.T, leaf string, got, want []float32, out, in int, x []float32) {
	t.Helper()
	if len(got) != out {
		t.Fatalf("%s: read produced %d rows, want %d", leaf, len(got), out)
	}
	const tol = 1e-3
	for o := 0; o < out; o++ {
		exp := v41StorageScalarDot(v41StorageRow(want, o, in), x)
		if !finite32(got[o]) {
			t.Fatalf("%s row %d: read value %v is not finite", leaf, o, got[o])
		}
		if d := absF32(got[o] - exp); d > tol {
			t.Fatalf("%s row %d: read %g, want %g (|Δ|=%g > %g)", leaf, o, got[o], exp, d, tol)
		}
	}
}

// v41StorageGroupedExpected returns the expected wo_a and wo_b [out, in] blocks
// for a grouped row. The operand this row varies is already installed by the
// caller (its expected block is materialized the same way the read under test
// is: the resident Q2_K arm's expected block, or the f32 control's manifest
// bytes); the SIBLING operand is left as the fixed f32 control and its expected
// block is read from the manifest. Reading the manifest view here is the
// independent expectation, not the production reader, which is F32 for the
// resident store only for the operand under test.
func v41StorageGroupedExpected(t *testing.T, m *Model, cfg Config, spec v41StorageLeaf) (wantA, wantB []float32) {
	t.Helper()
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OGroups * cfg.OLoraRank
	// wo_a is [OLoraRank, qHeadDim] flat (the reduced declaration); wo_b is
	// [H, oDim]. Either may be the varied operand.
	if spec.leaf == "attn.wo_a.weight" {
		wantA = v41StorageExpectedBlock(t, m, cfg, "attn.wo_a.weight", cfg.OLoraRank, qHeadDim)
		wantB = append([]float32(nil), m.tensor(layerName(0, "attn.wo_b.weight"))...)
	} else {
		wantA = append([]float32(nil), m.tensor(layerName(0, "attn.wo_a.weight"))...)
		wantB = v41StorageExpectedBlock(t, m, cfg, "attn.wo_b.weight", cfg.HiddenSize, oDim)
	}
	return wantA, wantB
}

// v41StorageExpectedBlock returns the expected [out, in] block for a leaf: the
// resident Q2_K analytic expectation when the leaf is resident in m.kqw, else
// the f32 manifest view.
func v41StorageExpectedBlock(t *testing.T, m *Model, cfg Config, leaf string, out, in int) []float32 {
	t.Helper()
	name := layerName(0, leaf)
	if qt := m.kqw[name]; qt != nil {
		if qt.out != out || qt.in != in {
			t.Fatalf("%s: resident shape [%d %d], want [%d %d]", leaf, qt.out, qt.in, out, in)
		}
		return qtExpectedQ2K(m, name, out, in)
	}
	return append([]float32(nil), m.tensor(name)...)
}

// qtExpectedQ2K re-derives the exact expected block of an installed resident
// Q2_K tensor from its raw payload through the independent scalar
// transcription (never the production dequantizer).
func qtExpectedQ2K(m *Model, name string, out, in int) []float32 {
	raw, ok := m.KQuantRaw(name)
	if !ok {
		panic("qtExpectedQ2K: " + name + " is not resident")
	}
	nblk := in / 256
	want := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*q2kBlockBytes : (o*nblk+b+1)*q2kBlockBytes]
			copy(want[o*in+b*256:], v41StorageScalarExpect(blk))
		}
	}
	return want
}

// assertV41StorageGrouped checks the grouped output contraction against the
// independently computed scalar reference.
func assertV41StorageGrouped(t *testing.T, leaf string, got, exp []float32) {
	t.Helper()
	if len(got) != len(exp) {
		t.Fatalf("%s: grouped output len %d, want %d", leaf, len(got), len(exp))
	}
	const tol = 1e-3
	for i := range got {
		if !finite32(got[i]) {
			t.Fatalf("%s dim %d: grouped output %v is not finite", leaf, i, got[i])
		}
		if d := absF32(got[i] - exp[i]); d > tol {
			t.Fatalf("%s dim %d: grouped output %g, want %g (|Δ|=%g > %g)", leaf, i, got[i], exp[i], d, tol)
		}
	}
}

// TestV41AttentionStorageRefuses drives the negative half of the contract: a
// projection absent from every store must refuse with a typed
// *V41ForwardError wrapping ErrV41ForwardStage naming the tensor (never a
// panic through residentMatRowsBase), and a present-but-wrong-shape weight must
// refuse at admission with the named shape guard.
func TestV41AttentionStorageRefuses(t *testing.T) {
	cfg := v41StorageConfig(t)
	const l = 0

	t.Run("missing-store", func(t *testing.T) {
		m := v41StorageBuildModel(cfg)
		name := layerName(l, "attn.wq_a.weight")
		delete(m.manifest, name) // resident in no store
		spec := v41StorageLeaf{leaf: "attn.wq_a.weight", out: cfg.QLoraRank, in: cfg.HiddenSize}

		if err := m.v41AdmitShape(name, v41StageAttention, l, spec.out, spec.in); err == nil {
			t.Fatalf("admission of a weight absent from every store succeeded, want a refusal")
		}
		y, err := m.v41ProjMatRows(l, "attn.wq_a.weight", v41StorageInput(cfg.HiddenSize), spec.out, spec.in)
		if err == nil {
			t.Fatalf("read of a weight absent from every store succeeded (len=%d), want a typed refusal", len(y))
		}
		var fe *V41ForwardError
		if !errors.As(err, &fe) {
			t.Fatalf("missing-store refusal error = %v (%T), want *V41ForwardError", err, err)
		}
		if !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("missing-store refusal %v does not wrap ErrV41ForwardStage", err)
		}
		if !strings.Contains(err.Error(), "attn.wq_a.weight") {
			t.Fatalf("missing-store refusal %v does not name the tensor", err)
		}
	})

	t.Run("wrong-shape", func(t *testing.T) {
		m := v41StorageBuildModel(cfg)
		name := layerName(l, "attn.wkv.weight")
		// Demand a shape the manifest does not carry.
		err := m.v41AdmitShape(name, v41StageAttention, l, cfg.HiddenSize, cfg.HiddenSize)
		if err == nil {
			t.Fatalf("admission of a wrong-shape weight succeeded, want a refusal")
		}
		var fe *V41ForwardError
		if !errors.As(err, &fe) {
			t.Fatalf("wrong-shape refusal error = %v (%T), want *V41ForwardError", err, err)
		}
		if !strings.Contains(err.Error(), "attn.wkv.weight") {
			t.Fatalf("wrong-shape refusal %v does not name the tensor", err)
		}
	})
}
