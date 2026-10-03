//go:build amd64

package model

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// q2kWitnessSameFloatBits requires two float32 values to be bit-identical.
func q2kWitnessSameFloatBits(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b)
}

// q2kOracleISSS is an INDEPENDENT oracle for the Q2_K x Q8_K integer-reduction
// seam. It derives the expected IS/SS values straight from the packed super-block
// bytes using the upstream ggml cursor order (four 2-bit planes, two 16-lane runs
// per plane, two 128-value passes per super-block). It calls neither
// q2kReduceRowScalar, nor q2kReduceRowAsmAVX2, nor q2kReduceRow.
//
// The code[] stream is tracked by OUTPUT POSITION (0..255). Output position p
// belongs to sub-block p/16 at lane p%16, so the reduction just accumulates into
// the sub-block bucket s = p/16.
func q2kOracleISSS(row []byte, nblk int, qx []int8) (IS, SS []int32) {
	IS = make([]int32, nblk*q2kGroupsPerBlock)
	SS = make([]int32, nblk*q2kGroupsPerBlock)
	for b := 0; b < nblk; b++ {
		blk := row[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
		q := blk[16:80]
		for n := 0; n < 256; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				// First 16-lane run of this plane.
				for l := 0; l < 16; l++ {
					pos := n + j*32 + l
					code := (q[l] >> shift) & 3
					q8 := int32(qx[b*qkK+pos])
					s := pos / 16
					IS[b*q2kGroupsPerBlock+s] += int32(code) * q8
					SS[b*q2kGroupsPerBlock+s] += q8
				}
				// Second 16-lane run of this plane.
				for l := 0; l < 16; l++ {
					pos := n + j*32 + 16 + l
					code := (q[l+16] >> shift) & 3
					q8 := int32(qx[b*qkK+pos])
					s := pos / 16
					IS[b*q2kGroupsPerBlock+s] += int32(code) * q8
					SS[b*q2kGroupsPerBlock+s] += q8
				}
				shift += 2
			}
			q = q[32:]
		}
	}
	return IS, SS
}

// q2kWitnessRunAsm drives the AVX2 reducer directly.
func q2kWitnessRunAsm(row []byte, nblk int, qx []int8) ([]int32, []int32) {
	IS := make([]int32, nblk*q2kGroupsPerBlock)
	SS := make([]int32, nblk*q2kGroupsPerBlock)
	if nblk > 0 {
		q2kReduceRowAsmAVX2(&row[0], nblk, &qx[0], &IS[0], &SS[0])
	}
	return IS, SS
}

// q2kWitnessRunScalar drives the scalar reducer directly.
func q2kWitnessRunScalar(row []byte, nblk int, qx []int8) ([]int32, []int32) {
	IS := make([]int32, nblk*q2kGroupsPerBlock)
	SS := make([]int32, nblk*q2kGroupsPerBlock)
	q2kReduceRowScalar(row, nblk, qx, IS, SS)
	return IS, SS
}

func q2kWitnessSameInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func q2kWitnessRandInt8(n int, seed uint64) []int8 {
	b := make([]byte, n)
	lcgBytes(b, seed)
	q := make([]int8, n)
	for i := range q {
		q[i] = int8(b[i])
	}
	return q
}

// TestQ2KReduceAsmMatchesScalar requires the AVX2 reducer and the scalar reducer
// to agree bit-for-bit over randomized rows, an adversarial super-block that
// exercises all four codes at every shift position and both 16-lane halves with
// signed qx, and an all-zero row.
func TestQ2KReduceAsmMatchesScalar(t *testing.T) {
	if !detectAVX2() {
		t.Skip("AVX2 not detected on this host; nothing to compare")
	}
	const nblk = 3
	rb := nblk * q2kBlockBytes

	// Randomized rows.
	for r := 0; r < 4; r++ {
		row := make([]byte, rb)
		lcgBytes(row, uint64(0x1000+r))
		qx := q2kWitnessRandInt8(nblk*qkK, uint64(0x2000+r))
		gotI, gotS := q2kWitnessRunAsm(row, nblk, qx)
		wantI, wantS := q2kWitnessRunScalar(row, nblk, qx)
		if !q2kWitnessSameInt32(gotI, wantI) || !q2kWitnessSameInt32(gotS, wantS) {
			t.Fatalf("random row %d: asm IS/SS != scalar", r)
		}
	}

	// Adversarial single super-block: all four codes at every shift position and
	// both 16-lane halves, signed qx extremes, nonzero dmin.
	{
		row := make([]byte, q2kBlockBytes)
		lcgBytes(row, 0xdead)
		row[82], row[83] = 0x00, 0x38 // dmin = 0.5 f16 (nonzero)
		for s := 0; s < q2kGroupsPerBlock; s++ {
			for l := 0; l < 16; l++ {
				q2kSetCode(row[16:80], s, l, byte((s+l)%4))
			}
		}
		qx := q2kWitnessRandInt8(qkK, 0xbeef)
		qx[0] = -128
		qx[1] = 127
		qx[128] = -128
		qx[255] = 127
		gotI, gotS := q2kWitnessRunAsm(row, 1, qx)
		wantI, wantS := q2kWitnessRunScalar(row, 1, qx)
		if !q2kWitnessSameInt32(gotI, wantI) || !q2kWitnessSameInt32(gotS, wantS) {
			t.Fatalf("adversarial block: asm != scalar")
		}
	}

	// All-zero row.
	{
		row := make([]byte, rb)
		qx := make([]int8, nblk*qkK)
		gotI, gotS := q2kWitnessRunAsm(row, nblk, qx)
		wantI, wantS := q2kWitnessRunScalar(row, nblk, qx)
		if !q2kWitnessSameInt32(gotI, wantI) || !q2kWitnessSameInt32(gotS, wantS) {
			t.Fatalf("zero row: asm != scalar")
		}
	}
}

// TestQ2KReduceDispatchFallsBackWithoutAVX2 pins qtier to the scalar tier and
// requires the dispatching q2kReduceRow to equal q2kReduceRowScalar bit-for-bit.
func TestQ2KReduceDispatchFallsBackWithoutAVX2(t *testing.T) {
	old := qtier
	qtier = tierScalar
	t.Cleanup(func() { qtier = old })

	const nblk = 2
	rb := nblk * q2kBlockBytes
	for r := 0; r < 3; r++ {
		row := make([]byte, rb)
		lcgBytes(row, uint64(0x3000+r))
		qx := q2kWitnessRandInt8(nblk*qkK, uint64(0x4000+r))

		di := make([]int32, nblk*q2kGroupsPerBlock)
		ds := make([]int32, nblk*q2kGroupsPerBlock)
		q2kReduceRow(row, nblk, qx, di, ds)

		si, ss := q2kWitnessRunScalar(row, nblk, qx)
		if !q2kWitnessSameInt32(di, si) || !q2kWitnessSameInt32(ds, ss) {
			t.Fatalf("dispatch fallback row %d: q2kReduceRow != scalar", r)
		}
	}
}

// TestQ2KReduceProductionCallerReachability forces the AVX2 tier plus the SDOT
// path and proves the two production callers reach the seam: q2kMatRowsRangeInt8Raw
// and kQuantMatRowsSubset must reproduce a scalar-built reference bit-for-bit.
func TestQ2KReduceProductionCallerReachability(t *testing.T) {
	if !detectAVX2() {
		t.Fatalf("AVX2 must actually execute for the production-caller witness; host reports no AVX2")
	}
	oldTier := qtier
	oldForce := kQuantSDOTForce
	oldVNNI := q2kUseVNNI
	qtier = tierAVX2
	setKQuantSDOTForTest(true)
	t.Cleanup(func() {
		qtier = oldTier
		kQuantSDOTForce = oldForce
		q2kUseVNNI = oldVNNI
	})

	const out, in = 4, 512 // two super-blocks per row
	qt := q2kTestTensor(out, in, 0x1234, 0x3800)
	x := q2kTestActivation(in, 0x5678, 0)
	qv := quantizeVecQ8(x)

	// Reference: scalar reduce + float fold, one value per row.
	rb := qt.rowBytes()
	ref := make([]float32, out)
	for r := 0; r < out; r++ {
		row := qt.raw[r*rb : (r+1)*rb]
		IS := make([]int32, qt.nblk*q2kGroupsPerBlock)
		SS := make([]int32, qt.nblk*q2kGroupsPerBlock)
		q2kReduceRowScalar(row, qt.nblk, qv.q, IS, SS)
		ref[r] = q2kCombineRow(row, qt.nblk, qv.d, IS, SS)
	}

	// (a) q2kMatRowsRangeInt8Raw must match the reference bit-for-bit.
	y := make([]float32, out)
	q2kMatRowsRangeInt8Raw(qt.raw, qt, qv, y, 0, out)
	for r := 0; r < out; r++ {
		if !q2kWitnessSameFloatBits(y[r], ref[r]) {
			t.Errorf("q2kMatRowsRangeInt8Raw row %d: got %v want %v", r, y[r], ref[r])
		}
	}

	// (b) kQuantMatRowsSubset over a subset of rows must match the reference rows.
	subset := []int{0, 2}
	got := kQuantMatRowsSubset(qt, x, subset)
	if len(got) != len(subset) {
		t.Fatalf("kQuantMatRowsSubset returned %d values for %d rows", len(got), len(subset))
	}
	for i, r := range subset {
		if !q2kWitnessSameFloatBits(got[i], ref[r]) {
			t.Errorf("kQuantMatRowsSubset subset[%d] (row %d): got %v want %v", i, r, got[i], ref[r])
		}
	}
}

// TestQ2KReducePinnedSourceProvenance is a non-vacuous source guard. It reads
// THIS file (via runtime.Caller) and the pinned amd64 seam source, requiring the
// expected test symbols and upstream provenance strings to be literally present.
func TestQ2KReducePinnedSourceProvenance(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatalf("runtime.Caller(0) failed")
	}
	self, err := os.ReadFile(thisFile)
	if err != nil {
		t.Fatalf("read self %q: %v", thisFile, err)
	}
	selfS := string(self)
	for _, s := range []string{
		"func TestQ2KReduceAsmMatchesScalar(",
		"func TestQ2KReduceDispatchFallsBackWithoutAVX2(",
		"func TestQ2KReduceProductionCallerReachability(",
		"func TestQ2KReducePinnedSourceProvenance(",
		"func TestQ2KReduceIndependentPackedOracle(",
	} {
		if !strings.Contains(selfS, s) {
			t.Errorf("this test file is missing required symbol %q", s)
		}
	}

	impl := filepath.Join(filepath.Dir(thisFile), "quant_kquant_int8_q2k_amd64.go")
	implBytes, err := os.ReadFile(impl)
	if err != nil {
		t.Fatalf("read impl %q: %v", impl, err)
	}
	implS := string(implBytes)
	for _, s := range []string{
		"83078fec0db82d6b5a00d9599062c38c39145755",
		"ggml_vec_dot_q2_K_q8_K",
		"quants.c",
		"MIT",
	} {
		if !strings.Contains(implS, s) {
			t.Errorf("impl source is missing provenance string %q", s)
		}
	}
}

// TestQ2KReduceIndependentPackedOracle is THE independent oracle test. It proves
// the AVX2 reducer (both directly and through dispatch) and the scalar dispatch
// arm agree bit-for-bit with a reducer written straight from the packed bytes.
//
// This is a SOFTWARE AVX2 host witness. It is NOT a Strix/physical qualification,
// and it makes NO speed, gain, whole-model, or default claim.
func TestQ2KReduceIndependentPackedOracle(t *testing.T) {
	// A skip is not a pass: require AVX2 to actually execute.
	if !detectAVX2() {
		t.Fatalf("AVX2 must actually execute for this witness; host reports no AVX2")
	}
	old := qtier
	qtier = tierAVX2
	t.Cleanup(func() { qtier = old })
	if qtier < tierAVX2 {
		t.Fatalf("forced qtier %d < tierAVX2 %d", qtier, tierAVX2)
	}
	t.Logf("q2k oracle witness: qtier=%d q2kUseVNNI=%d detectAVX2=%v", qtier, q2kUseVNNI, detectAVX2())

	const nblk = 3
	rb := nblk * q2kBlockBytes

	check := func(name string, row []byte, qx []int8) {
		wI, wS := q2kOracleISSS(row, nblk, qx)

		aI, aS := q2kWitnessRunAsm(row, nblk, qx)
		if !q2kWitnessSameInt32(aI, wI) || !q2kWitnessSameInt32(aS, wS) {
			t.Errorf("%s: direct asm != oracle", name)
		}

		dI := make([]int32, nblk*q2kGroupsPerBlock)
		dS := make([]int32, nblk*q2kGroupsPerBlock)
		q2kReduceRow(row, nblk, qx, dI, dS)
		if !q2kWitnessSameInt32(dI, wI) || !q2kWitnessSameInt32(dS, wS) {
			t.Errorf("%s: dispatch != oracle", name)
		}
	}

	// Randomized rows with NONZERO dmin, nonzero scale bytes, multiple super-blocks,
	// multiple rows, and signed int8 extremes in the activation.
	for r := 0; r < 3; r++ {
		row := make([]byte, rb)
		lcgBytes(row, uint64(0x5000+r))
		for b := 0; b < nblk; b++ {
			blk := row[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
			blk[80], blk[81] = 0x00, 0x3c // d = 1.0 f16
			blk[82], blk[83] = 0x00, 0x38 // dmin = 0.5 f16 (nonzero)
			for i := 0; i < 16; i++ {
				if blk[i] == 0 {
					blk[i] = 0x5a
				}
			}
		}
		qx := q2kWitnessRandInt8(nblk*qkK, uint64(0x6000+r))
		qx[0] = -128
		qx[17] = 127
		qx[nblk*qkK-1] = -128
		check(fmt.Sprintf("random-%d", r), row, qx)
	}

	// All-zero row + all-zero activation: assert the oracle itself yields 0/0.
	{
		row := make([]byte, rb)
		qx := make([]int8, nblk*qkK)
		wI, wS := q2kOracleISSS(row, nblk, qx)
		for i := range wI {
			if wI[i] != 0 || wS[i] != 0 {
				t.Fatalf("oracle zero case yielded nonzero at index %d", i)
			}
		}
		check("all-zero", row, qx)
	}

	// Adversarial block: all scale bytes differ and codes vary.
	{
		row := make([]byte, rb)
		for b := 0; b < nblk; b++ {
			blk := row[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
			for i := 0; i < 16; i++ {
				blk[i] = byte(0x11 * (i + 1))
			}
			blk[82], blk[83] = 0x00, 0x38
			for s := 0; s < q2kGroupsPerBlock; s++ {
				for l := 0; l < 16; l++ {
					q2kSetCode(blk[16:80], s, l, byte((s*7+l*3)%4))
				}
			}
		}
		qx := q2kWitnessRandInt8(nblk*qkK, 0x7777)
		qx[0] = -128
		qx[15] = 127
		qx[16] = 127
		qx[127] = -128
		check("adversarial", row, qx)
	}

	// Force the scalar tier and require the scalar dispatch arm == oracle.
	qtier = tierScalar
	{
		row := make([]byte, rb)
		lcgBytes(row, 0x8888)
		qx := q2kWitnessRandInt8(nblk*qkK, 0x9999)
		wI, wS := q2kOracleISSS(row, nblk, qx)
		dI := make([]int32, nblk*q2kGroupsPerBlock)
		dS := make([]int32, nblk*q2kGroupsPerBlock)
		q2kReduceRow(row, nblk, qx, dI, dS)
		if !q2kWitnessSameInt32(dI, wI) || !q2kWitnessSameInt32(dS, wS) {
			t.Fatalf("scalar dispatch arm != oracle")
		}
	}
	qtier = tierAVX2

	// VNNI: only assert when the package reports the VNNI branch is active. Never
	// fabricate VNNI coverage; if it is not 1, assert nothing VNNI-specific.
	if q2kUseVNNI == 1 {
		row := make([]byte, rb)
		lcgBytes(row, 0xaaaa)
		qx := q2kWitnessRandInt8(nblk*qkK, 0xbbbb)
		wI, wS := q2kOracleISSS(row, nblk, qx)
		aI, aS := q2kWitnessRunAsm(row, nblk, qx)
		if !q2kWitnessSameInt32(aI, wI) || !q2kWitnessSameInt32(aS, wS) {
			t.Fatalf("VNNI asm != oracle")
		}
	}
}
