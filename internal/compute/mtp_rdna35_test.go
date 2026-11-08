package compute

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// TestMTPK4CausalVerificationTreeMask verifies that the 2D causal verification
// tree mask generator satisfies Mask[i, j] = 1 for j <= i, and 0 for j > i.
func TestMTPK4CausalVerificationTreeMask(t *testing.T) {
	mask := MTPK4CausalVerificationTreeMask()

	// Verify exact 4x4 matrix values
	expected := [4][4]float32{
		{1.0, 0.0, 0.0, 0.0},
		{1.0, 1.0, 0.0, 0.0},
		{1.0, 1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0, 1.0},
	}

	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if mask[i][j] != expected[i][j] {
				t.Errorf("mask[%d][%d] = %g, want %g", i, j, mask[i][j], expected[i][j])
			}
			if j <= i {
				if mask[i][j] != 1.0 {
					t.Errorf("causal invariant violated: mask[%d][%d] must be 1.0 for j <= i", i, j)
				}
				if !IsCausalVerificationMaskAllowed(i, j) {
					t.Errorf("IsCausalVerificationMaskAllowed(%d, %d) must be true", i, j)
				}
			} else {
				if mask[i][j] != 0.0 {
					t.Errorf("future mask violated: mask[%d][%d] must be 0.0 for j > i", i, j)
				}
				if IsCausalVerificationMaskAllowed(i, j) {
					t.Errorf("IsCausalVerificationMaskAllowed(%d, %d) must be false", i, j)
				}
			}
		}
	}

	// Verify general K causal tree mask
	genMask := CausalVerificationTreeMask(4)
	if len(genMask) != 4 {
		t.Fatalf("len(genMask) = %d, want 4", len(genMask))
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if genMask[i][j] != expected[i][j] {
				t.Errorf("genMask[%d][%d] = %g, want %g", i, j, genMask[i][j], expected[i][j])
			}
		}
	}

	// Verify additive attention bias mask
	bias := CausalAttentionBiasMask(4)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if j <= i {
				if bias[i][j] != 0.0 {
					t.Errorf("bias[%d][%d] = %g, want 0.0 (allowed)", i, j, bias[i][j])
				}
			} else {
				if bias[i][j] > -1e8 {
					t.Errorf("bias[%d][%d] = %g, want negative infinity (masked)", i, j, bias[i][j])
				}
			}
		}
	}
}

// TestMTPSinglePassWeightReuseAcross40CUs witnesses single-pass weight reuse,
// 40 CU execution, and quadrupled arithmetic intensity from LPDDR5X DRAM.
func TestMTPSinglePassWeightReuseAcross40CUs(t *testing.T) {
	outDim := 128
	inDim := 64
	weights := make([]float32, outDim*inDim)
	for i := range weights {
		weights[i] = float32(i%17) * 0.05
	}

	drafts := make([][]float32, 4)
	for k := 0; k < 4; k++ {
		drafts[k] = make([]float32, inDim)
		for d := 0; d < inDim; d++ {
			drafts[k][d] = float32((k+1)*(d+1)%13) * 0.1
		}
	}

	treeMask := MTPK4CausalVerificationTreeMask()
	outputs, audit, err := MTPK4MicroBatchVerify(weights, outDim, inDim, drafts, treeMask)
	if err != nil {
		t.Fatalf("MTPK4MicroBatchVerify failed: %v", err)
	}

	if len(outputs) != 4 {
		t.Fatalf("len(outputs) = %d, want 4", len(outputs))
	}
	for k := 0; k < 4; k++ {
		if len(outputs[k]) != outDim {
			t.Fatalf("outputs[%d] length = %d, want %d", k, len(outputs[k]), outDim)
		}
	}

	// Verify audit invariants on gfx1151
	if audit.ComputeUnitsEngaged <= 0 || audit.ComputeUnitsEngaged > StrixHaloComputeUnits {
		t.Errorf("ComputeUnitsEngaged = %d, want <= %d", audit.ComputeUnitsEngaged, StrixHaloComputeUnits)
	}
	if audit.WavefrontSize != 32 {
		t.Errorf("WavefrontSize = %d, want 32 (Wave32)", audit.WavefrontSize)
	}
	if audit.DraftDepthK != 4 {
		t.Errorf("DraftDepthK = %d, want 4", audit.DraftDepthK)
	}
	if !audit.CausalTreeMaskApplied {
		t.Errorf("CausalTreeMaskApplied must be true")
	}

	// Verify weight reuse ratio: sequential reads 4x weights, single pass reads 1x weights
	// Expect WeightReuseRatio close to ~4.0x
	if audit.WeightReuseRatio < 3.5 || audit.WeightReuseRatio > 4.1 {
		t.Errorf("WeightReuseRatio = %g, want ~4.0x", audit.WeightReuseRatio)
	}

	// Verify arithmetic intensity: ~2.0 FLOP/byte for 4 draft proposals
	if audit.ArithmeticIntensity < 1.5 {
		t.Errorf("ArithmeticIntensity = %g FLOP/byte, want >= 1.5 FLOP/byte", audit.ArithmeticIntensity)
	}

	// Verify correctness against direct computation
	for k := 0; k < 4; k++ {
		for r := 0; r < outDim; r++ {
			var expected float32
			wBase := r * inDim
			for d := 0; d < inDim; d++ {
				expected += weights[wBase+d] * drafts[k][d]
			}
			diff := math.Abs(float64(outputs[k][r] - expected))
			if diff > 1e-4 {
				t.Fatalf("token %d row %d mismatch: got %g, want %g (diff %g)", k, r, outputs[k][r], expected, diff)
			}
		}
	}
}

// TestMTPAcceptanceScoringAndRollback verifies sequential prefix acceptance,
// rollback of rejected speculative branches, and speedup evaluation in the 75-85% band.
func TestMTPAcceptanceScoringAndRollback(t *testing.T) {
	// Case 1: All 4 tokens accepted (100% acceptance, 0 rollbacks)
	draftAll := []int{101, 102, 103, 104}
	targetAll := []int{101, 102, 103, 104, 105} // including 5th bonus token
	resAll := EvaluateDraftAcceptance(draftAll, targetAll)
	if resAll.AcceptedCount != 4 {
		t.Errorf("resAll.AcceptedCount = %d, want 4", resAll.AcceptedCount)
	}
	if resAll.RollbackCount != 0 {
		t.Errorf("resAll.RollbackCount = %d, want 0", resAll.RollbackCount)
	}
	if resAll.RejectedAt != -1 {
		t.Errorf("resAll.RejectedAt = %d, want -1", resAll.RejectedAt)
	}
	if len(resAll.NextTokens) != len(targetAll) || resAll.NextTokens[4] != 105 {
		t.Errorf("resAll.NextTokens = %+v, want %d tokens ending with 105", resAll.NextTokens, len(targetAll))
	}

	// Case 2: First 3 tokens accepted, 4th rejected (75% acceptance, 1 rollback)
	draft3 := []int{101, 102, 103, 999}
	target3 := []int{101, 102, 103, 204}
	res3 := EvaluateDraftAcceptance(draft3, target3)
	if res3.AcceptedCount != 3 {
		t.Errorf("res3.AcceptedCount = %d, want 3", res3.AcceptedCount)
	}
	if res3.RollbackCount != 1 {
		t.Errorf("res3.RollbackCount = %d, want 1", res3.RollbackCount)
	}
	if res3.RejectedAt != 3 {
		t.Errorf("res3.RejectedAt = %d, want 3", res3.RejectedAt)
	}
	if res3.AcceptanceRate != 0.75 {
		t.Errorf("res3.AcceptanceRate = %g, want 0.75 (75%%)", res3.AcceptanceRate)
	}
	// NextTokens should contain [101, 102, 103, 204] (3 accepted + 1 replacement)
	if len(res3.NextTokens) != 4 || res3.NextTokens[3] != 204 {
		t.Errorf("res3.NextTokens = %+v, want [101 102 103 204]", res3.NextTokens)
	}

	// Case 3: First token rejected (0% acceptance, 4 rollbacks)
	draft0 := []int{999, 102, 103, 104}
	target0 := []int{201, 102, 103, 104}
	res0 := EvaluateDraftAcceptance(draft0, target0)
	if res0.AcceptedCount != 0 {
		t.Errorf("res0.AcceptedCount = %d, want 0", res0.AcceptedCount)
	}
	if res0.RollbackCount != 4 {
		t.Errorf("res0.RollbackCount = %d, want 4", res0.RollbackCount)
	}
	if res0.RejectedAt != 0 {
		t.Errorf("res0.RejectedAt = %d, want 0", res0.RejectedAt)
	}
	if len(res0.NextTokens) != 1 || res0.NextTokens[0] != 201 {
		t.Errorf("res0.NextTokens = %+v, want [201]", res0.NextTokens)
	}

	// Test speedup calculation across the 75-85% acceptance band
	speedup75 := CalculateExpectedSpeedup(0.75, 4)
	speedup80 := CalculateExpectedSpeedup(0.80, 4)
	speedup85 := CalculateExpectedSpeedup(0.85, 4)

	// Theoretical:
	// alpha=0.75: 1 + 0.75 + 0.5625 + 0.421875 + 0.31640625 = 3.05
	// alpha=0.80: 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.36
	// alpha=0.85: 1 + 0.85 + 0.7225 + 0.614125 + 0.52200625 = 3.71
	if speedup75 < 2.9 || speedup75 > 3.2 {
		t.Errorf("speedup75 = %g, want ~3.05", speedup75)
	}
	if speedup80 < 3.2 || speedup80 > 3.5 {
		t.Errorf("speedup80 = %g, want ~3.36", speedup80)
	}
	if speedup85 < 3.5 || speedup85 > 3.9 {
		t.Errorf("speedup85 = %g, want ~3.71", speedup85)
	}
}

// batchedVerifyRow builds a deterministic fused logit row of the given width
// whose unique argmax is winner: every distractor stays below 1.0 while the
// winner sits at 2.0.
func batchedVerifyRow(width, winner int, rng *rand.Rand) []float32 {
	row := make([]float32, width)
	for i := range row {
		row[i] = rng.Float32()
	}
	row[winner] = 2.0
	return row
}

// TestBatchedBlockVerifyWholeBlockAccept pins the all-drafts-match outcome:
// every drafted token is accepted, nothing rolls back, and the optional final
// fused row contributes the trailing bonus token.
func TestBatchedBlockVerifyWholeBlockAccept(t *testing.T) {
	rng := rand.New(rand.NewSource(0xBADC0FFEE))
	draft := []int{101, 102, 103, 104}
	targets := []int{101, 102, 103, 104, 105}

	rows := make([][]float32, len(targets))
	for i, want := range targets {
		rows[i] = batchedVerifyRow(512, want, rng)
	}
	res, err := BatchedBlockVerify(draft, rows)
	if err != nil {
		t.Fatalf("BatchedBlockVerify failed: %v", err)
	}
	if res.AcceptedCount != len(draft) {
		t.Errorf("AcceptedCount = %d, want %d", res.AcceptedCount, len(draft))
	}
	if res.RejectedAt != -1 {
		t.Errorf("RejectedAt = %d, want -1 (whole block accepted)", res.RejectedAt)
	}
	if res.RollbackCount != 0 {
		t.Errorf("RollbackCount = %d, want 0", res.RollbackCount)
	}
	if len(res.NextTokens) != len(targets) || res.NextTokens[len(targets)-1] != 105 {
		t.Errorf("NextTokens = %+v, want %d tokens ending with bonus 105", res.NextTokens, len(targets))
	}

	// Without the bonus row the whole block still accepts, carrying no bonus.
	resNoBonus, err := BatchedBlockVerify(draft, rows[:len(draft)])
	if err != nil {
		t.Fatalf("BatchedBlockVerify (k rows) failed: %v", err)
	}
	if resNoBonus.AcceptedCount != len(draft) || resNoBonus.RejectedAt != -1 {
		t.Errorf("k-row whole-block accept = %+v, want all accepted", resNoBonus)
	}
	if len(resNoBonus.NextTokens) != len(draft) {
		t.Errorf("k-row NextTokens = %+v, want exactly the %d accepted drafts", resNoBonus.NextTokens, len(draft))
	}
}

// TestBatchedBlockVerifyMidBlockReject pins a divergence at a pinned interior
// position: the prefix accepts, the tail rejects, and the correction token is
// the diverging row's argmax.
func TestBatchedBlockVerifyMidBlockReject(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5EED))
	draft := []int{101, 102, 103, 104}
	targets := []int{101, 102, 300, 104} // divergence at position 2

	rows := make([][]float32, len(targets))
	for i, want := range targets {
		rows[i] = batchedVerifyRow(512, want, rng)
	}
	res, err := BatchedBlockVerify(draft, rows)
	if err != nil {
		t.Fatalf("BatchedBlockVerify failed: %v", err)
	}
	if res.RejectedAt != 2 {
		t.Errorf("RejectedAt = %d, want 2", res.RejectedAt)
	}
	if res.AcceptedCount != 2 {
		t.Errorf("AcceptedCount = %d, want 2", res.AcceptedCount)
	}
	if res.RollbackCount != len(draft)-2 {
		t.Errorf("RollbackCount = %d, want %d", res.RollbackCount, len(draft)-2)
	}
	wantNext := []int{101, 102, 300} // accepted prefix + correction = argmax(row 2)
	if len(res.NextTokens) != len(wantNext) {
		t.Fatalf("NextTokens = %+v, want %v", res.NextTokens, wantNext)
	}
	for i, tok := range wantNext {
		if res.NextTokens[i] != tok {
			t.Errorf("NextTokens[%d] = %d, want %d", i, res.NextTokens[i], tok)
		}
	}
}

// TestBatchedBlockVerifyTailOnlyAccept pins a divergence at the last draft
// position: every earlier token accepts, only the tail rejects.
func TestBatchedBlockVerifyTailOnlyAccept(t *testing.T) {
	rng := rand.New(rand.NewSource(0x7A1D0))
	draft := []int{101, 102, 103, 104}
	targets := []int{101, 102, 103, 300} // divergence at position 3 (tail)

	rows := make([][]float32, len(targets))
	for i, want := range targets {
		rows[i] = batchedVerifyRow(512, want, rng)
	}
	res, err := BatchedBlockVerify(draft, rows)
	if err != nil {
		t.Fatalf("BatchedBlockVerify failed: %v", err)
	}
	if res.RejectedAt != len(draft)-1 {
		t.Errorf("RejectedAt = %d, want %d", res.RejectedAt, len(draft)-1)
	}
	if res.AcceptedCount != len(draft)-1 {
		t.Errorf("AcceptedCount = %d, want %d", res.AcceptedCount, len(draft)-1)
	}
	if res.RollbackCount != 1 {
		t.Errorf("RollbackCount = %d, want 1", res.RollbackCount)
	}
	wantNext := []int{101, 102, 103, 300}
	if len(res.NextTokens) != len(wantNext) || res.NextTokens[len(wantNext)-1] != 300 {
		t.Errorf("NextTokens = %+v, want %v", res.NextTokens, wantNext)
	}
}

// TestBatchedBlockVerifyParityFixedSeed pins the owed parity: over a pinned
// step count at a fixed seed, the greedy token stream emitted by batched
// whole-block verify is identical to the serial per-position walk over the
// same fused logit rows.
func TestBatchedBlockVerifyParityFixedSeed(t *testing.T) {
	const (
		vocab     = 97
		k         = 4
		steps     = 256
		seed      = int64(0x5EEDF00D)
		acceptPct = 70 // drafter hits the true token 70% of the time
	)
	rng := rand.New(rand.NewSource(seed))

	// True greedy target stream long enough for every step's full panel.
	total := steps*(k+1) + k
	truth := make([]int, total)
	for i := range truth {
		truth[i] = rng.Intn(vocab)
	}

	emitSerial := func(draft []int, rows [][]float32) []int {
		out := make([]int, 0, len(rows))
		for j, row := range rows {
			target := argmaxFloat32(row)
			if j < len(draft) && draft[j] == target {
				out = append(out, draft[j])
				continue
			}
			out = append(out, target) // correction at the first divergence
			break
		}
		return out
	}

	pos := 0
	batchedStream, serialStream := make([]int, 0, total), make([]int, 0, total)
	for s := 0; s < steps; s++ {
		draft := make([]int, k)
		for i := 0; i < k; i++ {
			if rng.Intn(100) < acceptPct {
				draft[i] = truth[pos+i]
			} else {
				draft[i] = (truth[pos+i] + 1 + rng.Intn(vocab-1)) % vocab // guaranteed wrong
			}
		}
		rows := make([][]float32, k+1)
		for j := 0; j <= k; j++ {
			rows[j] = batchedVerifyRow(vocab, truth[pos+j], rng)
		}
		res, err := BatchedBlockVerify(draft, rows)
		if err != nil {
			t.Fatalf("step %d: BatchedBlockVerify failed: %v", s, err)
		}
		if len(res.NextTokens) == 0 {
			t.Fatalf("step %d: batched verify emitted no tokens", s)
		}
		batchedStream = append(batchedStream, res.NextTokens...)
		serialStream = append(serialStream, emitSerial(draft, rows)...)
		pos += len(res.NextTokens)
	}
	if len(batchedStream) != len(serialStream) {
		t.Fatalf("stream lengths differ: batched %d vs serial %d", len(batchedStream), len(serialStream))
	}
	for i := range batchedStream {
		if batchedStream[i] != serialStream[i] {
			t.Fatalf("parity broken at emission %d: batched %d vs serial %d", i, batchedStream[i], serialStream[i])
		}
	}
}

// TestBatchedBlockVerifySentinels pins the closed error contract: malformed
// panels report ErrBatchedBlockVerifyShape, non-finite logits report
// ErrBatchedBlockVerifyNonFinite, and an empty draft with no rows is a clean
// no-op.
func TestBatchedBlockVerifySentinels(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	draft := []int{101, 102, 103, 104}
	goodRows := func(n int) [][]float32 {
		rows := make([][]float32, n)
		for i := range rows {
			rows[i] = batchedVerifyRow(64, rng.Intn(64), rng)
		}
		return rows
	}

	shapeCases := map[string]func() [][]float32{
		"k-1 rows":     func() [][]float32 { return goodRows(len(draft) - 1) },
		"k+2 rows":     func() [][]float32 { return goodRows(len(draft) + 2) },
		"draft+rows 0": func() [][]float32 { return goodRows(0) }, // k>=1 needs at least k rows
		"empty draft with rows": func() [][]float32 {
			rows := goodRows(1)
			_ = draft
			return rows
		},
	}
	for name, build := range shapeCases {
		var d []int
		if name != "empty draft with rows" {
			d = draft
		}
		if _, err := BatchedBlockVerify(d, build()); !errors.Is(err, ErrBatchedBlockVerifyShape) {
			t.Errorf("%s: err = %v, want ErrBatchedBlockVerifyShape", name, err)
		}
	}
	ragged := goodRows(len(draft))
	ragged[2] = ragged[2][:4] // narrower than the panel width
	if _, err := BatchedBlockVerify(draft, ragged); !errors.Is(err, ErrBatchedBlockVerifyShape) {
		t.Errorf("ragged rows: err = %v, want ErrBatchedBlockVerifyShape", err)
	}
	emptyRow := goodRows(len(draft))
	emptyRow[0] = emptyRow[0][:0]
	if _, err := BatchedBlockVerify(draft, emptyRow); !errors.Is(err, ErrBatchedBlockVerifyShape) {
		t.Errorf("empty row: err = %v, want ErrBatchedBlockVerifyShape", err)
	}

	nanRow := goodRows(len(draft))
	nanRow[1][3] = float32(math.NaN())
	if _, err := BatchedBlockVerify(draft, nanRow); !errors.Is(err, ErrBatchedBlockVerifyNonFinite) {
		t.Errorf("NaN row: err = %v, want ErrBatchedBlockVerifyNonFinite", err)
	}
	infRow := goodRows(len(draft))
	infRow[2][5] = float32(math.Inf(1))
	if _, err := BatchedBlockVerify(draft, infRow); !errors.Is(err, ErrBatchedBlockVerifyNonFinite) {
		t.Errorf("+Inf row: err = %v, want ErrBatchedBlockVerifyNonFinite", err)
	}

	res, err := BatchedBlockVerify(nil, nil)
	if err != nil {
		t.Fatalf("empty draft with no rows must be a clean no-op, got %v", err)
	}
	if res.RejectedAt != -1 || res.AcceptedCount != 0 || len(res.NextTokens) != 0 {
		t.Errorf("empty-draft no-op result = %+v, want zero acceptance with RejectedAt=-1", res)
	}
}

// TestBatchedBlockVerifyModeSelection pins the admission probe: batched only
// when the drafter is live AND the block holds more than one token; otherwise
// the serial path remains the selected mode.
func TestBatchedBlockVerifyModeSelection(t *testing.T) {
	cases := []struct {
		drafterAvailable bool
		blockSize        int
		want             BlockVerifyMode
	}{
		{false, 0, BlockVerifyModeSerial},
		{false, 1, BlockVerifyModeSerial},
		{false, 4, BlockVerifyModeSerial},
		{true, 0, BlockVerifyModeSerial},
		{true, 1, BlockVerifyModeSerial},
		{true, 2, BlockVerifyModeBatched},
		{true, 3, BlockVerifyModeBatched},
		{true, 4, BlockVerifyModeBatched},
	}
	for _, tc := range cases {
		got := SelectBlockVerifyMode(tc.drafterAvailable, tc.blockSize)
		if got != tc.want {
			t.Errorf("SelectBlockVerifyMode(%v, %d) = %v, want %v", tc.drafterAvailable, tc.blockSize, got, tc.want)
		}
	}
	if BlockVerifyModeSerial.String() != "serial" || BlockVerifyModeBatched.String() != "batched" {
		t.Errorf("mode String() = %q / %q, want serial / batched", BlockVerifyModeSerial.String(), BlockVerifyModeBatched.String())
	}
}

// TestMTPDynamicThrottling tests adaptive throttling of K between 4 and 1.
func TestMTPDynamicThrottling(t *testing.T) {
	gov := NewMTPAdaptiveGovernor()
	if gov.CurrentK() != 4 {
		t.Fatalf("initial K = %d, want 4", gov.CurrentK())
	}

	// 1. Simulate high acceptance (80%): K remains 4
	for i := 0; i < 20; i++ {
		gov.RecordStep(4, 3, 1) // 75% per step
	}
	if gov.CurrentK() != 4 {
		t.Errorf("after high acceptance, K = %d, want 4", gov.CurrentK())
	}

	// 2. Simulate degraded acceptance (25% < 50% threshold): K throttles down to 1
	for i := 0; i < 20; i++ {
		gov.RecordStep(4, 1, 3) // 25% acceptance
	}
	if gov.CurrentK() != 1 {
		t.Errorf("after low acceptance, K = %d, want 1 (throttled)", gov.CurrentK())
	}

	// 3. Simulate recovery (80% >= 75% threshold): K recovers back to 4
	for i := 0; i < 20; i++ {
		gov.RecordStep(4, 4, 0) // 100% acceptance
	}
	if gov.CurrentK() != 4 {
		t.Errorf("after recovery, K = %d, want 4 (restored)", gov.CurrentK())
	}
}
