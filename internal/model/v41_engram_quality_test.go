package model

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// v41_engram_quality_test.go — focused witness for the quantized-Engram quality
// gate (issue #12969). It builds quantized Engram fixtures with the package's
// independent E4M3 encoder (encodeE4M3, fp8_blockscale_parity_test.go), derives
// an independent fp32 reference the fixture does NOT read back, and pins every
// rung: NMSE ceiling, held-out/zero-overlap perplexity, beat-random control, and
// the fail-closed typed missing-evidence refusal.

// quantizeV41EngramRow encodes one row of fp32 reference values into the
// 32x32-block E4M3 + E8M0 layout the gate decodes. It is deliberately written
// from the storage contract (not from decodeV41EngramQuantRow) so the fixture is
// an independent producer: for each 32-element tile it maps the tile absmax onto
// e4m3's 448 max, chooses the smallest E8M0 exponent that fits, and encodes each
// element as encodeE4M3(v / 2^exp).
func quantizeV41EngramRow(ref []float32) (weight []byte, scales []byte) {
	width := len(ref)
	scaleCols := (width-1)/v41FP8BlockDim + 1
	scales = make([]byte, scaleCols)
	weight = make([]byte, width)
	for tile := 0; tile < scaleCols; tile++ {
		lo := tile * v41FP8BlockDim
		hi := lo + v41FP8BlockDim
		if hi > width {
			hi = width
		}
		var amax float64
		for i := lo; i < hi; i++ {
			if a := math.Abs(float64(ref[i])); a > amax {
				amax = a
			}
		}
		exp := 0
		if amax > 0 {
			exp = int(math.Ceil(math.Log2(amax / e4m3Max)))
		}
		scales[tile] = byte(exp + 127)
		scale := math.Ldexp(1, exp)
		for i := lo; i < hi; i++ {
			if amax == 0 {
				weight[i] = 0
				continue
			}
			weight[i] = encodeE4M3(float32(float64(ref[i]) / scale))
		}
	}
	return weight, scales
}

// buildV41EngramQuantTable quantizes a full row-major [rows,width] reference.
func buildV41EngramQuantTable(ref [][]float32) V41EngramQuantTable {
	if len(ref) == 0 {
		return V41EngramQuantTable{}
	}
	width := len(ref[0])
	scaleCols := (width-1)/v41FP8BlockDim + 1
	scaleRows := (len(ref)-1)/v41FP8BlockDim + 1
	table := V41EngramQuantTable{
		RowCount: len(ref),
		RowWidth: width,
		Weight:   make([]byte, len(ref)*width),
		Scales:   make([]byte, scaleRows*scaleCols),
	}
	for row := 0; row < len(ref); row++ {
		w, s := quantizeV41EngramRow(ref[row])
		copy(table.Weight[row*width:], w)
		tileRow := (row / v41FP8BlockDim) * scaleCols
		copy(table.Scales[tileRow:], s)
	}
	return table
}

// referenceRows builds a deterministic, non-degenerate fp32 reference table.
func referenceRows(rows, width int, seed uint64) [][]float32 {
	out := make([][]float32, rows)
	state := seed | 1
	next := func() float64 {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		return float64(state%20000)/1000.0 - 10.0
	}
	for r := range out {
		row := make([]float32, width)
		for c := range row {
			row[c] = float32(next())
		}
		out[r] = row
	}
	return out
}

// qualityOptions pairs disjoint calibration/held-out token (row) splits.
func qualityOptions() V41EngramQualityOptions {
	return V41EngramQualityOptions{
		Artifact:          "enq-fixture-a",
		CalibrationTokens: []int{0, 1, 2, 3, 4, 5, 6, 7},
		HeldOutTokens:     []int{16, 17, 18, 19},
		RandomSeed:        0xC0FFEE,
	}
}

// TestV41EngramQuantQualityPassesWithinCeiling is the core GREEN witness: a
// well-quantized fixture passes NMSE, held-out, and beat-random, and the receipt
// admits it.
func TestV41EngramQuantQualityPassesWithinCeiling(t *testing.T) {
	ref := referenceRows(40, 64, 0xBEEF)
	quant := buildV41EngramQuantTable(ref)

	verdict, err := EvaluateV41EngramQuality(quant, ref, qualityOptions())
	if err != nil {
		t.Fatalf("EvaluateV41EngramQuality: %v", err)
	}
	if !verdict.Pass {
		t.Fatalf("expected pass, got %+v", verdict)
	}
	if !verdict.NMSEWithinCeiling {
		t.Fatalf("expected NMSE within ceiling: nmse=%g ceiling=%g", verdict.BlockNMSE, verdict.NMSECeiling)
	}
	if verdict.BlockNMSE <= 0 || verdict.BlockNMSE >= 1 {
		t.Fatalf("implausible NMSE %g", verdict.BlockNMSE)
	}
	if verdict.OverlapTokens != 0 {
		t.Fatalf("expected zero overlap, got %d", verdict.OverlapTokens)
	}
	if !(verdict.HeldOutPerplexity < verdict.RandomControlPerplexity) {
		t.Fatalf("held-out PPL %g should beat random %g", verdict.HeldOutPerplexity, verdict.RandomControlPerplexity)
	}
	t.Logf("NMSE=%g (ceiling %g) held-out PPL=%g random PPL=%g",
		verdict.BlockNMSE, verdict.NMSECeiling, verdict.HeldOutPerplexity, verdict.RandomControlPerplexity)

	receipt := &V41EngramQualityReceipt{Artifact: qualityOptions().Artifact, Verdict: verdict}
	if err := AdmitV41EngramQuantizedArtifact(qualityOptions().Artifact, receipt); err != nil {
		t.Fatalf("expected receipt to admit: %v", err)
	}
}

// TestV41EngramQuantQualityNMSEExceedsCeiling: a badly quantized table (every
// non-zero weight forced to a far-off code) must exceed the 0.06 ceiling and
// refuse.
func TestV41EngramQuantQualityNMSEExceedsCeiling(t *testing.T) {
	ref := referenceRows(32, 32, 0xABCD)
	quant := buildV41EngramQuantTable(ref)

	// Corrupt the payload: flip every code to a large, sign-consistent value
	// while leaving geometry valid, so only the NMSE rung can catch it.
	for i := range quant.Weight {
		if ref[i/quant.RowWidth][i%quant.RowWidth] != 0 {
			quant.Weight[i] = encodeE4M3(400)
		}
	}

	verdict, err := EvaluateV41EngramQuality(quant, ref, qualityOptions())
	if err != nil {
		t.Fatalf("EvaluateV41EngramQuality: %v", err)
	}
	if verdict.Pass {
		t.Fatalf("corrupted table must not pass: %+v", verdict)
	}
	if verdict.NMSEWithinCeiling {
		t.Fatalf("expected NMSE over ceiling, got %g <= %g", verdict.BlockNMSE, verdict.NMSECeiling)
	}
}

// TestV41EngramQuantQualityOverlapRefused: a held-out split that shares a token
// with the calibration set must refuse before any scoring.
func TestV41EngramQuantQualityOverlapRefused(t *testing.T) {
	ref := referenceRows(8, 32, 0x1)
	quant := buildV41EngramQuantTable(ref)
	opts := qualityOptions()
	opts.HeldOutTokens = []int{30, 4, 31} // 4 is in calibration
	_, err := EvaluateV41EngramQuality(quant, ref, opts)
	if err == nil {
		t.Fatal("expected overlap refusal")
	}
	if !errors.Is(err, ErrV41EngramQuality) {
		t.Fatalf("expected ErrV41EngramQuality, got %v", err)
	}
}

// TestV41EngramQuantQualityMissingSplitsFailClosed: empty calibration or
// held-out split returns the typed missing-evidence error.
func TestV41EngramQuantQualityMissingSplitsFailClosed(t *testing.T) {
	ref := referenceRows(8, 32, 0x2)
	quant := buildV41EngramQuantTable(ref)

	cases := []struct {
		name string
		opts V41EngramQualityOptions
	}{
		{"no held-out", func() V41EngramQualityOptions { o := qualityOptions(); o.HeldOutTokens = nil; return o }()},
		{"no calibration", func() V41EngramQualityOptions { o := qualityOptions(); o.CalibrationTokens = nil; return o }()},
	}
	for _, tc := range cases {
		_, err := EvaluateV41EngramQuality(quant, ref, tc.opts)
		var missing *V41EngramMissingQualityError
		if !errors.As(err, &missing) {
			t.Fatalf("%s: expected V41EngramMissingQualityError, got %v", tc.name, err)
		}
	}
}

// TestV41EngramQuantQualityAdmissionFailClosed: the admission guard refuses a
// nil receipt, a mismatched artifact identity, and a non-passing verdict.
func TestV41EngramQuantQualityAdmissionFailClosed(t *testing.T) {
	if err := AdmitV41EngramQuantizedArtifact("art", nil); err == nil {
		t.Fatal("nil receipt must refuse")
	} else {
		var missing *V41EngramMissingQualityError
		if !errors.As(err, &missing) {
			t.Fatalf("expected typed error, got %v", err)
		}
	}

	mismatch := &V41EngramQualityReceipt{Artifact: "other", Verdict: V41EngramQualityVerdict{Pass: true}}
	if err := AdmitV41EngramQuantizedArtifact("art", mismatch); err == nil {
		t.Fatal("mismatched artifact must refuse")
	}

	failing := &V41EngramQualityReceipt{Artifact: "art", Verdict: V41EngramQualityVerdict{Pass: false, Reason: "nmse"}}
	if err := AdmitV41EngramQuantizedArtifact("art", failing); err == nil {
		t.Fatal("non-passing verdict must refuse")
	}

	passing := &V41EngramQualityReceipt{Artifact: "art", Verdict: V41EngramQualityVerdict{Pass: true}}
	if err := AdmitV41EngramQuantizedArtifact("art", passing); err != nil {
		t.Fatalf("passing receipt must admit: %v", err)
	}
}

// TestV41EngramQuantQualityFailClosedGeometry exercises every malformed input
// rung: short weight, short scales, E8M0 NaN, E4M3 NaN, out-of-range row, and
// reference width mismatch.
func TestV41EngramQuantQualityFailClosedGeometry(t *testing.T) {
	ref := referenceRows(3, 32, 0x3)
	good := buildV41EngramQuantTable(ref)

	shortWeight := good
	shortWeight.Weight = good.Weight[:len(good.Weight)-1]
	shortScale := good
	shortScale.Scales = good.Scales[:len(good.Scales)-1]
	nanScale := good
	nanScale.Scales = append([]byte(nil), good.Scales...)
	nanScale.Scales[0] = 0xff
	nanWeight := good
	nanWeight.Weight = append([]byte(nil), good.Weight...)
	nanWeight.Weight[0] = 0x7f

	for _, tc := range []struct {
		name  string
		table V41EngramQuantTable
	}{
		{"short weight", shortWeight},
		{"short scales", shortScale},
		{"E8M0 NaN", nanScale},
		{"E4M3 NaN", nanWeight},
	} {
		if _, err := V41EngramBlockNMSE(tc.table, ref); err == nil {
			t.Fatalf("%s: expected refusal", tc.name)
		} else if !errors.Is(err, ErrV41EngramQuality) {
			t.Fatalf("%s: expected ErrV41EngramQuality, got %v", tc.name, err)
		}
	}

	if _, err := decodeV41EngramQuantRow(good, -1); err == nil {
		t.Fatal("negative row must refuse")
	}
	if _, err := decodeV41EngramQuantRow(good, good.RowCount); err == nil {
		t.Fatal("out-of-range row must refuse")
	}
	if _, err := V41EngramBlockNMSE(good, ref[:1]); err == nil {
		t.Fatal("reference row-count mismatch must refuse")
	}
	badWidth := [][]float32{{1, 2, 3}}
	if _, err := V41EngramBlockNMSE(good, badWidth); err == nil {
		t.Fatal("reference width mismatch must refuse")
	}
}

// TestV41EngramQuantQualityMustMoveControls proves each load-bearing axis is
// live: a shuffled table scores strictly worse, and the residual scale changes
// the held-out score, so the result is not a constant.
func TestV41EngramQuantQualityMustMoveControls(t *testing.T) {
	ref := referenceRows(40, 64, 0x99)
	quant := buildV41EngramQuantTable(ref)
	rows := []int{20, 21, 22, 23}

	sigma, n, err := v41EngramResidualScale(quant, ref, qualityOptions().CalibrationTokens)
	if err != nil {
		t.Fatalf("residual scale: %v", err)
	}
	if n == 0 || !(sigma > 0) {
		t.Fatalf("residual scale must be positive: sigma=%g n=%d", sigma, n)
	}
	heldOut, err := v41EngramHeldOutPerplexity(quant, ref, rows, sigma)
	if err != nil {
		t.Fatalf("held-out: %v", err)
	}
	randomPPL, err := v41EngramHeldOutPerplexityShuffled(quant, ref, rows, sigma, qualityOptions().RandomSeed)
	if err != nil {
		t.Fatalf("shuffled held-out: %v", err)
	}
	if !(heldOut < randomPPL) {
		t.Fatalf("shuffle must move the score: real %g should beat shuffled %g", heldOut, randomPPL)
	}
	t.Logf("held-out PPL=%g shuffled PPL=%g sigma=%g", heldOut, randomPPL, sigma)
}

// TestV41EngramQuantQualityUninformativeTableCannotPass is the anti-vacuity
// control the issue requires: an uninformative table must not pass. The all-zero
// table against an all-zero reference has NMSE 0 (well within ceiling) but
// carries no information, so its held-out score ties the shuffled control
// exactly and the gate must refuse on the beat-random rung.
func TestV41EngramQuantQualityUninformativeTableCannotPass(t *testing.T) {
	ref := referenceRows(40, 64, 0x1234)
	for r := range ref {
		for c := range ref[r] {
			ref[r][c] = 0
		}
	}
	quant := buildV41EngramQuantTable(ref)
	opts := qualityOptions()

	verdict, err := EvaluateV41EngramQuality(quant, ref, opts)
	if err != nil {
		t.Fatalf("EvaluateV41EngramQuality: %v", err)
	}
	if verdict.Pass {
		t.Fatalf("uninformative table must not pass: %+v", verdict)
	}
	if verdict.BlockNMSE != 0 {
		t.Fatalf("all-zero table vs all-zero reference must have NMSE 0, got %g", verdict.BlockNMSE)
	}
	if verdict.HeldOutPerplexity != verdict.RandomControlPerplexity {
		t.Logf("held-out PPL=%g random PPL=%g", verdict.HeldOutPerplexity, verdict.RandomControlPerplexity)
	}
	t.Logf("uninformative table refused: NMSE=%g reason=%s", verdict.BlockNMSE, verdict.Reason)
}

// TestV41EngramQuantQualityDeterministic pins replay determinism: the same
// inputs yield identical verdicts, including the beat-random control.
func TestV41EngramQuantQualityDeterministic(t *testing.T) {
	ref := referenceRows(40, 32, 0x5)
	quant := buildV41EngramQuantTable(ref)
	a, err := EvaluateV41EngramQuality(quant, ref, qualityOptions())
	if err != nil {
		t.Fatalf("first eval: %v", err)
	}
	b, err := EvaluateV41EngramQuality(quant, ref, qualityOptions())
	if err != nil {
		t.Fatalf("second eval: %v", err)
	}
	if a != b {
		t.Fatalf("non-deterministic verdict:\n a=%+v\n b=%+v", a, b)
	}
	// The verdict must be JSON-serializable (no Inf/NaN leak through the
	// receipt surface).
	if _, err := json.Marshal(&V41EngramQualityReceipt{Artifact: a.Artifact, Verdict: a}); err != nil {
		t.Fatalf("verdict must marshal: %v", err)
	}
}

// TestV41EngramQuantQualityResidualHelpersFailClosed exercises the residual and
// held-out helper fail-closed rungs.
func TestV41EngramQuantQualityResidualHelpersFailClosed(t *testing.T) {
	ref := referenceRows(4, 32, 0x6)
	quant := buildV41EngramQuantTable(ref)

	if _, _, err := v41EngramResidualScale(quant, ref, nil); err == nil {
		t.Fatal("empty calibration rows must refuse")
	}
	if _, err := v41EngramHeldOutPerplexity(quant, ref, []int{1}, 0); err == nil {
		t.Fatal("non-positive sigma must refuse")
	}
	if _, err := v41EngramHeldOutPerplexity(quant, ref, nil, 1); err == nil {
		t.Fatal("empty held-out rows must refuse")
	}
	if _, err := v41EngramHeldOutPerplexityShuffled(quant, ref, []int{1}, 1, 1); err == nil {
		t.Fatal("single-row control must refuse")
	}
}
