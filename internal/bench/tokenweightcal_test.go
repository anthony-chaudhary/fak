package bench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The synthetic ledger's ground truth. A real sanctioned-node ledger replaces
// these with measurements; they exist so the fit can be checked against an
// answer that is known independently of the fit.
const (
	truthFixedMS       = 25.0
	truthUncachedRate  = 0.40
	truthCachedRate    = 0.04
	truthDecodeRate    = 9.00
	syntheticLedgerRel = "testdata/token_service_synthetic.jsonl"
)

type tokenShape struct {
	id      string
	u, c, d int64
	holdout bool
}

// tokenShapes are controlled: the three classes vary close to independently, so
// the normal matrix is well conditioned and each weight is identifiable.
func tokenShapes() []tokenShape {
	return []tokenShape{
		{"f01", 200, 0, 16, false},
		{"f02", 200, 2000, 128, false},
		{"f03", 400, 6000, 32, false},
		{"f04", 800, 0, 256, false},
		{"f05", 800, 1000, 16, false},
		{"f06", 1200, 4000, 512, false},
		{"f07", 1600, 500, 64, false},
		{"f08", 2000, 0, 32, false},
		{"f09", 2000, 6000, 256, false},
		{"f10", 2400, 2000, 16, false},
		{"f11", 2800, 1000, 512, false},
		{"f12", 3200, 4000, 128, false},
		{"f13", 3600, 0, 64, false},
		{"f14", 4000, 6000, 16, false},
		{"f15", 4000, 500, 256, false},
		{"f16", 600, 3000, 384, false},
		{"f17", 1000, 5000, 96, false},
		{"f18", 3000, 2500, 192, false},
		{"h01", 300, 1500, 48, true},
		{"h02", 900, 4500, 320, true},
		{"h03", 1800, 250, 192, true},
		{"h04", 2600, 5500, 80, true},
		{"h05", 3400, 3000, 448, true},
		{"h06", 500, 0, 224, true},
		{"h07", 1400, 6000, 24, true},
		{"h08", 3800, 1200, 144, true},
	}
}

// syntheticTokenRows regenerates the committed ledger byte-for-byte. The noise
// is a fixed LCG seeded on the issue number, so the fixture is reproducible
// from source rather than being an opaque committed blob.
func syntheticTokenRows(prefix string) []TokenServiceRow {
	state := uint64(5778)
	next := func() float64 {
		state = (1103515245*state + 12345) % (1 << 31)
		return (float64(state)/float64(uint64(1)<<31)*2 - 1) * 0.015
	}
	var rows []TokenServiceRow
	for _, s := range tokenShapes() {
		exact := truthFixedMS + float64(s.u)*truthUncachedRate + float64(s.c)*truthCachedRate + float64(s.d)*truthDecodeRate
		// Round to 3 decimals so the JSONL is stable across platforms.
		noisy := math.Round(exact*(1+next())*1000) / 1000
		split := TokenSplitFit
		if s.holdout {
			split = TokenSplitHoldout
		}
		rows = append(rows, TokenServiceRow{
			Schema:              TokenServiceRowSchema,
			ShapeID:             s.id,
			Split:               split,
			UncachedInputTokens: s.u,
			CachedInputTokens:   s.c,
			DecodeTokens:        s.d,
			ServiceMS:           noisy,
			Model:               "llama-3.1-8b-instruct",
			Engine:              "vllm-0.6.3",
			Hardware:            "synthetic-node-a",
			BatchSize:           1,
			Repetitions:         5,
			Provenance:          prefix + "fak/internal/bench syntheticTokenRows, lcg seed 5778, +/-1.5% multiplicative noise; NOT hardware truth",
		})
	}
	return rows
}

func marshalTokenLedger(t *testing.T, rows []TokenServiceRow) []byte {
	t.Helper()
	var buf bytes.Buffer
	for _, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal row %s: %v", r.ShapeID, err)
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}
	return buf.Bytes()
}

// TestWriteSyntheticTokenLedger regenerates the committed fixture. It is
// env-gated so a bare `go test` never rewrites testdata:
//
//	FAK_TOKENCAL_FIXTURE_OUT=internal/bench/testdata/token_service_synthetic.jsonl \
//	  go test ./internal/bench -run TestWriteSyntheticTokenLedger -count=1
func TestWriteSyntheticTokenLedger(t *testing.T) {
	out := os.Getenv("FAK_TOKENCAL_FIXTURE_OUT")
	if out == "" {
		t.Skip("set FAK_TOKENCAL_FIXTURE_OUT to regenerate the synthetic ledger")
	}
	if err := os.WriteFile(out, marshalTokenLedger(t, syntheticTokenRows(SyntheticProvenancePrefix)), 0o644); err != nil {
		t.Fatalf("write ledger: %v", err)
	}
	t.Logf("wrote %s", out)
}

// TestSyntheticLedgerIsReproducible is the "reproducible benchmark ledger" half
// of the #5778 witness: the committed bytes are exactly what the documented
// generator produces.
func TestSyntheticLedgerIsReproducible(t *testing.T) {
	committed, err := os.ReadFile(filepath.FromSlash(syntheticLedgerRel))
	if err != nil {
		t.Fatalf("read committed ledger: %v", err)
	}
	want := marshalTokenLedger(t, syntheticTokenRows(SyntheticProvenancePrefix))
	if !bytes.Equal(bytes.ReplaceAll(committed, []byte("\r\n"), []byte("\n")), want) {
		t.Fatalf("committed ledger drifted from its generator; regenerate with FAK_TOKENCAL_FIXTURE_OUT")
	}
	rows, err := ReadTokenServiceJSONL(bytes.NewReader(committed))
	if err != nil {
		t.Fatalf("read rows: %v", err)
	}
	if len(rows) != len(tokenShapes()) {
		t.Fatalf("want %d rows, got %d", len(tokenShapes()), len(rows))
	}
}

func mustCalibrate(t *testing.T, rows []TokenServiceRow, bound float64) TokenWeightCalibration {
	t.Helper()
	cal, err := CalibrateTokenWeights(rows, bound)
	if err != nil {
		t.Fatalf("calibrate: %v", err)
	}
	return cal
}

// TestCalibrateRecoversGroundTruth checks the fit against an answer it never
// saw, and checks the held-out error bound.
func TestCalibrateRecoversGroundTruth(t *testing.T) {
	cal := mustCalibrate(t, syntheticTokenRows(SyntheticProvenancePrefix), 3.0)

	for _, tc := range []struct {
		name      string
		got, want float64
		tolPct    float64
		rate      TokenWeightRate
	}{
		{"uncached_input", cal.Weights.UncachedInput.MSPerToken, truthUncachedRate, 25, cal.Weights.UncachedInput},
		{"cached_input", cal.Weights.CachedInput.MSPerToken, truthCachedRate, 40, cal.Weights.CachedInput},
		{"decode", cal.Weights.Decode.MSPerToken, truthDecodeRate, 10, cal.Weights.Decode},
	} {
		relPct := math.Abs(tc.got-tc.want) / tc.want * 100
		if relPct > tc.tolPct {
			t.Errorf("%s: fitted %.5f ms/token vs truth %.5f (%.1f%% off, tolerance %.0f%%)", tc.name, tc.got, tc.want, relPct, tc.tolPct)
		}
		if !tc.rate.Identified {
			t.Errorf("%s: weight not identified (rate %.5f, ci95 half-width %.5f)", tc.name, tc.rate.MSPerToken, tc.rate.CI95HalfWidth)
		}
	}

	if !cal.WithinDeclaredBound {
		t.Errorf("held-out MAPE %.3f%% exceeded the declared 3%% bound", cal.Holdout.MAPEPct)
	}
	if cal.Confidence.RSquared < 0.99 {
		t.Errorf("R^2 %.4f below 0.99 on a near-linear generator", cal.Confidence.RSquared)
	}
	if cal.Confidence.DegreesOfFreedom != 18-4 {
		t.Errorf("want 14 degrees of freedom, got %d", cal.Confidence.DegreesOfFreedom)
	}
	if cal.Provenance.HardwareTruth {
		t.Error("a synthetic ledger must never be labelled hardware truth")
	}
	if cal.Provenance.Model == "" || cal.Provenance.Engine == "" || cal.Provenance.Hardware == "" || cal.Provenance.BatchSize == 0 {
		t.Errorf("calibration is missing model/engine/hardware/batch provenance: %+v", cal.Provenance)
	}
	if cal.Digest == "" || cal.Schema != TokenWeightCalibrationSchema {
		t.Errorf("unversioned artifact: schema=%q digest=%q", cal.Schema, cal.Digest)
	}

	// The per-class fit must beat the *tuned* scalar-total alternative on the
	// held-out split. The scalar got its own least-squares fit first.
	if cal.Baseline.MSPerTotalToken <= 0 {
		t.Fatalf("scalar baseline was not tuned: %+v", cal.Baseline)
	}
	if cal.Holdout.MAPEPct >= cal.Baseline.Holdout.MAPEPct {
		t.Errorf("calibrated MAPE %.3f%% did not beat tuned scalar %.3f%%", cal.Holdout.MAPEPct, cal.Baseline.Holdout.MAPEPct)
	}
	t.Logf("held-out MAPE: calibrated %.3f%% vs tuned scalar-total %.3f%%; RMSE %.2fms vs %.2fms",
		cal.Holdout.MAPEPct, cal.Baseline.Holdout.MAPEPct, cal.Holdout.RMSEMS, cal.Baseline.Holdout.RMSEMS)

	// The relative weights are the drop-in replacement for the illustrative
	// tokenprofile.DefaultWeights, and they differ from it by a lot: that gap is
	// the entire reason #5778 exists.
	sw := cal.SchedulerWeights
	if sw.InputUncached != 1 {
		t.Errorf("scheduler weights must normalize uncached input to 1, got %v", sw.InputUncached)
	}
	if sw.Output <= 4 {
		t.Errorf("measured decode weight %.2f did not exceed the illustrative default of 4", sw.Output)
	}
	t.Logf("calibrated scheduler weights: uncached=%.2f cached=%.3f output=%.2f (illustrative default: 1 / 0.25 / 4)",
		sw.InputUncached, sw.InputCached, sw.Output)
}

func TestCalibrateRefusals(t *testing.T) {
	base := func() []TokenServiceRow { return syntheticTokenRows(SyntheticProvenancePrefix) }

	for _, tc := range []struct {
		name    string
		mutate  func([]TokenServiceRow) []TokenServiceRow
		bound   float64
		wantSub string
	}{
		{"non-positive bound", func(r []TokenServiceRow) []TokenServiceRow { return r }, 0, "positive percentage"},
		{"empty ledger", func([]TokenServiceRow) []TokenServiceRow { return nil }, 3, "empty token service ledger"},
		{"wrong row schema", func(r []TokenServiceRow) []TokenServiceRow {
			r[0].Schema = "fak-token-service-sample/0"
			return r
		}, 3, "want schema"},
		{"undeclared provenance", func(r []TokenServiceRow) []TokenServiceRow {
			r[2].Provenance = "some node somewhere"
			return r
		}, 3, "must declare"},
		{"mixed measured and synthetic", func(r []TokenServiceRow) []TokenServiceRow {
			r[4].Provenance = MeasuredProvenancePrefix + "node-b"
			return r
		}, 3, "mixes measured and synthetic"},
		{"mixed hardware", func(r []TokenServiceRow) []TokenServiceRow {
			r[5].Hardware = "other-node"
			return r
		}, 3, "do not pool"},
		{"mixed batch size", func(r []TokenServiceRow) []TokenServiceRow {
			r[6].BatchSize = 8
			return r
		}, 3, "do not pool"},
		{"split leak", func(r []TokenServiceRow) []TokenServiceRow {
			leak := r[0]
			leak.Split = TokenSplitHoldout
			return append(r, leak)
		}, 3, "must not leak"},
		{"unknown split", func(r []TokenServiceRow) []TokenServiceRow {
			r[1].Split = "validation"
			return r
		}, 3, "unknown split"},
		{"too few fit rows", func(r []TokenServiceRow) []TokenServiceRow {
			var keep []TokenServiceRow
			fits := 0
			for _, row := range r {
				if row.Split == TokenSplitFit {
					if fits >= minTokenFitRows-1 {
						continue
					}
					fits++
				}
				keep = append(keep, row)
			}
			return keep
		}, 3, "at least 6 fit rows"},
		{"too few holdout rows", func(r []TokenServiceRow) []TokenServiceRow {
			var keep []TokenServiceRow
			held := 0
			for _, row := range r {
				if row.Split == TokenSplitHoldout {
					if held >= minTokenHoldoutRows-1 {
						continue
					}
					held++
				}
				keep = append(keep, row)
			}
			return keep
		}, 3, "at least 3 held-out rows"},
		{"non-positive service time", func(r []TokenServiceRow) []TokenServiceRow {
			r[3].ServiceMS = 0
			return r
		}, 3, "non-positive or non-finite service time"},
		{"empty token shape", func(r []TokenServiceRow) []TokenServiceRow {
			r[7].UncachedInputTokens, r[7].CachedInputTokens, r[7].DecodeTokens = 0, 0, 0
			return r
		}, 3, "no tokens in any class"},
		// The load-bearing one: if the shapes are not controlled — here every
		// fit row holds cached at a fixed multiple of uncached and decode — the
		// design is singular and no per-class weight is recoverable. Refuse
		// rather than emit a confident-looking pseudo-inverse.
		{"uncontrolled (singular) shapes", func(r []TokenServiceRow) []TokenServiceRow {
			for i := range r {
				if r[i].Split != TokenSplitFit {
					continue
				}
				r[i].CachedInputTokens = r[i].UncachedInputTokens * 2
				r[i].DecodeTokens = r[i].UncachedInputTokens / 10
			}
			return r
		}, 3, "singular"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := CalibrateTokenWeights(tc.mutate(base()), tc.bound)
			if err == nil {
				t.Fatalf("want refusal containing %q, got none", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("want refusal containing %q, got %v", tc.wantSub, err)
			}
		})
	}
}

func measuredOverhead() SchedulingOverhead {
	return SchedulingOverhead{
		CalibratedDecisionMS: 0.004,
		BaselineDecisionMS:   0.001,
		Measured:             true,
		Provenance:           MeasuredProvenancePrefix + "go test -bench admission decision, synthetic-node-a",
	}
}

// TestNetValueRefusesUnmeasuredInputs is done-condition bullet 4: no net-true
// gain claim without the measured alternative and the measured overhead.
func TestNetValueRefusesUnmeasuredInputs(t *testing.T) {
	cal := mustCalibrate(t, syntheticTokenRows(SyntheticProvenancePrefix), 3.0)

	for _, tc := range []struct {
		name    string
		cal     func(TokenWeightCalibration) TokenWeightCalibration
		oh      func(SchedulingOverhead) SchedulingOverhead
		wantSub string
	}{
		{"overhead not measured", nil, func(o SchedulingOverhead) SchedulingOverhead {
			o.Measured = false
			return o
		}, "overhead is not measured"},
		{"overhead has no provenance", nil, func(o SchedulingOverhead) SchedulingOverhead {
			o.Provenance = ""
			return o
		}, "overhead is not measured"},
		{"negative overhead", nil, func(o SchedulingOverhead) SchedulingOverhead {
			o.CalibratedDecisionMS = -1
			return o
		}, "finite and non-negative"},
		{"alternative never scored", func(c TokenWeightCalibration) TokenWeightCalibration {
			c.Baseline.Holdout.Samples = 0
			return c
		}, nil, "measured alternative was not scored"},
		{"alternative missing", func(c TokenWeightCalibration) TokenWeightCalibration {
			c.Baseline.MSPerTotalToken, c.Baseline.FixedMS = 0, 0
			return c
		}, nil, "tuned scalar-total alternative is missing"},
		{"wrong artifact schema", func(c TokenWeightCalibration) TokenWeightCalibration {
			c.Schema = "fak-token-weight-calibration/0"
			return c
		}, nil, "want calibration schema"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, o := cal, measuredOverhead()
			if tc.cal != nil {
				c = tc.cal(c)
			}
			if tc.oh != nil {
				o = tc.oh(o)
			}
			if _, err := EvaluateNetSchedulingValue(c, o); err == nil || !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("want refusal containing %q, got %v", tc.wantSub, err)
			}
		})
	}
}

// TestNetValueWithheldOnSyntheticLedger is the anti-fabrication guard: the
// committed fixture predicts well and beats the tuned scalar, and still cannot
// produce a net-true gain claim, because it is not hardware truth. Only a
// ledger measured on a sanctioned node can flip NetTrueGain.
func TestNetValueWithheldOnSyntheticLedger(t *testing.T) {
	cal := mustCalibrate(t, syntheticTokenRows(SyntheticProvenancePrefix), 3.0)
	got, err := EvaluateNetSchedulingValue(cal, measuredOverhead())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if got.MispredictionReducedMS <= 0 {
		t.Fatalf("expected the synthetic fit to reduce misprediction, got %.3fms", got.MispredictionReducedMS)
	}
	if got.NetTrueGain {
		t.Fatal("a synthetic fixture must not yield a net-true gain claim")
	}
	if !strings.Contains(got.WhyNot, "synthetic fixture") {
		t.Fatalf("want the synthetic-provenance reason, got %q", got.WhyNot)
	}
}

// TestNetValueOnMeasuredLedger exercises the hardware-truth branch. The rows
// are the same synthetic numbers relabelled: this asserts the arithmetic and
// the gating, and makes NO claim about any real device.
func TestNetValueOnMeasuredLedger(t *testing.T) {
	relabelled := syntheticTokenRows(MeasuredProvenancePrefix)
	cal := mustCalibrate(t, relabelled, 3.0)
	if !cal.Provenance.HardwareTruth {
		t.Fatal("measured-prefixed rows must set HardwareTruth")
	}

	got, err := EvaluateNetSchedulingValue(cal, measuredOverhead())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if !got.NetTrueGain {
		t.Fatalf("want a net-true gain, got %+v", got)
	}
	wantNet := got.MispredictionReducedMS - got.AddedOverheadMS
	if math.Abs(got.NetValueMSPerRequest-wantNet) > 1e-9 {
		t.Errorf("net value %.6f != reduction %.6f - overhead %.6f", got.NetValueMSPerRequest, got.MispredictionReducedMS, got.AddedOverheadMS)
	}
	if got.AddedOverheadMS != 0.003 {
		t.Errorf("want the overhead delta on the books, got %.6f", got.AddedOverheadMS)
	}
	if got.CalibrationDigest != cal.Digest {
		t.Error("net-value claim is not bound to the calibration digest")
	}

	// Overhead large enough to swamp the accuracy win must cancel the claim.
	heavy := measuredOverhead()
	heavy.CalibratedDecisionMS = got.MispredictionReducedMS + 10
	swamped, err := EvaluateNetSchedulingValue(cal, heavy)
	if err != nil {
		t.Fatalf("evaluate heavy: %v", err)
	}
	if swamped.NetTrueGain || !strings.Contains(swamped.WhyNot, "overhead cancels") {
		t.Fatalf("want the overhead to cancel the gain, got %+v", swamped)
	}

	// A bound the fit cannot meet must also withhold the claim.
	tight := mustCalibrate(t, relabelled, 0.0001)
	strictErr, err := EvaluateNetSchedulingValue(tight, measuredOverhead())
	if err != nil {
		t.Fatalf("evaluate tight: %v", err)
	}
	if strictErr.NetTrueGain || !strings.Contains(strictErr.WhyNot, "exceeds the declared bound") {
		t.Fatalf("want the declared bound to withhold the claim, got %+v", strictErr)
	}
}

func TestTokenCalibrationDigestIsSelfVerifying(t *testing.T) {
	cal := mustCalibrate(t, syntheticTokenRows(SyntheticProvenancePrefix), 3.0)
	again, err := tokenCalibrationDigest(cal)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if again != cal.Digest {
		t.Fatalf("digest not reproducible: %s vs %s", again, cal.Digest)
	}
	tampered := cal
	tampered.Weights.Decode.MSPerToken *= 2
	if d, err := tokenCalibrationDigest(tampered); err != nil || d == cal.Digest {
		t.Fatalf("tampered weights kept the digest (%v)", err)
	}
}

// TestWriteTokenWeightCalibration is the operator path on a sanctioned compute
// node: point it at a measured ledger and it emits the calibration artifact.
//
//	FAK_TOKENCAL_LEDGER=/tmp/token-service-da33.jsonl \
//	FAK_TOKENCAL_OUT=/tmp/token-weights-da33.json \
//	FAK_TOKENCAL_BOUND_PCT=5 \
//	  go test ./internal/bench -run TestWriteTokenWeightCalibration -count=1
func TestWriteTokenWeightCalibration(t *testing.T) {
	ledger, out := os.Getenv("FAK_TOKENCAL_LEDGER"), os.Getenv("FAK_TOKENCAL_OUT")
	if ledger == "" || out == "" {
		t.Skip("set FAK_TOKENCAL_LEDGER and FAK_TOKENCAL_OUT to calibrate a measured ledger")
	}
	bound := 5.0
	if v := os.Getenv("FAK_TOKENCAL_BOUND_PCT"); v != "" {
		if _, err := fmt.Sscanf(v, "%g", &bound); err != nil {
			t.Fatalf("FAK_TOKENCAL_BOUND_PCT %q: %v", v, err)
		}
	}
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	rows, err := ReadTokenServiceJSONL(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("parse ledger: %v", err)
	}
	cal, err := CalibrateTokenWeights(rows, bound)
	if err != nil {
		// A refusal is a valid recorded outcome, not a reason to fabricate.
		t.Fatalf("calibration refused: %v", err)
	}
	buf, err := json.MarshalIndent(cal, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(out, append(buf, '\n'), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	t.Logf("wrote %s (hardware_truth=%v, held-out MAPE %.3f%% vs tuned scalar %.3f%%)",
		out, cal.Provenance.HardwareTruth, cal.Holdout.MAPEPct, cal.Baseline.Holdout.MAPEPct)
}
