package fitplan

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const goldenEpoch = "epoch-a"

// decodeReceipt is one tg@512 decode receipt: bytes per token and measured tok/s.
func decodeReceipt(label, family string, gb, tokS float64) CalibrationReceipt {
	return CalibrationReceipt{
		Key:          CalibrationKey{Epoch: goldenEpoch, Backend: "llama.cpp-vulkan", QuantFamily: family, KernelPath: "fa-on", BatchBucket: BatchBucket(1)},
		Kind:         ReceiptDecode,
		Label:        label,
		BytesPerStep: gb * 1e9,
		StepSeconds:  1 / tokS,
		MeasuredAt:   "2026-10-07T00:34:00Z",
		Provenance:   Provenance{Path: "receipts/" + label + ".json", SHA256: "sha-" + label},
	}
}

// goldenReceipts are four llama.cpp Vulkan decode receipts on one APU and build: two K-quant
// arms spanning 5.1x bytes, one Q8_0 arm and one IQ4_XS arm.
func goldenReceipts() []CalibrationReceipt {
	return []CalibrationReceipt{
		decodeReceipt("dense-27b-q2k", "k", 9.80, 18.54),
		decodeReceipt("moe-35b-a3b-q4k", "k", 1.92, 56.61),
		decodeReceipt("moe-35b-a3b-q8", "q8_0", 3.15, 51.33),
		decodeReceipt("moe-125b-a6b-iq4xs", "iq", 4.50, 25.66),
	}
}

func goldenStore() CalibrationStore {
	s := NewCalibrationStore()
	s.Add(goldenReceipts()...)
	return s
}

func rowFor(t *testing.T, s CalibrationStore, family string) CalibrationRow {
	t.Helper()
	for _, r := range s.Rows {
		if r.Key.QuantFamily == family && r.Fit != FitAchievedBandwidth {
			return r
		}
	}
	t.Fatalf("no row for quant family %q in %d rows", family, len(s.Rows))
	return CalibrationRow{}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationGoldenTwoPointLine(t *testing.T) {
	k := rowFor(t, goldenStore(), "k")
	if k.Fit != FitLeastSquares || k.State != RowFit || k.Tier != TierCalibrated || k.Receipts != 2 {
		t.Fatalf("k row = %+v", k)
	}
	if bw := k.BytesPerSec / 1e9; math.Abs(bw-217.2) > 0.5 {
		t.Fatalf("BW_eff = %.2f GB/s, want 217.2 +/- 0.5", bw)
	}
	if c := k.OverheadSeconds * 1e3; math.Abs(c-8.83) > 0.05 {
		t.Fatalf("c = %.3f ms, want 8.83 +/- 0.05", c)
	}
	if len(k.Provenance) != 2 || k.Provenance[0].SHA256 == "" {
		t.Fatalf("provenance = %+v", k.Provenance)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationGoldenHeldOutFamiliesDrift(t *testing.T) {
	s := goldenStore()
	for _, tc := range []struct {
		family string
		resid  float64 // (predicted - measured) / measured step time against the k line
	}{
		{"q8_0", 0.197},
		{"iq", -0.242},
	} {
		r := rowFor(t, s, tc.family)
		if r.State != RowDrift || r.DriftReason != DriftSiblingLine || r.Tier != TierEmpirical {
			t.Fatalf("%s: state %s reason %s, want drift residual_vs_sibling_line", tc.family, r.State, r.DriftReason)
		}
		if len(r.Residuals) != 1 || math.Abs(r.Residuals[0].Relative-tc.resid) > 0.005 || r.Residuals[0].Against != "line:k" {
			t.Fatalf("%s: residuals %+v, want %+.3f against line:k", tc.family, r.Residuals, tc.resid)
		}
		err := r.Err()
		var de *DriftError
		if !errors.Is(err, ErrCalibrationDrift) || !errors.As(err, &de) || de.Reason != DriftSiblingLine {
			t.Fatalf("%s: Err() = %v", tc.family, err)
		}
	}
	if err := rowFor(t, s, "k").Err(); err != nil {
		t.Fatalf("k line Err() = %v, want nil", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationOneReceiptFoldsOverheadIntoBandwidth(t *testing.T) {
	s := NewCalibrationStore()
	s.Add(decodeReceipt("dense-27b-q2k", "k", 9.80, 18.54))
	r := rowFor(t, s, "k")
	if r.Fit != FitOnePoint || r.OverheadSeconds != 0 || r.State != RowFit {
		t.Fatalf("row = %+v", r)
	}
	if want := 9.80e9 * 18.54; !relEq(r.BytesPerSec, want) {
		t.Fatalf("BW = %v, want bytes x tok/s = %v", r.BytesPerSec, want)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationNarrowSpreadPoolsAndChecksOwnFit(t *testing.T) {
	s := NewCalibrationStore()
	// 2x spread is below LineMinSpread: the intercept is not identifiable, so the receipts pool.
	s.Add(decodeReceipt("a", "k", 4.0, 40), decodeReceipt("b", "k", 8.0, 10))
	r := rowFor(t, s, "k")
	if r.Fit != FitOnePoint || r.Receipts != 2 || len(r.Residuals) != 2 {
		t.Fatalf("row = %+v", r)
	}
	if r.State != RowDrift || r.DriftReason != DriftOwnFit {
		t.Fatalf("state %s reason %s, want drift residual_vs_own_fit", r.State, r.DriftReason)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationAddDedupesAndValidates(t *testing.T) {
	s := goldenStore()
	if n := s.Add(goldenReceipts()...); n != 0 {
		t.Fatalf("re-adding the same receipts added %d", n)
	}
	bad := decodeReceipt("x", "k", 1, 1)
	bad.Key.Epoch = ""
	if n := s.Add(bad); n != 0 || !errors.Is(bad.Validate(), ErrCalibrationReceiptInvalid) {
		t.Fatalf("invalid receipt added %d, Validate %v", n, bad.Validate())
	}
	if len(s.Receipts) != 4 {
		t.Fatalf("receipts = %d, want 4", len(s.Receipts))
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationStoreRoundTripRefitsRows(t *testing.T) {
	s := goldenStore()
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeCalibrationStore(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Rows, s.Rows) {
		t.Fatalf("round-trip rows differ:\n got %+v\nwant %+v", got.Rows, s.Rows)
	}
	var doc map[string]any
	_ = json.Unmarshal(raw, &doc)
	doc["schema"] = "fak.fitplan.calibration.v0"
	foreign, _ := json.Marshal(doc)
	if _, err := DecodeCalibrationStore(foreign); !errors.Is(err, ErrCalibrationSchema) {
		t.Fatalf("foreign schema err = %v, want ErrCalibrationSchema", err)
	}
}

func probeReceipt(epoch, sha string, gbps float64, at string) CalibrationReceipt {
	return CalibrationReceipt{
		Key:                 CalibrationKey{Epoch: epoch, Backend: "cpu"},
		Kind:                ReceiptBandwidth,
		AchievedBytesPerSec: gbps * 1e9,
		MeasuredAt:          at,
		Provenance:          Provenance{Path: "roofline-" + sha + ".json", SHA256: sha},
	}
}

func engineQuery(epoch string) CalibrationQuery {
	return CalibrationQuery{
		Key:          CalibrationKey{Epoch: epoch, Backend: "fak-cpu", QuantFamily: "q8_0", KernelPath: "cpu-lean-q8", BatchBucket: "1"},
		ProbeBackend: "cpu",
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationResolveAchievedBandwidthProbe(t *testing.T) {
	s := NewCalibrationStore()
	s.Add(probeReceipt("e1", "p1", 40, "2026-10-20T00:00:00Z"), probeReceipt("e1", "p2", 44, "2026-10-21T00:00:00Z"),
		probeReceipt("e1", "p3", 42, "2026-10-22T00:00:00Z"))
	cal, ok := s.Resolve(engineQuery("e1"))
	// A hardware probe is an upper bound, not an engine fit: EMPIRICAL, never CALIBRATED.
	if !ok || cal.Tier != TierEmpirical || cal.Row == nil || cal.Row.Fit != FitAchievedBandwidth || cal.State != RowFit {
		t.Fatalf("resolve = %+v ok %v", cal, ok)
	}
	if !relEq(cal.AchievedBytesPerSec, 42e9) || cal.OverheadSeconds != 0 || cal.ComputeTier != TierEmpirical {
		t.Fatalf("achieved %v overhead %v compute tier %s", cal.AchievedBytesPerSec, cal.OverheadSeconds, cal.ComputeTier)
	}
	if !cal.valid() {
		t.Fatal("a probe calibration with no efficiency is not valid")
	}
	if _, ok := s.Resolve(CalibrationQuery{Key: CalibrationKey{Epoch: "e1", Backend: "fak-vulkan"}, ProbeBackend: "vulkan0"}); ok {
		t.Fatal("a cpu probe resolved for another device")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationProbeSpreadDrifts(t *testing.T) {
	s := NewCalibrationStore()
	// Each reading is within 10% of the 43 GB/s median; 40 vs 47 is 17.5% apart.
	s.Add(probeReceipt("e1", "p1", 40, ""), probeReceipt("e1", "p2", 43, ""), probeReceipt("e1", "p3", 47, ""))
	cal, _ := s.Resolve(engineQuery("e1"))
	var de *DriftError
	if cal.State != RowDrift || cal.Tier != TierEmpirical || !errors.As(cal.Row.Err(), &de) || de.Reason != DriftProbeSpread {
		t.Fatalf("spread probes: state %s tier %s err %v", cal.State, cal.Tier, cal.Row.Err())
	}
}

// engineReceipt is a fak decode receipt on engineQuery's key.
func engineReceipt(epoch, sha string, gb, tokS float64, at string) CalibrationReceipt {
	q := engineQuery(epoch)
	return CalibrationReceipt{Key: q.Key, Kind: ReceiptDecode, BytesPerStep: gb * 1e9, StepSeconds: 1 / tokS,
		MeasuredAt: at, Provenance: Provenance{Path: sha + ".json", SHA256: sha}}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationEpochChangeCarriesPriorAsEmpirical(t *testing.T) {
	s := NewCalibrationStore()
	// The later stamp is in a non-UTC offset: ordering must follow the instant, not the string.
	s.Add(engineReceipt("old-a", "d1", 2, 10, "2026-10-10T20:00:00-08:00"), engineReceipt("old-b", "d2", 2, 12, "2026-10-11T01:00:00Z"))
	cal, ok := s.Resolve(engineQuery("new"))
	if !ok || cal.Tier != TierEmpirical || cal.Row.PriorEpoch != "old-a" || !relEq(cal.AchievedBytesPerSec, 2e9*10) {
		t.Fatalf("carried = %+v ok %v", cal, ok)
	}
	s.Add(engineReceipt("new", "d3", 2, 15, "2026-10-23T00:00:00Z"))
	cal, _ = s.Resolve(engineQuery("new"))
	if cal.Tier != TierCalibrated || cal.Row.Key.Epoch != "new" || cal.Row.PriorEpoch != "" {
		t.Fatalf("new epoch row = %+v", cal.Row)
	}
	if n := len(s.Rows); n != 3 {
		t.Fatalf("rows = %d, want one per epoch", n)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationIdentityKeepsUnlabelledPointsOnOneKey(t *testing.T) {
	s := NewCalibrationStore()
	a, b := engineReceipt("e1", "same-file", 1, 50, ""), engineReceipt("e1", "same-file", 5, 15, "")
	if n := s.Add(a, b); n != 2 {
		t.Fatalf("added %d of two distinct points from one file", n)
	}
	if r := s.Rows[0]; r.Fit != FitLeastSquares || r.Receipts != 2 {
		t.Fatalf("row = %+v, want a two-point line", r)
	}
	if n := s.Add(a); n != 0 {
		t.Fatalf("re-adding a point added %d", n)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationNegativeInterceptRefitsThroughOriginWithResiduals(t *testing.T) {
	s := NewCalibrationStore()
	// (1 GB, 1 ms) and (4 GB, 20 ms): the free line has c < 0; through the origin it misses the
	// first point by far more than 15%, which a one-parameter fit has the dof to see.
	s.Add(engineReceipt("e1", "o1", 1, 1000, ""), engineReceipt("e1", "o2", 4, 50, ""))
	r := s.Rows[0]
	if r.Fit != FitThroughOrigin || r.OverheadSeconds != 0 || len(r.Residuals) != 2 || r.State != RowDrift || r.DriftReason != DriftOwnFit {
		t.Fatalf("row = %+v", r)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestAchievedBandwidthIsCappedAtTheSpecPeak(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Hardware.Peaks.FLOPsPerSec = 0
	s := NewCalibrationStore()
	s.Add(probeReceipt("e1", "p1", 128, ""))
	in.Calibration, _ = s.Resolve(engineQuery("e1"))
	rep := plan(t, in)
	pr := rep.Candidates[0]
	if !relEq(rep.Evidence.Calibration.BandwidthEfficiency, 0.5) {
		t.Fatalf("evidence e_bw = %v, want achieved/peak = 0.5", rep.Evidence.Calibration.BandwidthEfficiency)
	}
	in.Calibration, _ = func() (Calibration, bool) {
		s := NewCalibrationStore()
		s.Add(probeReceipt("e1", "p2", 300, ""))
		return s.Resolve(engineQuery("e1"))
	}()
	over := plan(t, in).Candidates[0]
	if !relEq(over.DecodeTokSPerSeq, over.DecodeTokSCeiling) || !(over.DecodeTokSPerSeq > pr.DecodeTokSPerSeq) {
		t.Fatalf("achieved above peak: decode %v ceiling %v", over.DecodeTokSPerSeq, over.DecodeTokSCeiling)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCalibrationResolveOrderExactThenLineThenProbe(t *testing.T) {
	s := goldenStore()
	s.Add(probeReceipt(goldenEpoch, "p1", 120, ""))
	q := CalibrationQuery{Key: rowFor(t, s, "k").Key, ProbeBackend: "cpu"}
	if cal, _ := s.Resolve(q); cal.Row.Fit != FitLeastSquares || cal.Row.InheritedFromFamily != "" {
		t.Fatalf("exact k: %+v", cal.Row)
	}
	q.Key.QuantFamily = "legacy" // no row of its own: the k line stands in
	if cal, _ := s.Resolve(q); cal.Row.Fit != FitLeastSquares || cal.Row.InheritedFromFamily != "k" {
		t.Fatalf("inherited: %+v", cal.Row)
	}
	q.Key.QuantFamily = "iq" // its own row drifted: stated with the verdict, not replaced
	if cal, _ := s.Resolve(q); cal.State != RowDrift || cal.Row.Key.QuantFamily != "iq" {
		t.Fatalf("drifted: %+v", cal.Row)
	}
	q.Key.Backend = "fak-cpu" // no decode row on this backend: the probe
	if cal, _ := s.Resolve(q); cal.Row.Fit != FitAchievedBandwidth {
		t.Fatalf("probe: %+v", cal.Row)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeStepAddsOverhead(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Hardware.Peaks.FLOPsPerSec = 0
	base := plan(t, in).Candidates[0]
	in.Calibration.OverheadSeconds = 0.005
	pr := plan(t, in).Candidates[0]
	if want := 1 / (1/base.DecodeTokSPerSeq + 0.005); !relEq(pr.DecodeTokSPerSeq, want) {
		t.Fatalf("decode with t_ovh = %v, want 1/(1/%v + 5ms) = %v", pr.DecodeTokSPerSeq, base.DecodeTokSPerSeq, want)
	}
	if pr.DecodeTokSCeiling != base.DecodeTokSCeiling {
		t.Fatalf("SOL ceiling moved with t_ovh: %v -> %v", base.DecodeTokSCeiling, pr.DecodeTokSCeiling)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestAchievedBandwidthStatesDecodeWithoutPeak(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Hardware.Peaks = Peaks{}
	s := NewCalibrationStore()
	s.Add(probeReceipt("e1", "p1", 50, ""))
	in.Calibration, _ = s.Resolve(engineQuery("e1"))
	rep := plan(t, in)
	pr := rep.Candidates[0]
	if want := 50e9 / float64(pr.DecodeBytesPerStep); !relEq(pr.DecodeTokSPerSeq, want) {
		t.Fatalf("decode = %v, want achieved / step bytes = %v", pr.DecodeTokSPerSeq, want)
	}
	if pr.DecodeTokSCeiling != 0 || !hasOmission(pr.Omitted, OmitPeakBandwidthUnknown) || hasOmission(pr.Omitted, OmitCalibrationInvalid) {
		t.Fatalf("ceiling %v omitted %v", pr.DecodeTokSCeiling, pr.Omitted)
	}
	if rep.Evidence.Calibration.Tier != TierEmpirical || rep.Evidence.Calibration.Row == nil || len(rep.Evidence.Calibration.Row.Provenance) != 1 {
		t.Fatalf("evidence calibration = %+v", rep.Evidence.Calibration)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestQuantFamilyAndBatchBucketAreClosed(t *testing.T) {
	for in, want := range map[string]string{"Q2_K": "k", "q6_k": "k", "Q8_0": "q8_0", "IQ4_XS": "iq", "Q4_0": "legacy", "BF16": "float", "MXFP4": "mxfp4", "": ""} {
		if got := QuantFamily(in); got != want {
			t.Fatalf("QuantFamily(%q) = %q, want %q", in, got, want)
		}
	}
	for b, want := range map[int]string{0: "1", 1: "1", 3: "3", 6: "5-8", 64: "9-64", 65: ">64"} {
		if got := BatchBucket(b); got != want {
			t.Fatalf("BatchBucket(%d) = %q, want %q", b, got, want)
		}
	}
}
