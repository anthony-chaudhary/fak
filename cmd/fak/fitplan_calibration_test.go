package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

var fitPlanTestTime = time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)

// Every cmd/fak test prices fit_plan against an empty calibration store unless it sets its own,
// so a developer's real store never moves a fit_plan number under test.
func init() {
	_ = os.Setenv(calibrationStoreEnv, filepath.Join(os.TempDir(), fmt.Sprintf("fak-cmd-test-no-calibration-%d.json", os.Getpid())))
}

func fitPlanTestStore(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "calibration.json")
	t.Setenv(calibrationStoreEnv, path)
	return path
}

func hostRoofline(gbps float64) modelperfobs.RooflineMeasurement {
	return modelperfobs.RooflineMeasurement{
		Schema: modelperfobs.RooflineMeasurementSchema, Scope: "host-memory", Method: "parallel-copy",
		MachineClass: runtime.GOOS + "/" + runtime.GOARCH, TrafficAccounting: "read-plus-write-2-bytes-per-copied-byte",
		WorkingSetBytes: 64 << 20, Threads: runtime.GOMAXPROCS(0), MeasuredSustainableGBS: gbps,
	}
}

func writeFitPlanTestJSON(t *testing.T, name string, v any) (string, string) {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return path, hex.EncodeToString(sum[:])
}

func planSynthetic(t *testing.T) serveSizingArtifact {
	t.Helper()
	art, err := buildServeSizingArtifact(serveSynthConfiguredWeightSource(t), nil, false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	return art
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanHostProbeStatesDecode(t *testing.T) {
	serveFitPlanTestEnv(t)
	store := fitPlanTestStore(t)
	receipt, sha := writeFitPlanTestJSON(t, "roofline.json", hostRoofline(40))
	var out bytes.Buffer
	if err := modelObserveBandwidthCalibrate([]string{"--store", store, "--measured-here", receipt}, &out); err != nil {
		t.Fatal(err)
	}
	var report fitPlanCalibrationReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil || report.Added != 1 || len(report.Rows) != 1 {
		t.Fatalf("calibrate report %+v (%v)", report, err)
	}

	art := planSynthetic(t)
	cal := art.FitPlan.Evidence.Calibration
	// A hardware probe is an achieved-bandwidth upper bound, not an engine fit: EMPIRICAL.
	if cal.Tier != fitplan.TierEmpirical || cal.Row == nil || cal.Row.Fit != fitplan.FitAchievedBandwidth || cal.State != fitplan.RowFit {
		t.Fatalf("evidence calibration = %+v", cal)
	}
	if len(cal.Row.Provenance) != 1 || cal.Row.Provenance[0].SHA256 != sha || cal.Lookup == nil || cal.Lookup.Epoch != fitPlanHardwareEpoch(nil).Digest() {
		t.Fatalf("provenance %+v lookup %+v", cal.Row.Provenance, cal.Lookup)
	}
	cur := serveFitPlanCurrent(t, art.FitPlan)
	if want := 40e9 / float64(cur.DecodeBytesPerStep); !(cur.DecodeTokSPerSeq > 0) || math.Abs(cur.DecodeTokSPerSeq-want) > 1e-9*want {
		t.Fatalf("decode %v tok/s, want achieved / step bytes = %v", cur.DecodeTokSPerSeq, want)
	}
	if !serveFitPlanOmits(cur.Omitted, fitplan.OmitPeakBandwidthUnknown) || serveFitPlanOmits(cur.Omitted, fitplan.OmitCalibrationInvalid) || cur.DecodeTokSCeiling != 0 {
		t.Fatalf("omitted %v ceiling %v: the SOL ceiling still needs a spec peak", cur.Omitted, cur.DecodeTokSCeiling)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanEngineReceiptsAreCalibrated(t *testing.T) {
	serveFitPlanTestEnv(t)
	store := fitPlanTestStore(t)
	look := planSynthetic(t).FitPlan.Evidence.Calibration.Lookup
	if look == nil {
		t.Fatal("no lookup key on an empty store")
	}
	// Two engine decode receipts on this host's key, 4x apart in bytes: a least-squares line.
	key := *look
	doc := fitplan.CalibrationReceipts{Schema: fitplan.CalibrationReceiptsSchema, Receipts: []fitplan.CalibrationReceipt{
		{Key: key, Kind: fitplan.ReceiptDecode, Label: "small", BytesPerStep: 1e9, StepSeconds: 0.015},
		{Key: key, Kind: fitplan.ReceiptDecode, Label: "large", BytesPerStep: 4e9, StepSeconds: 0.045},
	}}
	path, _ := writeFitPlanTestJSON(t, "receipts.json", doc)
	if err := modelObserveBandwidthCalibrate([]string{"--store", store, path}, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	art := planSynthetic(t)
	cal := art.FitPlan.Evidence.Calibration
	if cal.Tier != fitplan.TierCalibrated || cal.Row == nil || cal.Row.Fit != fitplan.FitLeastSquares || cal.Row.InheritedFromFamily != "" {
		t.Fatalf("evidence calibration = %+v", cal)
	}
	bw, c := 3e9/0.030, 0.005 // a = 30 ms / 3 GB, c = 15 ms - 10 ms
	if math.Abs(cal.AchievedBytesPerSec-bw) > 1e-6*bw || math.Abs(cal.OverheadSeconds-c) > 1e-9 {
		t.Fatalf("achieved %v overhead %v, want %v and %v", cal.AchievedBytesPerSec, cal.OverheadSeconds, bw, c)
	}
	cur := serveFitPlanCurrent(t, art.FitPlan)
	if want := 1 / (float64(cur.DecodeBytesPerStep)/bw + c); math.Abs(cur.DecodeTokSPerSeq-want) > 1e-6*want {
		t.Fatalf("decode %v tok/s, want 1/(bytes/BW + t_ovh) = %v", cur.DecodeTokSPerSeq, want)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanEmptyStoreKeepsGenericEmpirical(t *testing.T) {
	serveFitPlanTestEnv(t)
	fitPlanTestStore(t)
	art := planSynthetic(t)
	cal := art.FitPlan.Evidence.Calibration
	if cal.Tier != fitplan.TierEmpirical || cal.Row != nil || cal.Lookup == nil || cal.Lookup.Backend != "fak-cpu" || cal.Lookup.KernelPath != art.Arm {
		t.Fatalf("empty-store calibration = %+v", cal)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestServeSizingFitPlanUnreadableStoreFallsBackToGeneric(t *testing.T) {
	store := fitPlanTestStore(t)
	if err := os.WriteFile(store, []byte(`{"schema":"other"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if cal := resolveServeCalibration(nil, fitplan.Model{}, "cpu-lean-q8"); cal.Tier != fitplan.TierEmpirical || cal.Row != nil || cal.Lookup == nil {
		t.Fatalf("unreadable store calibration = %+v", cal)
	}
	if _, err := loadFitPlanCalibration(store); !errors.Is(err, fitplan.ErrCalibrationSchema) {
		t.Fatalf("load err = %v, want ErrCalibrationSchema", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestFitPlanReceiptRefusalsAreTyped(t *testing.T) {
	raw := func(s string) string { p, _ := writeFitPlanTestJSON(t, "r.json", json.RawMessage(s)); return p }
	small := hostRoofline(400)
	small.WorkingSetBytes = 1 << 20
	partial := hostRoofline(40)
	partial.Threads = runtime.GOMAXPROCS(0) - 1
	foreign := hostRoofline(40)
	foreign.MachineClass = "plan9/mips"
	file := func(m modelperfobs.RooflineMeasurement) string {
		p, _ := writeFitPlanTestJSON(t, "m.json", m)
		return p
	}
	cases := []struct {
		path string
		here bool
		want error
	}{
		{raw(`{"schema":"fak.roofline.empirical/v1","simulated":true,"dram_bandwidth":{"sustained_gbps":210}}`), true, errFitPlanReceiptSimulated},
		{file(small), true, errFitPlanReceiptBelowDRAM},
		{file(foreign), true, errFitPlanReceiptForeignMachine},
		{file(hostRoofline(40)), false, errFitPlanReceiptNoEpoch},
		{raw(`{"schema":"` + modelperfobs.BandwidthCollectionSchema + `"}`), true, errFitPlanReceiptNoMeasure},
		{raw(`{"schema":"fak-model-bandwidth/1"}`), true, errFitPlanReceiptUnsupported},
		{raw(`{"schema":"` + fitplan.CalibrationReceiptsSchema + `","receipts":[{"kind":"decode","bytes_per_step":1e9,"step_s":0.02}]}`), true, fitplan.ErrCalibrationReceiptInvalid},
	}
	if runtime.GOMAXPROCS(0) > 1 {
		cases = append(cases, struct {
			path string
			here bool
			want error
		}{file(partial), true, errFitPlanReceiptPartialThreads})
	}
	for i, tc := range cases {
		b, err := os.ReadFile(tc.path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fitPlanReceiptsFromFile(tc.path, b, fitPlanTestTime, tc.here); !errors.Is(err, tc.want) {
			t.Fatalf("case %d: err = %v, want %v", i, err, tc.want)
		}
	}
	store := fitPlanTestStore(t)
	if err := modelObserveBandwidthCalibrate([]string{"--store", store, "--measured-here", file(small)}, &bytes.Buffer{}); !errors.Is(err, errFitPlanReceiptBelowDRAM) {
		t.Fatalf("calibrate err = %v", err)
	}
	if _, err := os.Stat(store); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a refused ingest wrote the store: %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestFitPlanDecodeReceiptsFileKeepsItsEpoch(t *testing.T) {
	doc := fitplan.CalibrationReceipts{Schema: fitplan.CalibrationReceiptsSchema, Receipts: []fitplan.CalibrationReceipt{
		{Key: fitplan.CalibrationKey{Epoch: "other-host", Backend: "llama.cpp-vulkan", QuantFamily: "k"}, Kind: fitplan.ReceiptDecode, BytesPerStep: 9.8e9, StepSeconds: 1 / 18.54},
	}}
	raw, _ := json.Marshal(doc)
	rs, err := fitPlanReceiptsFromFile("receipts.json", raw, fitPlanTestTime, false)
	if err != nil || len(rs) != 1 || rs[0].Key.Epoch != "other-host" || rs[0].Provenance.SHA256 == "" || rs[0].MeasuredAt == "" {
		t.Fatalf("receipts %+v err %v", rs, err)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestModelObserveCollectCalibrateAddsHostProbe(t *testing.T) {
	store := fitPlanTestStore(t)
	if err := runModelObserveBandwidthCollect([]string{"--calibrate"}); err == nil {
		t.Fatal("--calibrate without --measure-host-roofline accepted")
	}
	output := filepath.Join(t.TempDir(), "collection.json")
	args := []string{"--count", "1", "--interval", "10ms", "--nvidia-device", "__fak_no_device__", "--measure-host-roofline",
		"--roofline-bytes", fmt.Sprint(64 << 20), "--roofline-trials", "3", "--roofline-duration", "10ms",
		"--output", output, "--pretty=false", "--calibrate"}
	if err := runModelObserveBandwidthCollect(args); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	s, err := loadFitPlanCalibration(store)
	if err != nil || len(s.Rows) != 1 {
		t.Fatalf("store rows %+v err %v", s.Rows, err)
	}
	row := s.Rows[0]
	abs, _ := filepath.Abs(output)
	if row.Fit != fitplan.FitAchievedBandwidth || !(row.BytesPerSec > 0) || row.Key.Epoch != fitPlanHardwareEpoch(nil).Digest() ||
		row.Provenance[0].Path != abs || row.Provenance[0].SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("row = %+v", row)
	}
}
