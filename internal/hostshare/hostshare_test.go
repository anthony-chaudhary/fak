package hostshare

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func approx(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("got %v, want %v (tol %v)", got, want, tol)
	}
}

func mustReading(t *testing.T, rs []Reading, pathSuffix string) Reading {
	t.Helper()
	for _, r := range rs {
		if strings.HasSuffix(r.Path, pathSuffix) {
			return r
		}
	}
	t.Fatalf("no reading for %q in %+v", pathSuffix, rs)
	return Reading{}
}

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestScanTestdata walks the committed fixture tree end to end: every evidence
// shape derives its documented value, recency orders oldest-first, and the gate
// fires on the most recent (launch-bound) reading.
func TestScanTestdata(t *testing.T) {
	rep, err := ScanReport([]string{"testdata"})
	if err != nil {
		t.Fatal(err)
	}
	if rep.FilesScanned != 8 || len(rep.Readings) != 9 {
		t.Fatalf("files=%d readings=%d, want 8 files / 9 readings: %+v", rep.FilesScanned, len(rep.Readings), rep.Readings)
	}
	if len(rep.MalformedFiles) != 0 {
		t.Fatalf("malformed: %v", rep.MalformedFiles)
	}

	cases := map[string]struct {
		want float64
		src  string
	}{
		"phl051-device-busy-note.json":      {0.13, SourceDeviceBusyFraction},
		"phl127-ms-split.json":              {1 - 71.9/75.2, SourceMSPerTokenSplit},
		"launch-bound-kernels.json":         {0.45, SourceDeviceBusyFraction},
		"recorded-host-share-wins.json":     {0.13, SourceHostShareRecorded},
		"perf-receipt-v2-nobusy.json":       {math.NaN(), ""},                    // UNKNOWN: no device-busy data
		"perf-receipt-v3-hostoverhead.json": {0.2, SourceHostPhaseDecomposition}, // native v3 decomposition
		"invalid-evidence-20261015.json":    {math.NaN(), ""},                    // corrupt meter: never fabricated
	}
	for suffix, want := range cases {
		r := mustReading(t, rep.Readings, suffix)
		if math.IsNaN(want.want) {
			if r.HostShare != nil {
				t.Errorf("%s: HostShare = %v, want nil (UNKNOWN)", suffix, *r.HostShare)
			}
		} else if r.HostShare == nil {
			t.Errorf("%s: HostShare nil, want %v", suffix, want.want)
		} else {
			approx(t, *r.HostShare, want.want, 1e-4)
		}
		if r.Source != want.src {
			t.Errorf("%s: source %q, want %q", suffix, r.Source, want.src)
		}
	}

	// Both JSONL lines are admitted, newest last.
	var jsonl []Reading
	for _, r := range rep.Readings {
		if strings.HasSuffix(r.Path, "perf-record-jsonl.jsonl") {
			jsonl = append(jsonl, r)
		}
	}
	if len(jsonl) != 2 {
		t.Fatalf("jsonl readings = %d, want 2", len(jsonl))
	}
	approx(t, *jsonl[0].HostShare, 0.23, 1e-9) // gpu_busy_percent 77
	approx(t, *jsonl[1].HostShare, 0.60, 1e-9) // gpu_busy_percent 40
	if !jsonl[0].Timestamp.Before(jsonl[1].Timestamp) {
		t.Errorf("jsonl recency order wrong: %v then %v", jsonl[0].Timestamp, jsonl[1].Timestamp)
	}

	// Most recent reading overall is the launch-bound window; the gate fires.
	gated, exceeded, ok := Evaluate(rep.Readings, 0.30)
	if !ok || !exceeded {
		t.Fatalf("Evaluate = (%+v, %v, %v), want exceeded launch-bound reading", gated, exceeded, ok)
	}
	if !strings.HasSuffix(gated.Path, "launch-bound-kernels.json") {
		t.Errorf("gated = %s, want launch-bound-kernels.json", gated.Path)
	}
}

// TestRecordedHostShareWins pins the priority rule: a directly recorded
// host_share beats the device-busy derivation even when both are present.
func TestRecordedHostShareWins(t *testing.T) {
	rs, err := Scan([]string{"testdata/recorded-host-share-wins.json"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 {
		t.Fatalf("readings = %+v", rs)
	}
	if rs[0].HostShare == nil || *rs[0].HostShare != 0.13 {
		t.Fatalf("HostShare = %v, want 0.13 (recorded wins over device_busy 0.8)", rs[0].HostShare)
	}
	if rs[0].Model != "Qwen3.8-27B-UD-Q2_K_XL" || rs[0].Hardware != "strix1" {
		t.Errorf("identity = (%q, %q)", rs[0].Model, rs[0].Hardware)
	}
}

// TestPerfReceiptV2WithoutDeviceBusyIsUnknown guards the honesty invariant: a
// perf-receipt-v2 whose repetitions carry no device-busy data reads UNKNOWN,
// never zero, and never fails the gate.
func TestPerfReceiptV2WithoutDeviceBusyIsUnknown(t *testing.T) {
	rs, err := Scan([]string{"testdata/perf-receipt-v2-nobusy.json"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 {
		t.Fatalf("readings = %+v", rs)
	}
	r := rs[0]
	if r.HostShare != nil {
		t.Fatalf("HostShare = %v, want nil", *r.HostShare)
	}
	if Verdict(r.HostShare, 0.30) != "UNKNOWN" {
		t.Errorf("verdict = %q, want UNKNOWN", Verdict(r.HostShare, 0.30))
	}
	if r.Model != "qwen38-cuda-hostshare-fixture" || r.Hardware != "linux/amd64/cuda" {
		t.Errorf("identity = (%q, %q)", r.Model, r.Hardware)
	}
	if _, exceeded, _ := Evaluate(rs, 0.30); exceeded {
		t.Error("UNKNOWN reading must not exceed the threshold")
	}
}

// TestPerfReceiptV3HostOverheadDecomposition is the consumer smoke: a v3
// receipt whose repetitions natively carry the host-overhead decomposition
// yields a measured host-share reading (mean of host_phase_us over
// decode_milliseconds across recording repetitions) instead of the UNKNOWN
// reading its v2 predecessor produced for the same workload.
func TestPerfReceiptV3HostOverheadDecomposition(t *testing.T) {
	rs, err := Scan([]string{"testdata/perf-receipt-v3-hostoverhead.json"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 {
		t.Fatalf("readings = %+v", rs)
	}
	r := rs[0]
	if r.HostShare == nil {
		t.Fatalf("HostShare nil: v3 decomposition must yield a measured reading")
	}
	approx(t, *r.HostShare, 0.2, 1e-9) // (16000/80000 + 20000/100000) / 2
	if r.Source != SourceHostPhaseDecomposition {
		t.Errorf("source = %q, want %q", r.Source, SourceHostPhaseDecomposition)
	}
	if Verdict(r.HostShare, 0.30) != "OK" {
		t.Errorf("verdict = %q, want OK at 0.2 below the 0.30 gate", Verdict(r.HostShare, 0.30))
	}
	if r.Model != "qwen38-cuda-hostoverhead-fixture" || r.Hardware != "linux/amd64/cuda" {
		t.Errorf("identity = (%q, %q)", r.Model, r.Hardware)
	}
}

// TestNeverFabricatesZero checks both halves of the invariant: corrupt meter
// values yield no derivation, and a record with no decode-step signal at all
// yields no reading rather than a zero row.
func TestNeverFabricatesZero(t *testing.T) {
	rs, err := Scan([]string{"testdata/invalid-evidence-20261015.json"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 || rs[0].HostShare != nil {
		t.Fatalf("corrupt meter readings = %+v, want one UNKNOWN reading", rs)
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokens-only.json"),
		[]byte(`{"prompt_tokens":64,"model":"m"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	rep, err := ScanReport([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Readings) != 0 || rep.EvidenceFree != 1 {
		t.Fatalf("non-decode record: readings=%d evidenceFree=%d, want 0/1", len(rep.Readings), rep.EvidenceFree)
	}
}

// TestRecencyOrdering pins the sort contract: recorded timestamp fields beat
// file-name dates, file-name dates beat nothing, and undated readings sort
// first.
func TestRecencyOrdering(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"undated.json":        `{"device_busy":0.9,"decode_steps":1}`,
		"named-20260901.json": `{"device_busy":0.8,"decode_steps":1}`,
		"explicit.json":       `{"device_busy":0.7,"decode_steps":1,"observed_utc":"2026-08-01T00:00:00Z"}`,
		"x-20260101.json":     `{"device_busy":0.6,"decode_steps":1,"observed_utc":"2026-08-01T00:00:00Z"}`,
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	rs, err := Scan([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 4 {
		t.Fatalf("readings = %+v", rs)
	}
	// undated (zero) < 2026-08-01 field-dated pair (path tie-break) < 2026-09-01 file-name dated
	wantOrder := []string{"undated.json", "explicit.json", "x-20260101.json", "named-20260901.json"}
	for i, want := range wantOrder {
		if !strings.HasSuffix(rs[i].Path, want) {
			t.Errorf("position %d = %s, want %s", i, rs[i].Path, want)
		}
	}
	fieldBeatsName := mustReading(t, rs, "x-20260101.json")
	if !fieldBeatsName.Timestamp.Equal(time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)) {
		t.Errorf("recorded observed_utc lost to the file-name date: %v", fieldBeatsName.Timestamp)
	}
	named := mustReading(t, rs, "named-20260901.json")
	if !named.Timestamp.Equal(time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)) {
		t.Errorf("file-name date = %v, want 2026-09-01 UTC", named.Timestamp)
	}
}

// TestGateMostRecentDecides pins the gate semantics: only the most recent
// reading decides, in either direction.
func TestGateMostRecentDecides(t *testing.T) {
	mk := func(name, busy, date string) string {
		return `{"device_busy":` + busy + `,"decode_steps":1,"observed_utc":"` + date + `"}`
	}
	oldBadNewGood := t.TempDir()
	writeTemp(t, "old-launch-bound.json", mk("x", "0.55", "2026-01-01T00:00:00Z"))
	if err := os.WriteFile(filepath.Join(oldBadNewGood, "new-healthy.json"),
		[]byte(mk("y", "0.87", "2026-02-01T00:00:00Z")), 0o644); err != nil {
		t.Fatal(err)
	}
	rs, err := Scan([]string{oldBadNewGood})
	if err != nil {
		t.Fatal(err)
	}
	if _, exceeded, _ := Evaluate(rs, 0.30); exceeded {
		t.Error("most recent reading is healthy (0.13); gate must not fire on an older 0.45")
	}

	oldGoodNewBad := t.TempDir()
	if err := os.WriteFile(filepath.Join(oldGoodNewBad, "old-healthy.json"),
		[]byte(mk("x", "0.87", "2026-01-01T00:00:00Z")), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(oldGoodNewBad, "new-launch-bound.json"),
		[]byte(mk("y", "0.55", "2026-02-01T00:00:00Z")), 0o644); err != nil {
		t.Fatal(err)
	}
	rs, err = Scan([]string{oldGoodNewBad})
	if err != nil {
		t.Fatal(err)
	}
	gated, exceeded, ok := Evaluate(rs, 0.30)
	if !ok || !exceeded {
		t.Fatalf("Evaluate = (%+v, %v, %v), want the new launch-bound reading to exceed", gated, exceeded, ok)
	}
	if !strings.HasSuffix(gated.Path, "new-launch-bound.json") {
		t.Errorf("gated = %s", gated.Path)
	}
}

// TestMalformedContentIsSkipped keeps a shared-tree scan robust against peers'
// half-written JSON: the bad file is listed, the good file still reads.
func TestMalformedContentIsSkipped(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "broken.json"), []byte(`{"device_busy": 0.`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "good.json"), []byte(`{"device_busy":0.87,"decode_steps":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	rep, err := ScanReport([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Readings) != 1 || len(rep.MalformedFiles) != 1 {
		t.Fatalf("rep = %+v", rep)
	}
	if !strings.HasSuffix(rep.MalformedFiles[0], "broken.json") {
		t.Errorf("malformed = %v", rep.MalformedFiles)
	}
}
