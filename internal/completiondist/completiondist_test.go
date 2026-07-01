package completiondist

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetcap"
)

// fixture is a hand-verified set of 10 historical issue closures. Durations (in
// seconds) are chosen so the nearest-rank median and p95 land on exact input
// elements and every DefaultBuckets bucket is populated:
//
//	sorted seconds: 300 600 720 1200 1800 2400 3000 5400 9000 20000
//	  <15m  (<900):        300, 600, 720            -> 3
//	  15-60m (900..3600):  1200, 1800, 2400, 3000   -> 4
//	  1-4h  (3600..14400): 5400, 9000               -> 2
//	  >4h   (>=14400):     20000                     -> 1
//	  p50 = rank ceil(.50*10)=5  -> sorted[4]  = 1800
//	  p95 = rank ceil(.95*10)=10 -> sorted[9]  = 20000
//	  min = 300, max = 20000, sum = 44420, mean = 4442
func fixture() []ClosureSample {
	return []ClosureSample{
		{Issue: 1, DurationSec: 1800},
		{Issue: 2, DurationSec: 300},
		{Issue: 3, DurationSec: 9000},
		{Issue: 4, DurationSec: 2400},
		{Issue: 5, DurationSec: 600},
		{Issue: 6, DurationSec: 20000},
		{Issue: 7, DurationSec: 3000},
		{Issue: 8, DurationSec: 720},
		{Issue: 9, DurationSec: 5400},
		{Issue: 10, DurationSec: 1200},
	}
}

func TestBuild_HandVerifiedStats(t *testing.T) {
	d := Build(fixture())

	if d.Count != 10 {
		t.Errorf("Count = %d, want 10", d.Count)
	}
	if d.MinSec != 300 {
		t.Errorf("MinSec = %g, want 300", d.MinSec)
	}
	if d.MaxSec != 20000 {
		t.Errorf("MaxSec = %g, want 20000", d.MaxSec)
	}
	if d.MeanSec != 4442 {
		t.Errorf("MeanSec = %g, want 4442", d.MeanSec)
	}
	if d.MedianSec() != 1800 {
		t.Errorf("MedianSec() = %g, want 1800", d.MedianSec())
	}
	if d.P95() != 20000 {
		t.Errorf("P95() = %g, want 20000", d.P95())
	}
}

func TestBuild_HandVerifiedBuckets(t *testing.T) {
	d := Build(fixture())
	got := map[string]int{}
	for _, b := range d.Buckets {
		got[b.Label] = b.Count
	}
	want := map[string]int{"<15m": 3, "15-60m": 4, "1-4h": 2, ">4h": 1}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("bucket counts = %v, want %v", got, want)
	}
	// Every sample must land in exactly one bucket: counts sum to Count.
	total := 0
	for _, b := range d.Buckets {
		total += b.Count
	}
	if total != d.Count {
		t.Errorf("bucket counts sum to %d, want Count=%d", total, d.Count)
	}
}

// TestMedianDrivesRequiredWorkers is the composition this leaf exists to prove:
// the distribution's REAL median (not a guessed one) drives
// fleetcap.RequiredWorkers. Median 1800s == 30 min; at 400 issues/hour Little's
// law gives L = 400 * (30/60) = 200 workers.
func TestMedianDrivesRequiredWorkers(t *testing.T) {
	d := Build(fixture())

	// The capacity model consumes the distribution's median, converted to the
	// minutes fleetcap expects — no hard-coded median anywhere in this chain.
	medianMinutes := d.MedianSec() / 60.0
	want := fleetcap.RequiredWorkers(400, medianMinutes)
	if want != 200 {
		t.Fatalf("sanity: RequiredWorkers(400, %g) = %d, want 200", medianMinutes, want)
	}

	// The convenience method must agree with the explicit composition.
	if got := d.RequiredWorkersAtMedian(400); got != want {
		t.Errorf("RequiredWorkersAtMedian(400) = %d, want %d", got, want)
	}

	// Prove it is DATA-driven, not fixed: a distribution with a smaller median
	// must require fewer workers at the same rate.
	fast := Build([]ClosureSample{
		{Issue: 1, DurationSec: 300}, // 5 min
		{Issue: 2, DurationSec: 600}, // 10 min -> median
		{Issue: 3, DurationSec: 900}, // 15 min
	})
	// median = sorted[ceil(.5*3)-1] = sorted[1] = 600s = 10 min.
	// L = 400 * (10/60) = 66.67 -> 67 workers.
	if got := fast.RequiredWorkersAtMedian(400); got != 67 {
		t.Errorf("fast fixture RequiredWorkersAtMedian(400) = %d, want 67", got)
	}
	if fast.RequiredWorkersAtMedian(400) >= want {
		t.Errorf("smaller median (%gs) should need fewer workers than %gs",
			fast.MedianSec(), d.MedianSec())
	}
}

func TestBuild_Empty(t *testing.T) {
	d := Build(nil)
	if d.Count != 0 || d.MinSec != 0 || d.MaxSec != 0 || d.MeanSec != 0 ||
		d.MedianSec() != 0 || d.P95() != 0 {
		t.Errorf("empty distribution not all-zero: %+v", d)
	}
	// Buckets are still present (edges preserved) but every count is 0.
	if len(d.Buckets) != len(DefaultBuckets()) {
		t.Errorf("empty distribution Buckets len = %d, want %d",
			len(d.Buckets), len(DefaultBuckets()))
	}
	for _, b := range d.Buckets {
		if b.Count != 0 {
			t.Errorf("empty distribution bucket %q count = %d, want 0", b.Label, b.Count)
		}
	}
	// Empty median -> fleetcap needs no standing worker.
	if got := d.RequiredWorkersAtMedian(400); got != 0 {
		t.Errorf("empty RequiredWorkersAtMedian(400) = %d, want 0", got)
	}
}

func TestBuild_Single(t *testing.T) {
	d := Build([]ClosureSample{{Issue: 42, DurationSec: 1200}})
	// One sample: min==max==mean==p50==p95==that value.
	if d.Count != 1 {
		t.Fatalf("Count = %d, want 1", d.Count)
	}
	for name, got := range map[string]float64{
		"MinSec": d.MinSec, "MaxSec": d.MaxSec, "MeanSec": d.MeanSec,
		"MedianSec": d.MedianSec(), "P95": d.P95(),
	} {
		if got != 1200 {
			t.Errorf("single-sample %s = %g, want 1200", name, got)
		}
	}
}

func TestBuildWith_CustomBuckets(t *testing.T) {
	// Two coarse buckets split at 1000s over the fixture:
	//   under1000 (<1000): 300, 600, 720            -> 3
	//   over1000  (>=1000): 1200,1800,2400,3000,5400,9000,20000 -> 7
	buckets := []Bucket{
		{Label: "under1000", MinSec: 0, MaxSec: 1000},
		{Label: "over1000", MinSec: 1000, MaxSec: 0}, // unbounded top
	}
	d := BuildWith(fixture(), buckets)
	got := map[string]int{}
	for _, b := range d.Buckets {
		got[b.Label] = b.Count
	}
	want := map[string]int{"under1000": 3, "over1000": 7}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("custom bucket counts = %v, want %v", got, want)
	}
	// Summary stats are unaffected by bucket choice.
	if d.MedianSec() != 1800 || d.P95() != 20000 {
		t.Errorf("custom-bucket median/p95 = %g/%g, want 1800/20000", d.MedianSec(), d.P95())
	}
}

func TestParseSamples(t *testing.T) {
	data := []byte(`{"issue":1,"duration_sec":300}
{"issue":2,"duration_sec":1800}

  {"issue":3,"duration_sec":9000}
`)
	got, err := ParseSamples(data)
	if err != nil {
		t.Fatalf("ParseSamples: %v", err)
	}
	want := []ClosureSample{
		{Issue: 1, DurationSec: 300},
		{Issue: 2, DurationSec: 1800},
		{Issue: 3, DurationSec: 9000},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ParseSamples = %v, want %v", got, want)
	}

	// A negative duration is a located error, not a silent bad sample.
	if _, err := ParseSamples([]byte(`{"issue":1,"duration_sec":-5}`)); err == nil {
		t.Error("ParseSamples accepted negative duration_sec, want error")
	}
	// Malformed JSON is a located error.
	if _, err := ParseSamples([]byte("not json")); err == nil {
		t.Error("ParseSamples accepted malformed line, want error")
	}
}

// TestParseThenBuild proves the JSONL reader feeds Build end-to-end: parsed
// fixture durations fold to the same hand-verified median as the struct fixture.
func TestParseThenBuild(t *testing.T) {
	data := []byte(`{"issue":1,"duration_sec":300}
{"issue":2,"duration_sec":600}
{"issue":3,"duration_sec":720}
{"issue":4,"duration_sec":1200}
{"issue":5,"duration_sec":1800}
{"issue":6,"duration_sec":2400}
{"issue":7,"duration_sec":3000}
{"issue":8,"duration_sec":5400}
{"issue":9,"duration_sec":9000}
{"issue":10,"duration_sec":20000}`)
	samples, err := ParseSamples(data)
	if err != nil {
		t.Fatalf("ParseSamples: %v", err)
	}
	d := Build(samples)
	if d.Count != 10 || d.MedianSec() != 1800 || d.P95() != 20000 {
		t.Errorf("parsed-then-built = count %d median %g p95 %g, want 10/1800/20000",
			d.Count, d.MedianSec(), d.P95())
	}
}
