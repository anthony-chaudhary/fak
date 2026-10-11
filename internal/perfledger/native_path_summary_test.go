package perfledger

import (
	"encoding/json"
	"math"
	"testing"
)

type nativePathStatsWire struct {
	Count              int      `json:"count"`
	PrefillMeasured    int      `json:"prefill_measured"`
	DecodeMeasured     int      `json:"decode_measured"`
	PrefillTPSP50      *float64 `json:"prefill_tps_p50"`
	PrefillTPSWeighted *float64 `json:"prefill_tps_weighted"`
	DecodeTPSP50       *float64 `json:"decode_tps_p50"`
	DecodeTPSWeighted  *float64 `json:"decode_tps_weighted"`
	CacheHitShare      float64  `json:"cache_hit_share"`
}

func nativePathRecord(path string, prompt, completion, cached, prefillTokens, decodeTokens int, prefillMS, decodeMS float64) Record {
	return Record{
		Schema:           Schema,
		PromptTokens:     prompt,
		CompletionTokens: completion,
		CachedTokens:     cached,
		TTFTMS:           1,
		PrefillTPS:       999999,
		DecodeTPS:        888888,
		NativeTiming:     NewNativeTiming(prefillTokens, decodeTokens, prefillMS, decodeMS),
		Engine:           &Engine{Path: path},
	}
}

func closeNativePathStat(t *testing.T, field string, got *float64, want float64) {
	t.Helper()
	if got == nil || math.Abs(*got-want) > 0.0001 {
		t.Fatalf("%s=%v, want %v", field, got, want)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestNativePathSummaryAssociatesOnlyMeasuredNativePaths(t *testing.T) {
	recs := []Record{
		nativePathRecord(PathSerial, 200, 10, 100, 200, 10, 2000, 1000),
		nativePathRecord(PathSerial, 300, 20, 0, 300, 20, 1000, 1000),
		nativePathRecord(PathSerial, 50, 9, 0, 50, 9, 0, 0),
		nativePathRecord(PathBatched, 128, 6, 128, 128, 6, 1000, 1000),
		nativePathRecord(PathBatched, 256, 16, 0, 256, 16, 2000, 1000),
		nativePathRecord(PathSpeculative, 200, 16, 200, 200, 16, 500, 0),
		nativePathRecord("unknown", 1000, 101, 1000, 1000, 101, 1, 1),
		{Schema: Schema, Locality: LocalityVendor, PromptTokens: 1000, CompletionTokens: 101, CachedTokens: 1000, NativeTiming: NewNativeTiming(1000, 101, 1, 1)},
	}
	// Persisted derived rates and client-observed rates are both untrusted inputs;
	// the fold must recompute from the engine-authored token/duration pairs.
	recs[0].NativeTiming.PrefillTPS = 777777
	recs[0].NativeTiming.DecodeTPS = 666666

	report := BuildReport(recs, len(recs), false, 0)
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Summary struct {
			ByPath      map[string]int                 `json:"by_path"`
			ByPathStats map[string]nativePathStatsWire `json:"by_path_stats"`
		} `json:"summary"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Summary.ByPath[PathSerial] != 3 || wire.Summary.ByPath[PathBatched] != 2 || wire.Summary.ByPath[PathSpeculative] != 1 || wire.Summary.ByPath["unknown"] != 1 {
		t.Fatalf("legacy by_path changed: %v", wire.Summary.ByPath)
	}
	if len(wire.Summary.ByPathStats) != len(Paths) {
		t.Fatalf("by_path_stats=%v, want exactly %v", wire.Summary.ByPathStats, Paths)
	}
	if _, ok := wire.Summary.ByPathStats["unknown"]; ok {
		t.Fatalf("unknown path entered native stats: %v", wire.Summary.ByPathStats)
	}

	serial := wire.Summary.ByPathStats[PathSerial]
	if serial.Count != 3 || serial.PrefillMeasured != 2 || serial.DecodeMeasured != 2 {
		t.Fatalf("serial count/measured=%d/%d/%d, want 3/2/2", serial.Count, serial.PrefillMeasured, serial.DecodeMeasured)
	}
	closeNativePathStat(t, "serial p50 prefill", serial.PrefillTPSP50, 100)
	closeNativePathStat(t, "serial weighted prefill", serial.PrefillTPSWeighted, 166.67)
	closeNativePathStat(t, "serial p50 decode", serial.DecodeTPSP50, 9)
	closeNativePathStat(t, "serial weighted decode", serial.DecodeTPSWeighted, 14)
	if math.Abs(serial.CacheHitShare-100.0/650.0) > 0.0001 {
		t.Fatalf("serial cache share=%v, want %v", serial.CacheHitShare, 100.0/650.0)
	}

	batched := wire.Summary.ByPathStats[PathBatched]
	if batched.Count != 2 || batched.PrefillMeasured != 2 || batched.DecodeMeasured != 2 {
		t.Fatalf("batched count/measured=%d/%d/%d, want 2/2/2", batched.Count, batched.PrefillMeasured, batched.DecodeMeasured)
	}
	closeNativePathStat(t, "batched p50 prefill", batched.PrefillTPSP50, 128)
	closeNativePathStat(t, "batched weighted prefill", batched.PrefillTPSWeighted, 128)
	closeNativePathStat(t, "batched p50 decode", batched.DecodeTPSP50, 5)
	closeNativePathStat(t, "batched weighted decode", batched.DecodeTPSWeighted, 10)
	if math.Abs(batched.CacheHitShare-0.25) > 0.0001 {
		t.Fatalf("batched cache share=%v, want 0.25", batched.CacheHitShare)
	}

	speculative := wire.Summary.ByPathStats[PathSpeculative]
	if speculative.Count != 1 || speculative.PrefillMeasured != 1 || speculative.DecodeMeasured != 0 {
		t.Fatalf("speculative count/measured=%d/%d/%d, want 1/1/0", speculative.Count, speculative.PrefillMeasured, speculative.DecodeMeasured)
	}
	closeNativePathStat(t, "speculative p50 prefill", speculative.PrefillTPSP50, 400)
	closeNativePathStat(t, "speculative weighted prefill", speculative.PrefillTPSWeighted, 400)
	if speculative.DecodeTPSP50 != nil || speculative.DecodeTPSWeighted != nil {
		t.Fatalf("unknown speculative decode phase emitted rates: p50=%v weighted=%v", speculative.DecodeTPSP50, speculative.DecodeTPSWeighted)
	}
	if math.Abs(speculative.CacheHitShare-0.5) > 0.0001 {
		t.Fatalf("speculative cache share=%v, want 0.5", speculative.CacheHitShare)
	}

	proxy := BuildReport([]Record{{Schema: Schema, Locality: LocalityVendor, PromptTokens: 8, CompletionTokens: 2, NativeTiming: NewNativeTiming(8, 2, 1, 1)}}, 1, false, 0)
	if proxy.Summary.Native != 0 || proxy.Summary.ByPath != nil || proxy.Summary.ByPathStats != nil {
		t.Fatalf("proxy-only window exposed native path aggregates: %+v", proxy.Summary)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestNativePathSummaryRespectsBounded1024RecordWindow(t *testing.T) {
	recs := make([]Record, 0, RingCap+1)
	recs = append(recs, nativePathRecord(PathSpeculative, 128, 2, 0, 128, 2, 1000, 1000))
	for i := 0; i < RingCap; i++ {
		recs = append(recs, nativePathRecord(PathSerial, 128, 2, 0, 128, 2, 1000, 1000))
	}
	report := BuildReport(recs, RingCap+1, false, 0)
	if report.Window.Requests != RingCap || report.Window.RetainedCap != RingCap || len(report.Records) != RingCap {
		t.Fatalf("bounded window requests=%d retained_cap=%d records=%d, want %d", report.Window.Requests, report.Window.RetainedCap, len(report.Records), RingCap)
	}
	serial := report.Summary.ByPathStats[PathSerial]
	if serial.Count != RingCap || serial.PrefillMeasured != RingCap || serial.DecodeMeasured != RingCap {
		t.Fatalf("serial retained count/measured=%d/%d/%d, want %d", serial.Count, serial.PrefillMeasured, serial.DecodeMeasured, RingCap)
	}
	if _, ok := report.Summary.ByPathStats[PathSpeculative]; ok {
		t.Fatalf("evicted record influenced bounded summary: %v", report.Summary.ByPathStats)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNativePathSummarySeparatesMeasuredAndRateEligiblePhases(t *testing.T) {
	t.Parallel()
	recs := []Record{
		nativePathRecord(PathSerial, 127, 1, 0, 127, 1, 100000, 100000),
		nativePathRecord(PathSerial, 128, 2, 0, 128, 2, 1000, 1000),
		nativePathRecord(PathSerial, 256, 4, 0, 256, 4, 2000, 2000),
	}
	for i := range recs {
		recs[i].NativeTiming.PrefillTPS = 999999
		recs[i].NativeTiming.DecodeTPS = 999999
	}
	s := Summarize(recs)
	n := s.NativeTiming
	if n == nil || n.Count != 3 || n.PrefillMeasured != 3 || n.DecodeMeasured != 3 ||
		n.PrefillP50MS != 2000 || n.DecodeP50MS != 2000 {
		t.Fatalf("measured phases lost from coverage or latency: %+v", n)
	}
	if n.PrefillTPSP50 != 128 || n.DecodeTPSP50 != 1 ||
		n.PrefillTPSWeighted != 128 || n.DecodeTPSWeighted != 1.33 {
		t.Fatalf("rates include ineligible phases or persisted rates: %+v", n)
	}
	p := s.ByPathStats[PathSerial]
	if p.Count != 3 || p.PrefillMeasured != 3 || p.DecodeMeasured != 3 ||
		p.PrefillTPSP50 != 128 || p.DecodeTPSP50 != 1 ||
		p.PrefillTPSWeighted != 128 || p.DecodeTPSWeighted != 1.33 {
		t.Fatalf("path rates/coverage differ from eligible native phases: %+v", p)
	}
	// Rounding a valid per-record rate to zero does not remove its tokens or
	// duration from the weighted cohort.
	rounded := Summarize([]Record{
		nativePathRecord(PathSerial, 128, 2, 0, 128, 2, 30000000, 300000),
		nativePathRecord(PathSerial, 128, 2, 0, 128, 2, 1000, 1000),
	})
	if n := rounded.NativeTiming; n == nil || n.PrefillMeasured != 2 || n.DecodeMeasured != 2 ||
		n.PrefillTPSWeighted != 0.01 || n.DecodeTPSWeighted != 0.01 {
		t.Fatalf("rounded-zero eligible rates lost weighted contribution: %+v", n)
	}
	if p := rounded.ByPathStats[PathSerial]; p.PrefillTPSWeighted != 0.01 || p.DecodeTPSWeighted != 0.01 {
		t.Fatalf("path fold dropped rounded-zero eligible rates: %+v", p)
	}
	// A measured-only window keeps latency/coverage while omitting all rates,
	// even if the persisted row claims rates for an ineligible phase.
	only := Summarize(recs[:1])
	raw, err := json.Marshal(only.ByPathStats[PathSerial])
	if err != nil {
		t.Fatal(err)
	}
	var wire nativePathStatsWire
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Count != 1 || wire.PrefillMeasured != 1 || wire.DecodeMeasured != 1 ||
		wire.PrefillTPSP50 != nil || wire.DecodeTPSP50 != nil ||
		wire.PrefillTPSWeighted != nil || wire.DecodeTPSWeighted != nil {
		t.Fatalf("ineligible measured path invented rates: %s", raw)
	}
	if only.NativeTiming == nil || only.NativeTiming.PrefillP50MS != 100000 || only.NativeTiming.DecodeP50MS != 100000 {
		t.Fatalf("ineligible measured path lost latency: %+v", only.NativeTiming)
	}
}
