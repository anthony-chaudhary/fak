package codexlifecycle

// #10662 regression fixture tests. The committed fixture
// (testdata/posttool/issue-10662/session-a.jsonl) is one long rollout whose
// post-tool gaps grow monotonically with ordinal and context while tool
// execution stays flat, plus the four audited special shapes: a genuinely slow
// tool, a compaction inside a gap, a stall-length gap, and a live-tail result.
// conformance.json carries the exact expected values, derived from the timeline
// the fixture was generated from — independent of the implementation under test.

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// postToolFixtureDir is the committed #10662 regression fixture directory.
var postToolFixtureDir = filepath.Join("testdata", "posttool", "issue-10662")

// confSpan / confBucket mirror conformance.json exactly.
type confSpan struct {
	Ordinal        int    `json:"ordinal"`
	CallKind       string `json:"call_kind"`
	ToolMS         int64  `json:"tool_ms"`
	GapMS          int64  `json:"gap_ms"`
	NextRecordKind string `json:"next_record_kind"`
	Compactions    int    `json:"compactions"`
	InputTokens    int    `json:"input_tokens"`
	ContextBand    string `json:"context_band"`
	Attribution    string `json:"attribution"`
	Stall          bool   `json:"stall"`
	Tool           string `json:"tool"`
	ToolClass      string `json:"tool_class"`
}

type confBucket struct {
	Key     string  `json:"key"`
	N       int     `json:"n"`
	P50     float64 `json:"p50"`
	P90     float64 `json:"p90"`
	P95     float64 `json:"p95"`
	Over30s int     `json:"over30s"`
	ToolP50 float64 `json:"tool_p50"`
}

type confFile struct {
	Schema string `json:"schema"`
	Expect struct {
		Sessions        int `json:"sessions"`
		Spans           int `json:"spans"`
		TailSkipped     int `json:"tail_skipped"`
		StallSpans      int `json:"stall_spans"`
		CompactionInGap int `json:"compaction_in_gap"`
	} `json:"expect"`
	Spans    []confSpan   `json:"spans"`
	Bands    []confBucket `json:"bands"`
	Ordinals []confBucket `json:"ordinals"`
}

func eqF(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

func loadPostToolFixture(t *testing.T) (Meta, []ARecord, []PostToolSpan, PostToolReport, confFile) {
	t.Helper()
	fh, err := os.Open(filepath.Join(postToolFixtureDir, "session-a.jsonl"))
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer fh.Close()
	meta, records, err := ReadAnalyticsRollout(fh)
	if err != nil {
		t.Fatalf("ReadAnalyticsRollout: %v", err)
	}
	spans := AnalyzePostToolSpans(meta, records)
	rep, err := ScanPostToolCorpus(postToolFixtureDir, ScanOptions{})
	if err != nil {
		t.Fatalf("ScanPostToolCorpus: %v", err)
	}
	blob, err := os.ReadFile(filepath.Join(postToolFixtureDir, "conformance.json"))
	if err != nil {
		t.Fatalf("read conformance: %v", err)
	}
	var conf confFile
	if err := json.Unmarshal(blob, &conf); err != nil {
		t.Fatalf("decode conformance: %v", err)
	}
	return meta, records, spans, rep, conf
}

// The fixture reproduces the audited #10662 signature end to end and matches the
// conformance values exactly.
func TestPostToolFixture_Conformance(t *testing.T) {
	meta, records, spans, rep, conf := loadPostToolFixture(t)

	if meta.RolloutID != "posttool-fixture-0001" {
		t.Fatalf("rollout id = %q", meta.RolloutID)
	}
	// Ingestion captures the token_count usage the banding consumes.
	firstTokens := 0
	for _, r := range records {
		if r.Kind == kindTokens && r.InputTokens > 0 {
			firstTokens = r.InputTokens
			break
		}
	}
	if firstTokens != 5000 {
		t.Fatalf("first token_count input_tokens = %d, want 5000 (ingestion must parse payload.info.last_token_usage)", firstTokens)
	}

	if len(spans) != conf.Expect.Spans {
		t.Fatalf("spans = %d, want %d", len(spans), conf.Expect.Spans)
	}
	if rep.Sessions != conf.Expect.Sessions || rep.Spans != conf.Expect.Spans ||
		rep.TailSkipped != conf.Expect.TailSkipped || rep.StallSpans != conf.Expect.StallSpans ||
		rep.CompactionInGap != conf.Expect.CompactionInGap {
		t.Fatalf("report totals = sessions %d spans %d tail %d stall %d compaction %d, want %+v",
			rep.Sessions, rep.Spans, rep.TailSkipped, rep.StallSpans, rep.CompactionInGap, conf.Expect)
	}
	if rep.Gap.N != conf.Expect.Spans || rep.ToolMS.N != conf.Expect.Spans {
		t.Fatalf("overall percentiles n = gap %d tool %d, want %d", rep.Gap.N, rep.ToolMS.N, conf.Expect.Spans)
	}

	byOrdinal := map[int]PostToolSpan{}
	for _, s := range spans {
		byOrdinal[s.Ordinal] = s
	}
	for _, want := range conf.Spans {
		got, ok := byOrdinal[want.Ordinal]
		if !ok {
			t.Fatalf("conformance ordinal %d missing from spans", want.Ordinal)
		}
		if got.CallKind != want.CallKind || got.ToolMS != want.ToolMS || got.GapMS != want.GapMS ||
			got.NextRecordKind != want.NextRecordKind || got.Compactions != want.Compactions ||
			got.InputTokens != want.InputTokens || got.ContextBand != want.ContextBand ||
			got.Attribution != want.Attribution || got.Stall != want.Stall ||
			got.Tool != want.Tool || string(got.ToolClass) != want.ToolClass {
			t.Errorf("ordinal %d = %+v, want %+v", want.Ordinal, got, want)
		}
	}
	if s := byOrdinal[150]; len(s.Subspans) != 2 ||
		s.Subspans[0].Kind != "pre_compaction" || s.Subspans[0].MS != 5000 ||
		s.Subspans[1].Kind != "compaction" || s.Subspans[1].MS != 5000 {
		t.Errorf("compaction span subspans = %+v, want [pre_compaction 5000 compaction 5000]", s.Subspans)
	}

	if !eqBucketRows(rep.ByBand, conf.Bands) {
		t.Errorf("by_band = %+v, want %+v", rep.ByBand, conf.Bands)
	}
	if !eqBucketRows(rep.ByOrdinal, conf.Ordinals) {
		t.Errorf("by_ordinal = %+v, want %+v", rep.ByOrdinal, conf.Ordinals)
	}
}

func eqBucketRows(got []PostToolBucket, want []confBucket) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		g := got[i]
		w := want[i]
		if g.Key != w.Key || g.N != w.N || g.Over30s != w.Over30s ||
			!eqF(g.P50, w.P50) || !eqF(g.P90, w.P90) || !eqF(g.P95, w.P95) || !eqF(g.ToolP50, w.ToolP50) {
			return false
		}
	}
	return true
}

// THE NO-DOUBLE-COUNTING WITNESS: ToolMS measures call → its own output and
// GapMS measures that output → the next model-emitted record, so ToolMS + GapMS
// must equal the call → next-record interval for EVERY span — tool time and
// post-tool model time are disjoint by construction, never double-booked.
func TestPostToolSpans_NoDoubleCounting(t *testing.T) {
	_, records, spans, _, _ := loadPostToolFixture(t)

	// Re-derive the intervals straight from the fixture timestamps.
	callTS := map[string]time.Time{}
	type interval struct{ call, out, next time.Time }
	var intervals []interval
	for i, r := range records {
		if r.Kind == kindToolCall {
			callTS[r.CallID] = r.TS
		}
		if r.Kind != "function_call_output" {
			continue
		}
		for ni := i + 1; ni < len(records); ni++ {
			if modelEmitted(records[ni]) {
				intervals = append(intervals, interval{callTS[r.CallID], r.TS, records[ni].TS})
				break
			}
		}
	}
	if len(intervals) != len(spans) {
		t.Fatalf("derived %d completed intervals for %d spans", len(intervals), len(spans))
	}
	for i, s := range spans {
		want := intervals[i].next.Sub(intervals[i].call).Milliseconds()
		if s.ToolMS+s.GapMS != want {
			t.Errorf("span ordinal %d: tool %d + gap %d != call→next interval %d (double counting)",
				s.Ordinal, s.ToolMS, s.GapMS, want)
		}
		var sub int64
		for _, ss := range s.Subspans {
			sub += ss.MS
		}
		if sub != 0 && sub != s.GapMS {
			t.Errorf("span ordinal %d: subspans sum %d != gap %d (subspans must tile the gap)", s.Ordinal, sub, s.GapMS)
		}
	}
}

// The report DISTINGUISHES tool slowness from post-tool model latency: gap
// medians grow with ordinal and context while tool execution stays flat.
func TestPostToolReport_GrowthSignature(t *testing.T) {
	_, _, _, rep, _ := loadPostToolFixture(t)

	bucket := func(rows []PostToolBucket, key string) PostToolBucket {
		for _, b := range rows {
			if b.Key == key {
				return b
			}
		}
		t.Fatalf("bucket %q missing", key)
		return PostToolBucket{}
	}
	early, late := bucket(rep.ByOrdinal, OrdBucket1_20), bucket(rep.ByOrdinal, OrdBucketGTE201)
	if late.P50 <= early.P50 {
		t.Errorf("gte201 p50 %.2f must exceed 1_20 p50 %.2f (the audited growth signature)", late.P50, early.P50)
	}
	if late.P50 <= 55 {
		t.Errorf("gte201 p50 = %.2f, want > 55s of post-tool model latency", late.P50)
	}
	if early.ToolP50 >= 1 || late.ToolP50 >= 1 {
		t.Errorf("tool execution must stay sub-second while gaps grow: 1_20 tool_p50 %.2f gte201 tool_p50 %.2f",
			early.ToolP50, late.ToolP50)
	}
	lowBand, hotBand := bucket(rep.ByBand, BandLT10K), bucket(rep.ByBand, Band50K100K)
	if hotBand.P50 <= lowBand.P50 {
		t.Errorf("band 50k_100k p50 %.2f must exceed lt10k p50 %.2f (context-size hotspot)", hotBand.P50, lowBand.P50)
	}
}

// Scrubbing: the report and every span carry ids, numbers, and closed tokens
// only — never the fixture's command text, result body text, or cwd.
func TestPostToolReport_Scrubbing(t *testing.T) {
	_, _, spans, rep, _ := loadPostToolFixture(t)
	forbidden := []string{
		"posttool-fixture-cmd-10662", // command text
		"posttool-fixture-out-10662", // result body text
		"/tmp/fak-posttool-fixture",  // session cwd
	}
	blob, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range spans {
		sb, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		blob = append(blob, sb...)
	}
	for _, f := range forbidden {
		if bytes.Contains(blob, []byte(f)) {
			t.Errorf("scrubbing violation: %q appears in exported JSON", f)
		}
	}
}

// Closed-vocabulary boundaries: band edges and ordinal bucket edges.
func TestPostToolBoundaries(t *testing.T) {
	for _, tc := range []struct {
		tokens int
		want   string
	}{
		{0, BandUnobserved}, {-5, BandUnobserved},
		{1, BandLT10K}, {9999, BandLT10K},
		{10000, Band10K25K}, {24999, Band10K25K},
		{25000, Band25K50K}, {49999, Band25K50K},
		{50000, Band50K100K}, {99999, Band50K100K},
		{100000, BandGTE100K}, {250000, BandGTE100K},
	} {
		if got := contextBand(tc.tokens); got != tc.want {
			t.Errorf("contextBand(%d) = %s, want %s", tc.tokens, got, tc.want)
		}
	}
	for _, tc := range []struct {
		ordinal int
		want    string
	}{
		{1, OrdBucket1_20}, {20, OrdBucket1_20},
		{21, OrdBucket21_50}, {50, OrdBucket21_50},
		{51, OrdBucket51_100}, {100, OrdBucket51_100},
		{101, OrdBucket101_200}, {200, OrdBucket101_200},
		{201, OrdBucketGTE201}, {999, OrdBucketGTE201},
	} {
		if got := postToolOrdinalBucket(tc.ordinal); got != tc.want {
			t.Errorf("postToolOrdinalBucket(%d) = %s, want %s", tc.ordinal, got, tc.want)
		}
	}
}

// Unobserved tokens band as unobserved: a span whose gap ends after the last
// token_count carries InputTokens 0 (the fixture's final span proves the same).
func TestAnalyzePostToolSpans_UnobservedTokens(t *testing.T) {
	meta, recs, err := ReadAnalyticsRollout(strings.NewReader(strings.Join([]string{
		meta("s", "fak", "0.44.0", `C:\work\fak`),
		started("2026-09-01T10:00:00.000Z", "A"),
		callLine("2026-09-01T10:00:01.000Z", "c1", "shell_command", "echo hi"),
		outLine("2026-09-01T10:00:02.000Z", "c1", 0, 0.5, "hi"),
		complete("2026-09-01T10:00:10.000Z", "A"),
	}, "\n") + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	spans := AnalyzePostToolSpans(meta, recs)
	if len(spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(spans))
	}
	s := spans[0]
	if s.InputTokens != 0 || s.ContextBand != BandUnobserved {
		t.Errorf("input_tokens/band = %d/%s, want 0/unobserved", s.InputTokens, s.ContextBand)
	}
	if s.NextRecordKind != "task_complete" {
		t.Errorf("next_record_kind = %s, want task_complete", s.NextRecordKind)
	}
}

// custom_tool_call records fold into the shared tool-call kind at ingestion;
// PayloadKind must recover them so span kinds stay honest without re-parsing.
func TestAnalyzePostToolSpans_CustomToolCalls(t *testing.T) {
	customCall := func(ts, id, tool string) string {
		return `{"timestamp":"` + ts + `","type":"response_item","payload":{"type":"custom_tool_call","name":"` + tool + `","call_id":"` + id + `","input":"do"}}`
	}
	customOut := func(ts, id string) string {
		return `{"timestamp":"` + ts + `","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"` + id + `","output":"done"}}`
	}
	meta, recs, err := ReadAnalyticsRollout(strings.NewReader(strings.Join([]string{
		meta("s", "fak", "0.44.0", `C:\work\fak`),
		started("2026-09-01T10:00:00.000Z", "A"),
		customCall("2026-09-01T10:00:01.000Z", "k1", "browser"),
		customOut("2026-09-01T10:00:03.000Z", "k1"),
		customCall("2026-09-01T10:00:06.000Z", "k2", "browser"),
		customOut("2026-09-01T10:00:07.000Z", "k2"),
		tokensLine("2026-09-01T10:00:20.000Z"),
		complete("2026-09-01T10:00:21.000Z", "A"),
	}, "\n") + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	spans := AnalyzePostToolSpans(meta, recs)
	if len(spans) != 2 {
		t.Fatalf("spans = %d, want 2 (custom outputs are first-class results)", len(spans))
	}
	if spans[0].Ordinal != 1 || spans[1].Ordinal != 2 {
		t.Errorf("ordinals = %d/%d, want 1/2", spans[0].Ordinal, spans[1].Ordinal)
	}
	if spans[0].CallKind != "custom_tool_call" || spans[1].CallKind != "custom_tool_call" {
		t.Errorf("call kinds = %s/%s, want custom_tool_call twice", spans[0].CallKind, spans[1].CallKind)
	}
	if spans[0].NextRecordKind != "custom_tool_call" {
		t.Errorf("next_record_kind = %s, want custom_tool_call (the next model-emitted record)", spans[0].NextRecordKind)
	}
	if spans[0].ToolMS != 2000 || spans[0].GapMS != 3000 {
		t.Errorf("span 1 tool/gap = %d/%d, want 2000/3000", spans[0].ToolMS, spans[0].GapMS)
	}
	if spans[1].GapMS != 13000 {
		t.Errorf("span 2 gap = %d, want 13000 (output → token_count)", spans[1].GapMS)
	}
}
