package perfledger

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func jsonKeys(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

// fak-test:runtime fast est=10ms lane=default
func TestNewRecordDerivesRatesWhenTTFTMeasured(t *testing.T) {
	now := time.UnixMilli(1_700_000_000_000)
	r := NewRecord(now, "stop", LocalitySelfHosted, 1000, 201, 500, 5*time.Second, time.Second)
	if r.Schema != Schema || r.UnixMS != 1_700_000_000_000 {
		t.Fatalf("schema/unix_ms = %q/%d", r.Schema, r.UnixMS)
	}
	if r.PromptTokens != 1000 || r.CompletionTokens != 201 || r.CachedTokens != 500 {
		t.Fatalf("tokens = %d/%d/%d", r.PromptTokens, r.CompletionTokens, r.CachedTokens)
	}
	if r.E2EMS != 5000 || r.TTFTMS != 1000 {
		t.Fatalf("e2e/ttft = %v/%v, want 5000/1000", r.E2EMS, r.TTFTMS)
	}
	if r.PrefillTPS != 1000 {
		t.Fatalf("prefill_tps = %v, want 1000 (1000 tok / 1s)", r.PrefillTPS)
	}
	if r.DecodeTPS != 50 {
		t.Fatalf("decode_tps = %v, want 50 (200 tokens after the first / 4s)", r.DecodeTPS)
	}
	m := jsonKeys(t, r)
	for _, k := range []string{"schema", "unix_ms", "finish_reason", "locality", "prompt_tokens", "completion_tokens", "cached_tokens", "e2e_ms", "ttft_ms", "prefill_tps", "decode_tps"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("measured record JSON missing key %q: %v", k, m)
		}
	}
	if m["locality"] != LocalitySelfHosted || m["finish_reason"] != "stop" {
		t.Fatalf("locality/finish_reason = %v/%v", m["locality"], m["finish_reason"])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewRecordOmitsOptionalFieldsWhenTTFTUnmeasured(t *testing.T) {
	r := NewRecord(time.Now(), "stop", LocalityVendor, 100, 10, 0, 2*time.Second, 0)
	if r.TTFTMS != 0 || r.PrefillTPS != 0 || r.DecodeTPS != 0 {
		t.Fatalf("unmeasured ttft produced rates: %+v", r)
	}
	if r.E2EMS != 2000 {
		t.Fatalf("e2e_ms = %v, want 2000", r.E2EMS)
	}
	m := jsonKeys(t, r)
	for _, k := range []string{"ttft_ms", "prefill_tps", "decode_tps"} {
		if _, ok := m[k]; ok {
			t.Fatalf("unmeasured record JSON carries %q: %v", k, m)
		}
	}
	for _, k := range []string{"prompt_tokens", "completion_tokens", "cached_tokens", "e2e_ms"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("unmeasured record JSON missing %q: %v", k, m)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewRecordClampsTTFTToDuration(t *testing.T) {
	r := NewRecord(time.Now(), "stop", "", 10, 5, 0, time.Second, 3*time.Second)
	if r.TTFTMS != 1000 {
		t.Fatalf("ttft_ms = %v, want clamped to e2e 1000", r.TTFTMS)
	}
	if r.DecodeTPS != 0 {
		t.Fatalf("decode_tps = %v, want 0 when decode window is empty", r.DecodeTPS)
	}
}

// Halo witness (strix2, llama-server behind fak serve): warm hits prefilled 4-53
// uncached tokens in a 140-660ms TTFT floor and logged 9-100 "tok/s" next to a
// real ~290 tok/s cold prefill, dragging prefill p50 to a third of the truth.
//
// fak-test:runtime fast est=10ms lane=default
func TestNewRecordSkipsPrefillRateBelowMinTokens(t *testing.T) {
	warm := NewRecord(time.Now(), "stop", "", 4, 41, 3209, 5*time.Second, 140*time.Millisecond)
	if warm.TTFTMS != 140 {
		t.Fatalf("ttft_ms = %v, want 140", warm.TTFTMS)
	}
	if warm.PrefillTPS != 0 {
		t.Fatalf("prefill_tps = %v for a 4-token prefill, want omitted", warm.PrefillTPS)
	}
	if _, ok := jsonKeys(t, warm)["prefill_tps"]; ok {
		t.Fatalf("warm record JSON carries prefill_tps")
	}
	cold := NewRecord(time.Now(), "stop", "", MinPrefillRateTokens, 2, 0, 2*time.Second, time.Second)
	if cold.PrefillTPS != MinPrefillRateTokens {
		t.Fatalf("prefill_tps = %v at the floor, want %d", cold.PrefillTPS, MinPrefillRateTokens)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewRecordDecodeRateExcludesFirstToken(t *testing.T) {
	r := NewRecord(time.Now(), "stop", "", 200, 5, 0, 2*time.Second, time.Second)
	if r.DecodeTPS != 4 {
		t.Fatalf("decode_tps = %v, want 4 (4 tokens after the first / 1s)", r.DecodeTPS)
	}
	one := NewRecord(time.Now(), "stop", "", 200, 1, 0, 2*time.Second, time.Second)
	if one.DecodeTPS != 0 {
		t.Fatalf("decode_tps = %v for a single-token completion, want omitted", one.DecodeTPS)
	}
}

func fixedRecords() []Record {
	return []Record{
		{Schema: Schema, TTFTMS: 100, PrefillTPS: 10, DecodeTPS: 1, E2EMS: 1000, PromptTokens: 200, CachedTokens: 0},
		{Schema: Schema, TTFTMS: 200, PrefillTPS: 20, DecodeTPS: 2, E2EMS: 2000, PromptTokens: 200, CachedTokens: 100},
		{Schema: Schema, TTFTMS: 300, PrefillTPS: 30, DecodeTPS: 3, E2EMS: 3000, PromptTokens: 200, CachedTokens: 100},
		{Schema: Schema, TTFTMS: 400, PrefillTPS: 40, DecodeTPS: 4, E2EMS: 4000, PromptTokens: 200, CachedTokens: 100},
		{Schema: Schema, E2EMS: 5000, PromptTokens: 0, CachedTokens: 0},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeNearestRankOnFixedSet(t *testing.T) {
	s := Summarize(fixedRecords())
	if s.Count != 5 || s.TTFTMeasured != 4 {
		t.Fatalf("count/ttft_measured = %d/%d, want 5/4", s.Count, s.TTFTMeasured)
	}
	// nearest-rank over [100,200,300,400]: p50 -> rank 2, p99 -> rank 4.
	if s.TTFTP50MS != 200 || s.TTFTP99MS != 400 {
		t.Fatalf("ttft p50/p99 = %v/%v, want 200/400", s.TTFTP50MS, s.TTFTP99MS)
	}
	if s.PrefillTPSP50 != 20 || s.DecodeTPSP50 != 2 {
		t.Fatalf("prefill/decode p50 = %v/%v, want 20/2", s.PrefillTPSP50, s.DecodeTPSP50)
	}
	// 4 x 200 uncached tokens over 100+200+300+400 ms of TTFT.
	if s.PrefillTPSWeighted != 800 {
		t.Fatalf("prefill_tps_weighted = %v, want 800", s.PrefillTPSWeighted)
	}
	// nearest-rank over [1000..5000]: p50 -> rank 3, p99 -> rank 5.
	if s.E2EP50MS != 3000 || s.E2EP99MS != 5000 {
		t.Fatalf("e2e p50/p99 = %v/%v, want 3000/5000", s.E2EP50MS, s.E2EP99MS)
	}
	// cached 300 / (uncached 800 + cached 300).
	if s.CacheHitShare != 0.2727 {
		t.Fatalf("cache_hit_share = %v, want 0.2727", s.CacheHitShare)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeOmitsUnmeasuredAxes(t *testing.T) {
	s := Summarize([]Record{{Schema: Schema, E2EMS: 10, PromptTokens: 5}})
	m := jsonKeys(t, s)
	for _, k := range []string{"ttft_p50_ms", "ttft_p99_ms", "prefill_tps_p50", "decode_tps_p50"} {
		if _, ok := m[k]; ok {
			t.Fatalf("summary JSON carries unmeasured axis %q: %v", k, m)
		}
	}
	for _, k := range []string{"count", "ttft_measured", "e2e_p50_ms", "cache_hit_share"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("summary JSON missing %q: %v", k, m)
		}
	}
	if s.CacheHitShare != 0 {
		t.Fatalf("cache_hit_share = %v, want 0", s.CacheHitShare)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBuildReportWindowAndCompact(t *testing.T) {
	rep := BuildReport(fixedRecords(), 3, true, 7)
	if rep.Schema != ReportSchema {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if rep.Window.Requests != 3 || rep.Window.RetainedCap != RingCap || !rep.Window.Capped || rep.Window.DroppedWrites != 7 {
		t.Fatalf("window = %+v", rep.Window)
	}
	if len(rep.Records) != 3 || rep.Records[0].TTFTMS != 300 || rep.Records[2].E2EMS != 5000 {
		t.Fatalf("records not the newest 3 oldest-first: %+v", rep.Records)
	}
	line := RenderCompact(rep)
	if !strings.HasPrefix(line, "PERF n=3") {
		t.Fatalf("compact = %q, want prefix PERF n=3", line)
	}
	if strings.Contains(line, "\n") {
		t.Fatalf("compact is not one line: %q", line)
	}
	m := jsonKeys(t, rep)
	for _, k := range []string{"schema", "window", "summary", "records"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("report JSON missing %q", k)
		}
	}
	w := m["window"].(map[string]any)
	for _, k := range []string{"requests", "retained_cap", "capped", "dropped_writes"} {
		if _, ok := w[k]; !ok {
			t.Fatalf("window JSON missing %q", k)
		}
	}
	if got := BuildReport(nil, 0, false, 0); got.Window.Requests != 0 || !strings.HasPrefix(RenderCompact(got), "PERF n=0") {
		t.Fatalf("empty report = %+v", got)
	}
}

func writeLines(t *testing.T, path string, lines []string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func recLine(t *testing.T, unixMS int64) string {
	t.Helper()
	b, err := json.Marshal(Record{Schema: Schema, UnixMS: unixMS, E2EMS: 1})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// fak-test:runtime fast est=10ms lane=default
func TestReadTailMissingFile(t *testing.T) {
	recs, truncated := ReadTail(filepath.Join(t.TempDir(), "absent.jsonl"))
	if recs != nil || truncated {
		t.Fatalf("missing file = %v/%v, want nil/false", recs, truncated)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestReadTailSkipsMalformedAndKeepsOrder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "perf.jsonl")
	writeLines(t, path, []string{
		recLine(t, 1),
		"{not json",
		`{"schema":"some.other.v1","unix_ms":99}`,
		recLine(t, 2),
		recLine(t, 3),
	})
	recs, truncated := ReadTail(path)
	if truncated {
		t.Fatal("small ledger reported truncated")
	}
	if len(recs) != 3 || recs[0].UnixMS != 1 || recs[1].UnixMS != 2 || recs[2].UnixMS != 3 {
		t.Fatalf("recs = %+v, want unix_ms 1,2,3", recs)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestReadTailBoundedKeepsNewestOldestFirst(t *testing.T) {
	path := filepath.Join(t.TempDir(), "perf.jsonl")
	total := RingCap + 10
	lines := make([]string, 0, total+1)
	for i := 0; i < total; i++ {
		lines = append(lines, recLine(t, int64(i)))
		if i == total-5 {
			lines = append(lines, "garbage")
		}
	}
	writeLines(t, path, lines)
	recs, truncated := ReadTail(path)
	if !truncated {
		t.Fatal("over-cap ledger not reported truncated")
	}
	if len(recs) != RingCap {
		t.Fatalf("len = %d, want %d", len(recs), RingCap)
	}
	if recs[0].UnixMS != int64(total-RingCap) || recs[len(recs)-1].UnixMS != int64(total-1) {
		t.Fatalf("window = [%d..%d], want [%d..%d]", recs[0].UnixMS, recs[len(recs)-1].UnixMS, total-RingCap, total-1)
	}
	for i := 1; i < len(recs); i++ {
		if recs[i].UnixMS <= recs[i-1].UnixMS {
			t.Fatalf("not oldest-first at %d: %d after %d", i, recs[i].UnixMS, recs[i-1].UnixMS)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestSummarizeSplitsTTFTByCacheRegime(t *testing.T) {
	now := time.UnixMilli(1_700_000_000_000)
	recs := []Record{
		NewRecord(now, "stop", LocalitySelfHosted, 100, 10, 0, 200*time.Millisecond, 100*time.Millisecond),
		NewRecord(now, "stop", LocalitySelfHosted, 500, 10, 500, 150*time.Millisecond, 40*time.Millisecond),
		NewRecord(now, "stop", LocalitySelfHosted, 0, 10, 100, 60*time.Millisecond, 2*time.Millisecond),
		NewRecord(now, "stop", LocalitySelfHosted, 0, 10, 0, 50*time.Millisecond, 0),
	}
	if recs[0].CacheRegime != "cold" || recs[1].CacheRegime != "partial" || recs[2].CacheRegime != "frozen" || recs[3].CacheRegime != "unknown" {
		t.Fatalf("regimes = %q/%q/%q/%q", recs[0].CacheRegime, recs[1].CacheRegime, recs[2].CacheRegime, recs[3].CacheRegime)
	}
	legacy := recs[0]
	legacy.CacheRegime = ""
	s := Summarize([]Record{legacy, recs[1], recs[2], recs[3]})
	want := map[string]RegimeSummary{
		"cold":    {Count: 1, TTFTMeasured: 1, TTFTP50MS: 100, TTFTP99MS: 100, E2EP50MS: 200},
		"partial": {Count: 1, TTFTMeasured: 1, TTFTP50MS: 40, TTFTP99MS: 40, E2EP50MS: 150},
		"frozen":  {Count: 1, TTFTMeasured: 1, TTFTP50MS: 2, TTFTP99MS: 2, E2EP50MS: 60},
		"unknown": {Count: 1, E2EP50MS: 50},
	}
	if len(s.ByRegime) != len(want) {
		t.Fatalf("by_regime = %+v", s.ByRegime)
	}
	for k, w := range want {
		if s.ByRegime[k] != w {
			t.Fatalf("by_regime[%s] = %+v, want %+v", k, s.ByRegime[k], w)
		}
	}
	line := RenderCompact(BuildReport([]Record{legacy, recs[1], recs[2]}, 0, false, 0))
	if !strings.Contains(line, "ttft p50 by regime: frozen=2ms(n=1) partial=40ms(n=1) cold=100ms(n=1)") {
		t.Fatalf("compact line missing regime split: %s", line)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFailureRecordCountsInSummaryWithoutRates(t *testing.T) {
	now := time.UnixMilli(1_700_000_000_000)
	recs := []Record{
		NewRecord(now, "stop", LocalitySelfHosted, 500, 11, 0, 2*time.Second, time.Second),
		NewFailureRecord(now, LocalitySelfHosted, ErrorFirstTokenTimeout, 504, 60*time.Second, 0),
		NewFailureRecord(now, LocalitySelfHosted, ErrorStall, 200, 9*time.Second, 400*time.Millisecond),
	}
	f := recs[1]
	if f.FinishReason != FinishReasonError || f.Error != ErrorFirstTokenTimeout || f.Status != 504 || f.CacheRegime != "unknown" {
		t.Fatalf("failure row = %+v", f)
	}
	if f.E2EMS != 60000 || f.TTFTMS != 0 || f.PrefillTPS != 0 || f.DecodeTPS != 0 {
		t.Fatalf("failure row timings = %+v", f)
	}
	if recs[2].TTFTMS != 400 {
		t.Fatalf("mid-stream failure ttft = %v, want 400", recs[2].TTFTMS)
	}
	s := Summarize(recs)
	if s.Errors != 2 || s.ByError[ErrorFirstTokenTimeout] != 1 || s.ByError[ErrorStall] != 1 {
		t.Fatalf("errors = %d %v", s.Errors, s.ByError)
	}
	if s.E2EP99MS != 60000 {
		t.Fatalf("e2e p99 = %v, want the timed-out turn's 60000", s.E2EP99MS)
	}
	if s.DecodeTPSP50 != 10 {
		t.Fatalf("decode p50 = %v, want only the served row's 10", s.DecodeTPSP50)
	}
	line := RenderCompact(BuildReport(recs, 0, false, 0))
	if !strings.Contains(line, "errors=2 first_token_timeout=1 stall=1") {
		t.Fatalf("compact = %q", line)
	}
	if _, ok := jsonKeys(t, recs[0])["error"]; ok {
		t.Fatalf("served row JSON carries error")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeFoldsNativeEngineAnatomy(t *testing.T) {
	now := time.UnixMilli(1_700_000_000_000)
	proxy := NewRecord(now, "stop", LocalityVendor, 100, 10, 0, 200*time.Millisecond, 50*time.Millisecond)
	serial := NewRecord(now, "stop", LocalitySelfHosted, 100, 10, 0, 200*time.Millisecond, 50*time.Millisecond)
	serial.Model, serial.Engine = "m", &Engine{Path: PathSerial}
	batched := serial
	batched.Engine = &Engine{Path: PathBatched, CohortSize: 4}
	specA := serial
	specA.Engine = &Engine{Path: PathSpeculative, SpecRounds: 3, SpecDraftTokens: 9, SpecAcceptedTokens: 6}
	specB := serial
	specB.Engine = &Engine{Path: PathSpeculative, SpecRounds: 1, SpecDraftTokens: 3, SpecAcceptedTokens: 3}

	s := Summarize([]Record{proxy, serial, batched, specA, specB})
	if s.Native != 4 || s.ByPath[PathSerial] != 1 || s.ByPath[PathBatched] != 1 || s.ByPath[PathSpeculative] != 2 {
		t.Fatalf("native/by_path = %d/%v, want 4 split 1/1/2 (the proxy row is not native)", s.Native, s.ByPath)
	}
	if s.SpecRounds != 4 || s.SpecDraftTokens != 12 || s.SpecAcceptedTokens != 9 || s.SpecAcceptRate != 0.75 {
		t.Fatalf("spec = %d/%d/%d rate %v, want 4/12/9 rate 0.75", s.SpecRounds, s.SpecDraftTokens, s.SpecAcceptedTokens, s.SpecAcceptRate)
	}
	line := RenderCompact(BuildReport([]Record{proxy, serial, batched, specA, specB}, 0, false, 0))
	if !strings.Contains(line, "| path: serial=1 batched=1 speculative=2") || !strings.Contains(line, "| spec accept=75.0% (9/12 over 4 rounds)") {
		t.Fatalf("compact line missing native anatomy: %s", line)
	}

	proxyOnly := Summarize([]Record{proxy})
	keys := jsonKeys(t, proxyOnly)
	for _, k := range []string{"native", "by_path", "spec_rounds", "spec_accept_rate"} {
		if _, ok := keys[k]; ok {
			t.Fatalf("proxy-only summary carries %q; native fields must be absent, not 0", k)
		}
	}
	if line := RenderCompact(BuildReport([]Record{proxy}, 0, false, 0)); strings.Contains(line, "path:") || strings.Contains(line, "spec") {
		t.Fatalf("proxy-only compact line carries native segments: %s", line)
	}
	if _, ok := jsonKeys(t, proxy)["engine"]; ok {
		t.Fatal("proxy row serializes an engine object")
	}
}

// A cache hit re-feeds a few tokens in fixed overhead; its "rate" is not prefill
// and must not lower either prefill figure (live strix2: 4 tok/149ms read 27 tok/s).
// fak-test:runtime fast est=5ms lane=default
func TestCacheHitRowCannotLowerPrefill(t *testing.T) {
	now := time.UnixMilli(1_700_000_000_000)
	cold := NewRecord(now, "stop", LocalitySelfHosted, 3537, 33, 0, 13278*time.Millisecond, 11490*time.Millisecond)
	hit := NewRecord(now, "stop", LocalitySelfHosted, 4, 33, 3533, 1944*time.Millisecond, 149*time.Millisecond)
	if _, ok := jsonKeys(t, hit)["prefill_tps"]; ok {
		t.Fatalf("cache-hit row carries prefill_tps: %+v", hit)
	}
	base := Summarize([]Record{cold})
	for _, recs := range [][]Record{
		{cold, hit},
		{cold, hit, hit, hit},
		// A row written before the floor existed still carries its tiny-prompt rate.
		{cold, {Schema: Schema, PromptTokens: 4, CachedTokens: 3533, TTFTMS: 149, PrefillTPS: 26.84, E2EMS: 1944}},
	} {
		s := Summarize(recs)
		if s.PrefillTPSP50 < base.PrefillTPSP50 || s.PrefillTPSWeighted < base.PrefillTPSWeighted {
			t.Fatalf("cache hits lowered prefill: p50 %v->%v weighted %v->%v", base.PrefillTPSP50, s.PrefillTPSP50, base.PrefillTPSWeighted, s.PrefillTPSWeighted)
		}
	}
	if base.PrefillTPSWeighted < 300 {
		t.Fatalf("cold prefill_tps_weighted = %v, want ~308", base.PrefillTPSWeighted)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestSummarizeFoldsUpstreamSpecIntoAcceptRate(t *testing.T) {
	native := Record{Schema: Schema, E2EMS: 10, Engine: &Engine{Path: PathSpeculative, SpecRounds: 5, SpecDraftTokens: 20, SpecAcceptedTokens: 15}}
	upstream := Record{Schema: Schema, E2EMS: 10, UpstreamSpecDraftTokens: 30, UpstreamSpecAcceptedTokens: 20}
	plain := Record{Schema: Schema, E2EMS: 10}
	for _, k := range []string{"upstream_spec_draft_tokens", "upstream_spec_accepted_tokens"} {
		if _, ok := jsonKeys(t, upstream)[k]; !ok {
			t.Fatalf("upstream record JSON missing %q", k)
		}
		if _, ok := jsonKeys(t, plain)[k]; ok {
			t.Fatalf("plain record JSON carries %q", k)
		}
	}
	s := Summarize([]Record{native, upstream, plain})
	if s.SpecDraftTokens != 50 || s.SpecAcceptedTokens != 35 || s.SpecAcceptRate != 0.7 || s.SpecRounds != 5 {
		t.Fatalf("spec = %d/%d rate %v rounds %d, want 35/50 0.7 5", s.SpecAcceptedTokens, s.SpecDraftTokens, s.SpecAcceptRate, s.SpecRounds)
	}
	if s.Native != 1 {
		t.Fatalf("native = %d, want 1 (an upstream draft is not a native row)", s.Native)
	}
	if line := RenderCompact(BuildReport([]Record{upstream}, 0, false, 0)); !strings.Contains(line, "spec accept=66.7% (20/30)") {
		t.Fatalf("compact line for an upstream-only window: %s", line)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeCountsServedByAndKeepsNewest(t *testing.T) {
	old := Identity{Planner: "inkernel", Backend: "cpu-ref", Host: "h1", Version: "1.0.0"}
	cur := Identity{Planner: "inkernel", Backend: "vulkan", Host: "h1", Version: "1.1.0"}
	recs := []Record{
		{Schema: Schema, E2EMS: 10},
		{Schema: Schema, E2EMS: 10, Model: "m", Identity: old},
		{Schema: Schema, E2EMS: 10, Model: "m", Identity: cur},
		{Schema: Schema, E2EMS: 10, Model: "m", Identity: cur},
	}
	s := Summarize(recs)
	if s.Identities != 2 {
		t.Fatalf("identities = %d, want 2 (unstamped row not counted)", s.Identities)
	}
	if want := (ServedBy{Model: "m", Identity: cur}); s.ServedBy == nil || *s.ServedBy != want {
		t.Fatalf("newest served_by = %+v, want %+v", s.ServedBy, want)
	}
	line := RenderCompact(BuildReport(recs, 0, false, 0))
	for _, want := range []string{"planner=inkernel/vulkan", "fak=1.1.0", "window mixes 2"} {
		if !strings.Contains(line, want) {
			t.Fatalf("compact line missing %q: %s", want, line)
		}
	}
	if s := Summarize(recs[:1]); s.ServedBy != nil || s.Identities != 0 {
		t.Fatalf("unstamped window served_by = %+v/%d, want nil/0", s.ServedBy, s.Identities)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRecordIdentityRoundTripsFlat(t *testing.T) {
	rec := Record{Schema: Schema, Model: "m", Engine: &Engine{Path: PathSerial}, Identity: Identity{Planner: "proxy", Host: "h", Version: "v"}}
	raw, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	var flat map[string]any
	if err := json.Unmarshal(raw, &flat); err != nil {
		t.Fatal(err)
	}
	if flat["model"] != "m" || flat["planner"] != "proxy" || flat["fak_version"] != "v" {
		t.Fatalf("identity not flattened into the row: %s", raw)
	}
	if _, ok := flat["engine"].(map[string]any); !ok {
		t.Fatalf("engine anatomy object lost beside identity: %s", raw)
	}
	if _, ok := flat["backend"]; ok {
		t.Fatalf("empty backend serialized: %s", raw)
	}
	var back Record
	if err := json.Unmarshal(raw, &back); err != nil || back.Identity != rec.Identity || back.Model != "m" {
		t.Fatalf("round trip = %+v (%v)", back, err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestSummarizeSplitsByCacheTierAndRestore(t *testing.T) {
	now := time.UnixMilli(1_700_000_000_000)
	row := func(prompt, cached int, ttft time.Duration, tier, restore string) Record {
		r := NewRecord(now, "stop", LocalitySelfHosted, prompt, 10, cached, 500*time.Millisecond, ttft)
		r.CacheTier, r.CacheRestore = tier, restore
		return r
	}
	cases := []struct {
		name         string
		recs         []Record
		wantTier     map[string]TierSummary
		wantRestore  map[string]int
		wantUnserved int
		wantCompact  []string
		wantNoTier   bool
	}{
		{
			name: "every tier plus a proxied row",
			recs: []Record{
				row(100, 300, 20*time.Millisecond, TierDeviceL1, RestoreHit),
				row(100, 300, 40*time.Millisecond, TierDeviceL1, RestoreHit),
				row(200, 200, 80*time.Millisecond, TierHostL2, RestoreHit),
				row(300, 100, 150*time.Millisecond, TierRemoteL3, RestoreHit),
				row(400, 0, 200*time.Millisecond, TierNone, RestoreUnserved),
				row(400, 0, 0, TierNone, RestoreMiss),
				NewRecord(now, "stop", LocalityVendor, 200, 10, 300, 500*time.Millisecond, 0),
			},
			// window prompt = 1700 uncached + 1200 cached = 2900
			wantTier: map[string]TierSummary{
				TierDeviceL1: {Count: 2, CachedTokens: 600, PromptServedShare: 0.2069, TTFTMeasured: 2, TTFTP50MS: 20},
				TierHostL2:   {Count: 1, CachedTokens: 200, PromptServedShare: 0.069, TTFTMeasured: 1, TTFTP50MS: 80},
				TierRemoteL3: {Count: 1, CachedTokens: 100, PromptServedShare: 0.0345, TTFTMeasured: 1, TTFTP50MS: 150},
				TierNone:     {Count: 2, TTFTMeasured: 1, TTFTP50MS: 200},
				TierUnknown:  {Count: 1, CachedTokens: 300, PromptServedShare: 0.1034},
			},
			wantRestore:  map[string]int{RestoreHit: 4, RestoreMiss: 1, RestoreUnserved: 1},
			wantUnserved: 1,
			wantCompact: []string{
				"tier served/ttft p50: device_l1=20.7%/20ms(n=2) host_dram_l2=6.9%/80ms(n=1) remote_http_l3=3.5%/150ms(n=1) none=0.0%/200ms(n=2) unknown=10.3%/n/a(n=1)",
				"restore hit=4 miss=1 unserved=1",
			},
		},
		{
			name: "proxy-only window stays unknown and renders no tier segment",
			recs: []Record{
				NewRecord(now, "stop", LocalityVendor, 100, 10, 100, 500*time.Millisecond, 50*time.Millisecond),
			},
			wantTier:   map[string]TierSummary{TierUnknown: {Count: 1, CachedTokens: 100, PromptServedShare: 0.5, TTFTMeasured: 1, TTFTP50MS: 50}},
			wantNoTier: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := Summarize(tc.recs)
			if len(s.ByTier) != len(tc.wantTier) {
				t.Fatalf("by_tier = %+v, want %+v", s.ByTier, tc.wantTier)
			}
			for k, w := range tc.wantTier {
				if s.ByTier[k] != w {
					t.Errorf("by_tier[%s] = %+v, want %+v", k, s.ByTier[k], w)
				}
			}
			if len(s.ByRestore) != len(tc.wantRestore) || s.RestoreUnserved != tc.wantUnserved {
				t.Fatalf("by_restore = %v unserved=%d, want %v/%d", s.ByRestore, s.RestoreUnserved, tc.wantRestore, tc.wantUnserved)
			}
			for k, w := range tc.wantRestore {
				if s.ByRestore[k] != w {
					t.Errorf("by_restore[%s] = %d, want %d", k, s.ByRestore[k], w)
				}
			}
			line := RenderCompact(BuildReport(tc.recs, 0, false, 0))
			for _, want := range tc.wantCompact {
				if !strings.Contains(line, want) {
					t.Errorf("compact = %s\nwant substring %q", line, want)
				}
			}
			if tc.wantNoTier && (strings.Contains(line, "tier served") || strings.Contains(line, "restore ")) {
				t.Errorf("compact = %s, want no tier/restore segment on a proxy-only window", line)
			}
			keys := jsonKeys(t, tc.recs[0])
			if _, ok := keys["cache_tier"]; ok != (tc.recs[0].CacheTier != "") {
				t.Errorf("cache_tier omitempty broken: %v", keys)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRecordQueueAndModelRoundTripAndOldRowsStillParse(t *testing.T) {
	// A row written before queue_ms/model existed must still parse as schema v1.
	old := `{"schema":"fak.gateway.perf-record.v1","unix_ms":1,"prompt_tokens":10,"completion_tokens":2,"cached_tokens":0,"e2e_ms":50,"ttft_ms":20}`
	var r Record
	if err := json.Unmarshal([]byte(old), &r); err != nil {
		t.Fatalf("old row: %v", err)
	}
	if r.QueueMS != nil || r.Model != "" || r.TTFTMS != 20 {
		t.Fatalf("old row decoded = %+v, want queue unknown, no model", r)
	}
	if m := jsonKeys(t, r); m["queue_ms"] != nil || m["model"] != nil {
		t.Fatalf("old row re-encodes new keys: %v", m)
	}

	n := NewRecord(time.Now(), "stop", LocalitySelfHosted, 10, 2, 0, time.Second, 200*time.Millisecond).WithQueue(1500 * time.Microsecond)
	n.Model = "qwen"
	raw, _ := json.Marshal(n)
	var back Record
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("new row: %v", err)
	}
	if back.QueueMS == nil || *back.QueueMS != 1.5 || back.Model != "qwen" {
		t.Fatalf("round trip = %+v, want queue 1.5ms model qwen", back)
	}
	// An immediate admit is a KNOWN zero wait, not an absent one.
	zero := NewRecord(time.Now(), "stop", "", 1, 1, 0, time.Second, 0).WithQueue(0)
	if m := jsonKeys(t, zero); m["queue_ms"] != float64(0) {
		t.Fatalf("zero wait encoded as %v, want 0", m["queue_ms"])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeSplitsTTFTIntoQueueAndService(t *testing.T) {
	q := func(ms float64) *float64 { return &ms }
	recs := []Record{
		{Schema: Schema, TTFTMS: 100, QueueMS: q(0)},
		{Schema: Schema, TTFTMS: 200, QueueMS: q(300)},
		{Schema: Schema, TTFTMS: 300, QueueMS: q(10)},
		{Schema: Schema, QueueMS: q(900)}, // queue known, ttft unmeasured
		{Schema: Schema, TTFTMS: 5000},    // no scheduler: excluded from the split
	}
	s := Summarize(recs)
	if s.QueueMeasured != 4 || s.QueueP50MS != 10 || s.QueueP99MS != 900 {
		t.Fatalf("queue = %d p50=%v p99=%v, want 4/10/900", s.QueueMeasured, s.QueueP50MS, s.QueueP99MS)
	}
	// client = queue+ttft over {100, 500, 310}; service = ttft over {100, 200, 300}.
	if s.ClientTTFTP50MS != 310 || s.ServiceTTFTP50MS != 200 {
		t.Fatalf("client/service p50 = %v/%v, want 310/200", s.ClientTTFTP50MS, s.ServiceTTFTP50MS)
	}
	line := RenderCompact(Report{Summary: s})
	for _, want := range []string{"queue p50=10ms p99=900ms (measured 4/5)", "client_ttft p50=310ms service_ttft p50=200ms"} {
		if !strings.Contains(line, want) {
			t.Fatalf("compact line missing %q: %s", want, line)
		}
	}
	if none := RenderCompact(Report{Summary: Summarize([]Record{{Schema: Schema, TTFTMS: 5}})}); strings.Contains(none, "queue") {
		t.Fatalf("queue axis rendered with no queue data: %s", none)
	}
}
