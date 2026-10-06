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
	r := NewRecord(now, "stop", LocalitySelfHosted, 1000, 200, 500, 5*time.Second, time.Second)
	if r.Schema != Schema || r.UnixMS != 1_700_000_000_000 {
		t.Fatalf("schema/unix_ms = %q/%d", r.Schema, r.UnixMS)
	}
	if r.PromptTokens != 1000 || r.CompletionTokens != 200 || r.CachedTokens != 500 {
		t.Fatalf("tokens = %d/%d/%d", r.PromptTokens, r.CompletionTokens, r.CachedTokens)
	}
	if r.E2EMS != 5000 || r.TTFTMS != 1000 {
		t.Fatalf("e2e/ttft = %v/%v, want 5000/1000", r.E2EMS, r.TTFTMS)
	}
	if r.PrefillTPS != 1000 {
		t.Fatalf("prefill_tps = %v, want 1000 (1000 tok / 1s)", r.PrefillTPS)
	}
	if r.DecodeTPS != 50 {
		t.Fatalf("decode_tps = %v, want 50 (200 tok / 4s)", r.DecodeTPS)
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

func fixedRecords() []Record {
	return []Record{
		{Schema: Schema, TTFTMS: 100, PrefillTPS: 10, DecodeTPS: 1, E2EMS: 1000, PromptTokens: 100, CachedTokens: 0},
		{Schema: Schema, TTFTMS: 200, PrefillTPS: 20, DecodeTPS: 2, E2EMS: 2000, PromptTokens: 100, CachedTokens: 100},
		{Schema: Schema, TTFTMS: 300, PrefillTPS: 30, DecodeTPS: 3, E2EMS: 3000, PromptTokens: 100, CachedTokens: 100},
		{Schema: Schema, TTFTMS: 400, PrefillTPS: 40, DecodeTPS: 4, E2EMS: 4000, PromptTokens: 100, CachedTokens: 100},
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
	// nearest-rank over [1000..5000]: p50 -> rank 3, p99 -> rank 5.
	if s.E2EP50MS != 3000 || s.E2EP99MS != 5000 {
		t.Fatalf("e2e p50/p99 = %v/%v, want 3000/5000", s.E2EP50MS, s.E2EP99MS)
	}
	// cached 300 / (uncached 400 + cached 300).
	if s.CacheHitShare != 0.4286 {
		t.Fatalf("cache_hit_share = %v, want 0.4286", s.CacheHitShare)
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
