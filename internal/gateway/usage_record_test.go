package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

func TestUsageRecordOncePerCompletedTurn(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	s := &Server{metrics: m, model: "test-model"}

	s.logInferenceTurn("trace-A", "anthropic_messages", true, agent.Usage{
		PromptTokens:             10,
		CompletionTokens:         4,
		TotalTokens:              14,
		CacheReadInputTokens:     8,
		CacheCreationInputTokens: 1,
	}, "end_turn", time.Millisecond, false)
	s.logInferenceTurn("trace-A", "anthropic_messages", true, agent.Usage{
		PromptTokens:             6,
		CompletionTokens:         2,
		CacheReadInputTokens:     900,
		CacheCreationInputTokens: 0,
	}, "stop", time.Millisecond, false)

	records, capped := m.usageRecordsSnapshot()
	if capped {
		t.Fatal("window reported capped before the cap was reached")
	}
	if len(records) != 2 {
		t.Fatalf("usage records = %d, want exactly one per completed turn (2)", len(records))
	}
	first := records[0]
	if first.Ordinal != 1 || first.Family != "trace-A" || first.Wire != "anthropic_messages" || !first.Stream {
		t.Fatalf("first record = %+v", first)
	}
	if first.InputTokens != 10 || first.OutputTokens != 4 || first.CachedTokens != 8 || first.CacheWriteTokens != 1 {
		t.Fatalf("first record token axes = %+v", first)
	}
	if first.ReasoningTokens != 0 {
		t.Fatalf("first record reasoning_tokens = %d, want 0 when the provider reports none", first.ReasoningTokens)
	}
	if first.UnixMillis == 0 {
		t.Fatal("first record unix_millis unset")
	}
	// The join key against the native per-turn plane: the record's family+millis must
	// match the vcache Turn the same chokepoint recorded.
	turns, _ := m.vcacheTurnsSnapshot()
	if len(turns) != 2 || turns[0].Family != first.Family || turns[0].UnixMillis != first.UnixMillis {
		t.Fatalf("vcache turns do not join to usage records: turns=%+v record=%+v", turns, first)
	}
	second := records[1]
	if second.Ordinal != 2 || second.CachedTokens != 900 {
		t.Fatalf("second record = %+v, want ordinal 2 with the provider cache read carried", second)
	}
}

func TestUsageRecordOrdinalPerTrace(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	s := &Server{metrics: m}

	s.logInferenceTurn("t1", "wire", false, agent.Usage{PromptTokens: 5}, "stop", time.Millisecond, false)
	s.logInferenceTurn("t1", "wire", false, agent.Usage{PromptTokens: 5}, "stop", time.Millisecond, false)
	s.logInferenceTurn("t2", "wire", false, agent.Usage{PromptTokens: 5}, "stop", time.Millisecond, false)

	records, _ := m.usageRecordsSnapshot()
	if len(records) != 3 {
		t.Fatalf("records = %d, want 3", len(records))
	}
	if records[0].Ordinal != 1 || records[1].Ordinal != 2 {
		t.Fatalf("t1 ordinals = %d,%d, want 1,2 (per-trace monotonic)", records[0].Ordinal, records[1].Ordinal)
	}
	if records[2].Family != "t2" || records[2].Ordinal != 1 {
		t.Fatalf("t2 record = %+v, want a fresh per-trace ordinal 1", records[2])
	}
}

func TestUsageRecordCacheRatioAndAlignment(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	s := &Server{metrics: m}

	// 80%+ cached: aligned. 50% cached: not. Anthropic shape (prompt excludes the
	// cached span) and OpenAI shape (prompt includes it) must read the same ratio.
	s.logInferenceTurn("aligned", "wire", false, agent.Usage{
		PromptTokens:         20,
		CacheReadInputTokens: 80,
	}, "stop", time.Millisecond, false)
	s.logInferenceTurn("missed", "wire", false, agent.Usage{
		PromptTokens:         50,
		CacheReadInputTokens: 50,
	}, "stop", time.Millisecond, false)
	s.logInferenceTurn("openai-folded", "wire", false, agent.Usage{
		PromptTokens:        100,
		PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 90},
	}, "stop", time.Millisecond, false)
	s.logInferenceTurn("empty", "wire", false, agent.Usage{}, "stop", time.Millisecond, false)

	records, _ := m.usageRecordsSnapshot()
	byOrdinal := map[string]UsageRecord{}
	for _, r := range records {
		byOrdinal[r.Family] = r
	}
	if got := byOrdinal["aligned"]; got.CacheRatio < CacheAlignedThreshold || !got.CacheAligned {
		t.Fatalf("aligned record = %+v, want ratio >= %v and aligned", got, CacheAlignedThreshold)
	}
	if got := byOrdinal["missed"]; got.CacheRatio >= CacheAlignedThreshold || got.CacheAligned {
		t.Fatalf("missed record = %+v, want ratio < %v and not aligned", got, CacheAlignedThreshold)
	}
	if got := byOrdinal["openai-folded"]; got.CacheRatio < CacheAlignedThreshold || !got.CacheAligned {
		t.Fatalf("openai-folded record = %+v, want the folded cached span normalized into the ratio", got)
	}
	if got := byOrdinal["empty"]; got.CacheRatio != 0 || got.CacheAligned {
		t.Fatalf("empty record = %+v, want ratio 0 and not aligned (no divide-by-zero)", got)
	}
}

func TestUsageRecordReasoningTokensCarried(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	s := &Server{metrics: m}

	s.logInferenceTurn("think", "wire", false, agent.Usage{
		PromptTokens:            100,
		CompletionTokens:        30,
		CompletionTokensDetails: &agent.UsageCompletionTokenDetails{ReasoningTokens: 22},
	}, "stop", time.Millisecond, false)

	records, _ := m.usageRecordsSnapshot()
	if len(records) != 1 || records[0].ReasoningTokens != 22 {
		t.Fatalf("records = %+v, want reasoning_tokens 22 carried", records)
	}
}

func TestUsageRecordLogEventIsSiblingOfInferenceTurn(t *testing.T) {
	srv := newTestServer(t)
	var lines []string
	srv.logf = func(format string, args ...any) {
		lines = append(lines, formatLog(format, args...))
	}

	srv.logInferenceTurn("trace-turn", "anthropic_messages", true, agent.Usage{
		PromptTokens:             10,
		CompletionTokens:         2,
		CacheReadInputTokens:     8,
		CacheCreationInputTokens: 1,
	}, "end_turn", time.Millisecond, false)

	var inference, usage []map[string]any
	for _, line := range lines {
		var ev map[string]any
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("log line is not JSON: %v\n%s", err, line)
		}
		switch ev["event"] {
		case "gateway_inference_turn":
			inference = append(inference, ev)
		case UsageRecordEventName:
			usage = append(usage, ev)
		}
	}
	if len(inference) != 1 || len(usage) != 1 {
		t.Fatalf("want exactly one gateway_inference_turn AND exactly one %s per completed turn, got %d/%d: %v",
			UsageRecordEventName, len(inference), len(usage), lines)
	}
	ev := usage[0]
	if ev["trace_id"] != "trace-turn" || ev["family"] != "trace-turn" || ev["wire"] != "anthropic_messages" {
		t.Fatalf("usage event identity = %v", ev)
	}
	if ev["request_ordinal"] != float64(1) || ev["input_tokens"] != float64(10) ||
		ev["output_tokens"] != float64(2) || ev["cached_tokens"] != float64(8) ||
		ev["cache_write_tokens"] != float64(1) {
		t.Fatalf("usage event axes = %v", ev)
	}
	if _, ok := ev["unix_millis"].(float64); !ok {
		t.Fatalf("usage event unix_millis missing/non-number: %v", ev)
	}
}

func TestFakUsageCacheAlignmentIdleShape(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{}).handleFakUsageCacheAlignment(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/usage/cache-alignment", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var rep usageAlignmentReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if rep.Schema != usageAlignmentSchema || rep.Summary.Count != 0 || len(rep.Records) != 0 {
		t.Fatalf("idle report = %+v", rep)
	}
	if rep.AlignedThreshold != CacheAlignedThreshold {
		t.Fatalf("aligned_threshold = %v, want the canonical %v visible in the response", rep.AlignedThreshold, CacheAlignedThreshold)
	}
}

func TestFakUsageCacheAlignmentRejectsNonGet(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{}).handleFakUsageCacheAlignment(rec, httptest.NewRequest(http.MethodPost, "/v1/fak/usage/cache-alignment", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST: status=%d want 405", rec.Code)
	}
}

func TestFakUsageCacheAlignmentSummaryAndJoin(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	s := &Server{metrics: m}

	// Warm family: a cold write, then a served-from-cache read — the native governor
	// posture (ride natural) and the provider behavior agree on the second turn.
	s.logInferenceTurn("warm", "wire", false, agent.Usage{
		PromptTokens:             100,
		CacheCreationInputTokens: 900,
	}, "stop", time.Millisecond, false)
	s.logInferenceTurn("warm", "wire", false, agent.Usage{
		PromptTokens:         50,
		CacheReadInputTokens: 950,
	}, "stop", time.Millisecond, false)
	// Misaligned family: fak's warm posture says the prefix should be warm, but the
	// provider served nothing from cache (the invalidated-prefix case).
	s.logInferenceTurn("broken", "wire", false, agent.Usage{
		PromptTokens:             100,
		CacheCreationInputTokens: 900,
	}, "stop", time.Millisecond, false)
	s.logInferenceTurn("broken", "wire", false, agent.Usage{
		PromptTokens: 1000,
	}, "stop", time.Millisecond, false)
	// No native warm-state receipt: a turn with no cache axes at all.
	s.logInferenceTurn("tiny", "wire", false, agent.Usage{PromptTokens: 10}, "stop", time.Millisecond, false)

	rec := httptest.NewRecorder()
	s.handleFakUsageCacheAlignment(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/usage/cache-alignment", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var rep usageAlignmentReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if rep.Summary.Count != 5 {
		t.Fatalf("summary.count = %d, want 5", rep.Summary.Count)
	}
	if rep.Summary.CacheAligned != 1 {
		t.Fatalf("summary.cache_aligned = %d, want 1 (only the 950/1000 read clears the threshold)", rep.Summary.CacheAligned)
	}
	// Per-record class against the family's latest native posture: warm t1 (the
	// write) and both broken rows ride a warm posture without a provider read;
	// warm t2 agrees; tiny's family never engaged the cache plane.
	if rep.Summary.JoinAligned != 1 || rep.Summary.JoinMisaligned != 3 || rep.Summary.JoinWarmUnknown != 1 {
		t.Fatalf("summary join classes = %+v, want aligned=1 misaligned=3 warm_unknown=1", rep.Summary)
	}
	if rep.Summary.CacheAlignedRatio <= 0 || rep.Summary.CacheAlignedRatio > 1 {
		t.Fatalf("summary.cache_aligned_ratio = %v, want a ratio in (0,1]", rep.Summary.CacheAlignedRatio)
	}
	if len(rep.Records) != 5 {
		t.Fatalf("records = %d, want 5", len(rep.Records))
	}

	classes := map[string]string{}
	for _, row := range rep.Records {
		classes[row.Family] = row.Join.Class
		switch row.Join.Class {
		case usageJoinAligned, usageJoinMisaligned, usageJoinWarmUnknown:
		default:
			t.Fatalf("record family %s carries unknown join class %q", row.Family, row.Join.Class)
		}
		if row.Join.Provenance["provider_cache"] != "OBSERVED" || row.Join.Provenance["native_warm_state"] != "DECISION" {
			t.Fatalf("record family %s provenance = %v", row.Family, row.Join.Provenance)
		}
	}
	if classes["warm"] != usageJoinAligned || classes["broken"] != usageJoinMisaligned || classes["tiny"] != usageJoinWarmUnknown {
		t.Fatalf("join classes by family = %v", classes)
	}

	// The misaligned row must expose both sides of the divergence: the native warm
	// posture expected a warm prefix, the provider served nothing from cache.
	for _, row := range rep.Records {
		if row.Family == "broken" && row.Ordinal == 2 {
			if !row.Join.NativeWarmExpected || row.Join.ProviderServedFromCache {
				t.Fatalf("misaligned row join = %+v, want native warm expected + provider did not serve from cache", row.Join)
			}
			if row.Join.NativeGovernorDecision == "" {
				t.Fatalf("misaligned row missing native_governor_decision: %+v", row.Join)
			}
		}
		if row.Family == "warm" && row.Ordinal == 2 && !row.Join.ProviderServedFromCache {
			t.Fatalf("warm row join = %+v, want provider served from cache", row.Join)
		}
	}
}

func TestFakUsageCacheAlignmentNParamBounds(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	s := &Server{metrics: m}
	for i := 0; i < 7; i++ {
		s.logInferenceTurn("t", "wire", false, agent.Usage{PromptTokens: 10, CacheReadInputTokens: 90}, "stop", time.Millisecond, false)
	}

	rec := httptest.NewRecorder()
	s.handleFakUsageCacheAlignment(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/usage/cache-alignment?n=3", nil))
	var rep usageAlignmentReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(rep.Records) != 3 || rep.Summary.Count != 3 {
		t.Fatalf("n=3 gave %d records / count %d", len(rep.Records), rep.Summary.Count)
	}
	if rep.Records[0].Ordinal != 5 || rep.Records[2].Ordinal != 7 {
		t.Fatalf("n=3 records = ordinals %d..%d, want the LAST 3 (5..7)", rep.Records[0].Ordinal, rep.Records[2].Ordinal)
	}

	rec = httptest.NewRecorder()
	s.handleFakUsageCacheAlignment(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/usage/cache-alignment?n=99999", nil))
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(rep.Records) != 7 {
		t.Fatalf("oversized n clamped to the retained window: got %d records, want 7", len(rep.Records))
	}
}

func TestUsageRecordWindowCapDropOldest(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	for i := 0; i < usageRecordCap+50; i++ {
		m.recordUsageTurn("s", "wire", false, agent.Usage{PromptTokens: 10}, int64(i))
	}
	records, capped := m.usageRecordsSnapshot()
	if !capped {
		t.Fatal("capped flag false after drop-oldest")
	}
	if len(records) != usageRecordCap {
		t.Fatalf("window length %d, want cap %d", len(records), usageRecordCap)
	}
	if records[0].Ordinal != uint64(51) || records[len(records)-1].Ordinal != uint64(usageRecordCap+50) {
		t.Fatalf("oldest retained ordinal = %d, newest = %d, want drop-oldest continuity 51..%d",
			records[0].Ordinal, records[len(records)-1].Ordinal, usageRecordCap+50)
	}
}

func TestUsageRecordEffortHonestWhenUnavailable(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	s := &Server{metrics: m}
	s.logInferenceTurn("t", "wire", false, agent.Usage{PromptTokens: 10}, "stop", time.Millisecond, false)

	records, _ := m.usageRecordsSnapshot()
	if len(records) != 1 {
		t.Fatalf("records = %d, want 1", len(records))
	}
	b, err := json.Marshal(records[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "\"effort\":") {
		t.Fatalf("record carries an effort value (%s); effort is not threaded on this surface and must stay honestly absent", b)
	}
}

func TestUsageAlignmentClassReconciliation(t *testing.T) {
	dec := func(d vcachegov.GovernorDecision) *vcacheGovernorDecisionRecord {
		return &vcacheGovernorDecisionRecord{Decision: string(d)}
	}
	cases := []struct {
		name     string
		decision *vcacheGovernorDecisionRecord
		cached   int64
		written  int64
		want     string
	}{
		{"no receipt", nil, 0, 0, usageJoinWarmUnknown},
		{"warm posture served", dec(vcachegov.DecisionRideNatural), 100, 0, usageJoinAligned},
		{"warm posture cold", dec(vcachegov.DecisionHeartbeatPin), 0, 0, usageJoinMisaligned},
		{"lapse posture stayed cold", dec(vcachegov.DecisionLazyRebuild), 0, 0, usageJoinAligned},
		{"lapse posture still hot", dec(vcachegov.DecisionEvict), 100, 0, usageJoinMisaligned},
		{"never-warm stayed silent", dec(vcachegov.DecisionNoCache), 0, 0, usageJoinAligned},
		{"never-warm wrote", dec(vcachegov.DecisionExplicitCache), 0, 40000, usageJoinMisaligned},
	}
	for _, tc := range cases {
		rec := UsageRecord{CachedTokens: tc.cached, CacheWriteTokens: tc.written}
		if got := usageAlignmentClass(rec, tc.decision); got != tc.want {
			t.Fatalf("%s: class = %s, want %s", tc.name, got, tc.want)
		}
	}
}
