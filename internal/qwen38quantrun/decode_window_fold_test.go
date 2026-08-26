package qwen38quantrun

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestDecodeWindows(t *testing.T) {
	trace := nativeTrace([]int64{100, 200, 300, 400, 500, 600, 700, 850, 1000})
	report, err := BuildDecodeWindows(trace, NativeDecodeContract, 9)
	if err != nil {
		t.Fatal(err)
	}
	wantElapsed := []int64{300, 300, 400}
	wantRates := []float64{10_000_000, 10_000_000, 7_500_000}
	for i, window := range report.Windows {
		if window.Tokens != 3 || window.ElapsedNS != wantElapsed[i] || window.TokensPerSecond != wantRates[i] {
			t.Fatalf("window[%d]=%+v", i, window)
		}
	}
	if report.RawCompletionTokens != 9 || report.TimedTokens != 9 || report.ElapsedNS != 1000 || report.LateEarlyRatio != 0.75 || report.LinearSlopeTokensPerSec >= 0 {
		t.Fatalf("report=%+v", report)
	}

	pass, err := BuildDecodeWindows(nativeTrace([]int64{100, 200, 300, 400, 500, 600, 700, 800, 900}), NativeDecodeContract, 9)
	if err != nil {
		t.Fatal(err)
	}
	summary, err := FoldDecodeRepetitions([]DecodeWindowReport{pass, pass, pass})
	if err != nil {
		t.Fatal(err)
	}
	if summary.Verdict != "PASS" || summary.Confidence.Repetitions != 3 || summary.Confidence.LateEarlyStdDev != 0 || summary.LateEarlyRatio != 1 {
		t.Fatalf("summary=%+v", summary)
	}
	for i, window := range summary.Windows {
		if window.FirstTokenIndex != 0 || window.LastTokenIndex != 0 || window.Tokens != pass.Windows[i].Tokens*MinimumDecodeRepetitions {
			t.Fatalf("aggregate window[%d]=%+v", i, window)
		}
	}
	if _, err := FoldDecodeRepetitions([]DecodeWindowReport{pass, pass, pass, pass}); err == nil || !strings.Contains(err.Error(), "exactly 3") {
		t.Fatalf("four-repetition err=%v", err)
	}
	equalNative, err := BuildDecodeWindows(nativeTrace([]int64{10, 10, 20, 30, 40, 50}), NativeDecodeContract, 6)
	if err != nil || equalNative.Windows[0].ElapsedNS != 10 {
		t.Fatalf("equal native timestamps report=%+v err=%v", equalNative, err)
	}
	hold, err := FoldDecodeRepetitions([]DecodeWindowReport{report, report, report})
	if err != nil {
		t.Fatal(err)
	}
	if hold.Verdict != "HOLD" || !strings.Contains(hold.Failure, "below 0.85") {
		t.Fatalf("hold=%+v", hold)
	}
}

func TestDecodeWindowsRejectsInvalidTrace(t *testing.T) {
	valid := nativeTrace([]int64{10, 20, 30})
	tests := []struct {
		name string
		edit func(*DecodeTrace) int
		want string
	}{
		{"schema", func(trace *DecodeTrace) int { trace.Schema = "wrong"; return 3 }, "schema mismatch"},
		{"engine", func(trace *DecodeTrace) int { trace.Engine = "llama.cpp"; return 3 }, "engine mismatch"},
		{"provenance", func(trace *DecodeTrace) int { trace.Provenance = "sse-fragment"; return 3 }, "provenance mismatch"},
		{"missing", func(trace *DecodeTrace) int { trace.Events = trace.Events[:2]; return 3 }, "completion-token mismatch"},
		{"duplicate", func(trace *DecodeTrace) int { trace.Events[1].TokenIndex = 1; return 3 }, "duplicate or out-of-order"},
		{"gap", func(trace *DecodeTrace) int { trace.Events[1].TokenIndex = 3; return 3 }, "gapped"},
		{"non-monotonic", func(trace *DecodeTrace) int { trace.Events[1].ElapsedNS = 9; return 3 }, "non-monotonic"},
		{"completion mismatch", func(*DecodeTrace) int { return 4 }, "completion-token mismatch"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			trace := valid
			trace.Events = append([]DecodeTraceEvent(nil), valid.Events...)
			if _, err := BuildDecodeWindows(trace, NativeDecodeContract, tc.edit(&trace)); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want %q", err, tc.want)
			}
		})
	}
}

func TestDecodeWindowsNativeWire(t *testing.T) {
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "exact", "choices": []any{map[string]any{"message": map[string]any{"content": "ok"}}},
			"usage": map[string]int{"completion_tokens": 6},
			"fak": map[string]any{"decode_trace": map[string]any{
				"schema": NativeDecodeTraceSchema, "engine": NativeDecodeTraceEngine,
				"events": []any{
					map[string]any{"token_index": 1, "elapsed_ns": 10}, map[string]any{"token_index": 2, "elapsed_ns": 20},
					map[string]any{"token_index": 3, "elapsed_ns": 30}, map[string]any{"token_index": 4, "elapsed_ns": 40},
					map[string]any{"token_index": 5, "elapsed_ns": 50}, map[string]any{"token_index": 6, "elapsed_ns": 60},
				},
			}},
		})
	}))
	defer server.Close()
	response, err := runOne(context.Background(), server.Client(), Config{Endpoint: server.URL, APIKey: "secret", Model: "exact", NativeDecodeTrace: true}, qwen38quant.Fixture{ID: "long", Workload: "coding_reasoning", Prompt: "x", MaxOutputTokens: 6})
	if err != nil {
		t.Fatal(err)
	}
	if request["fak_decode_trace"] != true || request["stream"] != nil {
		t.Fatalf("request=%#v", request)
	}
	if response.Fak.DecodeTrace.Provenance != NativeDecodeTraceProvenance || response.DecodeWindows == nil || response.DecodeWindows.LateEarlyRatio != 1 {
		t.Fatalf("response=%+v", response)
	}
}

func TestDecodeWindowsNativeWireFailsClosed(t *testing.T) {
	for _, tc := range []struct {
		name, response, want string
	}{
		{"missing", `{"model":"exact","choices":[{"message":{"content":"ok"}}],"usage":{"completion_tokens":3}}`, "missing fak.decode_trace"},
		{"wrong engine", `{"model":"exact","choices":[{"message":{"content":"ok"}}],"usage":{"completion_tokens":3},"fak":{"decode_trace":{"schema":"fak.native-decode-trace/1","engine":"llama.cpp","events":[{"token_index":1,"elapsed_ns":1},{"token_index":2,"elapsed_ns":2},{"token_index":3,"elapsed_ns":3}]}}}`, "engine mismatch"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(tc.response)) }))
			defer server.Close()
			_, err := runOne(context.Background(), server.Client(), Config{Endpoint: server.URL, APIKey: "secret", Model: "exact", NativeDecodeTrace: true}, qwen38quant.Fixture{ID: "long", Workload: "coding_reasoning", Prompt: "x", MaxOutputTokens: 3})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want=%q", err, tc.want)
			}
		})
	}
}

func TestDecodeWindowsComparatorProvenanceFailsClosed(t *testing.T) {
	trace := nativeTrace([]int64{10, 20, 30})
	comparator := DecodeTraceContract{Schema: "llama.decode-transport/1", Engine: qwen38quant.EngineLlamaCpp, Provenance: "transport-arrival"}
	if _, err := BuildDecodeWindows(trace, comparator, 3); err == nil || !strings.Contains(err.Error(), "schema mismatch") {
		t.Fatalf("err=%v", err)
	}
	trace.Schema, trace.Engine, trace.Provenance = comparator.Schema, comparator.Engine, "sse-fragment"
	if _, err := BuildDecodeWindows(trace, comparator, 3); err == nil || !strings.Contains(err.Error(), "provenance mismatch") {
		t.Fatalf("err=%v", err)
	}
}

func TestDecodeWindowsLlamaClientArrivalContract(t *testing.T) {
	events := make([]LlamaClientArrivalEvent, MinimumLongDecodeTokens)
	for i := range events {
		events[i] = LlamaClientArrivalEvent{TokenIDs: []int{1000 + i}, TokensPredicted: i + 1, ElapsedNS: int64(i+1) * 1_000_000}
	}
	final := LlamaClientFinal{StopType: "limit", TokensPredicted: len(events)}
	trace, report, err := BuildLlamaClientDecodeWindows(events, final)
	if err != nil {
		t.Fatal(err)
	}
	if trace.Provenance != LlamaClientDecodeTraceProvenance || len(trace.Events) != MinimumLongDecodeTokens-1 || trace.Events[0].TokenID == nil || *trace.Events[0].TokenID != 1001 || report.RawCompletionTokens != MinimumLongDecodeTokens || report.TimedTokens != MinimumLongDecodeTokens-1 || report.LateEarlyRatio <= 0 {
		t.Fatalf("trace/report=%+v %+v", trace.Events[0], report)
	}
	offsetEvents := cloneLlamaEvents(events)
	for i := range offsetEvents {
		offsetEvents[i].ElapsedNS += 987_654_321
	}
	offsetTrace, offsetReport, err := BuildLlamaClientDecodeWindows(offsetEvents, final)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(trace, offsetTrace) || !reflect.DeepEqual(report, offsetReport) {
		t.Fatalf("first-arrival offset changed decode windows")
	}
	sameReadEvents := cloneLlamaEvents(events)
	for i := range sameReadEvents {
		sameReadEvents[i].ElapsedNS = int64(i/2+1) * 1_000_000
	}
	sameReadTrace, sameReadReport, err := BuildLlamaClientDecodeWindows(sameReadEvents, final)
	if err != nil {
		t.Fatal(err)
	}
	if sameReadTrace.Events[0].ElapsedNS != 0 || sameReadReport.Windows[0].ElapsedNS <= 0 || sameReadReport.Windows[1].ElapsedNS <= 0 || sameReadReport.Windows[2].ElapsedNS <= 0 {
		t.Fatalf("same-read trace/report=%+v %+v", sameReadTrace.Events[:2], sameReadReport)
	}
	allSameEvents := cloneLlamaEvents(events)
	for i := range allSameEvents {
		allSameEvents[i].ElapsedNS = 1_000_000
	}
	if _, _, err := BuildLlamaClientDecodeWindows(allSameEvents, final); err == nil || !strings.Contains(err.Error(), "report is incomplete") {
		t.Fatalf("all-same arrival err=%v", err)
	}
	for _, tc := range []struct {
		name string
		edit func([]LlamaClientArrivalEvent, *LlamaClientFinal)
		want string
	}{
		{"stop", func(_ []LlamaClientArrivalEvent, final *LlamaClientFinal) { final.StopType = "eos" }, "stop_type"},
		{"short", func(events []LlamaClientArrivalEvent, final *LlamaClientFinal) {
			final.TokensPredicted = len(events) - 1
		}, "below 2048"},
		{"final tokens", func(_ []LlamaClientArrivalEvent, final *LlamaClientFinal) { final.TokenIDs = []int{1} }, "final event tokens"},
		{"counter", func(events []LlamaClientArrivalEvent, _ *LlamaClientFinal) { events[4].TokensPredicted = 4 }, "tokens_predicted at event"},
		{"singleton", func(events []LlamaClientArrivalEvent, _ *LlamaClientFinal) { events[4].TokenIDs = []int{1, 2} }, "want singleton"},
		{"backwards time", func(events []LlamaClientArrivalEvent, _ *LlamaClientFinal) {
			events[4].ElapsedNS = events[3].ElapsedNS - 1
		}, "non-monotonic"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotEvents := append([]LlamaClientArrivalEvent(nil), events...)
			for i := range gotEvents {
				gotEvents[i].TokenIDs = append([]int(nil), events[i].TokenIDs...)
			}
			gotFinal := final
			tc.edit(gotEvents, &gotFinal)
			if _, _, err := BuildLlamaClientDecodeWindows(gotEvents, gotFinal); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want=%q", err, tc.want)
			}
		})
	}
}

func TestDecodeWindowsFoldRequiresOneThreeRepFixture(t *testing.T) {
	traceTemplate := nativeLinearTrace(MinimumLongDecodeTokens, 10)
	report, err := BuildDecodeWindows(traceTemplate, NativeDecodeContract, MinimumLongDecodeTokens)
	if err != nil {
		t.Fatal(err)
	}
	results := make([]Result, 3)
	for i := range results {
		trace := nativeLinearTrace(MinimumLongDecodeTokens, 10)
		window := report
		window.Windows = append([]DecodeWindow(nil), report.Windows...)
		results[i] = Result{FixtureID: "long", Repeat: i + 1, Usage: map[string]int{"completion_tokens": MinimumLongDecodeTokens}, Quality: "PASS", DecodeTrace: &trace, DecodeWindows: &window}
	}
	results[2].FixtureID = "other"
	if summary := FoldDecodeResults(results); summary.Verdict != "HOLD" || !strings.Contains(summary.Failure, "unmatched decode repetition identity") {
		t.Fatalf("summary=%+v", summary)
	}
	results[2].FixtureID = "long"
	results[1].Quality = "FAIL"
	if summary := FoldDecodeResults(results); summary.Verdict != "HOLD" || !strings.Contains(summary.Failure, `quality="FAIL"`) {
		t.Fatalf("quality summary=%+v", summary)
	}
	results[1].Quality, results[1].Failure = "PASS", "graded failure"
	if summary := FoldDecodeResults(results); summary.Verdict != "HOLD" || !strings.Contains(summary.Failure, "graded failure") {
		t.Fatalf("failure summary=%+v", summary)
	}
	results[1].Failure = ""
	results[1].Usage["completion_tokens"] = MinimumLongDecodeTokens - 1
	if summary := FoldDecodeResults(results); summary.Verdict != "HOLD" || !strings.Contains(summary.Failure, "below 2048") {
		t.Fatalf("short summary=%+v", summary)
	}
}

func TestDecodeWindowsMatchedCampaignVerdict(t *testing.T) {
	makeNative := func(slowLate bool) []Result {
		events := make([]DecodeTraceEvent, MinimumLongDecodeTokens)
		var elapsed int64
		for i := range events {
			step := int64(1_000_000)
			if slowLate && i >= 2*MinimumLongDecodeTokens/3 {
				step = 2_000_000
			}
			elapsed += step
			events[i] = DecodeTraceEvent{TokenIndex: i + 1, ElapsedNS: elapsed}
		}
		trace := DecodeTrace{Schema: NativeDecodeContract.Schema, Engine: NativeDecodeContract.Engine, Provenance: NativeDecodeContract.Provenance, Events: events}
		report, err := BuildDecodeWindows(trace, NativeDecodeContract, MinimumLongDecodeTokens)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]Result, MinimumDecodeRepetitions)
		for i := range out {
			gotTrace := trace
			gotTrace.Events = append([]DecodeTraceEvent(nil), trace.Events...)
			gotReport := report
			gotReport.Windows = append([]DecodeWindow(nil), report.Windows...)
			out[i] = Result{FixtureID: "long", Workload: LongDecodeWorkload, Repeat: i + 1, Usage: map[string]int{"completion_tokens": MinimumLongDecodeTokens}, Quality: "PASS", DecodeTrace: &gotTrace, DecodeWindows: &gotReport}
		}
		return out
	}
	makeComparator := func() []LlamaClientDecodeResult {
		out := make([]LlamaClientDecodeResult, MinimumDecodeRepetitions)
		for repetition := range out {
			events := make([]LlamaClientArrivalEvent, MinimumLongDecodeTokens)
			for i := range events {
				events[i] = LlamaClientArrivalEvent{TokenIDs: []int{i}, TokensPredicted: i + 1, ElapsedNS: 100_000_000 + int64(i+1)*1_000_000}
			}
			out[repetition] = LlamaClientDecodeResult{FixtureID: "long", Repeat: repetition + 1, Events: events, Final: LlamaClientFinal{StopType: "limit", TokensPredicted: MinimumLongDecodeTokens}}
		}
		return out
	}
	native := makeNative(false)
	comparator := makeComparator()
	matched, err := FoldMatchedDecodeCampaign(native, comparator)
	if err != nil || matched.Verdict != "PASS" {
		t.Fatalf("matched=%+v err=%v", matched, err)
	}
	if matched.Native.RawCompletionTokensPerRun != MinimumLongDecodeTokens || matched.Native.TimedTokensPerRun != MinimumLongDecodeTokens || matched.Comparator.RawCompletionTokensPerRun != MinimumLongDecodeTokens || matched.Comparator.TimedTokensPerRun != MinimumLongDecodeTokens-1 {
		t.Fatalf("matched raw/timed cardinality=%+v", matched)
	}
	slow := makeNative(true)
	matched, err = FoldMatchedDecodeCampaign(slow, comparator)
	if err != nil || matched.Verdict != "HOLD" || !strings.Contains(matched.Failure, "below 0.85") {
		t.Fatalf("matched=%+v err=%v", matched, err)
	}
	tampered := makeNative(false)
	tampered[0].DecodeWindows.Windows[0].ElapsedNS++
	if _, err := FoldMatchedDecodeCampaign(tampered, comparator); err == nil || !strings.Contains(err.Error(), "readback mismatch") {
		t.Fatalf("err=%v", err)
	}
	if _, err := FoldMatchedDecodeCampaign(append(native, native[0]), comparator); err == nil || !strings.Contains(err.Error(), "exactly 3") {
		t.Fatalf("err=%v", err)
	}
	mismatchedComparator := makeComparator()
	for i := range mismatchedComparator {
		mismatchedComparator[i].FixtureID = "other-long"
	}
	if _, err := FoldMatchedDecodeCampaign(native, mismatchedComparator); err == nil || !strings.Contains(err.Error(), "fixture mismatch") {
		t.Fatalf("cross-side fixture err=%v", err)
	}
}

func cloneLlamaEvents(events []LlamaClientArrivalEvent) []LlamaClientArrivalEvent {
	out := append([]LlamaClientArrivalEvent(nil), events...)
	for i := range out {
		out[i].TokenIDs = append([]int(nil), events[i].TokenIDs...)
	}
	return out
}

func nativeLinearTrace(tokens int, step int64) DecodeTrace {
	elapsed := make([]int64, tokens)
	for i := range elapsed {
		elapsed[i] = int64(i+1) * step
	}
	return nativeTrace(elapsed)
}

func nativeTrace(elapsed []int64) DecodeTrace {
	trace := DecodeTrace{Schema: NativeDecodeTraceSchema, Engine: NativeDecodeTraceEngine, Provenance: NativeDecodeTraceProvenance}
	for i, value := range elapsed {
		trace.Events = append(trace.Events, DecodeTraceEvent{TokenIndex: i + 1, ElapsedNS: value})
	}
	return trace
}
