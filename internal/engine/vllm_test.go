package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func TestVLLMEngineIsRegisteredLifecycleDriver(t *testing.T) {
	eng := abi.Engine(VLLMEngineID)
	if eng == nil {
		t.Fatalf("engine %q is not registered", VLLMEngineID)
	}
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatalf("engine %q must implement the lifecycle seam", VLLMEngineID)
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatalf("engine %q must advertise lifecycle support", VLLMEngineID)
	}
}

func TestVLLMHTTPAdapterStreamsChatAndCompletions(t *testing.T) {
	ctx := context.Background()
	type seenRequest struct {
		path string
		body map[string]any
	}
	seen := make(chan seenRequest, 2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want Bearer test-key", got)
		}
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("request body JSON: %v", err)
		}
		seen <- seenRequest{path: r.URL.Path, body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		switch r.URL.Path {
		case "/v1/chat/completions":
			io.WriteString(w, "data: {\"model\":\"served\",\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n")
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
		case "/v1/completions":
			io.WriteString(w, "data: {\"model\":\"served\",\"choices\":[{\"text\":\"o\"}]}\n\n")
			io.WriteString(w, "data: {\"choices\":[{\"text\":\"k\",\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	e := NewVLLMEngine(VLLMConfig{
		BaseURL:  srv.URL + "/v1",
		Model:    "served",
		APIKey:   "test-key",
		WorkerID: "worker-a",
	})

	chat, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "chat",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
	})
	if err != nil {
		t.Fatalf("chat Complete: %v", err)
	}
	assertVLLMResult(t, ctx, chat, "chat", "hello", "stop", "3", "2", "5")

	comp, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "completions",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"prompt":"hi"}`)},
		Meta: map[string]string{"openai_endpoint": "completions"},
	})
	if err != nil {
		t.Fatalf("completions Complete: %v", err)
	}
	assertVLLMResult(t, ctx, comp, "completions", "ok", "length", "7", "2", "9")

	first := <-seen
	if first.path != "/v1/chat/completions" {
		t.Fatalf("first path = %s, want chat completions", first.path)
	}
	if first.body["stream"] != true {
		t.Fatalf("chat stream flag = %#v, want true", first.body["stream"])
	}
	if first.body["stream_options"] == nil {
		t.Fatalf("chat request missing stream_options: %#v", first.body)
	}
	second := <-seen
	if second.path != "/v1/completions" {
		t.Fatalf("second path = %s, want completions", second.path)
	}
	if second.body["stream"] != true || second.body["prompt"] != "hi" {
		t.Fatalf("completion body not normalized: %#v", second.body)
	}
}

func assertVLLMResult(t *testing.T, ctx context.Context, res *abi.Result, endpoint, text, finish, in, out, total string) {
	t.Helper()
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("result = %+v, want StatusOK", res)
	}
	if res.Meta["engine"] != VLLMEngineID || res.Meta["endpoint"] != endpoint || res.Meta["finish_reason"] != finish {
		t.Fatalf("unexpected result meta: %+v", res.Meta)
	}
	if res.Meta["input_tokens"] != in || res.Meta["output_tokens"] != out || res.Meta["total_tokens"] != total {
		t.Fatalf("unexpected token meta: %+v", res.Meta)
	}
	body := res.Payload.Inline
	if res.Payload.Kind != abi.RefInline {
		resolver := abi.ActiveResolver()
		if resolver == nil {
			t.Fatalf("payload was %v but ActiveResolver is nil", res.Payload.Kind)
		}
		b, err := resolver.Resolve(ctx, res.Payload)
		if err != nil {
			t.Fatalf("resolve payload: %v", err)
		}
		body = b
	}
	if !strings.Contains(string(body), `"text":"`+text+`"`) {
		t.Fatalf("payload missing assembled text %q: %s", text, body)
	}
}

func TestVLLMKVEventSubscriptionFeedsResidencyAndCacheMetrics(t *testing.T) {
	idx := NewPrefixResidencyIndex()
	rec := NewCacheEventRecorder()
	src := NewVLLMJSONKVEventSource(io.NopCloser(strings.NewReader(strings.Join([]string{
		`{"ts":1.25,"worker_id":"worker-a","model_id":"m","tokenizer_id":"tok","events":[{"type":"BlockStored","block_hashes":["h1","h2"],"token_ids":[1,2,3,4],"block_size":2,"medium":"GPU","group_idx":0},{"type":"BlockRemoved","block_hashes":["h1"],"block_size":2,"medium":"CPU","group_idx":0}]}`,
		``,
	}, "\n"))))
	e := NewVLLMEngine(VLLMConfig{
		WorkerID:      "worker-a",
		Model:         "m",
		Residency:     idx,
		CacheRecorder: rec,
		KVEvents:      src,
	})

	if err := e.RunKVEventSubscription(context.Background()); err != nil {
		t.Fatalf("RunKVEventSubscription: %v", err)
	}
	if idx.Has("worker-a", "h1") {
		t.Fatal("removed block h1 is still resident")
	}
	if !idx.Has("worker-a", "h2") {
		t.Fatal("stored block h2 was not marked resident")
	}
	rows := idx.Snapshot("worker-a")
	if len(rows) != 1 || rows[0].ModelID != "m" || rows[0].TokenizerID != "tok" || rows[0].Tokens != 2 {
		t.Fatalf("residency row not normalized: %+v", rows)
	}
	snap := rec.Metrics().Snapshot()
	if snap.Events != 3 || snap.Hits != 3 {
		t.Fatalf("cache event metrics not fed by KV events: %+v", snap)
	}
}

func TestVLLMKVEventVisibilityConsolidatesSourcesThroughFinalRemove(t *testing.T) {
	rec := NewCacheEventRecorder()
	idx := NewPrefixResidencyIndex()
	batch := func(source, eventID string, sequence uint64, typ string) VLLMKVEventBatch {
		return VLLMKVEventBatch{
			TS:       float64(sequence),
			SourceID: source,
			EventID:  eventID,
			Sequence: sequence,
			Events: []VLLMKVEvent{{
				Type:        typ,
				BlockHashes: []any{"shared-block"},
				BlockSize:   16,
				Medium:      "GPU",
			}},
		}
	}

	storeA := RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, batch("rank-0", "a/store", 1, "BlockStored"))
	storeB := RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, batch("rank-1", "b/store", 1, "BlockStored"))
	removeA := RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, batch("rank-0", "a/remove", 2, "BlockRemoved"))
	removeB := RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, batch("rank-1", "b/remove", 2, "BlockRemoved"))
	if len(storeA) != 1 || len(storeB) != 1 || len(removeA) != 1 || len(removeB) != 1 {
		t.Fatalf("one block must lower to one decision: %d/%d/%d/%d", len(storeA), len(storeB), len(removeA), len(removeB))
	}
	if !storeA[0].Published || storeB[0].Published || removeA[0].Published || !removeB[0].Published {
		t.Fatalf("vLLM visibility decisions = storeA:%+v storeB:%+v removeA:%+v removeB:%+v", storeA[0], storeB[0], removeA[0], removeB[0])
	}
	if storeA[0].Entry.Labels["source_id"] != "rank-0" || storeA[0].Entry.Labels["event_id"] == "" {
		t.Fatalf("stable source/event identity was not attached: %+v", storeA[0].Entry.Labels)
	}
	if storeA[0].Entry.Labels["logical_block_key_version"] != cachemeta.CacheLogicalBlockKeyVersion {
		t.Fatalf("logical block key was not versioned: %+v", storeA[0].Entry.Labels)
	}
	if snap := rec.Metrics().Snapshot(); snap.Events != 2 || snap.SuppressedProducer != 1 || snap.SuppressedRemove != 1 {
		t.Fatalf("real vLLM visibility path did not consolidate: %+v", snap)
	}
}

func TestVLLMAllBlocksClearedWaitsForFinalSource(t *testing.T) {
	rec := NewCacheEventRecorder()
	idx := NewPrefixResidencyIndex()
	store := func(source string) VLLMKVEventBatch {
		return VLLMKVEventBatch{
			SourceID: source,
			EventID:  source + "/store",
			Sequence: 1,
			Events:   []VLLMKVEvent{{Type: "BlockStored", BlockHashes: []any{"shared-block"}, BlockSize: 16}},
		}
	}
	clear := func(source string) VLLMKVEventBatch {
		return VLLMKVEventBatch{
			SourceID: source,
			EventID:  source + "/clear",
			Sequence: 2,
			Events:   []VLLMKVEvent{{Type: "AllBlocksCleared"}},
		}
	}
	RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, store("rank-0"))
	RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, store("rank-1"))
	first := RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, clear("rank-0"))
	final := RecordVLLMKVEventBatch("worker", "Qwen3.8", "qwen3", idx, rec, clear("rank-1"))
	if len(first) != 1 || first[0].Published || first[0].Suppression != cachemeta.CacheEventSourceStillResident {
		t.Fatalf("first source clear = %+v, want suppressed remove", first)
	}
	if len(final) != 1 || !final[0].Published || final[0].Entry.Labels["visibility_action"] != "remove" {
		t.Fatalf("final source clear = %+v, want published REMOVE", final)
	}
}

func TestVLLMPrometheusNormalization(t *testing.T) {
	snap := ParseVLLMPrometheus("worker-a", `
vllm:time_to_first_token_seconds_sum{model_name="m"} 1.5
vllm:time_to_first_token_seconds_count{model_name="m"} 3
vllm:request_time_per_output_token_seconds_sum 2.5
vllm:request_time_per_output_token_seconds_count 5
vllm:inter_token_latency_seconds_sum 0.25
vllm:inter_token_latency_seconds_count 4
vllm:request_queue_time_seconds_sum 0.75
vllm:request_queue_time_seconds_count 6
vllm:kv_cache_usage_perc 0.8
vllm:num_requests_running 2
vllm:num_requests_waiting 1
vllm:num_requests_swapped 0
vllm:request_success_total 9
vllm:prefix_cache_queries 11
vllm:prefix_cache_hits 7
`)
	if snap.TTFT.Sum != 1.5 || snap.TTFT.Count != 3 || snap.TPOT.Sum != 2.5 || snap.TPOT.Count != 5 {
		t.Fatalf("serving latency metrics not normalized: %+v", snap)
	}
	if snap.KVCacheUsage != 0.8 || snap.RequestsRunning != 2 || snap.PrefixQueries != 11 || snap.PrefixHits != 7 {
		t.Fatalf("serving gauges/counters not normalized: %+v", snap)
	}
	prom := snap.Prometheus()
	for _, want := range []string{
		`fak_serving_ttft_seconds_sum{engine="vllm",worker="worker-a"} 1.5`,
		`fak_serving_tpot_seconds_count{engine="vllm",worker="worker-a"} 5`,
		`fak_serving_kv_cache_usage_ratio{engine="vllm",worker="worker-a"} 0.8`,
		`fak_serving_prefix_cache_hits_total{engine="vllm",worker="worker-a"} 7`,
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("Prometheus output missing %q:\n%s", want, prom)
		}
	}
}

// joinURLErr lets the JoinEndpoint test supply its own invalid-URL wording without
// pulling a new import into this file -- the point of the parameter is that the
// CALLER owns that sentence (the vLLM adapter and `fak llmd-smoke` word it
// differently and both wordings are load-bearing for their operators).
type joinURLErr string

func (e joinURLErr) Error() string { return string(e) }

func TestJoinEndpointNormalizesPathQueryAndFragment(t *testing.T) {
	bad := func(b string) error { return joinURLErr("invalid " + b) }

	got, err := JoinEndpoint("https://example.invalid:8000/v1/", "/models", bad)
	if err != nil {
		t.Fatalf("JoinEndpoint(trailing slash) error: %v", err)
	}
	if got != "https://example.invalid:8000/v1/models" {
		t.Fatalf("JoinEndpoint(trailing slash) = %q, want the slash collapsed", got)
	}

	got, err = JoinEndpoint("  https://example.invalid:8000/v1?key=v#frag  ", "/chat/completions", bad)
	if err != nil {
		t.Fatalf("JoinEndpoint(query+fragment) error: %v", err)
	}
	if got != "https://example.invalid:8000/v1/chat/completions" {
		t.Fatalf("JoinEndpoint(query+fragment) = %q, want query and fragment dropped", got)
	}

	if _, err := JoinEndpoint("example.invalid/v1", "/models", bad); err == nil {
		t.Fatalf("JoinEndpoint(no scheme) must refuse")
	} else if err.Error() != "invalid example.invalid/v1" {
		t.Fatalf("JoinEndpoint refusal = %q, want the caller-supplied wording", err.Error())
	}
}

func TestVLLMKVEventPinnedTaggedArrayPreservesIdentityAndLowers(t *testing.T) {
	fixture, err := os.ReadFile("testdata/vllm_kv_events/block_stored_v0.17.1.json")
	if err != nil {
		t.Fatal(err)
	}
	src := NewVLLMJSONKVEventSource(io.NopCloser(strings.NewReader(string(fixture) + "\n")))
	batch, err := src.Next(context.Background())
	if err != nil {
		t.Fatalf("decode upstream-derived fixture: %v", err)
	}
	if batch.DataParallelRank == nil || *batch.DataParallelRank != 3 || batch.Raw["encoding"] != "msgspec-array" {
		t.Fatalf("native batch boundary was not retained: %+v", batch)
	}
	if len(batch.Events) != 1 {
		t.Fatalf("got %d events, want 1", len(batch.Events))
	}
	ev := batch.Events[0]
	wantExtra := `[["mm","image-1"],null,["salt",42]]`
	if string(ev.ExtraKeys) != wantExtra {
		t.Fatalf("extra_keys = %s, want %s", ev.ExtraKeys, wantExtra)
	}
	if ev.LoraID == nil || *ev.LoraID != 7 || ev.LoraName != "adapter" || ev.ParentBlockHash != "parent" {
		t.Fatalf("BlockStored identity fields were not preserved: %+v", ev)
	}

	idx := NewPrefixResidencyIndex()
	rec := NewCacheEventRecorder()
	got := RecordVLLMKVEventBatch("worker-a", "model-a", "tokenizer-a", idx, rec, batch)
	if len(got) != 2 || !idx.Has("worker-a", "h1") || !idx.Has("worker-a", "h2") {
		t.Fatalf("native event was not lowered into two resident blocks: results=%+v rows=%+v", got, idx.Snapshot("worker-a"))
	}
	rows := idx.Snapshot("worker-a")
	if len(rows) != 2 || rows[0].ModelID != "model-a" || rows[0].TokenizerID != "tokenizer-a" {
		t.Fatalf("residency scope was not retained: %+v", rows)
	}
	if snap := rec.Metrics().Snapshot(); snap.Events != 2 || snap.Hits != 2 {
		t.Fatalf("cache-create telemetry not emitted: %+v", snap)
	}
}

func TestVLLMKVEventPinnedFieldOrderRejectsDrift(t *testing.T) {
	var batch VLLMKVEventBatch
	err := json.Unmarshal([]byte(`[1,[["BlockStored",["h"],null,[1],1,null,"GPU","",null,"unexpected"]],null]`), &batch)
	if err == nil || !strings.Contains(err.Error(), "want 8 or 9") {
		t.Fatalf("schema drift was not rejected: %v", err)
	}
}

func TestVLLMKVEventProvenancePinsUpstreamRevision(t *testing.T) {
	data, err := os.ReadFile("testdata/vllm_kv_events/upstream_provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"revision": "95c0f928cdeeaa21c4906e73cee6a156e1b3b995"`,
		`"license": "Apache-2.0"`,
		`"source_path": "vllm/distributed/kv_events.py"`,
		`"BlockStored_fields"`,
		`"extra_keys"`,
	} {
		if !strings.Contains(string(data), want) {
			t.Fatalf("provenance missing %s: %s", want, data)
		}
	}
}
