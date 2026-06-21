package gateway

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// This file proves the issue #21 OpenAI-API parity acceptance for the gateway:
// (1) an OpenAI client shape round-trips on /v1/embeddings and /v1/moderations,
// (2) /v1/chat/completions streaming emits incremental SSE chunks, and
// (3) batching (array input) returns per-item results. The structs below are
// minimal independent mirrors of the OpenAI SDK response shapes — decoding the
// gateway's bytes into them is the round-trip proof (we do NOT lean on the
// gateway's own DTOs to decode).

type oaiEmbeddingsResp struct {
	Object string `json:"object"`
	Model  string `json:"model"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// embeddings with encoding_format=base64 — the `embedding` field is a string.
type oaiEmbeddingsB64Resp struct {
	Data []struct {
		Index     int    `json:"index"`
		Embedding string `json:"embedding"`
	} `json:"data"`
}

type oaiModerationsResp struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Results []struct {
		Flagged        bool               `json:"flagged"`
		Categories     map[string]bool    `json:"categories"`
		CategoryScores map[string]float64 `json:"category_scores"`
	} `json:"results"`
}

// ---------------------------------------------------------------------------
// /v1/embeddings — OpenAI client shape round-trips + determinism + batching.
// ---------------------------------------------------------------------------

func TestEmbeddingsRoundTripsOpenAIShape(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp oaiEmbeddingsResp
	code := postJSON(t, ts.URL+"/v1/embeddings", map[string]any{
		"model": "text-embedding-3-small",
		"input": "the quick brown fox",
	}, &resp)
	if code != 200 {
		t.Fatalf("status = %d, want 200", code)
	}
	if resp.Object != "list" {
		t.Errorf("object = %q, want list", resp.Object)
	}
	if resp.Model != "test-model" {
		t.Errorf("model = %q, want the gateway model", resp.Model)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("data items = %d, want 1", len(resp.Data))
	}
	d := resp.Data[0]
	if d.Object != "embedding" || d.Index != 0 {
		t.Errorf("data[0] object/index = %q/%d, want embedding/0", d.Object, d.Index)
	}
	if len(d.Embedding) != embedDefaultDim {
		t.Errorf("embedding dim = %d, want %d", len(d.Embedding), embedDefaultDim)
	}
	// A real (L2-normalized) embedding has unit norm.
	if norm := l2norm(d.Embedding); math.Abs(norm-1.0) > 1e-9 {
		t.Errorf("embedding L2 norm = %v, want 1.0 (normalized)", norm)
	}
	if resp.Usage.PromptTokens != 4 || resp.Usage.TotalTokens != 4 {
		t.Errorf("usage = %+v, want 4 prompt/total tokens (the 4 words)", resp.Usage)
	}
}

func TestEmbeddingsCustomDimensionsAndBase64(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Float form at a custom dimension.
	var fl oaiEmbeddingsResp
	postJSON(t, ts.URL+"/v1/embeddings", map[string]any{
		"input": "hello world", "dimensions": 64,
	}, &fl)
	if len(fl.Data) != 1 || len(fl.Data[0].Embedding) != 64 {
		t.Fatalf("custom dimensions not honored: %+v", fl.Data)
	}

	// base64 form at the same dimension must decode to the identical float vector.
	var b64 oaiEmbeddingsB64Resp
	postJSON(t, ts.URL+"/v1/embeddings", map[string]any{
		"input": "hello world", "dimensions": 64, "encoding_format": "base64",
	}, &b64)
	if len(b64.Data) != 1 {
		t.Fatalf("base64 data items = %d, want 1", len(b64.Data))
	}
	raw, err := base64.StdEncoding.DecodeString(b64.Data[0].Embedding)
	if err != nil {
		t.Fatalf("base64 decode: %v", err)
	}
	if len(raw) != 64*4 {
		t.Fatalf("base64 byte len = %d, want %d (float32 per component)", len(raw), 64*4)
	}
	for i := 0; i < 64; i++ {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.Abs(float64(got)-fl.Data[0].Embedding[i]) > 1e-6 {
			t.Fatalf("base64 component %d = %v, want %v (float form)", i, got, fl.Data[0].Embedding[i])
		}
	}
}

// TestEmbeddingsAreDeterministicAndContentSensitive proves the backend is a real,
// honest embedding: identical text -> identical vector (cacheable/testable), and
// token-overlapping texts score higher cosine similarity than disjoint ones
// (useful as a drop-in). This is the substance behind "OpenAI client compatible".
func TestEmbeddingsAreDeterministicAndContentSensitive(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	embed := func(text string) []float64 {
		var r oaiEmbeddingsResp
		if code := postJSON(t, ts.URL+"/v1/embeddings", map[string]any{"input": text}, &r); code != 200 {
			t.Fatalf("embed %q: status %d", text, code)
		}
		return r.Data[0].Embedding
	}

	a1 := embed("the quick brown fox")
	a2 := embed("the quick brown fox")
	similar := embed("the quick brown dog")
	disjoint := embed("zebra mango xylophone parsec")

	if cosine(a1, a2) < 1.0-1e-9 {
		t.Errorf("identical text must embed identically, cosine = %v", cosine(a1, a2))
	}
	simScore := cosine(a1, similar)
	disScore := cosine(a1, disjoint)
	if simScore <= disScore {
		t.Errorf("token-overlap similarity (%v) must exceed disjoint similarity (%v)", simScore, disScore)
	}
	if math.Abs(disScore) > 1e-9 {
		t.Errorf("fully disjoint inputs should be ~orthogonal, cosine = %v", disScore)
	}
}

// TestEmbeddingsBatchReturnsPerItemResults is the embeddings half of the batching
// acceptance: an array `input` returns one indexed vector per item, identical
// items embed identically, and distinct items differ — exactly OpenAI's batch
// embedding contract.
func TestEmbeddingsBatchReturnsPerItemResults(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp oaiEmbeddingsResp
	code := postJSON(t, ts.URL+"/v1/embeddings", map[string]any{
		"input": []string{"alpha beta", "gamma delta", "alpha beta"},
	}, &resp)
	if code != 200 {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(resp.Data) != 3 {
		t.Fatalf("batch returned %d items, want 3 (per-item results)", len(resp.Data))
	}
	for i, d := range resp.Data {
		if d.Index != i {
			t.Errorf("data[%d].index = %d, want %d (request order preserved)", i, d.Index, i)
		}
	}
	// item 0 and item 2 are the same text -> identical vectors; item 1 differs.
	if cosine(resp.Data[0].Embedding, resp.Data[2].Embedding) < 1.0-1e-9 {
		t.Error("identical batch items must embed identically")
	}
	if cosine(resp.Data[0].Embedding, resp.Data[1].Embedding) >= 1.0-1e-9 {
		t.Error("distinct batch items must embed differently")
	}
	if resp.Usage.PromptTokens != 6 {
		t.Errorf("batch usage prompt_tokens = %d, want 6 (2 tokens x 3 items)", resp.Usage.PromptTokens)
	}
}

func TestEmbeddingsTokenIDArrayInput(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// A pre-tokenized batch ([][]int) is a valid OpenAI input shape; the gateway
	// must return one deterministic vector per token-id array.
	var resp oaiEmbeddingsResp
	code := postJSON(t, ts.URL+"/v1/embeddings", map[string]any{
		"input": [][]int{{1, 2, 3}, {4, 5, 6}},
	}, &resp)
	if code != 200 {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("token-id batch returned %d items, want 2", len(resp.Data))
	}
}

func TestEmbeddingsRejectsEmptyInput(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code := postJSON(t, ts.URL+"/v1/embeddings", map[string]any{"model": "m"}, nil)
	if code != 400 {
		t.Errorf("missing input must be 400, got %d", code)
	}
	code = postJSON(t, ts.URL+"/v1/embeddings", map[string]any{"input": "x", "encoding_format": "weird"}, nil)
	if code != 400 {
		t.Errorf("unknown encoding_format must be 400, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// /v1/moderations — OpenAI client shape round-trips + per-item batching.
// ---------------------------------------------------------------------------

func TestModerationsRoundTripsAndFlags(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp oaiModerationsResp
	code := postJSON(t, ts.URL+"/v1/moderations", map[string]any{
		"input": "I love sunny days and gardening",
	}, &resp)
	if code != 200 {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("results = %d, want 1", len(resp.Results))
	}
	if resp.Results[0].Flagged {
		t.Errorf("benign input must not be flagged: %+v", resp.Results[0])
	}
	// The full OpenAI category vocabulary must be present (a client keying on a
	// specific category never reads a missing field).
	if len(resp.Results[0].Categories) != len(moderationLexicon) {
		t.Errorf("categories map has %d keys, want %d (full vocabulary)", len(resp.Results[0].Categories), len(moderationLexicon))
	}
	for _, key := range moderationCategoryKeys() {
		if _, ok := resp.Results[0].CategoryScores[key]; !ok {
			t.Errorf("category_scores missing %q", key)
		}
	}

	// A harmful input must flag the right category.
	var harmful oaiModerationsResp
	postJSON(t, ts.URL+"/v1/moderations", map[string]any{
		"input": "I will kill him and shoot them",
	}, &harmful)
	if !harmful.Results[0].Flagged {
		t.Fatalf("violent input must be flagged: %+v", harmful.Results[0])
	}
	if !harmful.Results[0].Categories["violence"] {
		t.Errorf("violence category must be set: %+v", harmful.Results[0].Categories)
	}
}

// TestModerationsBatchReturnsPerItemResults is the moderations half of the
// batching acceptance: an array `input` returns one classification per item, in
// order.
func TestModerationsBatchReturnsPerItemResults(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp oaiModerationsResp
	code := postJSON(t, ts.URL+"/v1/moderations", map[string]any{
		"input": []string{"a peaceful walk in the park", "i will kill him and shoot them"},
	}, &resp)
	if code != 200 {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("batch returned %d results, want 2 (per-item)", len(resp.Results))
	}
	if resp.Results[0].Flagged {
		t.Errorf("item 0 (benign) must not be flagged: %+v", resp.Results[0])
	}
	if !resp.Results[1].Flagged {
		t.Errorf("item 1 (violent) must be flagged: %+v", resp.Results[1])
	}
}

func TestModerationsRejectsEmptyInput(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	if code := postJSON(t, ts.URL+"/v1/moderations", map[string]any{"model": "m"}, nil); code != 400 {
		t.Errorf("missing input must be 400, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — streaming emits incremental SSE chunks.
// ---------------------------------------------------------------------------

// TestChatCompletionsStreamingEmitsIncrementalSSEChunks proves the streaming
// surface independently of any upstream: a stub planner yields a content + a kept
// tool call + a denied tool call, and the gateway emits an SSE stream whose
// opening delta carries the (adjudicated) content/tool calls, whose terminal
// chunk carries finish_reason + usage, and which ends with the [DONE] sentinel.
// The denied call never reaches the streamed delta.
func TestChatCompletionsStreamingEmitsIncrementalSSEChunks(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "checking now", ToolCalls: []agent.ToolCall{
			{ID: "c1", Type: "function", Function: agent.Func{Name: "allow_a", Arguments: `{"x":1}`}},
			{ID: "c2", Type: "function", Function: agent.Func{Name: "deny_b", Arguments: `{}`}},
		}},
		FinishReason: "tool_calls",
		Usage:        agent.Usage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(ChatRequest{
		Model:    "test-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "go"}},
		Stream:   true,
	})
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	if ct := httpResp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}
	raw, _ := io.ReadAll(httpResp.Body)

	chunks, sawDone := parseSSEChunks(t, raw)
	if !sawDone {
		t.Fatalf("stream missing [DONE] sentinel:\n%s", raw)
	}
	if len(chunks) < 2 {
		t.Fatalf("want >=2 incremental chunks (an opening delta then a terminal finish), got %d:\n%s", len(chunks), raw)
	}
	for _, c := range chunks {
		if c.Object != "chat.completion.chunk" {
			t.Errorf("chunk object = %q, want chat.completion.chunk", c.Object)
		}
	}

	// The opening chunk announces the assistant role and the surviving (allow)
	// tool call, and carries no finish_reason.
	first := chunks[0]
	if first.Choices[0].Delta.Role != agent.RoleAssistant {
		t.Errorf("opening delta role = %q, want %q", first.Choices[0].Delta.Role, agent.RoleAssistant)
	}
	if first.Choices[0].FinishReason != nil {
		t.Errorf("opening chunk must not carry finish_reason, got %v", *first.Choices[0].FinishReason)
	}

	// Content arrives INCREMENTALLY: the multi-word reply is split across more than
	// one content-bearing chunk (not collapsed into a single delta), the denied
	// tool call never reaches any delta, the allow call survives, and concatenating
	// every content delta in order reproduces the assistant content exactly.
	var reassembled strings.Builder
	contentChunks := 0
	var sawAllow bool
	for _, c := range chunks {
		if seg := c.Choices[0].Delta.Content; seg != "" {
			reassembled.WriteString(seg)
			contentChunks++
		}
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.Function.Name == "deny_b" {
				t.Error("denied tool call must not reach the streamed delta")
			}
			if tc.Function.Name == "allow_a" {
				sawAllow = true
			}
		}
	}
	if !sawAllow {
		t.Error("streamed deltas missing the surviving allow_a tool call")
	}
	if got := reassembled.String(); got != "checking now" {
		t.Errorf("reassembled stream content = %q, want %q", got, "checking now")
	}
	if contentChunks < 2 {
		t.Errorf("want content split across >=2 incremental chunks, got %d", contentChunks)
	}

	// The terminal chunk carries finish_reason + usage.
	last := chunks[len(chunks)-1]
	if last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("terminal finish_reason = %v, want tool_calls", last.Choices[0].FinishReason)
	}
	if last.Usage == nil || last.Usage.TotalTokens != 12 {
		t.Errorf("terminal usage = %+v, want total 12", last.Usage)
	}
}

// TestSegmentContentIsIncrementalAndLossless table-proves the streaming
// segmentation that backs the incremental SSE content deltas: a multi-word reply
// splits into more than one fragment, a single token stays one fragment, an empty
// reply (a pure tool-call turn) yields no fragments, and — crucially — for every
// case the concatenation of the fragments reproduces the input byte-for-byte, so
// the wire stream never drops or reorders content.
func TestSegmentContentIsIncrementalAndLossless(t *testing.T) {
	cases := []struct {
		name     string
		in       string
		wantSegs int // exact fragment count
	}{
		{"empty", "", 0},
		{"single token", "hello", 1},
		{"two words", "checking now", 2},
		{"many words", "the quick brown fox jumps", 5},
		{"trailing space", "done ", 1},
		{"double inner space", "a  b", 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			segs := segmentContent(tc.in)
			if len(segs) != tc.wantSegs {
				t.Errorf("segmentContent(%q) gave %d fragments, want %d: %q", tc.in, len(segs), tc.wantSegs, segs)
			}
			if got := strings.Join(segs, ""); got != tc.in {
				t.Errorf("reassembled %q, want %q (lossless concat)", got, tc.in)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func parseSSEChunks(t *testing.T, raw []byte) (chunks []ChatStreamResponse, sawDone bool) {
	t.Helper()
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			t.Fatalf("non-SSE line: %q", line)
		}
		if data == "[DONE]" {
			sawDone = true
			continue
		}
		var c ChatStreamResponse
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatalf("decode SSE chunk: %v (%s)", err, data)
		}
		chunks = append(chunks, c)
	}
	return chunks, sawDone
}

func l2norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func cosine(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.NaN()
	}
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}
