package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/stepspan"
)

func TestOTLPExporterCollectorPayloadAndDrain(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	var payloads []map[string]any
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var p map[string]any
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			t.Error(err)
		}
		mu.Lock()
		paths = append(paths, r.URL.Path)
		payloads = append(payloads, p)
		mu.Unlock()
		w.WriteHeader(200)
	}))
	defer collector.Close()
	e, err := newOTLPExporter(collector.URL, 4, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	e.enqueue(otlpSpan{TraceID: "4bf92f3577b34da6a3ce929d0e0e4736", SpanID: "00f067aa0ba902b7", Name: "POST /v1/chat/completions", Route: "/v1/chat/completions", Method: "POST", Status: 200, Start: time.Unix(1, 0), End: time.Unix(1, 100)})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := e.close(ctx); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(payloads) != 1 || paths[0] != "/v1/traces" {
		t.Fatalf("paths=%v payloads=%v", paths, payloads)
	}
	body, _ := json.Marshal(payloads[0])
	text := string(body)
	for _, want := range []string{"4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", "http.route", "/v1/chat/completions", "fak-gateway"} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q: %s", want, text)
		}
	}
	st := e.stats()
	if st.Accepted != 1 || st.Exported != 1 || st.Failed != 0 {
		t.Fatalf("stats=%+v", st)
	}
}
func TestOTLPExporterUnavailableNeverBlocksAdmission(t *testing.T) {
	e, err := newOTLPExporter("http://127.0.0.1:1", 1, 20*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	for i := 0; i < 100; i++ {
		e.enqueue(otlpSpan{})
	}
	if time.Since(start) > 100*time.Millisecond {
		t.Fatal("enqueue blocked")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = e.close(ctx)
	st := e.stats()
	if st.Accepted < 1 || st.Dropped < 1 || st.Failed < 1 {
		t.Fatalf("stats=%+v", st)
	}
}
func TestOTLPExporterConfigFailClosed(t *testing.T) {
	for _, endpoint := range []string{"file:///tmp/x", "relative", "ftp://x"} {
		if _, err := newOTLPExporter(endpoint, 1, time.Second); err == nil {
			t.Fatalf("accepted %s", endpoint)
		}
	}
	if e, err := newOTLPExporter("", 256, time.Second); err != nil || e != nil {
		t.Fatalf("disabled=%v err=%v", e, err)
	}
}

func TestOTLPExporterConcurrentEnqueueAndCloseRaceSafe(t *testing.T) {
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}))
	defer collector.Close()

	for iter := 0; iter < 50; iter++ {
		e, err := newOTLPExporter(collector.URL, 16, time.Second)
		if err != nil {
			t.Fatal(err)
		}

		var wg sync.WaitGroup
		// Launch concurrent enqueuers
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < 50; j++ {
					e.enqueue(otlpSpan{TraceID: "test-trace", SpanID: "test-span", Name: "GET /"})
				}
			}()
		}

		// Launch concurrent closers (asserting idempotency and race safety)
		for i := 0; i < 3; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				_ = e.close(ctx)
			}()
		}

		wg.Wait()
		// Late enqueue after close should drop cleanly, never panic on closed channel
		e.enqueue(otlpSpan{TraceID: "late-trace"})
	}
}

func exportGenAIOnce(t *testing.T, span otlpSpan) map[string]any {
	t.Helper()
	var mu sync.Mutex
	var bodies [][]byte
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var buf strings.Builder
		if _, err := io.Copy(&buf, r.Body); err != nil {
			t.Error(err)
		}
		mu.Lock()
		bodies = append(bodies, []byte(buf.String()))
		mu.Unlock()
		w.WriteHeader(200)
	}))
	defer collector.Close()
	e, err := newOTLPExporter(collector.URL, 4, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	e.enqueue(span)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := e.close(ctx); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 1 {
		t.Fatalf("posted %d bodies, stats=%+v", len(bodies), e.stats())
	}
	var p struct {
		ResourceSpans []struct {
			ScopeSpans []struct {
				Spans []map[string]any `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	if err := json.Unmarshal(bodies[0], &p); err != nil {
		t.Fatal(err)
	}
	return p.ResourceSpans[0].ScopeSpans[0].Spans[0]
}

func otlpGenAIStringAttrs(t *testing.T, span map[string]any) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, raw := range span["attributes"].([]any) {
		a := raw.(map[string]any)
		v := a["value"].(map[string]any)
		if sv, ok := v["stringValue"].(string); ok {
			out[a["key"].(string)] = sv
		} else {
			out[a["key"].(string)] = "int:" + v["intValue"].(string)
		}
	}
	return out
}

func genAIBaseSpan() otlpSpan {
	return otlpSpan{TraceID: "4bf92f3577b34da6a3ce929d0e0e4736", SpanID: "00f067aa0ba902b7", Name: "POST /v1/chat/completions", Route: "/v1/chat/completions", Method: "POST", Status: 200, Start: time.Unix(1, 0), End: time.Unix(1, 100)}
}

func TestOTLPExportGenAIAttributesArriveInPostedJSON(t *testing.T) {
	g, ok := stepspan.GenAI(stepspan.Span{Plane: stepspan.Agent, Kind: stepspan.ToolCall, Status: stepspan.OK, Attrs: stepspan.Attrs{Tool: "Flights"}, IDs: stepspan.IDs{CallID: "call_9", SessionID: "conv_1"}}, stepspan.GenAIOptions{})
	if !ok {
		t.Fatal("tool call span not mapped")
	}
	span := genAIBaseSpan()
	span.GenAI = &g
	got := exportGenAIOnce(t, span)
	if got["name"] != "execute_tool Flights" {
		t.Fatalf("name=%v", got["name"])
	}
	want := map[string]string{
		"http.request.method":       "POST",
		"http.route":                "/v1/chat/completions",
		"http.response.status_code": "int:200",
		"gen_ai.operation.name":     "execute_tool",
		"gen_ai.tool.name":          "Flights",
		"gen_ai.tool.call.id":       "call_9",
		"gen_ai.conversation.id":    "conv_1",
	}
	if attrs := otlpGenAIStringAttrs(t, got); !reflect.DeepEqual(attrs, want) {
		t.Fatalf("attrs=%v want %v", attrs, want)
	}
}

func TestOTLPExportGenAIAbsentLeavesSpanUnchanged(t *testing.T) {
	got := exportGenAIOnce(t, genAIBaseSpan())
	if got["name"] != "POST /v1/chat/completions" {
		t.Fatalf("name=%v", got["name"])
	}
	want := map[string]string{"http.request.method": "POST", "http.route": "/v1/chat/completions", "http.response.status_code": "int:200"}
	if attrs := otlpGenAIStringAttrs(t, got); !reflect.DeepEqual(attrs, want) {
		t.Fatalf("attrs=%v", attrs)
	}
	plain, err := json.Marshal(genAIBaseSpan())
	if err != nil || strings.Contains(string(plain), "GenAI") {
		t.Fatalf("privacy payload changed without GenAI: %s err=%v", plain, err)
	}
}

func TestOTLPExportGenAINeverPostsContentAttributes(t *testing.T) {
	span := genAIBaseSpan()
	span.GenAI = &stepspan.GenAISpan{Name: "generate_content m", Attrs: []stepspan.GenAIAttr{
		{Key: stepspan.GenAIOperationName, Value: stepspan.GenAIOpGenerateContent},
		{Key: "gen_ai.input.messages", Value: "hello"},
		{Key: "gen_ai.output.messages", Value: "world"},
		{Key: "gen_ai.system_instructions", Value: "be brief"},
		{Key: "gen_ai.tool.call.arguments", Value: "{}"},
		{Key: "gen_ai.tool.call.result", Value: "{}"},
	}}
	attrs := otlpGenAIStringAttrs(t, exportGenAIOnce(t, span))
	if attrs["gen_ai.operation.name"] != "generate_content" {
		t.Fatalf("attrs=%v", attrs)
	}
	for k := range attrs {
		if stepspan.IsGenAIContentKey(k) {
			t.Fatalf("content attribute %q posted: %v", k, attrs)
		}
	}
	if len(attrs) != 4 {
		t.Fatalf("attrs=%v", attrs)
	}
}

func TestOTLPExportOpenInferenceAttrs(t *testing.T) {
	span := genAIBaseSpan()
	span.OpenInference = &OpenInferenceSpan{Kind: OpenInferenceKindLLM, ModelName: "qwen3.8", PromptTokens: 10, CompletionTokens: 5}
	attrs := otlpGenAIStringAttrs(t, exportGenAIOnce(t, span))
	want := map[string]string{
		"http.request.method": "POST", "http.route": "/v1/chat/completions", "http.response.status_code": "int:200",
		"openinference.span.kind": "LLM", "llm.model_name": "qwen3.8",
		"llm.token_count.prompt": "int:10", "llm.token_count.completion": "int:5", "llm.token_count.total": "int:15",
	}
	if !reflect.DeepEqual(attrs, want) {
		t.Fatalf("attrs=%v want %v", attrs, want)
	}
	bad := genAIBaseSpan()
	bad.OpenInference = &OpenInferenceSpan{Kind: "BOGUS", ModelName: "m"}
	if got := otlpGenAIStringAttrs(t, exportGenAIOnce(t, bad)); len(got) != 3 {
		t.Fatalf("unknown kind must emit nothing: %v", got)
	}
}

func TestOTLPExportToolCallAttrs(t *testing.T) {
	span := genAIBaseSpan()
	span.OpenInference = &OpenInferenceSpan{Kind: OpenInferenceKindTool, ToolName: "Flights", ToolCallID: "call_9", ToolCallArguments: `{"q":"x"}`, ToolCallResult: "ok"}
	attrs := otlpGenAIStringAttrs(t, exportGenAIOnce(t, span))
	if attrs["tool.name"] != "Flights" || attrs["tool.call.id"] != "call_9" || attrs["openinference.span.kind"] != "TOOL" {
		t.Fatalf("attrs=%v", attrs)
	}
	if _, ok := attrs["tool.call.arguments"]; ok {
		t.Fatalf("content exported without opt-in: %v", attrs)
	}
	if _, ok := attrs["tool.call.result"]; ok {
		t.Fatalf("content exported without opt-in: %v", attrs)
	}
	span.OpenInference.IncludeContent = true
	attrs = otlpGenAIStringAttrs(t, exportGenAIOnce(t, span))
	if attrs["tool.call.arguments"] != `{"q":"x"}` || attrs["tool.call.result"] != "ok" {
		t.Fatalf("opt-in content missing: %v", attrs)
	}
}
