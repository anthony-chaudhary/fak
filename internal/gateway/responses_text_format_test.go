package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

const responsesTextFormatSchema = `{"type":"object","properties":{"answer":{"type":"string","enum":["a","b"]}},"required":["answer"],"additionalProperties":false}`

type responsesTextFormatPlanner struct {
	mu     sync.Mutex
	calls  int
	format json.RawMessage
	tools  []string
}

func (p *responsesTextFormatPlanner) record(tools []agent.ToolDef, opts []agent.SampleOpt) {
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	names := []string{}
	for _, tool := range tools {
		names = append(names, tool.Function.Name)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	p.format = sp.ResponseFormat
	p.tools = names
}

func (p *responsesTextFormatPlanner) Complete(_ context.Context, _ []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(tools, opts)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: `{"answer":"a"}`}, FinishReason: "stop"}, nil
}

func (*responsesTextFormatPlanner) Model() string { return "text-format" }

type responsesTextFormatStreamingPlanner struct{ responsesTextFormatPlanner }

func (*responsesTextFormatStreamingPlanner) StreamingSupported() bool { return true }

func (p *responsesTextFormatStreamingPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(tools, opts)
	if err := sink(`{"answer":"a"}`); err != nil {
		return nil, err
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: `{"answer":"a"}`}, FinishReason: "stop"}, nil
}

func postResponsesTextFormat(t *testing.T, srv *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

func responsesTextFormatBody(model, text, tools string, stream bool) string {
	body := `{"model":"` + model + `","input":"answer with a or b","max_output_tokens":32,"temperature":0`
	if text != "" {
		body += `,"text":` + text
	}
	if tools != "" {
		body += `,"tools":` + tools
	}
	if stream {
		body += `,"stream":true`
	}
	return body + `}`
}

// fak-test:runtime fast est=120ms lane=default
func TestResponsesTextFormatReachesPlannerAsChatResponseFormat(t *testing.T) {
	jsonSchema := `{"format":{"type":"json_schema","name":"answer","strict":true,"schema":` + responsesTextFormatSchema + `}}`
	chatJSONSchema := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":` + responsesTextFormatSchema + `}}`)
	for _, stream := range []bool{false, true} {
		name := "buffered"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			srv := newTestServer(t)
			planner := &responsesTextFormatStreamingPlanner{}
			srv.planner = planner
			if !stream {
				srv.planner = &planner.responsesTextFormatPlanner
			}
			serve := func(text, clientTools string) (json.RawMessage, []string) {
				t.Helper()
				rr := postResponsesTextFormat(t, srv, responsesTextFormatBody("client", text, clientTools, stream))
				if rr.Code != http.StatusOK {
					t.Fatalf("text %s tools %s: status=%d body=%s", text, clientTools, rr.Code, rr.Body.String())
				}
				planner.mu.Lock()
				defer planner.mu.Unlock()
				return planner.format, planner.tools
			}
			ordinaryFormat, ordinaryTools := serve("", "")
			if len(ordinaryFormat) != 0 || len(ordinaryTools) == 0 {
				t.Fatalf("ordinary request format=%s tools=%v, want no format and the gateway-owned tools", ordinaryFormat, ordinaryTools)
			}
			for _, text := range []string{`null`, `{}`, `{"format":null}`, `{"format":{"type":"text"}}`, `{"format":{"type":"text"},"verbosity":"low"}`} {
				format, tools := serve(text, "")
				if len(format) != 0 || !slices.Equal(tools, ordinaryTools) {
					t.Fatalf("text %s: format=%s tools=%v, want ordinary route (no format, tools %v)", text, format, tools, ordinaryTools)
				}
			}
			for _, tc := range []struct {
				text string
				want json.RawMessage
			}{
				{jsonSchema, chatJSONSchema},
				{`{"format":{"type":"json_object"}}`, json.RawMessage(`{"type":"json_object"}`)},
			} {
				format, tools := serve(tc.text, "")
				if len(format) == 0 || !jsonEqual(t, format, tc.want) {
					t.Fatalf("text %s: planner response_format=%s, want %s", tc.text, format, tc.want)
				}
				if len(tools) != 0 {
					t.Fatalf("text %s: planner tools=%v, want only the client's tools (none)", tc.text, tools)
				}
			}
			_, ordinaryWithClientTools := serve("", `[{"type":"function","name":"lookup"}]`)
			format, tools := serve(jsonSchema, `[{"type":"function","name":"lookup"}]`)
			if len(format) == 0 || !jsonEqual(t, format, chatJSONSchema) {
				t.Fatalf("client tools: planner response_format=%s, want %s", format, chatJSONSchema)
			}
			if !slices.Contains(tools, "lookup") || !slices.Equal(tools, ordinaryWithClientTools) {
				t.Fatalf("client tools: planner tools=%v, want the unstructured turn's tools %v (client tools keep the restore tool)", tools, ordinaryWithClientTools)
			}
			planner.mu.Lock()
			calls := planner.calls
			planner.mu.Unlock()
			if calls != 10 {
				t.Fatalf("planner calls=%d, want 10", calls)
			}
		})
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestResponsesTextFormatRideForwardsChatCarrierUpstream(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	var mu sync.Mutex
	var forwarded json.RawMessage
	var forwardedTools []json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			ResponseFormat json.RawMessage   `json:"response_format"`
			Tools          []json.RawMessage `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode upstream request: %v", err)
		}
		mu.Lock()
		forwarded, forwardedTools = request.ResponseFormat, request.Tools
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"m","choices":[{"message":{"role":"assistant","content":"{\"answer\":\"a\"}"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	t.Cleanup(upstream.Close)
	srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	jsonSchema := `{"format":{"type":"json_schema","name":"answer","strict":true,"schema":` + responsesTextFormatSchema + `}}`
	chatJSONSchema := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":` + responsesTextFormatSchema + `}}`)
	const lookup = `[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{}}}]`
	for _, tc := range []struct {
		text      string
		tools     string
		want      json.RawMessage
		wantTools []string
	}{
		{jsonSchema, "", chatJSONSchema, nil},
		{`{"format":{"type":"json_object"}}`, "", json.RawMessage(`{"type":"json_object"}`), nil},
		{"", lookup, nil, []string{"lookup", "mcp__fak__fak_context_restore"}},
		{jsonSchema, lookup, chatJSONSchema, []string{"lookup", "mcp__fak__fak_context_restore"}},
	} {
		rr := postResponsesTextFormat(t, srv, responsesTextFormatBody("m", tc.text, tc.tools, false))
		if rr.Code != http.StatusOK {
			t.Fatalf("text %s tools %s: status=%d body=%s", tc.text, tc.tools, rr.Code, rr.Body.String())
		}
		mu.Lock()
		got, tools := forwarded, forwardedTools
		mu.Unlock()
		if len(tc.want) == 0 {
			if len(got) != 0 {
				t.Fatalf("text %s tools %s: upstream response_format=%s, want none", tc.text, tc.tools, got)
			}
		} else if len(got) == 0 || !jsonEqual(t, got, tc.want) {
			t.Fatalf("text %s tools %s: upstream response_format=%s, want %s", tc.text, tc.tools, got, tc.want)
		}
		names := []string{}
		for _, raw := range tools {
			var tool struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if err := json.Unmarshal(raw, &tool); err != nil {
				t.Fatalf("decode upstream tool %s: %v", raw, err)
			}
			names = append(names, tool.Function.Name)
		}
		slices.Sort(names)
		if !slices.Equal(names, tc.wantTools) {
			t.Fatalf("text %s tools %s: upstream tools=%v, want %v", tc.text, tc.tools, names, tc.wantTools)
		}
	}
}

// fak-test:runtime fast est=700ms lane=default
func TestResponsesTextFormatNativeRoute(t *testing.T) {
	srv := nativeReceiptServer(t)
	rr := postResponsesTextFormat(t, srv, responsesTextFormatBody("synthetic-live", `{"format":{"type":"json_schema","name":"answer","strict":true,"schema":`+responsesTextFormatSchema+`}}`, "", false))
	if rr.Code != http.StatusOK {
		t.Fatalf("json_schema: status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp responsesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	var answer map[string]string
	if resp.Status != "completed" || json.Unmarshal([]byte(resp.OutputText), &answer) != nil || len(answer) != 1 || (answer["answer"] != "a" && answer["answer"] != "b") {
		t.Fatalf("json_schema: status=%q output_text=%q, want a completed object matching the schema", resp.Status, resp.OutputText)
	}
	jsonObjectBody := strings.Replace(responsesTextFormatBody("synthetic-live", `{"format":{"type":"json_object"}}`, "", false), `"max_output_tokens":32`, `"max_output_tokens":128`, 1)
	rr = postResponsesTextFormat(t, srv, jsonObjectBody)
	if rr.Code != http.StatusOK {
		t.Fatalf("json_object: status=%d body=%s", rr.Code, rr.Body.String())
	}
	resp = responsesResponse{}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	var object map[string]any
	if resp.Status != "completed" || !strings.HasPrefix(resp.OutputText, "{") || json.Unmarshal([]byte(resp.OutputText), &object) != nil {
		t.Fatalf("json_object: status=%q output_text=%q, want a completed JSON object", resp.Status, resp.OutputText)
	}
	for _, text := range []string{
		`{"format":{"type":"json_object","schema":{}}}`,
		`{"format":{"type":"json_schema","name":"answer","strict":false,"schema":` + responsesTextFormatSchema + `}}`,
	} {
		rr := postResponsesTextFormat(t, srv, responsesTextFormatBody("synthetic-live", text, "", false))
		if rr.Code != http.StatusBadRequest || chatErrorCode(t, rr.Body.Bytes()) != "native_structured_output_unsupported" {
			t.Fatalf("text %s: status=%d body=%s, want 400 native_structured_output_unsupported", text, rr.Code, rr.Body.String())
		}
	}
}

// fak-test:runtime fast est=60ms lane=default
func TestResponsesTextFormatMalformedRefusedBeforePlanner(t *testing.T) {
	srv := newTestServer(t)
	planner := &responsesTextFormatPlanner{}
	srv.planner = planner
	for _, tc := range []struct {
		text   string
		status int
		code   string
	}{
		{`"json"`, http.StatusBadRequest, "invalid_text_format"},
		{`[]`, http.StatusBadRequest, "invalid_text_format"},
		{`{"format":"json_schema"}`, http.StatusBadRequest, "invalid_text_format"},
		{`{"format":{}}`, http.StatusBadRequest, "invalid_text_format"},
		{`{"format":{"type":7}}`, http.StatusBadRequest, "invalid_text_format"},
		{`{"format":{"type":" "}}`, http.StatusBadRequest, "invalid_text_format"},
		{`{"format":{"type":"json_schema","name":"answer"}}`, http.StatusUnprocessableEntity, "json_schema_missing"},
		{`{"format":{"type":"json_schema","name":"answer","schema":[]}}`, http.StatusUnprocessableEntity, "json_schema_not_object"},
	} {
		rr := postResponsesTextFormat(t, srv, responsesTextFormatBody("client", tc.text, "", false))
		if rr.Code != tc.status || chatErrorCode(t, rr.Body.Bytes()) != tc.code {
			t.Fatalf("text %s: status=%d body=%s, want %d %s", tc.text, rr.Code, rr.Body.String(), tc.status, tc.code)
		}
	}
	if planner.calls != 0 {
		t.Fatalf("planner calls=%d, want 0", planner.calls)
	}
}
