package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestChatProxyForwardsStructuredOutputFieldsToRideEngine is the GPU-free witness
// for #907's pass-through deliverable: when a client asks for OpenAI-compatible
// structured outputs (`response_format`) and a `logit_bias` mask, those constraint
// carriers survive the fak gateway and reach the ride engine (vLLM/SGLang) VERBATIM,
// AND the tool candidate the constrained generation produced still enters fak's
// adjudication plane before any survivor is forwarded.
//
// This proves the answer to the issue's first question — "Which constraints are
// pass-through-only in ride mode?" — is wired, not prose: vLLM/SGLang enforce the
// JSON-schema/grammar during generation; fak forwards the constraint and adjudicates
// the result. The gateway is the proxy planner (a BaseURL is set), so the assertion
// runs against the actual bytes that crossed the upstream wire, not an internal seam.
//
// The constraint is generic structured-output JSON (a `response_format` object) — the
// shape vLLM `guided_json`/`response_format` and SGLang `json_schema` both accept — so
// the one test stands in for the whole ride-mode pass-through lane.
func TestChatProxyForwardsStructuredOutputFieldsToRideEngine(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	// The exact structured-output constraint the client sends. The gateway must forward
	// it byte-equivalent: a json_schema response_format pinning the tool-call shape.
	clientResponseFormat := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"tool_call","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"arguments":{"type":"object"}},"required":["name","arguments"]}}}`)
	clientLogitBias := map[int]float64{50256: -100, 1024: 12.5}
	clientGuidedGrammar := json.RawMessage(`"root ::= tool_call"`)
	clientGuidedChoice := json.RawMessage(`["tool_call"]`)

	// The upstream (a stand-in vLLM/SGLang OpenAI surface) captures what the gateway
	// actually forwarded, so the test asserts the constraint crossed the wire.
	var gotResponseFormat json.RawMessage
	var gotLogitBias map[int]float64
	var gotGuidedGrammar json.RawMessage
	var gotGuidedChoice json.RawMessage
	upstreamHits := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream body: %v", err)
		}
		var req struct {
			ResponseFormat json.RawMessage `json:"response_format"`
			LogitBias      map[int]float64 `json:"logit_bias"`
			GuidedGrammar  json.RawMessage `json:"guided_grammar"`
			GuidedChoice   json.RawMessage `json:"guided_choice"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("decode upstream request: %v\n%s", err, raw)
		}
		gotResponseFormat = req.ResponseFormat
		gotLogitBias = req.LogitBias
		gotGuidedGrammar = req.GuidedGrammar
		gotGuidedChoice = req.GuidedChoice
		w.Header().Set("Content-Type", "application/json")
		// A constrained generation: one allow*, one deny* call — proof the proposed set
		// reaches the gate AFTER generation, where deny is dropped and allow is kept.
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-907",
			"object":"chat.completion",
			"created":1718900007,
			"model":"Qwen/Qwen3.6-27B",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"content":null,
					"tool_calls":[
						{"id":"call_0","type":"function","function":{"name":"allow_read","arguments":"{\"path\":\"/etc/hosts\"}"}},
						{"id":"call_1","type":"function","function":{"name":"deny_write","arguments":"{\"path\":\"/etc/passwd\"}"}}
					]
				},
				"logprobs":null,
				"finish_reason":"tool_calls"
			}],
			"usage":{"prompt_tokens":40,"completion_tokens":20,"total_tokens":60}
		}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test",
		Model:    "qwen3.6-27b",
		BaseURL:  upstream.URL + "/v1",
		Provider: "openai-compatible",
		VDSO:     true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, err := json.Marshal(map[string]any{
		"model": "qwen3.6-27b",
		"messages": []map[string]any{
			{"role": "user", "content": "read /etc/hosts"},
		},
		"tools": []map[string]any{
			{"type": "function", "function": map[string]any{"name": "allow_read", "description": "read", "parameters": map[string]any{"type": "object"}}},
			{"type": "function", "function": map[string]any{"name": "deny_write", "description": "write", "parameters": map[string]any{"type": "object"}}},
		},
		"response_format": clientResponseFormat,
		"logit_bias":      clientLogitBias,
		"guided_grammar":  clientGuidedGrammar,
		"guided_choice":   clientGuidedChoice,
	})
	if err != nil {
		t.Fatal(err)
	}
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, respRaw)
	}
	if upstreamHits != 1 {
		t.Fatalf("upstream hits = %d, want 1", upstreamHits)
	}

	// (1) The structured-output constraint reached the ride engine VERBATIM.
	if !jsonEqual(t, gotResponseFormat, clientResponseFormat) {
		t.Errorf("response_format forwarded to ride engine = %s\nwant (verbatim) = %s", gotResponseFormat, clientResponseFormat)
	}
	if len(gotLogitBias) != len(clientLogitBias) {
		t.Fatalf("logit_bias forwarded = %v, want %v", gotLogitBias, clientLogitBias)
	}
	for tok, bias := range clientLogitBias {
		if gotLogitBias[tok] != bias {
			t.Errorf("logit_bias[%d] forwarded = %v, want %v", tok, gotLogitBias[tok], bias)
		}
	}
	if !jsonEqual(t, gotGuidedGrammar, clientGuidedGrammar) {
		t.Errorf("guided_grammar forwarded to ride engine = %s\nwant (verbatim) = %s", gotGuidedGrammar, clientGuidedGrammar)
	}
	if !jsonEqual(t, gotGuidedChoice, clientGuidedChoice) {
		t.Errorf("guided_choice forwarded to ride engine = %s\nwant (verbatim) = %s", gotGuidedChoice, clientGuidedChoice)
	}

	// (2) The candidate the constrained generation produced still entered the gate.
	var resp ChatResponse
	if err := json.Unmarshal(respRaw, &resp); err != nil {
		t.Fatalf("decode response: %v (%s)", err, respRaw)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(resp.Choices))
	}
	kept := resp.Choices[0].Message.ToolCalls
	if len(kept) != 1 {
		t.Fatalf("surviving tool calls = %d, want 1 (deny dropped): %+v", len(kept), kept)
	}
	if kept[0].Function.Name != "allow_read" {
		t.Errorf("surviving call = %q, want allow_read", kept[0].Function.Name)
	}
	if resp.Fak == nil || len(resp.Fak.Adjudications) != 2 {
		t.Fatalf("fak adjudications = %+v, want 2 (every proposed call adjudicated AFTER generation)", resp.Fak)
	}
}

// TestChatProxyOmitsStructuredOutputFieldsWhenAbsent pins the bit-exact drop-in half:
// a client that sends NO response_format / logit_bias must produce an upstream body
// with neither key present (omitempty), so the unconstrained path is byte-identical to
// the pre-#907 wire and a non-structured client is never silently constrained.
func TestChatProxyOmitsStructuredOutputFieldsWhenAbsent(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var rawUpstream []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawUpstream, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	io.Copy(io.Discard, httpResp.Body)

	var up map[string]json.RawMessage
	if err := json.Unmarshal(rawUpstream, &up); err != nil {
		t.Fatalf("decode upstream body: %v\n%s", err, rawUpstream)
	}
	if _, ok := up["response_format"]; ok {
		t.Errorf("response_format present on upstream wire for an unconstrained request: %s", rawUpstream)
	}
	if _, ok := up["logit_bias"]; ok {
		t.Errorf("logit_bias present on upstream wire for an unconstrained request: %s", rawUpstream)
	}
	if _, ok := up["guided_grammar"]; ok {
		t.Errorf("guided_grammar present on upstream wire for an unconstrained request: %s", rawUpstream)
	}
}

// jsonEqual reports whether two JSON documents are semantically equal (key order /
// whitespace insensitive), so a verbatim-forwarding assertion does not pin a
// re-marshal's incidental key ordering.
func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var av, bv any
	if err := json.Unmarshal(a, &av); err != nil {
		t.Fatalf("unmarshal a: %v (%s)", err, a)
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		t.Fatalf("unmarshal b: %v (%s)", err, b)
	}
	na, _ := json.Marshal(av)
	nb, _ := json.Marshal(bv)
	return bytes.Equal(na, nb)
}

// newStructuredOutputProxy stands up a proxy-planner gateway in front of a counting
// stand-in upstream, so a refusal test can prove no upstream call was made and a
// pass-through test can capture the forwarded response_format bytes.
func newStructuredOutputProxy(t *testing.T) (gatewayURL string, hits *int, gotRF *json.RawMessage) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	hits = new(int)
	gotRF = new(json.RawMessage)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*hits++
		var req struct {
			ResponseFormat json.RawMessage `json:"response_format"`
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &req)
		*gotRF = req.ResponseFormat
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-rf","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	t.Cleanup(upstream.Close)
	srv, err := New(Config{EngineID: "test", Model: "qwen3.6-27b", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts.URL, hits, gotRF
}

func postResponseFormat(t *testing.T, gatewayURL string, rf json.RawMessage) (int, string) {
	t.Helper()
	status, code, _ := postResponseFormatFault(t, gatewayURL, rf)
	return status, code
}

// postResponseFormatFault posts rf and returns the status plus the typed error code
// and schema_path of a refusal (empty on success).
func postResponseFormatFault(t *testing.T, gatewayURL string, rf json.RawMessage) (status int, code, schemaPath string) {
	t.Helper()
	body := []byte(`{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"response_format":` + string(rf) + `}`)
	resp, err := http.Post(gatewayURL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var env struct {
		Error struct {
			Code       string `json:"code"`
			Type       string `json:"type"`
			SchemaPath string `json:"schema_path"`
		} `json:"error"`
	}
	_ = json.Unmarshal(raw, &env)
	if resp.StatusCode == http.StatusUnprocessableEntity && env.Error.Type != "invalid_request_error" {
		t.Fatalf("422 error type = %q, want invalid_request_error: %s", env.Error.Type, raw)
	}
	return resp.StatusCode, env.Error.Code, env.Error.SchemaPath
}

// TestChatRejectsMalformedJSONSchema422 pins oss-port-gateway-json-schema-validate
// (ADAPT of TGI router/src/validation.rs:341-406@b4adbf2): a malformed json_schema
// response_format is refused with 422 + a closed code and schema_path BEFORE any
// upstream call, while every valid JSON Schema shape clients send (no "type", open
// objects, root anyOf/oneOf/allOf, root $ref + $defs/definitions, boolean schemas)
// and json_object pass through verbatim.
func TestChatRejectsMalformedJSONSchema422(t *testing.T) {
	gw, hits, gotRF := newStructuredOutputProxy(t)
	rejects := []struct {
		name, rf, code, path string
	}{
		{"response_format not an object", `"json_schema"`, errCodeInvalidResponseFormat, ""},
		{"json_schema missing", `{"type":"json_schema"}`, errCodeJSONSchemaMissing, ""},
		{"json_schema null", `{"type":"json_schema","json_schema":null}`, errCodeJSONSchemaMissing, ""},
		{"json_schema not an object", `{"type":"json_schema","json_schema":"x"}`, errCodeJSONSchemaNotObject, ""},
		{"schema missing", `{"type":"json_schema","json_schema":{"name":"x"}}`, errCodeJSONSchemaMissing, ""},
		{"schema array", `{"type":"json_schema","json_schema":{"name":"x","schema":["object"]}}`, errCodeJSONSchemaNotObject, "#"},
		{"schema string", `{"type":"json_schema","json_schema":{"name":"x","schema":"object"}}`, errCodeJSONSchemaNotObject, "#"},
		{"schema number", `{"type":"json_schema","json_schema":{"name":"x","schema":1}}`, errCodeJSONSchemaNotObject, "#"},
		{"type unknown name", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"date"}}}`, errCodeJSONSchemaInvalidType, "#/type"},
		{"type number", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":1}}}`, errCodeJSONSchemaInvalidType, "#/type"},
		{"type empty string", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":""}}}`, errCodeJSONSchemaInvalidType, "#/type"},
		{"type empty array", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":[]}}}`, errCodeJSONSchemaInvalidType, "#/type"},
		{"type array with unknown", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":["string","date"]}}}`, errCodeJSONSchemaInvalidType, "#/type"},
		{"nested property type unknown", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"a":{"type":"str"}}}}}`, errCodeJSONSchemaInvalidType, "#/properties/a/type"},
		{"properties array", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":["a"]}}}`, errCodeJSONSchemaInvalidProps, "#/properties"},
		{"properties string", `{"type":"json_schema","json_schema":{"name":"x","schema":{"properties":"a"}}}`, errCodeJSONSchemaInvalidProps, "#/properties"},
		{"required string", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{},"required":"a"}}}`, errCodeJSONSchemaInvalidRequired, "#/required"},
		{"required non-string items", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{},"required":[1]}}}`, errCodeJSONSchemaInvalidRequired, "#/required"},
		{"required boolean", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"a":{"type":"string","required":true}}}}}`, errCodeJSONSchemaInvalidRequired, "#/properties/a/required"},
		{"anyOf branch type unknown", `{"type":"json_schema","json_schema":{"name":"x","schema":{"anyOf":[{"type":"string"},{"type":"nope"}]}}}`, errCodeJSONSchemaInvalidType, "#/anyOf/1/type"},
		{"$defs entry properties malformed", `{"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"#/$defs/a~1b","$defs":{"a/b":{"properties":1}}}}}`, errCodeJSONSchemaInvalidProps, "#/$defs/a~1b/properties"},
	}
	for _, tc := range rejects {
		status, code, path := postResponseFormatFault(t, gw, json.RawMessage(tc.rf))
		if status != http.StatusUnprocessableEntity || code != tc.code || path != tc.path {
			t.Errorf("%s: status=%d code=%q schema_path=%q, want 422 %q %q", tc.name, status, code, path, tc.code, tc.path)
		}
	}
	if *hits != 0 {
		t.Fatalf("upstream hits = %d after malformed schemas, want 0 (refuse before forwarding)", *hits)
	}
	passes := []struct{ name, rf string }{
		{"strict object", `{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}}`},
		{"string", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"string"}}}`},
		{"json_object", `{"type":"json_object"}`},
		{"no type", `{"type":"json_schema","json_schema":{"name":"x","schema":{"properties":{"a":{}}}}}`},
		{"no type enum only", `{"type":"json_schema","json_schema":{"name":"x","schema":{"enum":["a","b"]}}}`},
		{"empty schema", `{"type":"json_schema","json_schema":{"name":"x","schema":{}}}`},
		{"object without properties", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}`},
		{"object additionalProperties only", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","additionalProperties":{"type":"integer"}}}}`},
		{"type array", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":["string","null"]}}}`},
		{"root anyOf", `{"type":"json_schema","json_schema":{"name":"x","schema":{"anyOf":[{"type":"string"},{"type":"object","properties":{"a":{"type":"number"}}}]}}}`},
		{"root oneOf", `{"type":"json_schema","json_schema":{"name":"x","schema":{"oneOf":[{"type":"integer"},{"type":"null"}]}}}`},
		{"root allOf", `{"type":"json_schema","json_schema":{"name":"x","schema":{"allOf":[{"properties":{"a":{"type":"string"}}},{"required":["a"]}]}}}`},
		{"root $ref $defs", `{"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"#/$defs/item","$defs":{"item":{"type":"object","properties":{"id":{"type":"integer"}}}}}}}`},
		{"root $ref definitions", `{"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"#/definitions/item","definitions":{"item":{"type":"string"}}}}}`},
		{"boolean schema true", `{"type":"json_schema","json_schema":{"name":"x","schema":true}}`},
		{"boolean schema false", `{"type":"json_schema","json_schema":{"name":"x","schema":false}}`},
		{"boolean subschemas", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"a":true,"b":false},"additionalProperties":false}}}`},
		{"data keywords not walked", `{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"type":{"type":"string"},"properties":{"type":"string"}},"default":{"type":"bogus","properties":1},"const":{"required":"x"}}}}`},
	}
	for _, tc := range passes {
		status, code, _ := postResponseFormatFault(t, gw, json.RawMessage(tc.rf))
		if status != http.StatusOK {
			t.Errorf("%s: status=%d code=%q, want 200", tc.name, status, code)
			continue
		}
		if !jsonEqual(t, *gotRF, json.RawMessage(tc.rf)) {
			t.Errorf("%s: not forwarded verbatim: got %s want %s", tc.name, *gotRF, tc.rf)
		}
	}
	if *hits != len(passes) {
		t.Fatalf("upstream hits = %d, want %d", *hits, len(passes))
	}
}

// jsonSchemaRF wraps a raw schema in a json_schema response_format.
func jsonSchemaRF(schema string) json.RawMessage {
	return json.RawMessage(`{"type":"json_schema","json_schema":{"name":"x","schema":` + schema + `}}`)
}

// paddedStringSchema returns a valid {"type":"string","description":"..."} schema
// of exactly n bytes.
func paddedStringSchema(t *testing.T, n int) string {
	t.Helper()
	base := `{"type":"string","description":""}`
	if n < len(base) {
		t.Fatalf("n=%d below base %d", n, len(base))
	}
	return `{"type":"string","description":"` + strings.Repeat("a", n-len(base)) + `"}`
}

// nestedArraySchema returns a valid schema whose object/array nesting depth is
// exactly depth: (depth-1) array levels wrapping a string leaf.
func nestedArraySchema(depth int) string {
	s := `{"type":"string"}`
	for i := 1; i < depth; i++ {
		s = `{"type":"array","items":` + s + `}`
	}
	return s
}

// TestChatRejectsOversizedJSONSchema422 pins oss-port-gateway-json-schema-size-cap
// (ADAPT of TGI router/src/validation.rs:341-406@b4adbf2): the json_schema schema is
// accepted exactly at the byte (64 KiB) and nesting-depth (32) caps, refused with 422
// + a closed code at cap+1 before any upstream call, the caps are env-overridable,
// and the cap never touches tool definitions.
func TestChatRejectsOversizedJSONSchema422(t *testing.T) {
	if got := jsonNestingDepth([]byte(nestedArraySchema(defaultJSONSchemaMaxDepth)), 1<<20); got != defaultJSONSchemaMaxDepth {
		t.Fatalf("nestedArraySchema depth = %d, want %d", got, defaultJSONSchemaMaxDepth)
	}
	if got := jsonNestingDepth([]byte(`{"a":"{[{[\"}"}`), 99); got != 1 {
		t.Fatalf("brackets inside strings counted: depth = %d, want 1", got)
	}
	gw, hits, gotRF := newStructuredOutputProxy(t)

	atBytes := jsonSchemaRF(paddedStringSchema(t, defaultJSONSchemaMaxBytes))
	if status, code := postResponseFormat(t, gw, atBytes); status != http.StatusOK {
		t.Fatalf("schema at byte cap: status=%d code=%q, want 200", status, code)
	}
	if !jsonEqual(t, *gotRF, atBytes) {
		t.Fatal("schema at byte cap not forwarded verbatim")
	}
	atDepth := jsonSchemaRF(nestedArraySchema(defaultJSONSchemaMaxDepth))
	if status, code := postResponseFormat(t, gw, atDepth); status != http.StatusOK {
		t.Fatalf("schema at depth cap: status=%d code=%q, want 200", status, code)
	}
	accepted := *hits

	if status, code := postResponseFormat(t, gw, jsonSchemaRF(paddedStringSchema(t, defaultJSONSchemaMaxBytes+1))); status != http.StatusUnprocessableEntity || code != errCodeJSONSchemaTooLarge {
		t.Fatalf("schema at byte cap+1: status=%d code=%q, want 422 %q", status, code, errCodeJSONSchemaTooLarge)
	}
	if status, code := postResponseFormat(t, gw, jsonSchemaRF(nestedArraySchema(defaultJSONSchemaMaxDepth+1))); status != http.StatusUnprocessableEntity || code != errCodeJSONSchemaTooDeep {
		t.Fatalf("schema at depth cap+1: status=%d code=%q, want 422 %q", status, code, errCodeJSONSchemaTooDeep)
	}
	if *hits != accepted {
		t.Fatalf("upstream hits grew %d -> %d on refused schemas", accepted, *hits)
	}

	// Env override tightens both caps.
	t.Setenv("FAK_GATEWAY_JSON_SCHEMA_MAX_BYTES", "64")
	t.Setenv("FAK_GATEWAY_JSON_SCHEMA_MAX_DEPTH", "3")
	if status, _ := postResponseFormat(t, gw, jsonSchemaRF(paddedStringSchema(t, 64))); status != http.StatusOK {
		t.Fatalf("override byte cap exact: status=%d, want 200", status)
	}
	if status, code := postResponseFormat(t, gw, jsonSchemaRF(paddedStringSchema(t, 65))); code != errCodeJSONSchemaTooLarge {
		t.Fatalf("override byte cap+1: status=%d code=%q, want %q", status, code, errCodeJSONSchemaTooLarge)
	}
	t.Setenv("FAK_GATEWAY_JSON_SCHEMA_MAX_BYTES", "1024")
	if status, _ := postResponseFormat(t, gw, jsonSchemaRF(nestedArraySchema(3))); status != http.StatusOK {
		t.Fatalf("override depth cap exact: status=%d, want 200", status)
	}
	if status, code := postResponseFormat(t, gw, jsonSchemaRF(nestedArraySchema(4))); code != errCodeJSONSchemaTooDeep {
		t.Fatalf("override depth cap+1: status=%d code=%q, want %q", status, code, errCodeJSONSchemaTooDeep)
	}

	// The cap is scoped to response_format: an oversized tool definition still passes.
	bigParams := nestedArraySchema(10)
	body := []byte(`{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"allow_read","description":"` + strings.Repeat("d", 200) + `","parameters":` + bigParams + `}}]}`)
	resp, err := http.Post(gw+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("oversized tool definition: status=%d, want 200 (cap must not apply to tools)", resp.StatusCode)
	}
}
