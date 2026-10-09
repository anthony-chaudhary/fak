package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1s lane=default
func TestChatResponseFormatCarrierRejectsBeforeDispatch(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	format := func(schema string) string {
		return `{"type":"json_schema","json_schema":{"schema":` + schema + `}}`
	}
	sized := func(n int) string {
		const base = `{"description":""}`
		return `{"description":"` + strings.Repeat("a", n-len(base)) + `"}`
	}
	nested := func(depth int) string {
		// Data keywords count toward resource depth just like schema keywords do.
		return `{"default":` + strings.Repeat("[", depth-1) + "null" + strings.Repeat("]", depth-1) + `}`
	}

	for _, fanout := range []bool{false, true} {
		name := "without_followers"
		if fanout {
			name = "with_followers"
		}
		t.Run(name, func(t *testing.T) {
			var upstreamHits, followerHits atomic.Int32
			var forwardedMu sync.Mutex
			var forwarded json.RawMessage
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				upstreamHits.Add(1)
				var request struct {
					ResponseFormat json.RawMessage `json:"response_format"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("decode upstream request: %v", err)
				}
				forwardedMu.Lock()
				forwarded = request.ResponseFormat
				forwardedMu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`)
			}))
			t.Cleanup(upstream.Close)
			follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				followerHits.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				_, _ = io.WriteString(w, `{}`)
			}))
			t.Cleanup(follower.Close)
			addrs := ""
			if fanout {
				addrs = follower.URL
			}
			t.Setenv("FAK_EP_FANOUT_ADDRS", addrs)
			srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			ts := httptest.NewServer(srv.Handler())
			t.Cleanup(ts.Close)
			client := &http.Client{Timeout: 5 * time.Second}

			for _, tc := range []struct {
				name, format, code, path string
				maxBytes, maxDepth       string
				tools                    json.RawMessage
				stream                   bool
			}{
				{name: "non_object", format: `"json_schema"`, code: "invalid_response_format"},
				{name: "missing_carrier", format: `{"type":"json_schema"}`, code: "json_schema_missing"},
				{name: "non_object_carrier", format: `{"type":"json_schema","json_schema":[]}`, code: "json_schema_not_object"},
				{name: "missing_schema", format: `{"type":"json_schema","json_schema":{}}`, code: "json_schema_missing"},
				{name: "array_schema_stream", format: `{"type":"json_schema","json_schema":{"schema":[]}}`, code: "json_schema_not_object", path: "#", stream: true},
				{name: "open_object", format: `{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}`},
				{name: "boolean_true", format: `{"type":"json_schema","json_schema":{"schema":true}}`},
				{name: "boolean_false", format: `{"type":"json_schema","json_schema":{"schema":false}}`},
				{name: "draft_three", format: `{"type":"json_schema","json_schema":{"schema":{"$schema":"http://json-schema.org/draft-03/schema#","properties":{"a":{"type":"string","required":true}}}}}`},
				{name: "other_format", format: `{"type":"json_object"}`},
				{name: "default_byte_cap", format: format(sized(defaultJSONSchemaMaxBytes))},
				{name: "default_byte_cap_exceeded", format: format(sized(defaultJSONSchemaMaxBytes + 1)), code: "json_schema_too_large"},
				{name: "default_byte_cap_exceeded_stream", format: format(sized(defaultJSONSchemaMaxBytes + 1)), code: "json_schema_too_large", stream: true},
				{name: "default_depth_cap", format: format(nested(defaultJSONSchemaMaxDepth))},
				{name: "default_depth_cap_exceeded", format: format(nested(defaultJSONSchemaMaxDepth + 1)), code: "json_schema_too_deep"},
				{name: "default_depth_cap_exceeded_stream", format: format(nested(defaultJSONSchemaMaxDepth + 1)), code: "json_schema_too_deep", stream: true},
				{name: "override_byte_cap", format: format(sized(64)), maxBytes: " 64 "},
				{name: "override_byte_cap_exceeded", format: format(sized(65)), maxBytes: "64", code: "json_schema_too_large"},
				{name: "schema_outer_whitespace", format: format(" \n" + sized(64) + "\t "), maxBytes: "64"},
				{name: "schema_inner_whitespace", format: format("{ " + sized(64)[1:]), maxBytes: "64", code: "json_schema_too_large"},
				{name: "override_depth_cap", format: format(nested(3)), maxDepth: " 3 "},
				{name: "override_depth_cap_exceeded", format: format(nested(4)), maxDepth: "3", code: "json_schema_too_deep"},
				{name: "raised_byte_cap", format: format(sized(defaultJSONSchemaMaxBytes + 1)), maxBytes: strconv.Itoa(defaultJSONSchemaMaxBytes + 1)},
				{name: "raised_depth_cap", format: format(nested(defaultJSONSchemaMaxDepth + 1)), maxDepth: strconv.Itoa(defaultJSONSchemaMaxDepth + 1)},
				{name: "escaped_brackets", format: format(`{"description":"[{\\\"}]\\\\"}`), maxDepth: "1"},
				{name: "utf8_bytes", format: format(`{"description":"é"}`), maxBytes: "19", code: "json_schema_too_large"},
				{name: "invalid_byte_override", format: format(sized(defaultJSONSchemaMaxBytes + 1)), maxBytes: "0", code: "json_schema_too_large"},
				{name: "invalid_depth_override", format: format(nested(defaultJSONSchemaMaxDepth + 1)), maxDepth: "-1", code: "json_schema_too_deep"},
				{name: "tools_outside_caps", format: format(`{}`), maxBytes: "64", maxDepth: "3", tools: json.RawMessage(`[{"type":"function","function":{"name":"allow_read","description":"` + strings.Repeat("d", 128) + `","parameters":` + nested(10) + `}}]`)},
			} {
				t.Run(tc.name, func(t *testing.T) {
					t.Setenv("FAK_GATEWAY_JSON_SCHEMA_MAX_BYTES", tc.maxBytes)
					t.Setenv("FAK_GATEWAY_JSON_SCHEMA_MAX_DEPTH", tc.maxDepth)
					beforeUpstream, beforeFollower := upstreamHits.Load(), followerHits.Load()
					// Preserve raw schema whitespace so the byte-cap cases exercise wire bytes.
					request := `{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":` + tc.format + `,"stream":` + strconv.FormatBool(tc.stream)
					if tc.tools != nil {
						request += `,"tools":` + string(tc.tools)
					}
					body := []byte(request + `}`)
					response, err := client.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
					if err != nil {
						t.Fatal(err)
					}
					defer response.Body.Close()
					raw, err := io.ReadAll(response.Body)
					if err != nil {
						t.Fatal(err)
					}
					if tc.code != "" {
						var envelope struct {
							Error struct {
								Code       string `json:"code"`
								Type       string `json:"type"`
								SchemaPath string `json:"schema_path"`
							} `json:"error"`
						}
						if err := json.Unmarshal(raw, &envelope); err != nil {
							t.Fatalf("decode refusal: %v: %s", err, raw)
						}
						if response.StatusCode != http.StatusUnprocessableEntity || envelope.Error.Code != tc.code || envelope.Error.Type != "invalid_request_error" || envelope.Error.SchemaPath != tc.path {
							t.Fatalf("status=%d refusal=%s, want 422 code=%s path=%s", response.StatusCode, raw, tc.code, tc.path)
						}
						if upstreamHits.Load() != beforeUpstream || followerHits.Load() != beforeFollower {
							t.Fatal("rejected response format dispatched to upstream or EP follower")
						}
						return
					}
					if response.StatusCode != http.StatusOK || upstreamHits.Load() != beforeUpstream+1 {
						t.Fatalf("accepted carrier: status=%d upstream calls=%d: %s", response.StatusCode, upstreamHits.Load()-beforeUpstream, raw)
					}
					forwardedMu.Lock()
					got := append(json.RawMessage(nil), forwarded...)
					forwardedMu.Unlock()
					if !jsonEqual(t, got, json.RawMessage(tc.format)) {
						t.Fatalf("forwarded format = %s, want %s", got, tc.format)
					}
					wantFollowers := beforeFollower
					if fanout {
						wantFollowers++
					}
					if followerHits.Load() != wantFollowers {
						t.Fatalf("follower calls=%d, want %d", followerHits.Load(), wantFollowers)
					}
				})
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestJSONSchemaCapOverrideParsing(t *testing.T) {
	if defaultJSONSchemaMaxBytes != 64<<10 || defaultJSONSchemaMaxDepth != 32 {
		t.Fatalf("default caps = %d bytes / %d depth, want 65536 / 32", defaultJSONSchemaMaxBytes, defaultJSONSchemaMaxDepth)
	}
	for _, raw := range []string{"", " ", "0", "-1", "no", "1.5", strings.Repeat("9", 100)} {
		if got := jsonSchemaCap(raw, 32); got != 32 {
			t.Errorf("jsonSchemaCap(%q) = %d, want default 32", raw, got)
		}
	}
	if got := jsonSchemaCap(" 17 ", 32); got != 17 {
		t.Fatalf("trimmed positive override = %d, want 17", got)
	}
}
