package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// model_alias_fallback_test.go — ticket claude-alias-fallback (audit finding 2):
// a variant-spelled model id (`<alias>[1m]`, `<alias>-1m`) must resolve to its
// served base so the turn is served, a variant with no resolvable base must fail
// closed with a typed model_not_found, and a non-variant id must keep the
// facade's pass-through contract untouched.

// TestResolveAliasVariantTable is the table-driven edge witness for the pure
// resolver: marker shapes, case, whitespace, stacking, the modelreg-equivalent
// base, and the three verdict classes (pass-through, fallback, fail-closed).
func TestResolveAliasVariantTable(t *testing.T) {
	canonical := func(id string) string {
		switch strings.ToLower(strings.TrimSpace(id)) {
		case "boostrun.medium", "boostrun.medium-q4": // served id + a modelreg-equivalent spelling
			return "boostrun.medium"
		case "roster.model":
			return "roster.model"
		}
		return ""
	}
	tests := []struct {
		name       string
		in         string
		wantServed string
		wantFell   bool
		wantOK     bool
	}{
		{"known exact id passes through as sent", "boostrun.medium", "boostrun.medium", false, true},
		{"bracket variant of served base falls back", "boostrun.medium[1m]", "boostrun.medium", true, true},
		{"dash variant of served base falls back", "boostrun.medium-1m", "boostrun.medium", true, true},
		{"marker match is case-insensitive", "Boostrun.Medium[1M]", "boostrun.medium", true, true},
		{"dash marker match is case-insensitive", "Boostrun.Medium-1M", "boostrun.medium", true, true},
		{"surrounding whitespace is trimmed", "  boostrun.medium[1m]\t", "boostrun.medium", true, true},
		{"stacked markers strip repeatedly", "boostrun.medium[1m]-1m", "boostrun.medium", true, true},
		{"modelreg-equivalent base falls back to canonical", "boostrun.medium-q4[1m]", "boostrun.medium", true, true},
		{"roster-bound base falls back to its exact id", "Roster.Model[1m]", "roster.model", true, true},
		{"variant of unknown base fails closed", "zzz.unknown[1m]", "zzz.unknown[1m]", false, false},
		{"dash variant of unknown base fails closed", "zzz.unknown-1m", "zzz.unknown-1m", false, false},
		{"bare marker id fails closed", "[1m]", "[1m]", false, false},
		{"bare dash marker id fails closed", "-1m", "-1m", false, false},
		{"non-variant unknown id passes through", "zzz.unknown", "zzz.unknown", false, true},
		{"non-marker bracket suffix passes through", "zzz.unknown[2m]", "zzz.unknown[2m]", false, true},
		{"inner marker that is not a suffix passes through", "zz[1m].unknown", "zz[1m].unknown", false, true},
		{"empty model is not a fallback", "", "", false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			served, fell, ok := resolveAliasVariant(tt.in, canonical)
			if served != tt.wantServed || fell != tt.wantFell || ok != tt.wantOK {
				t.Fatalf("resolveAliasVariant(%q) = (%q, %v, %v), want (%q, %v, %v)",
					tt.in, served, fell, ok, tt.wantServed, tt.wantFell, tt.wantOK)
			}
		})
	}
}

// TestStripAliasVariantMarkers pins the stripper's contract directly: only
// trailing markers strip, the base keeps the client's case, and a bare marker
// strips to the empty string while still reporting marker-shaped.
func TestStripAliasVariantMarkers(t *testing.T) {
	tests := []struct {
		in         string
		want       string
		wantShaped bool
	}{
		{"boostrun.medium[1m]", "boostrun.medium", true},
		{"Boostrun.Medium[1M]", "Boostrun.Medium", true},
		{"boostrun.medium-1m", "boostrun.medium", true},
		{"boostrun.medium[1m]-1m", "boostrun.medium", true},
		{"boostrun.medium", "boostrun.medium", false},
		{"boostrun.medium[2m]", "boostrun.medium[2m]", false},
		{"[1m]", "", true},
		{"", "", false},
	}
	for _, tt := range tests {
		got, shaped := stripAliasVariantMarkers(tt.in)
		if got != tt.want || shaped != tt.wantShaped {
			t.Errorf("stripAliasVariantMarkers(%q) = (%q, %v), want (%q, %v)", tt.in, got, shaped, tt.want, tt.wantShaped)
		}
	}
}

// postMessages posts an Anthropic /v1/messages body and returns the response.
func postMessages(t *testing.T, ts *httptest.Server, model string, stream bool) *http.Response {
	t.Helper()
	reqBody := map[string]any{
		"model":      model,
		"max_tokens": 64,
		"stream":     stream,
		"messages":   []map[string]any{{"role": "user", "content": "hello"}},
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("post /v1/messages: %v", err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// TestAnthropicMessagesAliasVariantFallsBack is the audited DOA shape, witnessed
// on both response surfaces: a variant-spelled id resolves to the served base,
// the turn is SERVED, the response names the base, and HeaderAliasFallback
// annotates the rewrite.
func TestAnthropicMessagesAliasVariantFallsBack(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "buffered"
		if stream {
			name = "sse"
		}
		t.Run(name, func(t *testing.T) {
			srv := newTestServer(t)
			planner := &mockQwenPlanner{output: "served answer"}
			srv.planner = bufferedQwenPlanner{inner: planner}
			ts := httptest.NewServer(srv.AnthropicMessagesHandler())
			defer ts.Close()

			resp := postMessages(t, ts, "test-model[1m]", stream)
			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, body)
			}
			if got := resp.Header.Get(HeaderAliasFallback); got != "test-model[1m] -> test-model" {
				t.Fatalf("%s header = %q, want %q", HeaderAliasFallback, got, "test-model[1m] -> test-model")
			}
			if stream {
				events, err := parseSSEStream(resp.Body)
				if err != nil {
					t.Fatalf("parse SSE stream: %v", err)
				}
				var start struct {
					Message struct {
						Model string `json:"model"`
					} `json:"message"`
				}
				if len(events) == 0 || events[0].Event != "message_start" {
					t.Fatalf("first SSE event = %+v, want message_start", events)
				}
				if err := json.Unmarshal([]byte(events[0].Data), &start); err != nil {
					t.Fatalf("decode message_start: %v", err)
				}
				if start.Message.Model != "test-model" {
					t.Fatalf("message_start model = %q, want the resolved base test-model", start.Message.Model)
				}
				return
			}
			var out struct {
				Model   string `json:"model"`
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if out.Model != "test-model" {
				t.Fatalf("response model = %q, want the resolved base test-model", out.Model)
			}
			if len(out.Content) == 0 || out.Content[0].Text == "" {
				t.Fatalf("response content empty — the audited DOA shape, want a served turn: %+v", out)
			}
		})
	}
}

// TestAnthropicMessagesAliasVariantUnknownBaseFailsClosed pins the fail-closed
// class: a variant-shaped id whose base resolves to nothing is refused with a
// typed model_not_found BEFORE any planner turn is spent.
func TestAnthropicMessagesAliasVariantUnknownBaseFailsClosed(t *testing.T) {
	for _, model := range []string{"zzz.unknown[1m]", "zzz.unknown-1m", "[1m]"} {
		t.Run(model, func(t *testing.T) {
			srv := newTestServer(t)
			planner := &mockQwenPlanner{output: "must not be reached"}
			srv.planner = bufferedQwenPlanner{inner: planner}
			ts := httptest.NewServer(srv.AnthropicMessagesHandler())
			defer ts.Close()

			resp := postMessages(t, ts, model, false)
			if resp.StatusCode != http.StatusBadRequest {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("status = %d, want 400: %s", resp.StatusCode, body)
			}
			if resp.Header.Get(HeaderAliasFallback) != "" {
				t.Fatalf("%s set on a fail-closed response", HeaderAliasFallback)
			}
			var out struct {
				Error struct {
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
				t.Fatalf("decode error body: %v", err)
			}
			if out.Error.Code != ErrCodeAliasModelNotFound {
				t.Fatalf("error code = %q, want %q", out.Error.Code, ErrCodeAliasModelNotFound)
			}
			if !strings.Contains(out.Error.Message, "zzz.unknown") && model != "[1m]" {
				t.Fatalf("error message %q does not name the unresolvable base", out.Error.Message)
			}
			if len(planner.seenMessages) != 0 {
				t.Fatalf("planner saw %d turns, want 0 — a failed resolution must not spend a turn", len(planner.seenMessages))
			}
		})
	}
}

// TestAnthropicMessagesNonVariantUnknownPassesThrough pins the boundary of the
// fallback: an id with NO variant marker keeps the facade's existing
// pass-through contract (the client's label is not this seam's concern), so the
// audited fix cannot flip admission for the ids the adapter tests rely on.
func TestAnthropicMessagesNonVariantUnknownPassesThrough(t *testing.T) {
	srv := newTestServer(t)
	planner := &mockQwenPlanner{output: "served answer"}
	srv.planner = bufferedQwenPlanner{inner: planner}
	ts := httptest.NewServer(srv.AnthropicMessagesHandler())
	defer ts.Close()

	resp := postMessages(t, ts, "zzz.unknown", false)
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200 (pass-through contract): %s", resp.StatusCode, body)
	}
	if got := resp.Header.Get(HeaderAliasFallback); got != "" {
		t.Fatalf("%s = %q on a non-variant request, want unset", HeaderAliasFallback, got)
	}
}

// TestNormalizeAnthropicRequestModelSplicesRaw witnesses the raw half of the
// fallback wiring: when the fallback fires, the request's verbatim bytes carry
// the resolved base too, so the anthropic→anthropic passthrough forwards the
// resolved id; a non-variant request leaves req.Raw byte-identical.
func TestNormalizeAnthropicRequestModelSplicesRaw(t *testing.T) {
	srv := newTestServer(t)
	decode := func(t *testing.T, raw []byte) *agent.AnthropicMessagesRequest {
		t.Helper()
		req, err := agent.DecodeAnthropicMessagesRequest(raw)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		return req
	}

	t.Run("variant raw is spliced to the base", func(t *testing.T) {
		req := decode(t, []byte(`{"model":"test-model[1m]","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`))
		rec := httptest.NewRecorder()
		if !srv.normalizeAnthropicRequestModel(rec, req) {
			t.Fatalf("normalize refused a resolvable variant")
		}
		if req.Model != "test-model" {
			t.Fatalf("req.Model = %q, want test-model", req.Model)
		}
		if rec.Header().Get(HeaderAliasFallback) != "test-model[1m] -> test-model" {
			t.Fatalf("header = %q, want the fallback annotation", rec.Header().Get(HeaderAliasFallback))
		}
		spliced := decode(t, req.Raw)
		if spliced.Model != "test-model" {
			t.Fatalf("raw model = %q, want the spliced base test-model (raw: %s)", spliced.Model, req.Raw)
		}
		if !bytes.Contains(req.Raw, []byte(`"content":"hi"`)) {
			t.Fatalf("raw lost unrelated bytes: %s", req.Raw)
		}
	})
	t.Run("non-variant raw is byte-identical", func(t *testing.T) {
		raw := []byte(`{"model":"zzz.unknown","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
		req := decode(t, raw)
		rec := httptest.NewRecorder()
		if !srv.normalizeAnthropicRequestModel(rec, req) {
			t.Fatalf("normalize refused a pass-through id")
		}
		if !bytes.Equal(req.Raw, raw) {
			t.Fatalf("raw changed on a non-variant request: %s", req.Raw)
		}
		if rec.Header().Get(HeaderAliasFallback) != "" {
			t.Fatalf("%s set without a fallback", HeaderAliasFallback)
		}
	})
}

// TestSpliceAnthropicModel is the table-driven edge witness for the raw splice:
// only the top-level model value is touched, a "model" literal inside a client
// string or a nested schema is never mistaken for the key, and every ambiguity
// is fail-safe identity.
func TestSpliceAnthropicModel(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string // spliced model value; "" means ok=false (identity)
	}{
		{"plain body", `{"model":"test-model[1m]","max_tokens":8}`, "test-model"},
		{"whitespace around key and value", `{ "model" : "test-model[1m]" , "max_tokens":8}`, "test-model"},
		{"nested model key before the top-level one", `{"tools":[{"name":"t","input_schema":{"model":"nested[1m]"}}],"model":"test-model[1m]"}`, "test-model"},
		{"model literal inside a client string is untouched", `{"model":"test-model[1m]","messages":[{"role":"user","content":"say \"model\":\"zzz[1m]\" please"}]}`, "test-model"},
		{"missing model key", `{"max_tokens":8}`, ""},
		{"non-string model value", `{"model":8}`, ""},
		{"escaped byte inside the value", `{"model":"a\"b[1m]"}`, ""},
		{"unterminated value", `{"model":"test-model`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, ok := spliceAnthropicModel([]byte(tt.raw), "test-model")
			if tt.want == "" {
				if ok {
					t.Fatalf("spliceAnthropicModel(%s) succeeded, want identity", tt.raw)
				}
				return
			}
			if !ok {
				t.Fatalf("spliceAnthropicModel(%s) failed, want model %q", tt.raw, tt.want)
			}
			dec, err := agent.DecodeAnthropicMessagesRequest(out)
			if err != nil {
				t.Fatalf("spliced body does not decode: %v (%s)", err, out)
			}
			if dec.Model != tt.want {
				t.Fatalf("spliced model = %q, want %q (%s)", dec.Model, tt.want, out)
			}
		})
	}
}
