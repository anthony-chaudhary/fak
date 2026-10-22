package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const structuredOutputsTestCarrier = `{"regex":"[ab]{2}","disable_any_whitespace":true}`

func structuredOutputsChatBody(model string, stream bool, extra string) string {
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"answer"}],"max_tokens":4,"temperature":0,"stream":%t`, model, stream)
	if extra != "" {
		body += "," + extra
	}
	return body + "}"
}

// fak-test:runtime fast est=5ms lane=default
func TestStructuredOutputsCarrierJoinsGuidedDecodeFields(t *testing.T) {
	carriers := map[string]string{
		"guided_json":             `{"type":"object"}`,
		"guided_regex":            `"[ab]{2}"`,
		"guided_choice":           `["a","b"]`,
		"guided_grammar":          `"root ::= \"a\""`,
		"guided_decoding_backend": `"xgrammar"`,
		"json_schema":             `{"type":"object"}`,
		"regex":                   `"[ab]{2}"`,
		"ebnf":                    `"root ::= \"a\""`,
		"structured_outputs":      structuredOutputsTestCarrier,
	}
	parts := []string{`"model":"m"`}
	for _, key := range slices.Sorted(maps.Keys(carriers)) {
		parts = append(parts, fmt.Sprintf("%q:%s", key, carriers[key]))
	}
	var req ChatRequest
	if err := json.Unmarshal([]byte("{"+strings.Join(parts, ",")+"}"), &req); err != nil {
		t.Fatal(err)
	}
	got := req.GuidedDecodeFields()
	if len(got) != len(carriers) {
		t.Fatalf("guided carriers = %v, want exactly %v", slices.Sorted(maps.Keys(got)), slices.Sorted(maps.Keys(carriers)))
	}
	for key, raw := range carriers {
		if string(got[key]) != raw {
			t.Fatalf("carrier %s = %s, want %s", key, got[key], raw)
		}
	}
	var absent ChatRequest
	if err := json.Unmarshal([]byte(`{"model":"m"}`), &absent); err != nil {
		t.Fatal(err)
	}
	if fields := absent.GuidedDecodeFields(); fields != nil {
		t.Fatalf("absent carriers = %v, want nil", fields)
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestStructuredOutputsCarrierRideForwardsVerbatim(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	var mu sync.Mutex
	var forwarded map[string]json.RawMessage
	hits := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode upstream request: %v", err)
		}
		mu.Lock()
		forwarded = body
		hits++
		mu.Unlock()
		if string(body["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"model":"m","choices":[{"delta":{"role":"assistant","content":"ab"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`+"\n\n"+
				"data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"m","choices":[{"message":{"role":"assistant","content":"ab"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	t.Cleanup(upstream.Close)
	srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	const guidedRegex, guidedChoice = `"[ab]{2}"`, `["ab","ba"]`
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct {
			name  string
			extra string
			want  map[string]string
		}{
			{"structured_outputs_alone", `"structured_outputs":` + structuredOutputsTestCarrier, map[string]string{"structured_outputs": structuredOutputsTestCarrier}},
			{"with_existing_guided_fields", `"structured_outputs":` + structuredOutputsTestCarrier + `,"guided_regex":` + guidedRegex + `,"guided_choice":` + guidedChoice, map[string]string{"structured_outputs": structuredOutputsTestCarrier, "guided_regex": guidedRegex, "guided_choice": guidedChoice}},
			{"existing_guided_fields_alone", `"guided_regex":` + guidedRegex + `,"guided_choice":` + guidedChoice, map[string]string{"guided_regex": guidedRegex, "guided_choice": guidedChoice}},
			{"absent", "", map[string]string{}},
		} {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, tc.name), func(t *testing.T) {
				mu.Lock()
				before := hits
				mu.Unlock()
				rr := postNativeReceipt(t, srv, structuredOutputsChatBody("m", stream, tc.extra))
				if rr.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
				}
				mu.Lock()
				got, calls := forwarded, hits-before
				mu.Unlock()
				if calls != 1 {
					t.Fatalf("upstream hits=%d, want 1", calls)
				}
				for _, key := range []string{"structured_outputs", "guided_regex", "guided_choice"} {
					want, wanted := tc.want[key]
					raw, present := got[key]
					if present != wanted {
						t.Fatalf("upstream %s present=%t, want %t (body keys %v)", key, present, wanted, slices.Sorted(maps.Keys(got)))
					}
					if wanted && string(raw) != want {
						t.Fatalf("upstream %s = %s, want verbatim %s", key, raw, want)
					}
				}
			})
		}
	}
}

// fak-test:runtime fast est=60ms lane=default
func TestStructuredOutputsCarrierReachesPlannerCarrierMap(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, stream := range []bool{false, true} {
		for _, raw := range []string{structuredOutputsTestCarrier, `{}`, `null`} {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, raw), func(t *testing.T) {
				srv := newTestServer(t)
				p := &guidedRegexLimitPlanner{}
				srv.planner = p
				rr := postNativeReceipt(t, srv, structuredOutputsChatBody("m", stream, `"structured_outputs":`+raw+`,"guided_grammar":"root ::= \"a\""`))
				if rr.Code != http.StatusOK || p.calls != 1 {
					t.Fatalf("status=%d calls=%d body=%s", rr.Code, p.calls, rr.Body.String())
				}
				if len(p.guided) != 2 || string(p.guided["structured_outputs"]) != raw || string(p.guided["guided_grammar"]) != `"root ::= \"a\""` {
					t.Fatalf("planner carriers = %v, want structured_outputs=%s beside the unchanged guided_grammar", p.guided, raw)
				}
			})
		}
	}
}

// fak-test:runtime fast est=40ms lane=default
func TestStructuredOutputsCarrierMalformedRefusedBeforePlanner(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, stream := range []bool{false, true} {
		for _, raw := range []string{`"regex"`, `"{\"regex\":\"a\"}"`, `[]`, `[{"regex":"a"}]`, `17`, `true`, `false`} {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, raw), func(t *testing.T) {
				srv := newTestServer(t)
				p := &guidedRegexLimitPlanner{}
				srv.planner = p
				rr := postNativeReceipt(t, srv, structuredOutputsChatBody("m", stream, `"structured_outputs":`+raw))
				if rr.Code != http.StatusBadRequest || chatErrorCode(t, rr.Body.Bytes()) != "invalid_structured_outputs" {
					t.Fatalf("status=%d body=%s, want 400 invalid_structured_outputs", rr.Code, rr.Body.String())
				}
				if p.calls != 0 {
					t.Fatalf("planner calls=%d, want 0", p.calls)
				}
			})
		}
	}
}

// fak-test:runtime fast est=600ms lane=default
func TestStructuredOutputsCarrierNativeRouteRefusesInsteadOfDropping(t *testing.T) {
	srv := nativeReceiptServer(t)
	for _, stream := range []bool{false, true} {
		for _, extra := range []string{
			`"structured_outputs":` + structuredOutputsTestCarrier,
			`"structured_outputs":{"json":{"type":"object"}}`,
			`"structured_outputs":{"choice":["ab","ba"]}`,
			`"structured_outputs":{}`,
			`"structured_outputs":{"regex":"[ab]{2}"},"guided_regex":"[ab]{2}"`,
		} {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, extra), func(t *testing.T) {
				rr := postNativeReceipt(t, srv, structuredOutputsChatBody("synthetic-live", stream, extra))
				if rr.Code != http.StatusBadRequest || chatErrorCode(t, rr.Body.Bytes()) != "native_structured_output_unsupported" {
					t.Fatalf("status=%d body=%s, want 400 native_structured_output_unsupported", rr.Code, rr.Body.String())
				}
			})
		}
		t.Run(fmt.Sprintf("stream=%t/null_is_absent", stream), func(t *testing.T) {
			rr := postNativeReceipt(t, srv, structuredOutputsChatBody("synthetic-live", stream, `"structured_outputs":null`))
			if rr.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s, want 200", rr.Code, rr.Body.String())
			}
		})
	}
	rr := postNativeReceipt(t, srv, structuredOutputsChatBody("synthetic-live", false, `"guided_regex":"[ab]{2}"`))
	if rr.Code != http.StatusOK {
		t.Fatalf("guided_regex alone: status=%d body=%s, want 200", rr.Code, rr.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Choices) != 1 || !slices.Contains([]string{"aa", "ab", "ba", "bb"}, resp.Choices[0].Message.Content) {
		t.Fatalf("guided_regex alone: response %s, want content matching [ab]{2}", rr.Body.String())
	}
}
