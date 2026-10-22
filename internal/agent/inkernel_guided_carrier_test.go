package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

var nativeGuidedTestCarriers = map[string]json.RawMessage{
	"guided_json":             json.RawMessage(`{"type":"object"}`),
	"guided_regex":            json.RawMessage(`"[ab]{2}"`),
	"guided_choice":           json.RawMessage(`["a","b"]`),
	"guided_grammar":          json.RawMessage(`"root ::= \"a\""`),
	"guided_decoding_backend": json.RawMessage(`"xgrammar"`),
	"json_schema":             json.RawMessage(`{"type":"object"}`),
	"regex":                   json.RawMessage(`"[ab]{2}"`),
	"ebnf":                    json.RawMessage(`"root ::= \"a\""`),
	"structured_outputs":      json.RawMessage(`{"regex":"[ab]{2}"}`),
}

type nativeGuidedEntry struct {
	name string
	call func(p *InKernelPlanner, sink StreamSink, opts ...SampleOpt) (*Completion, error)
}

func nativeGuidedEntries() []nativeGuidedEntry {
	messages := []Message{{Role: RoleUser, Content: "guided"}}
	return []nativeGuidedEntry{
		{"complete", func(p *InKernelPlanner, _ StreamSink, opts ...SampleOpt) (*Completion, error) {
			return p.Complete(context.Background(), messages, nil, opts...)
		}},
		{"stream_buffered", func(p *InKernelPlanner, sink StreamSink, opts ...SampleOpt) (*Completion, error) {
			return p.CompleteStream(context.Background(), sink, messages, nil, append(opts, WithPerTokenStream(false))...)
		}},
		{"stream_per_token", func(p *InKernelPlanner, sink StreamSink, opts ...SampleOpt) (*Completion, error) {
			return p.CompleteStream(context.Background(), sink, messages, nil, append(opts, WithPerTokenStream(true))...)
		}},
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestNativeGuidedCarrierRefusedBeforeSampling(t *testing.T) {
	type refusal struct {
		name    string
		opts    []SampleOpt
		carrier string
	}
	var cases []refusal
	for key, value := range nativeGuidedTestCarriers {
		if slices.Contains(nativeGuidedCarriers[:], key) {
			// Enforced natively; refusals of bad enforced carriers live in
			// inkernel_regex_test.go, inkernel_guided_json_choice_test.go and
			// inkernel_structured_outputs_test.go.
			continue
		}
		cases = append(cases, refusal{key, []SampleOpt{WithGuidedDecode(map[string]json.RawMessage{key: value})}, key})
	}
	cases = append(cases,
		refusal{"guided_with_json_schema", []SampleOpt{WithResponseFormat(nativeStructuredTestFormat), WithGuidedDecode(map[string]json.RawMessage{"guided_regex": nativeGuidedTestCarriers["guided_regex"]})}, "guided_regex"},
		refusal{"regex_alias_with_json_schema", []SampleOpt{WithResponseFormat(nativeStructuredTestFormat), WithGuidedDecode(map[string]json.RawMessage{"regex": nativeGuidedTestCarriers["regex"]})}, "regex"},
		refusal{"regex_with_guided_json", []SampleOpt{WithGuidedDecode(map[string]json.RawMessage{"guided_regex": nativeGuidedTestCarriers["guided_regex"], "guided_json": nativeGuidedTestCarriers["guided_json"]})}, "guided_json"},
		refusal{"regex_with_backend", []SampleOpt{WithGuidedDecode(map[string]json.RawMessage{"regex": nativeGuidedTestCarriers["regex"], "guided_decoding_backend": nativeGuidedTestCarriers["guided_decoding_backend"]})}, "guided_decoding_backend"},
		refusal{"guided_with_text", []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"text"}`)), WithGuidedDecode(map[string]json.RawMessage{"guided_grammar": nativeGuidedTestCarriers["guided_grammar"]})}, "guided_grammar"},
		refusal{"several_carriers", []SampleOpt{WithGuidedDecode(map[string]json.RawMessage{"regex": nativeGuidedTestCarriers["regex"], "guided_regex": nativeGuidedTestCarriers["guided_regex"], "ebnf": nativeGuidedTestCarriers["ebnf"]})}, "ebnf"},
		refusal{"response_format_json_object_with_schema", []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"json_object","schema":{"type":"object"}}`))}, ""},
		refusal{"guided_regex_with_json_object", []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"json_object"}`)), WithGuidedDecode(map[string]json.RawMessage{"guided_regex": nativeGuidedTestCarriers["guided_regex"]})}, "guided_regex"},
		refusal{"response_format_regex", []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"regex","regex":"[ab]{2}"}`))}, ""},
		refusal{"response_format_grammar", []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"grammar","grammar":"root ::= \"a\""}`))}, ""},
	)
	p := nativeStructuredTestPlanner(t)
	sampled, streamed := 0, 0
	p.cachePrimeSamplerHook = func() { sampled++ }
	for _, entry := range nativeGuidedEntries() {
		for _, tc := range cases {
			t.Run(entry.name+"/"+tc.name, func(t *testing.T) {
				comp, err := entry.call(p, func(string) error { streamed++; return nil }, tc.opts...)
				if !errors.Is(err, ErrNativeStructuredUnsupported) {
					t.Fatalf("err = %v, want ErrNativeStructuredUnsupported", err)
				}
				var guided *nativeGuidedCarrierError
				if tc.carrier != "" && (!errors.As(err, &guided) || guided.Carrier != tc.carrier) {
					t.Fatalf("refused carrier = %+v, want %q", guided, tc.carrier)
				}
				if comp != nil || sampled != 0 || streamed != 0 {
					t.Fatalf("refusal ran decode: completion=%v sampled=%d streamed=%d", comp != nil, sampled, streamed)
				}
				if _, ok := p.LatestNativePhaseObservation(); ok {
					t.Fatal("refused request reached the native decode path")
				}
			})
		}
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestNativeGuidedCarrierAbsentKeepsOrdinaryRoute(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	sampled := 0
	p.cachePrimeSamplerHook = func() { sampled++ }
	plain, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "guided"}}, nil, WithMaxTokens(3))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		opts []SampleOpt
	}{
		{"absent", nil},
		{"null_carriers", []SampleOpt{WithGuidedDecode(map[string]json.RawMessage{"guided_regex": json.RawMessage(`null`), "ebnf": json.RawMessage(" null "), "json_schema": json.RawMessage(`null`), "structured_outputs": json.RawMessage(`null`)})}},
		{"response_format_text", []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"text"}`))}},
	} {
		for _, entry := range nativeGuidedEntries() {
			t.Run(tc.name+"/"+entry.name, func(t *testing.T) {
				before := sampled
				comp, err := entry.call(p, func(string) error { return nil }, append(tc.opts, WithMaxTokens(3))...)
				if err != nil {
					t.Fatal(err)
				}
				if comp.Message.Content != plain.Message.Content || comp.Usage.CompletionTokens != plain.Usage.CompletionTokens {
					t.Fatalf("got %q/%d, want ordinary %q/%d", comp.Message.Content, comp.Usage.CompletionTokens, plain.Message.Content, plain.Usage.CompletionTokens)
				}
				if sampled == before {
					t.Fatal("ordinary request did not sample")
				}
			})
		}
	}
}

// fak-test:runtime fast est=450ms lane=default
func TestNativeGuidedCarrierJSONSchemaKeepsWorking(t *testing.T) {
	nullGuided := WithGuidedDecode(map[string]json.RawMessage{"guided_json": json.RawMessage(`null`)})
	p := nativeStructuredTestPlanner(t)
	for _, entry := range nativeGuidedEntries() {
		t.Run(entry.name, func(t *testing.T) {
			var streamed []byte
			comp, err := entry.call(p, func(delta string) error { streamed = append(streamed, delta...); return nil }, WithResponseFormat(nativeStructuredTestFormat), nullGuided)
			if err != nil {
				t.Fatal(err)
			}
			nativeStructuredCheckAnswer(t, comp.Message.Content)
			if entry.name != "complete" && string(streamed) != comp.Message.Content {
				t.Fatalf("streamed %q, completion %q", streamed, comp.Message.Content)
			}
		})
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestNativeGuidedCarrierPassthroughForwardsUnchanged(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)
	p := NewHTTPPlanner(srv.URL, "m", "")
	if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "guided"}}, nil, WithGuidedDecode(nativeGuidedTestCarriers)); err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	for key, want := range nativeGuidedTestCarriers {
		if string(wire[key]) != string(want) {
			t.Fatalf("upstream %s = %s, want %s", key, wire[key], want)
		}
	}
}
