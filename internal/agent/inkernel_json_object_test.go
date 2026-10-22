package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var nativeJSONObjectTestFormat = json.RawMessage(`{"type":"json_object"}`)

// nativeJSONObjectCheck requires exactly one JSON object and reports how many
// containers deep it nests.
func nativeJSONObjectCheck(t *testing.T, content string) int {
	t.Helper()
	if !json.Valid([]byte(content)) || !strings.HasPrefix(content, "{") {
		t.Fatalf("native response is not one JSON object: %q", content)
	}
	var object map[string]any
	if err := json.Unmarshal([]byte(content), &object); err != nil {
		t.Fatalf("native response does not decode into an object: %q: %v", content, err)
	}
	decoder := json.NewDecoder(strings.NewReader(content))
	depth, deepest := 0, 0
	for {
		token, err := decoder.Token()
		if err != nil {
			break
		}
		switch token {
		case json.Delim('{'), json.Delim('['):
			depth++
			deepest = max(deepest, depth)
		case json.Delim('}'), json.Delim(']'):
			depth--
		}
	}
	return deepest
}

// nativeJSONObjectBias raises structural bytes over the probe tokenizer's other
// byte tokens so a random synthetic model closes strings and containers
// quickly while still choosing among them.
func nativeJSONObjectBias(t *testing.T, p *InKernelPlanner) map[int]float64 {
	t.Helper()
	bias := map[int]float64{}
	for text, weight := range map[string]float64{
		`"`: 7, ":": 6, "}": 6, "]": 6, "[": 6, "{": 5, ",": 4, "1": 4, "-": 2, ".": 2, "e": 2,
		"t": 3, "f": 3, "n": 3, `\`: 2, "é": 3,
	} {
		ids, err := p.tok.Encode(text)
		if err != nil || len(ids) == 0 {
			t.Fatalf("encode %q: ids=%v err=%v", text, ids, err)
		}
		bias[ids[0]] = weight
	}
	return bias
}

// fak-test:runtime fast est=400ms lane=default
func TestNativeJSONObjectCompleteAndStream(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	bias := nativeJSONObjectBias(t, p)
	deepest := 0
	for _, seed := range []int64{0, 1, 2, 3, 5, 7, 11, 13, -3, 42} {
		t.Run("seed_"+strconv.FormatInt(seed, 10), func(t *testing.T) {
			temperature := 1.0
			if seed == 0 {
				temperature = 0
			}
			opts := []SampleOpt{WithResponseFormat(nativeJSONObjectTestFormat), WithTemperature(&temperature), WithSeed(&seed), WithLogitBias(bias), WithMaxTokens(192)}
			nativeGuidedRunEntries(t, p, opts, func(t *testing.T, content string) {
				deepest = max(deepest, nativeJSONObjectCheck(t, content))
			})
		})
	}
	if deepest < 2 {
		t.Fatalf("no sampled object nested a container (deepest %d); the bias no longer exercises nesting", deepest)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeJSONObjectAfterReasoningSpan(t *testing.T) {
	p := nativeStructuredReasoningPlanner(t, false)
	open, close, body, eos := nativeStructuredReasoningIDs(t, p)
	brace, err := p.tok.Encode("}")
	if err != nil || len(brace) != 1 {
		t.Fatalf("encode brace: %v %v", brace, err)
	}
	zero := 0.0
	opts := []SampleOpt{
		WithResponseFormat(nativeJSONObjectTestFormat), WithThinkingBudget(2), WithTemperature(&zero),
		WithLogitBias(map[int]float64{body: 100, brace[0]: 90, open: 60, close: -100, eos: 200}),
		WithMaxTokens(48), WithDecodeTrace(true), WithNativeDecodeTokenIDs(true),
	}
	messages := []Message{{Role: RoleUser, Content: "Think briefly, then answer."}}
	check := func(t *testing.T, completion *Completion) {
		t.Helper()
		if completion.FinishReason != "stop" || completion.Message.Content != "{}" {
			t.Fatalf("finish=%q content=%q, want stop and the empty object", completion.FinishReason, completion.Message.Content)
		}
		if completion.Message.ReasoningContent != "xx" {
			t.Fatalf("reasoning = %q, want the two budgeted reasoning tokens", completion.Message.ReasoningContent)
		}
		ids := completion.NativeDecodeTokenIDs
		if ids == nil || len(ids.TokenIDs) < 5 || !slices.Equal(ids.TokenIDs[:4], []int{open, body, body, close}) {
			t.Fatalf("token IDs = %v, want opener, two reasoning tokens, then the forced close", ids)
		}
		if slices.Contains(ids.TokenIDs, eos) {
			t.Fatalf("EOS escaped the reasoning span or the object: %v", ids.TokenIDs)
		}
	}
	buffered, err := p.Complete(context.Background(), messages, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	check(t, buffered)
	var streamed strings.Builder
	stream, err := p.CompleteStream(context.Background(), func(piece string) error {
		streamed.WriteString(piece)
		return nil
	}, messages, nil, append(opts, WithPerTokenStream(true))...)
	if err != nil {
		t.Fatal(err)
	}
	check(t, stream)
	if streamed.String() != stream.Message.Content {
		t.Fatalf("reasoning leaked into the object stream: streamed=%q content=%q", streamed.String(), stream.Message.Content)
	}
}

// fak-test:runtime fast est=60ms lane=default
func TestNativeJSONObjectMasksControlTokensUntilComplete(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	encode := func(text string) []int {
		t.Helper()
		ids, err := p.tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	special := p.tok.SpecialTokens()
	if len(special) == 0 {
		t.Fatal("probe tokenizer declares no special tokens")
	}
	for _, tc := range []struct {
		history  string
		want     string
		complete bool
	}{
		{"", "{", false},
		{`{"a":"`, "", false},
		{`{"a":[1,{"b":null}]}`, "", true},
	} {
		c, err := p.newNativeRequestConstraint(SampleParams{ResponseFormat: nativeJSONObjectTestFormat}, stops)
		if err != nil {
			t.Fatal(err)
		}
		c.history = encode(tc.history)
		effective, err := c.maskLogits(make([]float32, p.m.Cfg.VocabSize), stops)
		if err != nil {
			t.Fatalf("after %q: %v", tc.history, err)
		}
		for id := range special {
			if finite := !math.IsInf(float64(effective[id]), -1); finite != (tc.complete && id == c.stopID) {
				t.Fatalf("after %q special token %d finite=%v, stop id %d", tc.history, id, finite, c.stopID)
			}
		}
		var admitted []byte
		for id, logit := range effective {
			if math.IsInf(float64(logit), -1) || id == c.stopID {
				continue
			}
			piece, err := p.tok.Decode([]int{id})
			if err != nil {
				t.Fatal(err)
			}
			admitted = append(admitted, piece...)
		}
		switch {
		case tc.want != "" && string(admitted) != tc.want:
			t.Fatalf("after %q admitted %q, want %q", tc.history, admitted, tc.want)
		case tc.complete && len(admitted) != 0:
			t.Fatalf("complete object still admits %q", admitted)
		case tc.history == `{"a":"` && (!bytes.Contains(admitted, []byte("<")) || !bytes.Contains(admitted, []byte(`"`))):
			t.Fatalf("string content after %q is not free text: %q", tc.history, admitted)
		}
	}
}

// fak-test:runtime fast est=80ms lane=default
func TestNativeJSONObjectRefusedBeforeSampling(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	sampled := 0
	p.cachePrimeSamplerHook = func() { sampled++ }
	messages := []Message{{Role: RoleUser, Content: "json"}}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}}
	for _, tc := range []struct {
		name  string
		tools []ToolDef
		opts  []SampleOpt
	}{
		{"schema_member", nil, []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"json_object","schema":{"type":"object"}}`))}},
		{"json_schema_member", nil, []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"json_object","json_schema":{"schema":{"type":"object"}}}`))}},
		{"not_an_object", nil, []SampleOpt{WithResponseFormat(json.RawMessage(`["json_object"]`))}},
		{"with_tools", tools, []SampleOpt{WithResponseFormat(nativeJSONObjectTestFormat)}},
		{"with_guided_json", nil, []SampleOpt{WithResponseFormat(nativeJSONObjectTestFormat), WithGuidedDecode(map[string]json.RawMessage{"guided_json": json.RawMessage(`{"type":"object"}`)})}},
	} {
		for _, entry := range nativeGuidedEntries() {
			t.Run(tc.name+"/"+entry.name, func(t *testing.T) {
				var comp *Completion
				var err error
				if tc.tools != nil {
					comp, err = p.Complete(context.Background(), messages, tc.tools, tc.opts...)
				} else {
					comp, err = entry.call(p, func(string) error { return nil }, tc.opts...)
				}
				if !errors.Is(err, ErrNativeStructuredUnsupported) || comp != nil {
					t.Fatalf("completion=%v err=%v, want ErrNativeStructuredUnsupported", comp != nil, err)
				}
			})
		}
	}
	_, err := p.Complete(context.Background(), messages, nil, WithResponseFormat(nativeJSONObjectTestFormat), WithNativeInferenceReceipt(true))
	var unsupported *model.NativeInferenceReceiptUnsupportedError
	if !errors.As(err, &unsupported) || !errors.Is(err, ErrNativeStructuredUnsupported) {
		t.Fatalf("unmodified receipt+json_object: %v", err)
	}
	if sampled != 0 {
		t.Fatalf("refused json_object request sampled %d times", sampled)
	}
	zero := 0.0
	comp, err := p.Complete(context.Background(), messages, nil, WithResponseFormat(json.RawMessage(` {"schema":null,"type":"json_object"} `)), WithTemperature(&zero), WithLogitBias(nativeJSONObjectBias(t, p)), WithMaxTokens(96))
	if err != nil {
		t.Fatalf("json_object with a null member: %v", err)
	}
	nativeJSONObjectCheck(t, comp.Message.Content)
}
