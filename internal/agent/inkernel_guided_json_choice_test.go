package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

var nativeGuidedJSONTestSchema = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string","enum":["a","b"]}},"required":["answer"],"additionalProperties":false}`)

func nativeGuidedChoiceRaw(t *testing.T, choices ...string) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(choices)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func nativeGuidedJSONString(t *testing.T, schema json.RawMessage) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(string(schema))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func nativeGuidedCheckChoice(t *testing.T, choices []string, content string) {
	t.Helper()
	if !slices.Contains(choices, content) {
		t.Fatalf("native response %q is not exactly one of %q", content, choices)
	}
}

type nativeGuidedSampling struct {
	name string
	opts func() []SampleOpt
}

func nativeGuidedSamplingRuns() []nativeGuidedSampling {
	fixed := func(temp float64, seed int64, extra ...SampleOpt) func() []SampleOpt {
		return func() []SampleOpt {
			return append([]SampleOpt{WithTemperature(&temp), WithSeed(&seed)}, extra...)
		}
	}
	topK, topP, minP := 4, 0.9, 0.05
	return []nativeGuidedSampling{
		{"greedy", fixed(0, 0)},
		{"seed_7", fixed(1, 7)},
		{"seed_-3_min_p_top_k_top_p", fixed(0.8, -3, WithMinP(&minP), WithTopK(&topK), WithTopP(&topP))},
	}
}

func nativeGuidedRunEntries(t *testing.T, p *InKernelPlanner, opts []SampleOpt, check func(*testing.T, string)) {
	t.Helper()
	want := ""
	for _, entry := range nativeGuidedEntries() {
		var streamed strings.Builder
		comp, err := entry.call(p, func(delta string) error {
			if !utf8.ValidString(delta) {
				t.Errorf("%s delta %q splits a UTF-8 rune", entry.name, delta)
			}
			streamed.WriteString(delta)
			return nil
		}, opts...)
		if err != nil {
			t.Fatalf("%s: %v", entry.name, err)
		}
		check(t, comp.Message.Content)
		if comp.FinishReason != "stop" {
			t.Fatalf("%s finish reason = %q, want stop", entry.name, comp.FinishReason)
		}
		if want == "" {
			want = comp.Message.Content
		} else if comp.Message.Content != want {
			t.Fatalf("%s content %q, complete %q", entry.name, comp.Message.Content, want)
		}
		if entry.name != "complete" && streamed.String() != comp.Message.Content {
			t.Fatalf("%s streamed %q, completion %q", entry.name, streamed.String(), comp.Message.Content)
		}
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestNativeGuidedJSONConstrainsCompletion(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	reordered := json.RawMessage(` { "required":["answer"], "additionalProperties":false, "properties":{"answer":{"enum":["a","b"],"type":"string"}}, "type":"object" } `)
	for _, tc := range []struct {
		name   string
		guided json.RawMessage
		extra  []SampleOpt
	}{
		{"object", nativeGuidedJSONTestSchema, nil},
		{"json_string", nativeGuidedJSONString(t, nativeGuidedJSONTestSchema), nil},
		{"with_text_format", nativeGuidedJSONTestSchema, []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"text"}`))}},
		{"same_schema_as_response_format", reordered, []SampleOpt{WithResponseFormat(nativeStructuredTestFormat)}},
		{"json_string_same_as_response_format", nativeGuidedJSONString(t, reordered), []SampleOpt{WithResponseFormat(nativeStructuredTestFormat)}},
	} {
		for _, run := range nativeGuidedSamplingRuns() {
			t.Run(tc.name+"/"+run.name, func(t *testing.T) {
				opts := append(append([]SampleOpt{WithGuidedDecode(map[string]json.RawMessage{"guided_json": tc.guided})}, run.opts()...), tc.extra...)
				nativeGuidedRunEntries(t, p, opts, nativeStructuredCheckAnswer)
			})
		}
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeGuidedJSONSharesResponseFormatSubset(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	seventeen, seventeenRequired := map[string]any{}, []string{}
	for i := range 17 {
		name := fmt.Sprintf("p%02d", i)
		seventeen[name] = map[string]any{"type": "boolean"}
		seventeenRequired = append(seventeenRequired, name)
	}
	for _, tc := range []struct {
		name   string
		schema any
		admit  bool
	}{
		{"enum_string", json.RawMessage(nativeGuidedJSONTestSchema), true},
		{"nested_object", map[string]any{"type": "object", "properties": map[string]any{"outer": map[string]any{"type": "object", "properties": map[string]any{"ok": map[string]any{"type": "boolean"}}, "required": []string{"ok"}, "additionalProperties": false}}, "required": []string{"outer"}, "additionalProperties": false}, true},
		{"integer_enum", map[string]any{"type": "object", "properties": map[string]any{"count": map[string]any{"type": "integer", "enum": []int{1, 2, 3}}}, "required": []string{"count"}, "additionalProperties": false}, true},
		{"scalar_array", map[string]any{"type": "object", "properties": map[string]any{"tags": map[string]any{"type": "array", "items": map[string]any{"type": "boolean"}, "maxItems": 2}}, "required": []string{"tags"}, "additionalProperties": false}, true},
		{"top_level_array", map[string]any{"type": "array"}, false},
		{"additional_properties_true", map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}, "required": []string{"answer"}, "additionalProperties": true}, false},
		{"min_length", map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string", "minLength": 1}}, "required": []string{"answer"}, "additionalProperties": false}, false},
		{"seventeen_properties", map[string]any{"type": "object", "properties": seventeen, "required": seventeenRequired, "additionalProperties": false}, false},
		{"over_64_kib", map[string]any{"type": "object", "description": strings.Repeat("x", 65536), "properties": map[string]any{"answer": map[string]any{"type": "string"}}, "required": []string{"answer"}, "additionalProperties": false}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			schema, err := json.Marshal(tc.schema)
			if err != nil {
				t.Fatal(err)
			}
			viaFormat, formatErr := p.newNativeRequestConstraint(SampleParams{ResponseFormat: nativeStructuredFormatForSchema(t, tc.schema)}, stops)
			for _, guided := range []json.RawMessage{schema, nativeGuidedJSONString(t, schema)} {
				viaGuided, guidedErr := p.newNativeRequestConstraint(SampleParams{GuidedDecode: map[string]json.RawMessage{"guided_json": guided}}, stops)
				if (formatErr == nil) != tc.admit || (guidedErr == nil) != tc.admit {
					t.Fatalf("admit = response_format:%v guided_json:%v, want %v (%v / %v)", formatErr == nil, guidedErr == nil, tc.admit, formatErr, guidedErr)
				}
				if !tc.admit {
					if !errors.Is(formatErr, ErrNativeStructuredUnsupported) || !errors.Is(guidedErr, ErrNativeStructuredUnsupported) {
						t.Fatalf("refusals = %v / %v, want ErrNativeStructuredUnsupported", formatErr, guidedErr)
					}
					continue
				}
				logits := make([]float32, p.m.Cfg.VocabSize)
				for i := range logits {
					logits[i] = float32(i % 5)
				}
				formatMask, err := viaFormat.maskLogits(logits, stops)
				if err != nil {
					t.Fatal(err)
				}
				formatMask = slices.Clone(formatMask)
				guidedMask, err := viaGuided.maskLogits(logits, stops)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(formatMask, guidedMask) || viaFormat.stopID != viaGuided.stopID {
					t.Fatal("guided_json and response_format compiled different first-token masks")
				}
			}
		})
	}
}

type nativeGuidedRefusal struct {
	name    string
	opts    []SampleOpt
	carrier string
}

func nativeGuidedAssertRefusals(t *testing.T, cases []nativeGuidedRefusal) {
	t.Helper()
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
				var refused *nativeGuidedCarrierError
				if tc.carrier != "" && (!errors.As(err, &refused) || refused.Carrier != tc.carrier) {
					t.Fatalf("refused carrier = %+v, want %q", refused, tc.carrier)
				}
				if comp != nil || sampled != 0 || streamed != 0 {
					t.Fatalf("refusal ran decode: completion=%v sampled=%d streamed=%d", comp != nil, sampled, streamed)
				}
			})
		}
	}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}}
	for _, guided := range []map[string]json.RawMessage{
		{"guided_json": nativeGuidedJSONTestSchema},
		{"guided_choice": nativeGuidedChoiceRaw(t, "a", "b")},
	} {
		if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "guided"}}, tools, WithGuidedDecode(guided)); !errors.Is(err, ErrNativeStructuredUnsupported) {
			t.Fatalf("tools+%v: %v", guided, err)
		}
	}
	if sampled != 0 {
		t.Fatalf("tools+guided sampled %d times", sampled)
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeGuidedJSONRefusedBeforeSampling(t *testing.T) {
	guided := func(fields map[string]json.RawMessage) SampleOpt { return WithGuidedDecode(fields) }
	schema := nativeGuidedJSONTestSchema
	other := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string","enum":["a","c"]}},"required":["answer"],"additionalProperties":false}`)
	malformed := func(name, raw string) nativeGuidedRefusal {
		return nativeGuidedRefusal{name, []SampleOpt{guided(map[string]json.RawMessage{"guided_json": json.RawMessage(raw)})}, "guided_json"}
	}
	nativeGuidedAssertRefusals(t, []nativeGuidedRefusal{
		malformed("number", `42`),
		malformed("boolean", `true`),
		malformed("array", `[{"type":"object"}]`),
		malformed("empty_string", `""`),
		malformed("string_not_json", `"not a schema"`),
		malformed("string_array", `"[{\"type\":\"object\"}]"`),
		malformed("string_trailing_data", `"{\"type\":\"object\"} {}"`),
		malformed("invalid_json", `{"type":`),
		{"unsupported_schema", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": json.RawMessage(`{"type":"array"}`)})}, ""},
		{"mismatched_response_format", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema}), WithResponseFormat(nativeStructuredFormatForSchema(t, other))}, "guided_json"},
		{"mismatched_response_format_json_string", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": nativeGuidedJSONString(t, other)}), WithResponseFormat(nativeStructuredTestFormat)}, "guided_json"},
		{"response_format_strict_false", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema}), WithResponseFormat(json.RawMessage(`{"type":"json_schema","json_schema":{"strict":false,"schema":` + string(schema) + `}}`))}, ""},
		{"with_json_object_format", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema}), WithResponseFormat(json.RawMessage(`{"type":"json_object"}`))}, "guided_json"},
		{"with_regex_format", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema}), WithResponseFormat(json.RawMessage(`{"type":"regex","regex":"a"}`))}, "guided_json"},
		{"with_guided_regex", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema, "guided_regex": json.RawMessage(`"a"`)})}, "guided_json"},
		{"with_regex_alias", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema, "regex": json.RawMessage(`"a"`)})}, "guided_json"},
		{"with_guided_choice", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema, "guided_choice": json.RawMessage(`["a"]`)})}, "guided_choice"},
		{"with_backend", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema, "guided_decoding_backend": json.RawMessage(`"xgrammar"`)})}, "guided_decoding_backend"},
		{"with_ebnf", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema, "ebnf": json.RawMessage(`"root ::= \"a\""`)})}, "ebnf"},
		{"with_sglang_json_schema", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema, "json_schema": nativeGuidedJSONString(t, schema)})}, "json_schema"},
	})
}

// fak-test:runtime fast est=250ms lane=default
func TestNativeGuidedChoiceConstrainsCompletion(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	for _, tc := range []struct {
		name    string
		choices []string
		extra   []SampleOpt
	}{
		{"labels", []string{"positive", "negative", "neutral"}, nil},
		{"metacharacters", []string{`a.c`, `(x)`, `1+1=2`, `$5`, `a|b`, `[^z]*`, `\d`, `^`}, nil},
		{"multibyte", []string{"é", "日本", "naïve"}, nil},
		{"duplicates", []string{"yes", "no", "yes", "no"}, nil},
		{"with_text_format", []string{"yes", "no"}, []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"text"}`))}},
	} {
		for _, run := range nativeGuidedSamplingRuns() {
			t.Run(tc.name+"/"+run.name, func(t *testing.T) {
				opts := append(append([]SampleOpt{WithGuidedDecode(map[string]json.RawMessage{"guided_choice": nativeGuidedChoiceRaw(t, tc.choices...)})}, run.opts()...), tc.extra...)
				nativeGuidedRunEntries(t, p, opts, func(t *testing.T, content string) {
					t.Helper()
					nativeGuidedCheckChoice(t, tc.choices, content)
				})
			})
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeGuidedChoicePrefixMembers(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	one := func(text string) int {
		t.Helper()
		ids, err := p.tok.Encode(text)
		if err != nil || len(ids) != 1 {
			t.Fatalf("encode %q: ids=%v err=%v", text, ids, err)
		}
		return ids[0]
	}
	a, b, c := one("a"), one("b"), one("c")
	choices := []string{"abc", "a", "ab"}
	guided := WithGuidedDecode(map[string]json.RawMessage{"guided_choice": nativeGuidedChoiceRaw(t, choices...)})

	stops := StopIDs(p.tok, p.m.Cfg)
	constraint, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: map[string]json.RawMessage{"guided_choice": nativeGuidedChoiceRaw(t, choices...)}}, stops)
	if err != nil {
		t.Fatal(err)
	}
	eos := constraint.stopID
	for _, step := range []struct {
		history []int
		admit   []int
	}{
		{nil, []int{a}},
		{[]int{a}, []int{b, eos}},
		{[]int{a, b}, []int{c, eos}},
		{[]int{a, b, c}, []int{eos}},
	} {
		constraint.history = append(constraint.history[:0], step.history...)
		effective, err := constraint.maskLogits(make([]float32, p.m.Cfg.VocabSize), stops)
		if err != nil {
			t.Fatal(err)
		}
		var admitted []int
		for id, logit := range effective {
			if !math.IsInf(float64(logit), -1) {
				admitted = append(admitted, id)
			}
		}
		slices.Sort(step.admit)
		if !slices.Equal(admitted, step.admit) {
			t.Fatalf("after %v admitted %v, want %v", step.history, admitted, step.admit)
		}
	}

	zero := 0.0
	for _, tc := range []struct {
		want string
		bias map[int]float64
	}{
		{"a", map[int]float64{a: 100, eos: 100}},
		{"ab", map[int]float64{a: 100, b: 100, c: -100}},
		{"abc", map[int]float64{a: 100, b: 100, c: 100, eos: -100}},
	} {
		t.Run(tc.want, func(t *testing.T) {
			opts := []SampleOpt{guided, WithTemperature(&zero), WithLogitBias(tc.bias)}
			nativeGuidedRunEntries(t, p, opts, func(t *testing.T, content string) {
				t.Helper()
				if content != tc.want {
					t.Fatalf("content = %q, want %q", content, tc.want)
				}
			})
		})
	}
}

func nativeGuidedChoiceMembers(n, width int) []string {
	const metacharacters = `()[]*$?|.+^\{}`
	members := make([]string, n)
	for i := range members {
		members[i] = fmt.Sprintf("%02x", i) + metacharacters[:width-2]
	}
	return members
}

// fak-test:runtime fast est=250ms lane=default
func TestNativeGuidedChoiceBoundsAdmitAtCap(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	atCap := nativeGuidedChoiceMembers(128, 8)
	withDuplicates := append(slices.Clone(atCap), atCap...)
	for name, choices := range map[string][]string{"at_cap": atCap, "duplicates_over_raw_cap": withDuplicates} {
		t.Run(name, func(t *testing.T) {
			if _, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: map[string]json.RawMessage{"guided_choice": nativeGuidedChoiceRaw(t, choices...)}}, stops); err != nil {
				t.Fatal(err)
			}
		})
	}
	seed, temp := int64(5), 1.0
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "pick"}}, nil,
		WithGuidedDecode(map[string]json.RawMessage{"guided_choice": nativeGuidedChoiceRaw(t, atCap...)}), WithTemperature(&temp), WithSeed(&seed))
	if err != nil {
		t.Fatal(err)
	}
	nativeGuidedCheckChoice(t, atCap, comp.Message.Content)
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeGuidedChoiceRefusedBeforeSampling(t *testing.T) {
	guided := func(fields map[string]json.RawMessage) SampleOpt { return WithGuidedDecode(fields) }
	choice := func(raw json.RawMessage) map[string]json.RawMessage {
		return map[string]json.RawMessage{"guided_choice": raw}
	}
	malformed := func(name string, raw json.RawMessage) nativeGuidedRefusal {
		return nativeGuidedRefusal{name, []SampleOpt{guided(choice(raw))}, "guided_choice"}
	}
	overBytes := nativeGuidedChoiceMembers(128, 8)
	overBytes[0] += "?"
	ab := nativeGuidedChoiceRaw(t, "a", "b")
	with := func(key string, raw json.RawMessage) map[string]json.RawMessage {
		return map[string]json.RawMessage{"guided_choice": ab, key: raw}
	}
	nativeGuidedAssertRefusals(t, []nativeGuidedRefusal{
		malformed("empty_array", json.RawMessage(`[]`)),
		malformed("string", json.RawMessage(`"a"`)),
		malformed("object", json.RawMessage(`{"a":"b"}`)),
		malformed("number_member", json.RawMessage(`["a",1]`)),
		malformed("null_member", json.RawMessage(`["a",null]`)),
		malformed("boolean_member", json.RawMessage(`[true]`)),
		malformed("nested_array", json.RawMessage(`[["a"]]`)),
		malformed("empty_member", json.RawMessage(`["a",""]`)),
		malformed("only_empty", json.RawMessage(`[""]`)),
		malformed("invalid_json", json.RawMessage(`["a"`)),
		malformed("over_count", nativeGuidedChoiceRaw(t, nativeGuidedChoiceMembers(129, 3)...)),
		malformed("over_bytes", nativeGuidedChoiceRaw(t, overBytes...)),
		{"with_json_schema_format", []SampleOpt{guided(choice(ab)), WithResponseFormat(nativeStructuredTestFormat)}, "guided_choice"},
		{"with_json_object_format", []SampleOpt{guided(choice(ab)), WithResponseFormat(json.RawMessage(`{"type":"json_object"}`))}, "guided_choice"},
		{"with_guided_regex", []SampleOpt{guided(with("guided_regex", json.RawMessage(`"a"`)))}, "guided_choice"},
		{"with_regex_alias", []SampleOpt{guided(with("regex", json.RawMessage(`"a"`)))}, "guided_choice"},
		{"with_guided_grammar", []SampleOpt{guided(with("guided_grammar", json.RawMessage(`"root ::= \"a\""`)))}, "guided_grammar"},
		{"with_backend", []SampleOpt{guided(with("guided_decoding_backend", json.RawMessage(`"xgrammar"`)))}, "guided_decoding_backend"},
	})
}

// fak-test:runtime fast est=400ms lane=default
func TestNativeGuidedJSONAndChoiceAfterReasoningSpan(t *testing.T) {
	choices := []string{"ab", "b"}
	for _, tc := range []struct {
		name   string
		guided map[string]json.RawMessage
		check  func(*testing.T, string)
	}{
		{"guided_json", map[string]json.RawMessage{"guided_json": nativeGuidedJSONTestSchema}, nativeStructuredCheckAnswer},
		{"guided_choice", map[string]json.RawMessage{"guided_choice": nativeGuidedChoiceRaw(t, choices...)}, func(t *testing.T, content string) {
			t.Helper()
			nativeGuidedCheckChoice(t, choices, content)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := nativeStructuredReasoningPlanner(t, false)
			open, close, body, eos := nativeStructuredReasoningIDs(t, p)
			zero := 0.0
			opts := []SampleOpt{
				WithGuidedDecode(tc.guided), WithThinkingBudget(2), WithTemperature(&zero),
				WithLogitBias(map[int]float64{body: 100, open: 60, close: -100, eos: 200}),
				WithMaxTokens(48), WithDecodeTrace(true), WithNativeDecodeTokenIDs(true),
			}
			messages := []Message{{Role: RoleUser, Content: "Think briefly, then answer."}}
			check := func(t *testing.T, completion *Completion) {
				t.Helper()
				if completion.FinishReason != "stop" {
					t.Fatalf("finish reason = %q, want stop", completion.FinishReason)
				}
				tc.check(t, completion.Message.Content)
				if completion.Message.ReasoningContent != "xx" {
					t.Fatalf("reasoning = %q, want the two budgeted reasoning tokens", completion.Message.ReasoningContent)
				}
				ids := completion.NativeDecodeTokenIDs
				if ids == nil || len(ids.TokenIDs) < 5 || !slices.Equal(ids.TokenIDs[:4], []int{open, body, body, close}) {
					t.Fatalf("token IDs = %v, want opener, two reasoning tokens, then the forced close", ids)
				}
				if slices.Contains(ids.TokenIDs, eos) {
					t.Fatalf("EOS escaped the reasoning span or the constraint: %v", ids.TokenIDs)
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
			if streamed.String() != stream.Message.Content || stream.Message.Content != buffered.Message.Content {
				t.Fatalf("reasoning leaked into the guided stream: streamed=%q content=%q buffered=%q", streamed.String(), stream.Message.Content, buffered.Message.Content)
			}
		})
	}
}
