package agent

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
)

func nativeStructuredOutputsGuided(t *testing.T, fields map[string]any, extra ...string) map[string]json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	guided := map[string]json.RawMessage{"structured_outputs": raw}
	for i := 0; i+1 < len(extra); i += 2 {
		guided[extra[i]] = json.RawMessage(extra[i+1])
	}
	return guided
}

const nativeStructuredOutputsTestPattern = `[ab]{2}-[ab]`

var nativeStructuredOutputsTestChoices = []string{"positive", "negative", "neutral"}

func nativeStructuredOutputsCheckRegex(t *testing.T, content string) {
	t.Helper()
	nativeRegexCheckFullMatch(t, nativeStructuredOutputsTestPattern, content)
}

func nativeStructuredOutputsCheckChoice(t *testing.T, content string) {
	t.Helper()
	nativeGuidedCheckChoice(t, nativeStructuredOutputsTestChoices, content)
}

// fak-test:runtime fast est=700ms lane=default
func TestNativeStructuredOutputsConstrainsCompletion(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	schema := nativeGuidedJSONTestSchema
	schemaString := string(schema)
	reordered := json.RawMessage(` { "required":["answer"], "additionalProperties":false, "properties":{"answer":{"enum":["a","b"],"type":"string"}}, "type":"object" } `)
	choices := nativeStructuredOutputsTestChoices
	multibyte := []string{"é", "日本", "naïve"}
	pattern, _ := json.Marshal(nativeStructuredOutputsTestPattern)
	so := func(fields map[string]any, extra ...string) map[string]json.RawMessage {
		return nativeStructuredOutputsGuided(t, fields, extra...)
	}
	for _, tc := range []struct {
		name   string
		guided map[string]json.RawMessage
		extra  []SampleOpt
		check  func(*testing.T, string)
	}{
		{"json_object", so(map[string]any{"json": schema}), nil, nativeStructuredCheckAnswer},
		{"json_string", so(map[string]any{"json": schemaString}), nil, nativeStructuredCheckAnswer},
		{"json_compatible_options", so(map[string]any{"json": schema, "disable_any_whitespace": true, "disable_additional_properties": true, "whitespace_pattern": nil}), nil, nativeStructuredCheckAnswer},
		{"json_default_options", so(map[string]any{"json": schema, "disable_any_whitespace": false, "disable_additional_properties": false}), nil, nativeStructuredCheckAnswer},
		{"json_null_members", so(map[string]any{"json": schema, "regex": nil, "choice": nil, "grammar": nil, "json_object": nil, "structural_tag": nil}), nil, nativeStructuredCheckAnswer},
		{"json_with_text_format", so(map[string]any{"json": schema}), []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"text"}`))}, nativeStructuredCheckAnswer},
		{"json_same_as_response_format", so(map[string]any{"json": reordered}), []SampleOpt{WithResponseFormat(nativeStructuredTestFormat)}, nativeStructuredCheckAnswer},
		{"json_same_as_guided_json", so(map[string]any{"json": schemaString}, "guided_json", string(reordered)), nil, nativeStructuredCheckAnswer},
		{"regex", so(map[string]any{"regex": nativeStructuredOutputsTestPattern}), nil, nativeStructuredOutputsCheckRegex},
		{"regex_with_options", so(map[string]any{"regex": nativeStructuredOutputsTestPattern, "disable_any_whitespace": true}), nil, nativeStructuredOutputsCheckRegex},
		{"regex_same_as_guided_regex_and_alias", so(map[string]any{"regex": nativeStructuredOutputsTestPattern}, "guided_regex", string(pattern), "regex", string(pattern)), nil, nativeStructuredOutputsCheckRegex},
		{"choice", so(map[string]any{"choice": choices}), nil, nativeStructuredOutputsCheckChoice},
		{"choice_multibyte", so(map[string]any{"choice": multibyte}), nil, func(t *testing.T, content string) {
			t.Helper()
			nativeGuidedCheckChoice(t, multibyte, content)
		}},
		{"choice_same_as_guided_choice", so(map[string]any{"choice": append(slices.Clone(choices), choices...)}, "guided_choice", string(nativeGuidedChoiceRaw(t, choices...))), nil, nativeStructuredOutputsCheckChoice},
		{"sglang_json_schema_string", map[string]json.RawMessage{"json_schema": nativeGuidedJSONString(t, schema)}, nil, nativeStructuredCheckAnswer},
		{"sglang_json_schema_object", map[string]json.RawMessage{"json_schema": schema}, nil, nativeStructuredCheckAnswer},
		{"sglang_json_schema_same_as_guided_json", map[string]json.RawMessage{"json_schema": nativeGuidedJSONString(t, schema), "guided_json": reordered}, nil, nativeStructuredCheckAnswer},
		{"sglang_json_schema_same_as_response_format", map[string]json.RawMessage{"json_schema": nativeGuidedJSONString(t, reordered)}, []SampleOpt{WithResponseFormat(nativeStructuredTestFormat)}, nativeStructuredCheckAnswer},
		{"sglang_json_schema_same_as_structured_outputs", so(map[string]any{"json": reordered}, "json_schema", string(nativeGuidedJSONString(t, schema))), nil, nativeStructuredCheckAnswer},
	} {
		for _, run := range nativeGuidedSamplingRuns() {
			t.Run(tc.name+"/"+run.name, func(t *testing.T) {
				opts := append(append([]SampleOpt{WithGuidedDecode(tc.guided)}, run.opts()...), tc.extra...)
				nativeGuidedRunEntries(t, p, opts, tc.check)
			})
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeStructuredOutputsMatchesGuidedCarrier(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	pattern, _ := json.Marshal(nativeStructuredOutputsTestPattern)
	for _, tc := range []struct {
		name  string
		alias map[string]json.RawMessage
		base  map[string]json.RawMessage
	}{
		{"json", nativeStructuredOutputsGuided(t, map[string]any{"json": nativeGuidedJSONTestSchema}), map[string]json.RawMessage{"guided_json": nativeGuidedJSONTestSchema}},
		{"json_string", nativeStructuredOutputsGuided(t, map[string]any{"json": string(nativeGuidedJSONTestSchema)}), map[string]json.RawMessage{"guided_json": nativeGuidedJSONTestSchema}},
		{"sglang_json_schema", map[string]json.RawMessage{"json_schema": nativeGuidedJSONString(t, nativeGuidedJSONTestSchema)}, map[string]json.RawMessage{"guided_json": nativeGuidedJSONTestSchema}},
		{"regex", nativeStructuredOutputsGuided(t, map[string]any{"regex": nativeStructuredOutputsTestPattern}), map[string]json.RawMessage{"guided_regex": pattern}},
		{"choice", nativeStructuredOutputsGuided(t, map[string]any{"choice": nativeStructuredOutputsTestChoices}), map[string]json.RawMessage{"guided_choice": nativeGuidedChoiceRaw(t, nativeStructuredOutputsTestChoices...)}},
	} {
		for _, run := range nativeGuidedSamplingRuns() {
			t.Run(tc.name+"/"+run.name, func(t *testing.T) {
				traced := func(guided map[string]json.RawMessage) []SampleOpt {
					return append(run.opts(), WithDecodeTrace(true), WithNativeDecodeTokenIDs(true), WithGuidedDecode(guided))
				}
				messages := []Message{{Role: RoleUser, Content: "guided"}}
				base, err := p.Complete(context.Background(), messages, nil, traced(tc.base)...)
				if err != nil {
					t.Fatal(err)
				}
				alias, err := p.Complete(context.Background(), messages, nil, traced(tc.alias)...)
				if err != nil {
					t.Fatal(err)
				}
				if alias.Message.Content != base.Message.Content || alias.FinishReason != base.FinishReason {
					t.Fatalf("alias %q/%s, guided carrier %q/%s", alias.Message.Content, alias.FinishReason, base.Message.Content, base.FinishReason)
				}
				if base.NativeDecodeTokenIDs == nil || alias.NativeDecodeTokenIDs == nil || !slices.Equal(alias.NativeDecodeTokenIDs.TokenIDs, base.NativeDecodeTokenIDs.TokenIDs) {
					t.Fatalf("token IDs differ: alias %v, guided carrier %v", alias.NativeDecodeTokenIDs, base.NativeDecodeTokenIDs)
				}
				viaBase, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: tc.base}, stops)
				if err != nil {
					t.Fatal(err)
				}
				viaAlias, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: tc.alias}, stops)
				if err != nil {
					t.Fatal(err)
				}
				if viaAlias.stopID != viaBase.stopID {
					t.Fatalf("stop IDs %d / %d", viaAlias.stopID, viaBase.stopID)
				}
				logits := make([]float32, p.m.Cfg.VocabSize)
				for i := range logits {
					logits[i] = float32(i % 7)
				}
				ids := base.NativeDecodeTokenIDs.TokenIDs
				for step := 0; step <= len(ids); step++ {
					history := slices.DeleteFunc(slices.Clone(ids[:step]), func(id int) bool { return id == viaBase.stopID })
					viaBase.history = append(viaBase.history[:0], history...)
					viaAlias.history = append(viaAlias.history[:0], history...)
					baseMask, baseErr := viaBase.maskLogits(logits, stops)
					baseMask = slices.Clone(baseMask)
					aliasMask, aliasErr := viaAlias.maskLogits(logits, stops)
					if !errors.Is(aliasErr, baseErr) || !slices.Equal(aliasMask, baseMask) {
						t.Fatalf("step %d: alias mask differs from the guided carrier (%v / %v)", step, aliasErr, baseErr)
					}
				}
			})
		}
	}
}

// fak-test:runtime fast est=900ms lane=default
func TestNativeStructuredOutputsAfterReasoningSpan(t *testing.T) {
	choices := []string{"ab", "b"}
	for _, tc := range []struct {
		name   string
		guided func(*testing.T) map[string]json.RawMessage
		check  func(*testing.T, string)
	}{
		{"json", func(t *testing.T) map[string]json.RawMessage {
			return nativeStructuredOutputsGuided(t, map[string]any{"json": nativeGuidedJSONTestSchema, "disable_any_whitespace": true})
		}, nativeStructuredCheckAnswer},
		{"regex", func(t *testing.T) map[string]json.RawMessage {
			return nativeStructuredOutputsGuided(t, map[string]any{"regex": nativeStructuredOutputsTestPattern})
		}, nativeStructuredOutputsCheckRegex},
		{"choice", func(t *testing.T) map[string]json.RawMessage {
			return nativeStructuredOutputsGuided(t, map[string]any{"choice": choices})
		}, func(t *testing.T, content string) {
			t.Helper()
			nativeGuidedCheckChoice(t, choices, content)
		}},
		{"sglang_json_schema", func(t *testing.T) map[string]json.RawMessage {
			return map[string]json.RawMessage{"json_schema": nativeGuidedJSONString(t, nativeGuidedJSONTestSchema)}
		}, nativeStructuredCheckAnswer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := nativeStructuredReasoningPlanner(t, false)
			open, close, body, eos := nativeStructuredReasoningIDs(t, p)
			zero := 0.0
			opts := []SampleOpt{
				WithGuidedDecode(tc.guided(t)), WithThinkingBudget(2), WithTemperature(&zero),
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
				t.Fatalf("reasoning leaked into the stream: streamed=%q content=%q buffered=%q", streamed.String(), stream.Message.Content, buffered.Message.Content)
			}
		})
	}
}

// fak-test:runtime fast est=250ms lane=default
func TestNativeStructuredOutputsRefusedBeforeSampling(t *testing.T) {
	guided := func(fields map[string]json.RawMessage) SampleOpt { return WithGuidedDecode(fields) }
	raw := func(value string) map[string]json.RawMessage {
		return map[string]json.RawMessage{"structured_outputs": json.RawMessage(value)}
	}
	so := func(fields map[string]any, extra ...string) SampleOpt {
		return guided(nativeStructuredOutputsGuided(t, fields, extra...))
	}
	schema := nativeGuidedJSONTestSchema
	other := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string","enum":["a","c"]}},"required":["answer"],"additionalProperties":false}`)
	regex := map[string]any{"regex": "a"}
	nativeGuidedAssertRefusals(t, []nativeGuidedRefusal{
		{"string", []SampleOpt{guided(raw(`"regex"`))}, "structured_outputs"},
		{"array", []SampleOpt{guided(raw(`[{"regex":"a"}]`))}, "structured_outputs"},
		{"number", []SampleOpt{guided(raw(`17`))}, "structured_outputs"},
		{"empty_object", []SampleOpt{guided(raw(`{}`))}, "structured_outputs"},
		{"only_null_members", []SampleOpt{guided(raw(`{"json":null,"regex":null}`))}, "structured_outputs"},
		{"options_without_member", []SampleOpt{guided(raw(`{"disable_any_whitespace":true}`))}, "structured_outputs"},
		{"two_members", []SampleOpt{so(map[string]any{"json": schema, "regex": "a"})}, "structured_outputs"},
		{"regex_and_choice", []SampleOpt{so(map[string]any{"regex": "a", "choice": []string{"a"}})}, "structured_outputs"},
		{"json_object", []SampleOpt{guided(raw(`{"json_object":true}`))}, "structured_outputs.json_object"},
		{"json_object_false", []SampleOpt{guided(raw(`{"json_object":false}`))}, "structured_outputs.json_object"},
		{"grammar", []SampleOpt{guided(raw(`{"grammar":"root ::= \"a\""}`))}, "structured_outputs.grammar"},
		{"structural_tag", []SampleOpt{guided(raw(`{"structural_tag":"{}"}`))}, "structured_outputs.structural_tag"},
		{"json_with_grammar", []SampleOpt{so(map[string]any{"json": schema, "grammar": `root ::= "a"`})}, "structured_outputs.grammar"},
		{"whitespace_pattern", []SampleOpt{so(map[string]any{"json": schema, "whitespace_pattern": `[ ]?`})}, "structured_outputs.whitespace_pattern"},
		{"whitespace_pattern_empty", []SampleOpt{so(map[string]any{"regex": "a", "whitespace_pattern": ""})}, "structured_outputs.whitespace_pattern"},
		{"disable_any_whitespace_string", []SampleOpt{so(map[string]any{"json": schema, "disable_any_whitespace": "true"})}, "structured_outputs.disable_any_whitespace"},
		{"disable_any_whitespace_number", []SampleOpt{so(map[string]any{"json": schema, "disable_any_whitespace": 1})}, "structured_outputs.disable_any_whitespace"},
		{"disable_additional_properties_string", []SampleOpt{so(map[string]any{"json": schema, "disable_additional_properties": "yes"})}, "structured_outputs.disable_additional_properties"},
		{"disable_fallback", []SampleOpt{so(map[string]any{"json": schema, "disable_fallback": true})}, "structured_outputs.disable_fallback"},
		{"private_backend", []SampleOpt{so(map[string]any{"regex": "a", "_backend": "xgrammar"})}, "structured_outputs._backend"},
		{"unknown_option", []SampleOpt{so(map[string]any{"choice": []string{"a"}, "backend": "xgrammar"})}, "structured_outputs.backend"},
		{"json_number", []SampleOpt{guided(raw(`{"json":42}`))}, "structured_outputs.json"},
		{"json_array", []SampleOpt{guided(raw(`{"json":[{"type":"object"}]}`))}, "structured_outputs.json"},
		{"json_string_not_schema", []SampleOpt{guided(raw(`{"json":"not a schema"}`))}, "structured_outputs.json"},
		{"json_unsupported_schema", []SampleOpt{guided(raw(`{"json":{"type":"array"}}`))}, ""},
		{"regex_not_string", []SampleOpt{guided(raw(`{"regex":["a"]}`))}, "structured_outputs.regex"},
		{"choice_empty", []SampleOpt{guided(raw(`{"choice":[]}`))}, "structured_outputs.choice"},
		{"choice_number_member", []SampleOpt{guided(raw(`{"choice":["a",1]}`))}, "structured_outputs.choice"},
		{"choice_string", []SampleOpt{guided(raw(`{"choice":"a"}`))}, "structured_outputs.choice"},
		{"regex_with_guided_json", []SampleOpt{so(regex, "guided_json", string(schema))}, "guided_json"},
		{"json_with_guided_regex", []SampleOpt{so(map[string]any{"json": schema}, "guided_regex", `"a"`)}, "guided_regex"},
		{"json_with_guided_choice", []SampleOpt{so(map[string]any{"json": schema}, "guided_choice", `["a"]`)}, "guided_choice"},
		{"choice_with_regex_alias", []SampleOpt{so(map[string]any{"choice": []string{"a"}}, "regex", `"a"`)}, "regex"},
		{"regex_differs_from_guided_regex", []SampleOpt{so(regex, "guided_regex", `"b"`)}, "structured_outputs.regex"},
		{"regex_differs_from_alias", []SampleOpt{so(regex, "regex", `"b"`)}, "structured_outputs.regex"},
		{"json_differs_from_guided_json", []SampleOpt{so(map[string]any{"json": other}, "guided_json", string(schema))}, "structured_outputs.json"},
		{"choice_differs_from_guided_choice", []SampleOpt{so(map[string]any{"choice": []string{"a", "c"}}, "guided_choice", `["a","b"]`)}, "structured_outputs.choice"},
		{"choice_order_differs_from_guided_choice", []SampleOpt{so(map[string]any{"choice": []string{"b", "a"}}, "guided_choice", `["a","b"]`)}, "structured_outputs.choice"},
		{"with_guided_grammar", []SampleOpt{so(regex, "guided_grammar", `"root ::= \"a\""`)}, "guided_grammar"},
		{"with_backend", []SampleOpt{so(regex, "guided_decoding_backend", `"xgrammar"`)}, "guided_decoding_backend"},
		{"with_ebnf", []SampleOpt{so(map[string]any{"json": schema}, "ebnf", `"root ::= \"a\""`)}, "ebnf"},
		{"regex_with_json_schema_format", []SampleOpt{so(regex), WithResponseFormat(nativeStructuredTestFormat)}, "structured_outputs.regex"},
		{"choice_with_json_schema_format", []SampleOpt{so(map[string]any{"choice": []string{"a"}}), WithResponseFormat(nativeStructuredTestFormat)}, "structured_outputs.choice"},
		{"json_with_json_object_format", []SampleOpt{so(map[string]any{"json": schema}), WithResponseFormat(json.RawMessage(`{"type":"json_object"}`))}, "structured_outputs.json"},
		{"json_mismatched_response_format", []SampleOpt{so(map[string]any{"json": other}), WithResponseFormat(nativeStructuredTestFormat)}, "structured_outputs.json"},
		{"sglang_json_schema_number", []SampleOpt{guided(map[string]json.RawMessage{"json_schema": json.RawMessage(`42`)})}, "json_schema"},
		{"sglang_json_schema_not_schema", []SampleOpt{guided(map[string]json.RawMessage{"json_schema": json.RawMessage(`"not a schema"`)})}, "json_schema"},
		{"sglang_json_schema_unsupported", []SampleOpt{guided(map[string]json.RawMessage{"json_schema": json.RawMessage(`"{\"type\":\"array\"}"`)})}, ""},
		{"sglang_json_schema_differs_from_guided_json", []SampleOpt{guided(map[string]json.RawMessage{"guided_json": schema, "json_schema": nativeGuidedJSONString(t, other)})}, "json_schema"},
		{"sglang_json_schema_with_guided_regex", []SampleOpt{guided(map[string]json.RawMessage{"json_schema": nativeGuidedJSONString(t, schema), "guided_regex": json.RawMessage(`"a"`)})}, "guided_regex"},
		{"sglang_json_schema_mismatched_response_format", []SampleOpt{guided(map[string]json.RawMessage{"json_schema": nativeGuidedJSONString(t, other)}), WithResponseFormat(nativeStructuredTestFormat)}, "json_schema"},
		{"sglang_json_schema_with_json_object_format", []SampleOpt{guided(map[string]json.RawMessage{"json_schema": schema}), WithResponseFormat(json.RawMessage(`{"type":"json_object"}`))}, "json_schema"},
	})
	p := nativeStructuredTestPlanner(t)
	sampled := 0
	p.cachePrimeSamplerHook = func() { sampled++ }
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}}
	for _, fields := range []map[string]json.RawMessage{
		nativeStructuredOutputsGuided(t, map[string]any{"json": schema}),
		nativeStructuredOutputsGuided(t, map[string]any{"regex": "a"}),
		nativeStructuredOutputsGuided(t, map[string]any{"choice": []string{"a"}}),
		{"json_schema": nativeGuidedJSONString(t, schema)},
	} {
		if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "guided"}}, tools, WithGuidedDecode(fields)); !errors.Is(err, ErrNativeStructuredUnsupported) {
			t.Fatalf("tools+%v: %v", fields, err)
		}
	}
	if sampled != 0 {
		t.Fatalf("tools+structured_outputs sampled %d times", sampled)
	}
}
