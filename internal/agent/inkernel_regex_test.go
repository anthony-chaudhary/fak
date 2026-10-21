package agent

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"regexp"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/grammar"
)

func nativeRegexGuided(t *testing.T, carriers ...string) map[string]json.RawMessage {
	t.Helper()
	guided := map[string]json.RawMessage{}
	for i := 0; i+1 < len(carriers); i += 2 {
		raw, err := json.Marshal(carriers[i+1])
		if err != nil {
			t.Fatal(err)
		}
		guided[carriers[i]] = raw
	}
	return guided
}

func nativeRegexCheckFullMatch(t *testing.T, pattern, content string) {
	t.Helper()
	if !regexp.MustCompile(`^(?:` + pattern + `)$`).MatchString(content) {
		t.Fatalf("native response %q does not fully match %q", content, pattern)
	}
}

// fak-test:runtime fast est=250ms lane=default
func TestNativeGuidedRegexConstrainsCompletion(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	messages := []Message{{Role: RoleUser, Content: "Return the requested code."}}
	type sampling struct {
		name string
		temp float64
		seed int64
	}
	runs := []sampling{{"greedy", 0, 0}, {"seed_-3", 1, -3}, {"seed_7", 1, 7}, {"seed_42", 1, 42}}
	for _, tc := range []struct {
		name    string
		pattern string
		guided  map[string]json.RawMessage
		extra   []SampleOpt
	}{
		{"guided_regex", `[ab]{2}-[ab]`, nativeRegexGuided(t, "guided_regex", `[ab]{2}-[ab]`), nil},
		{"regex_alias", `[ab]{2}-[ab]`, nativeRegexGuided(t, "regex", `[ab]{2}-[ab]`), nil},
		{"both_carriers_agree", `[ab]{2}-[ab]`, nativeRegexGuided(t, "guided_regex", `[ab]{2}-[ab]`, "regex", `[ab]{2}-[ab]`), nil},
		{"with_text_format", `[ab]{2}-[ab]`, nativeRegexGuided(t, "guided_regex", `[ab]{2}-[ab]`), []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"text"}`))}},
		{"multibyte_runes", `(?:é|\x{4A69}){2}[0-9]`, nativeRegexGuided(t, "guided_regex", `(?:é|\x{4A69}){2}[0-9]`), nil},
		{"anchored", `^\d{3}(?:-[a-c]{2})?$`, nativeRegexGuided(t, "guided_regex", `^\d{3}(?:-[a-c]{2})?$`), nil},
	} {
		for _, run := range runs {
			t.Run(tc.name+"/"+run.name, func(t *testing.T) {
				temp, seed := run.temp, run.seed
				opts := append([]SampleOpt{WithGuidedDecode(tc.guided), WithTemperature(&temp), WithSeed(&seed)}, tc.extra...)
				comp, err := p.Complete(context.Background(), messages, nil, opts...)
				if err != nil {
					t.Fatal(err)
				}
				nativeRegexCheckFullMatch(t, tc.pattern, comp.Message.Content)
				if comp.FinishReason != "stop" {
					t.Fatalf("finish reason = %q, want stop", comp.FinishReason)
				}
			})
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeGuidedRegexStreamMatchesComplete(t *testing.T) {
	const pattern = `é(?:[ab]|é){1,3}-[0-9]`
	p := nativeStructuredTestPlanner(t)
	temp, seed := 0.8, int64(11)
	opts := []SampleOpt{WithGuidedDecode(nativeRegexGuided(t, "guided_regex", pattern)), WithTemperature(&temp), WithSeed(&seed)}
	want := ""
	for _, entry := range nativeGuidedEntries() {
		t.Run(entry.name, func(t *testing.T) {
			var streamed strings.Builder
			comp, err := entry.call(p, func(delta string) error {
				if !utf8.ValidString(delta) {
					t.Errorf("delta %q splits a UTF-8 rune", delta)
				}
				streamed.WriteString(delta)
				return nil
			}, opts...)
			if err != nil {
				t.Fatal(err)
			}
			nativeRegexCheckFullMatch(t, pattern, comp.Message.Content)
			if want == "" {
				want = comp.Message.Content
			} else if comp.Message.Content != want {
				t.Fatalf("%s content %q, complete %q", entry.name, comp.Message.Content, want)
			}
			if entry.name != "complete" && streamed.String() != comp.Message.Content {
				t.Fatalf("streamed %q, completion %q", streamed.String(), comp.Message.Content)
			}
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeGuidedRegexAfterReasoningSpan(t *testing.T) {
	const pattern = `[ab]{2}-[ab]`
	p := nativeStructuredReasoningPlanner(t, false)
	open, close, body, eos := nativeStructuredReasoningIDs(t, p)
	zero := 0.0
	opts := []SampleOpt{
		WithGuidedDecode(nativeRegexGuided(t, "guided_regex", pattern)), WithThinkingBudget(2), WithTemperature(&zero),
		WithLogitBias(map[int]float64{body: 100, open: 60, close: -100, eos: 200}),
		WithMaxTokens(48), WithDecodeTrace(true), WithNativeDecodeTokenIDs(true),
	}
	messages := []Message{{Role: RoleUser, Content: "Think briefly, then answer."}}
	check := func(t *testing.T, completion *Completion) {
		t.Helper()
		if completion.FinishReason != "stop" {
			t.Fatalf("finish reason = %q, want stop", completion.FinishReason)
		}
		nativeRegexCheckFullMatch(t, pattern, completion.Message.Content)
		if completion.Message.ReasoningContent != "xx" {
			t.Fatalf("reasoning = %q, want the two budgeted reasoning tokens", completion.Message.ReasoningContent)
		}
		ids := completion.NativeDecodeTokenIDs
		if ids == nil || len(ids.TokenIDs) < 5 || !slices.Equal(ids.TokenIDs[:4], []int{open, body, body, close}) {
			t.Fatalf("token IDs = %v, want opener, two reasoning tokens, then the forced close", ids)
		}
		if slices.Contains(ids.TokenIDs, eos) {
			t.Fatalf("EOS escaped the reasoning span or the regex: %v", ids.TokenIDs)
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
		t.Fatalf("reasoning leaked into the regex stream: streamed=%q content=%q buffered=%q", streamed.String(), stream.Message.Content, buffered.Message.Content)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeGuidedRegexRefusedBeforeSampling(t *testing.T) {
	type refusal struct {
		name    string
		opts    []SampleOpt
		carrier string
		pattern bool
	}
	guided := func(fields map[string]json.RawMessage) SampleOpt { return WithGuidedDecode(fields) }
	cases := []refusal{
		{"non_string_guided_regex", []SampleOpt{guided(map[string]json.RawMessage{"guided_regex": json.RawMessage(`42`)})}, "guided_regex", false},
		{"non_string_regex", []SampleOpt{guided(map[string]json.RawMessage{"regex": json.RawMessage(`["a"]`)})}, "regex", false},
		{"conflicting_carriers", []SampleOpt{guided(nativeRegexGuided(t, "guided_regex", "a", "regex", "b"))}, "regex", false},
		{"with_json_object_format", []SampleOpt{guided(nativeRegexGuided(t, "regex", "a")), WithResponseFormat(json.RawMessage(`{"type":"json_object"}`))}, "regex", false},
		{"with_regex_format", []SampleOpt{guided(nativeRegexGuided(t, "guided_regex", "a")), WithResponseFormat(json.RawMessage(`{"type":"regex","regex":"a"}`))}, "guided_regex", false},
	}
	for _, pattern := range []string{`\bab`, `(?m)^a$`, `(a)\1`, `a(?=b)`, `a^b`, `(a|b)*a(a|b){16}`} {
		cases = append(cases, refusal{"pattern_" + pattern, []SampleOpt{guided(nativeRegexGuided(t, "guided_regex", pattern))}, "", true})
	}
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
				if tc.pattern != errors.Is(err, grammar.ErrUnsupportedSchema) {
					t.Fatalf("pattern refusal = %v, want %v: %v", errors.Is(err, grammar.ErrUnsupportedSchema), tc.pattern, err)
				}
				if comp != nil || sampled != 0 || streamed != 0 {
					t.Fatalf("refusal ran decode: completion=%v sampled=%d streamed=%d", comp != nil, sampled, streamed)
				}
			})
		}
	}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}}
	if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "regex"}}, tools, guided(nativeRegexGuided(t, "guided_regex", "a"))); !errors.Is(err, ErrNativeStructuredUnsupported) {
		t.Fatalf("tools+regex: %v", err)
	}
	if sampled != 0 {
		t.Fatalf("tools+regex sampled %d times", sampled)
	}
}

// fak-test:runtime fast est=80ms lane=default
func TestNativeGuidedRegexMasksControlTokens(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	c, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: nativeRegexGuided(t, "guided_regex", `(?s).*`)}, stops)
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, p.m.Cfg.VocabSize)
	effective, err := c.maskLogits(logits, stops)
	if err != nil {
		t.Fatal(err)
	}
	special := p.tok.SpecialTokens()
	if len(special) == 0 {
		t.Fatal("probe tokenizer declares no special tokens")
	}
	for id, content := range special {
		finite := !math.IsInf(float64(effective[id]), -1)
		if finite != (id == c.stopID) {
			t.Fatalf("special token %d %q finite=%v, stop id %d", id, content, finite, c.stopID)
		}
	}
	ordinary, err := p.tok.Encode("a")
	if err != nil || len(ordinary) != 1 {
		t.Fatalf("encode a: %v %v", ordinary, err)
	}
	if math.IsInf(float64(effective[ordinary[0]]), -1) {
		t.Fatal("ordinary text token masked under (?s).*")
	}
}
