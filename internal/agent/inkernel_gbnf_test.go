package agent

import (
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/grammar"
)

func nativeGBNFRaw(t *testing.T, source string) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// fak-test:runtime fast est=2s lane=default
func TestNativeGBNFCompletion(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	source := `root ::= start tail
start ::= "a" | "b"
tail ::= "a"{1,2}`
	raw := nativeGBNFRaw(t, source)
	for _, tc := range []struct {
		name   string
		guided map[string]json.RawMessage
		extra  []SampleOpt
	}{
		{"direct", map[string]json.RawMessage{"guided_grammar": raw}, nil},
		{"with_text", map[string]json.RawMessage{"guided_grammar": raw}, []SampleOpt{WithResponseFormat(json.RawMessage(`{"type":"text"}`))}},
		{"structured_outputs", nativeStructuredOutputsGuided(t, map[string]any{"grammar": source}), nil},
		{"same_aliases", nativeStructuredOutputsGuided(t, map[string]any{"grammar": source}, "guided_grammar", string(raw)), nil},
		{"false_options", nativeStructuredOutputsGuided(t, map[string]any{"grammar": source, "disable_any_whitespace": false, "disable_additional_properties": false}), nil},
	} {
		for _, run := range nativeGuidedSamplingRuns() {
			t.Run(tc.name+"/"+run.name, func(t *testing.T) {
				opts := append([]SampleOpt{WithGuidedDecode(tc.guided), WithMaxTokens(24)}, run.opts()...)
				opts = append(opts, tc.extra...)
				nativeGuidedRunEntries(t, p, opts, func(t *testing.T, content string) {
					t.Helper()
					if !slices.Contains([]string{"aa", "aaa", "ba", "baa"}, content) {
						t.Fatalf("outside complete grammar: %q", content)
					}
				})
			})
		}
	}
	// Preserve the exact formerly refused valid text-format carrier as a
	// positive normal-consumer witness rather than silently deleting its case.
	nativeGuidedRunEntries(t, p, []SampleOpt{
		WithGuidedDecode(map[string]json.RawMessage{"guided_grammar": nativeGuidedTestCarriers["guided_grammar"]}),
		WithResponseFormat(json.RawMessage(`{"type":"text"}`)), WithMaxTokens(8),
	}, func(t *testing.T, content string) {
		if content != "a" {
			t.Fatalf("got %q", content)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestNativeGBNFRefusedBeforeDecode(t *testing.T) {
	raw := func(s string) map[string]json.RawMessage {
		return map[string]json.RawMessage{"guided_grammar": nativeGBNFRaw(t, s)}
	}
	g := func(m map[string]json.RawMessage) SampleOpt { return WithGuidedDecode(m) }
	nativeGuidedAssertRefusals(t, []nativeGuidedRefusal{
		{"undefined", []SampleOpt{g(raw("root ::= missing"))}, "guided_grammar"},
		{"left_recursive", []SampleOpt{g(raw(`root ::= root "a" | ""`))}, "guided_grammar"},
		{"empty", []SampleOpt{g(raw(""))}, "guided_grammar"},
		{"not_string", []SampleOpt{g(map[string]json.RawMessage{"guided_grammar": json.RawMessage(`{"root":"a"}`)})}, "guided_grammar"},
		{"special_id_extension", []SampleOpt{g(raw("root ::= <[12]>"))}, "guided_grammar"},
		{"different_alias", []SampleOpt{g(nativeStructuredOutputsGuided(t, map[string]any{"grammar": `root ::= "b"`}, "guided_grammar", string(nativeGBNFRaw(t, `root ::= "a"`))))}, "structured_outputs.grammar"},
		{"schema_mix", []SampleOpt{g(raw(`root ::= "a"`)), WithResponseFormat(nativeStructuredTestFormat)}, "guided_grammar"},
		{"choice_mix", []SampleOpt{g(map[string]json.RawMessage{"guided_grammar": nativeGBNFRaw(t, `root ::= "a"`), "guided_choice": json.RawMessage(`["a"]`)})}, "guided_grammar"},
		{"whitespace_option", []SampleOpt{g(nativeStructuredOutputsGuided(t, map[string]any{"grammar": `root ::= "a"`, "disable_any_whitespace": true}))}, "structured_outputs.disable_any_whitespace"},
		{"property_option", []SampleOpt{g(nativeStructuredOutputsGuided(t, map[string]any{"grammar": `root ::= "a"`, "disable_additional_properties": true}))}, "structured_outputs.disable_additional_properties"},
		{"other_dialect", []SampleOpt{g(map[string]json.RawMessage{"ebnf": nativeGBNFRaw(t, `root ::= "a"`)})}, "ebnf"},
	})
}

type nativeGBNFDecoder [][]byte

func (d nativeGBNFDecoder) TokenBytes(id int) []byte {
	if id < 0 || id >= len(d) {
		return nil
	}
	return d[id]
}

// fak-test:runtime fast est=1s lane=default
func TestNativeGBNFMaskForkSpecialAndEOS(t *testing.T) {
	dec := nativeGBNFDecoder{[]byte("a"), []byte("ab"), []byte("cd"), []byte("a"), []byte("b"), []byte("c"), []byte("d"), []byte("ab")}
	mask, err := grammar.CompileGBNF(`root ::= "ab" | "cd"`, nativeRegexTokenDecoder{tok: dec, special: map[int]string{0: "eos", 7: "control"}}, grammar.CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	parent := &nativeStructuredConstraint{mask: mask, stopID: 0, openID: -1, closeID: -1}
	stops := map[int]bool{0: true}
	logits := []float32{100, 1, 1, 1, 1, 1, 1, 1000}
	originalLogits := slices.Clone(logits)
	effective, err := parent.maskLogits(logits, stops)
	if err != nil {
		t.Fatal(err)
	}
	if nativeStructuredLogitFinite(effective[0]) || nativeStructuredLogitFinite(effective[7]) {
		t.Fatal("EOS or control admitted before completion")
	}
	if err = parent.accept(3, effective); err != nil {
		t.Fatal(err)
	}
	fork := parent.fork()
	fork.history = []int{5}
	effective, err = fork.maskLogits(logits, stops)
	if err != nil {
		t.Fatal(err)
	}
	if !nativeStructuredLogitFinite(effective[6]) || nativeStructuredLogitFinite(effective[4]) {
		t.Fatal("fork branch lost")
	}
	effective, err = parent.maskLogits(logits, stops)
	if err != nil {
		t.Fatal(err)
	}
	if !nativeStructuredLogitFinite(effective[4]) || nativeStructuredLogitFinite(effective[6]) {
		t.Fatal("shared mask cursor contaminated parent")
	}
	if err = parent.accept(4, effective); err != nil {
		t.Fatal(err)
	}
	effective, err = parent.maskLogits(logits, stops)
	if err != nil {
		t.Fatal(err)
	}
	if !nativeStructuredLogitFinite(effective[0]) {
		t.Fatal("EOS not admitted by ID")
	}
	if err = parent.accept(0, effective); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(parent.history, []int{3, 4}) {
		t.Fatal("EOS entered grammar text history")
	}
	// Rejection/rollback uses the supplied history, never a stale shared cursor.
	parent.history = nil
	effective, err = parent.maskLogits(logits, stops)
	if err != nil {
		t.Fatal(err)
	}
	if !nativeStructuredLogitFinite(effective[3]) || !nativeStructuredLogitFinite(effective[5]) {
		t.Fatal("rollback did not return to root")
	}
	if !slices.Equal(logits, originalLogits) {
		t.Fatal("original logits mutated")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeGBNFResourceFailureStopsSelection(t *testing.T) {
	alts := make([]string, 256)
	for i := range alts {
		alts[i] = `"a" ("b" | "c")`
	}
	mask, err := grammar.CompileGBNF("root ::= "+strings.Join(alts, " | "), nativeGBNFDecoder{nil, []byte("a"), []byte("b"), []byte("c")}, grammar.CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	c := &nativeStructuredConstraint{mask: mask, stopID: 0, openID: -1, closeID: -1}
	got, err := c.maskLogits([]float32{100, 200, 300, 400}, map[int]bool{0: true})
	if got != nil || !errors.Is(err, ErrNativeStructuredNoToken) || !errors.Is(err, grammar.ErrGBNF) {
		t.Fatalf("got %v / %v", got, err)
	}
	for _, v := range c.maskBuffer {
		if !math.IsInf(float64(v), -1) {
			t.Fatal("resource failure did not fail closed")
		}
	}
	if len(c.history) != 0 {
		t.Fatal("resource failure advanced history")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeGBNFAbsentKeepsOrdinaryRoute(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	for _, guided := range []map[string]json.RawMessage{nil, {"guided_grammar": json.RawMessage(`null`)}} {
		c, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: guided}, stops)
		if err != nil || c != nil {
			t.Fatalf("absent grammar changed ordinary route: %v %v", c, err)
		}
	}
	// Null grammar alongside the pre-existing JSON member remains absent.
	c, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: nativeStructuredOutputsGuided(t, map[string]any{"json": nativeGuidedJSONTestSchema, "grammar": nil})}, stops)
	if err != nil || c == nil {
		t.Fatalf("JSON+null grammar changed: %v %v", c, err)
	}
}
