package gateway

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"
)

type spyMaskRecognizer struct {
	prefix         []byte
	accept         func([]byte) bool
	allowByte      func([]byte, byte) bool
	attempts       int
	acceptedPushes int
	pops           int
}

func (s *spyMaskRecognizer) TryPushByte(b byte) bool {
	s.attempts++
	if !s.allowByte(s.prefix, b) {
		return false
	}
	s.acceptedPushes++
	s.prefix = append(s.prefix, b)
	return true
}

func (s *spyMaskRecognizer) PopBytes(n int) {
	s.pops += n
	s.prefix = s.prefix[:len(s.prefix)-n]
}

func (s *spyMaskRecognizer) AllowEOS() bool {
	return s.accept(s.prefix)
}

// fak-test:runtime fast est=5ms lane=default
func TestGrammarMasksTokenTrieSubtreePruningAndBitset(t *testing.T) {
	pieces := [][]byte{
		[]byte("<eos>"), // 0: EOS (must never match via byte trie)
		[]byte("<pad>"), // 1: special token (excluded)
		nil,             // 2: empty token (excluded)
		[]byte("ok_a"),  // 3
		[]byte("ok_b"),  // 4
		[]byte("ok_b"),  // 5: duplicate spelling of token 4
		[]byte("ok"),    // 6: prefix of ok_a/ok_b
	}
	const badCount = 64
	for i := 0; i < badCount; i++ {
		pieces = append(pieces, []byte(fmt.Sprintf("bad_%02d", i)))
	}

	// Copy piece 3 so we can verify TokenTrie snapshots caller slices defensively.
	mutCheck := append([]byte(nil), pieces[3]...)
	dec := sliceTokenDecoder(pieces)
	trie, err := NewTokenTrie(len(pieces), 0, dec, 1)
	if err != nil {
		t.Fatalf("NewTokenTrie: %v", err)
	}
	pieces[3][0] = 'X' // mutate caller storage after trie construction
	if got := string(trie.TokenBytes(3)); got != string(mutCheck) {
		t.Fatalf("TokenBytes(3) mutated to %q, want %q", got, mutCheck)
	}

	spy := &spyMaskRecognizer{
		allowByte: func(prefix []byte, b byte) bool {
			// Admit only bytes along "ok_b" (plus "ok" prefix).
			want := "ok_b"
			if len(prefix) >= len(want) {
				return false
			}
			return b == want[len(prefix)]
		},
		accept: func(prefix []byte) bool {
			return string(prefix) == "ok_b"
		},
	}

	bitset, err := trie.AllowedTokens(spy)
	if err != nil {
		t.Fatalf("AllowedTokens: %v", err)
	}
	if spy.acceptedPushes != spy.pops || len(spy.prefix) != 0 {
		t.Fatalf("unbalanced recognizer stack: accepted=%d pops=%d remaining=%q", spy.acceptedPushes, spy.pops, spy.prefix)
	}
	// Subtree "bad_*" has 64 tokens all sharing 'b'; rejecting 'b' at root must skip all 64 in one attempt.
	if spy.attempts > 10 {
		t.Fatalf("expected subtree pruning to skip bad_* subtree, got %d TryPushByte attempts", spy.attempts)
	}

	// Token 0 (EOS) is admitted because spy.accept("ok_b") is false at root!
	// Wait: at root prefix == "", spy.AllowEOS() is false; "ok" (6), "ok_b" (4), "ok_b" dup (5) are allowed.
	wantAllowed := []int{4, 5, 6}
	if got := bitset.AllowedIDs(); !slices.Equal(got, wantAllowed) {
		t.Fatalf("AllowedIDs = %v, want %v", got, wantAllowed)
	}
	if bitset.Contains(0) || bitset.Contains(1) || bitset.Contains(2) || bitset.Contains(3) {
		t.Fatalf("unexpected special/empty/disallowed token in %v", bitset.AllowedIDs())
	}

	// Now advance spy to "ok_b" and check EOS admission + logit monotonicity.
	spy.prefix = []byte("ok_b")
	bitset2, err := trie.AllowedTokens(spy)
	if err != nil {
		t.Fatalf("AllowedTokens at accepting state: %v", err)
	}
	if !slices.Equal(bitset2.AllowedIDs(), []int{0}) {
		t.Fatalf("at accepting state with no continuations, AllowedIDs = %v, want [0]", bitset2.AllowedIDs())
	}

	// ApplyToLogits must never resurrect pre-existing -inf logits (even if EOS=0 were -inf).
	logits := make([]float32, len(pieces)+8)
	for i := range logits {
		logits[i] = float32(i)
	}
	logits[0] = float32(math.Inf(-1))
	for i := len(pieces); i < len(logits); i++ {
		logits[i] = float32(math.Inf(-1))
	}
	receipt, err := bitset.ApplyToLogits(logits)
	if err != nil {
		t.Fatalf("ApplyToLogits: %v", err)
	}
	if !receipt.DomainMonotonicityOK || receipt.WidenedTokens != 0 {
		t.Fatalf("unexpected monotonicity receipt: %+v", receipt)
	}
	if !math.IsInf(float64(logits[0]), -1) {
		t.Fatalf("expected pre-existing -inf token 0 to stay -inf")
	}
	for _, id := range wantAllowed {
		if math.IsInf(float64(logits[id]), -1) {
			t.Fatalf("expected allowed token %d to stay finite", id)
		}
	}
	if !math.IsInf(float64(logits[3]), -1) {
		t.Fatalf("expected disallowed token 3 to be masked to -inf")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGrammarMasksJSONSchemaPerStepAllowLists(t *testing.T) {
	schema := []byte(`{
		"type": "object",
		"properties": {
			"a": {"type": "string"},
			"b": {"type": "boolean"},
			"c": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
			"d": {"type": "string", "pattern": "^[a-z]{2}[0-9]$"}
		},
		"required": ["a"],
		"additionalProperties": false
	}`)
	grammar, err := CompileJSONSchemaGrammar(schema, GrammarOptions{CompactJSON: true})
	if err != nil {
		t.Fatalf("CompileJSONSchemaGrammar: %v", err)
	}

	const eos = 0
	pieces := [][]byte{
		[]byte("<eos>"),              // 0: EOS
		[]byte(`{"a":"`),             // 1: exact forced prefix
		[]byte(`hi"`),                // 2: closes "a"
		[]byte(`}`),                  // 3: close object with optional b,c,d omitted (also valid char inside string)
		[]byte(`,"b":true`),          // 4: optional b
		[]byte(`,"c":[10,20]`),       // 5: optional c
		[]byte(`,"d":"ab1"}`),        // 6: valid d + close
		[]byte(`,"d":"AB1"}`),        // 7: invalid uppercase for pattern
		[]byte(`,"b":true}tail`),     // 8: trailing garbage after object
		[]byte(`{"b":true,"a":"x"}`), // 9: wrong canonical key order
		[]byte(`,"extra":1}`),        // 10: violates additionalProperties:false
		[]byte(`{"a":"x"`),           // 11: partial without closing
		[]byte(`,"b":true,"b":false}`),
		[]byte("\n"), // 13: raw newline illegal inside JSON string
	}
	trie, err := NewTokenTrieFromPieces(pieces, eos)
	if err != nil {
		t.Fatalf("NewTokenTrieFromPieces: %v", err)
	}
	eng, err := NewGrammarMaskEngine(trie, grammar)
	if err != nil {
		t.Fatalf("NewGrammarMaskEngine: %v", err)
	}

	// Step 0: forced prefix {"a":" is uniquely determined and token 1 matches it.
	step0, err := eng.StepMask()
	if err != nil {
		t.Fatalf("StepMask 0: %v", err)
	}
	if step0.Accepting || step0.Stopped {
		t.Fatalf("unexpected initial terminal state: %+v", step0)
	}
	if string(step0.ForcedBytes) != `{"a":"` {
		t.Fatalf("ForcedBytes = %q, want %q", step0.ForcedBytes, `{"a":"`)
	}
	if !slices.Equal(step0.ForcedTokens, []int{1}) {
		t.Fatalf("ForcedTokens = %v, want [1]", step0.ForcedTokens)
	}
	if got := step0.Allowed.AllowedIDs(); !slices.Equal(got, []int{1, 11}) {
		t.Fatalf("step 0 allowed = %v, want [1 11]", got)
	}

	// Committing an illegal token must fail transactionally without mutating state.
	if err := eng.CommitToken(9); !errors.Is(err, ErrGrammarRejectedToken) {
		t.Fatalf("expected ErrGrammarRejectedToken on wrong-order token, got %v", err)
	}
	if len(eng.MatchedBytes()) != 0 {
		t.Fatalf("failed CommitToken mutated MatchedBytes: %q", eng.MatchedBytes())
	}

	// Commit token 1 (`{"a":"`), then check step 1: `hi"` (2) and `}` (3, valid string char) are allowed, raw "\n" (13) is rejected.
	if err := eng.CommitToken(1); err != nil {
		t.Fatalf("CommitToken(1): %v", err)
	}
	allowed1, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens 1: %v", err)
	}
	if !slices.Equal(allowed1, []int{2, 3}) {
		t.Fatalf("step 1 allowed = %v, want [2 3]", allowed1)
	}
	if err := eng.CommitToken(2); err != nil {
		t.Fatalf("CommitToken(2): %v", err)
	}

	// After `{"a":"hi"`, valid continuations are `}` (3), `,"b":true` (4), `,"c":[10,20]` (5), `,"d":"ab1"}` (6).
	allowed2, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens 2: %v", err)
	}
	if !slices.Equal(allowed2, []int{3, 4, 5, 6}) {
		t.Fatalf("step 2 allowed = %v, want [3 4 5 6]", allowed2)
	}

	// Walk through b -> c -> d -> EOS, checking invalid pattern/trailing/duplicate tokens stay excluded.
	if err := eng.CommitToken(4); err != nil {
		t.Fatalf("CommitToken(4): %v", err)
	}
	allowed3, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens 3: %v", err)
	}
	if !slices.Equal(allowed3, []int{3, 5, 6}) {
		t.Fatalf("step 3 allowed = %v, want [3 5 6]", allowed3)
	}
	if err := eng.CommitToken(5); err != nil {
		t.Fatalf("CommitToken(5): %v", err)
	}
	allowed4, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens 4: %v", err)
	}
	if !slices.Equal(allowed4, []int{3, 6}) {
		t.Fatalf("step 4 allowed = %v, want [3 6] (must reject pattern mismatch token 7)", allowed4)
	}
	if err := eng.CommitToken(6); err != nil {
		t.Fatalf("CommitToken(6): %v", err)
	}

	// Object complete: only EOS is admitted.
	stepFinal, err := eng.StepMask()
	if err != nil {
		t.Fatalf("StepMask final: %v", err)
	}
	if !stepFinal.Accepting || stepFinal.Stopped {
		t.Fatalf("expected Accepting=true Stopped=false, got %+v", stepFinal)
	}
	if got := stepFinal.Allowed.AllowedIDs(); !slices.Equal(got, []int{eos}) {
		t.Fatalf("final allowed = %v, want [%d]", got, eos)
	}

	// ApplyMask on logits with a padded tail -inf verifies domain monotonicity.
	logits := make([]float32, len(pieces)+2)
	for i := range logits {
		logits[i] = 1.0
	}
	logits[len(pieces)] = float32(math.Inf(-1))
	receipt, err := eng.ApplyMask(logits)
	if err != nil {
		t.Fatalf("ApplyMask: %v", err)
	}
	if !receipt.DomainMonotonicityOK || math.IsInf(float64(logits[eos]), -1) || !math.IsInf(float64(logits[3]), -1) {
		t.Fatalf("ApplyMask failed to isolate EOS cleanly: receipt=%+v", receipt)
	}

	if err := eng.CommitToken(eos); err != nil {
		t.Fatalf("CommitToken(eos): %v", err)
	}
	if !eng.IsStopped() {
		t.Fatal("expected IsStopped() == true after committing EOS")
	}
	if err := eng.CommitToken(eos); !errors.Is(err, ErrGrammarStopped) {
		t.Fatalf("expected ErrGrammarStopped after stop, got %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGrammarMasksRecursiveJSONRefAndAnyOf(t *testing.T) {
	schema := []byte(`{
		"$defs": {
			"node": {
				"type": "object",
				"properties": {
					"head": {"type": "integer"},
					"tail": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}
				},
				"required": ["head", "tail"],
				"additionalProperties": false
			}
		},
		"$ref": "#/$defs/node"
	}`)
	// Default GrammarOptions allows standard JSON whitespace [ \t\n\r]*.
	g, err := CompileJSONSchemaGrammar(schema, GrammarOptions{})
	if err != nil {
		t.Fatalf("CompileJSONSchemaGrammar recursive $ref: %v", err)
	}

	const eos = 0
	pieces := [][]byte{
		[]byte("<eos>"),                       // 0
		[]byte("{\n  \"head\": 1,\n"),         // 1
		[]byte("  \"tail\": "),                // 2
		[]byte("{\n    \"head\": -42,\n"),     // 3
		[]byte("    \"tail\": null\n  }\n}"),  // 4
		[]byte("null\n}"),                     // 5
		[]byte("  \"tail\": \"bad\"\n}"),      // 6: string violates anyOf[node, null]
		[]byte("    \"tail\": null\n  }\n}}"), // 7: extra closing brace
	}
	trie, err := NewTokenTrieFromPieces(pieces, eos)
	if err != nil {
		t.Fatalf("NewTokenTrieFromPieces: %v", err)
	}
	eng, err := NewGrammarMaskEngine(trie, g)
	if err != nil {
		t.Fatalf("NewGrammarMaskEngine: %v", err)
	}

	for _, tok := range []int{1, 2, 3} {
		allowed, err := eng.NextAllowedTokens()
		if err != nil {
			t.Fatalf("NextAllowedTokens before tok %d: %v", tok, err)
		}
		if !slices.Contains(allowed, tok) {
			t.Fatalf("expected token %d in allowed=%v", tok, allowed)
		}
		if err := eng.CommitToken(tok); err != nil {
			t.Fatalf("CommitToken(%d): %v", tok, err)
		}
	}

	// Nested node open: token 4 finishes nested + outer object; token 7 (extra '}') must be masked out.
	allowed, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens nested: %v", err)
	}
	if !slices.Contains(allowed, 4) || slices.Contains(allowed, 7) || slices.Contains(allowed, eos) {
		t.Fatalf("unexpected nested allowed tokens = %v", allowed)
	}
	if err := eng.CommitToken(4); err != nil {
		t.Fatalf("CommitToken(4): %v", err)
	}
	if !eng.IsAccepting() {
		t.Fatal("expected recursive object to be accepting")
	}
	finalAllowed, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens final: %v", err)
	}
	if !slices.Equal(finalAllowed, []int{eos}) {
		t.Fatalf("finalAllowed = %v, want [0]", finalAllowed)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGrammarMasksRegexUTF8SplitTokensAndChoices(t *testing.T) {
	// 4-byte UTF-8 grinning face U+1F600 is F0 9F 98 80.
	g, err := CompileRegexGrammar(`^id-[0-9]+-😀$`)
	if err != nil {
		t.Fatalf("CompileRegexGrammar: %v", err)
	}
	const eos = 0
	pieces := [][]byte{
		[]byte("<eos>"),          // 0
		[]byte("id-"),            // 1
		[]byte("7"),              // 2
		[]byte("-"),              // 3
		{0xF0, 0x9F},             // 4: first half of U+1F600
		{0x98, 0x80},             // 5: second half of U+1F600
		{0x98, 0x81},             // 6: wrong emoji U+1F601
		{0xF0, 0x28, 0x8C, 0xBC}, // 7: invalid UTF-8 sequence
		[]byte("id-7-😀trailing"), // 8: overshoots end
		[]byte("id-00-"),         // 9: valid prefix
		{0xF0, 0x9F, 0x98, 0x80}, // 10: whole U+1F600
	}
	trie, err := NewTokenTrieFromPieces(pieces, eos)
	if err != nil {
		t.Fatalf("NewTokenTrieFromPieces: %v", err)
	}
	eng, err := NewGrammarMaskEngine(trie, g)
	if err != nil {
		t.Fatalf("NewGrammarMaskEngine: %v", err)
	}

	// Step through split UTF-8 tokens: 1 ("id-") -> 2 ("7") -> 3 ("-") -> 4 (F0 9F) -> 5 (98 80) -> 0 (EOS).
	for _, tok := range []int{1, 2, 3, 4} {
		if err := eng.CommitToken(tok); err != nil {
			t.Fatalf("CommitToken(%d): %v", tok, err)
		}
	}
	midUTF8Allowed, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens mid-UTF8: %v", err)
	}
	if !slices.Equal(midUTF8Allowed, []int{5}) {
		t.Fatalf("midUTF8Allowed = %v, want [5] (rejecting wrong rune 6 and invalid UTF-8)", midUTF8Allowed)
	}
	if err := eng.CommitToken(5); err != nil {
		t.Fatalf("CommitToken(5): %v", err)
	}
	finalAllowed, err := eng.NextAllowedTokens()
	if err != nil {
		t.Fatalf("NextAllowedTokens final: %v", err)
	}
	if !slices.Equal(finalAllowed, []int{eos}) {
		t.Fatalf("finalAllowed = %v, want [0]", finalAllowed)
	}

	// CompileChoiceGrammar with prefix-shared choices.
	choiceG, err := CompileChoiceGrammar([]string{"tool_alpha", "tool_alpha_v2", "tool_beta"})
	if err != nil {
		t.Fatalf("CompileChoiceGrammar: %v", err)
	}
	cPieces := [][]byte{
		[]byte("<eos>"),
		[]byte("tool_"),
		[]byte("alpha"),
		[]byte("_v2"),
		[]byte("beta"),
		[]byte("gamma"),
	}
	cTrie, err := NewTokenTrieFromPieces(cPieces, eos)
	if err != nil {
		t.Fatalf("NewTokenTrieFromPieces choices: %v", err)
	}
	cEng, err := NewGrammarMaskEngine(cTrie, choiceG)
	if err != nil {
		t.Fatalf("NewGrammarMaskEngine choices: %v", err)
	}
	if err := cEng.CommitToken(1); err != nil {
		t.Fatalf("CommitToken(tool_): %v", err)
	}
	if got, err := cEng.NextAllowedTokens(); err != nil || !slices.Equal(got, []int{2, 4}) {
		t.Fatalf("choices after tool_ = %v (err=%v), want [2 4]", got, err)
	}
	if err := cEng.CommitToken(2); err != nil {
		t.Fatalf("CommitToken(alpha): %v", err)
	}
	// At "tool_alpha", both EOS (complete choice) and "_v2" (longer choice) must be admitted.
	if got, err := cEng.NextAllowedTokens(); err != nil || !slices.Equal(got, []int{0, 3}) {
		t.Fatalf("choices at tool_alpha = %v (err=%v), want [0 3]", got, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGrammarMasksCFGEBNFAndSpeculativeFork(t *testing.T) {
	cfgSrc := `
# Recursive arithmetic expression grammar
root ::= expr
expr ::= term | expr "+" term
term ::= [0-9]+ | "(" expr ")"
`
	g, err := CompileCFGGrammar(cfgSrc)
	if err != nil {
		t.Fatalf("CompileCFGGrammar: %v", err)
	}
	const eos = 0
	pieces := [][]byte{
		[]byte("<eos>"), // 0
		[]byte("("),     // 1
		[]byte("12"),    // 2
		[]byte("+"),     // 3
		[]byte("34"),    // 4
		[]byte(")"),     // 5
		[]byte("abc"),   // 6
	}
	trie, err := NewTokenTrieFromPieces(pieces, eos)
	if err != nil {
		t.Fatalf("NewTokenTrieFromPieces: %v", err)
	}
	eng, err := NewGrammarMaskEngine(trie, g)
	if err != nil {
		t.Fatalf("NewGrammarMaskEngine: %v", err)
	}

	// Root cannot accept empty input.
	if got, err := eng.NextAllowedTokens(); err != nil || !slices.Equal(got, []int{1, 2, 4}) {
		t.Fatalf("initial CFG allowed = %v (err=%v), want [1 2 4]", got, err)
	}

	// Speculative fork 1 explores "(12+34)".
	forkA := eng.Fork()
	for _, tok := range []int{1, 2, 3, 4} {
		if err := forkA.CommitToken(tok); err != nil {
			t.Fatalf("forkA CommitToken(%d): %v", tok, err)
		}
	}
	// Inside open paren after "(12+34", EOS must NOT be admitted until ")" closes; "+" and ")" and digits are valid.
	allowedA, err := forkA.NextAllowedTokens()
	if err != nil {
		t.Fatalf("forkA NextAllowedTokens: %v", err)
	}
	if slices.Contains(allowedA, eos) || !slices.Contains(allowedA, 5) || !slices.Contains(allowedA, 3) {
		t.Fatalf("forkA inside paren allowed = %v, want '+' and ')' without EOS", allowedA)
	}
	if err := forkA.CommitToken(5); err != nil {
		t.Fatalf("forkA CommitToken(')'): %v", err)
	}
	if !forkA.IsAccepting() {
		t.Fatal("expected forkA to accept balanced expression")
	}

	// Original engine must remain completely untouched at offset 0.
	if len(eng.MatchedBytes()) != 0 {
		t.Fatalf("parent MatchedBytes mutated by fork: %q", eng.MatchedBytes())
	}
	if err := eng.CommitToken(2); err != nil {
		t.Fatalf("parent CommitToken('12'): %v", err)
	}
	if !eng.IsAccepting() || string(eng.MatchedBytes()) != "12" {
		t.Fatalf("unexpected parent state: accepting=%v matched=%q", eng.IsAccepting(), eng.MatchedBytes())
	}
	if err := eng.CommitToken(5); !errors.Is(err, ErrGrammarRejectedToken) {
		t.Fatalf("expected unbalanced ')' to be rejected, got %v", err)
	}

	// Reset clears state back to initial position.
	eng.Reset()
	if eng.IsAccepting() || len(eng.MatchedBytes()) != 0 {
		t.Fatalf("expected Reset to clear parser state")
	}

	// Unsupported and invalid grammars fail closed with typed errors.
	if _, err := CompileJSONSchemaGrammar([]byte(`{"type":"integer","minimum":5}`), GrammarOptions{}); !errors.Is(err, ErrUnsupportedGrammar) {
		t.Fatalf("expected ErrUnsupportedGrammar for minimum, got %v", err)
	}
	if _, err := CompileJSONSchemaGrammar([]byte(`false`), GrammarOptions{}); !errors.Is(err, ErrInvalidGrammar) {
		t.Fatalf("expected ErrInvalidGrammar for false schema, got %v", err)
	}
	if _, err := CompileRegexGrammar(`\bword\b`); !errors.Is(err, ErrUnsupportedGrammar) {
		t.Fatalf("expected ErrUnsupportedGrammar for \\b assertion, got %v", err)
	}
	if _, err := CompileCFGGrammar(`root ::= missing_rule`); !errors.Is(err, ErrInvalidGrammar) {
		t.Fatalf("expected ErrInvalidGrammar for undefined rule, got %v", err)
	}
}
