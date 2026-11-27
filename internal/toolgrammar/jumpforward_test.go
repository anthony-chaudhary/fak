// fak-test:runtime fast est=100ms
package toolgrammar

import (
	"fmt"
	"strings"
	"testing"
)

func buildTestBPEVocab() *PrefixVocab {
	var toks []VocabToken
	var id int32
	// Single-byte fallback tokens (0..255)
	for b := 0; b < 256; b++ {
		toks = append(toks, VocabToken{ID: id, Text: string([]byte{byte(b)})})
		id++
	}
	// Realistic BPE subword tokens for JSON tool envelopes and identifiers
	multi := []string{
		`{"`, `"}`, `"}}`, `":"`, `","`, `":{"`, `":`, `,"`,
		`name`, `arguments`, `mode`, `path`, `old`, `new`, `_string`,
		`edit`, `_file`, `read`, `search`, `_kb`, `replace`, `view`, `write`,
		`query`, `top`, `_k`, `min`, `_score`, `include`, `_meta`, `offset`, `limit`,
		`true`, `false`, `src/`, `main.go`, `internal/`, `toolgrammar`,
	}
	for _, m := range multi {
		toks = append(toks, VocabToken{ID: id, Text: m})
		id++
	}
	return NewPrefixVocab(toks)
}

func tokenizeGreedy(vocab VocabMatcher, s string) []int32 {
	var out []int32
	rem := s
	for len(rem) > 0 {
		id, n, ok := vocab.MatchPrefix(rem)
		if !ok || n <= 0 {
			panic(fmt.Sprintf("vocab failed to match prefix of %q", rem))
		}
		out = append(out, id)
		rem = rem[n:]
	}
	return out
}

func TestForcedPrefixJumpForward(t *testing.T) {
	vocab := buildTestBPEVocab()

	editSchema := `{
		"name": "edit_file",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["replace", "view", "write"]},
				"path": {"type": "string"},
				"old_string": {"type": "string"},
				"new_string": {"type": "string"},
				"content": {"type": "string"}
			},
			"oneOf": [
				{
					"properties": {
						"mode": {"enum": ["replace"]},
						"path": {"type": "string"},
						"old_string": {"type": "string"},
						"new_string": {"type": "string"}
					},
					"required": ["mode", "path", "old_string", "new_string"]
				},
				{
					"properties": {
						"mode": {"enum": ["view"]},
						"path": {"type": "string"}
					},
					"required": ["mode", "path"]
				},
				{
					"properties": {
						"mode": {"enum": ["write"]},
						"path": {"type": "string"},
						"content": {"type": "string"}
					},
					"required": ["mode", "path", "content"]
				}
			]
		}
	}`

	searchSchema := `{
		"name": "search_kb",
		"parameters": {
			"type": "object",
			"properties": {
				"query": {"type": "string"},
				"top_k": {"type": "integer"},
				"min_score": {"type": "number"},
				"include_meta": {"type": "boolean"}
			},
			"required": ["query", "top_k", "min_score", "include_meta"]
		}
	}`

	readSchema := `{
		"name": "read_file",
		"parameters": {
			"type": "object",
			"properties": {
				"path": {"type": "string"},
				"offset": {"type": "integer"},
				"limit": {"type": "integer"}
			},
			"required": ["path"]
		}
	}`

	editCG, err := Compile([]byte(editSchema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
	if err != nil {
		t.Fatalf("compile editSchema: %v", err)
	}
	searchCG, err := Compile([]byte(searchSchema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
	if err != nil {
		t.Fatalf("compile searchSchema: %v", err)
	}
	readCG, err := Compile([]byte(readSchema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
	if err != nil {
		t.Fatalf("compile readSchema: %v", err)
	}

	t.Run("grammar_equivalence_openai_mcp_payloads", func(t *testing.T) {
		validEdit := []string{
			`{"name":"edit_file","arguments":{"mode":"replace","path":"src/main.go","old_string":"a","new_string":"b"}}`,
			`{"name": "edit_file", "arguments": {"mode": "view", "path": "README.md"}}`,
			`{"name":"edit_file","arguments":{"mode":"write","path":"a.txt","content":"hello \"world\""}}`,
		}
		for _, p := range validEdit {
			if !Accepts(editCG, p) {
				t.Errorf("expected editCG to accept %s", p)
			}
		}

		invalidEdit := []string{
			`{"name":"other_tool","arguments":{"mode":"view","path":"a"}}`,
			`{"name":"edit_file","arguments":{"mode":"delete","path":"a"}}`,
			`{"name":"edit_file","arguments":{"mode":"replace","path":"a"}}`,
			`{"name":"edit_file","arguments":{"mode":"view","path":"a","extra":"x"}}`,
			`{"name":"edit_file","arguments":{"mode":"view","path":"a"}}trailing`,
		}
		for _, p := range invalidEdit {
			if Accepts(editCG, p) {
				t.Errorf("expected editCG to reject %s", p)
			}
		}

		validSearch := []string{
			`{"name":"search_kb","arguments":{"query":"kv cache","top_k":10,"min_score":0.85,"include_meta":true}}`,
			`{"name":"search_kb","arguments":{"query":"jump","top_k":-1,"min_score":1,"include_meta":false}}`,
		}
		for _, p := range validSearch {
			if !Accepts(searchCG, p) {
				t.Errorf("expected searchCG to accept %s", p)
			}
		}

		invalidSearch := []string{
			`{"name":"search_kb","arguments":{"query":"x","top_k":"ten","min_score":0.5,"include_meta":true}}`,
			`{"name":"search_kb","arguments":{"query":"x","top_k":10,"min_score":0.5,"include_meta":"yes"}}`,
		}
		for _, p := range invalidSearch {
			if Accepts(searchCG, p) {
				t.Errorf("expected searchCG to reject %s", p)
			}
		}

		validRead := []string{
			`{"name":"read_file","arguments":{"path":"go.mod"}}`,
			`{"name":"read_file","arguments":{"path":"go.mod","limit":50}}`,
			`{"name":"read_file","arguments":{"path":"go.mod","offset":10,"limit":50}}`,
		}
		for _, p := range validRead {
			if !Accepts(readCG, p) {
				t.Errorf("expected readCG to accept %s", p)
			}
		}
	})

	t.Run("zero_false_forced_tokens_on_branching_and_value_positions", func(t *testing.T) {
		st, err := NewJumpForwardState(editCG)
		if err != nil {
			t.Fatal(err)
		}
		// 1. Initial forced prefix stops right before discriminator value
		toks, txt := ForcedDraft(st, vocab)
		wantPrefix := `{"name":"edit_file","arguments":{"mode":"`
		if txt != wantPrefix || len(toks) == 0 {
			t.Fatalf("got (%q, %d toks), want (%q, >0 toks)", txt, len(toks), wantPrefix)
		}
		if err := st.AdvanceText(txt); err != nil {
			t.Fatal(err)
		}
		// At discriminator branching position: MUST return 0 forced tokens
		toks2, txt2 := ForcedDraft(st, vocab)
		if len(toks2) != 0 || txt2 != "" {
			t.Fatalf("expected 0 forced tokens at discriminator branch, got %q (%v)", txt2, toks2)
		}

		// Model picks "replace"
		if err := st.AdvanceText(`replace"`); err != nil {
			t.Fatal(err)
		}
		// Next forced span: ,"path":"
		toks3, txt3 := ForcedDraft(st, vocab)
		if txt3 != `,"path":"` || len(toks3) == 0 {
			t.Fatalf("got %q, want %q", txt3, `,"path":"`)
		}
		if err := st.AdvanceText(txt3); err != nil {
			t.Fatal(err)
		}
		// At string value position: MUST return 0 forced tokens
		toks4, txt4 := ForcedDraft(st, vocab)
		if len(toks4) != 0 || txt4 != "" {
			t.Fatalf("expected 0 forced tokens at string value, got %q", txt4)
		}

		// Test scalar value positions (integer, number, boolean) in searchCG
		sst, err := NewJumpForwardState(searchCG)
		if err != nil {
			t.Fatal(err)
		}
		_, sPrefix := ForcedDraft(sst, vocab)
		if sPrefix != `{"name":"search_kb","arguments":{"query":"` {
			t.Fatalf("unexpected search prefix: %q", sPrefix)
		}
		_ = sst.AdvanceText(sPrefix + `kv"`)
		_, topKPrefix := ForcedDraft(sst, vocab)
		if topKPrefix != `,"top_k":` {
			t.Fatalf("unexpected top_k prefix: %q", topKPrefix)
		}
		_ = sst.AdvanceText(topKPrefix)
		// At integer value position: MUST return 0 forced tokens
		if intToks, intTxt := ForcedDraft(sst, vocab); len(intToks) != 0 || intTxt != "" {
			t.Fatalf("expected 0 forced tokens at integer position, got %q", intTxt)
		}

		// Test optional parameter branching in readCG
		rst, err := NewJumpForwardState(readCG)
		if err != nil {
			t.Fatal(err)
		}
		_, rPrefix := ForcedDraft(rst, vocab)
		_ = rst.AdvanceText(rPrefix + `go.mod"`)
		// After required "path", optional "offset"/"limit" or "}" may follow -> MUST return 0 forced tokens
		if optToks, optTxt := ForcedDraft(rst, vocab); len(optToks) != 0 || optTxt != "" {
			t.Fatalf("expected 0 forced tokens at optional param branch, got %q", optTxt)
		}
	})

	t.Run("decode_step_reduction_ratio", func(t *testing.T) {
		cases := []struct {
			name    string
			cg      *CompiledGrammar
			payload string
		}{
			{
				name:    "edit_file_replace",
				cg:      editCG,
				payload: `{"name":"edit_file","arguments":{"mode":"replace","path":"src/main.go","old_string":"x","new_string":"y"}}`,
			},
			{
				name:    "search_kb",
				cg:      searchCG,
				payload: `{"name":"search_kb","arguments":{"query":"kv","top_k":8,"min_score":0.9,"include_meta":true}}`,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				baselineTokens := tokenizeGreedy(vocab, tc.payload)
				baselineSteps := len(baselineTokens)

				st, err := NewJumpForwardState(tc.cg)
				if err != nil {
					t.Fatal(err)
				}

				rem := tc.payload
				jumpSteps := 0
				forcedTokensTotal := 0
				for !st.IsComplete() {
					fToks, fTxt := ForcedDraft(st, vocab)
					if len(fToks) > 0 && strings.HasPrefix(rem, fTxt) {
						forcedTokensTotal += len(fToks)
						if err := st.AdvanceText(fTxt); err != nil {
							t.Fatalf("AdvanceText forced %q: %v", fTxt, err)
						}
						rem = rem[len(fTxt):]
						// Speculative batch verification of forced span counts as 1 forward pass
						jumpSteps++
						continue
					}
					if len(rem) == 0 {
						t.Fatalf("incomplete state with empty remainder")
					}
					_, n, ok := vocab.MatchPrefix(rem)
					if !ok || n <= 0 {
						t.Fatalf("vocab match failed on %q", rem)
					}
					if err := st.AdvanceText(rem[:n]); err != nil {
						t.Fatalf("AdvanceText model %q: %v", rem[:n], err)
					}
					rem = rem[n:]
					jumpSteps++
				}

				ratio := float64(baselineSteps) / float64(jumpSteps)
				t.Logf("%s: baseline_steps=%d jump_forward_steps=%d forced_tokens=%d reduction=%.2fx",
					tc.name, baselineSteps, jumpSteps, forcedTokensTotal, ratio)
				if ratio < 2.0 {
					t.Errorf("decode step reduction %.2fx < 2.0x (baseline=%d, jump=%d)", ratio, baselineSteps, jumpSteps)
				}
			})
		}
	})
}

func BenchmarkForcedPrefixJumpForward(b *testing.B) {
	vocab := buildTestBPEVocab()
	schema := `{
		"name": "edit_file",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["replace", "view"]},
				"path": {"type": "string"},
				"old_string": {"type": "string"},
				"new_string": {"type": "string"}
			},
			"oneOf": [
				{
					"properties": {
						"mode": {"enum": ["replace"]},
						"path": {"type": "string"},
						"old_string": {"type": "string"},
						"new_string": {"type": "string"}
					},
					"required": ["mode", "path", "old_string", "new_string"]
				},
				{
					"properties": {
						"mode": {"enum": ["view"]},
						"path": {"type": "string"}
					},
					"required": ["mode", "path"]
				}
			]
		}
	}`
	cg, err := Compile([]byte(schema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
	if err != nil {
		b.Fatal(err)
	}
	baseState, err := NewJumpForwardState(cg)
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]int32, 0, 32)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toks, txt := ForcedDraftInto(baseState, vocab, dst[:0])
		if len(toks) == 0 || txt == "" {
			b.Fatal("expected non-empty forced draft")
		}
		dst = toks
	}
}
