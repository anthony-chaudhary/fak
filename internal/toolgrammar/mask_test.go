// fak-test:runtime fast est=100ms
package toolgrammar

import (
	"math"
	"sync"
	"testing"
)

func TestMaskAllowedNextBytesAndCanAcceptText(t *testing.T) {
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

	cases := []struct {
		name    string
		schema  string
		payload string
	}{
		{
			name:    "discriminated_union_replace",
			schema:  editSchema,
			payload: `{"name":"edit_file","arguments":{"mode":"replace","path":"main.go","old_string":"a","new_string":"b"}}`,
		},
		{
			name:    "scalars_bool_int_num",
			schema:  searchSchema,
			payload: `{"name":"search_kb","arguments":{"query":"hello","top_k":-10,"min_score":0.75,"include_meta":false}}`,
		},
		{
			name:    "optional_keys",
			schema:  readSchema,
			payload: `{"name":"read_file","arguments":{"path":"go.mod","limit":50}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cg, err := Compile([]byte(tc.schema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			st, err := NewJumpForwardState(cg)
			if err != nil {
				t.Fatalf("new state: %v", err)
			}

			for pos := 0; pos <= len(tc.payload); pos++ {
				allowed := st.AllowedNextBytes()
				for b := 0; b < 256; b++ {
					got := st.CanAcceptText(string([]byte{byte(b)}))
					if allowed[b] != got {
						t.Fatalf("pos=%d (%q): AllowedNextBytes[%q]=%v != CanAcceptText=%v",
							pos, tc.payload[:pos], byte(b), allowed[b], got)
					}
				}
				if pos < len(tc.payload) {
					if err := st.AdvanceText(tc.payload[pos : pos+1]); err != nil {
						t.Fatalf("pos=%d AdvanceText(%q): %v", pos, tc.payload[pos:pos+1], err)
					}
				}
			}
			if !st.IsComplete() {
				t.Fatalf("expected state to be complete after %q", tc.payload)
			}
		})
	}
}

func buildMaskTestTokens() []VocabToken {
	var toks []VocabToken
	var id int32
	for b := 0; b < 256; b++ {
		toks = append(toks, VocabToken{ID: id, Text: string([]byte{byte(b)})})
		id++
	}
	extra := []string{
		`replace`, `view`, `write`, `rep`, `delete`, `replace"`, `replace_bad`,
		`true`, `false`, `tr`, `fa`, `yes`, `truee`, `falsify`,
		`123`, `-42`, `0.75`, `1.`, `1..2`, `abc`,
		`offset`, `limit`, `off`, `lim`, `unknown`, `path`, `offset":`,
		`go.mod`, `"}}`, `":"`, `","`, `"}`, `"} `,
	}
	for _, s := range extra {
		toks = append(toks, VocabToken{ID: id, Text: s})
		id++
	}
	return toks
}

func tokenIDByText(tokens []VocabToken, text string) int32 {
	for _, tok := range tokens {
		if tok.Text == text {
			return tok.ID
		}
	}
	return -1
}

func assertMaskParity(t *testing.T, st *JumpForwardState, mc *MaskCache, tokens []VocabToken, valid []string, invalid []string) {
	t.Helper()
	n := len(tokens)
	logitsDirect := make([]float32, n)
	logitsCached := make([]float32, n)
	for i := 0; i < n; i++ {
		logitsDirect[i] = float32(i) + 1.0
		logitsCached[i] = float32(i) + 1.0
	}

	st.MaskLogits(tokens, logitsDirect)
	mc.MaskLogits(st, logitsCached)

	for i := 0; i < n; i++ {
		if logitsDirect[i] != logitsCached[i] {
			if !(math.IsInf(float64(logitsDirect[i]), -1) && math.IsInf(float64(logitsCached[i]), -1)) {
				t.Fatalf("MaskLogits vs MaskCache mismatch at token %d (%q): direct=%v cached=%v",
					i, tokens[i].Text, logitsDirect[i], logitsCached[i])
			}
		}
	}

	for _, txt := range valid {
		id := tokenIDByText(tokens, txt)
		if id < 0 {
			t.Fatalf("missing test token %q", txt)
		}
		if math.IsInf(float64(logitsDirect[id]), -1) {
			t.Errorf("expected valid token %q (id=%d) to remain unmasked, got -Inf", txt, id)
		}
	}
	for _, txt := range invalid {
		id := tokenIDByText(tokens, txt)
		if id < 0 {
			t.Fatalf("missing test token %q", txt)
		}
		if !math.IsInf(float64(logitsDirect[id]), -1) {
			t.Errorf("expected invalid token %q (id=%d) to be masked to -Inf, got %v", txt, id, logitsDirect[id])
		}
	}
}

func TestMaskLogitsBranchingPositions(t *testing.T) {
	tokens := buildMaskTestTokens()
	mc := NewMaskCache(tokens)

	t.Run("discriminator_enum", func(t *testing.T) {
		schema := `{
			"name": "edit_file",
			"parameters": {
				"type": "object",
				"oneOf": [
					{"properties": {"mode": {"enum": ["replace"]}, "path": {"type": "string"}}, "required": ["mode", "path"]},
					{"properties": {"mode": {"enum": ["view"]}, "path": {"type": "string"}}, "required": ["mode", "path"]},
					{"properties": {"mode": {"enum": ["write"]}, "path": {"type": "string"}}, "required": ["mode", "path"]}
				]
			}
		}`
		cg, err := Compile([]byte(schema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
		if err != nil {
			t.Fatal(err)
		}
		st, err := NewJumpForwardState(cg)
		if err != nil {
			t.Fatal(err)
		}
		if err := st.AdvanceText(`{"name":"edit_file","arguments":{"mode":"`); err != nil {
			t.Fatal(err)
		}
		assertMaskParity(t, st, mc, tokens,
			[]string{`replace`, `view`, `write`, `rep`, `replace"`, `r`, `v`, `w`},
			[]string{`delete`, `replace_bad`, `true`, `123`, `"}`, `"`},
		)
	})

	t.Run("boolean_integer_number", func(t *testing.T) {
		schema := `{
			"name": "search_kb",
			"parameters": {
				"type": "object",
				"properties": {
					"flag": {"type": "boolean"},
					"count": {"type": "integer"},
					"score": {"type": "number"}
				},
				"required": ["flag", "count", "score"]
			}
		}`
		cg, err := Compile([]byte(schema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
		if err != nil {
			t.Fatal(err)
		}
		st, err := NewJumpForwardState(cg)
		if err != nil {
			t.Fatal(err)
		}
		if err := st.AdvanceText(`{"name":"search_kb","arguments":{"flag":`); err != nil {
			t.Fatal(err)
		}
		assertMaskParity(t, st, mc, tokens,
			[]string{`true`, `false`, `tr`, `fa`, `t`, `f`, ` `},
			[]string{`yes`, `truee`, `falsify`, `123`, `"`},
		)

		if err := st.AdvanceText(`true,"count":`); err != nil {
			t.Fatal(err)
		}
		assertMaskParity(t, st, mc, tokens,
			[]string{`123`, `-42`, `-`, `0`, `9`, ` `},
			[]string{`0.75`, `1.`, `abc`, `true`, `"`},
		)

		if err := st.AdvanceText(`-42,"score":`); err != nil {
			t.Fatal(err)
		}
		assertMaskParity(t, st, mc, tokens,
			[]string{`123`, `-42`, `0.75`, `1.`, `-`, `0`},
			[]string{`1..2`, `abc`, `false`, `"`},
		)

		if err := st.AdvanceText(`1.`); err != nil {
			t.Fatal(err)
		}
		assertMaskParity(t, st, mc, tokens,
			[]string{`0`, `9`, `123`},
			[]string{`.`, `}`, `,`, ` `},
		)

		if err := st.AdvanceText(`5`); err != nil {
			t.Fatal(err)
		}
		assertMaskParity(t, st, mc, tokens,
			[]string{`0`, `}`, ` `},
			[]string{`.`, `,`, `"`},
		)
	})

	t.Run("optional_keys_and_closing_and_done", func(t *testing.T) {
		schema := `{
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
		cg, err := Compile([]byte(schema), GrammarOptions{Dialect: DialectJSON, IncludeToolTag: true})
		if err != nil {
			t.Fatal(err)
		}
		st, err := NewJumpForwardState(cg)
		if err != nil {
			t.Fatal(err)
		}
		if err := st.AdvanceText(`{"name":"read_file","arguments":{"path":"`); err != nil {
			t.Fatal(err)
		}
		// Inside string body: normal text and valid closing `"}}` allowed, `":"` invalid.
		assertMaskParity(t, st, mc, tokens,
			[]string{`go.mod`, `"}}`, `"`},
			[]string{`":"`},
		)

		if err := st.AdvanceText(`go.mod"`); err != nil {
			t.Fatal(err)
		}
		// After required param with 2 optionals remaining: `,` and `}` allowed.
		assertMaskParity(t, st, mc, tokens,
			[]string{`,`, `}`, ` `},
			[]string{`"`, `offset`, `123`},
		)

		if err := st.AdvanceText(`,"`); err != nil {
			t.Fatal(err)
		}
		// Optional key branching: offset/limit allowed, unknown/path masked.
		assertMaskParity(t, st, mc, tokens,
			[]string{`offset`, `limit`, `off`, `lim`, `offset":`},
			[]string{`unknown`, `path`, `123`, `}`},
		)

		if err := st.AdvanceText(`limit":10}`); err != nil {
			t.Fatal(err)
		}
		// At outer close: `}` and whitespace allowed.
		assertMaskParity(t, st, mc, tokens,
			[]string{`}`, ` `},
			[]string{`,`, `"`, `123`},
		)

		if err := st.AdvanceText(`}`); err != nil {
			t.Fatal(err)
		}
		if !st.IsComplete() {
			t.Fatal("expected IsComplete() == true")
		}
		// When complete: only whitespace allowed in vocab tokens.
		assertMaskParity(t, st, mc, tokens,
			[]string{` `, "\n", "\t"},
			[]string{`}`, `"`, `123`, `go.mod`},
		)
	})
}

func TestMaskQuotedDiscriminatorAndOptKeyAllowedNextBytesParity(t *testing.T) {
	quotedDiscSchema := `{
		"name": "tool",
		"parameters": {
			"type": "object",
			"oneOf": [
				{
					"properties": {
						"kind": {"enum": ["a\"b"]},
						"opt\"one": {"type": "string"},
						"opt\"two": {"type": "integer"}
					},
					"required": ["kind"]
				},
				{
					"properties": {
						"kind": {"enum": ["plain"]}
					},
					"required": ["kind"]
				}
			]
		}
	}`
	cg, err := Compile([]byte(quotedDiscSchema), GrammarOptions{Dialect: DialectJSON})
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	checkAllBytes := func(label string, st *JumpForwardState) {
		t.Helper()
		allowed := st.AllowedNextBytes()
		if allowed['"'] {
			t.Fatalf("%s: expected unescaped quote to be disallowed before full match", label)
		}
		if !allowed['\\'] {
			t.Fatalf("%s: expected backslash escape to be allowed before escaped quote", label)
		}
		for b := 0; b < 256; b++ {
			want := st.CanAcceptText(string([]byte{byte(b)}))
			if allowed[b] != want {
				t.Fatalf("%s: byte 0x%02x (%q): AllowedNextBytes=%v, CanAcceptText=%v", label, b, byte(b), allowed[b], want)
			}
		}
	}

	st, err := NewJumpForwardState(cg)
	if err != nil {
		t.Fatalf("NewJumpForwardState: %v", err)
	}
	if err := st.AdvanceText(`{"kind":"a`); err != nil {
		t.Fatalf("AdvanceText disc: %v", err)
	}
	checkAllBytes("disc_val_before_escaped_quote", st)

	if err := st.AdvanceText(`\"b","`); err != nil {
		t.Fatalf("AdvanceText opt key start: %v", err)
	}
	if err := st.AdvanceText(`opt`); err != nil {
		t.Fatalf("AdvanceText opt prefix: %v", err)
	}
	checkAllBytes("opt_key_before_escaped_quote", st)
}

func TestMaskCacheConcurrentMaskLogits(t *testing.T) {
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
	cg, err := Compile([]byte(readSchema), GrammarOptions{Dialect: DialectJSON})
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	tokens := buildMaskTestTokens()
	// Grow beyond stack scratch cap (256) so concurrent pool path executes.
	for len(tokens) < 512 {
		tokens = append(tokens, VocabToken{ID: int32(len(tokens)), Text: "extra"})
	}
	mc := NewMaskCache(tokens)
	base, err := NewJumpForwardState(cg)
	if err != nil {
		t.Fatalf("NewJumpForwardState: %v", err)
	}
	if err := base.AdvanceText(`{"path":"a",`); err != nil {
		t.Fatalf("AdvanceText: %v", err)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			st := base.Clone()
			logits := make([]float32, len(tokens))
			for iter := 0; iter < 50; iter++ {
				for i := range logits {
					logits[i] = 1.0
				}
				mc.MaskLogits(st, logits)
				if !math.IsInf(float64(logits[tokenIDByText(tokens, "}")]), -1) {
					t.Errorf("expected '}' masked before optional key after trailing comma")
					return
				}
				if math.IsInf(float64(logits[tokenIDByText(tokens, `"`)]), -1) {
					t.Errorf("expected '\"' preserved for optional key")
					return
				}
			}
		}()
	}
	wg.Wait()
}
