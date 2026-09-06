package toolgrammar

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestLiteralParameterEscaping(t *testing.T) {
	schemaJSON := `{
		"name": "edit_file",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["replace"]},
				"path": {"type": "string"},
				"new_string": {"type": "string"}
			},
			"required": ["mode", "path", "new_string"]
		}
	}`

	grammar, err := CompileDiscriminatedUnionGrammar([]byte(schemaJSON), GrammarOptions{
		Dialect: DialectDSML,
	})
	if err != nil {
		t.Fatalf("CompileDiscriminatedUnionGrammar failed: %v", err)
	}

	// Verify exact literal parameter escaping rule is emitted
	if !strings.Contains(grammar, LiteralParameterEscapeRule) {
		t.Fatalf("expected grammar to contain %q, got:\n%s", LiteralParameterEscapeRule, grammar)
	}

	// Test code snippets containing <, <=, #include <...>, templates, etc.
	testCases := []struct {
		name    string
		snippet string
	}{
		{
			name:    "for loop with relational less-than",
			snippet: "for (int i = 0; i < n; i++) {\n    count++;\n}",
		},
		{
			name:    "relational less-than-or-equal",
			snippet: "if (x <= 10 && y >= 5) { return true; }",
		},
		{
			name:    "C++ system include header",
			snippet: "#include <iostream>\n#include <vector>\n#include <stdio.h>",
		},
		{
			name:    "C++ template vector declaration",
			snippet: "std::vector<int> numbers = {1, 2, 3};",
		},
		{
			name:    "nested C++ templates",
			snippet: "std::map<std::string, std::vector<float>> matrix;",
		},
		{
			name:    "generic template class",
			snippet: "template <typename T> class SmartPointer {};",
		},
		{
			name:    "chained comparisons",
			snippet: "bool ok = (a < b) && (b < c);",
		},
		{
			name:    "bitwise left shift operator",
			snippet: "uint32_t mask = 1 << 16;",
		},
		{
			name:    "stream insertion operator",
			snippet: `std::cout << "value: " << val << std::endl;`,
		},
		{
			name:    "trailing less-than at boundary",
			snippet: "int x <",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Standalone match: entire snippet must be matched with no premature truncation
			matched, rem := MatchLiteralParameter(tc.snippet)
			if rem != "" {
				t.Errorf("expected no remainder, got %q (matched %q)", rem, matched)
			}
			if matched != tc.snippet {
				t.Errorf("snippet truncated!\nexpected: %q\ngot:      %q", tc.snippet, matched)
			}

			// 2. Trailing tag boundary match: parameter terminates exactly before closing tag </parameter>
			inputWithTag := tc.snippet + "</parameter>"
			matchedWithTag, remWithTag := MatchLiteralParameter(inputWithTag)
			if remWithTag != "</parameter>" {
				t.Errorf("expected remainder '</parameter>', got %q", remWithTag)
			}
			if matchedWithTag != tc.snippet {
				t.Errorf("snippet truncated before tag!\nexpected: %q\ngot:      %q", tc.snippet, matchedWithTag)
			}

			// 3. Contrast with naive rule [^<]* that prematurely truncates at '<'
			if strings.Contains(tc.snippet, "<") {
				naiveTruncated := tc.snippet[:strings.Index(tc.snippet, "<")]
				if matched == naiveTruncated {
					t.Fatalf("naive truncation failure reproduced: matched stopped at '<'")
				}
			}
		})
	}
}

func TestDiscriminatedUnionPinning(t *testing.T) {
	schemaJSON := `{
		"name": "edit_file",
		"description": "File editing tool with discriminated union modes",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {
					"type": "string",
					"enum": ["replace", "view", "write"]
				},
				"path": {"type": "string"},
				"old_string": {"type": "string"},
				"new_string": {"type": "string"},
				"content": {"type": "string"}
			},
			"required": ["mode", "path"],
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

	cg, err := Compile([]byte(schemaJSON), GrammarOptions{
		Dialect: DialectDSML,
	})
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}

	lines := strings.Split(cg.EBNF, "\n")
	ruleMap := make(map[string]string)
	for _, l := range lines {
		parts := strings.SplitN(l, "::=", 2)
		if len(parts) == 2 {
			ruleMap[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
		}
	}

	// 1. Verify discriminator parameter 'mode' is pinned FIRST in each branch
	replaceRule, ok := ruleMap["branch_replace"]
	if !ok {
		t.Fatalf("missing branch_replace rule in:\n%s", cg.EBNF)
	}
	if !strings.HasPrefix(replaceRule, `"<parameter=mode>replace</parameter>"`) {
		t.Errorf("branch_replace did not pin mode first: %s", replaceRule)
	}

	viewRule, ok := ruleMap["branch_view"]
	if !ok {
		t.Fatalf("missing branch_view rule in:\n%s", cg.EBNF)
	}
	if !strings.HasPrefix(viewRule, `"<parameter=mode>view</parameter>"`) {
		t.Errorf("branch_view did not pin mode first: %s", viewRule)
	}

	writeRule, ok := ruleMap["branch_write"]
	if !ok {
		t.Fatalf("missing branch_write rule in:\n%s", cg.EBNF)
	}
	if !strings.HasPrefix(writeRule, `"<parameter=mode>write</parameter>"`) {
		t.Errorf("branch_write did not pin mode first: %s", writeRule)
	}

	// 2. Verify linear rule scaling (no combinatorial explosion)
	// For K=3 branches and P=4 params, rule count is O(K*P), well below factorial explosion.
	if cg.RuleCount > 20 {
		t.Errorf("rule count too high (%d > 20), possible combinatorial explosion", cg.RuleCount)
	}

	// 3. Test scaling with larger schema (5 branches, 8 params each)
	// Without pinning, 8! * 5 = 201,600 rules. With pinning, rule count is linear <= 50 rules.
	var largeOneOf []map[string]any
	branchNames := []string{"create", "update", "patch", "inspect", "archive"}
	for _, bName := range branchNames {
		bProps := map[string]any{
			"action": map[string]any{"enum": []string{bName}},
		}
		req := []string{"action"}
		for p := 1; p <= 7; p++ {
			pName := fmt.Sprintf("field_%s_%d", bName, p)
			bProps[pName] = map[string]any{"type": "string"}
			req = append(req, pName)
		}
		largeOneOf = append(largeOneOf, map[string]any{
			"properties": bProps,
			"required":   req,
		})
	}
	largeSchema := map[string]any{
		"name": "complex_tool",
		"parameters": map[string]any{
			"type":  "object",
			"oneOf": largeOneOf,
		},
	}
	largeBytes, _ := json.Marshal(largeSchema)
	largeCG, err := Compile(largeBytes, GrammarOptions{Dialect: DialectDSML})
	if err != nil {
		t.Fatalf("large schema compile failed: %v", err)
	}

	// Rule count should be <= 50 (1 root + 5 branches + 35 params + 2 value/ws rules = 43 rules)
	if largeCG.RuleCount > 50 {
		t.Errorf("large schema rule count exceeded linear budget (%d > 50)", largeCG.RuleCount)
	}
	t.Logf("Linear scaling verified: K=5, P=8 produced %d rules (combinatorial 8! would be 40,320 rules)", largeCG.RuleCount)
}

func TestByteLevelSpaceProtection(t *testing.T) {
	// 1. Verify BPE space constants and byte mapping
	if BPESpaceRune != '\u0120' {
		t.Errorf("expected BPESpaceRune to be '\\u0120', got %q", BPESpaceRune)
	}
	if BPESpaceChar != "Ġ" {
		t.Errorf("expected BPESpaceChar to be 'Ġ', got %q", BPESpaceChar)
	}
	if string(BPESpaceBytes) != "Ġ" {
		t.Errorf("expected BPESpaceBytes to match 'Ġ', got %v", BPESpaceBytes)
	}

	// Test ByteToBPE mapping for space (0x20)
	if r := ByteToBPE(0x20); r != '\u0120' {
		t.Errorf("ByteToBPE(0x20) = %q, want '\\u0120'", r)
	}
	// Test BPEToByte inverse mapping
	if b, ok := BPEToByte('Ġ'); !ok || b != 0x20 {
		t.Errorf("BPEToByte('Ġ') = (0x%02x, %v), want (0x20, true)", b, ok)
	}

	// Test printable ASCII self-mapping
	for b := byte('!'); b <= byte('~'); b++ {
		if r := ByteToBPE(b); r != rune(b) {
			t.Errorf("ByteToBPE(%c) = %q, want self-mapping", b, r)
		}
	}

	// Test encoding and decoding strings with spaces and newlines
	original := "func edit(path string, mode string) error"
	encoded := EncodeByteLevel(original)
	expectedPrefix := "funcĠedit(pathĠstring,ĠmodeĠstring)Ġerror"
	if encoded != expectedPrefix {
		t.Errorf("EncodeByteLevel mismatch:\ngot:  %q\nwant: %q", encoded, expectedPrefix)
	}

	decoded := DecodeByteLevel(encoded)
	if decoded != original {
		t.Errorf("DecodeByteLevel roundtrip failed:\ngot:  %q\nwant: %q", decoded, original)
	}

	// 2. Test grammar compilation with VocabTypeByteLevel
	schemaJSON := `{
		"name": "edit_file",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["replace"]},
				"path": {"type": "string"}
			},
			"required": ["mode", "path"]
		}
	}`

	byteLevelGrammar, err := CompileDiscriminatedUnionGrammar([]byte(schemaJSON), GrammarOptions{
		Dialect:   DialectDSML,
		VocabType: VocabTypeByteLevel,
	})
	if err != nil {
		t.Fatalf("Compile with VocabTypeByteLevel failed: %v", err)
	}
	if !strings.Contains(byteLevelGrammar, `"Ġ"`) {
		t.Errorf("expected byte-level grammar to contain BPE space marker 'Ġ', got:\n%s", byteLevelGrammar)
	}

	// 3. Test grammar compilation with VocabTypeStandard
	standardGrammar, err := CompileDiscriminatedUnionGrammar([]byte(schemaJSON), GrammarOptions{
		Dialect:   DialectDSML,
		VocabType: VocabTypeStandard,
	})
	if err != nil {
		t.Fatalf("Compile with VocabTypeStandard failed: %v", err)
	}
	if strings.Contains(standardGrammar, `"Ġ"`) {
		t.Errorf("standard grammar should NOT contain 'Ġ', got:\n%s", standardGrammar)
	}
	if !strings.Contains(standardGrammar, `ws ::= [ \t\n\r]*`) {
		t.Errorf("expected standard grammar to contain 'ws ::= [ \\t\\n\\r]*', got:\n%s", standardGrammar)
	}
}

func TestDSMLFormatCompilation(t *testing.T) {
	schemaJSON := `{
		"name": "replace_text",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["exact", "regex"]},
				"path": {"type": "string"},
				"pattern": {"type": "string"},
				"replacement": {"type": "string"}
			},
			"required": ["mode", "path", "pattern", "replacement"],
			"oneOf": [
				{
					"properties": {
						"mode": {"enum": ["exact"]},
						"path": {"type": "string"},
						"pattern": {"type": "string"},
						"replacement": {"type": "string"}
					},
					"required": ["mode", "path", "pattern", "replacement"]
				},
				{
					"properties": {
						"mode": {"enum": ["regex"]},
						"path": {"type": "string"},
						"pattern": {"type": "string"},
						"replacement": {"type": "string"}
					},
					"required": ["mode", "path", "pattern", "replacement"]
				}
			]
		}
	}`

	grammar, err := CompileDiscriminatedUnionGrammar([]byte(schemaJSON), GrammarOptions{
		Dialect: DialectDSML,
	})
	if err != nil {
		t.Fatalf("DSML compilation failed: %v", err)
	}

	// Check DSML function tag
	if !strings.Contains(grammar, `root ::= "<function=replace_text>" ws (branch_exact | branch_regex) ws "</function>"`) {
		t.Errorf("missing or incorrect DSML root rule in:\n%s", grammar)
	}

	// Check parameter tags
	if !strings.Contains(grammar, `"<parameter=mode>exact</parameter>"`) {
		t.Errorf("missing exact mode parameter in branch_exact")
	}
	if !strings.Contains(grammar, `"<parameter=mode>regex</parameter>"`) {
		t.Errorf("missing regex mode parameter in branch_regex")
	}
	if !strings.Contains(grammar, `param_exact_pattern ::= "<parameter=pattern>" strval "</parameter>"`) {
		t.Errorf("missing exact pattern parameter rule in:\n%s", grammar)
	}

	// Check literal parameter escaping
	if !strings.Contains(grammar, LiteralParameterEscapeRule) {
		t.Errorf("missing literal parameter escape rule in:\n%s", grammar)
	}
}

func TestXMLFormatCompilation(t *testing.T) {
	schemaJSON := `{
		"name": "edit_file",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["replace", "view"]},
				"path": {"type": "string"}
			},
			"oneOf": [
				{
					"properties": {
						"mode": {"enum": ["replace"]},
						"path": {"type": "string"},
						"content": {"type": "string"}
					},
					"required": ["mode", "path", "content"]
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

	grammar, err := CompileDiscriminatedUnionGrammar([]byte(schemaJSON), GrammarOptions{
		Dialect: DialectXML,
	})
	if err != nil {
		t.Fatalf("XML compilation failed: %v", err)
	}

	if !strings.Contains(grammar, `root ::= "<edit_file>" ws (branch_replace | branch_view) ws "</edit_file>"`) {
		t.Errorf("missing XML root rule in:\n%s", grammar)
	}
	if !strings.Contains(grammar, `branch_replace ::= "<mode>replace</mode>" ws param_replace_path ws param_replace_content`) {
		t.Errorf("missing XML branch_replace rule in:\n%s", grammar)
	}
	if !strings.Contains(grammar, `param_replace_path ::= "<path>" strval "</path>"`) {
		t.Errorf("missing XML param_replace_path rule in:\n%s", grammar)
	}
}

func TestJSONFormatCompilation(t *testing.T) {
	schemaJSON := `{
		"name": "edit_file",
		"parameters": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["replace", "view"]},
				"path": {"type": "string"}
			},
			"oneOf": [
				{
					"properties": {
						"mode": {"enum": ["replace"]},
						"path": {"type": "string"}
					},
					"required": ["mode", "path"]
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

	grammar, err := CompileDiscriminatedUnionGrammar([]byte(schemaJSON), GrammarOptions{
		Dialect:        DialectJSON,
		IncludeToolTag: true,
	})
	if err != nil {
		t.Fatalf("JSON compilation failed: %v", err)
	}

	if !strings.Contains(grammar, `\"arguments\"`) {
		t.Errorf("missing JSON tool tag arguments in:\n%s", grammar)
	}
	if !strings.Contains(grammar, `json_strval ::= "\"" ([^"\\] | "\\" .)* "\""`) {
		t.Errorf("missing json_strval rule in:\n%s", grammar)
	}
}

func TestInvalidSchema(t *testing.T) {
	_, err := CompileDiscriminatedUnionGrammar([]byte("not json"), GrammarOptions{})
	if err == nil {
		t.Fatalf("expected error for invalid JSON")
	}

	_, err = CompileDiscriminatedUnionGrammar([]byte(`{}`), GrammarOptions{})
	if err == nil {
		t.Fatalf("expected error for empty schema with no properties or branches")
	}

	_, err = CompileDiscriminatedUnionGrammar([]byte(`{"type":"object"}`), GrammarOptions{
		Dialect: Dialect("unsupported"),
	})
	if err == nil {
		t.Fatalf("expected error for unsupported dialect")
	}
}
