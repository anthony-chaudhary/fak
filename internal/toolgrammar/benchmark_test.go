package toolgrammar

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// Global sinks prevent compiler optimizations from dead-code eliminating benchmark loops.
var (
	benchSinkGrammar *CompiledGrammar
	benchSinkString  string
	benchSinkRune    rune
	benchSinkByte    byte
	benchSinkBool    bool
	benchSinkMapB2U  map[byte]rune
)

const benchSmallSchemaJSON = `{
	"name": "read_file",
	"description": "Read file contents from local filesystem",
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

const benchUnionSchemaJSON = `{
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

func buildBenchmarkLargeSchemaJSON(branchCount int, paramsPerBranch int) []byte {
	var oneOf []map[string]any
	branchNames := make([]string, branchCount)
	for i := 0; i < branchCount; i++ {
		branchNames[i] = fmt.Sprintf("action_%d", i+1)
	}

	for _, bName := range branchNames {
		bProps := map[string]any{
			"action": map[string]any{"enum": []string{bName}},
		}
		req := []string{"action"}
		for p := 1; p <= paramsPerBranch; p++ {
			pName := fmt.Sprintf("field_%s_%d", bName, p)
			var pType string
			switch p % 4 {
			case 1:
				pType = "string"
			case 2:
				pType = "integer"
			case 3:
				pType = "number"
			default:
				pType = "boolean"
			}
			bProps[pName] = map[string]any{"type": pType}
			req = append(req, pName)
		}
		oneOf = append(oneOf, map[string]any{
			"properties": bProps,
			"required":   req,
		})
	}

	root := map[string]any{
		"name":        "orchestration_tool",
		"description": "Large multi-action tool schema",
		"parameters": map[string]any{
			"type":  "object",
			"oneOf": oneOf,
		},
	}
	data, _ := json.Marshal(root)
	return data
}

func buildSyntheticCodePayload(approxBytes int) string {
	var sb strings.Builder
	snippet := `for (int i = 0; i < n; i++) {
    if (a < b && b <= c) {
        std::vector<int> items = {1, 2, 3};
        std::map<std::string, std::vector<float>> matrix;
        uint32_t mask = 1 << 16;
        std::cout << "item: " << i << std::endl;
    }
}
`
	for sb.Len() < approxBytes {
		sb.WriteString(snippet)
	}
	return sb.String()
}

// -----------------------------------------------------------------------------
// 1. Schema Compiling Benchmarks
// -----------------------------------------------------------------------------

// BenchmarkCompile_SmallSchema measures compilation of a simple single-branch tool schema.
func BenchmarkCompile_SmallSchema(b *testing.B) {
	schemaBytes := []byte(benchSmallSchemaJSON)
	opts := GrammarOptions{Dialect: DialectDSML}

	b.ReportAllocs()
	b.SetBytes(int64(len(schemaBytes)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cg, err := Compile(schemaBytes, opts)
		if err != nil {
			b.Fatalf("Compile failed: %v", err)
		}
		benchSinkGrammar = cg
	}
}

// BenchmarkCompile_DiscriminatedUnion measures schema compilation of a multi-branch union
// with discriminator parameter pinning.
func BenchmarkCompile_DiscriminatedUnion(b *testing.B) {
	schemaBytes := []byte(benchUnionSchemaJSON)
	opts := GrammarOptions{Dialect: DialectDSML}

	b.ReportAllocs()
	b.SetBytes(int64(len(schemaBytes)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cg, err := Compile(schemaBytes, opts)
		if err != nil {
			b.Fatalf("Compile failed: %v", err)
		}
		benchSinkGrammar = cg
	}
}

// BenchmarkCompile_LargeSchema measures compilation throughput on complex discriminated
// union schemas scaling to 10 branches with 8 typed parameters each.
func BenchmarkCompile_LargeSchema(b *testing.B) {
	schemaBytes := buildBenchmarkLargeSchemaJSON(10, 8)
	opts := GrammarOptions{Dialect: DialectDSML}

	b.ReportAllocs()
	b.SetBytes(int64(len(schemaBytes)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cg, err := Compile(schemaBytes, opts)
		if err != nil {
			b.Fatalf("Compile failed: %v", err)
		}
		benchSinkGrammar = cg
	}
}

// BenchmarkCompile_Dialects measures schema compilation across each supported markup dialect.
func BenchmarkCompile_Dialects(b *testing.B) {
	schemaBytes := []byte(benchUnionSchemaJSON)
	dialects := []struct {
		name    string
		dialect Dialect
	}{
		{"DSML", DialectDSML},
		{"XML", DialectXML},
		{"JSON", DialectJSON},
	}

	for _, d := range dialects {
		b.Run(d.name, func(b *testing.B) {
			opts := GrammarOptions{Dialect: d.dialect}
			b.ReportAllocs()
			b.SetBytes(int64(len(schemaBytes)))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				cg, err := Compile(schemaBytes, opts)
				if err != nil {
					b.Fatalf("Compile failed: %v", err)
				}
				benchSinkGrammar = cg
			}
		})
	}
}

// BenchmarkCompileDiscriminatedUnionGrammar measures the full end-to-end convenience wrapper.
func BenchmarkCompileDiscriminatedUnionGrammar(b *testing.B) {
	schemaBytes := []byte(benchUnionSchemaJSON)
	opts := GrammarOptions{Dialect: DialectDSML}

	b.ReportAllocs()
	b.SetBytes(int64(len(schemaBytes)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		ebnf, err := CompileDiscriminatedUnionGrammar(schemaBytes, opts)
		if err != nil {
			b.Fatalf("CompileDiscriminatedUnionGrammar failed: %v", err)
		}
		benchSinkString = ebnf
	}
}

// -----------------------------------------------------------------------------
// 2. Grammar Generation Benchmarks
// -----------------------------------------------------------------------------

// BenchmarkGrammarGeneration_Dialects benchmarks the pure EBNF synthesis stage
// without JSON parsing overhead for DSML, XML, and JSON dialects.
func BenchmarkGrammarGeneration_Dialects(b *testing.B) {
	// Pre-extract branches to isolate EBNF rule generation
	var root rawSchema
	_ = json.Unmarshal([]byte(benchUnionSchemaJSON), &root)
	discProp, branches, _ := extractBranches(root.Parameters)

	dialects := []struct {
		name    string
		dialect Dialect
	}{
		{"DSML", DialectDSML},
		{"XML", DialectXML},
		{"JSON", DialectJSON},
	}

	for _, d := range dialects {
		b.Run(d.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ebnf, count := generateEBNF(d.dialect, VocabTypeStandard, "edit_file", "root", discProp, branches, false, "")
				if count == 0 || len(ebnf) == 0 {
					b.Fatal("unexpected empty EBNF")
				}
				benchSinkString = ebnf
			}
		})
	}
}

// BenchmarkGrammarGeneration_VocabType measures the difference between standard whitespace
// and byte-level BPE whitespace rule synthesis.
func BenchmarkGrammarGeneration_VocabType(b *testing.B) {
	var root rawSchema
	_ = json.Unmarshal([]byte(benchUnionSchemaJSON), &root)
	discProp, branches, _ := extractBranches(root.Parameters)

	vocabs := []struct {
		name      string
		vocabType VocabType
	}{
		{"Standard", VocabTypeStandard},
		{"ByteLevel", VocabTypeByteLevel},
	}

	for _, v := range vocabs {
		b.Run(v.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ebnf, count := generateEBNF(DialectDSML, v.vocabType, "edit_file", "root", discProp, branches, false, "")
				if count == 0 || len(ebnf) == 0 {
					b.Fatal("unexpected empty EBNF")
				}
				benchSinkString = ebnf
			}
		})
	}
}

// BenchmarkGrammarGeneration_BranchScaling measures EBNF generation scaling
// across varying branch counts (K=1, 5, 20) confirming linear O(K*P) behavior.
func BenchmarkGrammarGeneration_BranchScaling(b *testing.B) {
	branchCounts := []int{1, 5, 20}
	for _, k := range branchCounts {
		b.Run(fmt.Sprintf("K=%d", k), func(b *testing.B) {
			raw := buildBenchmarkLargeSchemaJSON(k, 6)
			var root rawSchema
			_ = json.Unmarshal(raw, &root)
			discProp, branches, _ := extractBranches(root.Parameters)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ebnf, count := generateEBNF(DialectDSML, VocabTypeStandard, "scaled_tool", "root", discProp, branches, false, "")
				if count == 0 || len(ebnf) == 0 {
					b.Fatal("unexpected empty EBNF")
				}
				benchSinkString = ebnf
			}
		})
	}
}

// -----------------------------------------------------------------------------
// 3. Token Validation & BPE Mapping Benchmarks
// -----------------------------------------------------------------------------

// BenchmarkTokenValidation_MatchLiteral_Snippet measures literal parameter parsing
// on C++/Go code snippets with relational and template '<' characters.
func BenchmarkTokenValidation_MatchLiteral_Snippet(b *testing.B) {
	snippet := "for (int i = 0; i < n; i++) { if (x <= 10) { std::vector<int> v; } }</parameter>"

	b.ReportAllocs()
	b.SetBytes(int64(len(snippet)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m, rem := MatchLiteralParameter(snippet)
		if len(m) == 0 || rem != "</parameter>" {
			b.Fatalf("unexpected match result: m=%q, rem=%q", m, rem)
		}
		benchSinkString = m
	}
}

// BenchmarkTokenValidation_MatchLiteral_PayloadSizes measures scanning throughput (MB/s)
// across 1KB, 16KB, and 64KB code buffers containing embedded relational symbols.
func BenchmarkTokenValidation_MatchLiteral_PayloadSizes(b *testing.B) {
	sizes := []struct {
		name  string
		bytes int
	}{
		{"1KB", 1024},
		{"16KB", 16 * 1024},
		{"64KB", 64 * 1024},
	}

	for _, sz := range sizes {
		payload := buildSyntheticCodePayload(sz.bytes) + "</parameter>"
		b.Run(sz.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(payload)))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				m, rem := MatchLiteralParameter(payload)
				if rem != "</parameter>" {
					b.Fatalf("expected remainder </parameter>, got %q", rem)
				}
				benchSinkString = m
			}
		})
	}
}

// BenchmarkTokenValidation_ByteToBPE measures single-byte to BPE rune conversion throughput.
func BenchmarkTokenValidation_ByteToBPE(b *testing.B) {
	bytesToTest := []byte{' ', '\t', '\n', 'a', 'Z', 0x00, 0x7F, 0xFF}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		target := bytesToTest[i%len(bytesToTest)]
		r := ByteToBPE(target)
		benchSinkRune = r
	}
}

// BenchmarkTokenValidation_BPEToByte measures inverse BPE rune to raw byte conversion throughput.
func BenchmarkTokenValidation_BPEToByte(b *testing.B) {
	runesToTest := []rune{BPESpaceRune, 'a', 'Z', 256, 270}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		target := runesToTest[i%len(runesToTest)]
		by, ok := BPEToByte(target)
		benchSinkByte = by
		benchSinkBool = ok
	}
}

// BenchmarkTokenValidation_EncodeByteLevel measures byte-level BPE text encoding throughput.
func BenchmarkTokenValidation_EncodeByteLevel(b *testing.B) {
	input := "func BenchmarkTokenValidation(b *testing.B) { std::vector<int> values = {1, 2, 3}; }"

	b.ReportAllocs()
	b.SetBytes(int64(len(input)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		enc := EncodeByteLevel(input)
		benchSinkString = enc
	}
}

// BenchmarkTokenValidation_DecodeByteLevel measures byte-level BPE text decoding throughput.
func BenchmarkTokenValidation_DecodeByteLevel(b *testing.B) {
	input := "func BenchmarkTokenValidation(b *testing.B) { std::vector<int> values = {1, 2, 3}; }"
	encoded := EncodeByteLevel(input)

	b.ReportAllocs()
	b.SetBytes(int64(len(encoded)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dec := DecodeByteLevel(encoded)
		benchSinkString = dec
	}
}

// BenchmarkTokenValidation_BuildBPEByteMap measures table construction throughput for BPE mappings.
func BenchmarkTokenValidation_BuildBPEByteMap(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m := BuildBPEByteMap()
		benchSinkMapB2U = m
	}
}

// -----------------------------------------------------------------------------
// Benchmark Sanity Check Test
// -----------------------------------------------------------------------------

// TestBenchmarkSanity ensures representative benchmark routines execute cleanly without panic.
func TestBenchmarkSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}

	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"Compile_DiscriminatedUnion", BenchmarkCompile_DiscriminatedUnion},
		{"GrammarGeneration_Dialects", BenchmarkGrammarGeneration_Dialects},
		{"TokenValidation_MatchLiteral_Snippet", BenchmarkTokenValidation_MatchLiteral_Snippet},
	}

	for _, bm := range benchmarks {
		res := testing.Benchmark(bm.fn)
		if res.N <= 0 {
			t.Errorf("benchmark %s performed 0 iterations", bm.name)
		}
	}
}
