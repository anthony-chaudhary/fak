package godsplitplan

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var (
	benchPlanSink    Plan
	benchStringSink  string
	benchAliasesSink []string
	benchKindSink    string
	benchNameSink    string
	benchIntSink     int
	benchBoolSink    bool
)

func makeMediumSource() string {
	var sb strings.Builder
	sb.WriteString("//go:build !windows\n\n")
	sb.WriteString("// Package mediumbench is a synthetic medium Go file for benchmarking.\n")
	sb.WriteString("package mediumbench\n\n")
	sb.WriteString("import (\n")
	sb.WriteString("\t\"context\"\n")
	sb.WriteString("\t\"fmt\"\n")
	sb.WriteString("\t_ \"image/png\"\n")
	sb.WriteString("\tfakctx \"github.com/anthony-chaudhary/fak/internal/ctxmmu\"\n")
	sb.WriteString("\tmodel \"github.com/anthony-chaudhary/fak/internal/model\"\n")
	sb.WriteString(")\n\n")

	for i := 0; i < 25; i++ {
		sb.WriteString(fmt.Sprintf("// Type%d represents synthetic benchmark data entity.\n", i))
		sb.WriteString(fmt.Sprintf("type Type%d struct {\n\tID int\n\tName string\n}\n\n", i))
		sb.WriteString(fmt.Sprintf("// Method%d performs operations on Type%d.\n", i, i))
		sb.WriteString(fmt.Sprintf("func (t *Type%d) Method%d(ctx context.Context) error {\n", i, i))
		sb.WriteString("\tmsg := \"inside method with {braces}\"\n")
		sb.WriteString("\tif msg != \"\" {\n")
		sb.WriteString("\t\treturn nil\n")
		sb.WriteString("\t}\n")
		sb.WriteString("\treturn fmt.Errorf(\"failed\")\n")
		sb.WriteString("}\n\n")
		sb.WriteString(fmt.Sprintf("// Worker%d processes standalone tasks.\n", i))
		sb.WriteString(fmt.Sprintf("func Worker%d(count int) int {\n\treturn count * 2\n}\n\n", i))
	}
	sb.WriteString("func init() {\n\tfmt.Println(\"init mediumbench\")\n}\n")
	return sb.String()
}

func makeGodFileSource() string {
	var sb strings.Builder
	sb.WriteString("//go:build linux || darwin\n\n")
	sb.WriteString("// Package godfilebench is a synthetic god-file exceeding FileHardMax lines.\n")
	sb.WriteString("package godfilebench\n\n")
	sb.WriteString("import (\n")
	sb.WriteString("\t\"context\"\n")
	sb.WriteString("\t\"fmt\"\n")
	sb.WriteString("\taliasedio \"io\"\n")
	sb.WriteString(")\n\n")

	// God-function exceeding FuncHardMax (200 lines)
	sb.WriteString("// GiantDispatcher is a god-function exceeding 200 lines.\n")
	sb.WriteString("func GiantDispatcher(ctx context.Context, payload []byte) (int, error) {\n")
	sb.WriteString("\tsum := 0\n")
	for j := 0; j < 210; j++ {
		sb.WriteString(fmt.Sprintf("\tstep%d := %d\n\tsum += step%d\n", j, j, j))
	}
	sb.WriteString("\treturn sum, nil\n}\n\n")

	// Regular functions and types to push total line count well above FileHardMax (1500 lines)
	for i := 0; i < 140; i++ {
		sb.WriteString(fmt.Sprintf("// Handler%d processes batch slice %d.\n", i, i))
		sb.WriteString(fmt.Sprintf("func Handler%d(input int) int {\n", i))
		sb.WriteString("\tres := input * 3\n")
		sb.WriteString("\tif res < 0 {\n\t\treturn 0\n\t}\n")
		sb.WriteString("\treturn res\n")
		sb.WriteString("}\n\n")
	}
	return sb.String()
}

func makeRawStringsSource() string {
	var sb strings.Builder
	sb.WriteString("package rawbench\n\n")
	for i := 0; i < 15; i++ {
		sb.WriteString(fmt.Sprintf("// RawTemplate%d contains embedded pseudo-Go code inside backticks.\n", i))
		sb.WriteString(fmt.Sprintf("var RawTemplate%d = `\n", i))
		sb.WriteString("func embeddedInsideRaw() {\n")
		sb.WriteString("\tfmt.Println(\"not a real top level func\")\n")
		sb.WriteString("}\n")
		sb.WriteString("type EmbeddedInsideRaw struct {\n")
		sb.WriteString("\tVal int\n")
		sb.WriteString("}\n")
		sb.WriteString("`\n\n")
	}
	sb.WriteString("// ActualRealFunc is a genuine top-level function.\n")
	sb.WriteString("func ActualRealFunc() bool {\n\treturn true\n}\n")
	return sb.String()
}

func TestBenchmarkSanity(t *testing.T) {
	pFixture := Compute(fixture)
	if pFixture.Package != "demo" {
		t.Fatalf("sanity fixture package: got %s, want demo", pFixture.Package)
	}

	mediumSrc := makeMediumSource()
	pMedium := Compute(mediumSrc)
	if pMedium.Package != "mediumbench" || pMedium.GodFile {
		t.Fatalf("sanity medium plan: package=%s godFile=%v", pMedium.Package, pMedium.GodFile)
	}

	godSrc := makeGodFileSource()
	pGod := Compute(godSrc)
	if !pGod.GodFile {
		t.Fatalf("sanity godfile plan: got godFile=false (lines=%d), want true", pGod.LineCount)
	}
	hasGodFunc := false
	for _, d := range pGod.Decls {
		if d.GodFunction != nil && *d.GodFunction {
			hasGodFunc = true
			break
		}
	}
	if !hasGodFunc {
		t.Fatalf("sanity godfile plan: expected at least one god function")
	}

	rawSrc := makeRawStringsSource()
	pRaw := Compute(rawSrc)
	if pRaw.Hazards.RawStrings != 15 {
		t.Fatalf("sanity raw plan: got %d raw strings, want 15", pRaw.Hazards.RawStrings)
	}

	rendered := Render("sample.go", pFixture)
	if len(rendered) == 0 {
		t.Fatalf("sanity render returned empty string")
	}

	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "fixture.go")
	if err := os.WriteFile(filePath, []byte(fixture), 0o644); err != nil {
		t.Fatal(err)
	}
	if code := Run(io.Discard, io.Discard, []string{filePath}); code != 0 {
		t.Fatalf("sanity Run text exit code: got %d, want 0", code)
	}
	if code := Run(io.Discard, io.Discard, []string{filePath, "--json"}); code != 0 {
		t.Fatalf("sanity Run json exit code: got %d, want 0", code)
	}
}

// BenchmarkCompute measures source text analysis across small, medium, god-file, and raw-string-heavy files.
func BenchmarkCompute(b *testing.B) {
	mediumSrc := makeMediumSource()
	godSrc := makeGodFileSource()
	rawSrc := makeRawStringsSource()

	b.Run("SmallFixture", func(b *testing.B) {
		b.SetBytes(int64(len(fixture)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := Compute(fixture)
			benchPlanSink = p
		}
	})

	b.Run("MediumFile_500L", func(b *testing.B) {
		b.SetBytes(int64(len(mediumSrc)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := Compute(mediumSrc)
			benchPlanSink = p
		}
	})

	b.Run("GodFile_1800L", func(b *testing.B) {
		b.SetBytes(int64(len(godSrc)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := Compute(godSrc)
			benchPlanSink = p
		}
	})

	b.Run("RawStrings_Intensive", func(b *testing.B) {
		b.SetBytes(int64(len(rawSrc)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := Compute(rawSrc)
			benchPlanSink = p
		}
	})
}

// BenchmarkRender measures formatting and output generation for human-readable reports.
func BenchmarkRender(b *testing.B) {
	pFixture := Compute(fixture)
	pGod := Compute(makeGodFileSource())

	b.Run("SmallPlan", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out := Render("demo.go", pFixture)
			benchStringSink = out
		}
	})

	b.Run("GodFilePlan", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out := Render("godfile.go", pGod)
			benchStringSink = out
		}
	})
}

// BenchmarkRun measures the end-to-end CLI execution pathway including disk read, plan computation, and serialization.
func BenchmarkRun(b *testing.B) {
	tmpDir := b.TempDir()
	filePath := filepath.Join(tmpDir, "fixture.go")
	if err := os.WriteFile(filePath, []byte(fixture), 0o644); err != nil {
		b.Fatal(err)
	}

	b.Run("TextOutput", func(b *testing.B) {
		argv := []string{filePath}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			code := Run(io.Discard, io.Discard, argv)
			benchIntSink = code
		}
	})

	b.Run("JSONOutput", func(b *testing.B) {
		argv := []string{filePath, "--json"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			code := Run(io.Discard, io.Discard, argv)
			benchIntSink = code
		}
	})
}

// BenchmarkImportAliases measures detection of genuine aliased imports over import line blocks.
func BenchmarkImportAliases(b *testing.B) {
	importLines := []string{
		"import (",
		"\t\"fmt\"",
		"\t\"os\"",
		"\t_ \"embed\"",
		"\t. \"github.com/stretchr/testify/assert\"",
		"\taliasedctx \"github.com/anthony-chaudhary/fak/internal/ctxmmu\"",
		"\tmodel \"github.com/anthony-chaudhary/fak/internal/model\"",
		"\tfastjson \"github.com/valyala/fastjson\"",
		")",
		"import singlealias \"github.com/example/pkg/v2\"",
		"import \"strings\"",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aliases := ImportAliases(importLines)
		benchAliasesSink = aliases
	}
}

// BenchmarkDeclName measures line classification across methods, funcs, types, vars, and grouped consts.
func BenchmarkDeclName(b *testing.B) {
	lines := []string{
		"func (r *Receiver) ProcessItem(ctx context.Context, item *Item) (*Result, error) {",
		"func ComputeMetrics(data []byte) (Metrics, error) {",
		"type ConfigurationManager struct {",
		"var GlobalRegistry = make(map[string]any)",
		"const (",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		line := lines[i%len(lines)]
		kind, name := DeclName(line)
		benchKindSink = kind
		benchNameSink = name
	}
}

// BenchmarkScanLine measures code-string-comment scanning across varied line patterns.
func BenchmarkScanLine(b *testing.B) {
	samples := []struct {
		line  string
		inRaw bool
	}{
		{"\tval := \"string literal with \\\"escaped\\\" quotes and {braces}\"", false},
		{"\t// line comment with {curly} braces and symbols", false},
		{"\tfor i := 0; i < len(items); i++ {", false},
		{"var Tmpl = `some text with backtick", false},
		{"func embeddedInRaw() { fmt.Println(\"raw\") }", true},
		{"closing backtick here ` // trailing comment", true},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := samples[i%len(samples)]
		code, inRawAfter := scanLine(s.line, s.inRaw)
		benchStringSink = code
		benchBoolSink = inRawAfter
	}
}
