package godsplitplan

import (
	"strings"
	"testing"
)

// fixture mirrors godsplit_plan_test.py's FIXTURE byte-for-byte: a build tag, an
// aliased import (and a basename look-alike that is NOT an alias), a doc-commented
// type and func, a one-line method, and an init(). Line numbers are referenced
// explicitly in the tests.
const fixture = "//go:build linux\n" + // 1
	"\n" + // 2
	"// Package demo is a fixture.\n" + // 3
	"package demo\n" + // 4
	"\n" + // 5
	"import (\n" + // 6
	"\t\"fmt\"\n" + // 7
	"\t_ \"embed\"\n" + // 8
	"\tfakmodel \"github.com/x/y/internal/model\"\n" + // 9 (real alias)
	"\tmodel \"github.com/x/y/model\"\n" + // 10 (alias == basename: NOT a real alias)
	")\n" + // 11
	"\n" + // 12
	"// Widget is a thing.\n" + // 13
	"// Second doc line.\n" + // 14
	"type Widget struct {\n" + // 15
	"\tN int\n" + // 16
	"}\n" + // 17
	"\n" + // 18
	"// Make builds a Widget.\n" + // 19
	"func Make() *Widget {\n" + // 20
	"\treturn &Widget{}\n" + // 21
	"}\n" + // 22
	"\n" + // 23
	"func (w *Widget) Inc() { w.N++ }\n" + // 24 (one-line method, no doc)
	"\n" + // 25
	"func init() {\n" + // 26
	"\tfmt.Println(\"init\")\n" + // 27
	"}\n" // 28

func byName(p Plan) map[string]Decl {
	m := make(map[string]Decl, len(p.Decls))
	for _, d := range p.Decls {
		m[d.Name] = d
	}
	return m
}

func TestPackageAndSize(t *testing.T) {
	p := Compute(fixture)
	if p.Package != "demo" {
		t.Errorf("package = %q, want demo", p.Package)
	}
	if p.LineCount != 28 {
		t.Errorf("line_count = %d, want 28", p.LineCount)
	}
	if p.GodFile {
		t.Errorf("god_file = true, want false")
	}
}

func TestHazards(t *testing.T) {
	p := Compute(fixture)
	hz := p.Hazards
	if len(hz.BuildTags) != 1 || hz.BuildTags[0] != "//go:build linux" {
		t.Errorf("build_tags = %v, want [//go:build linux]", hz.BuildTags)
	}
	if hz.InitFuncs != 1 {
		t.Errorf("init_funcs = %d, want 1", hz.InitFuncs)
	}
	// Only the genuine alias; `_ "embed"`, plain `"fmt"`, and the basename-matching
	// `model "..../model"` are all excluded.
	want := `fakmodel "github.com/x/y/internal/model"`
	if len(hz.ImportAliases) != 1 || hz.ImportAliases[0] != want {
		t.Errorf("import_aliases = %v, want [%s]", hz.ImportAliases, want)
	}
}

func TestDocCommentAwareBlockStarts(t *testing.T) {
	m := byName(Compute(fixture))
	// Block starts at the leading doc comment, not the decl keyword.
	if m["Widget"].BlockStart != 13 {
		t.Errorf("Widget block_start = %d, want 13", m["Widget"].BlockStart)
	}
	if m["Make"].BlockStart != 19 {
		t.Errorf("Make block_start = %d, want 19", m["Make"].BlockStart)
	}
	// A decl with no doc comment starts at itself.
	if m["Inc"].BlockStart != 24 {
		t.Errorf("Inc block_start = %d, want 24", m["Inc"].BlockStart)
	}
	if m["init"].BlockStart != 26 {
		t.Errorf("init block_start = %d, want 26", m["init"].BlockStart)
	}
}

func TestBlockEndsChainToNextStartAndEOF(t *testing.T) {
	m := byName(Compute(fixture))
	if m["Widget"].BlockEnd != 18 {
		t.Errorf("Widget block_end = %d, want 18", m["Widget"].BlockEnd)
	}
	if m["Make"].BlockEnd != 23 {
		t.Errorf("Make block_end = %d, want 23", m["Make"].BlockEnd)
	}
	if m["Inc"].BlockEnd != 25 {
		t.Errorf("Inc block_end = %d, want 25", m["Inc"].BlockEnd)
	}
	// The last decl runs to EOF.
	if m["init"].BlockEnd != 28 {
		t.Errorf("init block_end = %d, want 28", m["init"].BlockEnd)
	}
}

func TestFuncSpans(t *testing.T) {
	m := byName(Compute(fixture))
	if m["Make"].FuncLines == nil || *m["Make"].FuncLines != 3 {
		t.Errorf("Make func_lines = %v, want 3", m["Make"].FuncLines)
	}
	if m["Inc"].FuncLines == nil || *m["Inc"].FuncLines != 1 {
		t.Errorf("Inc func_lines = %v, want 1", m["Inc"].FuncLines)
	}
	if m["init"].FuncLines == nil || *m["init"].FuncLines != 3 {
		t.Errorf("init func_lines = %v, want 3", m["init"].FuncLines)
	}
	if m["Make"].GodFunction == nil || *m["Make"].GodFunction {
		t.Errorf("Make god_function = %v, want false", m["Make"].GodFunction)
	}
}

func TestDeclNameKinds(t *testing.T) {
	cases := []struct {
		line, kind, name string
	}{
		{"func (r *R) Foo(x int) {", "method", "Foo"},
		{"func Bar() {", "func", "Bar"},
		{"type T struct {", "type", "T"},
		{"var x = 1", "var", "x"},
		{"const (", "const", "(group)"},
	}
	for _, c := range cases {
		k, nm := DeclName(c.line)
		if k != c.kind || nm != c.name {
			t.Errorf("DeclName(%q) = (%q,%q), want (%q,%q)", c.line, k, nm, c.kind, c.name)
		}
	}
}

func TestGodFunctionFlag(t *testing.T) {
	big := "package p\n\nfunc Big() {\n" + strings.Repeat("\t_ = 1\n", 205) + "}\n"
	p := Compute(big)
	m := byName(p)
	d, ok := m["Big"]
	if !ok {
		t.Fatal("Big decl not found")
	}
	if d.FuncLines == nil || *d.FuncLines <= FuncHardMax {
		t.Errorf("Big func_lines = %v, want > %d", d.FuncLines, FuncHardMax)
	}
	if d.GodFunction == nil || !*d.GodFunction {
		t.Errorf("Big god_function = %v, want true", d.GodFunction)
	}
}

func TestGodFileFlag(t *testing.T) {
	huge := "package p\n" + strings.Repeat("// filler\n", 1600)
	p := Compute(huge)
	if !p.GodFile {
		t.Errorf("god_file = false, want true")
	}
}

func TestRawStringEmbeddedDeclsAreIgnored(t *testing.T) {
	// A backtick raw string holding column-0 func/type text must NOT produce phantom
	// decls, and the real `var Tmpl` must survive with a valid (non-inverted) range.
	raw := "package demo\n" + // 1
		"\n" + // 2
		"// Tmpl is generated source.\n" + // 3
		"var Tmpl = `\n" + // 4  opens raw string
		"func generated() {}\n" + // 5  INSIDE raw — ignore
		"type Thing struct{}\n" + // 6  INSIDE raw — ignore
		"`\n" + // 7  closes raw string
		"\n" + // 8
		"// Real is real.\n" + // 9
		"func Real() {}\n" // 10
	p := Compute(raw)
	var names []string
	for _, d := range p.Decls {
		names = append(names, d.Name)
	}
	if len(names) != 2 || names[0] != "Tmpl" || names[1] != "Real" {
		t.Errorf("names = %v, want [Tmpl Real]", names)
	}
	if p.Hazards.RawStrings != 1 {
		t.Errorf("raw_strings = %d, want 1", p.Hazards.RawStrings)
	}
	tmpl := p.Decls[0]
	if tmpl.BlockEnd < tmpl.BlockStart {
		t.Errorf("Tmpl range inverted: %d..%d", tmpl.BlockStart, tmpl.BlockEnd)
	}
}

func TestFirstDeclBlockStartNeverSwallowsPackage(t *testing.T) {
	// No blank between the package clause and the first decl's doc comment: block_start
	// must clamp BELOW the package line, never to line 1 (which would duplicate the
	// package clause in the extracted file → build break).
	noblank := "package p\n// Doc\ntype X struct{}\n" // package=1, //Doc=2, type=3
	p := Compute(noblank)
	x := p.Decls[0]
	if x.Name != "X" {
		t.Fatalf("first decl name = %q, want X", x.Name)
	}
	if x.BlockStart != 2 {
		t.Errorf("X block_start = %d, want 2 (the //Doc line, not package)", x.BlockStart)
	}
}

func TestFuncSpanBraceDepthRobustness(t *testing.T) {
	// Brace depth (over string/comment-stripped code) must walk past an indented inner
	// composite-literal close and a brace inside a string, ending at the real closing
	// brace — and handle a one-liner.
	src := "package p\n" + // 1
		"\n" + // 2
		"func F() {\n" + // 3
		"\tm := map[string]int{\n" + // 4
		"\t\t\"a\": 1,\n" + // 5
		"\t}\n" + // 6  indented inner close
		"\tx := \"}\"\n" + // 7  brace in string — must NOT end the func
		"\t_ = x\n" + // 8
		"}\n" + // 9  real close
		"\n" + // 10
		"func G() { return }\n" // 11 one-liner
	m := byName(Compute(src))
	if m["F"].FuncLines == nil || *m["F"].FuncLines != 7 {
		t.Errorf("F func_lines = %v, want 7 (lines 3..9)", m["F"].FuncLines)
	}
	if m["G"].FuncLines == nil || *m["G"].FuncLines != 1 {
		t.Errorf("G func_lines = %v, want 1", m["G"].FuncLines)
	}
}
