package ctxknobs

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// BenchmarkScan measures end-to-end scanning of a repository tree for context
// knobs (combining flags, env lookups, skill frontmatter, sorting, and classification).
func BenchmarkScan(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inv, err := Scan(fixtureRoot)
		if err != nil {
			b.Fatalf("Scan: %v", err)
		}
		if len(inv.Knobs) == 0 {
			b.Fatal("unexpected empty knobs inventory")
		}
	}
}

// BenchmarkRatchetCore measures evaluation of an inventory against a baseline set,
// identifying user-required overlay offenses while ignoring operator-debug knobs.
func BenchmarkRatchetCore(b *testing.B) {
	inv := Inventory{
		Knobs: []Knob{
			{Kind: KindSkill, Name: "memory-compact", Class: UserRequired, File: ".claude/skills/memory-compact/SKILL.md", Line: 5},
			{Kind: KindSkill, Name: "ctx-overlay", Class: UserRequired, File: ".claude/skills/ctx-overlay/SKILL.md", Line: 10},
			{Kind: KindSkill, Name: "unapproved-overlay", Class: UserRequired, File: ".claude/skills/unapproved/SKILL.md", Line: 12},
			{Kind: KindFlag, Name: "ctx-view-budget", Class: OperatorDebug, File: "cmd/fak/main.go", Line: 42},
			{Kind: KindEnv, Name: "FAK_CONTEXT_TOKENS", Class: OperatorDebug, File: "cmd/fak/main.go", Line: 60},
			{Kind: KindFlag, Name: "context-compact-ratio", Class: OperatorDebug, File: "cmd/fak/flags.go", Line: 15},
		},
		UserRequired:  3,
		OperatorDebug: 3,
	}
	baseline := []string{"skill:memory-compact", "skill:ctx-overlay"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		off := RatchetOffenses(inv, baseline)
		if len(off) != 1 {
			b.Fatalf("unexpected offense count: got %d, want 1", len(off))
		}
	}
}

// BenchmarkFindKnobInAst measures parsing and traversing Go AST structures to
// identify context flag registrations and environment lookups.
func BenchmarkFindKnobInAst(b *testing.B) {
	samplePath := filepath.Join(fixtureRoot, "cmd", "fak", "sample.go")
	src, err := os.ReadFile(samplePath)
	if err != nil {
		b.Fatalf("ReadFile: %v", err)
	}
	fset := token.NewFileSet()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		file, err := parser.ParseFile(fset, "sample.go", src, 0)
		if err != nil {
			b.Fatalf("ParseFile: %v", err)
		}
		var knobsFound int
		ast.Inspect(file, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if strings.Contains(sel.Sel.Name, "Var") || sel.Sel.Name == "String" || sel.Sel.Name == "Bool" {
				if len(call.Args) >= 2 {
					if lit, ok := call.Args[1].(*ast.BasicLit); ok && lit.Kind == token.STRING {
						name := strings.Trim(lit.Value, `"`)
						if isContextFlagName(name) {
							knobsFound++
						}
					}
				}
			}
			if sel.Sel.Name == "Getenv" || sel.Sel.Name == "LookupEnv" {
				if len(call.Args) >= 1 {
					if lit, ok := call.Args[0].(*ast.BasicLit); ok && lit.Kind == token.STRING {
						name := strings.Trim(lit.Value, `"`)
						if isContextEnvName(name) {
							knobsFound++
						}
					}
				}
			}
			return true
		})
		if knobsFound != 2 {
			b.Fatalf("unexpected knobs found in AST: got %d, want 2", knobsFound)
		}
	}
}

// BenchmarkScanGoFileForKnobs measures the production regex line-scanner on a Go source file.
func BenchmarkScanGoFileForKnobs(b *testing.B) {
	samplePath := filepath.Join(fixtureRoot, "cmd", "fak", "sample.go")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		knobs, err := scanGoFileForKnobs(samplePath, "cmd/fak/sample.go")
		if err != nil {
			b.Fatalf("scanGoFileForKnobs: %v", err)
		}
		if len(knobs) != 2 {
			b.Fatalf("unexpected knob count: got %d, want 2", len(knobs))
		}
	}
}

// BenchmarkScanSkills measures scanning skills in .claude/skills for frontmatter overlays.
func BenchmarkScanSkills(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		knobs, err := scanSkills(fixtureRoot)
		if err != nil {
			b.Fatalf("scanSkills: %v", err)
		}
		if len(knobs) != 1 {
			b.Fatalf("unexpected skill knobs: got %d, want 1", len(knobs))
		}
	}
}

// BenchmarkIsContextFlagName measures flag and env name classification.
func BenchmarkIsContextFlagName(b *testing.B) {
	candidates := []string{
		"ctx-view-budget",
		"FAK_CONTEXT_TOKENS",
		"context-compact-ratio",
		"managed-cache-size",
		"session-token-budget",
		"verbose",
		"timeout",
		"worker-concurrency",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, c := range candidates {
			_ = isContextFlagName(c)
		}
	}
}

// BenchmarkIsUserRequiredSkill measures skill name and description classification.
func BenchmarkIsUserRequiredSkill(b *testing.B) {
	skills := []struct {
		name string
		desc string
	}{
		{"memory-compact", "compact and prune the auto-memory context window"},
		{"ctx-overlay", "manage and trim context token budget overlays"},
		{"quality-report", "audit code quality scorecards across git commits"},
		{"agent-readiness", "score agent discovery and tooling friction"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range skills {
			_ = isUserRequiredSkill(s.name, s.desc)
		}
	}
}

// BenchmarkReadSkillFrontmatter measures reading and parsing YAML frontmatter from a skill.
func BenchmarkReadSkillFrontmatter(b *testing.B) {
	skillPath := filepath.Join(fixtureRoot, ".claude", "skills", "ctx-overlay", "SKILL.md")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		name, desc, line, err := readSkillFrontmatter(skillPath)
		if err != nil {
			b.Fatalf("readSkillFrontmatter: %v", err)
		}
		if name == "" || desc == "" || line == 0 {
			b.Fatal("unexpected empty frontmatter")
		}
	}
}

// TestBenchmarkSanity ensures all benchmarked code paths execute without error.
func TestBenchmarkSanity(t *testing.T) {
	inv, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(inv.Knobs) == 0 {
		t.Fatal("unexpected empty knobs inventory")
	}

	off := RatchetOffenses(inv, []string{"skill:memory-compact"})
	if len(off) == 0 {
		t.Fatal("expected at least one offense for unbaselined fixture")
	}

	samplePath := filepath.Join(fixtureRoot, "cmd", "fak", "sample.go")
	knobs, err := scanGoFileForKnobs(samplePath, "cmd/fak/sample.go")
	if err != nil {
		t.Fatalf("scanGoFileForKnobs: %v", err)
	}
	if len(knobs) != 2 {
		t.Fatalf("scanGoFileForKnobs got %d knobs, want 2", len(knobs))
	}

	skillKnobs, err := scanSkills(fixtureRoot)
	if err != nil {
		t.Fatalf("scanSkills: %v", err)
	}
	if len(skillKnobs) != 1 {
		t.Fatalf("scanSkills got %d skill knobs, want 1", len(skillKnobs))
	}
}
