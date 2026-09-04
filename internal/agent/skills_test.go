package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
)

func TestParseSkillFrontmatter(t *testing.T) {
	t.Run("comma-separated allowed-tools and metadata", func(t *testing.T) {
		raw := `---
name: audit-code
description: Performs deep code audit
allowed-tools: Read, Edit, Write, Grep, Glob, Bash
metadata:
  author: fak-eng
  version: 1.2.0
---

# Audit Code Procedure
1. Search for issues
2. Edit files
`
		skill, err := ParseSkill([]byte(raw), "")
		if err != nil {
			t.Fatalf("ParseSkill: %v", err)
		}
		if skill.Name != "audit-code" {
			t.Errorf("Name = %q, want audit-code", skill.Name)
		}
		if skill.Description != "Performs deep code audit" {
			t.Errorf("Description = %q", skill.Description)
		}
		wantTools := []string{"Read", "Edit", "Write", "Grep", "Glob", "Bash"}
		if len(skill.AllowedTools) != len(wantTools) {
			t.Fatalf("AllowedTools len = %d, want %d: %v", len(skill.AllowedTools), len(wantTools), skill.AllowedTools)
		}
		for i, w := range wantTools {
			if skill.AllowedTools[i] != w {
				t.Errorf("AllowedTools[%d] = %q, want %q", i, skill.AllowedTools[i], w)
			}
		}
		if skill.Metadata["author"] != "fak-eng" {
			t.Errorf("Metadata[author] = %q, want fak-eng", skill.Metadata["author"])
		}
		if skill.Metadata["version"] != "1.2.0" {
			t.Errorf("Metadata[version] = %q, want 1.2.0", skill.Metadata["version"])
		}
		if !strings.Contains(skill.Body, "# Audit Code Procedure") {
			t.Errorf("Body = %q, expected procedure header", skill.Body)
		}
	})

	t.Run("yaml list allowed-tools and top-level scalars", func(t *testing.T) {
		raw := `---
name: test-runner
description: "Runs tests safely"
allowed-tools:
  - Read
  - Bash
type: testing
license: MIT
---

## Test Steps
Run tests.
`
		skill, err := ParseSkill([]byte(raw), "")
		if err != nil {
			t.Fatalf("ParseSkill: %v", err)
		}
		if skill.Name != "test-runner" {
			t.Errorf("Name = %q", skill.Name)
		}
		if skill.Description != "Runs tests safely" {
			t.Errorf("Description = %q", skill.Description)
		}
		if len(skill.AllowedTools) != 2 || skill.AllowedTools[0] != "Read" || skill.AllowedTools[1] != "Bash" {
			t.Errorf("AllowedTools = %v, want [Read Bash]", skill.AllowedTools)
		}
		if skill.Metadata["type"] != "testing" {
			t.Errorf("Metadata[type] = %q", skill.Metadata["type"])
		}
		if skill.Metadata["license"] != "MIT" {
			t.Errorf("Metadata[license] = %q", skill.Metadata["license"])
		}
	})

	t.Run("underscore allowed_tools and bracketed list", func(t *testing.T) {
		raw := `---
name: bracket-skill
description: Bracket tools
allowed_tools: [Read, Write, Grep]
---

Body.
`
		skill, err := ParseSkill([]byte(raw), "")
		if err != nil {
			t.Fatalf("ParseSkill: %v", err)
		}
		if len(skill.AllowedTools) != 3 || skill.AllowedTools[0] != "Read" || skill.AllowedTools[1] != "Write" || skill.AllowedTools[2] != "Grep" {
			t.Errorf("AllowedTools = %v, want [Read Write Grep]", skill.AllowedTools)
		}
	})

	t.Run("canonical resolution", func(t *testing.T) {
		dir := t.TempDir()
		canonDir := filepath.Join(dir, ".claude", "skills", "my-skill")
		adaptDir := filepath.Join(dir, ".agents", "skills", "my-skill")
		if err := os.MkdirAll(canonDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(adaptDir, 0o755); err != nil {
			t.Fatal(err)
		}

		canonPath := filepath.Join(canonDir, "SKILL.md")
		canonContent := `---
name: my-skill
description: Canonical complete instructions
allowed-tools: Read, Bash
---

# Canonical Workflow Body
This is the authoritative procedure.
`
		if err := os.WriteFile(canonPath, []byte(canonContent), 0o644); err != nil {
			t.Fatal(err)
		}

		adaptPath := filepath.Join(adaptDir, "SKILL.md")
		adaptContent := `---
name: my-skill
description: Adapter stub
metadata:
  canonical: ../../../.claude/skills/my-skill/SKILL.md
---

# Canonical project skill adapter
This generated adapter contains no maintained workflow body.
`
		if err := os.WriteFile(adaptPath, []byte(adaptContent), 0o644); err != nil {
			t.Fatal(err)
		}

		skill, err := ParseSkillFile(adaptPath)
		if err != nil {
			t.Fatalf("ParseSkillFile(adaptPath): %v", err)
		}

		if skill.Name != "my-skill" {
			t.Errorf("Name = %q, want my-skill", skill.Name)
		}
		if filepath.Clean(skill.Canonical) != filepath.Clean(canonPath) {
			t.Errorf("Canonical = %q, want %q", skill.Canonical, canonPath)
		}
		if !strings.Contains(skill.Body, "This is the authoritative procedure.") {
			t.Errorf("Body was not resolved to canonical body: %s", skill.Body)
		}
		if len(skill.AllowedTools) != 2 || skill.AllowedTools[0] != "Read" || skill.AllowedTools[1] != "Bash" {
			t.Errorf("AllowedTools = %v, want canonical tools [Read Bash]", skill.AllowedTools)
		}
	})
}

func TestDiscoverSkills(t *testing.T) {
	ws := t.TempDir()
	extra := t.TempDir()

	writeSkill := func(dir, name, content string) string {
		skillDir := filepath.Join(dir, name)
		if err := os.MkdirAll(skillDir, 0o755); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(skillDir, "SKILL.md")
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}

	canonPath := writeSkill(filepath.Join(ws, ".claude", "skills"), "alpha", `---
name: alpha
description: Alpha canonical skill
allowed-tools: Read
---

# Alpha Procedure
Full instructions for Alpha.
`)

	writeSkill(filepath.Join(ws, ".agents", "skills"), "alpha", `---
name: alpha
description: Alpha adapter
metadata:
  canonical: ../../../.claude/skills/alpha/SKILL.md
---

# Canonical project skill adapter
Stub adapter body.
`)

	writeSkill(filepath.Join(ws, ".claude", "skills"), "beta", `---
name: beta
description: Beta skill
allowed-tools: Bash, Write
---

# Beta Procedure
`)

	writeSkill(extra, "gamma", `---
name: gamma
description: Gamma from extraDirs
allowed-tools: Glob
---

# Gamma Procedure
`)

	reg, err := DiscoverSkills(ws, extra)
	if err != nil {
		t.Fatalf("DiscoverSkills: %v", err)
	}

	if reg.Len() != 3 {
		t.Fatalf("reg.Len() = %d, want 3", reg.Len())
	}

	alpha, ok := reg.Get("alpha")
	if !ok {
		t.Fatal("skill alpha not found in registry")
	}
	if filepath.Clean(alpha.Path) != filepath.Clean(canonPath) {
		t.Errorf("alpha.Path = %q, want canonical path %q", alpha.Path, canonPath)
	}
	if !strings.Contains(alpha.Body, "Full instructions for Alpha.") {
		t.Errorf("alpha.Body = %q, expected canonical complete instructions", alpha.Body)
	}

	if _, ok := reg.Get("beta"); !ok {
		t.Error("skill beta not found")
	}
	if _, ok := reg.Get("gamma"); !ok {
		t.Error("skill gamma not found")
	}

	summary := reg.Summary()
	if !strings.Contains(summary, "alpha") || !strings.Contains(summary, "beta") || !strings.Contains(summary, "gamma") {
		t.Errorf("Summary missing discovered skills: %s", summary)
	}

	def := reg.ToolDef()
	if def.Function.Name != ToolSkill {
		t.Errorf("ToolDef name = %q, want %q", def.Function.Name, ToolSkill)
	}
}

func TestDiscoverSkillsRepoRoot(t *testing.T) {
	// Discovers real skills in this repo root if available
	reg, err := DiscoverSkills("../..")
	if err != nil {
		t.Fatalf("DiscoverSkills(repo root): %v", err)
	}
	if reg.Len() < 10 {
		t.Errorf("expected at least 10 skills discovered in repo root, got %d", reg.Len())
	}
	if s, ok := reg.Get("agent-readiness"); ok {
		if !strings.Contains(s.Body, "agent-readiness") {
			t.Errorf("agent-readiness body does not contain expected text: %s", s.Body)
		}
		if s.Description == "" {
			t.Errorf("agent-readiness description is empty")
		}
	} else {
		t.Errorf("expected agent-readiness skill to be discovered")
	}
}

func TestSkillToolExecution(t *testing.T) {
	reg := NewSkillRegistry()
	reg.Register(&Skill{
		Name:         "deploy-service",
		Description:  "Deploys microservices safely",
		AllowedTools: []string{"Bash", "Read"},
		Body:         "Step 1: check git status\nStep 2: build and deploy",
		Path:         "internal/deploy/SKILL.md",
	})

	t.Run("execute success", func(t *testing.T) {
		res, err := reg.Execute("deploy-service")
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if !strings.Contains(res, "deploy-service") {
			t.Errorf("result missing name: %s", res)
		}
		if !strings.Contains(res, "Deploys microservices safely") {
			t.Errorf("result missing description: %s", res)
		}
		if !strings.Contains(res, "Allowed Tools: Bash, Read") {
			t.Errorf("result missing allowed tools: %s", res)
		}
		if !strings.Contains(res, "Step 1: check git status") {
			t.Errorf("result missing body: %s", res)
		}
	})

	t.Run("execute unknown skill", func(t *testing.T) {
		_, err := reg.Execute("unknown-target")
		if err == nil {
			t.Fatal("expected error for unknown skill, got nil")
		}
		if !strings.Contains(err.Error(), "unknown-target") {
			t.Errorf("error missing requested name: %v", err)
		}
		if !strings.Contains(err.Error(), "deploy-service") {
			t.Errorf("error missing available skill list: %v", err)
		}
	})

	t.Run("execTool baseline execution", func(t *testing.T) {
		ArmSkills(reg)
		defer DisarmSkills()

		out, isErr := execTool(ToolSkill, map[string]any{"name": "deploy-service"})
		if isErr {
			t.Fatalf("execTool returned error: %s", string(out))
		}
		if !strings.Contains(string(out), "deploy-service") {
			t.Errorf("output missing name: %s", string(out))
		}

		errOut, isErr := execTool(ToolSkill, map[string]any{"name": "nonexistent"})
		if !isErr {
			t.Fatalf("execTool expected error for nonexistent skill")
		}
		if !strings.Contains(string(errOut), "nonexistent") || !strings.Contains(string(errOut), "deploy-service") {
			t.Errorf("error output = %s, expected unknown skill and available list", string(errOut))
		}

		emptyOut, isErr := execTool(ToolSkill, map[string]any{})
		if !isErr {
			t.Fatalf("execTool expected error for missing name")
		}
		if !strings.Contains(string(emptyOut), "missing required field: name") {
			t.Errorf("error output = %s, expected missing required field", string(emptyOut))
		}
	})
}

func TestSkillIntegrationThroughRunArm(t *testing.T) {
	root := t.TempDir()
	skillDir := filepath.Join(root, ".agents", "skills", "greeter")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	content := `---
name: greeter
description: Greets the team warmly
allowed-tools: Read
---

# Greeter Instructions
Emit a warm greeting to all engineers.
`
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	catalog, err := ArmCodeTools(root)
	if err != nil {
		t.Fatalf("ArmCodeTools: %v", err)
	}
	defer DisarmCodeTools()

	var hasSkillTool bool
	for _, def := range catalog {
		if def.Function.Name == ToolSkill {
			hasSkillTool = true
			break
		}
	}
	if !hasSkillTool {
		t.Fatalf("catalog does not contain tool %q: %#v", ToolSkill, catalog)
	}

	turns := []*Completion{
		toolCallTurn(ToolSkill, `{"name":"greeter"}`),
		{Message: Message{Content: "done"}},
	}
	var log []traceEvent
	planner := &recordingCodePlanner{turns: turns}

	m, err := RunArm(context.Background(), planner, "run greeter skill", true, len(turns)+1, &log, WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	var skillRows []traceEvent
	for _, e := range log {
		if e.Tool == ToolSkill {
			skillRows = append(skillRows, e)
		}
	}
	if len(skillRows) != 1 {
		t.Fatalf("expected 1 skill trace row, got %d: %+v", len(skillRows), log)
	}
	row := skillRows[0]
	if row.Verdict != "ALLOW" {
		t.Errorf("skill verdict = %q, want ALLOW", row.Verdict)
	}
	if row.By != codetools.RungName {
		t.Errorf("skill decided By = %q, want %q", row.By, codetools.RungName)
	}
	if m.EngineCalls < 1 {
		t.Errorf("EngineCalls = %d, want >= 1", m.EngineCalls)
	}

	lastResult := lastResultFromMessages(lastRecordingMessages)
	if !strings.Contains(lastResult, "Greeter Instructions") {
		t.Errorf("lastResult missing skill instructions: %s", lastResult)
	}
	if !strings.Contains(lastResult, "Greets the team warmly") {
		t.Errorf("lastResult missing skill description: %s", lastResult)
	}
}
