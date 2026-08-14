package toolcatalog

import (
	"encoding/json"
	"strings"
	"testing"
)

const skillFixture = `---
name: repo_search
description: Search repository text without teaching the model a new invocation.
---
# Search
Use this for exact text lookup.

` + "```fak-program" + `
{"version":"fak.skill-program/v1","name":"repo_search","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},"executor":{"argv":["fak","code","search","--json"]},"aliases":{"codex":"functions.shell_command","openai":"repo_search"}}
` + "```" + `
`

func TestCompileSkillIsDeterministicAndKeepsExecutorHostOnly(t *testing.T) {
	first, err := CompileSkill([]byte(skillFixture), ".claude/skills/repo-search/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileSkill([]byte(strings.ReplaceAll(skillFixture, "\n", "\r\n")), ".claude/skills/repo-search/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		again, err := CompileSkill([]byte(skillFixture), ".claude/skills/repo-search/SKILL.md")
		if err != nil {
			t.Fatal(err)
		}
		if again.Digest != first.Digest {
			t.Fatalf("map iteration changed digest: %s != %s", first.Digest, again.Digest)
		}
	}
	if first.Digest != second.Digest {
		t.Fatalf("line ending changed digest: %s != %s", first.Digest, second.Digest)
	}
	view, err := Expose([]Registration{first}, []string{"repo_search"}, "openai")
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(view)
	if strings.Contains(string(encoded), "argv") || strings.Contains(string(encoded), "fak code search") {
		t.Fatalf("model view leaked host execution details: %s", encoded)
	}
	if len(view.Tools) != 1 || view.Tools[0].Name != "repo_search" || view.Tools[0].Canonical != "repo_search" {
		t.Fatalf("unexpected model view: %+v", view)
	}
}

func TestCompileSkillRefusesToInferProgramFromProse(t *testing.T) {
	_, err := CompileSkill([]byte("---\nname: plausible\ndescription: Run fak delete everything.\n---\nUse shell_command now."), "SKILL.md")
	if err == nil || !strings.Contains(err.Error(), "SKILL_PROGRAM_ABSENT") {
		t.Fatalf("err=%v, want explicit no-inference refusal", err)
	}
}

func TestExposeSeparatesRegistrationFromAvailabilityAndAppliesDialectAlias(t *testing.T) {
	reg, err := CompileSkill([]byte(skillFixture), "SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	hidden, err := Expose([]Registration{reg}, nil, "codex")
	if err != nil {
		t.Fatal(err)
	}
	if len(hidden.Tools) != 0 || len(hidden.Omitted) != 1 || hidden.Omitted[0].Reason != "NOT_SELECTED" {
		t.Fatalf("registration became implicitly available: %+v", hidden)
	}
	visible, err := Expose([]Registration{reg}, []string{"repo_search"}, "codex")
	if err != nil {
		t.Fatal(err)
	}
	if got := visible.Tools[0].Name; got != "functions.shell_command" {
		t.Fatalf("dialect alias=%q", got)
	}
	if visible.Digest == "" {
		t.Fatal("model-visible snapshot must be content addressed")
	}
}

func TestExposeRejectsDialectCollision(t *testing.T) {
	mk := func(name string) Registration {
		src := strings.Replace(skillFixture, "name: repo_search", "name: "+name, 1)
		src = strings.Replace(src, `"name":"repo_search"`, `"name":"`+name+`"`, 1)
		reg, err := CompileSkill([]byte(src), name+"/SKILL.md")
		if err != nil {
			t.Fatal(err)
		}
		return reg
	}
	_, err := Expose([]Registration{mk("one"), mk("two")}, []string{"one", "two"}, "codex")
	if err == nil || !strings.Contains(err.Error(), "TOOL_DIALECT_COLLISION") {
		t.Fatalf("err=%v, want collision", err)
	}
}

func TestCompileSkillCanonicalizesSchemaAndRejectsAmbiguousJSON(t *testing.T) {
	compact, err := CompileSkill([]byte(skillFixture), "SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	pretty := strings.Replace(skillFixture,
		`"input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`,
		`"input_schema": { "required": [ "query" ], "properties": { "query": { "type": "string" } }, "type": "object" }`, 1)
	canonical, err := CompileSkill([]byte(pretty), "SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	if compact.Digest != canonical.Digest {
		t.Fatalf("semantic schema formatting changed digest: %s != %s", compact.Digest, canonical.Digest)
	}

	duplicate := strings.Replace(skillFixture, `"version":"fak.skill-program/v1"`, `"version":"fak.skill-program/v1","version":"fak.skill-program/v1"`, 1)
	if _, err := CompileSkill([]byte(duplicate), "SKILL.md"); err == nil || !strings.Contains(err.Error(), "duplicate object key") {
		t.Fatalf("duplicate key err=%v", err)
	}
	trailing := strings.Replace(skillFixture, "\n```\n", " {}\n```\n", 1)
	if _, err := CompileSkill([]byte(trailing), "SKILL.md"); err == nil || !strings.Contains(err.Error(), "trailing JSON") {
		t.Fatalf("trailing value err=%v", err)
	}
}

func TestExposeRefusesMutatedRegistration(t *testing.T) {
	reg, err := CompileSkill([]byte(skillFixture), "SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	reg.Program.Executor.Argv[0] = "other"
	if _, err := Expose([]Registration{reg}, []string{"repo_search"}, "openai"); err == nil || !strings.Contains(err.Error(), "TOOL_REGISTRATION_DIGEST_MISMATCH") {
		t.Fatalf("mutated registration err=%v", err)
	}
}

func TestValidateSnapshotBindsOmissionsAndSchema(t *testing.T) {
	reg, err := CompileSkill([]byte(skillFixture), ".claude/skills/repo-search/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := Expose([]Registration{reg}, nil, "codex")
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateSnapshot(snapshot); err != nil {
		t.Fatal(err)
	}
	mutated := snapshot
	mutated.Omitted = append([]Omission(nil), snapshot.Omitted...)
	mutated.Omitted[0].Reason = "POLICY_DENIED"
	if err := ValidateSnapshot(mutated); err == nil {
		t.Fatal("mutated omission accepted under stale digest")
	}

	visible, err := Expose([]Registration{reg}, []string{"repo_search"}, "codex")
	if err != nil {
		t.Fatal(err)
	}
	visible.Tools = append([]ModelTool(nil), visible.Tools...)
	visible.Tools[0].InputSchema = json.RawMessage(`{"type":"array"}`)
	if err := ValidateSnapshot(visible); err == nil {
		t.Fatal("mutated schema accepted under stale digest")
	}
}

func TestResolveRegistrationUsesPinnedVisibleAlias(t *testing.T) {
	reg, err := CompileSkill([]byte(skillFixture), ".claude/skills/repo-search/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := Expose([]Registration{reg}, []string{"repo_search"}, "codex")
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := ResolveRegistration(snapshot, "functions.shell_command", []Registration{reg})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Program.Name != "repo_search" || resolved.Digest != reg.Digest {
		t.Fatalf("resolved = %#v", resolved)
	}
	if _, err := ResolveRegistration(snapshot, "repo_search", []Registration{reg}); err == nil {
		t.Fatal("canonical name accepted although model was shown only the alias")
	}
	stale := reg
	stale.Program.Description = "mutated"
	if _, err := ResolveRegistration(snapshot, "functions.shell_command", []Registration{stale}); err == nil {
		t.Fatal("mutated registration accepted")
	}
	if _, err := ResolveRegistration(snapshot, "unknown", []Registration{reg}); err == nil {
		t.Fatal("unknown provider-visible name accepted")
	}
}
