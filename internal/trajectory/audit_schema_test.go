package trajectory

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuditSchemaDriftGolden(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-9564", "current")
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{
		{Name: AuditSourceClaude, Root: filepath.Join(root, "claude", "projects"), RootLabel: "claude/projects"},
		{Name: AuditSourceCodex, Root: filepath.Join(root, "codex", "sessions"), RootLabel: "codex/sessions"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("schema drift fixture must remain parseable: %+v", result.Refusals)
	}
	if result.ConclusionStatus.SchemaDriftCount != 5 || result.ConclusionStatus.BreakingSchemaDrift != 4 {
		t.Fatalf("schema drift status = %+v, want 5 changes / 4 breaking", result.ConclusionStatus)
	}
	assertAuditBuildIdentity(t, result, AuditSourceClaude, AuditBuildIdentity{
		Provider: "anthropic", ProviderBuild: "messages-v2", Harness: "claude", HarnessBuild: "2.0.0-fixture",
	})
	assertAuditBuildIdentity(t, result, AuditSourceCodex, AuditBuildIdentity{
		Provider: "openai", ProviderBuild: "responses-v2", Harness: "codex", HarnessBuild: "2.0.0-fixture",
	})

	var jsonl bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"kind":"event_schema"`, `"kind":"schema_drift"`, `"provider_build":"responses-v2"`,
		`"compatibility":"additive"`, `"compatibility":"breaking"`,
		`"parser_surface":"internal/trajectory/audit_parse.go:parseCodexAuditRecord"`,
		`"fixture_surface":"internal/trajectory/testdata/audit/claude/projects"`,
	} {
		if !strings.Contains(jsonl.String(), want) {
			t.Fatalf("schema audit JSONL missing %s:\n%s", want, jsonl.String())
		}
	}

	var markdown bytes.Buffer
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	got := auditMarkdownSection(t, markdown.String(), "## Transcript schema drift", "## Transcript storage and telemetry overhead")
	want, err := os.ReadFile(filepath.Join("testdata", "audit", "issue-9564", "schema-drift.golden.md"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(got) != strings.TrimSpace(string(want)) {
		t.Fatalf("schema drift report changed\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func TestAuditSchemaBaselineRejectsUnknownVersion(t *testing.T) {
	_, err := DecodeAuditSchemaBaseline(strings.NewReader(`{"schema":"fak-trajectory-schema-baseline/2","events":[]}`))
	if err == nil || !strings.Contains(err.Error(), "has no reader") {
		t.Fatalf("unknown baseline version error = %v", err)
	}
}

func assertAuditBuildIdentity(t *testing.T, result AuditResult, source string, want AuditBuildIdentity) {
	t.Helper()
	row := auditSession(t, result, source)
	for _, identity := range row.BuildIdentities {
		if identity == want {
			return
		}
	}
	t.Fatalf("%s build identities = %+v, want %+v", source, row.BuildIdentities, want)
}

func auditMarkdownSection(t *testing.T, markdown, start, end string) string {
	t.Helper()
	startIndex := strings.Index(markdown, start)
	if startIndex < 0 {
		t.Fatalf("markdown missing %q:\n%s", start, markdown)
	}
	endIndex := strings.Index(markdown[startIndex:], end)
	if endIndex < 0 {
		return markdown[startIndex:]
	}
	return strings.TrimSpace(markdown[startIndex : startIndex+endIndex])
}
