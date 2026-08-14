package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/toolcatalog"
)

func catalogRegistration(t *testing.T, name string, aliases map[string]string) toolcatalog.Registration {
	t.Helper()
	aliasJSON, err := json.Marshal(aliases)
	if err != nil {
		t.Fatal(err)
	}
	source := "---\nname: " + name + "\ndescription: Search the repository\n---\n```fak-program\n{\"version\":\"fak.skill-program/v1\",\"name\":\"" + name + "\",\"description\":\"Search the repository\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}},\"executor\":{\"argv\":[\"fak\",\"code\",\"search\"]},\"aliases\":" + string(aliasJSON) + "}\n```"
	reg, err := toolcatalog.CompileSkill([]byte(source), "skills/"+name+"/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

func TestToolCatalogPageFeedsOpenAIAndAnthropicRequests(t *testing.T) {
	reg := catalogRegistration(t, "repo_search", map[string]string{"codex": "functions.shell_command"})
	snapshot, err := toolcatalog.Expose([]toolcatalog.Registration{reg}, []string{"repo_search"}, "codex")
	if err != nil {
		t.Fatal(err)
	}
	pages := ctxmmu.NewToolPageTable(ctxmmu.New())
	pinned, err := pages.RegisterToolCatalogSnapshot(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	tools, audit, err := ToolDefsFromCatalogPage(context.Background(), pages, pinned.PageHash, pinned.SnapshotDigest)
	if err != nil {
		t.Fatal(err)
	}
	if audit.SnapshotDigest != snapshot.Digest || len(tools) != 1 || tools[0].Function.Name != "functions.shell_command" {
		t.Fatalf("tools=%#v audit=%#v", tools, audit)
	}
	for _, provider := range []Provider{ProviderOpenAI, ProviderAnthropic} {
		adapter, err := NewTranscriptAdapter(provider)
		if err != nil {
			t.Fatal(err)
		}
		body, err := adapter.MarshalRequest(adapterRequest{Model: "m", Messages: []Message{{Role: "user", Content: "find it"}}, Tools: tools, MaxTokens: 64})
		if err != nil {
			t.Fatal(err)
		}
		if !json.Valid(body) || !containsJSONToolName(body, "functions.shell_command") {
			t.Fatalf("provider %s request did not carry selected alias: %s", provider, body)
		}
		if containsJSONToolName(body, "repo_search") {
			t.Fatalf("provider %s leaked canonical name instead of dialect alias: %s", provider, body)
		}
	}
}

func TestInstalledButUnexposedToolCannotEnterProviderRequest(t *testing.T) {
	reg := catalogRegistration(t, "repo_search", nil)
	snapshot, err := toolcatalog.Expose([]toolcatalog.Registration{reg}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	pages := ctxmmu.NewToolPageTable(ctxmmu.New())
	pinned, err := pages.RegisterToolCatalogSnapshot(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	tools, audit, err := ToolDefsFromCatalogPage(context.Background(), pages, pinned.PageHash, pinned.SnapshotDigest)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 0 || len(audit.Omissions) != 1 || audit.Omissions[0].Reason != "NOT_SELECTED" {
		t.Fatalf("tools=%#v audit=%#v", tools, audit)
	}
	adapter, _ := NewTranscriptAdapter(ProviderOpenAI)
	body, err := adapter.MarshalRequest(adapterRequest{Model: "m", Messages: []Message{{Role: "user", Content: "find it"}}, Tools: tools, MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatal(err)
	}
	if _, exists := envelope["tools"]; exists {
		t.Fatalf("installed-but-unexposed tool entered request: %s", body)
	}
}

func containsJSONToolName(body []byte, name string) bool {
	needle, _ := json.Marshal(name)
	return json.Valid(body) && jsonContains(body, needle)
}

func jsonContains(body, needle []byte) bool {
	for i := 0; i+len(needle) <= len(body); i++ {
		if string(body[i:i+len(needle)]) == string(needle) {
			return true
		}
	}
	return false
}
