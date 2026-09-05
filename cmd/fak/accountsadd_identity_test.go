package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

func TestWriteClaudeJSONIdentityHandlesJSONNull(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".claude.json")
	if err := os.WriteFile(path, []byte("null"), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	id := accounts.ProbedIdentity{
		Email:       "test@example.com",
		AccountUUID: "uuid-1234",
	}

	if err := writeClaudeJSONIdentity(dir, id); err != nil {
		t.Fatalf("writeClaudeJSONIdentity returned unexpected error: %v", err)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read .claude.json: %v", err)
	}

	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("failed to parse .claude.json as json: %v", err)
	}

	rawAccount, ok := doc["oauthAccount"]
	if !ok || rawAccount == nil {
		t.Fatalf("expected oauthAccount in doc, got %v", doc)
	}

	accountMap, ok := rawAccount.(map[string]any)
	if !ok {
		t.Fatalf("expected oauthAccount to be map[string]any, got %T", rawAccount)
	}

	if accountMap["emailAddress"] != id.Email {
		t.Errorf("expected email %q, got %v", id.Email, accountMap["emailAddress"])
	}
	if accountMap["accountUuid"] != id.AccountUUID {
		t.Errorf("expected uuid %q, got %v", id.AccountUUID, accountMap["accountUuid"])
	}
}

func TestWriteClaudeJSONIdentityPreservesExistingKeys(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".claude.json")
	initial := `{"theme": "dark", "version": 1}`
	if err := os.WriteFile(path, []byte(initial), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	id := accounts.ProbedIdentity{
		Email:       "preserve@example.com",
		AccountUUID: "uuid-5678",
	}

	if err := writeClaudeJSONIdentity(dir, id); err != nil {
		t.Fatalf("writeClaudeJSONIdentity returned unexpected error: %v", err)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read .claude.json: %v", err)
	}

	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("failed to parse .claude.json as json: %v", err)
	}

	if doc["theme"] != "dark" {
		t.Errorf("expected theme 'dark', got %v", doc["theme"])
	}
	if doc["oauthAccount"] == nil {
		t.Errorf("expected oauthAccount to be populated")
	}
}

func TestWriteClaudeJSONIdentityReplacesCorruptFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".claude.json")
	if err := os.WriteFile(path, []byte("{corrupt-json"), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	id := accounts.ProbedIdentity{
		Email:       "corrupt@example.com",
		AccountUUID: "uuid-9999",
	}

	if err := writeClaudeJSONIdentity(dir, id); err != nil {
		t.Fatalf("writeClaudeJSONIdentity returned unexpected error: %v", err)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read .claude.json: %v", err)
	}

	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("failed to parse .claude.json as json: %v", err)
	}

	if doc["oauthAccount"] == nil {
		t.Errorf("expected oauthAccount in doc after replacing corrupt file")
	}
}
