package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunConversationProfile_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runConversationProfile(&stdout, &stderr, []string{"help"})
	if code != 0 {
		t.Fatalf("expected code 0, got %d", code)
	}
	if !strings.Contains(stdout.String(), "Usage: fak conversation-profile validate <file>") {
		t.Errorf("expected usage, got:\n%s", stdout.String())
	}
}

func TestRunConversationProfile_ValidateValid(t *testing.T) {
	tmpDir := t.TempDir()
	validFile := filepath.Join(tmpDir, "profile.json")
	validJSON := `{
		"schema": "fak.conversation-profile/v1",
		"id": "valid-profile",
		"settings": {
			"response.detail": { "value": "brief", "fidelity": "required" }
		}
	}`
	if err := os.WriteFile(validFile, []byte(validJSON), 0600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runConversationProfile(&stdout, &stderr, []string{"validate", validFile})
	if code != 0 {
		t.Fatalf("expected code 0, got %d; stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "OK") {
		t.Errorf("expected stdout to contain OK, got %q", stdout.String())
	}
}

func TestRunConversationProfile_ValidateInvalid(t *testing.T) {
	tmpDir := t.TempDir()
	invalidFile := filepath.Join(tmpDir, "invalid.json")
	invalidJSON := `{"schema": "bad-schema"}`
	if err := os.WriteFile(invalidFile, []byte(invalidJSON), 0600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runConversationProfile(&stdout, &stderr, []string{"validate", invalidFile})
	if code != 1 {
		t.Fatalf("expected code 1, got %d", code)
	}
	if !strings.Contains(stderr.String(), "validation error:") {
		t.Errorf("expected stderr to contain validation error, got %q", stderr.String())
	}
}
