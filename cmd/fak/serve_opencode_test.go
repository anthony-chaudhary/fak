package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestServeOpenCodeConfigPrintsSnippet(t *testing.T) {
	var buf bytes.Buffer
	sf := &serveFlags{
		addr:  strPtr("127.0.0.1:8080"),
		model: strPtr("qwen38:27b"),
	}

	runServeOpenCodeConfig(sf, &buf, false)
	out := buf.String()

	if !strings.Contains(out, "http://127.0.0.1:8080/v1") {
		t.Errorf("expected baseURL in output: %s", out)
	}
	if !strings.Contains(out, "qwen38:27b") {
		t.Errorf("expected model in output: %s", out)
	}
	if !strings.Contains(out, `"snapshot": false`) {
		t.Errorf("expected snapshot false in output: %s", out)
	}
}

func TestServeOpenCodeConfigWritesWorkspaceFile(t *testing.T) {
	ws := t.TempDir()
	t.Chdir(ws)

	var buf bytes.Buffer
	sf := &serveFlags{
		addr:  strPtr("127.0.0.1:9000"),
		model: strPtr("custom-model"),
	}

	runServeOpenCodeConfig(sf, &buf, true)

	configPath := filepath.Join(ws, "opencode.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read opencode.json: %v", err)
	}

	content := string(data)
	if !strings.Contains(content, "http://127.0.0.1:9000/v1") {
		t.Errorf("expected baseURL in file: %s", content)
	}
	if !strings.Contains(content, "custom-model") {
		t.Errorf("expected model in file: %s", content)
	}
}
