package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestServePiConfigPrintsSnippet(t *testing.T) {
	var buf bytes.Buffer
	sf := &serveFlags{
		addr:  strPtr("127.0.0.1:8080"),
		model: strPtr("qwen38:27b-q4"),
	}

	runServePiConfig(sf, &buf, false)
	out := buf.String()

	if !strings.Contains(out, "http://127.0.0.1:8080/v1") {
		t.Errorf("expected baseURL in output: %s", out)
	}
	if !strings.Contains(out, "qwen38:27b-q4") {
		t.Errorf("expected model in output: %s", out)
	}
	if !strings.Contains(out, `"openai-completions"`) {
		t.Errorf("expected openai-completions in output: %s", out)
	}
	if !strings.Contains(out, `"supportsDeveloperRole": false`) {
		t.Errorf("expected supportsDeveloperRole: false in output: %s", out)
	}
}

func TestServePiConfigWritesFile(t *testing.T) {
	ws := t.TempDir()
	configPath := filepath.Join(ws, "models.json")

	var buf bytes.Buffer
	sf := &serveFlags{
		addr:         strPtr("127.0.0.1:9000"),
		model:        strPtr("custom-model"),
		piConfigPath: strPtr(configPath),
	}

	runServePiConfig(sf, &buf, true)

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read models.json: %v", err)
	}

	content := string(data)
	if !strings.Contains(content, "http://127.0.0.1:9000/v1") {
		t.Errorf("expected baseURL in file: %s", content)
	}
	if !strings.Contains(content, "custom-model") {
		t.Errorf("expected model in file: %s", content)
	}
	if !strings.Contains(content, `"openai-completions"`) {
		t.Errorf("expected api in file: %s", content)
	}
}
