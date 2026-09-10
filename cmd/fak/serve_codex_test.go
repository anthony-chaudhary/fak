package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunServeCodexConfigPreview(t *testing.T) {
	addr := "127.0.0.1:8080"
	model := "qwen38:27b-q4"
	sf := &serveFlags{
		addr:  &addr,
		model: &model,
	}

	var buf bytes.Buffer
	runServeCodexConfig(sf, &buf, false)

	out := buf.String()
	if strings.Contains(out, `model_provider = "fak"`) {
		t.Errorf("expected preview NOT to hijack model_provider: %s", out)
	}
	if !strings.Contains(out, `[model_providers.fak]`) {
		t.Errorf("expected [model_providers.fak] in preview: %s", out)
	}
	if !strings.Contains(out, `base_url = "http://127.0.0.1:8080/v1"`) {
		t.Errorf("expected base_url in preview: %s", out)
	}
	if !strings.Contains(out, `wire_api = "responses"`) {
		t.Errorf("expected wire_api in preview: %s", out)
	}
}

func TestRunServeCodexConfigWrite(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "config.toml")

	addr := "127.0.0.1:8080"
	model := "qwen38:27b-q4"
	sf := &serveFlags{
		addr:            &addr,
		model:           &model,
		codexConfigPath: &targetPath,
	}

	var buf bytes.Buffer
	runServeCodexConfig(sf, &buf, true)

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("failed to read written config: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, `[model_providers.fak]`) {
		t.Errorf("written config missing provider table: %s", content)
	}
}
