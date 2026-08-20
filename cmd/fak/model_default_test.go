package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestModelDefaultJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runModelDefault(&stdout, &stderr, []string{"--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		Schema      string `json:"schema"`
		Alias       string `json:"alias"`
		Ref         string `json:"ref"`
		Coding      bool   `json:"coding"`
		ToolCapable bool   `json:"tool_capable"`
		Verdict     string `json:"verdict"`
		Decision    struct {
			Reasons []struct {
				Family string `json:"family"`
				Code   string `json:"code"`
			} `json:"reasons"`
		} `json:"decision"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-model-default/v1" || got.Alias != "qwen38:27b" || !strings.Contains(got.Ref, "Qwen3.8-27B-Q4_K_M.gguf") || !got.Coding || !got.ToolCapable || got.Verdict != "HOLD" || len(got.Decision.Reasons) != 6 {
		t.Fatalf("unexpected default: %+v", got)
	}
}

func TestModelDefaultRejectsMalformedEvidenceWithoutDownloading(t *testing.T) {
	path := t.TempDir() + "/evidence.json"
	if err := os.WriteFile(path, []byte(`{"schema":`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runModelDefault(&stdout, &stderr, []string{"--evidence", path, "--json"}); code != 1 {
		t.Fatalf("exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "decode evidence") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}
