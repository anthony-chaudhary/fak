package main

import (
	"bytes"
	"encoding/json"
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
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-model-default/v1" || got.Alias != "qwen38:27b" || !strings.Contains(got.Ref, "Qwen3.8-27B-Q4_K_M.gguf") || !got.Coding || !got.ToolCapable {
		t.Fatalf("unexpected default: %+v", got)
	}
}
