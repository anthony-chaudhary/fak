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
	if got.Schema != "fak-model-default/v1" || got.Alias != "qwen38:27b" || !strings.Contains(got.Ref, "Qwen3.8-27B-Q4_K_M.gguf") || !got.Coding || !got.ToolCapable || got.Verdict != "HOLD" {
		t.Fatalf("unexpected default: %+v", got)
	}
	wantReasons := map[string]string{
		"identity": "UNPROVEN_IDENTITY", "macbook": "MISSING_EVIDENCE", "nvidia": "MISSING_EVIDENCE",
		"cache": "MISSING_EVIDENCE", "comparison": "MISSING_EVIDENCE", "support": "MISSING_EVIDENCE",
	}
	for _, reason := range got.Decision.Reasons {
		if wantReasons[reason.Family] != reason.Code {
			t.Fatalf("unexpected default reason: %+v", reason)
		}
		delete(wantReasons, reason.Family)
	}
	if len(wantReasons) != 0 {
		t.Fatalf("default decision omitted reasons: %v", wantReasons)
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
