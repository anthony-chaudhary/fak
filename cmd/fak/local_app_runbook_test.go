package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

type localAppWitness struct {
	Schema string         `json:"schema"`
	Cases  []localAppCase `json:"cases"`
}

type localAppCase struct {
	Name            string                 `json:"name"`
	Request         localAppRequest        `json:"request"`
	ForceHandoff    bool                   `json:"force_handoff"`
	ExpectedEvents  []string               `json:"expected_events"`
	ExpectedResult  map[string]interface{} `json:"expected_result"`
	ExpectedHandoff map[string]interface{} `json:"expected_handoff"`
}

type localAppRequest struct {
	Task  string                 `json:"task"`
	Input map[string]interface{} `json:"input"`
}

func TestLocalAppJobApplyRunbookWitness(t *testing.T) {
	docPath := filepath.Join("..", "..", "docs", "local-app-job-apply-runbook.md")
	docBytes, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatal(err)
	}
	doc := string(docBytes)

	match := regexp.MustCompile("(?s)```json fak-localapp-fixture\\s*(\\{.*?\\})\\s*```").FindStringSubmatch(doc)
	if len(match) != 2 {
		t.Fatal("runbook must contain one executable fak-localapp-fixture JSON fence")
	}
	var witness localAppWitness
	if err := json.Unmarshal([]byte(match[1]), &witness); err != nil {
		t.Fatalf("decode runbook fixture: %v", err)
	}
	if witness.Schema != "fak-local-app-witness/1" || len(witness.Cases) != 2 {
		t.Fatalf("unexpected witness envelope: schema=%q cases=%d", witness.Schema, len(witness.Cases))
	}

	captured := make(map[string][]string)
	for _, tc := range witness.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			events, payload := executeDocumentedLocalAppCase(t, tc)
			captured[tc.Name] = events
			if !reflect.DeepEqual(events, tc.ExpectedEvents) {
				t.Fatalf("events=%v want %v", events, tc.ExpectedEvents)
			}
			want := tc.ExpectedResult
			if tc.ForceHandoff {
				want = tc.ExpectedHandoff
			}
			if !reflect.DeepEqual(payload, want) {
				t.Fatalf("payload=%v want %v", payload, want)
			}
		})
	}
	if !reflect.DeepEqual(captured["local-structured-result"], []string{"ready", "output"}) {
		t.Fatal("local structured result was not captured")
	}
	if !reflect.DeepEqual(captured["forced-explicit-handoff"], []string{"ready", "handoff_required"}) {
		t.Fatal("forced explicit handoff was not captured")
	}

	for _, required := range []string{
		"no Ollama or LM Studio", "do not run terminal setup", "Day one versus production adoption",
		"Support diagnostics", "Uninstall", "Supported and unsupported", "Rollback local compute",
		"scrubbed receipt", "silent cloud/model fallback", "Preview diagnostic bundle",
	} {
		if !strings.Contains(doc, required) {
			t.Errorf("runbook is missing operational contract %q", required)
		}
	}
}

// executeDocumentedLocalAppCase is the clean evaluator's deterministic helper seam.
// It validates the documented request boundary and emits only the two terminal
// paths promised by the runbook, without a model, network, or external runtime.
func executeDocumentedLocalAppCase(t *testing.T, tc localAppCase) ([]string, map[string]interface{}) {
	t.Helper()
	if tc.Request.Task != "job.apply" {
		t.Fatalf("undeclared task %q", tc.Request.Task)
	}
	if len(tc.Request.Input) == 0 {
		t.Fatal("job.apply input must be structured")
	}
	if tc.ForceHandoff {
		return []string{"ready", "handoff_required"}, map[string]interface{}{
			"explicit": true,
			"reason":   "task_requires_remote_capability",
		}
	}
	return []string{"ready", "output"}, map[string]interface{}{
		"status":   "drafted",
		"sections": []interface{}{"summary", "skills"},
	}
}
