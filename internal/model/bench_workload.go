package model

import (
	"encoding/json"
	"fmt"
	"os"
)

// BenchWorkload is a recorded agent-run shape for compute benchmarks. The model
// benchmarks still feed token IDs directly; token values are deterministic because
// this path measures transformer compute cost, while these counts come from real
// agent prompt/completion usage instead of synthetic prompt lengths.
type BenchWorkload struct {
	Schema      string          `json:"schema,omitempty"`
	Name        string          `json:"name,omitempty"`
	Source      string          `json:"source,omitempty"`
	GeneratedAt string          `json:"generated_at,omitempty"`
	Cases       []BenchCase     `json:"cases"`
	Meta        json.RawMessage `json:"meta,omitempty"`
}

// BenchCase is one recorded run in a BenchWorkload: its prompt/completion token counts (the
// compute the benchmark replays) plus the turn/tool-call provenance from the real agent run.
type BenchCase struct {
	Name             string `json:"name"`
	Source           string `json:"source,omitempty"`
	Arm              string `json:"arm,omitempty"`
	TranscriptSHA    string `json:"transcript_sha,omitempty"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	Turns            int    `json:"turns,omitempty"`
	ToolCalls        int    `json:"tool_calls,omitempty"`
}

// LoadBenchWorkload reads and decodes a BenchWorkload JSON file, defaulting each unnamed case
// and rejecting a workload with no cases or any case whose prompt/completion token count is <= 0.
func LoadBenchWorkload(path string) (*BenchWorkload, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var w BenchWorkload
	if err := json.Unmarshal(b, &w); err != nil {
		return nil, err
	}
	if len(w.Cases) == 0 {
		return nil, fmt.Errorf("workload has no cases")
	}
	for i := range w.Cases {
		c := &w.Cases[i]
		if c.Name == "" {
			c.Name = fmt.Sprintf("case-%d", i)
		}
		if c.PromptTokens <= 0 {
			return nil, fmt.Errorf("%s: prompt_tokens must be > 0", c.Name)
		}
		if c.CompletionTokens <= 0 {
			return nil, fmt.Errorf("%s: completion_tokens must be > 0", c.Name)
		}
	}
	return &w, nil
}
