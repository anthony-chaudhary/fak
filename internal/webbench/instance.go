// Package webbench turns frontier web/browser agent benchmarks into fak-native
// benchmarks whose results directly measure the value of fak's session value stack
// on multi-turn web automation tasks. The design mirrors swebench: load task
// instances, model agent-workload geometry, compute cost comparisons across arms
// (naive re-prefill vs per-agent KV vs fak fused), and evaluate against official
// harnesses where available.
//
// Supported frontier web benchmarks:
// - Browser Agent Benchmark (browser-use.com): 100 hard browser tasks
// - WebVoyager: 586 diverse web interaction tasks
// - BrowseComp (OpenAI): information location tasks
//
// The key metrics for web agents:
// 1. Prefill work-elimination (A/C, B/C ratios)
// 2. Navigation turns + tokens (turn-tax for web)
// 3. DOM processing cost (ingested page state)
// 4. Task success rate (where official harness exists)
package webbench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Instance is one web/browser agent task. The schema is designed to accommodate
// multiple frontier benchmarks while keeping the core fields consistent.
type Instance struct {
	// Core identification
	TaskID    string `json:"task_id"`    // unique task identifier
	Benchmark string `json:"benchmark"`  // source benchmark name (e.g., "browser-agent", "webvoyager")
	SourceURL string `json:"source_url"` // starting URL for the task

	// Task definition
	Description  string `json:"description"`            // natural language task description
	Instructions string `json:"instructions,omitempty"` // detailed step-by-step if available

	// Expected actions / trajectory (for geometry modeling)
	Actions []Action `json:"actions,omitempty"` // ground-truth action sequence

	// Evaluation criteria
	SuccessCriteria string `json:"success_criteria,omitempty"` // how success is judged
	ExpectedOutput  string `json:"expected_output,omitempty"`  // expected final state/value
	Difficulty      string `json:"difficulty,omitempty"`       // difficulty bucket if available

	// Metadata
	Category  string `json:"category,omitempty"` // task category (e.g., "form-fill", "navigation", "scraping")
	Domain    string `json:"domain,omitempty"`   // website domain
	CreatedAt string `json:"created_at,omitempty"`
}

// Action represents one step in a web automation trajectory.
type Action struct {
	Type      string `json:"type"`                 // "click", "type", "navigate", "scroll", "wait", "extract"
	Target    string `json:"target,omitempty"`     // selector, text, or URL
	Value     string `json:"value,omitempty"`      // text to type, data to extract
	Tokens    int    `json:"tokens,omitempty"`     // estimated assistant tokens for this action
	PageState int    `json:"page_state,omitempty"` // DOM tokens ingested after this action
}

// Dataset is an ordered collection of web task instances.
type Dataset struct {
	Instances []Instance
	byID      map[string]int
}

// NewDataset creates an indexed dataset from instances.
func NewDataset(insts []Instance) *Dataset {
	d := &Dataset{byID: make(map[string]int, len(insts))}
	for _, in := range insts {
		if j, ok := d.byID[in.TaskID]; ok {
			d.Instances[j] = in
			continue
		}
		d.byID[in.TaskID] = len(d.Instances)
		d.Instances = append(d.Instances, in)
	}
	return d
}

// Len returns the number of instances in the dataset.
func (d *Dataset) Len() int { return len(d.Instances) }

// Get returns the instance with id and whether it was present.
func (d *Dataset) Get(id string) (Instance, bool) {
	if j, ok := d.byID[id]; ok {
		return d.Instances[j], true
	}
	return Instance{}, false
}

// Limit returns a dataset with at most n instances.
func (d *Dataset) Limit(n int) *Dataset {
	if n >= d.Len() {
		return d
	}
	return &Dataset{
		Instances: d.Instances[:n],
		byID:      make(map[string]int, n),
	}
}

// LoadDataset reads a web benchmark dataset. Supports JSONL (one instance per line)
// or a single JSON array of instances.
func LoadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset: %w", err)
	}
	defer f.Close()

	// Detect format by first non-space byte.
	buf := make([]byte, 1)
	if _, err := f.Read(buf); err != nil {
		return nil, fmt.Errorf("read first byte: %w", err)
	}
	if _, err := f.Seek(0, 0); err != nil {
		return nil, fmt.Errorf("seek: %w", err)
	}

	var insts []Instance
	if buf[0] == '[' {
		// JSON array.
		if err := json.NewDecoder(f).Decode(&insts); err != nil {
			return nil, fmt.Errorf("decode JSON array: %w", err)
		}
	} else {
		// JSONL.
		s := bufio.NewScanner(f)
		for s.Scan() {
			line := strings.TrimSpace(s.Text())
			if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
				continue
			}
			var in Instance
			if err := json.Unmarshal([]byte(line), &in); err != nil {
				return nil, fmt.Errorf("decode JSONL line: %w", err)
			}
			insts = append(insts, in)
		}
		if err := s.Err(); err != nil {
			return nil, fmt.Errorf("scan JSONL: %w", err)
		}
	}

	return NewDataset(insts), nil
}

// LoadBrowserAgentBenchmark loads the browser-use.com benchmark format.
// This is a helper for the specific format used by that benchmark.
func LoadBrowserAgentBenchmark(path string) (*Dataset, error) {
	d, err := LoadDataset(path)
	if err != nil {
		return nil, err
	}
	// Tag all instances with the benchmark source if not already set.
	for i := range d.Instances {
		if d.Instances[i].Benchmark == "" {
			d.Instances[i].Benchmark = "browser-agent"
		}
	}
	return d, nil
}

// DifficultyStats returns per-difficulty instance counts.
func (d *Dataset) DifficultyStats() map[string]int {
	stats := make(map[string]int)
	for _, in := range d.Instances {
		diff := in.Difficulty
		if diff == "" {
			diff = "unknown"
		}
		stats[diff]++
	}
	return stats
}

// CategoryStats returns per-category instance counts.
func (d *Dataset) CategoryStats() map[string]int {
	stats := make(map[string]int)
	for _, in := range d.Instances {
		cat := in.Category
		if cat == "" {
			cat = "uncategorized"
		}
		stats[cat]++
	}
	return stats
}

// SortedDifficulties returns difficulty keys sorted by name.
func (d *Dataset) SortedDifficulties() []string {
	stats := d.DifficultyStats()
	keys := make([]string, 0, len(stats))
	for k := range stats {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// MergeDifficulty overlays difficulty annotations from a map.
// The map format matches swebench: {"task_id": "difficulty"}.
func (d *Dataset) MergeDifficulty(diffMap map[string]string) int {
	n := 0
	for i := range d.Instances {
		id := d.Instances[i].TaskID
		if diff, ok := diffMap[id]; ok && d.Instances[i].Difficulty == "" {
			d.Instances[i].Difficulty = diff
			n++
		}
	}
	return n
}

// EstimateTokens approximates token count for text using ~4 chars per token.
// This is the same estimator swebench uses for problem statements.
func EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return len(text) / 4
}
