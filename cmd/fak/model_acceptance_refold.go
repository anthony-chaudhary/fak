package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

type acceptanceRunKey struct {
	model, task string
	repetition  int
}

func runModelAcceptanceRefold(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model acceptance-refold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "completed report JSON containing immutable run identities")
	rawDir := fs.String("raw-dir", "", "directory containing immutable provider JSONL streams")
	output := fs.String("output", "", "corrected report JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*input) == "" || strings.TrimSpace(*rawDir) == "" || strings.TrimSpace(*output) == "" {
		fmt.Fprintln(stderr, "usage: fak model acceptance-refold --input REPORT --raw-dir DIR --output REPORT")
		return 2
	}
	in, err := decodeAcceptanceInput(*input)
	if err != nil {
		fmt.Fprintln(stderr, "fak model acceptance-refold:", err)
		return 2
	}
	refolded, err := refoldAcceptanceReport(in, *rawDir)
	if err != nil {
		fmt.Fprintln(stderr, "fak model acceptance-refold:", err)
		return 2
	}
	if err := writeJSONAtomic(*output, refolded, 0600); err != nil {
		fmt.Fprintln(stderr, "fak model acceptance-refold:", err)
		return 2
	}
	decision := modelaccept.Evaluate(refolded)
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(decision); err != nil {
		return 2
	}
	if decision.Verdict == modelaccept.Pass {
		return 0
	}
	return 4
}

func refoldAcceptanceReport(in modelaccept.Input, rawDir string) (modelaccept.Input, error) {
	tasks := make(map[string]modelaccept.Task, len(in.Corpus.Tasks))
	for _, task := range in.Corpus.Tasks {
		if _, exists := tasks[task.ID]; exists {
			return in, fmt.Errorf("duplicate task %q", task.ID)
		}
		tasks[task.ID] = task
	}
	models := make(map[string]modelaccept.ModelRequest, len(in.Models))
	for _, model := range in.Models {
		if _, exists := models[model.Model]; exists {
			return in, fmt.Errorf("duplicate model %q", model.Model)
		}
		models[model.Model] = model
	}
	expected := make(map[acceptanceRunKey]string)
	for _, model := range in.Models {
		for _, task := range in.Corpus.Tasks {
			if task.Tier < model.RequestedTier {
				continue
			}
			for rep := 1; rep <= task.Repetitions; rep++ {
				key := acceptanceRunKey{model: model.Model, task: task.ID, repetition: rep}
				expected[key] = fmt.Sprintf("%s--%s--%02d.jsonl", safeName(model.Model), safeName(task.ID), rep)
			}
		}
	}
	prior := make(map[acceptanceRunKey]modelaccept.Run, len(in.Runs))
	for _, run := range in.Runs {
		key := acceptanceRunKey{model: run.Model, task: run.Task, repetition: run.Repetition}
		if _, ok := expected[key]; !ok {
			return in, fmt.Errorf("unexpected prior run identity %s/%s/%d", run.Model, run.Task, run.Repetition)
		}
		if _, duplicate := prior[key]; duplicate {
			return in, fmt.Errorf("duplicate prior run identity %s/%s/%d", run.Model, run.Task, run.Repetition)
		}
		prior[key] = run
	}
	if len(prior) != len(expected) {
		return in, fmt.Errorf("prior report has %d runs, want %d", len(prior), len(expected))
	}
	entries, err := os.ReadDir(rawDir)
	if err != nil {
		return in, err
	}
	allowedFiles := make(map[string]bool, len(expected))
	for _, name := range expected {
		allowedFiles[name] = true
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".jsonl") {
			continue
		}
		if !allowedFiles[entry.Name()] {
			return in, fmt.Errorf("unexpected raw stream %q", entry.Name())
		}
	}
	keys := make([]acceptanceRunKey, 0, len(expected))
	for key := range expected {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].model != keys[j].model {
			return keys[i].model < keys[j].model
		}
		if keys[i].task != keys[j].task {
			return keys[i].task < keys[j].task
		}
		return keys[i].repetition < keys[j].repetition
	})
	refolded := make([]modelaccept.Run, 0, len(keys))
	for _, key := range keys {
		name := expected[key]
		raw, err := os.ReadFile(filepath.Join(rawDir, name))
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return in, fmt.Errorf("missing raw stream %q", name)
			}
			return in, err
		}
		parsed, err := parseClaudeAcceptance(raw, key.model, tasks[key.task])
		if err != nil {
			return in, fmt.Errorf("raw stream %q: %w", name, err)
		}
		old := prior[key]
		refolded = append(refolded, modelaccept.Run{
			Model: key.model, ActualModel: parsed.actualModel, Task: key.task, Repetition: key.repetition,
			Result: parsed.result, ToolValid: parsed.toolValid, ToolCalls: parsed.toolCalls, ToolTurns: parsed.toolTurns,
			Refusal: parsed.refusal, RetryCount: parsed.retryCount, Recovered: parsed.recovered,
			LatencyMS: parsed.latencyMS, InputTokens: parsed.inputTokens, CostUSD: parsed.costUSD,
			ObservedAt: old.ObservedAt,
		})
	}
	in.Runs = refolded
	return in, nil
}
