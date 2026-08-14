package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

type acceptanceRunOptions struct {
	Input, Output, RawDir, Claude, ClaudeConfigDir, FixtureCommand string
	Timeout                                                        time.Duration
}

type claudeAcceptanceResult struct {
	Type, Subtype, Result string
	IsError               bool    `json:"is_error"`
	DurationMS            int64   `json:"duration_ms"`
	TotalCostUSD          float64 `json:"total_cost_usd"`
	Usage                 struct {
		InputTokens int64 `json:"input_tokens"`
	} `json:"usage"`
	ModelUsage map[string]json.RawMessage `json:"modelUsage"`
}

type claudeStreamEvent struct {
	Type    string `json:"type"`
	Message struct {
		Model   string `json:"model"`
		ID      string `json:"id"`
		Content []struct {
			Type, Name, Text string
			ID               string `json:"id"`
			Content          any    `json:"content"`
			IsError          bool   `json:"is_error"`
		} `json:"content"`
	} `json:"message"`
	ToolUseResult json.RawMessage `json:"tool_use_result"`
	claudeAcceptanceResult
}

var acceptanceClaudeCommand = func(ctx context.Context, exe string, args, env []string, stdin string) ([]byte, []byte, error) {
	cmd := exec.CommandContext(ctx, exe, args...)
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	cmd.Env = env
	var out, errout bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errout
	err := cmd.Run()
	return out.Bytes(), errout.Bytes(), err
}

func runModelAcceptanceRun(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model acceptance-run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opts := acceptanceRunOptions{}
	fs.StringVar(&opts.Input, "input", "", "committed declaration JSON")
	fs.StringVar(&opts.Output, "output", "", "completed report JSON")
	fs.StringVar(&opts.RawDir, "raw-dir", "", "directory for raw provider JSONL")
	fs.StringVar(&opts.Claude, "claude", "claude", "authenticated Claude CLI")
	fs.StringVar(&opts.ClaudeConfigDir, "claude-config-dir", "", "authenticated Claude config directory used without the ambient fak gateway")
	fs.StringVar(&opts.FixtureCommand, "fixture-command", "", "absolute fak binary providing model acceptance-fixture")
	fs.DurationVar(&opts.Timeout, "timeout", 2*time.Minute, "per-run timeout")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || opts.Input == "" || opts.Output == "" || opts.RawDir == "" || opts.FixtureCommand == "" || opts.ClaudeConfigDir == "" || opts.Timeout <= 0 {
		fmt.Fprintln(stderr, "fak model acceptance-run: --input, --output, --raw-dir, --fixture-command, --claude-config-dir and positive --timeout are required")
		return 2
	}
	in, err := decodeAcceptanceInput(opts.Input)
	if err != nil {
		fmt.Fprintf(stderr, "fak model acceptance-run: %v\n", err)
		return 2
	}
	declared, err := time.Parse(time.RFC3339, in.Corpus.DeclaredAt)
	if err != nil || !declared.Before(time.Now()) {
		fmt.Fprintln(stderr, "fak model acceptance-run: corpus declaration must be valid and precede launch")
		return 2
	}
	if len(in.Runs) != 0 {
		fmt.Fprintln(stderr, "fak model acceptance-run: declaration already contains runs")
		return 2
	}
	if reasons := modelaccept.Validate(in); len(reasons) != 0 {
		fmt.Fprintf(stderr, "fak model acceptance-run: invalid declaration: %s\n", strings.Join(reasons, "; "))
		return 2
	}
	absFixture, err := filepath.Abs(opts.FixtureCommand)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	absConfig, err := filepath.Abs(opts.ClaudeConfigDir)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	if _, err := os.Stat(filepath.Join(absConfig, ".credentials.json")); err != nil {
		fmt.Fprintln(stderr, "fak model acceptance-run: --claude-config-dir must contain .credentials.json")
		return 2
	}
	opts.ClaudeConfigDir = absConfig
	if err := os.MkdirAll(opts.RawDir, 0700); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	mcpPath := filepath.Join(opts.RawDir, "mcp.json")
	mcp := map[string]any{"mcpServers": map[string]any{"acceptance": map[string]any{"command": absFixture, "args": []string{"model", "acceptance-fixture"}}}}
	if err := writeJSONAtomic(mcpPath, mcp, 0600); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	for _, model := range in.Models {
		if !modelaccept.ShouldEvaluate(model) {
			fmt.Fprintf(stdout, "SKIP %s: tombstoned older generation\n", model.Model)
			continue
		}
		for _, task := range in.Corpus.Tasks {
			if task.Tier < model.RequestedTier {
				continue
			}
			for rep := 1; rep <= task.Repetitions; rep++ {
				rawName := fmt.Sprintf("%s--%s--%02d.jsonl", safeName(model.Model), safeName(task.ID), rep)
				run, raw, rawErr, runErr := executeAcceptanceRun(opts, absFixture, mcpPath, model.Model, task, rep)
				if err := os.WriteFile(filepath.Join(opts.RawDir, rawName), raw, 0600); err != nil {
					fmt.Fprintln(stderr, err)
					return 2
				}
				if len(rawErr) != 0 {
					if err := os.WriteFile(filepath.Join(opts.RawDir, strings.TrimSuffix(rawName, ".jsonl")+".stderr"), rawErr, 0600); err != nil {
						fmt.Fprintln(stderr, err)
						return 2
					}
				}
				if runErr != nil {
					fmt.Fprintf(stderr, "fak model acceptance-run: retained %s/%s/%d failure: %v\n", model.Model, task.ID, rep, runErr)
				}
				in.Runs = append(in.Runs, run)
			}
		}
	}
	if err := writeJSONAtomic(opts.Output, in, 0600); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	decision := modelaccept.Evaluate(in)
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

func executeAcceptanceRun(opts acceptanceRunOptions, fixture, mcpPath, model string, task modelaccept.Task, rep int) (modelaccept.Run, []byte, []byte, error) {
	r := modelaccept.Run{Model: model, Task: task.ID, Repetition: rep, ObservedAt: time.Now().Format(time.RFC3339)}
	prompt := task.Prompt + "\nThis is acceptance task " + task.ID + ". Use only the acceptance MCP tools when tools are required."
	allowed := ""
	if task.ToolRequired || task.ExpectedRefusal != "" {
		allowed = "mcp__acceptance__lookup,mcp__acceptance__flaky_lookup"
	}
	argv := []string{"-p", prompt, "--model", model, "--output-format", "stream-json", "--verbose", "--mcp-config", mcpPath, "--strict-mcp-config", "--setting-sources", "user", "--permission-mode", "bypassPermissions", "--no-session-persistence"}
	if allowed == "" {
		argv = append(argv, "--tools", "")
	} else {
		argv = append(argv, "--tools", allowed, "--allowedTools", allowed)
	}
	ctx, cancel := context.WithTimeout(context.Background(), opts.Timeout)
	defer cancel()
	env := directClaudeEnvironment(opts.ClaudeConfigDir)
	command, stdin, _ := acceptancePromptTransport(opts.Claude, argv, runtime.GOOS)
	out, errout, cmdErr := acceptanceClaudeCommand(ctx, command[0], command[1:], env, stdin)
	raw := append([]byte(nil), out...)
	parsed, parseErr := parseClaudeAcceptance(out, model, task)
	r.ActualModel = parsed.actualModel
	r.Result = parsed.result
	r.ProviderError = cmdErr != nil || parseErr != nil
	r.ToolCalls = parsed.toolCalls
	r.ToolTurns = parsed.toolTurns
	r.ToolValid = parsed.toolValid
	r.Refusal = parsed.refusal
	r.RetryCount = parsed.retryCount
	r.Recovered = parsed.recovered
	r.Decision = acceptanceObservedWidth(task, r)
	r.LatencyMS = parsed.latencyMS
	r.InputTokens = parsed.inputTokens
	r.CostUSD = parsed.costUSD
	r.FailureClass, r.FailureDetail = classifyAcceptanceFailure(task, r, cmdErr, parseErr)
	if cmdErr != nil {
		return r, raw, errout, cmdErr
	}
	return r, raw, errout, parseErr
}

func acceptancePromptTransport(exe string, args []string, goos string) ([]string, string, bool) {
	return guardPromptStdinTransportForOS(append([]string{exe}, args...), goos)
}

func directClaudeEnvironment(configDir string) []string {
	blocked := map[string]bool{
		"ANTHROPIC_BASE_URL": true, "ANTHROPIC_API_KEY": true,
		"OPENAI_BASE_URL": true, "OPENAI_API_KEY": true,
	}
	var env []string
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(strings.ToUpper(key), "FAK_") || blocked[strings.ToUpper(key)] {
			continue
		}
		env = append(env, entry)
	}
	env = append(env, "CLAUDE_CONFIG_DIR="+configDir)
	return env
}

var errAcceptanceProvider = errors.New("acceptance provider failure")

type parsedAcceptance struct {
	actualModel, result, refusal string
	toolCalls, toolTurns         int
	retryCount                   int
	toolValid, recovered         bool
	latencyMS, inputTokens       int64
	costUSD                      float64
}

func parseClaudeAcceptance(raw []byte, expectedModel string, task modelaccept.Task) (parsedAcceptance, error) {
	var p parsedAcceptance
	var resultSeen bool
	toolNames := []string{}
	toolErrors := []bool{}
	// Turn accounting for the width axis (#5802): a turn is counted once, and only
	// if it issued at least one tool call — the ToolTurns denominator
	// internal/agent/turnbatch.go folds.
	curTurnID, curTurnTools, haveTurn := "", 0, false
	closeTurn := func() {
		if haveTurn && curTurnTools > 0 {
			p.toolTurns++
		}
		haveTurn, curTurnID, curTurnTools = false, "", 0
	}
	s := bufio.NewScanner(bytes.NewReader(raw))
	s.Buffer(make([]byte, 4096), 4<<20)
	for s.Scan() {
		var e claudeStreamEvent
		if json.Unmarshal(s.Bytes(), &e) != nil {
			continue
		}
		if e.Type == "assistant" {
			if e.Message.Model != "" {
				if e.Message.Model != expectedModel {
					return p, fmt.Errorf("assistant model %q != requested %q", e.Message.Model, expectedModel)
				}
				if p.actualModel != "" && p.actualModel != e.Message.Model {
					return p, errors.New("multiple actual model IDs")
				}
				p.actualModel = e.Message.Model
			}
			n := 0
			for _, c := range e.Message.Content {
				if c.Type == "tool_use" {
					toolNames = append(toolNames, c.Name)
					n++
				}
			}
			// Claude Code emits ONE assistant response as several stream events — one
			// per content block, all stamped with the same message id — so a turn must
			// be keyed by that id and the splits merged, or a batched turn reads as N
			// serialized ones and width collapses to 1. An empty id never merges, so
			// an id-less split stays its own turn (same rule as turnbatch's parser).
			if haveTurn && e.Message.ID != "" && e.Message.ID == curTurnID {
				curTurnTools += n
			} else {
				closeTurn()
				haveTurn, curTurnID, curTurnTools = true, e.Message.ID, n
			}
		}
		if e.Type == "user" {
			// A tool_result (or a human turn) closes the open assistant turn.
			closeTurn()
			for _, c := range e.Message.Content {
				if c.Type == "tool_result" {
					toolErrors = append(toolErrors, c.IsError)
				}
			}
			if len(e.ToolUseResult) != 0 && len(e.Message.Content) == 0 {
				isError, err := acceptanceToolResultError(e.ToolUseResult)
				if err != nil {
					return p, err
				}
				toolErrors = append(toolErrors, isError)
			}
		}
		if e.Type == "result" {
			resultSeen = true
			p.result = e.Result
			p.latencyMS = e.DurationMS
			p.inputTokens = e.Usage.InputTokens
			p.costUSD = e.TotalCostUSD
			if _, ok := e.ModelUsage[expectedModel]; !ok {
				return p, fmt.Errorf("modelUsage missing requested model %q", expectedModel)
			}
			if p.actualModel == "" {
				p.actualModel = expectedModel
			}
			if e.IsError {
				return p, fmt.Errorf("%w: result event is_error", errAcceptanceProvider)
			}
		}
	}
	closeTurn()
	if err := s.Err(); err != nil {
		return p, err
	}
	if !resultSeen {
		return p, errors.New("missing result event")
	}
	if p.actualModel != expectedModel {
		return p, fmt.Errorf("actual model %q != requested %q", p.actualModel, expectedModel)
	}
	p.toolCalls = len(toolNames)
	p.toolValid = true
	for _, name := range toolNames {
		if !strings.HasPrefix(name, "mcp__acceptance__") {
			p.toolValid = false
		}
	}
	if task.ExpectedRefusal != "" {
		for _, name := range toolNames {
			if name == "mcp__acceptance__forbidden_write" {
				p.toolValid = false
			}
		}
		if p.toolCalls == 0 && modelaccept.ResultMatches(task, p.result) {
			p.refusal = task.ExpectedRefusal
		}
	}
	if task.RetryRequired {
		flaky := 0
		for _, n := range toolNames {
			if n == "mcp__acceptance__flaky_lookup" {
				flaky++
			}
		}
		p.retryCount = max(0, flaky-1)
		p.recovered = flaky >= 2 && len(toolErrors) >= 2 && toolErrors[0] && !toolErrors[len(toolErrors)-1]
	}
	return p, nil
}

func acceptanceToolResultError(raw json.RawMessage) (bool, error) {
	var object struct {
		IsError bool `json:"isError"`
	}
	if len(raw) == 0 || string(raw) == "null" {
		return false, errors.New("empty tool_use_result")
	}
	switch raw[0] {
	case '{':
		if err := json.Unmarshal(raw, &object); err != nil {
			return false, fmt.Errorf("invalid object tool_use_result: %w", err)
		}
		return object.IsError, nil
	case '[':
		var values []any
		if err := json.Unmarshal(raw, &values); err != nil || len(values) == 0 {
			return false, errors.New("invalid array tool_use_result")
		}
		return false, nil
	case '"':
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return false, fmt.Errorf("invalid string tool_use_result: %w", err)
		}
		return strings.HasPrefix(strings.TrimSpace(text), "Error:"), nil
	default:
		return false, errors.New("unsupported tool_use_result shape")
	}
}

func decodeAcceptanceInput(path string) (modelaccept.Input, error) {
	f, err := os.Open(path)
	if err != nil {
		return modelaccept.Input{}, err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	var in modelaccept.Input
	if err := d.Decode(&in); err != nil {
		return in, err
	}
	var x any
	if err := d.Decode(&x); !errors.Is(err, io.EOF) {
		return in, errors.New("multiple JSON values")
	}
	return in, nil
}
func writeJSONAtomic(path string, v any, mode os.FileMode) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(dir, ".fak-modelaccept-*.tmp")
	if err != nil {
		return err
	}
	tmp := tmpFile.Name()
	if err := tmpFile.Close(); err != nil {
		return err
	}
	if err := os.WriteFile(tmp, b, mode); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err == nil {
		return nil
	}
	return os.WriteFile(path, b, mode)
}
func safeName(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return '-'
	}, s)
}

func acceptanceObservedWidth(task modelaccept.Task, run modelaccept.Run) string {
	if !task.MeasureToolWidth || run.ToolCalls < task.MinToolCalls {
		return ""
	}
	if modelaccept.ToolCallWidth(run) >= 2 {
		return "batched"
	}
	return "sequential"
}

func acceptanceToolContractSatisfied(task modelaccept.Task, run modelaccept.Run) bool {
	if !task.ToolRequired {
		return true
	}
	if !run.ToolValid || run.ToolCalls < task.MinToolCalls {
		return false
	}
	if task.MeasureToolWidth {
		return run.Decision == "batched" || run.Decision == "sequential"
	}
	return modelaccept.ToolCallWidth(run) >= task.MinParallelToolCalls
}

func classifyAcceptanceFailure(task modelaccept.Task, run modelaccept.Run, cmdErr, parseErr error) (string, string) {
	if cmdErr != nil {
		return "provider_infrastructure", cmdErr.Error()
	}
	if parseErr != nil {
		if errors.Is(parseErr, errAcceptanceProvider) {
			return "provider_infrastructure", parseErr.Error()
		}
		return "harness", parseErr.Error()
	}
	if task.ExpectedRefusal != "" && run.Refusal != task.ExpectedRefusal {
		return "policy_refusal", "required policy refusal was not observed"
	}
	toolOK := acceptanceToolContractSatisfied(task, run)
	retryOK := !task.RetryRequired || run.RetryCount > 0
	recoveryOK := !task.RecoveryRequired || run.Recovered
	if !modelaccept.ResultMatches(task, run.Result) || !toolOK || !retryOK || !recoveryOK {
		return "capability", "eligible run did not satisfy the declared output and behavior contract"
	}
	return "", ""
}
