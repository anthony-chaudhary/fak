package swebench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fleet.go is the fak-native SWE-bench solver: the "fleet" runner drives a real
// coding-agent loop against a fak gateway (an OpenAI-compatible /v1/chat/completions
// endpoint — `fak serve`, which may front SGLang OR serve the pure in-kernel engine
// via `--gguf --engine inkernel`). The gateway ADJUDICATES the model's proposed tool
// calls; this runner EXECUTES the allowed ones on a per-instance git worktree and
// captures the resulting `git diff` as the prediction patch the official SWE-bench
// harness grades.
//
// What is witnessed here (fleet_test.go, no GPU / model / network): the loop mechanics
// — tool dispatch, file edits applied to a real git repo, the unified-diff capture in
// the exact harness shape, the step cap, and the honest "no gateway" error. What is
// NOT witnessed here and remains the GPU/DGX residual: whether a real model actually
// RESOLVES instances (resolve-rate). Drive that with `fak swebench run --agent fleet`
// against a live `fak serve` on the DGX, then grade with the Docker harness.

const maxToolOutput = 8192 // cap a single tool result fed back to the model

const codingSystemPrompt = "You are an autonomous software engineer. You are given a bug report for a " +
	"repository checked out at a specific commit in your working directory. Use the tools to read files and " +
	"make edits. Make the smallest change that fixes the issue. Do not ask questions; act. When the fix is " +
	"complete, call the finish tool."

// fleetRunner executes instances via the fak gateway coding agent.
//
// planner and prepare are injectable so the loop is testable without a model,
// a GPU, or network: a test supplies a scripted in-process planner and a
// fixture worktree preparer. In production both are nil and are built from cfg
// (the gateway HTTP planner + a real `git clone`/checkout).
type fleetRunner struct {
	cfg      RunConfig
	planner  agent.Planner                                            // nil => agent.NewHTTPPlanner from cfg.GatewayAddr
	prepare  func(ctx context.Context, in Instance, dir string) error // nil => gitCheckoutWorkspace
	workRoot string                                                   // parent for the per-instance temp dir ("" => os.TempDir)
}

func (f *fleetRunner) RunInstance(ctx context.Context, in Instance) (Prediction, error) {
	planner := f.planner
	if planner == nil {
		base := gatewayBaseURL(f.cfg.GatewayAddr)
		if base == "" {
			return Prediction{}, fmt.Errorf("fleet runner: no gateway configured (pass --gateway ADDR pointing at a running `fak serve`)")
		}
		planner = agent.NewHTTPPlanner(base, f.cfg.Model, os.Getenv("FAK_API_KEY"))
	}
	prepare := f.prepare
	if prepare == nil {
		prepare = gitCheckoutWorkspace
	}
	root := f.workRoot
	if root == "" {
		root = os.TempDir()
	}

	dir, err := os.MkdirTemp(root, "fak-swe-"+sanitizeID(in.InstanceID)+"-")
	if err != nil {
		return Prediction{}, fmt.Errorf("workspace: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := prepare(ctx, in, dir); err != nil {
		return Prediction{}, fmt.Errorf("prepare workspace: %w", err)
	}
	if _, err := runGit(ctx, dir, "rev-parse", "--git-dir"); err != nil {
		return Prediction{}, fmt.Errorf("workspace %s is not a git repo after prepare: %w", dir, err)
	}

	if _, err := runCodingAgent(ctx, planner, in, dir, f.cfg.MaxSteps, f.cfg.AllowExec); err != nil {
		return Prediction{}, err
	}

	patch, err := capturePatch(ctx, dir)
	if err != nil {
		return Prediction{}, fmt.Errorf("capture patch: %w", err)
	}

	model := planner.Model()
	if model == "" {
		model = "fleet"
	}
	return Prediction{InstanceID: in.InstanceID, ModelNameOrPath: model, ModelPatch: patch}, nil
}

// runCodingAgent drives the read/edit loop for one instance. It returns the
// number of model turns taken. Hitting the step cap is NOT an error — the
// partial worktree diff is still a (possibly empty) prediction; only a planner
// failure or context cancellation propagates. Exposed within the package so the
// test can assert turn accounting directly.
func runCodingAgent(ctx context.Context, p agent.Planner, in Instance, dir string, maxSteps int, allowExec bool) (int, error) {
	if maxSteps <= 0 {
		maxSteps = 50
	}
	tools := codingTools(allowExec)
	messages := []agent.Message{
		{Role: "system", Content: codingSystemPrompt},
		{Role: "user", Content: codingTaskPrompt(in, allowExec)},
	}

	for step := 0; step < maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return step, err
		}
		comp, err := p.Complete(ctx, messages, tools)
		if err != nil {
			return step, fmt.Errorf("planner turn %d: %w", step+1, err)
		}
		msg := comp.Message
		messages = append(messages, msg)

		if len(msg.ToolCalls) == 0 {
			// No tool calls: the model produced a final answer. Done.
			return step + 1, nil
		}

		finished := false
		for _, tc := range msg.ToolCalls {
			result, fin := execTool(ctx, dir, tc, allowExec)
			messages = append(messages, agent.Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    result,
			})
			if fin {
				finished = true
			}
		}
		if finished {
			return step + 1, nil
		}
	}
	return maxSteps, nil // hit the step cap; the caller still captures the partial diff
}

// execTool runs one tool call against the worktree and returns the result text
// fed back to the model plus whether it was the finish signal. Tool-level
// failures (bad path, missing file) are returned AS results — the model sees the
// error and adapts; they are not run-fatal.
func execTool(ctx context.Context, dir string, tc agent.ToolCall, allowExec bool) (result string, finished bool) {
	switch tc.Function.Name {
	case "finish":
		return "ok: finishing", true

	case "read_file":
		var a struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal([]byte(orEmptyObj(tc.Function.Arguments)), &a); err != nil {
			return "error: invalid arguments: " + err.Error(), false
		}
		full, err := safeJoin(dir, a.Path)
		if err != nil {
			return "error: " + err.Error(), false
		}
		b, err := os.ReadFile(full)
		if err != nil {
			return "error: " + err.Error(), false
		}
		return truncateOutput(string(b)), false

	case "write_file":
		var a struct {
			Path    string `json:"path"`
			Content string `json:"content"`
		}
		if err := json.Unmarshal([]byte(orEmptyObj(tc.Function.Arguments)), &a); err != nil {
			return "error: invalid arguments: " + err.Error(), false
		}
		full, err := safeJoin(dir, a.Path)
		if err != nil {
			return "error: " + err.Error(), false
		}
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return "error: " + err.Error(), false
		}
		if err := os.WriteFile(full, []byte(a.Content), 0o644); err != nil {
			return "error: " + err.Error(), false
		}
		return fmt.Sprintf("ok: wrote %d bytes to %s", len(a.Content), a.Path), false

	case "run":
		if !allowExec {
			return "error: the run tool is disabled; enable with --allow-exec (use only in a sandboxed/containerized run)", false
		}
		var a struct {
			Cmd string `json:"cmd"`
		}
		if err := json.Unmarshal([]byte(orEmptyObj(tc.Function.Arguments)), &a); err != nil {
			return "error: invalid arguments: " + err.Error(), false
		}
		if strings.TrimSpace(a.Cmd) == "" {
			return "error: empty cmd", false
		}
		out := runShell(ctx, dir, a.Cmd)
		return truncateOutput(out), false

	default:
		return "error: unknown tool " + tc.Function.Name, false
	}
}

// capturePatch stages every change in the worktree and returns the unified diff
// against the base commit in the exact shape the SWE-bench harness applies. The
// `add -A` makes new files appear in the diff; `diff --cached` then renders the
// full index-vs-HEAD delta. An empty string means the agent changed nothing.
func capturePatch(ctx context.Context, dir string) (string, error) {
	if out, err := runGit(ctx, dir, "add", "-A"); err != nil {
		return "", fmt.Errorf("git add: %w (%s)", err, strings.TrimSpace(out))
	}
	out, err := runGit(ctx, dir, "diff", "--cached")
	if err != nil {
		return "", fmt.Errorf("git diff: %w (%s)", err, strings.TrimSpace(out))
	}
	return out, nil
}

// gitCheckoutWorkspace is the production worktree preparer: clone the instance's
// GitHub repo and check out its base commit. Requires network + git (the DGX
// path). Tests inject a fixture preparer instead.
func gitCheckoutWorkspace(ctx context.Context, in Instance, dir string) error {
	repo := in.RepoFull()
	if repo == "" {
		return fmt.Errorf("instance %s: cannot determine repo url", in.InstanceID)
	}
	url := "https://github.com/" + repo + ".git"
	if out, err := runGit(ctx, "", "clone", "--quiet", url, dir); err != nil {
		return fmt.Errorf("clone %s: %w (%s)", url, err, strings.TrimSpace(out))
	}
	if in.BaseCommit != "" {
		if out, err := runGit(ctx, dir, "checkout", "--quiet", in.BaseCommit); err != nil {
			return fmt.Errorf("checkout %s: %w (%s)", in.BaseCommit, err, strings.TrimSpace(out))
		}
	}
	return nil
}

// ---- tool catalog --------------------------------------------------------------

func codingTools(allowExec bool) []agent.ToolDef {
	defs := []agent.ToolDef{
		toolDef("read_file", "Read a UTF-8 text file from the repository.",
			`{"type":"object","properties":{"path":{"type":"string","description":"path relative to the repository root"}},"required":["path"]}`),
		toolDef("write_file", "Create or overwrite a file with the given full content.",
			`{"type":"object","properties":{"path":{"type":"string","description":"path relative to the repository root"},"content":{"type":"string","description":"the complete new file content"}},"required":["path","content"]}`),
		toolDef("finish", "Signal that the fix is complete.",
			`{"type":"object","properties":{}}`),
	}
	if allowExec {
		defs = append(defs, toolDef("run", "Run a shell command in the repository root (e.g. to reproduce the bug or run tests).",
			`{"type":"object","properties":{"cmd":{"type":"string","description":"the shell command"}},"required":["cmd"]}`))
	}
	return defs
}

func toolDef(name, desc, schema string) agent.ToolDef {
	return agent.ToolDef{
		Type:     "function",
		Function: agent.ToolDefFunction{Name: name, Description: desc, Parameters: json.RawMessage(schema)},
	}
}

func codingTaskPrompt(in Instance, allowExec bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Repository: %s\n", in.RepoFull())
	if in.BaseCommit != "" {
		fmt.Fprintf(&b, "Base commit: %s\n", in.BaseCommit)
	}
	b.WriteString("The repository is checked out at the base commit in your working directory.\n\n")
	b.WriteString("Problem statement:\n")
	b.WriteString(strings.TrimSpace(in.ProblemStatement))
	b.WriteString("\n\n")
	if h := strings.TrimSpace(in.Hints); h != "" {
		b.WriteString("Maintainer hints:\n")
		b.WriteString(h)
		b.WriteString("\n\n")
	}
	b.WriteString("Edit the repository files to fix the issue. ")
	if allowExec {
		b.WriteString("You may use the run tool to reproduce and verify. ")
	}
	b.WriteString("Make the smallest change that resolves it, then call finish.")
	return b.String()
}

// ---- small helpers -------------------------------------------------------------

// gatewayBaseURL normalizes a gateway address into an OpenAI-compatible base URL.
// "localhost:8080" -> "http://localhost:8080/v1"; a full URL is preserved (and a
// missing /v1 segment is appended). The OpenAI adapter then posts to
// <base>/chat/completions.
func gatewayBaseURL(addr string) string {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return ""
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	addr = strings.TrimRight(addr, "/")
	if !strings.Contains(addr, "/v1") {
		addr += "/v1"
	}
	return addr
}

// safeJoin resolves a model-supplied relative path under root, refusing absolute
// paths and any path that escapes the worktree (a poisoned `../../etc/passwd`).
func safeJoin(root, p string) (string, error) {
	if strings.TrimSpace(p) == "" {
		return "", fmt.Errorf("empty path")
	}
	clean := filepath.Clean(p)
	if filepath.IsAbs(clean) {
		return "", fmt.Errorf("absolute path not allowed: %s", p)
	}
	full := filepath.Join(root, clean)
	rel, err := filepath.Rel(root, full)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes workspace: %s", p)
	}
	return full, nil
}

func runGit(ctx context.Context, dir string, args ...string) (string, error) {
	c := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		c.Dir = dir
	}
	out, err := c.CombinedOutput()
	return string(out), err
}

func runShell(ctx context.Context, dir, cmd string) string {
	c := exec.CommandContext(ctx, "bash", "-c", cmd)
	c.Dir = dir
	out, err := c.CombinedOutput()
	s := string(out)
	if err != nil {
		s += "\n[exit: " + err.Error() + "]"
	}
	return s
}

func orEmptyObj(s string) string {
	if strings.TrimSpace(s) == "" {
		return "{}"
	}
	return s
}

func truncateOutput(s string) string {
	if len(s) <= maxToolOutput {
		return s
	}
	return s[:maxToolOutput] + "\n... [truncated]"
}

// sanitizeID keeps a temp-dir name filesystem-safe.
func sanitizeID(id string) string {
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	out := b.String()
	if len(out) > 60 {
		out = out[:60]
	}
	return out
}
