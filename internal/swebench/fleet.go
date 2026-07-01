package swebench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/codelint"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// fleet.go is the fak-native SWE-bench solver: the "fleet" runner drives a real
// coding-agent loop against an injected CodePlanner (in production an OpenAI-compatible
// /v1/chat/completions client wired by cmd/fak — `fak serve`, which may front SGLang
// OR serve the pure in-kernel engine via `--gguf --engine inkernel`). The gateway
// ADJUDICATES the model's proposed tool calls; this runner EXECUTES the allowed ones
// on a per-instance git worktree and captures the resulting `git diff` as the
// prediction patch the official SWE-bench harness grades.
//
// LAYERING: swebench is a foundation tier, so it owns NO chat-client dependency —
// the planner is an interface (CodePlanner) the integrator (cmd/fak, which owns the
// single outbound HTTPPlanner) injects via RunConfig.Planner. This keeps the
// layered-DAG (internal/architest) and the "agent is the one chat client" rule.
//
// What is witnessed here (fleet_test.go, no GPU / model / network): the loop mechanics
// — tool dispatch, file edits applied to a real git repo, the unified-diff capture in
// the exact harness shape, the step cap, and the honest "no planner" error. What is
// NOT witnessed here and remains the GPU/DGX residual: whether a real model actually
// RESOLVES instances (resolve-rate). Drive that with `fak swebench run --agent fleet`
// against a live `fak serve` on the GPU server, then grade with the Docker harness.

const maxToolOutput = 8192 // cap a single tool result fed back to the model

const codingSystemPrompt = "You are an autonomous software engineer. You are given a bug report for a " +
	"repository checked out at a specific commit in your working directory. Use the tools to read files and " +
	"make edits. Make the smallest change that fixes the issue. Do not ask questions; act. When the fix is " +
	"complete, call the finish tool."

// CodePlanner is the minimal model interface the fleet coding loop drives. It is
// declared in swebench (foundation) so the harness depends on no chat client; an
// integrator (cmd/fak) injects an HTTPPlanner-backed implementation via
// RunConfig.Planner. The types below are the OpenAI-shaped subset the loop needs,
// kept swebench-local for the same layering reason.
type CodePlanner interface {
	// Complete sends the running message list + tool catalog and returns the
	// assistant's next message (tool calls or a final answer).
	Complete(ctx context.Context, messages []ChatMessage, tools []ChatTool) (ChatTurn, error)
	// Model reports the model id (for provenance).
	Model() string
}

// ChatMessage is one chat-completions message (request or response).
type ChatMessage struct {
	Role       string         // system | user | assistant | tool
	Content    string         // text content (final answer or tool result)
	ToolCalls  []ChatToolCall // assistant tool calls
	ToolCallID string         // for role=tool: the call this result answers
	Name       string         // for role=tool: the tool name
}

// ChatToolCall is one function call the model emitted.
type ChatToolCall struct {
	ID   string // call id (echoed back on the tool result)
	Name string // function name
	Args string // raw JSON arguments string as the model emitted them
}

// ChatTool is one tool advertised to the model (an OpenAI function declaration).
type ChatTool struct {
	Name        string
	Description string
	Parameters  string // JSON Schema (raw JSON)
}

// ChatTurn is a planner's response for one turn.
type ChatTurn struct {
	Message ChatMessage
}

// fleetRunner executes instances via the injected coding-agent planner.
//
// prepare and workRoot are injectable so the loop is testable without a model,
// a GPU, or network: a test supplies a scripted CodePlanner (RunConfig.Planner)
// and a fixture worktree preparer. In production prepare is nil (a real `git
// clone`/checkout) and Planner is wired by cmd/fak.
type fleetRunner struct {
	cfg      RunConfig
	prepare  func(ctx context.Context, in Instance, dir string) error // nil => gitCheckoutWorkspace
	workRoot string                                                   // parent for the per-instance temp dir ("" => os.TempDir)
}

// RunInstance solves one SWE-bench instance end to end: it prepares an isolated git worktree,
// drives the coding-agent planner over it, captures the resulting diff, and returns it as a
// Prediction. It errors when no planner is configured or any workspace/agent step fails.
func (f *fleetRunner) RunInstance(ctx context.Context, in Instance) (Prediction, error) {
	planner := f.cfg.Planner
	if planner == nil {
		return Prediction{}, fmt.Errorf("fleet runner: no planner configured (the integrator must set RunConfig.Planner — cmd/fak builds one from --gateway pointing at a running `fak serve`)")
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

	// LintWrites turns on the kernel's language-server packs over every file the
	// agent writes (off by default, so a benchmark run's behavior is unchanged
	// unless the operator opts in). A nil linter disables the check entirely.
	var linter *codelint.Registry
	if f.cfg.LintWrites {
		linter = codelint.DefaultRegistry()
	}
	if _, err := runCodingAgent(ctx, planner, in, dir, f.cfg.MaxSteps, f.cfg.AllowExec, linter); err != nil {
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
func runCodingAgent(ctx context.Context, p CodePlanner, in Instance, dir string, maxSteps int, allowExec bool, linter *codelint.Registry) (int, error) {
	if maxSteps <= 0 {
		maxSteps = 50
	}
	tools := codingTools(allowExec)
	messages := []ChatMessage{
		{Role: "system", Content: codingSystemPrompt},
		{Role: "user", Content: codingTaskPrompt(in, allowExec)},
	}

	for step := 0; step < maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return step, err
		}
		turn, err := p.Complete(ctx, messages, tools)
		if err != nil {
			return step, fmt.Errorf("planner turn %d: %w", step+1, err)
		}
		msg := turn.Message
		messages = append(messages, msg)

		if len(msg.ToolCalls) == 0 {
			// No tool calls: the model produced a final answer. Done.
			return step + 1, nil
		}

		finished := false
		for _, tc := range msg.ToolCalls {
			result, fin := execTool(ctx, dir, tc, allowExec, linter)
			messages = append(messages, ChatMessage{
				Role:       "tool",
				ToolCallID: tc.ID,
				Name:       tc.Name,
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
func execTool(ctx context.Context, dir string, tc ChatToolCall, allowExec bool, linter *codelint.Registry) (result string, finished bool) {
	switch tc.Name {
	case "finish":
		return "ok: finishing", true

	case "read_file":
		var a struct {
			Path string `json:"path"`
		}
		if msg, ok := decodeToolArgs(tc, &a); !ok {
			return msg, false
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
		if msg, ok := decodeToolArgs(tc, &a); !ok {
			return msg, false
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
		msg := fmt.Sprintf("ok: wrote %d bytes to %s", len(a.Content), a.Path)
		if diag := lintWritten(ctx, linter, full, a.Path); diag != "" {
			msg += "\n" + diag
		}
		return msg, false

	case "bash":
		if !allowExec {
			return "error: the bash tool is disabled; enable with --allow-exec (use only in a sandboxed/containerized run)", false
		}
		var a struct {
			Command string `json:"command"`
		}
		if msg, ok := decodeToolArgs(tc, &a); !ok {
			return msg, false
		}
		if strings.TrimSpace(a.Command) == "" {
			return "error: empty command", false
		}
		out := runShell(ctx, dir, a.Command)
		return truncateOutput(out), false

	default:
		return "error: unknown tool " + tc.Name, false
	}
}

// decodeToolArgs unmarshals a tool call's JSON args into dst. On a decode
// failure it returns the model-facing "invalid arguments" result and false so
// the caller can return it directly; on success it returns "", true.
func decodeToolArgs(tc ChatToolCall, dst any) (string, bool) {
	if err := json.Unmarshal([]byte(orEmptyObj(tc.Args)), dst); err != nil {
		return "error: invalid arguments: " + err.Error(), false
	}
	return "", true
}

// lintWritten runs the kernel's language-server packs over a file the agent just
// wrote and returns a compact, model-facing block for any HARD (parse/compile)
// error, so the coding agent sees its own broken code and fixes it on the next
// turn — the same way a tool error is fed back. It is advisory: the write already
// landed, and a clean file, an unlinted language, or an absent checker returns "".
// A nil linter (LintWrites off) is a no-op. Lint is never run-fatal.
func lintWritten(ctx context.Context, linter *codelint.Registry, full, shown string) string {
	if linter == nil {
		return ""
	}
	fs, err := linter.LintFile(ctx, full)
	if err != nil || !codelint.HasError(fs) {
		return ""
	}
	hard := make([]codelint.Finding, 0, len(fs))
	for _, f := range fs {
		if f.Severity == codelint.Error {
			f.File = shown // address the diagnostic to the path the model used
			hard = append(hard, f)
		}
	}
	return "codelint: the file you just wrote has errors — fix them before continuing:\n" + codelint.Summary(hard)
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
// GitHub repo and check out its base commit. Requires network + git (the GPU server
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

func codingTools(allowExec bool) []ChatTool {
	defs := []ChatTool{
		{Name: "read_file", Description: "Read a UTF-8 text file from the repository.",
			Parameters: `{"type":"object","properties":{"path":{"type":"string","description":"path relative to the repository root"}},"required":["path"]}`},
		{Name: "write_file", Description: "Create or overwrite a file with the given full content.",
			Parameters: `{"type":"object","properties":{"path":{"type":"string","description":"path relative to the repository root"},"content":{"type":"string","description":"the complete new file content"}},"required":["path","content"]}`},
		{Name: "finish", Description: "Signal that the fix is complete.",
			Parameters: `{"type":"object","properties":{}}`},
	}
	if allowExec {
		defs = append(defs, ChatTool{Name: "bash", Description: "Run a shell command in the repository root (e.g. to reproduce the bug or run tests).",
			Parameters: `{"type":"object","properties":{"command":{"type":"string","description":"the shell command"}},"required":["command"]}`})
	}
	return defs
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
		b.WriteString("You may use the bash tool to reproduce and verify. ")
	}
	b.WriteString("Make the smallest change that resolves it, then call finish.")
	return b.String()
}

// ---- small helpers -------------------------------------------------------------

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
	windowgate.ConfigureBackgroundCommand(c)
	if dir != "" {
		c.Dir = dir
	}
	out, err := c.CombinedOutput()
	return string(out), err
}

func runShell(ctx context.Context, dir, cmd string) string {
	c := exec.CommandContext(ctx, "bash", "-c", cmd)
	windowgate.ConfigureBackgroundCommand(c)
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
