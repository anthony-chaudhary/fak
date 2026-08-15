package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// microCollapseReport is the captured dogfood receipt for one governed child
// pipeline. It reports the context counterfactual directly instead of calling a
// collapse "free": the parent pays folded_tokens, while saved_tokens is the
// bounded child chatter it did not admit.
type microCollapseReport struct {
	Schema             string `json:"schema"`
	Verdict            string `json:"verdict"`
	Calls              int    `json:"calls"`
	Allowed            int    `json:"allowed"`
	Denied             int    `json:"denied"`
	Errored            int    `json:"errored"`
	IntermediateTokens int    `json:"intermediate_tokens"`
	FoldedTokens       int    `json:"folded_tokens"`
	SavedTokens        int    `json:"saved_tokens"`
	JournalRows        int    `json:"journal_rows"`
	Collapsed          string `json:"collapsed"`
}

type collapseBackend struct{ payload string }

func (collapseBackend) Name() string { return microagent.BackendGoroutine }
func (b collapseBackend) Dispatch(_ context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
	return microagent.ToolResult{Stdout: []byte(act.Tool + ": " + b.payload)}, nil
}

func cmdMicroCollapse(args []string) {
	if len(args) > 0 && args[0] == "cache-witness" {
		os.Exit(runMicroCacheWitness(os.Stdout, os.Stderr, args[1:]))
	}
	if len(args) > 0 && args[0] == "readiness" {
		os.Exit(runMicroDogfoodReadiness(os.Stdout, os.Stderr, args[1:]))
	}
	if len(args) > 0 && args[0] == "repo-pulse" {
		cmdMicroCollapseRepoPulse(args[1:])
		return
	}
	fs := flag.NewFlagSet("micro collapse", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	calls := fs.Int("calls", 3, "number of governed child tool calls")
	payloadBytes := fs.Int("payload-bytes", 2048, "deterministic intermediate result bytes per call")
	jsonOut := fs.Bool("json", false, "emit the measured collapse receipt as JSON")
	if err := fs.Parse(args); err != nil {
		return
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "fak micro collapse: unexpected positional arguments")
		return
	}
	r, err := runMicroCollapse(*calls, *payloadBytes)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak micro collapse:", err)
		return
	}
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(r)
		return
	}
	fmt.Fprintf(os.Stdout, "micro collapse %s: calls=%d allowed=%d denied=%d intermediate_tokens=%d folded_tokens=%d saved_tokens=%d journal_rows=%d\n", r.Verdict, r.Calls, r.Allowed, r.Denied, r.IntermediateTokens, r.FoldedTokens, r.SavedTokens, r.JournalRows)
	fmt.Fprintln(os.Stdout, r.Collapsed)
}

func runMicroCollapse(calls, payloadBytes int) (microCollapseReport, error) {
	if calls < 1 || payloadBytes < 1 {
		return microCollapseReport{}, fmt.Errorf("--calls and --payload-bytes must be positive")
	}
	const tool = "dogfood_read"
	floor := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{adjudicator.New(adjudicator.Policy{Allow: map[string]bool{tool: true}})}))
	exec, err := microagent.NewToolExecBackend(floor, collapseBackend{payload: strings.Repeat("x", payloadBytes)})
	if err != nil {
		return microCollapseReport{}, err
	}
	jrnl := journal.OpenMemory()
	sub, err := microagent.NewRPCSubagent("dogfood-collapse", exec, calls*(payloadBytes/4+64), jrnl)
	if err != nil {
		return microCollapseReport{}, err
	}
	script := make([]microagent.ToolAction, calls)
	for i := range script {
		script[i] = microagent.ToolAction{Tool: tool, Args: map[string]any{"call": i + 1}}
	}
	got := sub.RunScript(context.Background(), script)
	rows := jrnl.Recent(calls)
	verdict := "PASS"
	if got.Allowed != calls || got.Denied != 0 || got.Errored != 0 || got.SavedTokens <= 0 || len(rows) != calls {
		verdict = "FAIL"
	}
	return microCollapseReport{Schema: "fak-micro-collapse/1", Verdict: verdict, Calls: calls, Allowed: got.Allowed, Denied: got.Denied, Errored: got.Errored, IntermediateTokens: got.IntermediateTokens, FoldedTokens: got.FoldedTokens, SavedTokens: got.SavedTokens, JournalRows: len(rows), Collapsed: got.Collapsed}, nil
}

type repoPulseReport struct {
	Schema                   string `json:"schema"`
	Verdict                  string `json:"verdict"`
	Calls                    int    `json:"calls"`
	Allowed                  int    `json:"allowed"`
	Denied                   int    `json:"denied"`
	Errored                  int    `json:"errored"`
	InlineTokens             int    `json:"inline_tokens"`
	FoldedTokens             int    `json:"folded_tokens"`
	SavedTokens              int    `json:"saved_tokens"`
	ParentToolTurnsInline    int    `json:"parent_tool_turns_inline"`
	ParentToolTurnsCollapsed int    `json:"parent_tool_turns_collapsed"`
	ToolTurnsSkipped         int    `json:"tool_turns_skipped"`
	JournalRows              int    `json:"journal_rows"`
	Collapsed                string `json:"collapsed"`
}

type repoPulseBackend struct{ dir string }

func (repoPulseBackend) Name() string { return microagent.BackendGoroutine }
func (b repoPulseBackend) Dispatch(ctx context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
	var argv []string
	switch act.Tool {
	case "repo_status":
		argv = []string{"status", "--short", "--branch"}
	case "repo_head":
		argv = []string{"log", "-1", "--oneline", "--decorate"}
	case "repo_diffstat":
		argv = []string{"diff", "--stat"}
	default:
		return microagent.ToolResult{}, fmt.Errorf("unknown repo-pulse tool %q", act.Tool)
	}
	cmd := windowgate.CommandContext(ctx, "git", argv...)
	cmd.Dir = b.dir
	out, err := cmd.Output()
	res := microagent.ToolResult{Ran: true, Stdout: boundedPulseOutput(out), ExitCode: 0}
	if err != nil {
		res.ExitCode = 1
		return res, err
	}
	return res, nil
}

const repoPulseOutputLimit = 32 << 10

func boundedPulseOutput(in []byte) []byte {
	if len(in) <= repoPulseOutputLimit {
		return in
	}
	out := append([]byte(nil), in[:repoPulseOutputLimit]...)
	return append(out, []byte("\n... output truncated by fak repo-pulse\n")...)
}

func cmdMicroCollapseRepoPulse(args []string) {
	fs := flag.NewFlagSet("micro collapse repo-pulse", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	dir := fs.String("dir", ".", "repository directory")
	jsonOut := fs.Bool("json", false, "emit JSON receipt")
	if err := fs.Parse(args); err != nil {
		return
	}
	r, err := runRepoPulse(pathutil.ExpandTilde(*dir))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak micro collapse repo-pulse:", err)
		return
	}
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(r)
		return
	}
	fmt.Fprintf(os.Stdout, "repo pulse %s: inline_tokens=%d folded_tokens=%d saved_tokens=%d turns_skipped=%d journal_rows=%d\n%s\n", r.Verdict, r.InlineTokens, r.FoldedTokens, r.SavedTokens, r.ToolTurnsSkipped, r.JournalRows, r.Collapsed)
}

func runRepoPulse(dir string) (repoPulseReport, error) {
	tools := []string{"repo_status", "repo_head", "repo_diffstat"}
	allow := make(map[string]bool, len(tools))
	for _, tool := range tools {
		allow[tool] = true
	}
	floor := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{adjudicator.New(adjudicator.Policy{Allow: allow})}))
	execSeam, err := microagent.NewToolExecBackend(floor, repoPulseBackend{dir: dir})
	if err != nil {
		return repoPulseReport{}, err
	}
	jrnl := journal.OpenMemory()
	sub, err := microagent.NewRPCSubagent("repo-pulse", execSeam, 32768, jrnl)
	if err != nil {
		return repoPulseReport{}, err
	}
	sub.WithSummarizer(summarizeRepoPulse)
	script := make([]microagent.ToolAction, len(tools))
	for i, tool := range tools {
		script[i] = microagent.ToolAction{Tool: tool}
	}
	runCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	got := sub.RunScript(runCtx, script)
	inline := microagent.NewContext(32768)
	for _, step := range got.Steps {
		inline.Append("assistant", "call "+step.Tool)
		inline.Append("tool", string(step.Stdout))
	}
	inlineTokens := inline.Tokens()
	rows := jrnl.Recent(len(tools))
	verdict := "PASS"
	if got.Allowed != len(tools) || got.Denied != 0 || got.Errored != 0 || len(rows) != len(tools) || inlineTokens <= got.FoldedTokens {
		verdict = "FAIL"
	}
	return repoPulseReport{Schema: "fak-micro-collapse-repo-pulse/1", Verdict: verdict, Calls: len(tools), Allowed: got.Allowed, Denied: got.Denied, Errored: got.Errored, InlineTokens: inlineTokens, FoldedTokens: got.FoldedTokens, SavedTokens: max(inlineTokens-got.FoldedTokens, 0), ParentToolTurnsInline: len(tools), ParentToolTurnsCollapsed: 1, ToolTurnsSkipped: len(tools) - 1, JournalRows: len(rows), Collapsed: got.Collapsed}, nil
}

func summarizeRepoPulse(_ string, steps []microagent.RPCStep) string {
	value := func(i int) string {
		if i >= len(steps) || steps[i].Err != nil {
			return "unavailable"
		}
		s := strings.TrimSpace(string(steps[i].Stdout))
		if s == "" {
			return "clean"
		}
		lines := strings.Split(s, "\n")
		if len(lines) > 4 {
			lines = append(lines[:4], fmt.Sprintf("... +%d lines", len(lines)-4))
		}
		return strings.Join(lines, " | ")
	}
	return fmt.Sprintf("repo pulse - status: %s; head: %s; diff: %s", value(0), value(1), value(2))
}
