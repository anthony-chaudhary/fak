package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/microagent"
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
