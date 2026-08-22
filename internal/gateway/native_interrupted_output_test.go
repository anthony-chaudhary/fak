package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

type prefixThenCancelPlanner struct {
	chunks []string
}

func (p *prefixThenCancelPlanner) Model() string { return "prefix-then-cancel" }

func (p *prefixThenCancelPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("buffered completion unexpectedly called")
}

func (p *prefixThenCancelPlanner) StreamingSupported() bool { return true }

func (p *prefixThenCancelPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	for _, chunk := range p.chunks {
		if err := sink(chunk); err != nil {
			return nil, err
		}
	}
	return nil, context.Canceled
}

func TestNativeInterruptedStreamPersistsPrefixAndTerminalAfterReopen(t *testing.T) {
	dir := t.TempDir()
	ledger, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("Open ledger: %v", err)
	}
	oldOpen := openNativeModelRequestLedger
	openNativeModelRequestLedger = func() (*sessionledger.Ledger, error) { return ledger, nil }
	t.Cleanup(func() { openNativeModelRequestLedger = oldOpen })

	srv := newTestServer(t)
	srv.native = true
	srv.nativeMaxTurns = 1
	planner := &prefixThenCancelPlanner{chunks: []string{"visible ", "assistant ", "prefix"}}
	srv.planner = planner

	req, err := agent.DecodeAnthropicMessagesRequest([]byte(`{
		"model":"prefix-then-cancel",
		"max_tokens":64,
		"stream":true,
		"messages":[{"role":"user","content":"continue the long answer"}]
	}`))
	if err != nil {
		t.Fatalf("DecodeAnthropicMessagesRequest: %v", err)
	}
	var visible strings.Builder
	metrics, err := srv.runNativeArmStream(context.Background(), req, "interrupted-prefix", func(chunk string) error {
		visible.WriteString(chunk)
		return nil
	}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("runNativeArmStream error = %v, want context canceled", err)
	}
	if got := visible.String(); got != "visible assistant prefix" {
		t.Fatalf("visible stream = %q, want exact emitted prefix", got)
	}
	if metrics.ToolCalls != 0 || metrics.EngineCalls != 0 {
		t.Fatalf("partial stream dispatched tools: tool_calls=%d engine_calls=%d", metrics.ToolCalls, metrics.EngineCalls)
	}

	reopened, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("reopen ledger: %v", err)
	}
	chain, err := reopened.Chain("interrupted-prefix")
	if err != nil {
		t.Fatalf("read reopened chain: %v", err)
	}
	var requestCount, interruptedCount, terminalCount int
	for _, entry := range chain {
		switch entry.Kind {
		case sessionledger.KindModelRequestReceipt:
			requestCount++
		case sessionledger.KindInterruptedAssistant:
			interruptedCount++
		case sessionledger.KindTurnTerminal:
			terminalCount++
			var terminal sessionledger.TurnTerminal
			if err := json.Unmarshal(entry.Content, &terminal); err != nil {
				t.Fatalf("decode turn terminal: %v", err)
			}
			if terminal.Turn != 1 || terminal.Status != sessionledger.TurnStatusInterrupted || terminal.Reason.Cause != agent.TerminationCanceled {
				t.Fatalf("turn terminal = %+v, want interrupted cancellation on turn 1", terminal)
			}
		}
	}
	if interruptedCount != 1 {
		t.Fatalf("interrupted assistant records = %d, want exactly 1; kinds=%v", interruptedCount, ledgerKinds(chain))
	}
	if requestCount != 1 || terminalCount != requestCount {
		t.Fatalf("model requests=%d turn terminals=%d, want balanced 1:1 terminal record; kinds=%v", requestCount, terminalCount, ledgerKinds(chain))
	}
	record, receipt, err := reopened.ReconstructInterruptedAssistant("interrupted-prefix", 1)
	if err != nil {
		t.Fatalf("reconstruct interrupted assistant: %v", err)
	}
	if record.Role != agent.RoleAssistant || record.Turn != 1 || strings.Join(record.Chunks, "|") != "visible |assistant |prefix" {
		t.Fatalf("interrupted assistant = %+v, want exact three-chunk assistant prefix on turn 1", record)
	}
	if record.Reason.Cause != agent.TerminationCanceled || record.Reason.Evidence != "request context ended" {
		t.Fatalf("interrupted reason = %+v, want typed cancellation", record.Reason)
	}
	if receipt.ChunkCount != len(planner.chunks) || receipt.ContentBytes != len(visible.String()) {
		t.Fatalf("interrupted receipt = %+v, want exact chunk and byte counts", receipt)
	}
}

func ledgerKinds(entries []sessionledger.Entry) []string {
	kinds := make([]string, 0, len(entries))
	for _, entry := range entries {
		kinds = append(kinds, entry.Kind)
	}
	return kinds
}
