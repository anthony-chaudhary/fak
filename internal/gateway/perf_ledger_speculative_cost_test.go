package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

type perfSpeculativeCostPlanner struct {
	completion *agent.Completion
	err        error
	buffered   atomic.Int32
	streamed   atomic.Int32
}

func (*perfSpeculativeCostPlanner) Model() string { return "cost-model" }

func (p *perfSpeculativeCostPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.buffered.Add(1)
	return p.completion, p.err
}

func (*perfSpeculativeCostPlanner) StreamingSupported() bool { return true }

func (p *perfSpeculativeCostPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.streamed.Add(1)
	if err := sink("plain partial response content that cannot be a tool call\n"); err != nil {
		return p.completion, err
	}
	return p.completion, p.err
}

// fak-test:runtime fast est=1s lane=default
func TestPerfLedgerSpeculativeCostServedAndFailedRoutes(t *testing.T) {
	completed := &agent.SpeculativeDecodeCost{
		Scope: "proposal-through-emission-and-advance", Attempts: 1,
		Completed: agent.SpeculativeAttemptCost{
			Attempts: 1, EmittedTokens: 4, MeasuredAttempts: 1,
			DraftElapsed: 3 * time.Millisecond, InclusiveElapsed: 13 * time.Millisecond,
		},
	}
	fallback := &agent.SpeculativeDecodeCost{
		Scope: completed.Scope, Attempts: 2,
		ProposalError: agent.SpeculativeAttemptCost{
			Attempts: 1, MeasuredAttempts: 1,
			DraftElapsed: 2 * time.Millisecond, InclusiveElapsed: 3 * time.Millisecond,
		},
		EmptyProposal: agent.SpeculativeAttemptCost{
			Attempts: 1, MeasuredAttempts: 1,
			DraftElapsed: time.Millisecond, InclusiveElapsed: time.Millisecond,
		},
	}
	verificationError, incomplete, cancelled := *completed, *completed, *completed
	verificationError.Attempts, incomplete.Attempts, cancelled.Attempts = 2, 2, 2
	verificationError.VerificationError = agent.SpeculativeAttemptCost{
		Attempts: 1, MeasuredAttempts: 1,
		DraftElapsed: 4 * time.Millisecond, InclusiveElapsed: 8 * time.Millisecond,
	}
	incomplete.Incomplete = agent.SpeculativeAttemptCost{
		Attempts: 1, EmittedTokens: 1, MeasuredAttempts: 1,
		DraftElapsed: 2 * time.Millisecond, InclusiveElapsed: 5 * time.Millisecond,
	}
	cancelled.Cancelled = agent.SpeculativeAttemptCost{Attempts: 1, UnmeasuredAttempts: 1}
	failure := errors.New("synthetic completion failure")
	for _, tc := range []struct {
		name, path, decodePath string
		stream                 bool
		cost                   *agent.SpeculativeDecodeCost
		err                    error
		wantStatus             int
		wantError              string
	}{
		{"completed", "/v1/chat/completions", agent.NativeDecodePathSpeculative, false, completed, nil, http.StatusOK, ""},
		{"serial-fallback", "/v1/chat/completions", agent.NativeDecodePathSerial, false, fallback, nil, http.StatusOK, ""},
		{"unobserved-cost", "/v1/chat/completions", agent.NativeDecodePathSpeculative, false, nil, nil, http.StatusOK, ""},
		{"buffered-failure", "/v1/chat/completions", agent.NativeDecodePathSpeculative, false, &verificationError, failure, http.StatusBadGateway, perfledger.ErrorUpstream},
		{"chat-stream-failure", "/v1/chat/completions", agent.NativeDecodePathSpeculative, true, &incomplete, failure, http.StatusOK, perfledger.ErrorUpstream},
		{"responses-stream-failure", "/v1/responses", agent.NativeDecodePathSpeculative, true, &cancelled, context.Canceled, http.StatusOK, perfledger.ErrorClientCanceled},
		{"messages-stream-failure", "/v1/messages", agent.NativeDecodePathSpeculative, true, &cancelled, context.Canceled, http.StatusOK, perfledger.ErrorClientCanceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var cost *agent.SpeculativeDecodeCost
			if tc.cost != nil {
				copy := *tc.cost
				cost = &copy
			}
			decode := &agent.NativeDecodeSummary{Path: tc.decodePath, SpeculativeCost: cost}
			if tc.decodePath == agent.NativeDecodePathSpeculative {
				decode.Speculative = &agent.SpeculativeDecodeTally{Rounds: 1, DraftTokens: 4, AcceptedTokens: 3}
			}
			comp := &agent.Completion{
				Model: "cost-model", Message: agent.Message{Role: agent.RoleAssistant, Content: "done"},
				FinishReason: "stop", Usage: agent.Usage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
				NativeDecode: decode,
			}
			if tc.err != nil {
				comp.Timings = agent.NewTimings(8, 0, 4, .001, .003)
			}
			planner := &perfSpeculativeCostPlanner{completion: comp, err: tc.err}
			h := newPerfServedHarness(t, planner)
			t.Cleanup(func() { _ = h.w.Close() })
			body := map[string]any{"model": "cost-model", "stream": tc.stream}
			if tc.path == "/v1/responses" {
				body["input"], body["max_output_tokens"] = "hello", 4
			} else {
				body["messages"] = []map[string]string{{"role": "user", "content": "hello"}}
				body["max_tokens"] = 4
			}
			raw, err := json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := http.Post(h.ts.URL+tc.path, "application/json", bytes.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			_, readErr := io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if readErr != nil {
				t.Fatal(readErr)
			}
			if resp.StatusCode != tc.wantStatus {
				t.Fatalf("wire status = %d, want %d", resp.StatusCode, tc.wantStatus)
			}
			if tc.stream {
				if planner.streamed.Load() != 1 || planner.buffered.Load() != 0 {
					t.Fatal("request did not reach the direct streaming planner seam once")
				}
			} else if planner.buffered.Load() != 1 || planner.streamed.Load() != 0 {
				t.Fatal("request did not reach the buffered planner seam once")
			}
			live := h.recent()
			if len(live.Records) != 1 {
				t.Fatalf("live rows = %d, want one request-owned row", len(live.Records))
			}
			assertPerfSpeculativeCostRow(t, live.Records[0], tc.decodePath, tc.cost, tc.wantError, tc.wantStatus)
			if cost != nil {
				cost.Completed.InclusiveElapsed = time.Hour
				cost.Cancelled.MeasuredAttempts = 99
			}
			// The writer and bounded reader must retain the request snapshot, not
			// a pointer into the planner's mutable Completion.
			durable := h.durable()
			if len(durable) != 1 {
				t.Fatalf("durable rows = %d, want one", len(durable))
			}
			offline := perfledger.BuildReport(durable, 1, false, 0)
			if len(offline.Records) != 1 {
				t.Fatalf("bounded offline rows = %d, want one", len(offline.Records))
			}
			assertPerfSpeculativeCostRow(t, offline.Records[0], tc.decodePath, tc.cost, tc.wantError, tc.wantStatus)
		})
	}
}

func assertPerfSpeculativeCostRow(t *testing.T, row perfledger.Record, path string, want *agent.SpeculativeDecodeCost, failure string, status int) {
	t.Helper()
	if row.Engine == nil || row.Engine.Path != path || row.Model != "cost-model" {
		t.Fatal("ledger lost the completion's native engine/model provenance")
	}
	if path == agent.NativeDecodePathSpeculative && (row.Engine.SpecRounds != 1 || row.Engine.SpecDraftTokens != 4 || row.Engine.SpecAcceptedTokens != 3) {
		t.Fatal("cost projection changed existing speculative verification counts")
	}
	if path == agent.NativeDecodePathSerial && (row.Engine.SpecRounds != 0 || row.Engine.SpecDraftTokens != 0 || row.Engine.SpecAcceptedTokens != 0) {
		t.Fatal("serial fallback acquired phantom verification counts")
	}
	if failure != "" {
		if row.Error != failure || row.Status != status || row.FinishReason != perfledger.FinishReasonError {
			t.Fatalf("failed row class/status/finish = %q/%d/%q", row.Error, row.Status, row.FinishReason)
		}
		if row.BenchEvidence != nil || row.NativeTiming != nil || row.DecodeTPS != 0 || row.PrefillTPS != 0 {
			t.Fatal("failed completion cost was promoted to successful benchmark/timing evidence")
		}
	} else if row.Error != "" {
		t.Fatalf("served row acquired failure class %q", row.Error)
	}
	if want == nil {
		if row.Engine.SpeculativeCost != nil {
			t.Fatal("unobserved attempt cost became measured zero")
		}
		return
	}
	if row.Engine.SpeculativeCost == nil {
		t.Fatal("ledger dropped request-owned attempt cost")
	}
	expected, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := json.Marshal(row.Engine.SpeculativeCost)
	if err != nil {
		t.Fatal(err)
	}
	var expectedFields, actualFields any
	if err := json.Unmarshal(expected, &expectedFields); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(actual, &actualFields); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(actualFields, expectedFields) {
		t.Fatal("request cost/outcome/duration measurement fields changed across the ledger")
	}
}
