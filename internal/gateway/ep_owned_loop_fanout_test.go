package gateway

// ep_owned_loop_fanout_test.go — #5532: the EP follower fanout where the served route
// runs fak's OWN multi-turn governed loop (agent.RunArm / agent.RunGovernedArm).
//
// #5528 left owned-loop routes exempt from the request-mirror bridge with a written reason
// (epExemptOwnedLoop) but nothing that CHECKED the reason: the exemption was prose. #5532
// asked for an in-loop rank barrier so the mirror could cover them. Investigating first
// found two things:
//
//  1. that barrier already exists — model.EPDecodeCoordinator / model.RunEPFollower (#4835),
//     announced from Session.Prefill and Session.Step, a unit finer than a turn and so
//     route-agnostic. It has no serve wiring, which is a wiring gap, not a missing design;
//     and
//  2. the premise that agent-sessions is the ONLY owned-loop route was false. Under
//     `fak serve --native`, /v1/messages drives the same owned loop — and /v1/messages is
//     classified COVERED and DID mirror the inbound body. A follower rank is another process
//     running the same binary and config, so it re-ran the whole loop and dispatched the
//     session's tool calls a second time.
//
// So the two tests below pin the two halves of the corrected rule: an owned-loop arm is
// never handed to the request mirror, whichever route it is reached through. Both were
// observed RED before the handler change (the first at "released 0 time(s)", which is what
// refuted the ticket's prescribed shape; the second at "4 forward passes and 2 tool-result
// turns across the ranks, want 2 and 1", which is the real defect).
//
// Modeled in-process with no weights and no device, like the rest of the EP fanout suite:
// what is asserted is the RELEASE and the loop's forward-pass count, not the collective.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// epOwnedLoopPlanner is a deterministic TWO-turn planner for the owned loop: the first
// forward pass of a transcript emits one tool call, and the pass whose transcript
// already carries that tool's result emits the final answer. It is stateless with
// respect to WHICH rank drives it — the decision is read off the transcript — so one
// instance can be shared by a leader and a follower rank without their turn counts
// interfering.
type epOwnedLoopPlanner struct {
	mu sync.Mutex
	// calls is every forward pass this planner served, across every rank.
	calls int
	// withToolResult is the subset of those passes whose transcript already carried a
	// RoleTool message — i.e. the number of ranks that DISPATCHED the tool. A tool
	// result in a transcript is the loop-level witness that the tool actually ran.
	withToolResult int
}

func (p *epOwnedLoopPlanner) Model() string { return "ep-owned-loop" }

func (p *epOwnedLoopPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	sawToolResult := false
	for _, m := range messages {
		if m.Role == agent.RoleTool {
			sawToolResult = true
			break
		}
	}
	p.mu.Lock()
	p.calls++
	if sawToolResult {
		p.withToolResult++
	}
	p.mu.Unlock()

	if sawToolResult {
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "Booked direct."},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: 8, CompletionTokens: 3, TotalTokens: 11},
		}, nil
	}
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
			ID: "call_user", Type: "function",
			Function: agent.Func{Name: "get_user_details", Arguments: `{"user_id":"mia_li_3668"}`},
		}}},
		FinishReason: "tool_calls",
		Usage:        agent.Usage{PromptTokens: 6, CompletionTokens: 2, TotalTokens: 8},
	}, nil
}

func (p *epOwnedLoopPlanner) counts() (calls, withToolResult int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls, p.withToolResult
}

// TestAgentSessionsHandsNoOwnedLoopToTheRequestMirror makes epExemptOwnedLoop's claim
// checkable instead of merely written down.
//
// #5532 asked for the opposite assertion — drive a two-turn session with a follower
// configured and require the follower to enter the collective exactly twice, once per
// forward pass. That was written first and observed RED at "follower rank was released 0
// time(s)", which confirms the narrow premise (agent-sessions releases nobody) but not the
// prescribed cure: a REQUEST mirror cannot produce one release per forward pass, because
// its unit is one inbound body. Handing this route to the mirror would give each follower
// rank its own governed loop and a second real execution of the session's tool calls —
// unrecoverable, where a leader alone in a collective merely stalls. The barrier that does
// cover it is model.EPDecodeCoordinator (#4835), below the HTTP layer entirely.
//
// What is pinned here is therefore the exemption itself: with a follower rank CONFIGURED
// (otherwise the fanout no-ops and this asserts nothing) and a session that really runs two
// forward passes and one tool dispatch, no follower rank is contacted at all. That catches
// the naive "wire the bridge here too" change, which is the change that would duplicate the
// side effects.
func TestAgentSessionsHandsNoOwnedLoopToTheRequestMirror(t *testing.T) {
	srv := newTestServer(t)
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	planner := &epOwnedLoopPlanner{}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	calls := startRecordingFollowerRank(t)

	body, _ := json.Marshal(AgentSessionRequest{Goal: "Book me a direct flight.", MaxTurns: 4})
	resp, err := http.Post(ts.URL+"/v1/fak/agent/sessions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/fak/agent/sessions: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, raw)
	}

	// The leader really did run a MULTI-turn governed session with a real tool dispatch —
	// otherwise the negative below would be asserting the absence of a mirror for a request
	// that had no owned loop to mirror in the first place, which is vacuous.
	if got, withTool := planner.counts(); got != 2 || withTool != 1 {
		t.Fatalf("leader ran %d forward pass(es), %d of them after a tool result; want 2 and 1 (a two-turn governed session)", got, withTool)
	}

	// A BOUNDED negative window, not a bare channel read: the leader's session has already
	// finished, but a mirrored request started during it could still be in flight. Waiting is
	// what makes the absence real.
	select {
	case got := <-calls:
		t.Fatalf("a follower rank was handed the owned loop at %q — the request mirror's unit is one inbound body, so a follower on an owned-loop route runs its OWN N turns and executes the session's tool calls a second time; keep this route exempt and wire model.EPDecodeCoordinator (#4835) instead (#5532)", got.path)
	case <-time.After(time.Second):
	}
}

// TestNativeMessagesFanoutDoesNotDuplicateTheOwnedLoop is the half #5532 assumed already
// held: that no served route mirrors an inbound body into fak's OWN governed loop on a
// follower rank. It did not hold.
//
// /v1/messages is classified COVERED by the #5528 gate and called startEPFanoutFollowers
// unconditionally — including under `serve --native`, where the same handler drives
// agent.RunArm(fak=true). A follower rank is another process running the same binary with
// the same config, so the mirrored body reached the same native branch and the follower ran
// the WHOLE governed loop: its own N forward passes and its own tool dispatches. Observed
// RED at "4 forward passes and 2 tool-result turns across the ranks, want 2 and 1" — the
// world size multiplied the governed loop's real side effects. The cure is the `!s.native`
// guard in handleAnthropicMessages: the proxy arm still mirrors (the arm the bridge's
// one-body-one-pass unit is sound for), the owned-loop arm never does.
func TestNativeMessagesFanoutDoesNotDuplicateTheOwnedLoop(t *testing.T) {
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	planner := &epOwnedLoopPlanner{}
	srv, err := New(Config{EngineID: "localtools", Model: "test-model", VDSO: true, Native: true, NativeMaxTurns: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// Rank 1 is another process running the SAME native binary. Pointing the bridge at
	// this server models that exactly: the mirrored body is served by the same handler
	// with the same config, which is what a real sharded serve does.
	t.Setenv("FAK_EP_FANOUT_ADDRS", ts.URL)

	body, _ := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 256,
		"messages":   []map[string]string{{"role": "user", "content": "Book me a direct flight."}},
	})
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, raw)
	}
	var got anthropicMessageResponse
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode response: %v; body=%s", err, raw)
	}
	if got.Fak == nil || got.Fak.NativeArm == nil {
		t.Fatalf("response carried no fak.native_arm — the leader did not drive the owned loop, so this test asserts nothing; body=%s", raw)
	}

	// A BOUNDED negative window, not a bare read: the follower's mirrored request runs
	// concurrently with the leader's loop, so waiting is what makes the absence real.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, withTool := planner.counts(); withTool > 1 {
			calls, wt := planner.counts()
			t.Fatalf("a follower rank ran the owned loop too: %d forward passes and %d tool-result turns across the ranks, want 2 and 1 — configuring an EP follower multiplied the governed loop's tool dispatches by the world size (#5532)", calls, wt)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if calls, wt := planner.counts(); calls != 2 || wt != 1 {
		t.Fatalf("planner served %d forward passes, %d after a tool result; want exactly the leader's 2 and 1", calls, wt)
	}
}
