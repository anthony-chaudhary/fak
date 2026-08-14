package gateway

// native_serve_progress_test.go — the golden witness for #5148 (harness-native program
// #2388/#2387): the native owned-loop stream carries the loop's typed STATE, not only its
// prose. A multi-turn `serve --native` stream in which one tool call is DENIED and one is
// EXECUTED emits the full typed lifecycle sequence in order —
//
//	turn_started → (tool_started → call_adjudicated{verdict} → result_admitted{taint})+ → turn_done
//
// per turn — interleaved with the text deltas, terminating on message_stop carrying the
// fak.native_arm ArmMetrics. The kernel verdict (DENY on delete_account, ALLOW on
// book_flight) rides the call_adjudicated event a client can gate on. Runs under a
// deterministic mock streaming planner: no model serving, so it is witnessable on any box.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// progressStreamingPlanner drives a deterministic 3-turn owned-loop run for the structured
// progress witness: turn 1 emits a policy-DENIED delete_account call, turn 2 an ALLOWED
// book_flight call, turn 3 a streamed text final answer. The tool-call turns do not stream
// text (sink is left untouched); only the final turn streams, exactly as a real turn would.
type progressStreamingPlanner struct {
	turns int32
}

func (p *progressStreamingPlanner) Model() string            { return "progress-stream" }
func (p *progressStreamingPlanner) StreamingSupported() bool { return true }

func (p *progressStreamingPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return p.complete(ctx, nil, messages, tools, opts...)
}

func (p *progressStreamingPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return p.complete(ctx, sink, messages, tools, opts...)
}

func (p *progressStreamingPlanner) complete(_ context.Context, sink agent.StreamSink, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	switch atomic.AddInt32(&p.turns, 1) {
	case 1:
		return &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
				ID: "call_del", Type: "function", Function: agent.Func{Name: "delete_account", Arguments: `{"user_id":"mia_li_3668"}`},
			}}},
			FinishReason: "tool_calls",
			Usage:        agent.Usage{PromptTokens: len(messages) * 4, CompletionTokens: 2},
		}, nil
	case 2:
		return &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
				ID: "call_book", Type: "function", Function: agent.Func{Name: "book_flight", Arguments: `{"user_id":"mia_li_3668","flight_id":"UA123"}`},
			}}},
			FinishReason: "tool_calls",
			Usage:        agent.Usage{PromptTokens: len(messages) * 4, CompletionTokens: 2},
		}, nil
	default:
		if sink != nil {
			if err := sink("All done."); err != nil {
				return nil, err
			}
		}
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "All done."},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: len(messages) * 4, CompletionTokens: 3},
		}, nil
	}
}

// progressStep is one custom lifecycle SSE frame decoded to the fields the sequence asserts.
type progressStep struct {
	kind    string
	turn    int
	callID  string
	tool    string
	verdict string
	taint   string
}

func TestNativeStreamStructuredEvents(t *testing.T) {
	// RunArm(fak=true) builds its own kernel.New("localtools") + Configure() (registers the
	// localtools engine and the agent policy that DENIES delete_account); Configure here so
	// New validates against a registered engine, and re-register the inline region backend
	// in case a sibling test's abi.ResetForTest wiped it (blob's init registers the default).
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	tbl := session.NewTable()
	const trace = "native-progress-trace"
	srv, err := New(Config{
		EngineID:       "localtools",
		Model:          "test-model",
		VDSO:           true,
		Native:         true,
		NativeMaxTurns: 8,
		DecideSession: func(_ context.Context, tr string) SessionVerdict {
			v := tbl.Decide(tr)
			return SessionVerdict{Proceed: v.Proceed, MaxTokens: v.MaxTokens, MinGapMs: v.MinGapMs, Stop: v.Stop, Reason: v.Reason}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	srv.planner = &progressStreamingPlanner{}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 256,
		"stream":     true,
		"messages":   []map[string]string{{"role": "user", "content": "Clean up my account then book a flight."}},
	})
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", trace)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	frames := readAnthropicSSE(t, bufio.NewReader(resp.Body))

	// Partition the frames: the ordered lifecycle steps, the streamed text, and whether the
	// terminal message_stop carried the ArmMetrics witness.
	progressKinds := map[string]bool{"turn_started": true, "tool_started": true, "call_adjudicated": true, "result_admitted": true, "turn_done": true}
	var steps []progressStep
	var text string
	var stopHasArm bool
	for _, f := range frames {
		var obj map[string]any
		if err := json.Unmarshal([]byte(f.data), &obj); err != nil {
			t.Fatalf("decode frame event=%q data=%q: %v", f.event, f.data, err)
		}
		switch {
		case progressKinds[f.event]:
			if f.id == "" {
				t.Fatalf("%s event has no SSE id", f.event)
			}
			// Every custom lifecycle event carries the session trace tag.
			if obj["session"] != trace {
				t.Fatalf("%s event missing//wrong session tag: got %v want %q", f.event, obj["session"], trace)
			}
			s := progressStep{kind: f.event}
			if v, ok := obj["turn"].(float64); ok {
				s.turn = int(v)
			}
			s.callID, _ = obj["call_id"].(string)
			s.tool, _ = obj["tool"].(string)
			s.verdict, _ = obj["verdict"].(string)
			s.taint, _ = obj["taint"].(string)
			steps = append(steps, s)
		case obj["type"] == "content_block_delta":
			if delta, _ := obj["delta"].(map[string]any); delta["type"] == "text_delta" {
				text += delta["text"].(string)
			}
		case obj["type"] == "message_stop":
			fak, _ := obj["fak"].(map[string]any)
			arm, _ := fak["native_arm"].(map[string]any)
			stopHasArm = arm["arm"] == "fak"
		}
	}

	// (1) The full typed lifecycle sequence, in order. Two tool turns (delete DENIED, book
	// ALLOWED) each wrap tool_started → call_adjudicated → result_admitted; a final
	// answer turn carries only its boundary events.
	want := []progressStep{
		{kind: "turn_started", turn: 1},
		{kind: "tool_started", turn: 1, callID: "call_del", tool: "delete_account"},
		{kind: "call_adjudicated", turn: 1, callID: "call_del", tool: "delete_account", verdict: "DENY"},
		{kind: "result_admitted", turn: 1, callID: "call_del", tool: "delete_account", taint: "clean"},
		{kind: "turn_done", turn: 1},
		{kind: "turn_started", turn: 2},
		{kind: "tool_started", turn: 2, callID: "call_book", tool: "book_flight"},
		{kind: "call_adjudicated", turn: 2, callID: "call_book", tool: "book_flight", verdict: "ALLOW"},
		{kind: "result_admitted", turn: 2, callID: "call_book", tool: "book_flight", taint: "clean"},
		{kind: "turn_done", turn: 2},
		{kind: "turn_started", turn: 3},
		{kind: "turn_done", turn: 3},
	}
	if len(steps) != len(want) {
		t.Fatalf("lifecycle step count = %d, want %d\n got: %+v\nwant: %+v", len(steps), len(want), steps, want)
	}
	for i := range want {
		if steps[i] != want[i] {
			t.Fatalf("lifecycle step %d = %+v, want %+v\nfull got: %+v", i, steps[i], want[i], steps)
		}
	}

	// (2) The denied call carried a closed refusal reason (a client can gate on it); the
	// executed call did not. Reason is checked separately so the ordered compare above can
	// stay reason-agnostic (the exact token is the policy's to name).
	for _, f := range frames {
		if f.event != "call_adjudicated" {
			continue
		}
		var obj map[string]any
		_ = json.Unmarshal([]byte(f.data), &obj)
		reason, _ := obj["reason"].(string)
		if obj["verdict"] == "DENY" && reason == "" {
			t.Fatalf("DENY call_adjudicated carried no closed refusal reason: %s", f.data)
		}
		if obj["verdict"] == "ALLOW" && reason != "" {
			t.Fatalf("ALLOW call_adjudicated should carry no refusal reason, got %q", reason)
		}
	}

	// (3) The text still streamed and the terminal witness still rode message_stop — the
	// structured events are ADDITIVE to the existing text+ArmMetrics stream, not a
	// replacement.
	if text != "All done." {
		t.Fatalf("streamed text = %q, want the final answer deltas", text)
	}
	if !stopHasArm {
		t.Fatalf("message_stop did not carry fak.native_arm; frames=%+v", frames)
	}
}

// TestNativeStreamUnwiredLoopIsUnchanged is the OTHER half of #5148's done-when: the
// observer is a pure additive RunOption, so with none wired the owned loop is
// byte-for-byte the historical loop. The golden test above witnesses the WIRED path (the
// HTTP handler always passes a non-nil observer, so it can never reach the unwired arm of
// runNativeArmStream's `if onProgress != nil`); this drives the same deterministic 3-turn
// run through runNativeArmStream twice — once unwired, once observed — and pins that
// observation changes NOTHING the loop produces: same streamed text, same ArmMetrics
// (compared as marshalled bytes, the literal reading of "byte-for-byte"). Watching the
// loop must not perturb it.
func TestNativeStreamUnwiredLoopIsUnchanged(t *testing.T) {
	// Same bootstrap as the golden test: RunArm(fak=true) builds its own kernel, and
	// runNativeArmStream's ensureGovernedRungs() self-heals a sibling's abi.ResetForTest.
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	// One fresh planner per run — the turn script is a stateful counter, so a shared
	// planner would make the second run start mid-script and the compare meaningless.
	run := func(obs agent.ProgressObserver) (agent.ArmMetrics, string) {
		t.Helper()
		srv := &Server{planner: &progressStreamingPlanner{}, nativeMaxTurns: 8}
		var text strings.Builder
		m, err := srv.runNativeArmStream(context.Background(), &agent.AnthropicMessagesRequest{
			Messages: []agent.Message{{Role: agent.RoleUser, Content: "Clean up my account then book a flight."}},
		}, "native-unwired-trace", func(delta string) error {
			text.WriteString(delta)
			return nil
		}, obs)
		if err != nil {
			t.Fatalf("runNativeArmStream: %v", err)
		}
		return m, text.String()
	}

	bareArm, bareText := run(nil)

	var seen []agent.ProgressEvent
	obsArm, obsText := run(func(ev agent.ProgressEvent) { seen = append(seen, ev) })

	// Non-vacuity FIRST: if the observed run never actually emitted, the equality below
	// would compare two identical unwired runs and prove nothing. Pin that the observed
	// run really did see the full lifecycle — both kernel verdicts included.
	//
	// The count is a RELATION over what this run actually did, not a frozen total. The
	// documented sequence is turn_started → (tool_started → call_adjudicated →
	// result_admitted)+ → turn_done, so the emission is two boundary events per turn plus
	// three per adjudicated call. Anchoring the turn count to the UNWIRED run's ArmMetrics
	// is what keeps this non-vacuous: an observer that emitted nothing scores zero turns
	// and fails here instead of agreeing with itself.
	byKind := map[agent.ProgressEventKind]int{}
	for _, ev := range seen {
		byKind[ev.Kind]++
	}
	turns, calls := byKind[agent.ProgressTurnStarted], byKind[agent.ProgressCallAdjudicated]
	if turns != int(bareArm.Turns) || byKind[agent.ProgressTurnDone] != turns {
		t.Fatalf("observed %d turn_started / %d turn_done, want one pair for each of the unwired run's %d turns: %+v",
			turns, byKind[agent.ProgressTurnDone], bareArm.Turns, seen)
	}
	if byKind[agent.ProgressToolStarted] != calls || byKind[agent.ProgressResultAdmitted] != calls {
		t.Fatalf("observed %d tool_started / %d call_adjudicated / %d result_admitted, want one of each per call: %+v",
			byKind[agent.ProgressToolStarted], calls, byKind[agent.ProgressResultAdmitted], seen)
	}
	if want := 2*turns + 3*calls; len(seen) != want {
		t.Fatalf("observed run emitted %d lifecycle events, want %d for %d turns and %d calls "+
			"— something emitted outside the documented sequence: %+v", len(seen), want, turns, calls, seen)
	}
	verdicts := map[string]int{}
	for _, ev := range seen {
		if ev.Kind == agent.ProgressCallAdjudicated {
			verdicts[ev.Verdict]++
		}
	}
	if verdicts["DENY"] != 1 || verdicts["ALLOW"] != 1 {
		t.Fatalf("observed verdicts = %v, want one DENY (delete_account) and one ALLOW (book_flight)", verdicts)
	}

	// ...and that the unwired run actually DROVE the loop. Without this the byte compare
	// below could pass on two zero-value ArmMetrics.
	if bareArm.Turns != 3 || bareArm.Denies != 1 || bareArm.FinalAnswer != "All done." {
		t.Fatalf("unwired run did not drive the 3-turn script: turns=%d denies=%d final=%q",
			bareArm.Turns, bareArm.Denies, bareArm.FinalAnswer)
	}

	// Second vacuity guard: the unwired run must have actually DRIVEN the loop. An
	// all-zero ArmMetrics on both sides would satisfy the byte compare below while
	// witnessing nothing, so pin the real outcome of the 3-turn script — the denied call,
	// the executed one, and the final answer.
	if bareArm.Turns != 3 || bareArm.ToolCalls != 2 || bareArm.Denies != 1 || bareArm.EngineCalls != 1 || bareArm.FinalAnswer != "All done." {
		t.Fatalf("unwired run did not drive the full script: %+v", bareArm)
	}

	// The loop's own output is identical wired vs unwired.
	if bareText != obsText {
		t.Fatalf("streamed text differs with an observer wired: unwired=%q observed=%q", bareText, obsText)
	}
	bareJSON, err := json.Marshal(bareArm)
	if err != nil {
		t.Fatalf("marshal unwired ArmMetrics: %v", err)
	}
	obsJSON, err := json.Marshal(obsArm)
	if err != nil {
		t.Fatalf("marshal observed ArmMetrics: %v", err)
	}
	if !bytes.Equal(bareJSON, obsJSON) {
		t.Fatalf("ArmMetrics differ with an observer wired — the observer perturbed the loop\nunwired : %s\nobserved: %s", bareJSON, obsJSON)
	}
}
