package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type scriptedCanaryRuntime struct {
	turns []hookCanaryTurn
	err   error
	calls int
}

func (s *scriptedCanaryRuntime) Run(context.Context, string, string) (hookCanaryTurn, error) {
	i := s.calls
	s.calls++
	if s.err != nil {
		return hookCanaryTurn{}, s.err
	}
	return s.turns[i], nil
}
func greenTurn(deny bool) hookCanaryTurn {
	return hookCanaryTurn{ItemCompleted: true, TurnCompleted: true, Denied: deny, PreToolUse: lifecycleCounts{Denominator: 1, Attempted: 1, Succeeded: 1}, Stop: lifecycleCounts{Denominator: 1, Attempted: 1, Succeeded: 1}}
}
func greenProfile() hookProfileReport {
	return hookProfileReport{Verdict: "HEALTHY", EventContracts: []hookEventContract{{EventName: "post_tool_use", Status: "intentionally_disabled"}}}
}

func TestRunHookCanaryReadbackAndClassification(t *testing.T) {
	rt := &scriptedCanaryRuntime{turns: []hookCanaryTurn{greenTurn(false), greenTurn(true)}}
	r := runHookCanary(context.Background(), greenProfile(), rt, t.TempDir())
	if r.State != canaryGreen || r.SentinelBefore == "" || r.SentinelBefore != r.SentinelAfter || rt.calls != 2 {
		t.Fatalf("receipt=%+v calls=%d", r, rt.calls)
	}
}
func TestClassifyHookCanaryStates(t *testing.T) {
	base := hookCanaryReceipt{ProfileVerdict: "HEALTHY", PostToolUseState: "intentionally_disabled", SentinelBefore: "x", SentinelAfter: "x", Turns: []hookCanaryTurn{greenTurn(false), greenTurn(true)}}
	cases := []struct {
		name, want string
		mut        func(*hookCanaryReceipt)
	}{{"launcher", canaryLauncherFailure, func(r *hookCanaryReceipt) { r.Turns = nil }}, {"plugin", canaryPluginNotActivated, func(r *hookCanaryReceipt) { r.Turns[0].Stop = lifecycleCounts{} }}, {"post", canaryPostToolUseReintroduced, func(r *hookCanaryReceipt) { r.Turns[0].PostToolUseRows = 1 }}, {"backend", canaryBackendFailure, func(r *hookCanaryReceipt) { r.Turns[0].ItemCompleted = false }}, {"deny", canaryDenySemanticsFailure, func(r *hookCanaryReceipt) { r.Turns[1].Denied = false }}, {"green", canaryGreen, func(*hookCanaryReceipt) {}}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := base
			r.Turns = append([]hookCanaryTurn(nil), base.Turns...)
			tc.mut(&r)
			got, _ := classifyHookCanary(r)
			if got != tc.want {
				t.Fatalf("got %s want %s", got, tc.want)
			}
		})
	}
}
func TestWriteHookCanaryReceiptAtomic(t *testing.T) {
	p := filepath.Join(t.TempDir(), "receipt.json")
	r := hookCanaryReceipt{Schema: codexHookCanarySchema, State: canaryGreen}
	if err := writeHookCanaryReceipt(p, r); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	var got hookCanaryReceipt
	if json.Unmarshal(b, &got) != nil || got.Schema != codexHookCanarySchema {
		t.Fatalf("receipt=%s", b)
	}
}

type fakeCanaryTransport struct {
	sent     []any
	messages []appServerMessage
}

func (f *fakeCanaryTransport) Send(v any) error { f.sent = append(f.sent, v); return nil }
func (f *fakeCanaryTransport) Receive(context.Context) (appServerMessage, error) {
	if len(f.messages) == 0 {
		return appServerMessage{}, errors.New("empty")
	}
	m := f.messages[0]
	f.messages = f.messages[1:]
	return m, nil
}
func (f *fakeCanaryTransport) Close() error { return nil }
func msg(id int, method string, v any) appServerMessage {
	b, _ := json.Marshal(v)
	if id > 0 {
		return appServerMessage{ID: id, Result: b}
	}
	return appServerMessage{Method: method, Params: b}
}
func hookMsg(method, event, status string) appServerMessage {
	return msg(0, method, map[string]any{"threadId": "th", "turnId": "tu", "run": map[string]any{"id": event, "eventName": event, "status": status}})
}
func TestRunHookCanaryTurnParsesLifecycle(t *testing.T) {
	f := &fakeCanaryTransport{messages: []appServerMessage{msg(2, "", map[string]any{"thread": map[string]string{"id": "th"}}), msg(3, "", map[string]any{"turn": map[string]string{"id": "tu"}}), hookMsg("hook/started", "preToolUse", "running"), hookMsg("hook/completed", "preToolUse", "completed"), msg(0, "item/completed", map[string]any{"item": map[string]string{"status": "policy_block"}}), hookMsg("hook/started", "stop", "running"), hookMsg("hook/completed", "stop", "completed"), msg(0, "turn/completed", map[string]any{})}}
	r, err := runHookCanaryTurn(context.Background(), f, t.TempDir(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if !r.ItemCompleted || !r.TurnCompleted || !r.Denied || r.PreToolUse.Succeeded != 1 || r.Stop.Succeeded != 1 || r.PostToolUseRows != 0 {
		t.Fatalf("%+v", r)
	}
	if len(f.sent) != 4 {
		t.Fatalf("sent=%d", len(f.sent))
	}
}
func TestRunHookCanaryTurnTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	time.Sleep(time.Millisecond)
	f := &fakeCanaryTransport{}
	_, err := runHookCanaryTurn(ctx, f, t.TempDir(), "x")
	if err == nil {
		t.Fatal("expected error")
	}
}
func TestCanaryJSONIsBounded(t *testing.T) {
	r := hookCanaryReceipt{Schema: codexHookCanarySchema, State: canaryGreen}
	b, _ := json.Marshal(r)
	if bytes.Contains(b, []byte("fak-codex-hook-canary\\n")) {
		t.Fatal("sentinel contents leaked")
	}
}
