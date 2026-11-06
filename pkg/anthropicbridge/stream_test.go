package anthropicbridge

// stream_test.go witnesses the streaming half: StreamTranslator's live text
// arm (holdback-gated text_delta emission), the held tool-call arm released by
// Finish, the stop holdback semantics, the incomplete-stream and in-band-error
// sentinels, and the SSE frame shape.

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// decodeEvent unmarshals one event's payload into a generic map for spot checks.
func decodeEvent(t *testing.T, e Event) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(e.Data, &m); err != nil {
		t.Fatalf("%s payload: %v (data: %s)", e.Name, err, e.Data)
	}
	return m
}

func TestStreamTranslatorTextAndToolUse(t *testing.T) {
	tr := NewStreamTranslator("msg_1", "m", 7, []string{"END"})
	var feedEvents []Event
	feed := func(data string) {
		t.Helper()
		evs, err := tr.Feed(data)
		if err != nil {
			t.Fatalf("Feed(%s): %v", data, err)
		}
		feedEvents = append(feedEvents, evs...)
	}

	// Role-only chunk: nothing to emit yet.
	feed(`{"model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`)
	// "He" sits inside the holdback window (max(len("END"))-1 == 2): held, so the
	// message_start + text block have not opened yet either.
	feed(`{"choices":[{"index":0,"delta":{"content":"He"}}]}`)
	// "llo" pushes the buffer past the window: the safe prefix "Hel" releases and
	// the tail "lo" stays held against a possible stop marker.
	feed(`{"choices":[{"index":0,"delta":{"content":"llo"}}]}`)

	if len(feedEvents) != 3 {
		t.Fatalf("feed events = %d (%v), want 3", len(feedEvents), feedEvents)
	}
	if feedEvents[0].Name != "message_start" {
		t.Fatalf("first event = %s, want message_start", feedEvents[0].Name)
	}
	ms := decodeEvent(t, feedEvents[0])
	msg := ms["message"].(map[string]any)
	if msg["id"] != "msg_1" || msg["model"] != "m" {
		t.Errorf("message_start echo = %v/%v, want msg_1/m", msg["id"], msg["model"])
	}
	if usage := msg["usage"].(map[string]any); usage["input_tokens"] != float64(7) {
		t.Errorf("message_start usage.input_tokens = %v, want 7", usage["input_tokens"])
	}
	if feedEvents[1].Name != "content_block_start" {
		t.Fatalf("second event = %s, want content_block_start", feedEvents[1].Name)
	}
	cbs := decodeEvent(t, feedEvents[1])
	if cbs["index"] != float64(0) {
		t.Errorf("content_block_start index = %v, want 0", cbs["index"])
	}
	if blk := cbs["content_block"].(map[string]any); blk["type"] != "text" {
		t.Errorf("content_block type = %v, want text", blk["type"])
	}
	if feedEvents[2].Name != "content_block_delta" {
		t.Fatalf("third event = %s, want content_block_delta", feedEvents[2].Name)
	}
	if delta := decodeEvent(t, feedEvents[2])["delta"].(map[string]any); delta["type"] != "text_delta" || delta["text"] != "Hel" {
		t.Errorf("first text_delta = %v, want text_delta Hel", delta)
	}

	// Tool-call fragments are HELD until Finish: no events at all — in particular
	// no content_block_stop and no tool_use block while the stream is live.
	feed(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"ls","arguments":""}}]}}]}`)
	feed(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]}}]}`)
	feed(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\".\"}"}}]}}]}`)
	feed(`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
	feed(`{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4}}`)
	feed("[DONE]")
	if len(feedEvents) != 3 {
		t.Fatalf("feed events after tool/finish/usage/DONE = %d, want 3 (tool fragments held)", len(feedEvents))
	}
	for _, e := range feedEvents {
		if e.Name == "content_block_stop" {
			t.Errorf("content_block_stop emitted before Finish")
		}
	}

	finishEvents, comp, err := tr.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	wantNames := []string{
		"content_block_delta", // the held tail "lo" flushes with the block
		"content_block_stop",  // index 0
		"content_block_start", // index 1: tool_use
		"content_block_delta", // input_json_delta
		"content_block_stop",  // index 1
		"message_delta",
		"message_stop",
	}
	if len(finishEvents) != len(wantNames) {
		t.Fatalf("finish events = %d (%v), want %d", len(finishEvents), finishEvents, len(wantNames))
	}
	for i, name := range wantNames {
		if finishEvents[i].Name != name {
			t.Errorf("finishEvents[%d] = %s, want %s", i, finishEvents[i].Name, name)
		}
	}
	if delta := decodeEvent(t, finishEvents[0])["delta"].(map[string]any); delta["text"] != "lo" {
		t.Errorf("holdback tail text_delta = %v, want lo", delta["text"])
	}
	if stop := decodeEvent(t, finishEvents[1]); stop["index"] != float64(0) {
		t.Errorf("text content_block_stop index = %v, want 0", stop["index"])
	}
	toolStart := decodeEvent(t, finishEvents[2])
	if toolStart["index"] != float64(1) {
		t.Errorf("tool content_block_start index = %v, want 1", toolStart["index"])
	}
	if blk := toolStart["content_block"].(map[string]any); blk["type"] != "tool_use" || blk["id"] != "call_9" || blk["name"] != "ls" {
		t.Errorf("tool content_block = %v, want tool_use call_9 ls", blk)
	}
	if delta := decodeEvent(t, finishEvents[3])["delta"].(map[string]any); delta["type"] != "input_json_delta" || delta["partial_json"] != `{"path":"."}` {
		t.Errorf("input_json_delta = %v, want the whole accumulated argument object", delta)
	}
	if stop := decodeEvent(t, finishEvents[4]); stop["index"] != float64(1) {
		t.Errorf("tool content_block_stop index = %v, want 1", stop["index"])
	}
	md := decodeEvent(t, finishEvents[5])
	if d := md["delta"].(map[string]any); d["stop_reason"] != "tool_use" {
		t.Errorf("message_delta stop_reason = %v, want tool_use", d["stop_reason"])
	}
	if usage := md["usage"].(map[string]any); usage["input_tokens"] != float64(11) || usage["output_tokens"] != float64(4) {
		t.Errorf("message_delta usage = %v, want 11/4 (the real usage chunk)", usage)
	}

	if comp == nil {
		t.Fatalf("Finish returned nil Completion")
	}
	if comp.Message.Content != "Hello" {
		t.Errorf("comp.Content = %q, want Hello", comp.Message.Content)
	}
	if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "call_9" {
		t.Errorf("comp.ToolCalls = %+v, want one call call_9", comp.Message.ToolCalls)
	}
	if comp.Message.ToolCalls[0].Function.Arguments != `{"path":"."}` {
		t.Errorf("comp arguments = %q, want the concatenated fragments", comp.Message.ToolCalls[0].Function.Arguments)
	}
	if comp.FinishReason != "tool_calls" {
		t.Errorf("comp.FinishReason = %q, want tool_calls", comp.FinishReason)
	}
	if comp.Usage.PromptTokens != 11 || comp.Usage.CompletionTokens != 4 {
		t.Errorf("comp.Usage = %d/%d, want 11/4", comp.Usage.PromptTokens, comp.Usage.CompletionTokens)
	}
	// Idempotent after success.
	if evs2, comp2, err := tr.Finish(); err != nil || comp2 != nil || len(evs2) != 0 {
		t.Errorf("second Finish = (%d events, %v, %v), want (0, nil, nil)", len(evs2), comp2, err)
	}
}

func TestStreamTranslatorStopHoldback(t *testing.T) {
	tr := NewStreamTranslator("msg_h", "m", 3, []string{"STOP"})
	// "AB" is within the holdback window (max(len("STOP"))-1 == 3): held.
	evs, err := tr.Feed(`{"choices":[{"index":0,"delta":{"content":"AB"}}]}`)
	if err != nil {
		t.Fatalf("Feed(AB): %v", err)
	}
	if len(evs) != 0 {
		t.Errorf("Feed(AB) events = %v, want none (held)", evs)
	}
	// "CSTOP" completes the match at index 3 of "ABCSTOP": the pre-stop text "ABC"
	// releases and the stop text itself is trimmed, never emitted.
	evs, err = tr.Feed(`{"choices":[{"index":0,"delta":{"content":"CSTOP"}}]}`)
	if err != nil {
		t.Fatalf("Feed(CSTOP): %v", err)
	}
	if len(evs) != 3 || evs[0].Name != "message_start" || evs[1].Name != "content_block_start" || evs[2].Name != "content_block_delta" {
		t.Fatalf("Feed(CSTOP) events = %v, want message_start + content_block_start + text_delta", evs)
	}
	if delta := decodeEvent(t, evs[2])["delta"].(map[string]any); delta["text"] != "ABC" {
		t.Errorf("post-match text_delta = %v, want ABC", delta["text"])
	}
	// Content after a matched stop is gated off (Stopped) but still accumulates.
	evs, err = tr.Feed(`{"choices":[{"index":0,"delta":{"content":"D"}}]}`)
	if err != nil {
		t.Fatalf("Feed(D): %v", err)
	}
	if len(evs) != 0 {
		t.Errorf("Feed(D) events = %v, want none (stopped)", evs)
	}
	if _, err := tr.Feed("[DONE]"); err != nil {
		t.Fatalf("Feed([DONE]): %v", err)
	}
	finishEvents, comp, err := tr.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	// Finish emits no further text: the block just closes, then the terminal pair.
	wantNames := []string{"content_block_stop", "message_delta", "message_stop"}
	if len(finishEvents) != len(wantNames) {
		t.Fatalf("finish events = %v, want %v", finishEvents, wantNames)
	}
	for i, name := range wantNames {
		if finishEvents[i].Name != name {
			t.Errorf("finishEvents[%d] = %s, want %s", i, finishEvents[i].Name, name)
		}
	}
	// Accumulation is raw (the deltas concatenate verbatim; the holdback gates
	// emission only), so the canonical completion carries the full raw text.
	if comp.Message.Content != "ABCSTOPD" {
		t.Errorf("comp.Content = %q, want the raw accumulation ABCSTOPD", comp.Message.Content)
	}
	// The stop text itself never reaches any emitted text_delta.
	for _, e := range append(append([]Event{}, evs...), finishEvents...) {
		if e.Name == "content_block_delta" {
			if delta := decodeEvent(t, e)["delta"].(map[string]any); strings.Contains(delta["text"].(string), "STOP") {
				t.Errorf("emitted text_delta leaks the stop text: %v", delta["text"])
			}
		}
	}
}

func TestStreamTranslatorIncomplete(t *testing.T) {
	tr := NewStreamTranslator("msg_i", "m", 1, nil)
	if _, err := tr.Feed(`{"choices":[{"index":0,"delta":{"content":"hi"}}]}`); err != nil {
		t.Fatalf("Feed: %v", err)
	}
	events, comp, err := tr.Finish()
	if err == nil || !errors.Is(err, ErrIncompleteStream) {
		t.Fatalf("Finish error = %v, want ErrIncompleteStream", err)
	}
	if comp != nil || len(events) != 0 {
		t.Errorf("Finish = (%d events, %v), want (0, nil) on incomplete stream", len(events), comp)
	}
}

func TestStreamTranslatorInBandError(t *testing.T) {
	tr := NewStreamTranslator("msg_e", "m", 1, nil)
	events, err := tr.Feed(`{"error":{"code":502,"message":"upstream boom"}}`)
	if err == nil || !errors.Is(err, ErrStreamError) {
		t.Fatalf("Feed error = %v, want ErrStreamError", err)
	}
	if !strings.Contains(err.Error(), "upstream boom") {
		t.Errorf("error text = %q, want the in-band message carried", err.Error())
	}
	if len(events) != 0 {
		t.Errorf("Feed events on error = %d, want 0", len(events))
	}
}

// A non-conformant upstream that omits tool_calls[].id must still yield an
// answerable tool_use block: the planner's streaming path mints call_fak_N ids
// (NormalizeStreamedToolCalls); the routed seam must not emit id:"" and strand
// the client's tool_result round-trip.
func TestStreamTranslatorMintsToolCallIDs(t *testing.T) {
	tr := NewStreamTranslator("msg_n", "m", 3, nil)
	for _, data := range []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"ls","arguments":"{}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		"[DONE]",
	} {
		if _, err := tr.Feed(data); err != nil {
			t.Fatalf("Feed(%s): %v", data, err)
		}
	}
	events, comp, err := tr.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if comp.Message.ToolCalls[0].ID != "call_fak_0" {
		t.Errorf("minted id = %q, want call_fak_0", comp.Message.ToolCalls[0].ID)
	}
	if comp.Message.ToolCalls[0].Type != "function" {
		t.Errorf("defaulted type = %q, want function", comp.Message.ToolCalls[0].Type)
	}
	// The emitted tool_use block carries the minted id, not the empty one.
	var found bool
	for _, ev := range events {
		var payload struct {
			Type         string `json:"type"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
			} `json:"content_block"`
		}
		if json.Unmarshal(ev.Data, &payload) == nil && payload.Type == "content_block_start" && payload.ContentBlock.Type == "tool_use" {
			found = true
			if payload.ContentBlock.ID != "call_fak_0" {
				t.Errorf("tool_use block id = %q, want call_fak_0", payload.ContentBlock.ID)
			}
		}
	}
	if !found {
		t.Errorf("no tool_use content_block_start among Finish events")
	}
	// The finish rewrite: a surviving tool call rewrites the finish reason.
	if comp.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls (normalize rewrite)", comp.FinishReason)
	}
}

func TestEventFrame(t *testing.T) {
	data, err := json.Marshal(map[string]any{"type": "ping"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	e := Event{Name: "ping", Data: data}
	want := "event: ping\ndata: {\"type\":\"ping\"}\n\n"
	if got := string(e.Frame()); got != want {
		t.Errorf("Frame() = %q, want %q", got, want)
	}
}
