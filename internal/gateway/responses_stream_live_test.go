package gateway

// responses_stream_live_test.go witnesses the #12765 live passthrough on
// POST /v1/responses: with a planner that can stream the wire live, a stream:true
// client receives the Responses SSE events INCREMENTALLY — content deltas arrive
// while the upstream planner is still running, before CompleteStream returns — and
// a planner the W1 gate refuses falls back to the untouched buffered synthesis
// (writeResponsesStream) with a byte-identical event stream for the same request.

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// scriptedLiveResponsesPlanner is a StreamingPlanner whose CompleteStream
// fragments arrive on a scripted schedule, so the test can prove deltas reach the
// client BEFORE the planner returns. It also carries W1's gate so the fallback
// witness can refuse the live arm explicitly.
type scriptedLiveResponsesPlanner struct {
	fragments []string      // content fragments sinked in order
	fragGap   time.Duration // sleep BETWEEN fragments (after every one but the last)
	comp      *agent.Completion

	supports   bool // StreamingSupported()
	gateAllows bool // ResponsesWireStreamsLive verdict

	returnedAt time.Time // captured when CompleteStream is about to return

	sawFragments []string
}

func (p *scriptedLiveResponsesPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.comp, nil
}

func (*scriptedLiveResponsesPlanner) Model() string { return "scripted-responses-live" }

// Provider lets plannerWireProvider resolve the wire family for the W1 gate;
// the live arm is only reachable for providers the gate recognizes (ProviderOpenAI).
func (*scriptedLiveResponsesPlanner) Provider() agent.Provider { return agent.ProviderOpenAI }

func (p *scriptedLiveResponsesPlanner) StreamingSupported() bool { return p.supports }

func (p *scriptedLiveResponsesPlanner) ResponsesWireStreamsLive(agent.Provider, bool) bool {
	return p.gateAllows
}

func (p *scriptedLiveResponsesPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	defer func() { p.returnedAt = time.Now() }()
	for i, frag := range p.fragments {
		p.sawFragments = append(p.sawFragments, frag)
		if err := sink(frag); err != nil {
			return nil, err
		}
		if p.fragGap > 0 && i < len(p.fragments)-1 {
			time.Sleep(p.fragGap)
		}
	}
	return p.comp, nil
}

// timedSSEEvent is one parsed Responses SSE frame with its client-side arrival time.
type timedSSEEvent struct {
	name string
	data string
	at   time.Time
}

// readTimedResponsesStream POSTs a stream:true /v1/responses request to base and
// parses the response body into timed SSE events (arrival order = wire order).
func readTimedResponsesStream(t *testing.T, base, body string) ([]timedSSEEvent, *http.Response) {
	t.Helper()
	resp, err := http.Post(base+"/v1/responses", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}
	var events []timedSSEEvent
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event:"):
			events = append(events, timedSSEEvent{name: strings.TrimSpace(strings.TrimPrefix(line, "event:")), at: time.Now()})
		case strings.HasPrefix(line, "data:"):
			if len(events) > 0 {
				events[len(events)-1].data += strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
		}
	}
	resp.Body.Close()
	return events, resp
}

func TestResponsesLiveStreamsDeltasBeforeUpstreamEOF(t *testing.T) {
	srv := newTestServer(t)
	planner := &scriptedLiveResponsesPlanner{
		fragments:  []string{"Hel", "lo"},
		fragGap:    150 * time.Millisecond,
		supports:   true,
		gateAllows: true,
		comp: &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "Hello"},
			FinishReason: "stop",
		},
	}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	events, _ := readTimedResponsesStream(t, ts.URL, `{"model":"test-model","input":"hi","stream":true}`)

	var names []string
	for _, ev := range events {
		names = append(names, ev.name)
	}
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.output_text.delta", // "Hel"
		"response.output_text.delta", // "lo"
		"response.output_text.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(names) != len(want) {
		t.Fatalf("event sequence = %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Fatalf("event sequence = %v, want %v (mismatch at %d)", names, want, i)
		}
	}

	// LIVE-wALITY witness: the first content delta reached the client's parser while
	// the planner was still streaming — strictly before CompleteStream returned. The
	// 150ms inter-fragment sleep makes the margin far larger than any scheduling
	// jitter; a buffered-only path emits its single fat delta only after the turn.
	returnedAt := planner.returnedAt
	if returnedAt.IsZero() {
		t.Fatal("planner return time not captured — CompleteStream never ran")
	}
	if !events[2].at.Before(returnedAt) {
		t.Fatalf("first delta arrived at %v, want strictly before CompleteStream returned at %v (the live passthrough streamed late)", events[2].at, returnedAt)
	}
	if len(planner.sawFragments) != 2 || planner.sawFragments[0] != "Hel" || planner.sawFragments[1] != "lo" {
		t.Fatalf("sink fragments = %v, want [Hel lo]", planner.sawFragments)
	}

	// Delta payloads: the two fragments streamed verbatim, in order.
	var deltas []string
	for _, ev := range events {
		if ev.name != "response.output_text.delta" {
			continue
		}
		var payload struct {
			Delta string `json:"delta"`
		}
		if err := json.Unmarshal([]byte(ev.data), &payload); err != nil {
			t.Fatalf("decode delta %s: %v", ev.data, err)
		}
		deltas = append(deltas, payload.Delta)
	}
	if len(deltas) != 2 || deltas[0] != "Hel" || deltas[1] != "lo" {
		t.Fatalf("deltas = %q, want [Hel lo]", deltas)
	}

	// response.created: in_progress, empty output/text, zero usage, output_index 0 item.
	created := decodeResponsesLiveEvent(t, events[0].data)
	if created.Status != "in_progress" || created.OutputText != "" || created.ID == "" {
		t.Fatalf("response.created = %+v, want in_progress + non-empty id + empty output_text", created)
	}
	if len(created.Output) != 0 || created.Usage.TotalTokens != 0 {
		t.Fatalf("response.created output/usage = %+v/%+v, want empty/zero", created.Output, created.Usage)
	}
	var addedEnvelope struct {
		OutputIndex int                 `json:"output_index"`
		Item        responsesOutputItem `json:"item"`
	}
	if err := json.Unmarshal([]byte(events[1].data), &addedEnvelope); err != nil {
		t.Fatalf("decode added: %v", err)
	}
	if addedEnvelope.OutputIndex != 0 || addedEnvelope.Item.Type != "message" || addedEnvelope.Item.ID != "msg_fak_live_0" {
		t.Fatalf("output_item.added = %+v, want output_index 0 message msg_fak_live_0", addedEnvelope)
	}

	// response.completed: full text, zero-OK usage, terminal event, model echoed.
	series := events[len(events)-1]
	completed := decodeResponsesLiveEvent(t, series.data)
	if completed.Status != "completed" {
		t.Fatalf("response.completed status = %q, want completed", completed.Status)
	}
	if completed.OutputText != "Hello" {
		t.Fatalf("response.completed output_text = %q, want Hello", completed.OutputText)
	}
	if completed.Usage.TotalTokens != 0 || completed.Usage.InputTokens != 0 || completed.Usage.OutputTokens != 0 {
		t.Fatalf("response.completed usage = %+v, want zero-OK", completed.Usage)
	}
	if completed.Model != "test-model" {
		t.Fatalf("response.completed model = %q, want test-model", completed.Model)
	}
	msgItem, ok := findResponsesLiveItem(completed.Output, "message")
	if !ok || len(msgItem.Content) != 1 || msgItem.Content[0].Text != "Hello" {
		t.Fatalf("response.completed message item = %+v, want one output_text part Hello", msgItem)
	}

	// Sequence discipline: 0,1,2,... one per event, in arrival order.
	for i, ev := range events {
		var envelope struct {
			SequenceNumber int `json:"sequence_number"`
		}
		if err := json.Unmarshal([]byte(ev.data), &envelope); err != nil {
			t.Fatalf("sequence_number at event %d (%s): %v: %s", i, ev.name, err, ev.data)
		}
		if envelope.SequenceNumber != i {
			t.Fatalf("sequence_number at event %d = %d, want %d", i, envelope.SequenceNumber, i)
		}
	}
}

func TestResponsesLiveFallsBackWhenUpstreamCannotStream(t *testing.T) {
	comp := &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "Hello"},
		FinishReason: "stop",
	}

	// Fallback server: the planner CAN stream (StreamingSupported) but W1's gate
	// REFUSES the Responses live arm — streamResponsesLive must return false having
	// written nothing, and the request must fall through to writeResponsesStream.
	fallback := newTestServer(t)
	fallback.planner = &scriptedLiveResponsesPlanner{supports: true, gateAllows: false, comp: comp}
	fallbackTS := httptest.NewServer(fallback.Handler())
	defer fallbackTS.Close()

	// Control server: a plain Complete-only planner (never a StreamingPlanner) on the
	// identical request — the buffered route this wire has always taken.
	control := newTestServer(t)
	control.planner = stubPlanner{comp: comp}
	controlTS := httptest.NewServer(control.Handler())
	defer controlTS.Close()

	const body = `{"model":"test-model","input":"hi","stream":true}`
	fallbackEvents, _ := readTimedResponsesStream(t, fallbackTS.URL, body)
	controlEvents, _ := readTimedResponsesStream(t, controlTS.URL, body)

	render := func(t *testing.T, events []timedSSEEvent) string {
		t.Helper()
		var b strings.Builder
		for _, ev := range events {
			b.WriteString("event: " + ev.name + "\n")
			b.WriteString("data: " + ev.data + "\n\n")
		}
		return b.String()
	}

	// Byte-identity witness. The ONLY normalization is the two values that are
	// inherently nondeterministic between two separate requests — the random
	// response id and the wall-clock created_at. Everything else (event names,
	// ordering, envelope field order, deltas, usage) must be byte-identical.
	var idRe = regexp.MustCompile(`"id":"resp_fak_[0-9a-f]+"`)
	var createdRe = regexp.MustCompile(`"created_at":[0-9]+`)
	mask := func(t *testing.T, raw string) string {
		t.Helper()
		masked := idRe.ReplaceAllString(raw, `"id":"resp_fak_NORMALIZED"`)
		return createdRe.ReplaceAllString(masked, `"created_at":0`)
	}

	fallbackBody := mask(t, render(t, fallbackEvents))
	controlBody := mask(t, render(t, controlEvents))
	if fallbackBody != controlBody {
		t.Fatalf("fallback SSE differs from the buffered synthesis:\n--fallback--\n%s\n--buffered--\n%s", fallbackBody, controlBody)
	}

	// Sanity: the fallback stream still carries the full synthesized shape.
	var names []string
	for _, ev := range fallbackEvents {
		names = append(names, ev.name)
	}
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.output_item.done",
		"response.completed",
	}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("fallback event sequence = %v, want %v", names, want)
	}
	last := decodeResponsesLiveEvent(t, fallbackEvents[len(fallbackEvents)-1].data)
	if last.Status != "completed" || last.OutputText != "Hello" {
		t.Fatalf("fallback response.completed = %+v, want completed/Hello", last)
	}
	if strings.Contains(fallbackBody, "[DONE]") {
		t.Fatal("fallback stream carried a [DONE] sentinel, want none")
	}
}

// decodeResponsesLiveEvent decodes one SSE `data:` payload's embedded response
// object (every Responses event wraps the full response in `response`).
func decodeResponsesLiveEvent(t *testing.T, data string) responsesResponse {
	t.Helper()
	var envelope struct {
		Response responsesResponse `json:"response"`
	}
	if err := json.Unmarshal([]byte(data), &envelope); err != nil {
		t.Fatalf("decode event data: %v: %s", err, data)
	}
	return envelope.Response
}

// findResponsesLiveItem returns the first output item of the given type.
func findResponsesLiveItem(items []responsesOutputItem, typ string) (responsesOutputItem, bool) {
	for _, it := range items {
		if it.Type == typ {
			return it, true
		}
	}
	return responsesOutputItem{}, false
}
