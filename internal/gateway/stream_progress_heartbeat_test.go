package gateway

// stream_progress_heartbeat_test.go — #10672: typed progress heartbeats on the OpenAI
// live-stream wire. A minutes-long stream is today indistinguishable from a hang from the
// outside: no frame carries any evidence the turn is advancing. During a long stream the
// gateway now emits a periodic, content-free heartbeat — an SSE comment frame (spec-legal
// in any position between events, dropped by conforming parsers, visible to any operator
// sniffing the wire) carrying counts and durations only: elapsed, phase, bytes/events
// emitted, last-event age. NO prompt or output content ever appears in one.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// hbBudget bounds every wait in this file: long enough that an in-process flush can never
// miss it, short enough that a missing heartbeat fails the test instead of hanging it.
const hbBudget = 4 * time.Second

// hbProbePlanner models a live-streaming upstream whose turn takes a controlled amount of
// wall clock. It emits firstFragment through the sink, parks for pause (the minutes-long
// "model is streaming" window the heartbeat exists to light up — ended by the pause
// elapsing OR the test releasing early), then either finishes the turn or fails mid-stream
// with failAfterPause. With parkFirst set it parks BEFORE its first fragment — the
// pre-first-token window where the gateway has not yet committed a response and must not
// write a heartbeat.
type hbProbePlanner struct {
	model          string
	firstFragment  string
	secondFragment string
	pause          time.Duration
	failAfterPause error
	parkFirst      bool

	joinedOnce  sync.Once
	releaseOnce sync.Once
	joined      chan struct{} // closed the moment the first fragment was sunk
	release     chan struct{} // closed by the test to end the modeled pause early
}

func newHBProbePlanner(model, first, second string, pause time.Duration, failAfterPause error) *hbProbePlanner {
	return &hbProbePlanner{
		model: model, firstFragment: first, secondFragment: second,
		pause: pause, failAfterPause: failAfterPause,
		joined: make(chan struct{}), release: make(chan struct{}),
	}
}

func (p *hbProbePlanner) releasePause() { p.releaseOnce.Do(func() { close(p.release) }) }

func (p *hbProbePlanner) Model() string { return p.model }

func (p *hbProbePlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("hbProbePlanner models a stream-only planner")
}

func (p *hbProbePlanner) StreamingSupported() bool { return true }

// park waits out the modeled pause: released early by the test, cut by the client, or
// ended by the pause itself — never unbounded.
func (p *hbProbePlanner) park(ctx context.Context) error {
	if p.parkFirst {
		// The pre-first-token park is release-driven only: the test must own the window.
		select {
		case <-p.release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(30 * time.Second):
			return errors.New("modeled pre-first-token pause was never released")
		}
	}
	t := time.NewTimer(p.pause)
	defer t.Stop()
	select {
	case <-p.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func (p *hbProbePlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if p.parkFirst {
		if err := p.park(ctx); err != nil {
			return nil, err
		}
	}
	if err := sink(p.firstFragment); err != nil {
		return nil, err
	}
	p.joinedOnce.Do(func() { close(p.joined) })
	if !p.parkFirst {
		if err := p.park(ctx); err != nil {
			return nil, err
		}
	}
	if p.failAfterPause != nil {
		return nil, p.failAfterPause
	}
	if err := sink(p.secondFragment); err != nil {
		return nil, err
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: p.firstFragment + p.secondFragment},
		FinishReason: "stop",
		Model:        p.model,
	}, nil
}

var _ agent.StreamingPlanner = (*hbProbePlanner)(nil)

// hbFrame is one decoded heartbeat comment line: `: fak-heartbeat {json}`.
type hbFrame struct {
	ElapsedMS      int64  `json:"elapsed_ms"`
	Phase          string `json:"phase"`
	BytesEmitted   int64  `json:"bytes_emitted"`
	EventsEmitted  int64  `json:"events_emitted"`
	LastEventAgeMS int64  `json:"last_event_age_ms"`
}

func decodeHeartbeat(t *testing.T, line string) hbFrame {
	t.Helper()
	raw, ok := strings.CutPrefix(line, ": fak-heartbeat ")
	if !ok {
		t.Fatalf("line is not a fak-heartbeat comment: %q", line)
	}
	var hb hbFrame
	if err := json.Unmarshal([]byte(raw), &hb); err != nil {
		t.Fatalf("decode heartbeat %q: %v", raw, err)
	}
	return hb
}

// heartbeatJSON re-renders the decoded heartbeat compactly for a content-leak check.
func heartbeatJSON(t *testing.T, hb hbFrame) string {
	t.Helper()
	b, err := json.Marshal(hb)
	if err != nil {
		t.Fatalf("marshal heartbeat: %v", err)
	}
	return string(b)
}

// waitHeartbeat reads frames until the first (or next) heartbeat comment arrives.
// It returns the heartbeat line and any data frames encountered (which the caller
// must prepend to the drained content to avoid losing the first fragment).
func waitHeartbeat(t *testing.T, tap *sseTap, stallMsg string, dataFrames *[]string) string {
	t.Helper()
	for i := 0; i < 32; i++ {
		line := tap.waitFrame(t, hbBudget, stallMsg)
		if strings.HasPrefix(line, ": fak-heartbeat") {
			return line
		}
		if strings.HasPrefix(line, "data:") {
			*dataFrames = append(*dataFrames, line)
		}
	}
	t.Fatal(stallMsg)
	return ""
}

// TestStreamChatLiveEmitsTypedHeartbeatsDuringLongStream is the #10672 heartbeat witness on
// the OpenAI live wire: while the modeled upstream sits in a long post-first-token pause,
// the client wire must carry typed heartbeats proving the turn is alive and ADVANCING
// (phase mid_stream, bytes/events emitted, last-event age) — and none of them may carry a
// single byte of the streamed content.
func TestStreamChatLiveEmitsTypedHeartbeatsDuringLongStream(t *testing.T) {
	t.Setenv("FAK_STREAM_HEARTBEAT_S", "1")
	const contentMark = "HEARTBEAT-CONTENT-MARKER "
	planner := newHBProbePlanner("test-model", contentMark, "tail", 3200*time.Millisecond, nil)

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// LIFO cleanup: end the modeled pause before the HTTP server closes its connections.
	t.Cleanup(planner.releasePause)

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"probe"}],"stream":true}`)
	tap := tapChatStream(ts.URL+"/v1/chat/completions", body)

	resp := tap.waitHead(t, hbBudget, "no HTTP head — the stream never opened")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	opening := decodeSSEChunk(t, tap.waitFrame(t, hbBudget, "no opening role chunk"))
	if got := opening.Choices[0].Delta.Role; got != agent.RoleAssistant {
		t.Fatalf("opening delta role = %q, want assistant", got)
	}

	// The pause is real at this point: the first fragment is sunk, the modeled upstream is
	// parked, and only the heartbeat can speak for the turn until the pause ends. The first
	// fragment's own data chunk is already in flight ahead of the pause — collect it.
	var hbDataFrames []string
	first := decodeHeartbeat(t, waitHeartbeat(t, tap,
		"no heartbeat frame arrived during a 3.2s live-stream pause with FAK_STREAM_HEARTBEAT_S=1 — a long stream is still indistinguishable from a hang (#10672)", &hbDataFrames))
	if first.Phase != "mid_stream" {
		t.Fatalf("heartbeat phase = %q, want mid_stream — the first token already landed", first.Phase)
	}
	if first.ElapsedMS <= 0 {
		t.Fatalf("heartbeat elapsed_ms = %d, want > 0", first.ElapsedMS)
	}
	if first.BytesEmitted < int64(len(contentMark)) {
		t.Fatalf("heartbeat bytes_emitted = %d, want >= %d (the first fragment was already streamed)", first.BytesEmitted, len(contentMark))
	}
	if first.EventsEmitted < 1 {
		t.Fatalf("heartbeat events_emitted = %d, want >= 1", first.EventsEmitted)
	}

	// Heartbeats are counts and durations ONLY: the streamed content must never appear in one.
	if strings.Contains(heartbeatJSON(t, first), contentMark) {
		t.Fatal("heartbeat carried streamed content — heartbeats must be content-free")
	}

	// A second heartbeat during the same pause must show ADVANCING elapsed and last-event
	// age (the fragment landed before the pause), proving the fields are live signals.
	second := decodeHeartbeat(t, waitHeartbeat(t, tap, "no second heartbeat during the pause", &hbDataFrames))
	if second.ElapsedMS <= first.ElapsedMS {
		t.Fatalf("second heartbeat elapsed_ms = %d, want > first %d — elapsed must advance", second.ElapsedMS, first.ElapsedMS)
	}
	if second.LastEventAgeMS <= first.LastEventAgeMS {
		t.Fatalf("second heartbeat last_event_age_ms = %d, want > first %d — the modeled pause means no new events between heartbeats, so the age must grow", second.LastEventAgeMS, first.LastEventAgeMS)
	}

	// Release: the turn completes and the stream must end well-formed — heartbeats must not
	// have corrupted the SSE framing (every data line still parses as a chat chunk).
	planner.releasePause()
	rest := tap.drain(t, 15*time.Second)
	// Prepend any data frames consumed during heartbeat wait (the first fragment).
	rest = append(hbDataFrames, rest...)
	var content strings.Builder
	sawDone := false
	for _, line := range rest {
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		if strings.HasPrefix(line, ": fak-heartbeat") {
			continue
		}
		chunk := decodeSSEChunk(t, line)
		content.WriteString(chunk.Choices[0].Delta.Content)
	}
	if got := content.String(); got != contentMark+"tail" {
		t.Fatalf("reassembled content = %q, want %q — the heartbeat frames corrupted the stream", got, contentMark+"tail")
	}
	if !sawDone {
		t.Fatal("stream never terminated with [DONE] after the heartbeats")
	}
}

// TestStreamChatLiveHeartbeatSilentBeforeFirstToken pins the wire-position rule: before the
// 200/SSE headers are committed (nothing on the wire yet) the heartbeat must NOT write — a
// premature write would break the "clean HTTP error before any byte" contract. While the
// modeled upstream parks BEFORE its first fragment, the client sees NOTHING at all: no
// head, no heartbeat comment, no chunk. After the park ends the turn completes normally.
func TestStreamChatLiveHeartbeatSilentBeforeFirstToken(t *testing.T) {
	t.Setenv("FAK_STREAM_HEARTBEAT_S", "1")
	silent := newHBProbePlanner("test-model", "late ", "tail", 2500*time.Millisecond, nil)
	silent.parkFirst = true

	srv := newTestServer(t)
	srv.planner = silent
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	t.Cleanup(silent.releasePause)

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"probe"}],"stream":true}`)
	tap := tapChatStream(ts.URL+"/v1/chat/completions", body)

	// Parked pre-first-token with the heartbeat ticker running: for a full heartbeat
	// interval + margin NOTHING may reach the client — the response is not committed yet,
	// so any early write (a heartbeat, a header) would forfeit the clean-error contract.
	silentWindow := time.After(hbBudget)
	select {
	case h := <-tap.head:
		if h.err == nil {
			t.Fatalf("HTTP head reached the client before the first token (status %d) — a heartbeat or chunk was written pre-commit", h.resp.StatusCode)
		}
		t.Fatalf("request failed during the pre-first-token park: %v", h.err)
	case line, ok := <-tap.frames:
		if !ok {
			t.Fatal("stream closed during the pre-first-token park")
		}
		t.Fatalf("a frame reached the client before the first token: %q", line)
	case <-silentWindow:
		silent.releasePause()
	}

	// The turn completes normally once the park ends: well-formed head, role chunk, content.
	resp := tap.waitHead(t, hbBudget, "no HTTP head after the pre-first-token park ended")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	opening := decodeSSEChunk(t, tap.waitFrame(t, hbBudget, "no opening role chunk after the park"))
	if got := opening.Choices[0].Delta.Role; got != agent.RoleAssistant {
		t.Fatalf("opening delta role = %q, want assistant", got)
	}
	rest := tap.drain(t, 15*time.Second)
	var content strings.Builder
	sawDone := false
	for _, line := range rest {
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		if strings.HasPrefix(line, ": fak-heartbeat") {
			t.Fatalf("heartbeat on the wire after first token on the parkFirst stream: %q", line)
		}
		chunk := decodeSSEChunk(t, line)
		content.WriteString(chunk.Choices[0].Delta.Content)
	}
	if got := content.String(); got != "late tail" {
		t.Fatalf("reassembled content = %q, want %q", got, "late tail")
	}
	if !sawDone {
		t.Fatal("stream never terminated with [DONE]")
	}
}
