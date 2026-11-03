package gateway

// chat_stream_preamble_test.go — pins the opt-in in-band route-decision preamble on
// the buffered chat stream (lemonade borrow, ADAPT): one SSE comment line before the
// first data chunk carrying the chosen upstream and the stream's announced model, its
// absence when the flag is off, and a byte-identical first data chunk either way.

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=5ms lane=default

const pinnedStreamModel = "qwen3.8-27b"

// newPinnedChatStreamWriter builds a buffered chat stream writer with a FIXED minted
// identity so two independently opened streams serialize to byte-identical frames —
// the comparison the route-decision contract needs (the real constructor mints id from
// the wall clock).
func newPinnedChatStreamWriter(t *testing.T, rec *httptest.ResponseRecorder) *chatStreamWriter {
	t.Helper()
	p := newChatStreamWriter(rec, pinnedStreamModel, false)
	p.id = "chatcmpl-fak-pinned"
	p.created = 1700000000
	return p
}

// firstDataEvent returns the first complete `data:` SSE event (including its blank-line
// terminator) in raw; it fails the test when no data event exists.
func firstDataEvent(t *testing.T, raw string) string {
	t.Helper()
	i := strings.Index(raw, "data: ")
	if i < 0 {
		t.Fatalf("no data event in stream %q", raw)
	}
	j := strings.Index(raw[i:], "\n\n")
	if j < 0 {
		t.Fatalf("first data event unterminated in stream %q", raw)
	}
	return raw[i : i+j+2]
}

// TestStreamFirstChunkCarriesRouteDecision pins the closed wire contract of the opt-in
// route-decision preamble: the exact comment line when enabled, its absence when
// disabled, and a byte-identical first data chunk either way.
func TestStreamFirstChunkCarriesRouteDecision(t *testing.T) {
	const wantComment = ": fak-route upstream=inkernel model=" + pinnedStreamModel + "\n\n"

	t.Run("enabled pins the exact comment line before the first data chunk", func(t *testing.T) {
		t.Setenv(streamRoutePreambleEnv, "inkernel")
		rec := httptest.NewRecorder()
		if err := newPinnedChatStreamWriter(t, rec).open(); err != nil {
			t.Fatalf("open: %v", err)
		}
		raw := rec.Body.String()
		if !strings.HasPrefix(raw, wantComment) {
			t.Fatalf("stream does not open with the pinned route-decision comment:\ngot  prefix=%q\nwant %q",
				raw[:min(len(raw), 96)], wantComment)
		}
		// The comment is a single frame: exactly one blank-terminated line precedes the
		// first data event.
		prefix := raw[:strings.Index(raw, "data: ")]
		if got := strings.Count(prefix, "\n\n"); got != 1 {
			t.Fatalf("%d blank-terminated frames before the first data event, want 1: %q", got, prefix)
		}
		ev := firstDataEvent(t, raw)
		var chunk ChatStreamResponse
		if err := json.Unmarshal([]byte(strings.TrimPrefix(ev, "data: ")), &chunk); err != nil {
			t.Fatalf("first data chunk is not valid JSON: %v (%q)", err, ev)
		}
		if chunk.Model != pinnedStreamModel || chunk.Object != "chat.completion.chunk" ||
			len(chunk.Choices) != 1 || chunk.Choices[0].Delta.Role != agent.RoleAssistant {
			t.Fatalf("first data chunk = %+v, want the opening role chunk announcing %s", chunk, pinnedStreamModel)
		}
		if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
			t.Fatalf("Content-Type = %q, want text/event-stream (the shim must forward headers)", ct)
		}
	})

	t.Run("disabled emits no comment and opens straight into the data chunk", func(t *testing.T) {
		for _, off := range []string{"", "0", "false", "off", "no"} {
			t.Setenv(streamRoutePreambleEnv, off)
			rec := httptest.NewRecorder()
			if err := newPinnedChatStreamWriter(t, rec).open(); err != nil {
				t.Fatalf("open (flag=%q): %v", off, err)
			}
			raw := rec.Body.String()
			if strings.Contains(raw, "fak-route") {
				t.Fatalf("flag=%q: route-decision comment leaked into a disabled stream: %q", off, raw)
			}
			if !strings.HasPrefix(raw, "data: ") {
				t.Fatalf("flag=%q: stream does not open with the first data chunk: %q", off, raw[:min(len(raw), 96)])
			}
		}
	})

	t.Run("first data chunk is byte-identical enabled vs disabled", func(t *testing.T) {
		t.Setenv(streamRoutePreambleEnv, "inkernel")
		recOn := httptest.NewRecorder()
		if err := newPinnedChatStreamWriter(t, recOn).open(); err != nil {
			t.Fatalf("open (enabled): %v", err)
		}
		t.Setenv(streamRoutePreambleEnv, "")
		recOff := httptest.NewRecorder()
		if err := newPinnedChatStreamWriter(t, recOff).open(); err != nil {
			t.Fatalf("open (disabled): %v", err)
		}
		on := firstDataEvent(t, recOn.Body.String())
		off := firstDataEvent(t, recOff.Body.String())
		if on != off {
			t.Fatalf("first data chunk differs with the preamble enabled:\nenabled  %q\ndisabled %q", on, off)
		}
	})

	t.Run("flag value cannot break out of the single-line comment frame", func(t *testing.T) {
		t.Setenv(streamRoutePreambleEnv, "bad\r\nupstream: x\ninkernel")
		rec := httptest.NewRecorder()
		if err := newPinnedChatStreamWriter(t, rec).open(); err != nil {
			t.Fatalf("open: %v", err)
		}
		raw := rec.Body.String()
		want := ": fak-route upstream=badupstream: xinkernel model=" + pinnedStreamModel + "\n\n"
		if !strings.HasPrefix(raw, want) {
			t.Fatalf("control characters were not stripped from the comment frame:\ngot  prefix=%q\nwant %q",
				raw[:min(len(raw), 96)], want)
		}
		prefix := raw[:strings.Index(raw, "data: ")]
		if got := strings.Count(prefix, "\n\n"); got != 1 {
			t.Fatalf("%d blank-terminated frames before the first data event, want 1: %q", got, prefix)
		}
	})
}
