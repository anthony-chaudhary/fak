// Package demoutil holds the small server-sent-events scaffolding shared by the
// browser-facing demo binaries (cmd/ctxdemo, cmd/demorace): both stream the same
// JSON-object events to their viewer over an identical text/event-stream writer.
// Keeping one copy here is the behavior-preserving de-dup of the per-demo sseWriter.
package demoutil

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// Event is one server-sent-event payload: a JSON object the demo streams to the
// viewer (each demo emits its own "type"-tagged shapes).
type Event = map[string]any

// Emitter consumes Events as a demo arm produces them (the headless -race path
// passes a printing emitter; the HTTP path passes SSEWriter.Emit).
type Emitter = func(Event)

// SSEWriter streams Events to an HTTP client as text/event-stream "data:" frames.
type SSEWriter struct {
	W       http.ResponseWriter
	Flusher http.Flusher
}

// Emit marshals e and writes it as one SSE frame, then flushes so the viewer sees
// each event as it happens.
func (s *SSEWriter) Emit(e Event) {
	b, _ := json.Marshal(e)
	fmt.Fprintf(s.W, "data: %s\n\n", b)
	s.Flusher.Flush()
}
