package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestStreamPromptProgressValidation(t *testing.T) {
	valid := json.RawMessage(`{"total":100,"cache":10,"processed":20,"time_ms":1}`)
	for _, request := range []string{`{}`, `{"return_progress":false}`, `{"return_progress":null}`, `{"return_progress":"true"}`} {
		s := newStreamPromptProgress([]byte(request))
		if s.advance(valid) {
			t.Fatalf("unconfigured request accepted progress: %s", request)
		}
	}
	for _, raw := range []string{
		`null`, `{}`, `[]`, `{"total":100,"cache":10,"processed":20}`,
		`{"total":null,"cache":10,"processed":20,"time_ms":1}`,
		`{"total":100,"cache":null,"processed":20,"time_ms":1}`,
		`{"total":100,"cache":10,"processed":null,"time_ms":1}`,
		`{"total":100,"cache":10,"processed":20,"time_ms":null}`,
		`{"total":100,"cache":10,"processed":"20","time_ms":1}`,
		`{"total":100,"cache":10,"processed":20.5,"time_ms":1}`,
		`{"total":0,"cache":0,"processed":0,"time_ms":1}`,
		`{"total":2147483648,"cache":10,"processed":20,"time_ms":1}`,
		`{"total":100,"cache":-1,"processed":20,"time_ms":1}`,
		`{"total":100,"cache":10,"processed":9,"time_ms":1}`,
		`{"total":100,"cache":10,"processed":101,"time_ms":1}`,
		`{"total":100,"cache":10,"processed":20,"time_ms":-1}`,
		`{"total":100,"cache":10,"processed":20,"time_ms":1.5}`,
	} {
		s := newStreamPromptProgress([]byte(`{"return_progress":true}`))
		if s.advance(json.RawMessage(raw)) || s.observed {
			t.Fatalf("invalid progress altered tracker: %s", raw)
		}
	}
	s := newStreamPromptProgress([]byte(`{"return_progress":true}`))
	for _, step := range []struct {
		processed, elapsed int
		want               bool
	}{{10, 0, false}, {20, 1, true}, {20, 2, false}, {15, 3, false}, {25, 3, true}, {100, 4, true}, {100, 5, false}} {
		raw := fmt.Sprintf(`{"total":100,"cache":10,"processed":%d,"time_ms":%d}`, step.processed, step.elapsed)
		if got := s.advance(json.RawMessage(raw)); got != step.want {
			t.Fatalf("processed=%d elapsed=%d: got %t want %t", step.processed, step.elapsed, got, step.want)
		}
	}
	for _, changed := range []string{
		`{"total":200,"cache":10,"processed":30,"time_ms":2}`,
		`{"total":100,"cache":0,"processed":30,"time_ms":2}`,
	} {
		s := newStreamPromptProgress([]byte(`{"return_progress":true}`))
		if !s.advance(valid) || s.advance(json.RawMessage(changed)) || s.enabled || s.advance(json.RawMessage(`{"total":100,"cache":10,"processed":30,"time_ms":3}`)) {
			t.Fatalf("tuple change permitted a new baseline or replay: %s", changed)
		}
	}
}

// This real HTTP/CompleteStream witness uses only existing public APIs, so it
// can also expose the dropped-progress defect with original production source.
// fak-test:runtime fast est=100ms lane=default
func TestCompleteStreamPromptProgress(t *testing.T) {
	for _, config := range []string{"omitted", "false", "true"} {
		t.Run(config, func(t *testing.T) {
			captured := make(chan []byte, 1)
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
				if err != nil {
					t.Errorf("read request: %v", err)
				}
				captured <- body
				w.Header().Set("Content-Type", "text/event-stream")
				for _, p := range []string{
					`{"total":100,"cache":10,"processed":10,"time_ms":0}`,
					`{"total":100,"cache":10,"processed":20,"time_ms":1}`,
					`{"total":100,"cache":10,"processed":20,"time_ms":2}`,
					`{"total":100,"cache":10,"processed":15,"time_ms":3}`,
					`{"total":100,"cache":10,"processed":"bad","time_ms":4}`,
					`{"total":100,"cache":10,"processed":25,"time_ms":5}`,
				} {
					fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"prompt_progress\":%s}\n\n", p)
				}
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
				// Prefill receipts after actual output cannot extend the turn.
				fmt.Fprint(w, "data: {\"prompt_progress\":{\"total\":100,\"cache\":10,\"processed\":90,\"time_ms\":6}}\n\n")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":1,\"total_tokens\":101}}\n\ndata: [DONE]\n\n")
			}))
			defer up.Close()
			p := NewHTTPPlanner(up.URL, "model", "")
			if config != "omitted" {
				if err := p.SetExtraBodyJSON(`{"return_progress":` + config + `}`); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			held, boundaries := 0, 0
			ctx = WithStreamHeldActivity(ctx, func() { held++ })
			ctx = WithSerialStreamBoundary(ctx, func() { boundaries++ })
			var output strings.Builder
			comp, err := p.CompleteStream(ctx, func(part string) error { output.WriteString(part); return nil }, []Message{{Role: RoleUser, Content: "hi"}}, nil)
			if err != nil || comp == nil {
				t.Fatalf("stream: %v, %v", comp, err)
			}
			wantHeld := 0
			if config == "true" {
				wantHeld = 2
			}
			if held != wantHeld || boundaries != wantHeld || output.String() != "ok" || comp.Message.Content != "ok" || comp.FinishReason != "stop" || comp.Usage.PromptTokens != 100 || comp.Usage.CompletionTokens != 1 {
				t.Fatalf("configured=%s held=%d boundaries=%d want=%d content=%q completion=%+v", config, held, boundaries, wantHeld, output.String(), comp)
			}
			var sent map[string]json.RawMessage
			if err := json.Unmarshal(<-captured, &sent); err != nil {
				t.Fatal(err)
			}
			want := config
			if config == "omitted" {
				want = ""
			}
			if string(sent["return_progress"]) != want || string(sent["stream"]) != "true" {
				t.Fatalf("request options changed: %v", sent)
			}
		})
	}
}
