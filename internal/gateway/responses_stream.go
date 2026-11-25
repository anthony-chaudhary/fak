package gateway

import (
	"encoding/json"
	"io"
	"net/http"
)

// writeResponsesStream synthesizes a well-formed Responses SSE stream from a
// buffered response, matching the request's stream flag. It emits the sequence:
// response.created → response.output_item.added (per item) →
// response.output_text.delta/done (for message items) → response.output_item.done
// (per item) → response.completed. This is the synthesized-stream analogue of
// writeChatCompletionStream: the gateway buffers the entire turn, adjudicates the
// proposed tool calls, then re-serializes the adjudicated turn as SSE.
func (s *Server) writeResponsesStream(w http.ResponseWriter, resp responsesResponse) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	nextSeq := 0
	type responseEvent struct {
		Type           string            `json:"type"`
		SequenceNumber int               `json:"sequence_number"`
		Response       responsesResponse `json:"response"`
	}
	writeResponseEvent := func(event string, response responsesResponse) {
		_ = writeSSEEvent(w, event, responseEvent{
			Type:           event,
			SequenceNumber: nextSeq,
			Response:       response,
		})
		nextSeq++
	}

	// response.created: initial event with response metadata. The generated turn is
	// already buffered, but the stream envelope still mirrors the live Responses API.
	created := resp
	created.Status = "in_progress"
	created.Output = []responsesOutputItem{}
	created.OutputText = ""
	created.Usage = responsesUsage{}
	writeResponseEvent("response.created", created)

	// Emit each output item: added → done
	for i, item := range resp.Output {
		// response.output_item.added
		type outputItemEvent struct {
			Type           string              `json:"type"`
			SequenceNumber int                 `json:"sequence_number"`
			OutputIndex    int                 `json:"output_index"`
			Item           responsesOutputItem `json:"item"`
		}
		_ = writeSSEEvent(w, "response.output_item.added", outputItemEvent{
			Type:           "response.output_item.added",
			SequenceNumber: nextSeq,
			OutputIndex:    i,
			Item:           item,
		})
		nextSeq++

		if item.Type == "message" {
			type outputTextDeltaEvent struct {
				Type           string `json:"type"`
				SequenceNumber int    `json:"sequence_number"`
				OutputIndex    int    `json:"output_index"`
				ContentIndex   int    `json:"content_index"`
				Delta          string `json:"delta"`
			}
			type outputTextDoneEvent struct {
				Type           string `json:"type"`
				SequenceNumber int    `json:"sequence_number"`
				OutputIndex    int    `json:"output_index"`
				ContentIndex   int    `json:"content_index"`
				Text           string `json:"text"`
			}
			for cIdx, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					_ = writeSSEEvent(w, "response.output_text.delta", outputTextDeltaEvent{
						Type:           "response.output_text.delta",
						SequenceNumber: nextSeq,
						OutputIndex:    i,
						ContentIndex:   cIdx,
						Delta:          part.Text,
					})
					nextSeq++
					_ = writeSSEEvent(w, "response.output_text.done", outputTextDoneEvent{
						Type:           "response.output_text.done",
						SequenceNumber: nextSeq,
						OutputIndex:    i,
						ContentIndex:   cIdx,
						Text:           part.Text,
					})
					nextSeq++
				}
			}
		} else if item.Type == "function_call" {
			type functionCallArgsDeltaEvent struct {
				Type           string `json:"type"`
				SequenceNumber int    `json:"sequence_number"`
				OutputIndex    int    `json:"output_index"`
				CallID         string `json:"call_id"`
				Delta          string `json:"delta"`
			}
			type functionCallArgsDoneEvent struct {
				Type           string `json:"type"`
				SequenceNumber int    `json:"sequence_number"`
				OutputIndex    int    `json:"output_index"`
				CallID         string `json:"call_id"`
				Arguments      string `json:"arguments"`
			}
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			if item.Arguments != "" {
				_ = writeSSEEvent(w, "response.function_call_arguments.delta", functionCallArgsDeltaEvent{
					Type:           "response.function_call_arguments.delta",
					SequenceNumber: nextSeq,
					OutputIndex:    i,
					CallID:         callID,
					Delta:          item.Arguments,
				})
				nextSeq++
				_ = writeSSEEvent(w, "response.function_call_arguments.done", functionCallArgsDoneEvent{
					Type:           "response.function_call_arguments.done",
					SequenceNumber: nextSeq,
					OutputIndex:    i,
					CallID:         callID,
					Arguments:      item.Arguments,
				})
				nextSeq++
			}
		}

		// response.output_item.done
		_ = writeSSEEvent(w, "response.output_item.done", outputItemEvent{
			Type:           "response.output_item.done",
			SequenceNumber: nextSeq,
			OutputIndex:    i,
			Item:           item,
		})
		nextSeq++
	}

	// Terminal event with optional fak extension and incomplete details. The event NAME
	// mirrors the response status: an `incomplete` response (length truncation, guard block,
	// or the sub-turn yield valve #11764) terminates the stream with `response.incomplete`,
	// so a client whose planner never reached a genuine finish cannot read a fabricated
	// `response.completed` as a finished model answer. Codex 0.142.4 treats a stream that
	// closes before a terminal event as incomplete and retries.
	terminalEvent := "response.completed"
	if resp.Status == "incomplete" {
		terminalEvent = "response.incomplete"
	}
	writeResponseEvent(terminalEvent, resp)

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// writeSSEEvent writes a single SSE event with a typed event name. The Responses
// wire uses named event types (response.created, response.completed) rather than the
// generic data: frames the chat wire uses.
func writeSSEEvent(w http.ResponseWriter, event string, data interface{}) error {
	raw, err := json.Marshal(data)
	if err != nil {
		return err
	}
	for _, line := range []string{"event: " + event + "\n", "data: " + string(raw) + "\n\n"} {
		n, err := io.WriteString(w, line)
		if err != nil {
			return err
		}
		if n != len(line) {
			return io.ErrShortWrite
		}
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}
