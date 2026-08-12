package generationctl

// anthropic_stream.go decodes the Anthropic `/v1/messages` SSE shape into a
// StreamBridge, one already-parsed event at a time. Taking events rather than
// bytes is deliberate: the gateway's live passthrough already has an SSE
// scanner and a per-event callback, so it wires generation control in by
// forwarding each event here instead of growing a second parser.
//
// This wire is the one that CAN stream tool arguments below the call boundary —
// `input_json_delta` arrives in fragments — so an adapter over it may honestly
// declare delta resolution. The bridge still measures rather than trusts: a turn
// whose arguments happen to arrive whole is reported as tool-call resolution
// even here.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// DecodeAnthropicStream feeds a whole Anthropic `/v1/messages` SSE body into
// the bridge. Live adapters that already own an SSE scanner call AnthropicEvent
// per event instead; this is the entry point for replaying a recorded body.
func DecodeAnthropicStream(b *StreamBridge, r io.Reader) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	event := ""
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		switch {
		case line == "":
			event = ""
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			name := event
			if name == "" {
				// A body with no event: lines still names the type in the frame.
				var typed struct {
					Type string `json:"type"`
				}
				if json.Unmarshal([]byte(payload), &typed) == nil {
					name = typed.Type
				}
			}
			stop, err := AnthropicEvent(b, name, []byte(payload))
			if err != nil {
				return err
			}
			if stop {
				return nil
			}
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("generationctl: read stream: %w", err)
	}
	return nil
}

// AnthropicEvent feeds one Anthropic SSE event into the bridge. It reports
// stop=true once a steering point closed the epoch, which is the caller's
// signal to cancel at its next safe boundary. Events the steering path does not
// care about (message_start, ping, usage-only frames) are ignored, not errors.
func AnthropicEvent(b *StreamBridge, event string, data []byte) (stop bool, err error) {
	if b.Cancelled() {
		return true, nil
	}
	switch event {
	case "content_block_start":
		var d struct {
			Index        int `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
		}
		if json.Unmarshal(data, &d) != nil {
			return false, nil // a frame this build cannot read steers nothing
		}
		if d.ContentBlock.Type != "tool_use" {
			return false, nil
		}
		id := d.ContentBlock.ID
		if id == "" {
			id = fmt.Sprintf("anthropic-tool-index-%d", d.Index)
		}
		if err := b.ToolCallStart(id, d.ContentBlock.Name, ""); err != nil {
			return true, err
		}
		b.BindIndex(d.Index, id)
		return false, nil

	case "content_block_delta":
		var d struct {
			Index int `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Thinking    string `json:"thinking"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if json.Unmarshal(data, &d) != nil {
			return false, nil
		}
		switch d.Delta.Type {
		case "text_delta":
			if _, err := b.Text(d.Delta.Text); err != nil {
				return true, err
			}
		case "thinking_delta":
			if _, err := b.Thinking(d.Delta.Thinking); err != nil {
				return true, err
			}
		case "input_json_delta":
			callID, ok := b.CallIDForIndex(d.Index)
			if !ok {
				// Arguments for a block whose start was never seen. Refusing to
				// invent a call is the fail-closed choice: an unregistered call
				// can never reach ToolCallBoundary as admissible.
				return false, nil
			}
			if _, err := b.ToolArgs(callID, d.Delta.PartialJSON); err != nil {
				return true, err
			}
		}
		return b.Cancelled(), nil

	case "content_block_stop":
		var d struct {
			Index int `json:"index"`
		}
		if json.Unmarshal(data, &d) != nil {
			return false, nil
		}
		if callID, ok := b.CallIDForIndex(d.Index); ok {
			if err := b.SealToolCall(callID); err != nil {
				return true, err
			}
		}
		return false, nil

	case "message_delta", "message_stop":
		b.SealAll()
		return false, nil
	}
	return false, nil
}
