// Streaming chat surface for the OpenAI-compatible /v1/chat/completions SSE
// stream the fak gateway serves (internal/gateway stream:true path, role-first
// opening chunk, data: [DONE] terminator).
//
// Unlike the verdict surface in client.go, a chat completion can legitimately
// take longer than the SDK's default 30s client timeout to finish arriving;
// StreamChatCompletions therefore issues its request on a per-call copy of the
// underlaying *http.Client with the total deadline zeroed, so the context is
// the only time bound. A mid-stream disconnect is NEVER folded into a
// successful result: it surfaces as *StreamInterruptedError carrying the
// fragments already delivered (issue #12767).
package fakclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// StreamDoneToken is the sentinel SSE payload the terminator of a Chat
// Completions stream is wrapped in: a final `data: [DONE]` line. Its presence
// is the only proof the server finished cleanly; a stream that ends without it
// is reported as interrupted, never as success.
const StreamDoneToken = "[DONE]"

// ErrStreamInterrupted is the sentinel every *StreamInterruptedError unwraps to,
// so a caller can distinguish "the stream did not run to [DONE]" from any other
// fault with errors.Is(err, ErrStreamInterrupted).
var ErrStreamInterrupted = errors.New("fak: stream interrupted before [DONE]")

// StreamMessage is one entry of the chat transcript sent to the gateway.
type StreamMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
}

// StreamToolFunction is the function shape of a tool the model may call,
// mirroring the OpenAI function schema the gateway proxies as-is.
type StreamToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// StreamTool declares one callable tool advertised to the model alongside the
// conversation.
type StreamTool struct {
	Type     string             `json:"type"`
	Function StreamToolFunction `json:"function"`
}

// StreamChatRequest is the body of StreamChatCompletions. Callers never set
// stream themselves — the method stamps `stream: true` internally, since the
// only supported path is the streamed one.
type StreamChatRequest struct {
	Model       string          `json:"model"`
	Messages    []StreamMessage `json:"messages"`
	Tools       []StreamTool    `json:"tools,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stop        json.RawMessage `json:"stop,omitempty"`

	// stream is always set to true by StreamChatCompletions, never by callers —
	// the only supported path is the streamed one. Left tagless because an
	// unexported field cannot be JSON-serialized; the method marshals it onto
	// the wire explicitly.
	stream bool
}

// StreamToolCall is one assembled tool invocation the model emitted: the
// gateway streams it as per-index argument fragments, which the client buffers
// and joins before surfacing. Arguments is the raw JSON object text.
type StreamToolCall struct {
	Index     int    `json:"index"`
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// StreamUsage is the token accounting the final chunk of a usage-enabled stream
// carries.
type StreamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatResult is the fully assembled outcome of one streamed completion: the
// content fragments concatenated in arrival order and the buffered tool calls
// ordered by their stream index.
type ChatResult struct {
	ID           string           `json:"id"`
	Object       string           `json:"object"`
	Created      int64            `json:"created"`
	Model        string           `json:"model"`
	Content      string           `json:"content"`
	ToolCalls    []StreamToolCall `json:"tool_calls,omitempty"`
	FinishReason string           `json:"finish_reason"`
	Usage        *StreamUsage     `json:"usage,omitempty"`
}

// StreamInterruptedError is returned when a stream ends before the gateway
// emitted data: [DONE] — a transport read failure or a bare server close. It
// carries every content fragment already handed to the caller's onDelta, so a
// retrying caller knows exactly what it has shown downstream.
type StreamInterruptedError struct {
	FragmentsDelivered []string
	Err                error
}

// Error reports the interruption with the delivered-fragment count, the number
// a caller needs to reason about partial output.
func (e *StreamInterruptedError) Error() string {
	return fmt.Sprintf("fak: stream interrupted after %d fragment(s) delivered: %v",
		len(e.FragmentsDelivered), e.Err)
}

// Unwrap pins the cause to ErrStreamInterrupted so errors.Is matches the
// sentinel; the underlying transport error is described in Error.
func (e *StreamInterruptedError) Unwrap() error { return ErrStreamInterrupted }

// StreamChatCompletions POSTs req to /v1/chat/completions with stream:true and
// consumes the SSE body, invoking onDelta with each content fragment the moment
// it arrives (onDelta may be nil to discard incremental text). It returns the
// assembled ChatResult after the gateway's data: [DONE] terminator. Anything
// short of that terminator — a read error or a bare EOF — returns a
// *StreamInterruptedError instead of a partial result, so a disconnect can
// never masquerade as a completed answer (issue #12767).
func (c *Client) StreamChatCompletions(ctx context.Context, req StreamChatRequest, onDelta func(frag string)) (*ChatResult, error) {
	// stream is unexported (callers never set it), so json.Marshal on the bare
	// request would drop it; marshal through a promoting wrapper instead.
	req.stream = true
	body, err := json.Marshal(struct {
		StreamChatRequest
		Stream bool `json:"stream"`
	}{StreamChatRequest: req, Stream: req.stream})
	if err != nil {
		return nil, fmt.Errorf("fak: marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if c.principal != "" {
		httpReq.Header.Set("X-Fak-Principal", c.principal)
	}
	// Streaming must not inherit the 30s default timeout: copy the client and
	// zero the total deadline so a long stream survives; ctx governs instead.
	hc := *c.httpClient
	hc.Timeout = 0
	resp, err := (&hc).Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
		return nil, parseAPIError(resp.StatusCode, data)
	}
	return consumeChatStream(resp, onDelta)
}

// consumeChatStream drains the SSE body: one text frame per "data:" line, a
// role-first opening chunk, then content / tool-call deltas, terminated by
// data: [DONE].
func consumeChatStream(resp *http.Response, onDelta func(frag string)) (*ChatResult, error) {
	res := &ChatResult{}
	var fragments []string
	var delivered []string
	toolCalls := map[int]*StreamToolCall{}
	var order []int
	sawDone := false

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == StreamDoneToken {
			sawDone = true
			break
		}
		if payload == "" {
			continue
		}
		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return nil, &APIError{
				StatusCode: resp.StatusCode,
				Type:       "stream_decode_error",
				Message:    fmt.Sprintf("undecodable SSE payload: %v", err),
			}
		}
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			// A role-first opening chunk (role set, no content) is an
			// announcement only and is intentionally not surfaced.
			if delta.Content != "" {
				if onDelta != nil {
					onDelta(delta.Content)
				}
				fragments = append(fragments, delta.Content)
				delivered = append(delivered, delta.Content)
			}
			for _, tc := range delta.ToolCalls {
				entry, ok := toolCalls[tc.Index]
				if !ok {
					entry = &StreamToolCall{
						Index:     tc.Index,
						ID:        tc.ID,
						Type:      tc.Type,
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					}
					toolCalls[tc.Index] = entry
					order = append(order, tc.Index)
					continue
				}
				entry.Arguments += tc.Function.Arguments
				if tc.ID != "" {
					entry.ID = tc.ID
				}
				if tc.Type != "" {
					entry.Type = tc.Type
				}
				if tc.Function.Name != "" {
					entry.Name = tc.Function.Name
				}
			}
			if fr := chunk.Choices[0].FinishReason; fr != nil && *fr != "" {
				res.FinishReason = *fr
			}
		}
		if chunk.ID != "" && res.ID == "" {
			res.ID = chunk.ID
		}
		if chunk.Object != "" && res.Object == "" {
			res.Object = chunk.Object
		}
		if chunk.Created != 0 && res.Created == 0 {
			res.Created = chunk.Created
		}
		if chunk.Model != "" && res.Model == "" {
			res.Model = chunk.Model
		}
		if chunk.Usage != nil {
			res.Usage = chunk.Usage
		}
	}
	if err := sc.Err(); err != nil {
		return nil, &StreamInterruptedError{FragmentsDelivered: delivered, Err: err}
	}
	if !sawDone {
		// An unterminated stream is an interruption, never silent success.
		return nil, &StreamInterruptedError{FragmentsDelivered: delivered, Err: io.EOF}
	}
	res.Content = strings.Join(fragments, "")
	for _, idx := range order {
		res.ToolCalls = append(res.ToolCalls, *toolCalls[idx])
	}
	return res, nil
}

// chatStreamChunk mirrors the gateway's ChatStreamResponse (internal/gateway
// wire.go) field-for-field; tool-call arguments arrive incrementally as raw
// JSON string text and are kept as-is until assembly.
type chatStreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *StreamUsage `json:"usage"`
}
