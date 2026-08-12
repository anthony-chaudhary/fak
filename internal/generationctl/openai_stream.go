package generationctl

// openai_stream.go decodes the OpenAI-compatible `chat.completion.chunk` SSE
// shape into a StreamBridge. It is the adapter half for every provider that
// speaks that wire (the fak gateway's own OpenAI surface, Groq, NVIDIA NIM,
// vLLM, and friends).
//
// A note the captures under testdata/captures back up: this wire carries a
// tool call's arguments in `delta.tool_calls[].function.arguments`, and the
// providers recorded there send that whole object in ONE chunk even though the
// prose arrives token by token. The decoder does not paper over that — it feeds
// exactly the fragments it received, so the bridge measures tool-call
// resolution and the adapter's declaration is judged against it.

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// openAIChunk is the subset of a chat.completion.chunk this decoder steers on.
type openAIChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning"`
			// Some OpenAI-compatible providers spell the reasoning channel
			// `reasoning_content`; both mean "not part of the durable prefix".
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error json.RawMessage `json:"error"`
}

// ErrProviderRefused reports that the stream carried an in-band provider error
// frame instead of a turn. It is deliberately distinct from a decode failure:
// nothing is wrong with the adapter, the provider declined.
var ErrProviderRefused = errors.New("generationctl: provider returned an in-band error frame")

// DecodeOpenAIStream feeds an OpenAI-compatible SSE body into the bridge in
// arrival order and returns when the stream ends, when the provider refuses, or
// as soon as a steering point closes the epoch — the last case being the cancel
// signal the caller acts on at its next safe boundary. A closed epoch is not an
// error: check bridge.Cancelled().
func DecodeOpenAIStream(b *StreamBridge, r io.Reader) error {
	sc := bufio.NewScanner(r)
	// Tool arguments and reasoning blocks routinely exceed the 64 KiB default,
	// and a truncated line would silently become a different tool call.
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}
		var chunk openAIChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return fmt.Errorf("generationctl: decode chunk: %w", err)
		}
		if len(chunk.Error) != 0 {
			return fmt.Errorf("%w: %s", ErrProviderRefused, string(chunk.Error))
		}
		stop, err := feedOpenAIChunk(b, chunk)
		if err != nil {
			return err
		}
		if stop {
			return nil
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("generationctl: read stream: %w", err)
	}
	return nil
}

// feedOpenAIChunk applies one chunk. It reports stop=true once the epoch closed,
// so the caller stops reading upstream rather than feeding a closed controller.
func feedOpenAIChunk(b *StreamBridge, chunk openAIChunk) (bool, error) {
	for _, choice := range chunk.Choices {
		d := choice.Delta
		if reasoning := firstNonEmpty(d.Reasoning, d.ReasoningContent); reasoning != "" {
			if _, err := b.Thinking(reasoning); err != nil {
				return true, err
			}
			if b.Cancelled() {
				return true, nil
			}
		}
		if d.Content != "" {
			if _, err := b.Text(d.Content); err != nil {
				return true, err
			}
			if b.Cancelled() {
				return true, nil
			}
		}
		for _, tc := range d.ToolCalls {
			callID, err := resolveOpenAICallID(b, tc.Index, tc.ID, tc.Function.Name)
			if err != nil {
				return true, err
			}
			if tc.Function.Arguments == "" {
				continue
			}
			if _, err := b.ToolArgs(callID, tc.Function.Arguments); err != nil {
				return true, err
			}
			if b.Cancelled() {
				return true, nil
			}
		}
		if choice.FinishReason != "" {
			// The provider closed the message; every proposed call now has all
			// the arguments it will ever have. Sealing here is what lets the
			// boundary tell "complete" from "cut off mid-arguments".
			b.SealAll()
		}
	}
	return false, nil
}

// resolveOpenAICallID maps a fragment to a registered call. The wire sends the
// id on the first fragment and the index on all of them, so the index binding
// is what keeps later fragments attached to the right call.
func resolveOpenAICallID(b *StreamBridge, index int, id, name string) (string, error) {
	if id != "" {
		if err := b.ToolCallStart(id, name, ""); err != nil {
			return "", err
		}
		b.BindIndex(index, id)
		return id, nil
	}
	if bound, ok := b.CallIDForIndex(index); ok {
		if name != "" {
			if err := b.ToolCallStart(bound, name, ""); err != nil {
				return "", err
			}
		}
		return bound, nil
	}
	// No id has ever been seen for this index. Synthesize one from the index so
	// the fragments still land on a single ordered stream rather than being
	// dropped; the provider, not the adapter, chose to omit the identifier.
	synth := fmt.Sprintf("openai-tool-index-%d", index)
	if err := b.ToolCallStart(synth, name, ""); err != nil {
		return "", err
	}
	b.BindIndex(index, synth)
	return synth, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// openAIResponse is the subset of a NON-streaming chat completion this package
// steers on.
type openAIResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Error json.RawMessage `json:"error"`
}

// DecodeOpenAIResponse feeds a whole non-streaming OpenAI-compatible response
// into the bridge. The adapter it is given must be non-streaming, so its
// resolution is reported as request-boundary — the point of routing a buffered
// turn through the same bridge is that it cannot pass itself off as live.
func DecodeOpenAIResponse(b *StreamBridge, body []byte) error {
	if b.adapter.Streaming {
		return fmt.Errorf("generationctl: adapter %q is declared streaming; a whole-response decode would misreport its resolution", b.adapter.Name)
	}
	var resp openAIResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return fmt.Errorf("generationctl: decode response: %w", err)
	}
	if len(resp.Error) != 0 {
		return fmt.Errorf("%w: %s", ErrProviderRefused, string(resp.Error))
	}
	for _, choice := range resp.Choices {
		if choice.Message.Content != "" {
			if _, err := b.Text(choice.Message.Content); err != nil {
				return err
			}
			if b.Cancelled() {
				return nil
			}
		}
		for _, tc := range choice.Message.ToolCalls {
			id := tc.ID
			if id == "" {
				id = "openai-buffered-" + tc.Function.Name
			}
			if err := b.ToolCallStart(id, tc.Function.Name, ""); err != nil {
				return err
			}
			if _, err := b.ToolArgs(id, tc.Function.Arguments); err != nil {
				return err
			}
			if b.Cancelled() {
				return nil
			}
		}
	}
	b.SealAll()
	return nil
}
