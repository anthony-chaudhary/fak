package zaitask

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	GLM53FlashModel       = "glm-5.3-flash"
	DefaultGeneralBaseURL = "https://api.z.ai/api/paas/v4"
	HostedProvider        = "z.ai"
	HostedEngine          = "zai-hosted"
)

// CapabilityProbes records provider behaviors that are documented inconsistently.
// A caller must opt in only after its endpoint has accepted the corresponding form.
type CapabilityProbes struct {
	ResponseFormat bool
	ToolStream     bool
	DirectFileURL  bool
	DirectVideoURL bool
}

type Request struct {
	Model           string
	Messages        []Message
	MaxTokens       int
	Stream          bool
	ReasoningEffort string
	ClearThinking   *bool
	Tools           []ToolDefinition
	ToolChoice      string
	ToolStream      bool
	ResponseFormat  *ResponseFormat
	Probes          CapabilityProbes
}

type Message struct {
	Role             string     `json:"role"`
	Content          any        `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

type ContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *URLContent   `json:"image_url,omitempty"`
	File     *UploadedFile `json:"file,omitempty"`
	FileURL  *URLContent   `json:"file_url,omitempty"`
	VideoURL *URLContent   `json:"video_url,omitempty"`
}

type URLContent struct {
	URL string `json:"url"`
}
type UploadedFile struct {
	FileID string `json:"file_id"`
}
type ResponseFormat struct {
	Type string `json:"type"`
}

type ToolDefinition struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}
type ToolCall struct {
	Index    int              `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}
type ToolCallFunction struct {
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// ArgumentsJSON accepts the actual OpenAI-style JSON-string argument form while
// also tolerating the object form exposed by the provider's current OpenAPI schema.
func (f ToolCallFunction) ArgumentsJSON() (json.RawMessage, error) {
	raw := bytes.TrimSpace(f.Arguments)
	if len(raw) == 0 {
		return nil, errors.New("tool call arguments are empty")
	}
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, fmt.Errorf("decode tool call argument string: %w", err)
		}
		raw = json.RawMessage(text)
	}
	if !json.Valid(raw) {
		return nil, errors.New("tool call arguments are not valid JSON")
	}
	return append(json.RawMessage(nil), raw...), nil
}

type chatRequestWire struct {
	Model           string           `json:"model"`
	Messages        []Message        `json:"messages"`
	MaxTokens       int              `json:"max_tokens,omitempty"`
	Stream          bool             `json:"stream"`
	Thinking        thinkingWire     `json:"thinking"`
	ReasoningEffort string           `json:"reasoning_effort"`
	Tools           []ToolDefinition `json:"tools,omitempty"`
	ToolChoice      string           `json:"tool_choice,omitempty"`
	ToolStream      bool             `json:"tool_stream,omitempty"`
	ResponseFormat  *ResponseFormat  `json:"response_format,omitempty"`
}
type thinkingWire struct {
	Type          string `json:"type"`
	ClearThinking *bool  `json:"clear_thinking,omitempty"`
}

type chatResponseWire struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Model     string `json:"model"`
	RequestID string `json:"request_id"`
	Usage     Usage  `json:"usage"`
	Error     *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error,omitempty"`
}

type streamResponseWire struct {
	Choices []struct {
		Delta        Message `json:"delta"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Model     string `json:"model"`
	RequestID string `json:"request_id"`
	Usage     Usage  `json:"usage"`
	Error     *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error,omitempty"`
}

func (c Client) RunChat(ctx context.Context, in Request) (Result, error) {
	if strings.TrimSpace(c.APIKey) == "" {
		return Result{}, errors.New("ZAI API key is required; recovery: set Client.APIKey from a ZAI credential")
	}
	if in.Model == "" {
		in.Model = GLM53FlashModel
	}
	if in.Model != GLM53FlashModel {
		return Result{}, fmt.Errorf("hosted conformance route requires model %q", GLM53FlashModel)
	}
	if len(in.Messages) == 0 {
		return Result{}, errors.New("messages are required; recovery: provide at least one user message")
	}
	effort := strings.TrimSpace(in.ReasoningEffort)
	if effort == "" {
		effort = "max"
	}
	switch effort {
	case "low", "high", "max":
	default:
		return Result{}, errors.New("GLM-5.3-Flash reasoning_effort must be low, high, or max")
	}
	if in.ResponseFormat != nil && !in.Probes.ResponseFormat {
		return Result{}, errors.New("response_format probe has not been accepted for this endpoint")
	}
	if in.ToolStream && !in.Probes.ToolStream {
		return Result{}, errors.New("tool_stream probe has not been accepted for this endpoint")
	}
	if in.ToolStream && !in.Stream {
		return Result{}, errors.New("tool_stream requires stream=true")
	}
	if in.ClearThinking != nil && !*in.ClearThinking {
		for _, m := range in.Messages {
			if m.Role == "assistant" && strings.TrimSpace(m.ReasoningContent) == "" {
				return Result{}, errors.New("clear_thinking=false requires complete reasoning_content for every historical assistant message")
			}
		}
	}
	if err := validateMediaProbes(in.Messages, in.Probes); err != nil {
		return Result{}, err
	}

	wire := chatRequestWire{Model: in.Model, Messages: in.Messages, MaxTokens: in.MaxTokens, Stream: in.Stream,
		Thinking: thinkingWire{Type: "enabled", ClearThinking: in.ClearThinking}, ReasoningEffort: effort,
		Tools: in.Tools, ToolChoice: in.ToolChoice, ToolStream: in.ToolStream, ResponseFormat: in.ResponseFormat}
	body, err := json.Marshal(wire)
	if err != nil {
		return Result{}, fmt.Errorf("encode zai request; recovery: report the request encoder defect: %w", err)
	}
	base := strings.TrimRight(c.BaseURL, "/")
	if base == "" || base == DefaultBaseURL {
		base = DefaultGeneralBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Result{}, fmt.Errorf("build zai request; recovery: set Client.BaseURL to a valid HTTP(S) URL: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")
	if in.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	hc := c.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 2 * time.Minute}
	}
	started := time.Now()
	resp, err := hc.Do(req)
	latency := time.Since(started).Milliseconds()
	if err != nil {
		return Result{}, fmt.Errorf("zai request failed; recovery: check endpoint/network availability and retry: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Result{}, decodeHostedHTTPError(resp)
	}
	if in.Stream {
		got, err := decodeSSE(resp.Body)
		got.LatencyMS = latency
		return got, err
	}
	got, err := decodeNonStream(resp.Body)
	got.LatencyMS = latency
	return got, err
}

func validateMediaProbes(messages []Message, probes CapabilityProbes) error {
	for _, m := range messages {
		parts, ok := m.Content.([]ContentPart)
		if !ok {
			continue
		}
		for _, p := range parts {
			switch p.Type {
			case "file_url":
				if p.FileURL != nil && !probes.DirectFileURL {
					return errors.New("direct file URL probe has not been accepted for this endpoint")
				}
			case "video_url":
				if p.VideoURL != nil && !probes.DirectVideoURL {
					return errors.New("direct video URL probe has not been accepted for this endpoint")
				}
			}
		}
	}
	return nil
}

func decodeHostedHTTPError(resp *http.Response) error {
	raw, err := io.ReadAll(io.LimitReader(resp.Body, (8<<20)+1))
	if err != nil {
		return fmt.Errorf("read zai response: %w", err)
	}
	if len(raw) > 8<<20 {
		return fmt.Errorf("zai HTTP %d response exceeds 8 MiB limit", resp.StatusCode)
	}
	msg := strings.TrimSpace(string(raw))
	var decoded chatResponseWire
	if json.Unmarshal(raw, &decoded) == nil && decoded.Error != nil && decoded.Error.Message != "" {
		msg = decoded.Error.Message
	}
	return fmt.Errorf("zai HTTP %d: %s; recovery: inspect provider error, quota, and credentials before retrying", resp.StatusCode, msg)
}

func decodeNonStream(r io.Reader) (Result, error) {
	raw, err := io.ReadAll(io.LimitReader(r, (8<<20)+1))
	if err != nil {
		return Result{}, fmt.Errorf("read zai response; recovery: retry the request or inspect the endpoint transport: %w", err)
	}
	if len(raw) > 8<<20 {
		return Result{}, errors.New("zai response exceeds 8 MiB limit; recovery: request a smaller max_tokens value")
	}
	var decoded chatResponseWire
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return Result{}, fmt.Errorf("zai response returned invalid JSON: %w", err)
	}
	if decoded.Error != nil {
		return Result{}, errors.New(decoded.Error.Message)
	}
	if len(decoded.Choices) == 0 {
		return Result{}, errors.New("zai response contained no choices; recovery: retry or inspect provider response compatibility")
	}
	choice := decoded.Choices[0]
	decoded.Usage = normalizeUsage(decoded.Usage)
	return hostedResult(choice.Message, choice.FinishReason, decoded.Model, decoded.RequestID, decoded.Usage, false, false), nil
}

func decodeSSE(r io.Reader) (Result, error) {
	scanner := bufio.NewScanner(io.LimitReader(r, (8<<20)+1))
	scanner.Buffer(make([]byte, 64<<10), 8<<20)
	got := Result{Provider: HostedProvider, Engine: HostedEngine, FakNative: false, Streamed: true}
	calls := map[int]*ToolCall{}
	order := []int{}
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			got.Done = true
			break
		}
		var chunk streamResponseWire
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return Result{}, fmt.Errorf("decode zai SSE event: %w", err)
		}
		if chunk.Error != nil {
			return Result{}, errors.New(chunk.Error.Message)
		}
		if chunk.Model != "" {
			got.Model = chunk.Model
		}
		if chunk.RequestID != "" {
			got.RequestID = chunk.RequestID
		}
		if chunk.Usage.TotalTokens != 0 || chunk.Usage.PromptTokens != 0 {
			got.Usage = normalizeUsage(chunk.Usage)
		}
		for _, choice := range chunk.Choices {
			got.Content += contentString(choice.Delta.Content)
			got.ReasoningContent += choice.Delta.ReasoningContent
			if choice.FinishReason != "" {
				got.FinishReason = choice.FinishReason
			}
			for _, part := range choice.Delta.ToolCalls {
				idx := part.Index
				call, ok := calls[idx]
				if !ok {
					call = &ToolCall{Index: idx}
					calls[idx] = call
					order = append(order, idx)
				}
				if part.ID != "" {
					call.ID = part.ID
				}
				if part.Type != "" {
					call.Type = part.Type
				}
				if part.Function.Name != "" {
					call.Function.Name += part.Function.Name
				}
				frag, err := argumentFragment(part.Function.Arguments)
				if err != nil {
					return Result{}, err
				}
				if frag != "" {
					prior, _ := argumentFragment(call.Function.Arguments)
					call.Function.Arguments = json.RawMessage(strconv.Quote(prior + frag))
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return Result{}, fmt.Errorf("read zai SSE response: %w", err)
	}
	if !got.Done {
		return Result{}, errors.New("zai SSE response ended before [DONE]")
	}
	for _, idx := range order {
		got.ToolCalls = append(got.ToolCalls, *calls[idx])
	}
	return got, nil
}

func argumentFragment(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", err
		}
		return s, nil
	}
	return string(raw), nil
}

func hostedResult(m Message, finish, model, requestID string, usage Usage, streamed, done bool) Result {
	return Result{Content: contentString(m.Content), ReasoningContent: m.ReasoningContent, ToolCalls: m.ToolCalls,
		FinishReason: finish, Model: model, RequestID: requestID, Usage: usage, Provider: HostedProvider,
		Engine: HostedEngine, FakNative: false, Streamed: streamed, Done: done}
}

func contentString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

func normalizeUsage(u Usage) Usage {
	if u.CachedTokens == 0 {
		u.CachedTokens = u.PromptTokensDetails.CachedTokens
	}
	return u
}
