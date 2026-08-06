package main

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
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type endpointStats struct {
	latencies        []time.Duration
	ttfts            []time.Duration
	promptTokens     int64
	completionTokens int64
	cachedTokens     int64
	usageResponses   int
}

type openAIEndpoint struct {
	baseURL string
	apiKey  string
	model   string
	base    *sharedBase
	client  *http.Client
	statsMu sync.Mutex
	stats   endpointStats
}

type chatRequest struct {
	Model         string          `json:"model"`
	Messages      []agent.Message `json:"messages"`
	MaxTokens     int             `json:"max_tokens"`
	Temperature   float64         `json:"temperature"`
	Stream        bool            `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		PromptDetails    struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage,omitempty"`
}

func newOpenAIEndpoint(baseURL, apiKey, model string, base *sharedBase, timeout time.Duration) (*openAIEndpoint, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" || model == "" {
		return nil, errors.New("endpoint and model are required")
	}
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL += "/v1"
	}
	return &openAIEndpoint{baseURL: baseURL, apiKey: apiKey, model: model, base: base, client: &http.Client{Timeout: timeout}}, nil
}

func (g *openAIEndpoint) Model() string { return g.model }

func (g *openAIEndpoint) Complete(ctx context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if len(messages) != 1 || messages[0].Role != agent.RoleUser {
		return nil, fmt.Errorf("delta contract: got %d messages", len(messages))
	}
	request := chatRequest{Model: g.model, MaxTokens: 8, Temperature: 0, Stream: true}
	request.StreamOptions.IncludeUsage = true
	request.Messages = []agent.Message{
		{Role: agent.RoleSystem, Content: g.base.instructions},
		messages[0],
	}
	body, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, g.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if g.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+g.apiKey)
	}
	start := time.Now()
	resp, err := g.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("endpoint status %s: %s", resp.Status, strings.TrimSpace(string(limited)))
	}

	var text strings.Builder
	var ttft time.Duration
	var prompt, completion, cached int64
	usageSeen := false
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return nil, fmt.Errorf("decode stream chunk: %w", err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				if ttft == 0 {
					ttft = time.Since(start)
				}
				text.WriteString(choice.Delta.Content)
			}
		}
		if chunk.Usage != nil {
			prompt = chunk.Usage.PromptTokens
			completion = chunk.Usage.CompletionTokens
			cached = chunk.Usage.PromptDetails.CachedTokens
			usageSeen = true
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	latency := time.Since(start)
	if ttft == 0 {
		ttft = latency
	}
	g.statsMu.Lock()
	g.stats.latencies = append(g.stats.latencies, latency)
	g.stats.ttfts = append(g.stats.ttfts, ttft)
	g.stats.promptTokens += prompt
	g.stats.completionTokens += completion
	g.stats.cachedTokens += cached
	if usageSeen {
		g.stats.usageResponses++
	}
	g.statsMu.Unlock()
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: text.String()}}, nil
}

func (g *openAIEndpoint) snapshot() endpointStats {
	g.statsMu.Lock()
	defer g.statsMu.Unlock()
	out := g.stats
	out.latencies = append([]time.Duration(nil), g.stats.latencies...)
	out.ttfts = append([]time.Duration(nil), g.stats.ttfts...)
	return out
}
