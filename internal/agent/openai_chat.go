package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const openAIChatResponseLimit = 4 << 20

type openAIChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// CompleteOpenAIChat performs one bounded, non-streaming completion against an
// OpenAI-compatible endpoint. It does not retry or select a fallback provider.
func CompleteOpenAIChat(ctx context.Context, client *http.Client, endpoint, model string, messages []Message) (string, error) {
	if client == nil {
		return "", errors.New("agent: nil OpenAI chat client")
	}
	payload := map[string]any{
		"model":       model,
		"stream":      false,
		"temperature": 0,
		"messages":    messages,
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint, "/")+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, openAIChatResponseLimit))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("OpenAI chat status %d", resp.StatusCode)
	}
	var chat openAIChatResponse
	if json.Unmarshal(body, &chat) != nil || len(chat.Choices) == 0 || strings.TrimSpace(chat.Choices[0].Message.Content) == "" {
		return "", errors.New("agent: invalid OpenAI chat response")
	}
	return chat.Choices[0].Message.Content, nil
}
