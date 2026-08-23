package canon

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

const TokenUsageSchema = "fak/token-usage/v1"

type TokenClass string

const (
	inputFreshClass      TokenClass = "input_uncached"
	inputReadClass       TokenClass = "input_reused"
	inputWriteClass      TokenClass = "input_prefilled"
	outputVisibleClass   TokenClass = "output_visible"
	outputReasoningClass TokenClass = "output_reasoning"
)

type TokenUsage struct {
	Schema   string               `json:"schema"`
	Provider string               `json:"provider"`
	Classes  map[TokenClass]int64 `json:"classes"`
	Input    int64                `json:"input_total"`
	Output   int64                `json:"output_total"`
	Total    int64                `json:"total"`
	Raw      json.RawMessage      `json:"raw"`
}

func AdaptTokenUsage(provider string, raw json.RawMessage) (TokenUsage, error) {
	if !json.Valid(raw) {
		return TokenUsage{}, errors.New("token usage: invalid native JSON")
	}
	var classes map[TokenClass]int64
	var input, output int64
	var err error
	switch provider {
	case "openai":
		classes, input, output, err = adaptOpenAIUsage(raw)
	case "anthropic":
		classes, input, output, err = adaptAnthropicUsage(raw)
	case "gemini":
		classes, input, output, err = adaptGeminiUsage(raw)
	case "local":
		classes, input, output, err = adaptLocalUsage(raw)
	default:
		return TokenUsage{}, fmt.Errorf("token usage: unsupported provider %q", provider)
	}
	if err != nil {
		return TokenUsage{}, err
	}
	for class, n := range classes {
		if n < 0 {
			return TokenUsage{}, fmt.Errorf("token usage: %s is negative", class)
		}
		if n == 0 {
			delete(classes, class)
		}
	}
	var reconciled int64
	for _, n := range classes {
		reconciled += n
	}
	if reconciled != input+output {
		return TokenUsage{}, fmt.Errorf("token usage: classes reconcile to %d, native totals to %d", reconciled, input+output)
	}
	return TokenUsage{Schema: TokenUsageSchema, Provider: provider, Classes: classes, Input: input, Output: output, Total: input + output, Raw: append(json.RawMessage(nil), bytes.TrimSpace(raw)...)}, nil
}

func adaptOpenAIUsage(raw []byte) (map[TokenClass]int64, int64, int64, error) {
	var u struct {
		InputTokens      int64 `json:"input_tokens"`
		OutputTokens     int64 `json:"output_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		InputDetails     struct {
			CacheWrite int64 `json:"cache_write_tokens"`
			Cached     int64 `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		PromptDetails struct {
			Cached int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		OutputDetails struct {
			Reasoning int64 `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
		CompletionDetails struct {
			Reasoning int64 `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	}
	if err := json.Unmarshal(raw, &u); err != nil {
		return nil, 0, 0, err
	}
	input := u.InputTokens
	if input == 0 {
		input = u.PromptTokens
	}
	output := u.OutputTokens
	if output == 0 {
		output = u.CompletionTokens
	}
	cacheWrite := u.InputDetails.CacheWrite
	cached := u.InputDetails.Cached
	if cached == 0 {
		cached = u.PromptDetails.Cached
	}
	reasoning := u.OutputDetails.Reasoning
	if reasoning == 0 {
		reasoning = u.CompletionDetails.Reasoning
	}
	if cacheWrite+cached > input || reasoning > output {
		return nil, 0, 0, errors.New("token usage: OpenAI detail exceeds its inclusive total")
	}
	return map[TokenClass]int64{inputFreshClass: input - cached - cacheWrite, inputReadClass: cached, inputWriteClass: cacheWrite, outputVisibleClass: output - reasoning, outputReasoningClass: reasoning}, input, output, nil
}

func adaptAnthropicUsage(raw []byte) (map[TokenClass]int64, int64, int64, error) {
	var u struct {
		Input      int64 `json:"input_tokens"`
		Output     int64 `json:"output_tokens"`
		CacheRead  int64 `json:"cache_read_input_tokens"`
		CacheWrite int64 `json:"cache_creation_input_tokens"`
	}
	if err := json.Unmarshal(raw, &u); err != nil {
		return nil, 0, 0, err
	}
	input := u.Input + u.CacheRead + u.CacheWrite
	return map[TokenClass]int64{inputFreshClass: u.Input, inputReadClass: u.CacheRead, inputWriteClass: u.CacheWrite, outputVisibleClass: u.Output}, input, u.Output, nil
}

// adaptGeminiUsage copies the inclusive prompt/cache semantics from Google's
// generated GenerateContentResponseUsageMetadata contract. Tool-result tokens
// are additional model input; thoughts are generated output distinct from the
// visible candidates.
func adaptGeminiUsage(raw []byte) (map[TokenClass]int64, int64, int64, error) {
	var u struct {
		PromptTokenCount        int64 `json:"promptTokenCount"`
		CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
		ToolUsePromptTokenCount int64 `json:"toolUsePromptTokenCount"`
		CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
		ThoughtsTokenCount      int64 `json:"thoughtsTokenCount"`
		TotalTokenCount         int64 `json:"totalTokenCount"`
	}
	if err := json.Unmarshal(raw, &u); err != nil {
		return nil, 0, 0, err
	}
	if u.CachedContentTokenCount > u.PromptTokenCount {
		return nil, 0, 0, errors.New("token usage: Gemini cached content exceeds inclusive prompt total")
	}
	input := u.PromptTokenCount + u.ToolUsePromptTokenCount
	output := u.CandidatesTokenCount + u.ThoughtsTokenCount
	if u.TotalTokenCount != 0 && u.TotalTokenCount != input+output {
		return nil, 0, 0, errors.New("token usage: Gemini total does not reconcile with generated SDK components")
	}
	return map[TokenClass]int64{
		inputFreshClass:      u.PromptTokenCount - u.CachedContentTokenCount + u.ToolUsePromptTokenCount,
		inputReadClass:       u.CachedContentTokenCount,
		inputWriteClass:      0, // Gemini reports cached reads but not provider-side cache creation tokens.
		outputVisibleClass:   u.CandidatesTokenCount,
		outputReasoningClass: u.ThoughtsTokenCount,
	}, input, output, nil
}

func adaptLocalUsage(raw []byte) (map[TokenClass]int64, int64, int64, error) {
	var u struct {
		Prompt     int64 `json:"prompt_tokens"`
		Completion int64 `json:"completion_tokens"`
	}
	if err := json.Unmarshal(raw, &u); err != nil {
		return nil, 0, 0, err
	}
	return map[TokenClass]int64{inputFreshClass: u.Prompt, outputVisibleClass: u.Completion}, u.Prompt, u.Completion, nil
}
