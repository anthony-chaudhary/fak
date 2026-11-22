// Adapted from ai-dynamo/aiperf, src/aiperf/endpoints/openai_chat.py:74 at ab610add4e8876fca223197596c728e17365f381. Apache-2.0; the
// streaming-only stream_options.include_usage injection (format_payload / _ensure_include_usage)
// re-expressed as a typed, non-clobbering default on loadgen's synthetic chat request.

package main

// streamOptions is the OpenAI stream_options object. IncludeUsage has no
// omitempty so an explicit caller opt-out (false) survives on the wire.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// syntheticChatRequest is the chat request oneRequest sends when no stored
// payload is being replayed.
func syntheticChatRequest(cfg *Config) chatRequest {
	return chatRequest{
		Model:       cfg.Model,
		Messages:    []chatMessage{{Role: "user", Content: cfg.Prompt}},
		MaxTokens:   cfg.MaxTokens,
		Temperature: 0,
		Stream:      true,
	}.withIncludeUsage()
}

// withIncludeUsage asks a streaming endpoint for the trailing usage chunk:
// OpenAI-compatible servers (vLLM, SGLang) omit usage from SSE streams unless
// include_usage is set, which would leave completion tokens unknown. It never
// overrides caller-supplied stream_options, and adds nothing when not
// streaming because servers reject stream_options on stream=false.
func (r chatRequest) withIncludeUsage() chatRequest {
	if r.Stream && r.StreamOptions == nil {
		r.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return r
}
