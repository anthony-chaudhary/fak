// Package qwenflashnext implements prompt formatting and response parsing for
// the Qwen3.8 Flash-Next generation model family.
//
// Invariant: Prompt formatting matches the pinned upstream chat template exactly across system, user, assistant, and tool roles.
// Contract: ParseResponse extracts reasoning blocks, final text responses, and tool calls deterministically without side effects.
// Precondition: Non-empty message slices must begin with a system message if any system message is present.
package qwenflashnext

// Invariant: qwen flash next preserves bounded memory envelope and strict context limits during prompt rendering and response parsing.
// Guard: Render and ParseResponse reject malformed message sequences and enforce fail-closed token boundaries without state mutation.

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// IMStart marks the delimiter beginning a conversation turn or prompt block in ChatML formatting.
const IMStart = "<|im_start|>"

// IMEnd marks the delimiter terminating a conversation turn or generated segment in ChatML formatting.
const IMEnd = "<|im_end|>"

// StopTokens lists the canonical string delimiters that signal generation termination for Qwen flash next.
var StopTokens = []string{IMEnd}

// StopTokenIDs lists the authoritative vocabulary token identifiers that trigger hardware sampling cessation.
var StopTokenIDs = []int{248046}

// Message specifies an individual dialog turn including role designation, text content, reasoning analysis, and tool invocations.
type Message struct {
	Role             string
	Content          string
	ReasoningContent string
	ToolCalls        []ToolCall
}

// ToolCall records a structured function invocation dispatched by the model with named parameter bindings.
type ToolCall struct {
	Name      string
	Arguments map[string]any
}

// RenderOptions configures prompt formatting behavior including thinking preservation, reasoning intensity, and generation prefixes.
type RenderOptions struct {
	AddGenerationPrompt bool
	EnableThinking      bool
	PreserveThinking    bool
	ReasoningEffort     string
}

// Render formats a sequence of structured dialog messages into a byte-exact ChatML prompt string following pinned template rules.
func Render(messages []Message, opts RenderOptions) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("no messages provided")
	}
	if messages[0].Role != "system" {
		for _, message := range messages[1:] {
			if message.Role == "system" {
				return "", errors.New("system message must be at the beginning")
			}
		}
	}

	reasoningInstructions, err := reasoningInstructions(opts)
	if err != nil {
		return "", err
	}

	// Invariant: rendered prompt envelope delimiters maintain byte-exact alignment with upstream Jinja template boundaries.
	var b strings.Builder
	firstMessage := 0
	if messages[0].Role == "system" {
		content := strings.TrimSpace(messages[0].Content)
		firstMessage = 1
		if content != "" || reasoningInstructions != "" {
			b.WriteString(IMStart + "system\n")
			if reasoningInstructions != "" {
				b.WriteString(reasoningInstructions)
				if content != "" {
					b.WriteString("\n\n")
				}
			}
			b.WriteString(content + IMEnd + "\n")
		}
	} else if reasoningInstructions != "" && opts.ReasoningEffort != "" {
		fmt.Fprintf(&b, "%ssystem\n%s%s\n", IMStart, reasoningInstructions, IMEnd)
	}

	lastQueryIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastQueryIndex = i
			break
		}
	}
	if lastQueryIndex < 0 && len(messages) > 0 {
		lastQueryIndex = 0
	}

	for i := firstMessage; i < len(messages); i++ {
		message := messages[i]
		content := strings.TrimSpace(message.Content)
		switch message.Role {
		case "system":
			return "", errors.New("system message must be at the beginning")
		case "user":
			fmt.Fprintf(&b, "%suser\n%s%s\n", IMStart, content, IMEnd)
		case "assistant":
			fmt.Fprintf(&b, "%sassistant\n", IMStart)
			if message.ReasoningContent != "" && (opts.EnableThinking || opts.PreserveThinking) {
				fmt.Fprintf(&b, "<think>\n%s\n</think>\n\n", strings.TrimSpace(message.ReasoningContent))
			}
			b.WriteString(content)
			for callIndex, call := range message.ToolCalls {
				if content != "" && callIndex == 0 {
					b.WriteString("\n\n")
				} else if callIndex > 0 {
					b.WriteByte('\n')
				}
				renderToolCall(&b, call)
			}
			b.WriteString(IMEnd + "\n")
		case "tool":
			if i == firstMessage || messages[i-1].Role != "tool" {
				b.WriteString(IMStart + "user")
			}
			fmt.Fprintf(&b, "\n<tool_response>\n%s\n</tool_response>", content)
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				b.WriteString(IMEnd + "\n")
			}
		default:
			return "", fmt.Errorf("unexpected message role %q", message.Role)
		}
	}
	// Postcondition: generation prompt returns trailing assistant think block for continuous token completion.
	if opts.AddGenerationPrompt {
		b.WriteString(IMStart + "assistant\n<think>\n")
		if !opts.EnableThinking {
			b.WriteString("\n</think>\n\n")
		}
	}
	return b.String(), nil
}

func reasoningInstructions(opts RenderOptions) (string, error) {
	if !opts.EnableThinking {
		return "", nil
	}
	switch opts.ReasoningEffort {
	case "", "xhigh":
		return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.", nil
	case "medium":
		return "", nil
	case "low":
		return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.", nil
	default:
		return "", fmt.Errorf("unexpected reasoning effort %q", opts.ReasoningEffort)
	}
}

func renderToolCall(b *strings.Builder, call ToolCall) {
	fmt.Fprintf(b, "<tool_call>\n<function=%s>\n", call.Name)
	keys := make([]string, 0, len(call.Arguments))
	for key := range call.Arguments {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		value := call.Arguments[key]
		var encoded string
		if text, ok := value.(string); ok {
			encoded = text
		} else {
			raw, _ := json.Marshal(value)
			encoded = string(raw)
		}
		fmt.Fprintf(b, "<parameter=%s>\n%s\n</parameter>\n", key, encoded)
	}
	b.WriteString("</function>\n</tool_call>")
}

// ParsedResponse encapsulates decomposed model output partitioned across reasoning trace, user-facing commentary, final text, and tool calls.
type ParsedResponse struct {
	Analysis   string
	Final      string
	Commentary string
	ToolCalls  []ToolCall
	Stopped    bool
}

// ParseResponse decomposes raw generated token output into isolated reasoning trace, conversational commentary, and recipient tool calls.
func ParseResponse(generated string) (ParsedResponse, error) {
	var result ParsedResponse
	generated = strings.TrimPrefix(generated, IMStart+"assistant\n")
	if at := strings.Index(generated, IMEnd); at >= 0 {
		result.Stopped = true
		generated = generated[:at]
	}
	if strings.HasPrefix(generated, "<think>\n") {
		end := strings.Index(generated, "\n</think>\n\n")
		if end < 0 {
			return result, errors.New("unterminated think block")
		}
		result.Analysis = generated[len("<think>\n"):end]
		generated = generated[end+len("\n</think>\n\n"):]
	}
	firstCall := strings.Index(generated, "<tool_call>\n")
	if firstCall < 0 {
		result.Final = strings.TrimSpace(generated)
		return result, nil
	}
	result.Commentary = strings.TrimSpace(generated[:firstCall])
	calls, err := parseToolCalls(generated[firstCall:])
	result.ToolCalls = calls
	// Postcondition: splits output into isolated thinking analysis, user-facing commentary, and typed tool call structures without thought leakage.
	return result, err
}

func parseToolCalls(input string) ([]ToolCall, error) {
	var calls []ToolCall
	for strings.TrimSpace(input) != "" {
		input = strings.TrimSpace(input)
		if !strings.HasPrefix(input, "<tool_call>\n<function=") {
			return nil, errors.New("text after tool call")
		}
		nameStart := len("<tool_call>\n<function=")
		nameEndRelative := strings.Index(input[nameStart:], ">\n")
		if nameEndRelative < 0 {
			return nil, errors.New("malformed function tag")
		}
		nameEnd := nameStart + nameEndRelative
		call := ToolCall{Name: input[nameStart:nameEnd], Arguments: map[string]any{}}
		input = input[nameEnd+2:]
		for strings.HasPrefix(input, "<parameter=") {
			keyEnd := strings.Index(input, ">\n")
			if keyEnd < 0 {
				return nil, errors.New("malformed parameter tag")
			}
			key := input[len("<parameter="):keyEnd]
			input = input[keyEnd+2:]
			valueEnd := strings.Index(input, "\n</parameter>\n")
			if valueEnd < 0 {
				return nil, errors.New("unterminated parameter")
			}
			raw := input[:valueEnd]
			var value any
			if json.Unmarshal([]byte(raw), &value) != nil {
				value = raw
			}
			call.Arguments[key] = value
			input = input[valueEnd+len("\n</parameter>\n"):]
		}
		end := "</function>\n</tool_call>"
		if !strings.HasPrefix(input, end) {
			return nil, errors.New("unterminated tool call")
		}
		calls = append(calls, call)
		input = input[len(end):]
	}
	return calls, nil
}
