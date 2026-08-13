package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Observed is what a recorded body empirically contains. Every field is
// re-derived from the bytes, never from what the request asked for — that is
// the whole point. A scenario named "tool-destructive" that came back as prose,
// or as a 400, produces an Observed that says so, and the manifest row carries
// it.
type Observed struct {
	Chunks              int      `json:"chunks"`
	TextDeltas          int      `json:"text_deltas"`
	ThinkingDeltas      int      `json:"thinking_deltas"`
	ToolCalls           int      `json:"tool_calls"`
	ToolArgDeltas       int      `json:"tool_arg_deltas"`
	MaxArgDeltasPerCall int      `json:"max_arg_deltas_per_call"`
	ToolArgsFragmented  bool     `json:"tool_args_fragmented"`
	ToolNames           []string `json:"tool_names,omitempty"`
	ArgumentsComplete   bool     `json:"arguments_complete"`
	FinishReason        string   `json:"finish_reason,omitempty"`
	Terminated          bool     `json:"terminated"`
	ProviderError       string   `json:"provider_error,omitempty"`
}

type wireToolCall struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireDelta struct {
	Content string `json:"content"`
	// Providers disagree on the reasoning field name. Neither is committable
	// output, so both are counted apart from text.
	Reasoning        string         `json:"reasoning"`
	ReasoningContent string         `json:"reasoning_content"`
	ToolCalls        []wireToolCall `json:"tool_calls"`
}

func (d wireDelta) thinking() string {
	if d.Reasoning != "" {
		return d.Reasoning
	}
	return d.ReasoningContent
}

type wireChunk struct {
	Choices []struct {
		Delta        wireDelta  `json:"delta"`
		Message      *wireDelta `json:"message"`
		FinishReason string     `json:"finish_reason"`
	} `json:"choices"`
	Error json.RawMessage `json:"error"`
}

type argAccumulator struct {
	order  []int
	byIdx  map[int]*strings.Builder
	deltas map[int]int
	names  map[int]string
}

func newArgAccumulator() *argAccumulator {
	return &argAccumulator{byIdx: map[int]*strings.Builder{}, deltas: map[int]int{}, names: map[int]string{}}
}

func (a *argAccumulator) apply(call wireToolCall) (argDelta string) {
	idx := 0
	if call.Index != nil {
		idx = *call.Index
	}
	if _, seen := a.byIdx[idx]; !seen {
		a.byIdx[idx] = &strings.Builder{}
		a.order = append(a.order, idx)
	}
	if call.Function.Name != "" {
		a.names[idx] = call.Function.Name
	}
	if call.Function.Arguments == "" {
		return ""
	}
	a.byIdx[idx].WriteString(call.Function.Arguments)
	a.deltas[idx]++
	return call.Function.Arguments
}

func (a *argAccumulator) fold(o *Observed) {
	o.ToolCalls = len(a.order)
	o.ArgumentsComplete = len(a.order) > 0
	for _, idx := range a.order {
		if name := a.names[idx]; name != "" {
			o.ToolNames = appendUnique(o.ToolNames, name)
		}
		if n := a.deltas[idx]; n > o.MaxArgDeltasPerCall {
			o.MaxArgDeltasPerCall = n
		}
		if !json.Valid([]byte(a.byIdx[idx].String())) {
			o.ArgumentsComplete = false
		}
	}
	o.ToolArgsFragmented = o.MaxArgDeltasPerCall > 1
}

// analyzeStream folds a recorded SSE body.
func analyzeStream(raw []byte) Observed {
	var o Observed
	acc := newArgAccumulator()
	for _, data := range sseEvents(raw, &o) {
		var chunk wireChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			o.ProviderError = "undecodable chunk: " + err.Error()
			continue
		}
		if message := errorMessage(chunk.Error); message != "" {
			o.ProviderError = message
			continue
		}
		o.Chunks++
		foldChoices(&o, acc, chunk)
	}
	if o.Chunks == 0 && o.ProviderError == "" {
		if message := errorBody(raw); message != "" {
			o.ProviderError = message
		}
	}
	acc.fold(&o)
	return o
}

// analyzeResponse folds a recorded non-streaming JSON body.
func analyzeResponse(raw []byte) Observed {
	var o Observed
	var chunk wireChunk
	if err := json.Unmarshal(raw, &chunk); err != nil {
		o.ProviderError = "undecodable response: " + err.Error()
		return o
	}
	if message := errorMessage(chunk.Error); message != "" {
		o.ProviderError = message
		return o
	}
	acc := newArgAccumulator()
	o.Chunks = 1
	o.Terminated = true
	foldChoices(&o, acc, chunk)
	acc.fold(&o)
	return o
}

func foldChoices(o *Observed, acc *argAccumulator, chunk wireChunk) {
	for _, choice := range chunk.Choices {
		if choice.FinishReason != "" {
			o.FinishReason = choice.FinishReason
		}
		delta := choice.Delta
		if choice.Message != nil {
			delta = *choice.Message
		}
		if delta.thinking() != "" {
			o.ThinkingDeltas++
		}
		if delta.Content != "" {
			o.TextDeltas++
		}
		for _, call := range delta.ToolCalls {
			if acc.apply(call) != "" {
				o.ToolArgDeltas++
			}
		}
	}
}

// sseEvents splits a recorded body into event payloads, setting Terminated when
// the terminal [DONE] sentinel is present.
func sseEvents(raw []byte, o *Observed) []string {
	var events []string
	var current strings.Builder
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	flush := func() {
		if current.Len() == 0 {
			return
		}
		data := current.String()
		current.Reset()
		if strings.TrimSpace(data) == "[DONE]" {
			o.Terminated = true
			return
		}
		events = append(events, data)
	}
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		if line == "" {
			flush()
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		payload, isData := strings.CutPrefix(line, "data:")
		if !isData {
			continue
		}
		if current.Len() > 0 {
			current.WriteString("\n")
		}
		current.WriteString(strings.TrimPrefix(payload, " "))
	}
	flush()
	return events
}

func errorMessage(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var probe struct {
		Message string `json:"message"`
		Code    string `json:"code"`
		Type    string `json:"type"`
	}
	if err := json.Unmarshal(raw, &probe); err == nil {
		for _, candidate := range []string{probe.Message, probe.Code, probe.Type} {
			if candidate != "" {
				return candidate
			}
		}
	}
	return truncate(strings.TrimSpace(string(raw)))
}

func errorBody(raw []byte) string {
	var probe struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
		Detail  string          `json:"detail"`
		Title   string          `json:"title"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		trimmed := strings.TrimSpace(string(raw))
		if trimmed == "" {
			return "empty body"
		}
		return truncate(trimmed)
	}
	if message := errorMessage(probe.Error); message != "" {
		return message
	}
	for _, candidate := range []string{probe.Message, probe.Detail, probe.Title} {
		if candidate != "" {
			return candidate
		}
	}
	return ""
}

func truncate(s string) string {
	if len(s) > 200 {
		return s[:200]
	}
	return s
}

func appendUnique(names []string, name string) []string {
	for _, existing := range names {
		if existing == name {
			return names
		}
	}
	return append(names, name)
}

func sha256Of(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// secretShapes are the token shapes that must never reach a committed capture.
// The recorder sends the key in a header and the provider does not echo it, but
// "should not happen" is not a control: the check runs before every write and
// refuses fail-closed.
var secretShapes = []*regexp.Regexp{
	regexp.MustCompile(`gsk_[A-Za-z0-9]{20,}`),
	regexp.MustCompile(`nvapi-[A-Za-z0-9_\-]{20,}`),
	regexp.MustCompile(`sk-[A-Za-z0-9_\-]{20,}`),
	regexp.MustCompile(`(?i)"?\bauthorization\b"?\s*[:=]`),
	regexp.MustCompile(`(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}`),
	regexp.MustCompile(`(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b`),
}

// scrubViolations reports every secret-or-PII shape found in a payload, plus
// any live key value from the environment. The caller refuses to write on a
// non-empty result.
func scrubViolations(payload []byte, envKeys []string) []string {
	var found []string
	for _, shape := range secretShapes {
		if match := shape.Find(payload); match != nil {
			found = append(found, fmt.Sprintf("%s matched %q", shape.String(), redact(string(match))))
		}
	}
	for _, key := range envKeys {
		if len(key) >= 8 && bytes.Contains(payload, []byte(key)) {
			found = append(found, "payload contains a live API key from the environment")
		}
	}
	return found
}

func redact(match string) string {
	if len(match) <= 6 {
		return "…"
	}
	return match[:6] + "…"
}
