package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"strconv"
	"strings"
	"unicode/utf8"
)

type opsRunProbeEvidence struct {
	Reason, ObservedModel, FinishReason, ResponseDigest string
}

type opsRunProbeTool struct {
	Index    *int    `json:"index"`
	ID       *string `json:"id"`
	Type     *string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type opsRunProbeChoice struct {
	Index json.RawMessage `json:"index"`
	Delta struct {
		Content   *string           `json:"content"`
		ToolCalls []opsRunProbeTool `json:"tool_calls"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

func opsRunReadInferenceProbe(ctx context.Context, r io.Reader, model string) opsRunProbeEvidence {
	var out opsRunProbeEvidence
	fail := func(reason string) opsRunProbeEvidence { out.Reason = reason; return out }
	var raw []byte
	var readErr error
	if ctx.Err() == nil {
		raw, readErr = io.ReadAll(io.LimitReader(r, opsRunInferenceProbeMaxBytes+1))
	}
	sum := sha256.Sum256(raw)
	out.ResponseDigest = "sha256:" + hex.EncodeToString(sum[:])
	if ctx.Err() != nil {
		return fail("timeout")
	}
	if len(raw) > opsRunInferenceProbeMaxBytes {
		return fail("response_too_large")
	}
	if readErr != nil {
		if e, ok := readErr.(interface{ Timeout() bool }); ok && e.Timeout() {
			return fail("timeout")
		}
		return fail("incomplete_sse")
	}
	parsed := opsRunParseInferenceProbe(raw, model)
	parsed.ResponseDigest = out.ResponseDigest
	return parsed
}

func opsRunParseInferenceProbe(raw []byte, model string) opsRunProbeEvidence {
	var out opsRunProbeEvidence
	fail := func(reason string) opsRunProbeEvidence { out.Reason = reason; return out }
	text := strings.ReplaceAll(string(raw), "\r\n", "\n")
	if !utf8.Valid(raw) || strings.ContainsRune(text, '\r') {
		return fail("malformed_sse")
	}
	var name, args strings.Builder
	var id string
	typed, finished, usageSeen := false, false, false
	events := strings.SplitAfter(text, "\n\n")
	for eventIndex, event := range events {
		if event == "" {
			continue
		}
		if !strings.HasSuffix(event, "\n\n") {
			return fail("incomplete_sse")
		}
		var data []string
		for _, line := range strings.Split(strings.TrimSuffix(event, "\n\n"), "\n") {
			if line == "" || strings.HasPrefix(line, ":") {
				continue
			}
			if !strings.HasPrefix(line, "data:") {
				return fail("malformed_sse")
			}
			data = append(data, strings.TrimPrefix(line[5:], " "))
		}
		if len(data) == 0 {
			continue
		}
		payload := strings.Join(data, "\n")
		if payload == "[DONE]" {
			if eventIndex != len(events)-2 || events[len(events)-1] != "" {
				return fail("malformed_sse")
			}
			if !finished || out.ObservedModel == "" {
				return fail("incomplete_sse")
			}
			if id == "" || !typed || name.String() != opsRunInferenceProbeTool || !opsRunProbeArgumentsValid(args.String()) {
				return fail("malformed_tool_call")
			}
			return out
		}
		var frame struct {
			Model   json.RawMessage      `json:"model"`
			Choices *[]opsRunProbeChoice `json:"choices"`
			Usage   json.RawMessage      `json:"usage"`
		}
		if json.Unmarshal([]byte(payload), &frame) != nil || frame.Choices == nil {
			return fail("malformed_sse")
		}
		if len(frame.Model) != 0 {
			var observed string
			m := bytes.TrimSpace(frame.Model)
			if len(m) == 0 || m[0] != '"' || json.Unmarshal(m, &observed) != nil || observed == "" || len(observed) > 512 {
				return fail("malformed_sse")
			}
			if out.ObservedModel == "" {
				out.ObservedModel = observed
			}
			if observed != model || observed != out.ObservedModel {
				return fail("model_mismatch")
			}
		}
		choices := *frame.Choices
		if len(choices) == 0 {
			if !finished || usageSeen || len(frame.Model) == 0 || !opsRunProbeUsageValid(frame.Usage) {
				return fail("malformed_sse")
			}
			usageSeen = true
			continue
		}
		if len(choices) != 1 || finished {
			return fail("malformed_tool_call")
		}
		c := choices[0]
		if !opsRunProbeChoiceIndexValid(c.Index) || (c.Delta.Content != nil && *c.Delta.Content != "") || len(c.Delta.ToolCalls) > 1 {
			return fail("malformed_tool_call")
		}
		for _, tool := range c.Delta.ToolCalls {
			if tool.Index == nil || *tool.Index != 0 {
				return fail("malformed_tool_call")
			}
			if tool.ID != nil {
				if id != "" || strings.TrimSpace(*tool.ID) == "" {
					return fail("malformed_tool_call")
				}
				id = *tool.ID
			}
			if tool.Type != nil {
				if typed || *tool.Type != "function" {
					return fail("malformed_tool_call")
				}
				typed = true
			}
			name.WriteString(tool.Function.Name)
			args.WriteString(tool.Function.Arguments)
		}
		if c.FinishReason != nil && *c.FinishReason != "" {
			if len(*c.FinishReason) > 64 {
				return fail("malformed_sse")
			}
			out.FinishReason = *c.FinishReason
			if out.FinishReason != "tool_calls" {
				return fail("incomplete_sse")
			}
			finished = true
		}
		u := bytes.TrimSpace(frame.Usage)
		if len(u) != 0 && !bytes.Equal(u, []byte("null")) {
			if !finished || usageSeen || !opsRunProbeUsageValid(u) {
				return fail("malformed_sse")
			}
			usageSeen = true
		}
	}
	return fail("incomplete_sse")
}

func opsRunProbeArgumentsValid(raw string) bool {
	d := json.NewDecoder(strings.NewReader(raw))
	start, err := d.Token()
	if err != nil || start != json.Delim('{') || !d.More() {
		return false
	}
	key, err := d.Token()
	if err != nil || key != "ok" {
		return false
	}
	var ok bool
	if d.Decode(&ok) != nil || !ok || d.More() {
		return false
	}
	end, err := d.Token()
	if err != nil || end != json.Delim('}') {
		return false
	}
	_, err = d.Token()
	return err == io.EOF
}

func opsRunProbeChoiceIndexValid(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return true
	}
	var index *int
	return json.Unmarshal(raw, &index) == nil && index != nil && *index == 0
}

func opsRunProbeUsageValid(raw json.RawMessage) bool {
	var counts map[string]json.RawMessage
	if json.Unmarshal(raw, &counts) != nil || counts == nil {
		return false
	}
	for _, key := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
		if _, err := strconv.ParseUint(string(bytes.TrimSpace(counts[key])), 10, 63); err != nil {
			return false
		}
	}
	return true
}
