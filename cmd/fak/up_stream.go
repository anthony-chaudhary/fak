package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// turnkeyChatStream owns the point after which an HTTP error status can no
// longer be returned. It opens lazily on the first safe content fragment, so a
// planner failure before any output retains the buffered endpoint's status.
type turnkeyChatStream struct {
	w       http.ResponseWriter
	flush   http.Flusher
	id      string
	created int64
	model   string
	started bool
	content strings.Builder
}

func newTurnkeyChatStream(w http.ResponseWriter, model string) *turnkeyChatStream {
	flusher, _ := w.(http.Flusher)
	return &turnkeyChatStream{w: w, flush: flusher, id: fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()), created: time.Now().Unix(), model: model}
}

func (s *turnkeyChatStream) send(delta map[string]any, finish *string, usage *agent.Usage) error {
	chunk := map[string]any{
		"id": s.id, "object": "chat.completion.chunk", "created": s.created, "model": s.model,
		"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
	}
	if usage != nil {
		chunk["usage"] = *usage
	}
	raw, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(s.w, "data: %s\n\n", raw)
	if s.flush != nil {
		s.flush.Flush()
	}
	return err
}

func (s *turnkeyChatStream) open() error {
	if s.started {
		return nil
	}
	s.w.Header().Set("Content-Type", "text/event-stream")
	s.w.Header().Set("Cache-Control", "no-cache")
	s.w.Header().Set("Connection", "keep-alive")
	s.w.WriteHeader(http.StatusOK)
	s.started = true
	return s.send(map[string]any{"role": "assistant"}, nil, nil)
}

func (s *turnkeyChatStream) contentDelta(delta string) error {
	if delta == "" {
		return nil
	}
	if err := s.open(); err != nil {
		return err
	}
	if err := s.send(map[string]any{"content": delta}, nil, nil); err != nil {
		return err
	}
	s.content.WriteString(delta)
	return nil
}

func (s *turnkeyChatStream) fail(message, code string) {
	if !s.started {
		return
	}
	if message == "" {
		message = "inference error"
	}
	_ = writeTurnkeySSEData(s.w, map[string]any{"error": map[string]any{"message": message, "type": "server_error", "code": code}})
	if s.flush != nil {
		s.flush.Flush()
	}
}

// turnkeyCompletionStream is the LEGACY text-completion twin of turnkeyChatStream
// for POST /v1/completions: same lazy-open and in-band-failure contract, but each
// SSE chunk is a `text_completion` object whose choice carries a bare `text`
// fragment instead of a chat `delta`. Legacy clients (and the subagent fan-out
// harness) read this wire, not the chat one.
type turnkeyCompletionStream struct {
	w       http.ResponseWriter
	flush   http.Flusher
	id      string
	created int64
	model   string
	started bool
	content strings.Builder
}

func newTurnkeyCompletionStream(w http.ResponseWriter, model string) *turnkeyCompletionStream {
	flusher, _ := w.(http.Flusher)
	return &turnkeyCompletionStream{w: w, flush: flusher, id: fmt.Sprintf("cmpl-fak-%d", time.Now().UnixNano()), created: time.Now().Unix(), model: model}
}

func (s *turnkeyCompletionStream) send(text string, finish *string, usage *agent.Usage) error {
	chunk := map[string]any{
		"id": s.id, "object": "text_completion", "created": s.created, "model": s.model,
		"choices": []map[string]any{{"index": 0, "text": text, "finish_reason": finish}},
	}
	if usage != nil {
		chunk["usage"] = *usage
	}
	raw, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(s.w, "data: %s\n\n", raw)
	if s.flush != nil {
		s.flush.Flush()
	}
	return err
}

func (s *turnkeyCompletionStream) open() error {
	if s.started {
		return nil
	}
	s.w.Header().Set("Content-Type", "text/event-stream")
	s.w.Header().Set("Cache-Control", "no-cache")
	s.w.Header().Set("Connection", "keep-alive")
	s.w.WriteHeader(http.StatusOK)
	s.started = true
	return nil
}

func (s *turnkeyCompletionStream) contentDelta(delta string) error {
	if delta == "" {
		return nil
	}
	if err := s.open(); err != nil {
		return err
	}
	if err := s.send(delta, nil, nil); err != nil {
		return err
	}
	s.content.WriteString(delta)
	return nil
}

func (s *turnkeyCompletionStream) fail(message, code string) {
	if !s.started {
		return
	}
	if message == "" {
		message = "inference error"
	}
	_ = writeTurnkeySSEData(s.w, map[string]any{"error": map[string]any{"message": message, "type": "server_error", "code": code}})
	if s.flush != nil {
		s.flush.Flush()
	}
}

// handleCompletionsStream mirrors handleChatCompletionsStream for the legacy wire:
// it drives the same StreamingPlanner per-token path, forwards each fragment as a
// legacy `text_completion` chunk, and reconciles the streamed prefix against the
// adjudicated completion before emitting the terminal finish/usage chunk.
func (s *turnkeyServer) handleCompletionsStream(w http.ResponseWriter, r *http.Request, req gateway.ChatRequest, modelID string, sp agent.StreamingPlanner) {
	stream := newTurnkeyCompletionStream(w, modelID)
	opts := append(turnkeyChatSampleOpts(req, s.plan.ContextBudgetTokens), agent.WithPerTokenStream(true))
	comp, err := sp.CompleteStream(r.Context(), stream.contentDelta, req.Messages, req.Tools, opts...)
	if err != nil {
		if !stream.started {
			writeTurnkeyInferenceError(w, err)
		} else if !errors.Is(err, context.Canceled) && r.Context().Err() == nil {
			stream.fail("inference error", "inference_error")
		}
		return
	}
	if r.Context().Err() != nil {
		return
	}
	if comp == nil {
		if !stream.started {
			writeTurnkeyInferenceError(w, errors.New("planner returned no completion"))
		} else {
			stream.fail("inference error", "inference_error")
		}
		return
	}
	if err := stream.open(); err != nil {
		return
	}
	remaining, err := turnkeyStreamRemainder(stream.content.String(), comp.Message.Content)
	if err != nil {
		stream.fail("streamed content did not match completed response", "stream_content_mismatch")
		return
	}
	if remaining != "" {
		if err := stream.contentDelta(remaining); err != nil {
			return
		}
	}
	if r.Context().Err() != nil {
		return
	}
	finish := comp.FinishReason
	if finish == "" {
		finish = "stop"
	}
	usage := comp.Usage
	if usage.CompletionTokens == 0 && comp.Message.Content != "" {
		usage.CompletionTokens = len(strings.Fields(comp.Message.Content))
		if usage.CompletionTokens == 0 {
			usage.CompletionTokens = 1
		}
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	atomic.AddInt64(&s.requestCount, 1)
	atomic.AddInt64(&s.totalTokens, int64(usage.CompletionTokens))
	if err := stream.send("", &finish, &usage); err != nil {
		return
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if stream.flush != nil {
		stream.flush.Flush()
	}
}

func writeTurnkeySSEData(w io.Writer, value any) error {
	raw, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", raw)
	return err
}

func (s *turnkeyServer) handleChatCompletionsStream(w http.ResponseWriter, r *http.Request, req gateway.ChatRequest, modelID string, sp agent.StreamingPlanner) {
	stream := newTurnkeyChatStream(w, modelID)
	// The turnkey path streams TRUE per-token deltas by default (no env var): ask the
	// planner to forward each decoded prose piece live. toolSpanGuard inside
	// CompleteStream holds explicit tool-call spans back for post-decode adjudication.
	opts := append(turnkeyChatSampleOpts(req, s.plan.ContextBudgetTokens), agent.WithPerTokenStream(true))
	comp, err := sp.CompleteStream(r.Context(), stream.contentDelta, req.Messages, req.Tools, opts...)
	if err != nil {
		if !stream.started {
			writeTurnkeyInferenceError(w, err)
		} else if !errors.Is(err, context.Canceled) && r.Context().Err() == nil {
			stream.fail("inference error", "inference_error")
		}
		return
	}
	if r.Context().Err() != nil {
		return
	}
	if comp == nil {
		if !stream.started {
			writeTurnkeyInferenceError(w, errors.New("planner returned no completion"))
		} else {
			stream.fail("inference error", "inference_error")
		}
		return
	}
	if comp.ToolCallsDropped && len(comp.Message.ToolCalls) == 0 {
		if !stream.started {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "upstream tool-call format not recognized; refusing to skip adjudication", "type": "server_error", "code": "tool_call_conformance"}})
		} else {
			stream.fail("upstream tool-call format not recognized; refusing to skip adjudication", "tool_call_conformance")
		}
		return
	}
	if err := stream.open(); err != nil {
		return
	}
	remaining, err := turnkeyStreamRemainder(stream.content.String(), comp.Message.Content)
	if err != nil {
		stream.fail("streamed content did not match completed response", "stream_content_mismatch")
		return
	}
	if remaining != "" {
		if err := stream.contentDelta(remaining); err != nil {
			return
		}
	}
	if len(comp.Message.ToolCalls) > 0 {
		calls := make([]gateway.ChatDeltaToolCall, 0, len(comp.Message.ToolCalls))
		for i, call := range comp.Message.ToolCalls {
			calls = append(calls, gateway.ChatDeltaToolCall{Index: i, ID: call.ID, Type: call.Type, Function: call.Function})
		}
		if err := stream.send(map[string]any{"tool_calls": calls}, nil, nil); err != nil {
			return
		}
	}
	if r.Context().Err() != nil {
		return
	}
	finish := comp.FinishReason
	if len(comp.Message.ToolCalls) > 0 {
		finish = "tool_calls"
	} else if finish == "" {
		finish = "stop"
	}
	usage := comp.Usage
	if usage.CompletionTokens == 0 && comp.Message.Content != "" {
		usage.CompletionTokens = len(strings.Fields(comp.Message.Content))
		if usage.CompletionTokens == 0 {
			usage.CompletionTokens = 1
		}
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	atomic.AddInt64(&s.requestCount, 1)
	atomic.AddInt64(&s.totalTokens, int64(usage.CompletionTokens))
	if err := stream.send(map[string]any{}, &finish, &usage); err != nil {
		return
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if stream.flush != nil {
		stream.flush.Flush()
	}
}

func turnkeyStreamRemainder(streamed, completed string) (string, error) {
	if streamed == "" {
		return completed, nil
	}
	if strings.HasPrefix(completed, streamed) {
		return strings.TrimPrefix(completed, streamed), nil
	}
	return "", errors.New("streamed content is not a prefix of completed response")
}
