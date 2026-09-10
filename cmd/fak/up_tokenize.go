package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const turnkeyTokenizeBodyLimit = 1 << 20

type turnkeyPromptEncoder interface {
	EncodePrompt(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (agent.PromptEncoding, error)
}

func turnkeyChatSampleOpts(req gateway.ChatRequest, contextTokens uint64) []agent.SampleOpt {
	var opts []agent.SampleOpt
	if req.MaxTokens > 0 {
		opts = append(opts, agent.WithMaxTokens(min(req.MaxTokens, turnkeyMaxOutputTokens(contextTokens))))
	}
	if req.Temperature != nil {
		opts = append(opts, agent.WithTemperature(req.Temperature))
	}
	return append(opts,
		agent.WithTopP(req.TopP),
		agent.WithToolChoice(req.ToolChoice),
		agent.WithResponseFormat(req.ResponseFormat),
		agent.WithLogitBias(req.LogitBias),
		agent.WithFrequencyPenalty(req.FrequencyPenalty),
		agent.WithPresencePenalty(req.PresencePenalty),
	)
}

func turnkeyPromptEncodingAvailable(s *turnkeyServer) bool {
	if s == nil || s.mock || s.planner == nil {
		return false
	}
	_, ok := s.planner.(turnkeyPromptEncoder)
	return ok
}

func (s *turnkeyServer) handleTokenize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.beginChatRequest() {
		http.Error(w, "server stopping", http.StatusServiceUnavailable)
		return
	}
	defer s.endChatRequest()
	if s.mock || s.planner == nil {
		http.Error(w, "native prompt encoding unavailable", http.StatusNotImplemented)
		return
	}
	encoder, ok := s.planner.(turnkeyPromptEncoder)
	if !ok {
		http.Error(w, "native prompt encoding unavailable", http.StatusNotImplemented)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, turnkeyTokenizeBodyLimit)
	var req gateway.ChatRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := ensureTurnkeyTokenizeEOF(decoder); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	servedModel := strings.TrimSpace(s.plan.Tier.ModelID)
	if requested := strings.TrimSpace(req.Model); requested != "" && requested != servedModel {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
			"message": "requested model does not match the loaded native model",
			"type":    "invalid_request_error", "code": "model_mismatch",
		}})
		return
	}
	encoding, err := encoder.EncodePrompt(r.Context(), req.Messages, req.Tools, turnkeyChatSampleOpts(req, s.plan.ContextBudgetTokens)...)
	if err != nil {
		writeTurnkeyInferenceError(w, err)
		return
	}
	if strings.TrimSpace(encoding.ModelID) != servedModel {
		http.Error(w, "native prompt encoding model identity mismatch", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(encoding)
}

func ensureTurnkeyTokenizeEOF(decoder *json.Decoder) error {
	var trailing any
	err := decoder.Decode(&trailing)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("multiple JSON values")
	}
	return err
}
