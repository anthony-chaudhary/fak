package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func (s *turnkeyServer) handleHealthz(w http.ResponseWriter, r *http.Request) {
	isReady, state, _ := s.readiness()
	var nativeStartup *turnkeyNativeStartup
	if s.native != nil {
		nativeStartup = &s.native.Startup
	}
	w.Header().Set("Content-Type", "application/json")
	// Liveness is preserved for a live-but-not-ready process: /healthz answers
	// 200 while the process can still respond, but the body's status must be
	// truthful about readiness — "warming_up" until the boot warmup gate
	// completes, "stopping" once shutdown began, otherwise "ok".
	w.WriteHeader(http.StatusOK)
	body := map[string]any{
		"ok":             isReady,
		"status":         state,
		"ready":          isReady,
		"mode":           "turnkey",
		"engine":         s.engineID,
		"tier":           s.plan.Tier.Name,
		"model":          s.plan.Tier.ModelID,
		"headroom_ratio": s.plan.HeadroomRatio,
		"native_startup": nativeStartup,
		"live_residency": s.liveResidencyReport(),
		"agent_warm":     s.agentWarm.agentWarmBlock(),
		"sessions":       s.capacityStats(),
	}
	// fak#13567: resident weight bytes by store + the LIVE lm_head route, the same shape the
	// gateway /healthz reports. Absent (not zero) when no native model is loaded.
	if s.native != nil {
		if wr := residentWeightsLiveView(s.native.Model, s.native.Startup.Resident); wr != nil {
			body["resident_weights"] = wr
		}
	}
	_ = json.NewEncoder(w).Encode(body)
}

// liveResidencyReport reads the LIVE Metal/CPU routing state at request time (#12875) instead of
// replaying the snapshot frozen into native.Startup at load. The frozen `native_startup` stays for
// backward compatibility; this block is the truthful answer to "did decode run on Metal on this
// run?". It re-samples device-resident weight counts (a lazy later upload is now visible) and
// reports the model's process-wide promised-CPU-fallback tally, which survives the per-request
// session churn. It returns nil when no native model is loaded (mock/custom-server paths), so the
// key is absent rather than a fabricated zero.
func (s *turnkeyServer) liveResidencyReport() map[string]any {
	if s.native == nil || s.native.Model == nil {
		return nil
	}
	q6k, q8 := s.native.Model.RefreshMetalResidency()
	fallbacks := s.native.Model.MetalFallbackSnapshot()
	return map[string]any{
		"metal_live_q8_weights":    q8,
		"metal_live_q6_weights":    q6k,
		"promised_cpu_fallbacks":   fallbacks.Total,
		"fallbacks_observed":       fallbacks.Observed,
		"fallbacks_by_route":       fallbacks.ByRoute,
		"startup_q8_weights":       s.native.Startup.MetalLiveQ8Weights,
		"startup_q6_weights":       s.native.Startup.MetalLiveQ6Weights,
		"metal_q8_residency_error": s.native.Startup.MetalQ8ResidencyError,
		// Re-read per probe: a later Metal Q6_K promotion moves the head (fak#13567).
		"lm_head": s.native.Model.LMHeadRoute(),
	}
}

func (s *turnkeyServer) handleReadyz(w http.ResponseWriter, r *http.Request) {
	isReady, state, reason := s.readiness()
	w.Header().Set("Content-Type", "application/json")
	if !isReady {
		w.Header().Set("Retry-After", "1")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": state,
			"ready":  false,
			"reason": reason,
		})
		return
	}
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status": "ready",
		"ready":  true,
		"mode":   "turnkey",
	})
}

func (s *turnkeyServer) handleModels(w http.ResponseWriter, r *http.Request) {
	row := map[string]any{
		"id":         s.plan.Tier.ModelID,
		"object":     "model",
		"created":    time.Now().Unix(),
		"owned_by":   "fak",
		"permission": []any{},
	}
	if contextWindow := turnkeyPlannerContextWindow(s.planner); contextWindow > 0 {
		row["context_length"] = contextWindow
		row["context_window"] = contextWindow
		row["max_output_tokens"] = turnkeyMaxOutputTokens(uint64(contextWindow))
	}
	if turnkeyPromptEncodingAvailable(s) {
		row["fak_capabilities"] = map[string]any{
			"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"},
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   []map[string]any{row},
	})
}

// handleCompletions serves the LEGACY text-completion wire the turnkey server
// previously omitted: it wraps the request prompt as a single user message and reuses
// the same planner path as the chat route, then emits `text_completion` frames (bare
// `text`, never a chat delta). This is the surface vLLM, SGLang, llama.cpp-server,
// and the subagent fan-out harness all speak.
func (s *turnkeyServer) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	switch s.admitChatRequest() {
	case admissionGranted:
		// admitted
	case admissionAtCapacity:
		writeTurnkeyBackpressure(w, "server_at_capacity", s.capacity().MaxSessions)
		return
	default:
		http.Error(w, "server stopping", http.StatusServiceUnavailable)
		return
	}
	defer s.endChatRequest()

	var req turnkeyCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	prompt := turnkeyNormalizePrompt(req.Prompt)
	if strings.TrimSpace(prompt) == "" {
		http.Error(w, "prompt: field required", http.StatusBadRequest)
		return
	}
	if req.MaxTokens < 0 {
		http.Error(w, "max_tokens: must be a positive integer", http.StatusBadRequest)
		return
	}

	chatReq := gateway.ChatRequest{
		Model:       req.Model,
		Messages:    []agent.Message{{Role: agent.RoleUser, Content: prompt}},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}
	modelID := s.plan.Tier.ModelID
	if chatReq.Model != "" {
		modelID = chatReq.Model
	}

	if chatReq.Stream && !s.mock {
		if sp, ok := s.planner.(agent.StreamingPlanner); ok && sp.StreamingSupported() {
			s.handleCompletionsStream(w, r, chatReq, modelID, sp)
			return
		}
	}

	finishReason := "stop"
	answer := agent.Message{Role: agent.RoleAssistant}
	var usage agent.Usage

	if !s.mock && s.planner != nil {
		sampleOpts := turnkeyChatSampleOpts(chatReq, s.plan.ContextBudgetTokens)
		comp, err := s.planner.Complete(turnkeyRequestContext(r.Context()), chatReq.Messages, nil, sampleOpts...)
		if err != nil {
			writeTurnkeyInferenceError(w, err)
			return
		}
		answer = comp.Message
		usage = comp.Usage
		if comp.FinishReason != "" {
			finishReason = comp.FinishReason
		}
	} else {
		answer.Content = fmt.Sprintf("Turnkey %s completion on Apple Silicon. Processed: %s", s.plan.Tier.Name, prompt)
	}
	if usage.CompletionTokens == 0 && answer.Content != "" {
		usage.CompletionTokens = len(strings.Fields(answer.Content))
		if usage.CompletionTokens == 0 {
			usage.CompletionTokens = 1
		}
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	atomic.AddInt64(&s.requestCount, 1)
	atomic.AddInt64(&s.totalTokens, int64(usage.CompletionTokens))
	created := time.Now().Unix()
	cmplID := fmt.Sprintf("cmpl-fak-%d", time.Now().UnixNano())

	if chatReq.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		sendChunk := func(text string, finish *string) {
			chunk := map[string]any{
				"id": cmplID, "object": "text_completion", "created": created, "model": modelID,
				"choices": []map[string]any{{"index": 0, "text": text, "finish_reason": finish}},
			}
			if finish != nil {
				chunk["usage"] = usage
			}
			raw, _ := json.Marshal(chunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", raw)
			if flusher != nil {
				flusher.Flush()
			}
		}
		if answer.Content != "" {
			sendChunk(answer.Content, nil)
		}
		stop := finishReason
		sendChunk("", &stop)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(gateway.CompletionResponse{
		ID:      cmplID,
		Object:  "text_completion",
		Created: created,
		Model:   modelID,
		Choices: []gateway.CompletionChoice{{Index: 0, Text: answer.Content, FinishReason: &finishReason}},
		Usage:   usage,
	})
}

func writeTurnkeyInferenceError(w http.ResponseWriter, err error) {
	var contextErr *agent.InKernelContextLengthError
	if errors.As(err, &contextErr) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
			"message": contextErr.Error(), "type": "invalid_request_error", "code": "context_length_exceeded",
		}})
		return
	}
	// A native Metal command-buffer wait that exceeded its bound is a local,
	// retryable resource stall — not a generic server fault. Surface it as 503
	// with a distinct code so a client can retry or shed load instead of
	// treating it as an opaque 500. The observation seam may wrap the typed
	// error, so errors.As (not a type assertion) recovers it.
	var stall metalgemm.MetalCommandBufferStallError
	if errors.As(err, &stall) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
			"message": fmt.Sprintf("Metal command buffer stall during %s: waited %.3fms at/over %.3fms limit",
				stall.Operation, stall.WaitedMilliseconds, stall.LimitMilliseconds),
			"type": "server_error", "code": "metal_command_buffer_stalled",
		}})
		return
	}
	// An in-kernel capacity refusal (the #13267 host-memory arm or the device precheck)
	// or a recovered device OOM is a local, retryable resource condition: the request was
	// declined before it could grow the resident server into a jetsam kill. Surface it as
	// 503 + Retry-After with the same in_kernel_oom code the gateway uses, not an opaque 500.
	var capErr *agent.InKernelCapacityError
	var oomErr *agent.InKernelOOMError
	if errors.As(err, &capErr) || errors.As(err, &oomErr) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", turnkeyCapacityRetryAfterSeconds)
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
			"message": err.Error() + "; retry later, or reduce the prompt/context size or max_tokens",
			"type":    "server_error", "code": "in_kernel_oom",
		}})
		return
	}
	http.Error(w, fmt.Sprintf("inference error: %v", err), http.StatusInternalServerError)
}

// turnkeyCapacityRetryAfterSeconds is the Retry-After hint on an in-kernel capacity
// refusal: long enough for an in-flight turn to finish and release its reservation.
const turnkeyCapacityRetryAfterSeconds = "5"
