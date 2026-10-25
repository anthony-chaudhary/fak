package agentxbench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Runner orchestrates the execution of multi-turn agent sessions.
type Runner struct {
	client *http.Client
}

// NewRunner returns a new Runner with an optional custom http.Client.
func NewRunner(client *http.Client) *Runner {
	if client == nil {
		client = &http.Client{Timeout: 120 * time.Second}
	}
	return &Runner{client: client}
}

// Run executes the AgentX benchmark according to the provided Config.
func (r *Runner) Run(ctx context.Context, cfg Config) (*AgentXReceipt, error) {
	if cfg.AgentCount <= 0 {
		cfg.AgentCount = 1
	}
	if cfg.TurnsPerAgent <= 0 {
		cfg.TurnsPerAgent = 1
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 32
	}
	if cfg.Model == "" {
		cfg.Model = "Qwen3.8-27B-Q4_K_M"
	}
	mode := ExecutionModeLive
	if cfg.MockExecution {
		mode = ExecutionModeMock
		if cfg.Engine == "" {
			cfg.Engine = "selfcheck"
		}
		if cfg.Hardware == "" {
			cfg.Hardware = "selfcheck (no endpoint)"
		}
	}
	if strings.TrimSpace(cfg.Engine) == "" || strings.TrimSpace(cfg.Hardware) == "" {
		return nil, fmt.Errorf("live run needs the engine and hardware it measures; no default is assumed")
	}

	benchmarkID := fmt.Sprintf("agentx-%d-%s", time.Now().Unix(), cfg.Model)
	startTime := time.Now()

	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		requests []RequestRecord
	)

	requests = make([]RequestRecord, 0, cfg.AgentCount*cfg.TurnsPerAgent)

	for a := 0; a < cfg.AgentCount; a++ {
		agentID := fmt.Sprintf("agent-%d", a+1)
		wg.Add(1)

		go func(agentIdx int, aid string) {
			defer wg.Done()

			agentRng := rand.New(rand.NewSource(cfg.DeterministicSeed + int64(agentIdx)*1000))

			for turn := 0; turn < cfg.TurnsPerAgent; turn++ {
				reqID := fmt.Sprintf("%s-t%d", aid, turn)
				var record RequestRecord

				if cfg.MockExecution {
					record = r.runMockTurn(agentRng, aid, reqID, turn, cfg)
				} else {
					record = r.runLiveTurn(ctx, aid, reqID, turn, cfg)
				}

				mu.Lock()
				requests = append(requests, record)
				mu.Unlock()
			}
		}(a, agentID)
	}

	wg.Wait()
	totalWallMS := float64(time.Since(startTime).Nanoseconds()) / 1e6
	if cfg.MockExecution || totalWallMS < 1.0 {
		agentTimes := make(map[string]float64)
		for _, req := range requests {
			agentTimes[req.AgentID] += req.ClientPhases.TotalLifecycleMS
		}
		var maxAgentTime float64
		for _, t := range agentTimes {
			if t > maxAgentTime {
				maxAgentTime = t
			}
		}
		if maxAgentTime > 0 {
			totalWallMS = maxAgentTime
		}
	}

	agg := Aggregate(requests, totalWallMS)

	receipt := &AgentXReceipt{
		Schema:           SchemaIdentifier,
		BenchmarkID:      benchmarkID,
		TimestampISO:     startTime.UTC().Format(time.RFC3339),
		ExecutionMode:    mode,
		Workload:         WorkloadSyntheticTurns,
		TokenCountSource: receiptTokenSource(requests),
		Hardware:         cfg.Hardware,
		Endpoint:         cfg.EndpointURL,
		Model:            cfg.Model,
		Engine:           cfg.Engine,
		AgentCount:       cfg.AgentCount,
		TurnsPerAgent:    cfg.TurnsPerAgent,
		Aggregated:       agg,
		Requests:         requests,
		ValidationStatus: "PENDING",
	}

	validationErrs := ValidateReceipt(receipt)
	receipt.ValidationErrors = validationErrs
	switch {
	case len(validationErrs) > 0:
		receipt.ValidationStatus = StatusVerifiedFail
	case mode == ExecutionModeMock:
		receipt.ValidationStatus = StatusSimulated
	default:
		receipt.ValidationStatus = StatusVerifiedPass
	}

	return receipt, nil
}

func (r *Runner) runMockTurn(rng *rand.Rand, agentID, reqID string, turn int, cfg Config) RequestRecord {
	queueWaitMS := 5.0 + rng.Float64()*10.0

	promptTokens := 1024 + turn*256
	cachedTokens := 0
	if turn > 0 {
		cachedTokens = 1024 + (turn-1)*256
	}

	var ttftMS float64
	if turn == 0 || cachedTokens == 0 {
		// Cold TTFT: ~1200ms
		ttftMS = 1100.0 + rng.Float64()*200.0
	} else {
		// Warm prefix hit TTFT: ~250ms (4.8x speedup)
		ttftMS = 220.0 + rng.Float64()*60.0
	}

	completionTokens := cfg.MaxTokens
	timestamps := make([]int64, completionTokens)
	nowNano := time.Now().UnixNano()

	// Initial token arrives at TTFT
	timestamps[0] = nowNano + int64(ttftMS*1e6)

	// Inter-token latency: ~15ms per token on A100 Q8 decode (approx 66 tok/s decode)
	for i := 1; i < completionTokens; i++ {
		itlMS := 12.0 + rng.Float64()*6.0
		timestamps[i] = timestamps[i-1] + int64(itlMS*1e6)
	}

	activeGenMS := float64(timestamps[completionTokens-1]-timestamps[0]) / 1e6
	execMS := ttftMS + activeGenMS
	evalMS := 2.0 + rng.Float64()*3.0
	lifecycleMS := queueWaitMS + execMS + evalMS

	interactivity := ComputeInteractivity(timestamps, ttftMS, execMS)

	return RequestRecord{
		RequestID:        reqID,
		AgentID:          agentID,
		TurnIndex:        turn,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		CachedTokens:     cachedTokens,
		CacheHitRatio:    float64(cachedTokens) / float64(promptTokens),
		ClientPhases: ClientPhases{
			QueueWaitMS:      queueWaitMS,
			AgentExecutionMS: execMS,
			EvaluationMS:     evalMS,
			TotalLifecycleMS: lifecycleMS,
		},
		TokenCountSource:        TokenCountsEstimated,
		Interactivity:           interactivity,
		TokenTimestampsUnixNano: timestamps,
		Success:                 true,
		ResponseText:            MockResponseText,
	}
}

// receiptTokenSource is the weakest source among successful requests, so a
// receipt reads server_usage only when every request carried full usage.
func receiptTokenSource(reqs []RequestRecord) string {
	rank := map[string]int{TokenCountsServerUsage: 2, TokenCountsServerUsageNoCacheRpt: 1}
	src, seen := TokenCountsEstimated, false
	for _, req := range reqs {
		if !req.Success {
			continue
		}
		if !seen || rank[req.TokenCountSource] < rank[src] {
			src = req.TokenCountSource
		}
		seen = true
	}
	if _, ok := rank[src]; !ok {
		return TokenCountsEstimated
	}
	return src
}

func (r *Runner) runLiveTurn(ctx context.Context, agentID, reqID string, turn int, cfg Config) RequestRecord {
	// Each agent issues its turns back to back; there is no client-side queue.
	queueWaitMS := 0.0

	execStart := time.Now()

	prompt := cfg.SharedPrefix
	if prompt == "" {
		prompt = "You are an autonomous coding agent solving system optimization tasks."
	}
	prompt += fmt.Sprintf("\nAgent ID: %s, Turn: %d. Provide structured status.", agentID, turn)

	reqPayload := map[string]any{
		"model":          cfg.Model,
		"messages":       []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":     cfg.MaxTokens,
		"temperature":    cfg.Temperature,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	bodyBytes, err := json.Marshal(reqPayload)
	if err != nil {
		return RequestRecord{
			RequestID: reqID,
			AgentID:   agentID,
			TurnIndex: turn,
			Success:   false,
			Error:     fmt.Sprintf("json marshal: %v", err),
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", cfg.EndpointURL+"/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return RequestRecord{
			RequestID: reqID,
			AgentID:   agentID,
			TurnIndex: turn,
			Success:   false,
			Error:     fmt.Sprintf("http request create: %v", err),
		}
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(httpReq)
	if err != nil {
		return RequestRecord{
			RequestID: reqID,
			AgentID:   agentID,
			TurnIndex: turn,
			Success:   false,
			Error:     fmt.Sprintf("http do: %v", err),
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return RequestRecord{
			RequestID: reqID,
			AgentID:   agentID,
			TurnIndex: turn,
			Success:   false,
			Error:     fmt.Sprintf("http status %d: %s", resp.StatusCode, string(respBody)),
		}
	}

	var timestamps []int64
	var ttftMS float64
	var responseBuilder strings.Builder
	var usage *streamUsage

	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			break
		}
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "data: ") {
			data := strings.TrimPrefix(trimmed, "data: ")
			if data == "[DONE]" {
				break
			}
			nowNano := time.Now().UnixNano()
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content          string `json:"content"`
						ReasoningContent string `json:"reasoning_content"`
					} `json:"delta"`
				} `json:"choices"`
				Usage *streamUsage `json:"usage"`
			}
			if json.Unmarshal([]byte(data), &chunk) == nil {
				if chunk.Usage != nil {
					usage = chunk.Usage
				}
				// Role-only and usage-only frames carry no token; TTFT and ITL
				// start at the first generated content.
				if len(chunk.Choices) > 0 {
					delta := chunk.Choices[0].Delta
					if delta.Content != "" || delta.ReasoningContent != "" {
						if len(timestamps) == 0 {
							ttftMS = float64(max(time.Since(execStart).Nanoseconds(), 1000)) / 1e6
						}
						timestamps = append(timestamps, nowNano)
						responseBuilder.WriteString(delta.Content)
					}
				}
			}
		}
		if err == io.EOF {
			break
		}
	}

	execMS := float64(time.Since(execStart).Microseconds()) / 1000.0
	evalStart := time.Now()
	// Validation of tool call response
	evalMS := float64(time.Since(evalStart).Microseconds()) / 1000.0
	lifecycleMS := queueWaitMS + execMS + evalMS

	numTokens := len(timestamps)
	if numTokens == 0 {
		return RequestRecord{
			RequestID: reqID,
			AgentID:   agentID,
			TurnIndex: turn,
			Success:   false,
			Error:     "ZERO_TOKENS_RECEIVED: stream completed without emitting tokens",
		}
	}

	interactivity := ComputeInteractivity(timestamps, ttftMS, execMS)

	// Without the server's usage block, counts are estimated and no cache hit
	// is claimed.
	promptTokens, completionTokens, cachedTokens := max(len(prompt)/4, 1), numTokens, 0
	source := TokenCountsEstimated
	if usage != nil && usage.PromptTokens > 0 {
		promptTokens, completionTokens = usage.PromptTokens, usage.CompletionTokens
		source = TokenCountsServerUsageNoCacheRpt
		if usage.PromptTokensDetails != nil {
			cachedTokens = usage.PromptTokensDetails.CachedTokens
			source = TokenCountsServerUsage
		}
	}

	return RequestRecord{
		RequestID:        reqID,
		AgentID:          agentID,
		TurnIndex:        turn,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		CachedTokens:     cachedTokens,
		CacheHitRatio:    float64(cachedTokens) / float64(promptTokens),
		TokenCountSource: source,
		ClientPhases: ClientPhases{
			QueueWaitMS:      queueWaitMS,
			AgentExecutionMS: execMS,
			EvaluationMS:     evalMS,
			TotalLifecycleMS: lifecycleMS,
		},
		Interactivity:           interactivity,
		TokenTimestampsUnixNano: timestamps,
		Success:                 true,
		ResponseText:            responseBuilder.String(),
	}
}

// streamUsage is the OpenAI-compatible usage block a server sends on the final
// stream frame when stream_options.include_usage is set.
type streamUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}
