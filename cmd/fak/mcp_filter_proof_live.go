package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

type mcpFilterLiveProof struct {
	Schema                  string           `json:"schema"`
	Verdict                 string           `json:"verdict"`
	Provider                string           `json:"provider"`
	Model                   string           `json:"model"`
	Active                  mcpFilterLiveArm `json:"active"`
	Control                 mcpFilterLiveArm `json:"control"`
	SecurityParityConfirmed bool             `json:"security_parity_confirmed"`
	Rollback                string           `json:"rollback"`
	Note                    string           `json:"note"`
}

type mcpFilterLiveArm struct {
	Mode                  string              `json:"mode"`
	Reason                string              `json:"reason"`
	Tasks                 []mcpFilterLiveTask `json:"tasks"`
	TaskSuccessRate       float64             `json:"task_success_rate"`
	SearchRecall          float64             `json:"search_recall"`
	FirstCallSuccess      float64             `json:"first_call_success"`
	ToolsBefore           int                 `json:"tools_before"`
	ToolsAfter            int                 `json:"tools_after"`
	DescriptorBytesBefore int                 `json:"descriptor_bytes_before"`
	DescriptorBytesAfter  int                 `json:"descriptor_bytes_after"`
	SavedDescriptorBytes  int                 `json:"saved_descriptor_bytes"`
	BailReasons           []string            `json:"bail_reasons"`
}

type mcpFilterLiveTask struct {
	ID               string `json:"id"`
	RequiredTool     string `json:"required_tool"`
	FirstCall        string `json:"first_call,omitempty"`
	FirstCallSuccess bool   `json:"first_call_success"`
	SearchUsed       bool   `json:"search_used"`
	SearchHit        bool   `json:"search_hit"`
	SelectedTool     string `json:"selected_tool,omitempty"`
	Success          bool   `json:"success"`
	Failure          string `json:"failure,omitempty"`
}

type openAIProofClient struct {
	endpoint    string
	apiKey      string
	model       string
	client      *http.Client
	lastCall    time.Time
	minInterval time.Duration
}

type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIProofResponse struct {
	Choices []struct {
		Message struct {
			Role      string           `json:"role"`
			Content   any              `json:"content"`
			ToolCalls []openAIToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func runLiveMCPFilterProof(ctx context.Context, srv *gateway.Server, endpoint, apiKey, model string) (mcpFilterLiveProof, error) {
	endpoint = strings.TrimRight(strings.TrimSpace(endpoint), "/")
	if endpoint == "" {
		endpoint = "https://api.openai.com/v1"
	}
	if model == "" {
		model = "gpt-4.1-mini"
	}
	c := &openAIProofClient{endpoint: endpoint + "/chat/completions", apiKey: apiKey, model: model, client: &http.Client{Timeout: 90 * time.Second}, minInterval: 25 * time.Second}
	if strings.Contains(endpoint, "127.0.0.1") || strings.Contains(endpoint, "localhost") {
		c.minInterval = 0
	}
	proof := mcpFilterLiveProof{
		Schema:   "fak-native-mcp-filter-live-proof/1",
		Verdict:  "NOT_YET",
		Provider: "openai-compatible",
		Model:    model,
		Rollback: "FAK_ABLATE_MCP_TOOL_FILTER=1",
		Note:     "Held synthetic tool-selection tasks; success means the model selected the required real MCP route, not that a side-effecting tool was executed.",
	}
	var err error
	proof.Active, err = runMCPFilterLiveArm(ctx, srv, c, false)
	if err != nil {
		return proof, fmt.Errorf("active arm: %w", err)
	}
	proof.Control, err = runMCPFilterLiveArm(ctx, srv, c, true)
	if err != nil {
		return proof, fmt.Errorf("control arm: %w", err)
	}
	// The existing proof checks the exposure predicate against both discovery and
	// invocation. Carry that independently witnessed result into this live run.
	proof.SecurityParityConfirmed = srv.NativeMCPFilterProof().SecurityParityConfirmed
	if proof.Active.TaskSuccessRate >= proof.Control.TaskSuccessRate &&
		proof.Active.SearchRecall == 1 && proof.Active.FirstCallSuccess >= proof.Control.FirstCallSuccess &&
		proof.Active.SavedDescriptorBytes > 0 && proof.SecurityParityConfirmed {
		proof.Verdict = "PASS"
	}
	return proof, nil
}

func runMCPFilterLiveArm(ctx context.Context, srv *gateway.Server, c *openAIProofClient, control bool) (mcpFilterLiveArm, error) {
	descriptors, receipt := srv.MCPToolListSnapshot(control)
	arm := mcpFilterLiveArm{
		Mode: receipt.Mode, Reason: receipt.Reason,
		ToolsBefore: receipt.ToolsBefore, ToolsAfter: receipt.ToolsAfter,
		DescriptorBytesBefore: receipt.DescriptorBytesBefore, DescriptorBytesAfter: receipt.DescriptorBytesAfter,
		SavedDescriptorBytes: receipt.SavedBytes,
	}
	if receipt.Mode == "bypass" && !control {
		arm.BailReasons = append(arm.BailReasons, receipt.Reason)
	}
	tasks := []struct{ id, prompt, tool string }{
		{"memory", "Identify the fak memory drivers. Use the available tools; do not answer from general knowledge.", "fak_memory_drivers"},
		{"context", "Change the fak context budget. Use the available tools; do not answer from general knowledge.", "fak_context_change"},
		{"features", "Query fak's available features. Use the available tools; do not answer from general knowledge.", "fak_feature_query"},
	}
	var successes, recalls, firsts int
	for _, tc := range tasks {
		result, err := runMCPFilterLiveTask(ctx, srv, c, descriptors, control, tc.id, tc.prompt, tc.tool)
		if err != nil {
			return arm, err
		}
		arm.Tasks = append(arm.Tasks, result)
		if result.Success {
			successes++
		}
		if !result.SearchUsed || result.SearchHit {
			recalls++
		}
		if result.FirstCallSuccess {
			firsts++
		}
	}
	n := float64(len(tasks))
	arm.TaskSuccessRate = float64(successes) / n
	arm.SearchRecall = float64(recalls) / n
	arm.FirstCallSuccess = float64(firsts) / n
	return arm, nil
}

func runMCPFilterLiveTask(ctx context.Context, srv *gateway.Server, c *openAIProofClient, descriptors []map[string]any, control bool, id, prompt, required string) (mcpFilterLiveTask, error) {
	result := mcpFilterLiveTask{ID: id, RequiredTool: required}
	messages := []map[string]any{{"role": "system", "content": "Select and call tools; never merely describe a call. If the needed tool is absent, use fak_tools_search first."}, {"role": "user", "content": prompt}}
	resp, err := c.call(ctx, messages, descriptors)
	if err != nil {
		result.Failure = classifyLiveProofError(err)
		return result, nil
	}
	if len(resp.Choices) == 0 || len(resp.Choices[0].Message.ToolCalls) == 0 {
		result.Failure = "no_first_tool_call"
		return result, nil
	}
	call := resp.Choices[0].Message.ToolCalls[0]
	result.FirstCall = call.Function.Name
	result.FirstCallSuccess = call.Function.Name == required || (!control && call.Function.Name == "fak_tools_search")
	if call.Function.Name == required {
		result.SelectedTool, result.Success = required, true
		return result, nil
	}
	if call.Function.Name != "fak_tools_search" {
		result.Failure = "wrong_first_tool"
		return result, nil
	}
	result.SearchUsed = true
	var args struct {
		Query string `json:"query"`
	}
	if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil || strings.TrimSpace(args.Query) == "" {
		result.Failure = "invalid_search_arguments"
		return result, nil
	}
	search, err := srv.MCPToolSearchSnapshot(args.Query, "full")
	if err != nil {
		return result, err
	}
	var discovered map[string]any
	for _, tool := range search.Tools {
		name, _ := tool["name"].(string)
		if name == required {
			result.SearchHit = true
			discovered = tool
			break
		}
	}
	if !result.SearchHit {
		result.Failure = "search_miss"
		return result, nil
	}
	assistant := map[string]any{"role": "assistant", "content": resp.Choices[0].Message.Content, "tool_calls": resp.Choices[0].Message.ToolCalls}
	searchJSON, _ := json.Marshal(search)
	messages = append(messages, assistant, map[string]any{"role": "tool", "tool_call_id": call.ID, "content": string(searchJSON)})
	resp2, err := c.call(ctx, messages, append(descriptors, discovered))
	if err != nil {
		result.Failure = classifyLiveProofError(err)
		return result, nil
	}
	if len(resp2.Choices) == 0 || len(resp2.Choices[0].Message.ToolCalls) == 0 {
		result.Failure = "no_post_search_tool_call"
		return result, nil
	}
	result.SelectedTool = resp2.Choices[0].Message.ToolCalls[0].Function.Name
	result.Success = result.SelectedTool == required
	if !result.Success {
		result.Failure = "wrong_post_search_tool"
	}
	return result, nil
}

func classifyLiveProofError(err error) string {
	text := strings.ToLower(err.Error())
	switch {
	case strings.Contains(text, "429"), strings.Contains(text, "rate limit"):
		return "provider_rate_limited"
	case strings.Contains(text, "tool"), strings.Contains(text, "function"):
		return "provider_rejected_tool_call"
	default:
		return "provider_request_failed"
	}
}

func (c *openAIProofClient) call(ctx context.Context, messages []map[string]any, descriptors []map[string]any) (openAIProofResponse, error) {
	tools := make([]map[string]any, 0, len(descriptors))
	for _, descriptor := range descriptors {
		fn := map[string]any{"name": descriptor["name"], "description": descriptor["description"], "parameters": descriptor["inputSchema"]}
		tools = append(tools, map[string]any{"type": "function", "function": fn})
	}
	body, _ := json.Marshal(map[string]any{"model": c.model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0, "max_tokens": 128})
	for attempt := 0; attempt < 3; attempt++ {
		// Keep a tuned live control from self-inducing provider TPM failures: the
		// full descriptor arm is roughly 20 KiB per request.
		if wait := c.minInterval - time.Since(c.lastCall); wait > 0 {
			select {
			case <-ctx.Done():
				return openAIProofResponse{}, ctx.Err()
			case <-time.After(wait):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
		if err != nil {
			return openAIProofResponse{}, err
		}
		if c.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := c.client.Do(req)
		c.lastCall = time.Now()
		if err != nil {
			return openAIProofResponse{}, err
		}
		raw, readErr := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if readErr != nil {
			return openAIProofResponse{}, readErr
		}
		var decoded openAIProofResponse
		if err := json.Unmarshal(raw, &decoded); err != nil {
			return decoded, fmt.Errorf("decode HTTP %d: %w", resp.StatusCode, err)
		}
		if resp.StatusCode == http.StatusTooManyRequests && attempt < 2 {
			wait := 12 * time.Second
			if seconds, parseErr := strconv.ParseFloat(resp.Header.Get("Retry-After"), 64); parseErr == nil && seconds > 0 {
				wait = time.Duration(seconds*float64(time.Second)) + time.Second
			}
			select {
			case <-ctx.Done():
				return decoded, ctx.Err()
			case <-time.After(wait):
				continue
			}
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			message := strings.TrimSpace(resp.Status)
			if decoded.Error != nil && decoded.Error.Message != "" {
				message = decoded.Error.Message
			}
			return decoded, fmt.Errorf("provider HTTP %d: %s", resp.StatusCode, message)
		}
		return decoded, nil
	}
	return openAIProofResponse{}, errors.New("provider retry budget exhausted")
}
func liveProofDefaults() (endpoint, apiKey, model string) {
	return os.Getenv("OPENAI_BASE_URL"), os.Getenv("OPENAI_API_KEY"), os.Getenv("FAK_MCP_FILTER_PROOF_MODEL")
}
