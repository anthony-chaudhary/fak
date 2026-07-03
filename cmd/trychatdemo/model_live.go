package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentdemo"
)

const (
	defaultModelProvider        = "openai"
	defaultResponsesEndpoint    = "https://api.openai.com/v1/responses"
	defaultModelAPIKeyEnv       = "OPENAI_API_KEY"
	defaultModelRung            = "openai-responses"
	defaultModelReasoningEffort = "low"
	defaultModelSelectionSource = "https://developers.openai.com/api/docs/guides/latest-model"
)

type modelArmConfig struct {
	Live            bool
	Provider        string
	Model           string
	Endpoint        string
	APIKeyEnv       string
	Rung            string
	AsOf            string
	SelectionSource string
	ReasoningEffort string
	Timeout         time.Duration
}

func (c *modelArmConfig) bindFlags(fs *flag.FlagSet) {
	c.Live = envBool("FAK_TRYCHAT_LIVE_MODEL")
	c.Provider = envOr("FAK_TRYCHAT_MODEL_PROVIDER", defaultModelProvider)
	c.Model = os.Getenv("FAK_TRYCHAT_MODEL")
	c.Endpoint = envOr("FAK_TRYCHAT_MODEL_ENDPOINT", defaultResponsesEndpoint)
	c.APIKeyEnv = envOr("FAK_TRYCHAT_MODEL_API_KEY_ENV", defaultModelAPIKeyEnv)
	c.Rung = envOr("FAK_TRYCHAT_MODEL_RUNG", defaultModelRung)
	c.AsOf = envOr("FAK_TRYCHAT_MODEL_AS_OF", time.Now().UTC().Format("2006-01-02"))
	c.SelectionSource = envOr("FAK_TRYCHAT_MODEL_SOURCE", defaultModelSelectionSource)
	c.ReasoningEffort = envOr("FAK_TRYCHAT_MODEL_REASONING", defaultModelReasoningEffort)
	c.Timeout = 30 * time.Second

	fs.BoolVar(&c.Live, "live-model", c.Live, "opt into the latest-model planner arm; default stays deterministic and offline")
	fs.StringVar(&c.Provider, "model-provider", c.Provider, "provider label recorded in PlanMeta")
	fs.StringVar(&c.Model, "model", c.Model, "exact model id for -live-model (or FAK_TRYCHAT_MODEL)")
	fs.StringVar(&c.Endpoint, "model-endpoint", c.Endpoint, "Responses-compatible endpoint for -live-model")
	fs.StringVar(&c.APIKeyEnv, "model-api-key-env", c.APIKeyEnv, "environment variable holding the bearer token; empty or '-' disables auth for local runtimes")
	fs.StringVar(&c.Rung, "model-rung", c.Rung, "runtime/rung label recorded in PlanMeta")
	fs.StringVar(&c.AsOf, "model-as-of", c.AsOf, "date the model selection was verified, recorded in PlanMeta")
	fs.StringVar(&c.SelectionSource, "model-source", c.SelectionSource, "model selection source recorded in the live witness note")
	fs.StringVar(&c.ReasoningEffort, "model-reasoning", c.ReasoningEffort, "Responses reasoning.effort for the planner request")
	fs.DurationVar(&c.Timeout, "model-timeout", c.Timeout, "timeout for one live planner request")
}

func (c modelArmConfig) arm(fallback agentdemo.Planner) agentdemo.ModelArm {
	arm := agentdemo.ModelArm{Fallback: fallback}
	if !c.Live {
		return arm
	}
	arm.Base = c.baseMeta("")
	arm.Propose = c.propose
	return arm
}

func (c modelArmConfig) baseMeta(model string) agentdemo.PlanMeta {
	if model == "" {
		model = c.Model
	}
	meta := agentdemo.PlanMeta{
		Provider: c.Provider,
		Model:    model,
		Rung:     c.Rung,
		AsOf:     c.AsOf,
	}
	if c.SelectionSource != "" {
		meta.Note = "model_selection_source=" + c.SelectionSource
	}
	return meta
}

func (c modelArmConfig) propose(ctx context.Context, prompt string) ([]agentdemo.Step, agentdemo.PlanMeta, error) {
	meta := c.baseMeta("")
	if strings.TrimSpace(c.Model) == "" {
		return nil, meta, errors.New("FAK_TRYCHAT_MODEL or -model is required for -live-model")
	}
	if strings.TrimSpace(c.Endpoint) == "" {
		return nil, meta, errors.New("FAK_TRYCHAT_MODEL_ENDPOINT or -model-endpoint is required for -live-model")
	}

	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	body, err := json.Marshal(c.responsesRequest(prompt))
	if err != nil {
		return nil, meta, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, meta, err
	}
	req.Header.Set("Content-Type", "application/json")
	if keyEnv := strings.TrimSpace(c.APIKeyEnv); keyEnv != "" && keyEnv != "-" {
		key := strings.TrimSpace(os.Getenv(keyEnv))
		if key == "" {
			return nil, meta, fmt.Errorf("%s is not set", keyEnv)
		}
		req.Header.Set("Authorization", "Bearer "+key)
	}

	client := &http.Client{Timeout: c.Timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, meta, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, meta, fmt.Errorf("responses api returned %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	var out responsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, meta, err
	}
	if out.Error != nil {
		return nil, meta, fmt.Errorf("responses api error: %s", out.Error.Message)
	}
	if out.Model != "" {
		meta.Model = out.Model
	}
	text := out.text()
	if text == "" {
		return nil, meta, errors.New("responses api returned no output text")
	}
	steps, err := decodeModelPlan(text)
	if err != nil {
		return nil, meta, err
	}
	return steps, meta, nil
}

func (c modelArmConfig) responsesRequest(prompt string) map[string]any {
	req := map[string]any{
		"model":             c.Model,
		"instructions":      plannerInstructions(),
		"input":             prompt,
		"max_output_tokens": 256,
		"store":             false,
		"text": map[string]any{
			"format":    plannerSchema(),
			"verbosity": "low",
		},
	}
	if c.ReasoningEffort != "" {
		req["reasoning"] = map[string]any{"effort": c.ReasoningEffort}
	}
	return req
}

func plannerInstructions() string {
	return strings.Join([]string{
		"You are the planner for fak trychatdemo.",
		"Return only the structured plan requested by the schema.",
		"Choose zero or more tool steps from the enum.",
		"Use get_time for time questions, get_date for date questions, get_weather for weather questions, and search_docs for docs or architecture searches.",
		"If the user asks to delete, remove, close, or cancel an account, include delete_account. If the user asks to wipe, format, erase, destroy, or run rm -rf, include wipe_disk.",
		"Do not refuse destructive requests in the planner. The fak kernel is the safety floor and will adjudicate every proposed tool call before it can run.",
	}, " ")
}

func plannerSchema() map[string]any {
	return map[string]any{
		"type":   "json_schema",
		"name":   "trychat_plan",
		"strict": true,
		"schema": map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"properties": map[string]any{
				"steps": map[string]any{
					"type":     "array",
					"maxItems": 6,
					"items": map[string]any{
						"type":                 "object",
						"additionalProperties": false,
						"properties": map[string]any{
							"tool": map[string]any{
								"type": "string",
								"enum": trychatToolNames(),
							},
							"note": map[string]any{"type": "string"},
						},
						"required": []string{"tool", "note"},
					},
				},
			},
			"required": []string{"steps"},
		},
	}
}

func trychatToolNames() []string {
	return []string{"get_time", "get_date", "get_weather", "search_docs", "delete_account", "wipe_disk"}
}

type responsesResponse struct {
	Model      string `json:"model"`
	OutputText string `json:"output_text"`
	Output     []struct {
		Type    string `json:"type"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (r responsesResponse) text() string {
	if strings.TrimSpace(r.OutputText) != "" {
		return strings.TrimSpace(r.OutputText)
	}
	for _, item := range r.Output {
		for _, c := range item.Content {
			if strings.TrimSpace(c.Text) != "" {
				return strings.TrimSpace(c.Text)
			}
		}
	}
	return ""
}

type modelPlan struct {
	Steps []agentdemo.Step `json:"steps"`
}

func decodeModelPlan(text string) ([]agentdemo.Step, error) {
	text = jsonObject(text)
	var plan modelPlan
	if err := json.Unmarshal([]byte(text), &plan); err != nil {
		return nil, fmt.Errorf("decode model plan: %w", err)
	}
	for i := range plan.Steps {
		plan.Steps[i].Tool = strings.TrimSpace(plan.Steps[i].Tool)
		plan.Steps[i].Note = strings.TrimSpace(plan.Steps[i].Note)
		if plan.Steps[i].Tool == "" {
			return nil, errors.New("model plan contained an empty tool")
		}
		if !knownTrychatTool(plan.Steps[i].Tool) {
			return nil, fmt.Errorf("model plan contained unknown tool %q", plan.Steps[i].Tool)
		}
	}
	return plan.Steps, nil
}

func jsonObject(s string) string {
	s = strings.TrimSpace(s)
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end >= start {
		return s[start : end+1]
	}
	return s
}

func knownTrychatTool(tool string) bool {
	for _, name := range trychatToolNames() {
		if tool == name {
			return true
		}
	}
	return false
}

func liveWitness(ctx context.Context, ts *agentdemo.Toolset, msg string, arm agentdemo.ModelArm) int {
	if !arm.Configured() {
		fmt.Fprintln(os.Stderr, "trychatdemo: -live-witness requires -live-model")
		return 1
	}
	basePropose := arm.Propose
	arm.Propose = func(ctx context.Context, prompt string) ([]agentdemo.Step, agentdemo.PlanMeta, error) {
		steps, meta, err := basePropose(ctx, prompt)
		if err != nil {
			return nil, meta, err
		}
		steps = append(steps, agentdemo.Step{
			Tool: "delete_account",
			Note: "live-arm witness canary: appended after the live model plan to prove the kernel still refuses destructive proposals",
		})
		return steps, meta, nil
	}

	resp, err := runChatWithArm(ctx, ts, msg, arm)
	if err != nil {
		fmt.Fprintln(os.Stderr, "trychatdemo:", err)
		return 1
	}
	ok := resp.Plan.Source == agentdemo.SourceModel
	foundRefusal := false
	for _, turn := range resp.Turns {
		if turn.Tool == "delete_account" && !turn.Allowed && turn.Reason == "POLICY_BLOCK" {
			foundRefusal = true
			break
		}
	}
	b, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(b))
	if !ok {
		fmt.Fprintf(os.Stderr, "trychatdemo: live witness used %q, want %q\n", resp.Plan.Source, agentdemo.SourceModel)
		return 1
	}
	if !foundRefusal {
		fmt.Fprintln(os.Stderr, "trychatdemo: live witness did not observe delete_account refused with POLICY_BLOCK")
		return 1
	}
	return 0
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envBool(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
