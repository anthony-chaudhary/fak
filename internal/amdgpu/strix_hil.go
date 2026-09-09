package amdgpu

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	StrixHILReceiptSchema = "fak.strix.hil-receipt/v1"
	DefaultHILPort        = 8080
	DefaultHILModel       = "Qwen3.8-27B-Q4_K_M"
)

// StrixHILInferenceWitness records a verified live inference round-trip on the Strix appliance.
type StrixHILInferenceWitness struct {
	Endpoint         string  `json:"endpoint"`
	Model            string  `json:"model"`
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	LatencyMS        float64 `json:"latency_ms"`
	TokensPerSec     float64 `json:"tokens_per_sec"`
	Verified         bool    `json:"verified"`
	Error            string  `json:"error,omitempty"`
}

// StrixHILOpts configures a hardware-in-the-loop validation run.
type StrixHILOpts struct {
	Host                 string
	RunValidation        bool
	RunSubkernels        bool
	Subkernels           []string
	RunAblations         bool
	Ablations            []string
	RunInference         bool
	InferenceModel       string
	InferencePrompt      string
	GatewayKey           string
	GitRef               string
	GitTip               string
	Command              string
	Timeout              time.Duration
	AdmissionTimeout     time.Duration
	RequireSourceBinding bool
	CandidateArchive     []byte
	SourceArchiveSHA256  string
}

// StrixHILReceipt models the unified hardware-in-the-loop receipt.
type StrixHILReceipt struct {
	Schema            string                    `json:"schema"`
	Timestamp         string                    `json:"timestamp"`
	Verdict           string                    `json:"verdict"` // "PASS" | "FAIL"
	Verified          bool                      `json:"verified"`
	Digest            string                    `json:"digest"`
	Target            StrixTarget               `json:"target"`
	ValidationReceipt *StrixValidationReceipt   `json:"validation_receipt,omitempty"`
	InferenceWitness  *StrixHILInferenceWitness `json:"inference_witness,omitempty"`
	SubkernelCount    int                       `json:"subkernel_count"`
	AblationCount     int                       `json:"ablation_count"`
	Failures          []string                  `json:"failures,omitempty"`
}

// ComputeDigest computes a canonical SHA-256 digest of the HIL receipt.
func (r *StrixHILReceipt) ComputeDigest() (string, error) {
	copyReceipt := *r
	copyReceipt.Digest = ""
	data, err := json.Marshal(copyReceipt)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(h[:]), nil
}

// Validate checks internal consistency and invariants of the HIL receipt.
func (r *StrixHILReceipt) Validate() error {
	if r.Schema != StrixHILReceiptSchema {
		return fmt.Errorf("invalid hil schema %q (want %q)", r.Schema, StrixHILReceiptSchema)
	}
	if r.Verdict != "PASS" && r.Verdict != "FAIL" {
		return fmt.Errorf("invalid hil verdict %q", r.Verdict)
	}
	if r.Digest == "" {
		return fmt.Errorf("hil receipt digest is required")
	}
	expected, err := r.ComputeDigest()
	if err != nil {
		return fmt.Errorf("cannot compute hil digest: %w", err)
	}
	if r.Digest != expected {
		return fmt.Errorf("hil digest mismatch: recorded %s != computed %s", r.Digest, expected)
	}
	if r.Verdict == "PASS" {
		if !r.Target.Reachable {
			return fmt.Errorf("hil passed but target is unreachable")
		}
		if len(r.Failures) > 0 {
			return fmt.Errorf("hil passed but contains %d failures", len(r.Failures))
		}
		if r.ValidationReceipt != nil {
			if err := r.ValidationReceipt.Validate(); err != nil {
				return fmt.Errorf("nested validation receipt invalid: %w", err)
			}
			if r.ValidationReceipt.Verdict != "PASS" {
				return fmt.Errorf("nested validation receipt verdict = %s", r.ValidationReceipt.Verdict)
			}
		}
		if r.InferenceWitness != nil && !r.InferenceWitness.Verified {
			return fmt.Errorf("inference witness is present but not verified")
		}
	}
	return nil
}

var (
	strixHTTPClient = &http.Client{Timeout: 15 * time.Second}
)

// resolveGatewayKey retrieves the gateway authentication bearer key.
func resolveGatewayKey(ctx context.Context, target *StrixTarget, explicitKey string) string {
	if explicitKey != "" {
		return explicitKey
	}
	if k := os.Getenv("FAK_GATEWAY_KEY"); k != "" {
		return k
	}
	// Try reading /etc/fak/gateway.env over SSH
	if target != nil && target.Reachable {
		out, err := runStrixTargetCommand(ctx, target, "cat /etc/fak/gateway.env 2>/dev/null", nil)
		if err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "FAK_GATEWAY_KEY=") {
					return strings.TrimSpace(strings.TrimPrefix(line, "FAK_GATEWAY_KEY="))
				}
			}
		}
	}
	return ""
}

// ExecuteStrixInferenceCheck queries the live model running on the Strix appliance.
func ExecuteStrixInferenceCheck(ctx context.Context, target *StrixTarget, model, prompt, key string) *StrixHILInferenceWitness {
	if model == "" {
		model = DefaultHILModel
	}
	if prompt == "" {
		prompt = "What is 2 + 2? Reply with just the number."
	}

	hostAddr := target.Host
	if hostAddr == "localhost" || hostAddr == "127.0.0.1" || target.Mode == "local" {
		hostAddr = "127.0.0.1"
	} else if hostAddr == "strix1" || hostAddr == "strix-agent" || hostAddr == "strix-halo-fak" {
		hostAddr = "192.168.1.208"
	}

	endpoint := fmt.Sprintf("http://%s:%d/v1/chat/completions", hostAddr, DefaultHILPort)
	if strings.Contains(hostAddr, ":") {
		endpoint = fmt.Sprintf("http://%s/v1/chat/completions", hostAddr)
	}
	if key != "" {
		endpoint += "?key=" + key
	}

	witness := &StrixHILInferenceWitness{
		Endpoint: endpoint,
		Model:    model,
		Prompt:   prompt,
	}

	reqBody := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": 16,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		witness.Error = fmt.Sprintf("marshal request body: %v", err)
		return witness
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		witness.Error = fmt.Sprintf("create http request: %v", err)
		return witness
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	start := time.Now()
	resp, err := strixHTTPClient.Do(req)
	witness.LatencyMS = float64(time.Since(start).Nanoseconds()) / 1e6
	if err != nil {
		witness.Error = fmt.Sprintf("http post: %v", err)
		return witness
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		witness.Error = fmt.Sprintf("read response body: %v", err)
		return witness
	}

	if resp.StatusCode != http.StatusOK {
		witness.Error = fmt.Sprintf("http status %d: %s", resp.StatusCode, string(respBytes))
		return witness
	}

	var chatResp struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(respBytes, &chatResp); err != nil {
		witness.Error = fmt.Sprintf("unmarshal chat response: %v", err)
		return witness
	}

	if len(chatResp.Choices) == 0 {
		witness.Error = "response contained zero choices"
		return witness
	}

	witness.Response = strings.TrimSpace(chatResp.Choices[0].Message.Content)
	witness.PromptTokens = chatResp.Usage.PromptTokens
	witness.CompletionTokens = chatResp.Usage.CompletionTokens
	witness.TotalTokens = chatResp.Usage.TotalTokens
	if witness.LatencyMS > 0 && witness.CompletionTokens > 0 {
		witness.TokensPerSec = float64(witness.CompletionTokens) / (witness.LatencyMS / 1000.0)
	}
	witness.Verified = witness.Response != "" && len(witness.Response) > 0
	return witness
}

// RunStrixHIL executes the complete Hardware-In-The-Loop evaluation suite on the Strix appliance.
func RunStrixHIL(ctx context.Context, opts StrixHILOpts) (*StrixHILReceipt, error) {
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}

	target, err := DiscoverStrixTarget(ctx, opts.Host)
	if err != nil || target == nil || !target.Reachable {
		host := opts.Host
		if target != nil && target.Host != "" {
			host = target.Host
		}
		errMsg := "target unreachable"
		if err != nil {
			errMsg = err.Error()
		}
		r := &StrixHILReceipt{
			Schema:    StrixHILReceiptSchema,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Verdict:   "FAIL",
			Verified:  false,
			Target:    StrixTarget{Host: host, Reachable: false, Mode: "ssh", TargetISA: "gfx1151", ComputeUnits: 40},
			Failures:  []string{"strix halo target unreachable: " + errMsg},
		}
		r.Digest, _ = r.ComputeDigest()
		return r, fmt.Errorf("strix halo target unreachable: %w", err)
	}

	receipt := &StrixHILReceipt{
		Schema:    StrixHILReceiptSchema,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Verdict:   "PASS",
		Verified:  true,
		Target:    *target,
		Failures:  make([]string, 0),
	}

	// 1. Live inference check if requested
	if opts.RunInference {
		key := resolveGatewayKey(ctx, target, opts.GatewayKey)
		witness := ExecuteStrixInferenceCheck(ctx, target, opts.InferenceModel, opts.InferencePrompt, key)
		receipt.InferenceWitness = witness
		if !witness.Verified {
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("live inference witness failed: %s", witness.Error))
		}
	}

	// 2. Subkernels & ablations validation if requested
	if opts.RunValidation {
		subkernelList := opts.Subkernels
		if opts.RunSubkernels && len(subkernelList) == 0 {
			subkernelList = append([]string(nil), DefaultCreditableSubkernelSelectors...)
		}
		ablationList := opts.Ablations
		if opts.RunAblations && len(ablationList) == 0 {
			ablationList = []string{"cpu_vs_vulkan_gpu", "fused_vs_discrete_norm_matmul"}
		}

		vOpts := StrixValidationOpts{
			Host:                 target.Host,
			RunSubkernels:        opts.RunSubkernels,
			Subkernels:           subkernelList,
			RunAblations:         opts.RunAblations,
			Ablations:            ablationList,
			GitRef:               opts.GitRef,
			GitTip:               opts.GitTip,
			Command:              opts.Command,
			Timeout:              opts.Timeout,
			AdmissionTimeout:     opts.AdmissionTimeout,
			RequireSourceBinding: opts.RequireSourceBinding,
			CandidateArchive:     opts.CandidateArchive,
			SourceArchiveSHA256:  opts.SourceArchiveSHA256,
		}

		valReceipt, valErr := RunStrixValidation(ctx, vOpts)
		receipt.ValidationReceipt = valReceipt
		if valReceipt != nil {
			receipt.SubkernelCount = len(valReceipt.Subkernels)
			receipt.AblationCount = len(valReceipt.Ablations)
			if valReceipt.Verdict != "PASS" {
				receipt.Failures = append(receipt.Failures, valReceipt.Failures...)
			}
		}
		if valErr != nil && len(receipt.Failures) == 0 {
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("validation error: %v", valErr))
		}
	}

	if len(receipt.Failures) > 0 {
		receipt.Verdict = "FAIL"
		receipt.Verified = false
	}

	digest, err := receipt.ComputeDigest()
	if err == nil {
		receipt.Digest = digest
	}

	if !receipt.Verified {
		return receipt, fmt.Errorf("strix hil verification failed: %s", strings.Join(receipt.Failures, "; "))
	}
	return receipt, nil
}
