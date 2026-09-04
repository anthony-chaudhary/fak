package qwen38quantrun

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestAMDDeterministicTokensAndLogitsPassWithinTolerance(t *testing.T) {
	in := validAMDScoreboardInput()
	// Slightly perturb reference logits within tolerance (1e-3)
	for i := range in.Reference.Trials {
		in.Reference.Trials[i].Logits = slices.Clone(in.Reference.Trials[i].Logits)
		in.Reference.Trials[i].Logits[0] += 0.0001
	}

	report := BuildAMDScoreboard(in)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		t.Fatalf("ValidateAMDScoreboardReport failed: %v", err)
	}
	if !report.Comparable || report.Verdict != "comparable" {
		t.Fatalf("expected comparable verdict, got %+v", report)
	}
	if report.ReferenceOverCandidate == nil {
		t.Fatal("expected non-nil ratios for comparable report")
	}

	// Direct trial validation helper should also pass
	for i := range in.Candidate.Trials {
		c, r := in.Candidate.Trials[i], in.Reference.Trials[i]
		if err := ValidateTrialTokensAndLogits(c, r, in.LogitTolerance); err != nil {
			t.Fatalf("trial %d tokens and logits should match within tolerance: %v", i+1, err)
		}
	}
}

func TestAMDScoreboardOutputTokenMismatchYieldsDiverged(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Reference.Trials = make([]AMDScoreboardTrial, len(in.Candidate.Trials))
	for i := range in.Candidate.Trials {
		in.Reference.Trials[i] = in.Candidate.Trials[i]
		in.Reference.Trials[i].OutputTokenIDs = slices.Clone(in.Candidate.Trials[i].OutputTokenIDs)
		in.Reference.Trials[i].Logits = slices.Clone(in.Candidate.Trials[i].Logits)
	}
	// Candidate produced [4, 5, 6, 7], change reference trial 0 to [4, 5, 6, 999]
	in.Reference.Trials[0].OutputTokenIDs[3] = 999

	report := BuildAMDScoreboard(in)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		t.Fatalf("report must be structurally valid: %v", err)
	}
	if report.Comparable {
		t.Fatal("expected report to not be comparable")
	}
	if report.Verdict != "diverged" && report.Verdict != "not-comparable" {
		t.Fatalf("expected verdict 'diverged' or 'not-comparable', got %q", report.Verdict)
	}
	if !slices.Contains(report.Reasons, "output-token-mismatch") {
		t.Fatalf("expected 'output-token-mismatch' in reasons: %v", report.Reasons)
	}
	if err := ValidateAMDScoreboardDivergence(report); err != nil {
		t.Fatalf("ValidateAMDScoreboardDivergence should pass: %v", err)
	}

	// ValidateTokenEquivalence helper verification
	err := ValidateTokenEquivalence(in.Candidate.Trials[0].OutputTokenIDs, in.Reference.Trials[0].OutputTokenIDs)
	if err == nil || !strings.Contains(err.Error(), "token ID mismatch") {
		t.Fatalf("expected token ID mismatch error, got: %v", err)
	}
}

func TestAMDScoreboardLogitToleranceExceededYieldsDiverged(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Reference.Trials = make([]AMDScoreboardTrial, len(in.Candidate.Trials))
	for i := range in.Candidate.Trials {
		in.Reference.Trials[i] = in.Candidate.Trials[i]
		in.Reference.Trials[i].OutputTokenIDs = slices.Clone(in.Candidate.Trials[i].OutputTokenIDs)
		in.Reference.Trials[i].Logits = slices.Clone(in.Candidate.Trials[i].Logits)
	}
	// Tolerance is 1e-3, add 0.05 to exceed tolerance
	in.Reference.Trials[0].Logits[0] += 0.05

	report := BuildAMDScoreboard(in)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		t.Fatalf("report must be structurally valid: %v", err)
	}
	if report.Comparable {
		t.Fatal("expected report to not be comparable")
	}
	if report.Verdict != "diverged" && report.Verdict != "not-comparable" {
		t.Fatalf("expected verdict 'diverged' or 'not-comparable', got %q", report.Verdict)
	}
	if !slices.Contains(report.Reasons, "logit-tolerance-exceeded") {
		t.Fatalf("expected 'logit-tolerance-exceeded' in reasons: %v", report.Reasons)
	}
	if err := ValidateAMDScoreboardDivergence(report); err != nil {
		t.Fatalf("ValidateAMDScoreboardDivergence should pass: %v", err)
	}

	// ValidateLogitsTolerance helper verification
	err := ValidateLogitsTolerance(in.Candidate.Trials[0].Logits, in.Reference.Trials[0].Logits, in.LogitTolerance)
	if err == nil || !strings.Contains(err.Error(), "logit tolerance exceeded") {
		t.Fatalf("expected logit tolerance exceeded error, got: %v", err)
	}
}

func TestAMDScoreboardLogitShapeMismatchYieldsDiverged(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Reference.Trials = make([]AMDScoreboardTrial, len(in.Candidate.Trials))
	for i := range in.Candidate.Trials {
		in.Reference.Trials[i] = in.Candidate.Trials[i]
		in.Reference.Trials[i].OutputTokenIDs = slices.Clone(in.Candidate.Trials[i].OutputTokenIDs)
		in.Reference.Trials[i].Logits = slices.Clone(in.Candidate.Trials[i].Logits)
	}
	in.Reference.Trials[0].Logits = append(in.Reference.Trials[0].Logits, 3.14)

	report := BuildAMDScoreboard(in)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		t.Fatalf("report must be structurally valid: %v", err)
	}
	if report.Comparable {
		t.Fatal("expected report to not be comparable")
	}
	if report.Verdict != "diverged" && report.Verdict != "not-comparable" {
		t.Fatalf("expected verdict 'diverged' or 'not-comparable', got %q", report.Verdict)
	}
	if !slices.Contains(report.Reasons, "logit-shape-mismatch") {
		t.Fatalf("expected 'logit-shape-mismatch' in reasons: %v", report.Reasons)
	}
	if err := ValidateAMDScoreboardDivergence(report); err != nil {
		t.Fatalf("ValidateAMDScoreboardDivergence should pass: %v", err)
	}

	err := ValidateLogitsTolerance(in.Candidate.Trials[0].Logits, in.Reference.Trials[0].Logits, in.LogitTolerance)
	if err == nil || !strings.Contains(err.Error(), "logit shape mismatch") {
		t.Fatalf("expected logit shape mismatch error, got: %v", err)
	}
}

func TestAMDScoreboardRejectsArtifactOrTokenizerOrPromptMismatch(t *testing.T) {
	t.Run("artifact mismatch", func(t *testing.T) {
		in := validAMDScoreboardInput()
		in.Reference.ArtifactSHA256 = "1111111111111111111111111111111111111111111111111111111111111111"
		report := BuildAMDScoreboard(in)
		if report.Comparable || report.Verdict != "not-comparable" || !slices.Contains(report.Reasons, "artifact-mismatch") {
			t.Fatalf("expected not-comparable on artifact mismatch: %+v", report)
		}
	})

	t.Run("tokenizer digest mismatch", func(t *testing.T) {
		in := validAMDScoreboardInput()
		in.Candidate.TokenizerDigest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		in.Reference.TokenizerDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
		report := BuildAMDScoreboard(in)
		if report.Comparable || report.Verdict != "not-comparable" || !slices.Contains(report.Reasons, "tokenizer-digest-mismatch") {
			t.Fatalf("expected not-comparable on tokenizer digest mismatch: %+v", report)
		}
	})

	t.Run("prompt token IDs mismatch", func(t *testing.T) {
		in := validAMDScoreboardInput()
		in.Reference.PromptTokenIDs = []int{99, 100, 101}
		report := BuildAMDScoreboard(in)
		if report.Comparable || report.Verdict != "not-comparable" || !slices.Contains(report.Reasons, "prompt-or-tokenization-mismatch") {
			t.Fatalf("expected not-comparable on prompt token mismatch: %+v", report)
		}
	})

	t.Run("prompt packet attestation mismatch", func(t *testing.T) {
		in := validAMDScoreboardInput()
		p1, err := FreezePromptPacket(validTestPromptPacket())
		if err != nil {
			t.Fatal(err)
		}
		p2 := validTestPromptPacket()
		p2.PromptTokenIDs = slices.Clone(p2.PromptTokenIDs)
		p2.PromptTokenIDs[0]++
		p2Frozen, err := FreezePromptPacket(p2)
		if err != nil {
			t.Fatal(err)
		}
		in.Candidate.PromptPacket = &p1
		in.Reference.PromptPacket = &p2Frozen
		report := BuildAMDScoreboard(in)
		if report.Comparable || report.Verdict != "not-comparable" || !slices.Contains(report.Reasons, "prompt-packet-mismatch") {
			t.Fatalf("expected not-comparable on prompt packet mismatch: %+v", report)
		}
	})
}

func TestAMDScoreboardSelectedTokenLogitsOptIn(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Candidate.DeterministicTokens = true
	in.Candidate.SelectedTokenLogits = true
	in.Reference.DeterministicTokens = true
	in.Reference.SelectedTokenLogits = true

	// Move tokens and logits to SelectedTokenIDs / SelectedTokenLogits
	for i := range in.Candidate.Trials {
		in.Candidate.Trials[i].SelectedTokenIDs = in.Candidate.Trials[i].OutputTokenIDs
		in.Candidate.Trials[i].SelectedTokenLogits = in.Candidate.Trials[i].Logits
		in.Candidate.Trials[i].OutputTokenIDs = nil
		in.Candidate.Trials[i].Logits = nil

		in.Reference.Trials[i].SelectedTokenIDs = in.Reference.Trials[i].OutputTokenIDs
		in.Reference.Trials[i].SelectedTokenLogits = in.Reference.Trials[i].Logits
		in.Reference.Trials[i].OutputTokenIDs = nil
		in.Reference.Trials[i].Logits = nil
	}

	report := BuildAMDScoreboard(in)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		t.Fatalf("ValidateAMDScoreboardReport failed: %v", err)
	}
	if !report.Comparable || report.Verdict != "comparable" {
		t.Fatalf("expected comparable verdict with selected tokens/logits, got: %+v", report)
	}
}

func TestAMDCaptureTrialFromNativeInferenceReceipt(t *testing.T) {
	receipt := &model.NativeInferenceReceipt{
		TokenIDs:       []int{4, 5, 6, 7},
		TokenLogprobs:  []float64{-0.12, -0.34, -0.56, -0.78},
		PrefillSeconds: 0.25,
		DecodeSeconds:  1.50,
		Model:          "Qwen3.8-27B",
		Engine:         "fak-native",
		Backend:        "vulkan",
	}

	trial, err := CaptureAMDScoreboardTrial(1, receipt, 5.0, 100.0, 25.0, 1024, 2048, 512, 10)
	if err != nil {
		t.Fatalf("CaptureAMDScoreboardTrial failed: %v", err)
	}

	if trial.Repetition != 1 {
		t.Fatalf("repetition = %d, want 1", trial.Repetition)
	}
	if !slices.Equal(trial.OutputTokenIDs, receipt.TokenIDs) {
		t.Fatalf("output token IDs = %v, want %v", trial.OutputTokenIDs, receipt.TokenIDs)
	}
	if !slices.Equal(trial.Logits, receipt.TokenLogprobs) {
		t.Fatalf("logits = %v, want %v", trial.Logits, receipt.TokenLogprobs)
	}
	if !slices.Equal(trial.SelectedTokenIDs, receipt.TokenIDs) {
		t.Fatalf("selected token IDs = %v, want %v", trial.SelectedTokenIDs, receipt.TokenIDs)
	}
	if !slices.Equal(trial.SelectedTokenLogits, receipt.TokenLogprobs) {
		t.Fatalf("selected token logits = %v, want %v", trial.SelectedTokenLogits, receipt.TokenLogprobs)
	}
	if trial.PrefillSeconds != 0.25 || trial.WarmDecodeSeconds != 1.50 {
		t.Fatalf("prefill=%f decode=%f mismatch", trial.PrefillSeconds, trial.WarmDecodeSeconds)
	}

	// Nil receipt fails
	if _, err := CaptureAMDScoreboardTrial(1, nil, 0, 0, 0, 0, 0, 0, 0); err == nil {
		t.Fatal("expected error on nil receipt")
	}
}

func TestAMDDefaultOpenAIFormattingUntouched(t *testing.T) {
	var capturedRequest map[string]any

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "exact"}}})
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&capturedRequest); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

		// Standard OpenAI response format
		response := map[string]any{
			"id":      "chatcmpl-test-123",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "exact",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "hello world",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     10,
				"completion_tokens": 2,
				"total_tokens":      12,
			},
		}

		// Only if opt-in native inference receipt was requested, attach fak receipt
		fakReq, _ := capturedRequest["fak"].(map[string]any)
		if fakReq != nil && fakReq["native_inference_receipt"] == true {
			response["fak"] = map[string]any{
				"native_inference_receipt": map[string]any{
					"token_ids":       []int{15339, 1917},
					"token_logprobs":  []float64{-0.005, -0.012},
					"prefill_seconds": 0.05,
					"decode_seconds":  0.01,
					"model":           "exact",
					"engine":          "fak-native",
					"backend":         "vulkan",
				},
			}
		}

		_ = json.NewEncoder(w).Encode(response)
	}))
	defer s.Close()

	fixture := qwen38quant.Fixture{
		ID:              "fix-text",
		Workload:        "text",
		Prompt:          "Say hello world",
		ExpectedExact:   "hello world",
		MaxOutputTokens: 16,
	}

	// Case 1: Default request (NativeInferenceReceipt = false)
	// Must NOT send fak.native_inference_receipt and default OpenAI response is parsed untouched.
	cfgDefault := Config{
		Endpoint:               s.URL,
		APIKey:                 "test-key",
		Model:                  "exact",
		NativeInferenceReceipt: false,
	}

	respDefault, err := runOne(context.Background(), s.Client(), cfgDefault, fixture)
	if err != nil {
		t.Fatalf("runOne default failed: %v", err)
	}
	if capturedRequest["fak"] != nil {
		t.Fatalf("default request must not send fak extension: %+v", capturedRequest["fak"])
	}
	if respDefault.Choices[0].Message.Content != "hello world" {
		t.Fatalf("choices content = %q, want 'hello world'", respDefault.Choices[0].Message.Content)
	}
	if respDefault.Usage["completion_tokens"] != 2 {
		t.Fatalf("usage completion tokens = %d, want 2", respDefault.Usage["completion_tokens"])
	}
	if respDefault.Fak != nil && respDefault.Fak.NativeInferenceReceipt != nil {
		t.Fatal("expected nil Fak.NativeInferenceReceipt for default request")
	}

	// Case 2: Opt-in request (NativeInferenceReceipt = true)
	// Must send fak.native_inference_receipt and capture exact tokens and logits debug trace.
	cfgOptIn := Config{
		Endpoint:               s.URL,
		APIKey:                 "test-key",
		Model:                  "exact",
		NativeInferenceReceipt: true,
	}

	respOptIn, err := runOne(context.Background(), s.Client(), cfgOptIn, fixture)
	if err != nil {
		t.Fatalf("runOne opt-in failed: %v", err)
	}
	fakReq, ok := capturedRequest["fak"].(map[string]any)
	if !ok || fakReq["native_inference_receipt"] != true {
		t.Fatalf("opt-in request must send native_inference_receipt: true, got: %#v", capturedRequest["fak"])
	}
	// OpenAI choices message content is still standard and intact
	if respOptIn.Choices[0].Message.Content != "hello world" {
		t.Fatalf("choices content = %q, want 'hello world'", respOptIn.Choices[0].Message.Content)
	}
	// Opt-in debug trace is captured
	if respOptIn.Fak == nil || respOptIn.Fak.NativeInferenceReceipt == nil {
		t.Fatal("expected non-nil Fak.NativeInferenceReceipt for opt-in request")
	}
	receipt := respOptIn.Fak.NativeInferenceReceipt
	if !slices.Equal(receipt.TokenIDs, []int{15339, 1917}) {
		t.Fatalf("token IDs = %v, want [15339, 1917]", receipt.TokenIDs)
	}
	if len(receipt.TokenLogprobs) != 2 || receipt.TokenLogprobs[0] != -0.005 || receipt.TokenLogprobs[1] != -0.012 {
		t.Fatalf("token logprobs = %v, want [-0.005, -0.012]", receipt.TokenLogprobs)
	}

	// Case 3: runFixtures populates Result.NativeInferenceReceipt on opt-in
	results, err := (Runner{}).runFixtures(context.Background(), s.Client(), cfgOptIn, []qwen38quant.Fixture{fixture}, 1)
	if err != nil {
		t.Fatalf("runFixtures failed: %v", err)
	}
	if len(results) != 1 || results[0].Quality != "PASS" {
		t.Fatalf("expected PASS: %+v", results)
	}
	if results[0].NativeInferenceReceipt == nil {
		t.Fatal("expected non-nil Result.NativeInferenceReceipt")
	}
	if !slices.Equal(results[0].NativeInferenceReceipt.TokenIDs, []int{15339, 1917}) {
		t.Fatalf("result token IDs = %v", results[0].NativeInferenceReceipt.TokenIDs)
	}

	// Case 4: runFixtures leaves Result.NativeInferenceReceipt nil on default
	resultsDef, err := (Runner{}).runFixtures(context.Background(), s.Client(), cfgDefault, []qwen38quant.Fixture{fixture}, 1)
	if err != nil {
		t.Fatalf("runFixtures default failed: %v", err)
	}
	if len(resultsDef) != 1 || resultsDef[0].Quality != "PASS" {
		t.Fatalf("expected PASS: %+v", resultsDef)
	}
	if resultsDef[0].NativeInferenceReceipt != nil {
		t.Fatal("expected nil Result.NativeInferenceReceipt for default config")
	}
}
