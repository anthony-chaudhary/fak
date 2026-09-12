//go:build vulkan && (windows || linux) && cgo

package model_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const q4KTiledF32FullModelWitnessEnv = "FAK_QWEN35_Q4K_TILED_FP32_FULL_MODEL_WITNESS"

type q4KTiledF32ModelArm struct {
	standaloneSequenceArm
	PrefillDispatches       uint64 `json:"q4_tiled_fp32_prefill_dispatches"`
	ContinuationDispatches  uint64 `json:"q4_tiled_fp32_continuation_dispatches"`
	PrefillH2DBytes         uint64 `json:"prefill_h2d_bytes"`
	PrefillD2HBytes         uint64 `json:"prefill_d2h_bytes_including_final_logits"`
	ContinuationH2DBytes    uint64 `json:"continuation_h2d_bytes"`
	ContinuationD2HBytes    uint64 `json:"continuation_d2h_bytes_including_logits"`
	ExactLogitBitMismatches int    `json:"exact_logit_bit_mismatches"`
	ExactTokenIDMismatches  int    `json:"exact_token_id_mismatches"`
}

type q4KTiledF32TransferWitness interface {
	VulkanDebugTransferBytes() (uint64, uint64)
}

type q4KTiledF32ExactComparison struct {
	LogitBitMismatches int `json:"logit_bit_mismatches"`
	TokenIDMismatches  int `json:"token_id_mismatches"`
}

func compareQ4KTiledF32Exact(got, want standaloneSequenceArm) (q4KTiledF32ExactComparison, []string) {
	var comparison q4KTiledF32ExactComparison
	var failures []string
	if len(got.logits) != len(want.logits) {
		failures = append(failures, fmt.Sprintf("logit step count %d != %d", len(got.logits), len(want.logits)))
	} else {
		for step := range want.logits {
			if len(got.logits[step]) != len(want.logits[step]) {
				failures = append(failures, fmt.Sprintf("step %d logit count %d != %d", step, len(got.logits[step]), len(want.logits[step])))
				continue
			}
			for index, value := range got.logits[step] {
				if math.Float32bits(value) != math.Float32bits(want.logits[step][index]) {
					comparison.LogitBitMismatches++
				}
			}
		}
	}
	if len(got.GeneratedTokenIDs) != len(want.GeneratedTokenIDs) {
		failures = append(failures, fmt.Sprintf("generated token count %d != %d", len(got.GeneratedTokenIDs), len(want.GeneratedTokenIDs)))
	} else {
		for index, id := range got.GeneratedTokenIDs {
			if id != want.GeneratedTokenIDs[index] {
				comparison.TokenIDMismatches++
			}
		}
	}
	if comparison.LogitBitMismatches != 0 {
		failures = append(failures, fmt.Sprintf("exact logit bit mismatches=%d", comparison.LogitBitMismatches))
	}
	if comparison.TokenIDMismatches != 0 {
		failures = append(failures, fmt.Sprintf("exact token ID mismatches=%d", comparison.TokenIDMismatches))
	}
	return comparison, failures
}

// TestVulkanQ4KTiledF32PrefillFullModel is deliberately separate from the default
// sequence witness. The latter pins the production scalar route; this test is
// the opt-in promotion gate for the tiled FP32 Q4_K prefill candidate.
func TestVulkanQ4KTiledF32PrefillFullModel(t *testing.T) {
	// Isolate Q4_K tiling from the independently qualified GDN panel candidate.
	t.Setenv("FAK_VULKAN_GDN_Q8_PANEL", "0")
	if os.Getenv(q4KTiledF32FullModelWitnessEnv) != "1" {
		t.Skip("set " + q4KTiledF32FullModelWitnessEnv + "=1 for the tiled FP32 Q4_K full-model witness")
	}
	path := strings.TrimSpace(os.Getenv("FAK_QWEN38_GGUF"))
	if path == "" || os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Fatal("an actual checkpoint and FAK_VULKAN_REQUIRE_DEVICE=1 are required")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil {
		t.Fatal("required Vulkan backend unavailable")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(be.Tier()), strings.ToLower(expected)) {
		t.Fatalf("unexpected device %q, want %q", be.Tier(), expected)
	}
	if _, ok := be.(standaloneQ4DispatchWitness); !ok {
		t.Fatal("backend lacks tiled FP32 Q4_K dispatch attribution")
	}
	if _, ok := be.(q4KTiledF32TransferWitness); !ok {
		t.Fatal("backend lacks Vulkan transfer-byte attribution")
	}
	digest := standaloneHashArtifact(t, path)
	if expected := os.Getenv("FAK_QWEN38_GGUF_SHA256"); expected != "" && !strings.EqualFold(digest, expected) {
		t.Fatal("checkpoint SHA-256 mismatch")
	}
	provenance := map[string]any{
		"gdn_q8_panel":          os.Getenv("FAK_VULKAN_GDN_Q8_PANEL"),
		"artifact_sha256": digest, "device": be.Tier(), "engine": "fak-native",
		"source_archive_sha256": os.Getenv("FAK_SOURCE_ARCHIVE_SHA256"),
		"binary_sha256":         os.Getenv("FAK_BINARY_SHA256"),
		"shader_bundle_sha256":  os.Getenv("FAK_SHADER_BUNDLE_SHA256"),
		"quality_gate":          map[string]float64{"cosine_min": .99999, "relative_l2_max": 1e-4},
	}
	logRecord := func(phase string, passed bool, fields map[string]any) {
		t.Helper()
		fields["schema"], fields["phase"], fields["passed"] = "fak.vulkan-q4k-tiled-fp32-full-model/v1", phase, passed
		fields["observed_utc"], fields["provenance"] = time.Now().UTC().Format(time.RFC3339Nano), provenance
		data, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		t.Log(string(data))
	}
	var invalidProvenance []string
	for _, key := range []string{"source_archive_sha256", "binary_sha256", "shader_bundle_sha256"} {
		value, _ := provenance[key].(string)
		decoded, err := hex.DecodeString(value)
		if err != nil || len(decoded) != sha256.Size {
			invalidProvenance = append(invalidProvenance, key)
		}
	}
	if len(invalidProvenance) != 0 {
		logRecord("provenance_failure", false, map[string]any{"invalid_or_missing_sha256": invalidProvenance})
		t.Fatal("missing or invalid source/binary/shader provenance")
	}
	m, err := ggufload.LoadModelQ4KProfileOptions(path, nil,
		ggufload.WithDenseKQuantResident(false), ggufload.WithDenseQ2KResident(true))
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	if standaloneHashArtifact(t, path) != digest {
		t.Fatal("checkpoint changed during load")
	}
	if !m.Cfg.IsQwen35Hybrid() || m.Cfg.NumLayers != 64 || m.Cfg.HiddenSize != 5120 ||
		m.Cfg.IntermediateSize != 17408 || m.Cfg.NumHeads != 24 || m.Cfg.NumKVHeads != 4 || m.Cfg.HeadDim != 256 {
		t.Fatal("witness requires the full dense Qwen3.8-27B geometry")
	}
	prompt := func(n, offset, stride int) []int {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (offset + i*stride) % m.Cfg.VocabSize
		}
		return ids
	}
	runPair := func(ids []int) (q4KTiledF32ModelArm, q4KTiledF32ModelArm) {
		t.Helper()
		scalar, err := runQ4KTiledF32ModelArm(m, be, "scalar", ids, 4)
		if err != nil {
			logRecord("execution_failure", false, map[string]any{"arm": "scalar", "prompt_tokens": len(ids), "result": scalar, "error": err.Error()})
			t.Fatal(err)
		}
		candidate, err := runQ4KTiledF32ModelArm(m, be, "candidate", ids, 4)
		if err != nil {
			logRecord("execution_failure", false, map[string]any{"arm": "candidate", "prompt_tokens": len(ids), "result": candidate, "error": err.Error()})
			t.Fatal(err)
		}
		if scalar.PrefillDispatches != 0 || scalar.ContinuationDispatches != 0 {
			err := fmt.Errorf("scalar arm used tiled FP32 Q4_K: prefill=%d continuation=%d", scalar.PrefillDispatches, scalar.ContinuationDispatches)
			logRecord("counter_failure", false, map[string]any{"error": err.Error(), "scalar": scalar})
			t.Fatal(err)
		}
		if candidate.PrefillDispatches == 0 {
			err := fmt.Errorf("candidate arm produced no tiled FP32 Q4_K prefill dispatch")
			logRecord("counter_failure", false, map[string]any{"error": err.Error(), "candidate": candidate})
			t.Fatal(err)
		}
		if candidate.ContinuationDispatches != 0 {
			err := fmt.Errorf("single-token continuation used %d tiled FP32 Q4_K dispatches", candidate.ContinuationDispatches)
			logRecord("counter_failure", false, map[string]any{"error": err.Error(), "candidate": candidate})
			t.Fatal(err)
		}
		failures := standaloneCompareArms(t, &candidate.standaloneSequenceArm, scalar.standaloneSequenceArm)
		exact, exactFailures := compareQ4KTiledF32Exact(candidate.standaloneSequenceArm, scalar.standaloneSequenceArm)
		candidate.ExactLogitBitMismatches = exact.LogitBitMismatches
		candidate.ExactTokenIDMismatches = exact.TokenIDMismatches
		failures = append(failures, exactFailures...)
		if len(failures) != 0 {
			logRecord("quality_failure", false, map[string]any{"prompt_tokens": len(ids), "scalar": scalar, "candidate": candidate, "failures": failures})
			t.Fatal(strings.Join(failures, "; "))
		}
		return scalar, candidate
	}

	// Required-device smoke: one alternate P32 sequence plus continuation.
	scalar, candidate := runPair(prompt(32, 53, 3571))
	logRecord("smoke", true, map[string]any{"prompt_tokens": 32, "scalar": scalar, "candidate": candidate})
	t.Logf("q4k_tiled_fp32_smoke artifact=%s device=%q scalar_ns=%d candidate_ns=%d candidate_prefill_dispatches=%d",
		digest, be.Tier(), scalar.PrefillNanoseconds, candidate.PrefillNanoseconds, candidate.PrefillDispatches)

	// Default admission is a separate witness from explicit candidate timing.
	defaultArm, err := runQ4KTiledF32ModelArm(m, be, "default", prompt(32, 53, 3571), 4)
	if err != nil {
		logRecord("default_execution_failure", false, map[string]any{"arm": "default", "prompt_tokens": 32, "result": defaultArm, "error": err.Error()})
		t.Fatal(err)
	}
	defaultFailures := standaloneCompareArms(t, &defaultArm.standaloneSequenceArm, scalar.standaloneSequenceArm)
	defaultExact, defaultExactFailures := compareQ4KTiledF32Exact(defaultArm.standaloneSequenceArm, scalar.standaloneSequenceArm)
	defaultArm.ExactLogitBitMismatches = defaultExact.LogitBitMismatches
	defaultArm.ExactTokenIDMismatches = defaultExact.TokenIDMismatches
	defaultFailures = append(defaultFailures, defaultExactFailures...)
	logRecord("default_smoke", len(defaultFailures) == 0, map[string]any{"prompt_tokens": 32, "scalar": scalar, "default": defaultArm, "exact": defaultExact, "failures": defaultFailures})
	if len(defaultFailures) != 0 {
		t.Fatal(strings.Join(defaultFailures, "; "))
	}

	// Split continuity proves that a candidate-prefilled session can append and
	// decode without changing the strict logits/greedy contract.
	t.Setenv("FAK_VULKAN_Q4K_ARM", "")
	t.Setenv("FAK_VULKAN_Q4K_TILED_FP32", "candidate")
	split, err := standaloneRunSplit(m, be, prompt(32, 107, 1877))
	if err != nil {
		logRecord("split_failure", false, map[string]any{"error": err.Error(), "candidate_split": split})
		t.Fatal(err)
	}
	if split.TiledF32Dispatches == nil || *split.TiledF32Dispatches == 0 {
		err := fmt.Errorf("candidate split prefill produced no tiled FP32 Q4_K dispatch")
		logRecord("split_failure", false, map[string]any{"error": err.Error(), "candidate_split": split})
		t.Fatal(err)
	}
	alternatePrompt := prompt(32, 107, 1877)
	whole, err := runQ4KTiledF32ModelArm(m, be, "candidate", alternatePrompt, 4)
	if err != nil {
		logRecord("alternate_execution_failure", false, map[string]any{"arm": "candidate", "prompt_tokens": len(alternatePrompt), "result": whole, "error": err.Error()})
		t.Fatal(err)
	}
	scalarAlternate, err := runQ4KTiledF32ModelArm(m, be, "scalar", alternatePrompt, 4)
	if err != nil {
		logRecord("alternate_execution_failure", false, map[string]any{"arm": "scalar", "prompt_tokens": len(alternatePrompt), "result": scalarAlternate, "error": err.Error()})
		t.Fatal(err)
	}
	failures := standaloneCompareArms(t, &whole.standaloneSequenceArm, scalarAlternate.standaloneSequenceArm)
	failures = append(failures, standaloneCompareArms(t, &split, scalarAlternate.standaloneSequenceArm)...)
	wholeExact, wholeExactFailures := compareQ4KTiledF32Exact(whole.standaloneSequenceArm, scalarAlternate.standaloneSequenceArm)
	splitExact, splitExactFailures := compareQ4KTiledF32Exact(split, scalarAlternate.standaloneSequenceArm)
	whole.ExactLogitBitMismatches, whole.ExactTokenIDMismatches = wholeExact.LogitBitMismatches, wholeExact.TokenIDMismatches
	failures = append(failures, wholeExactFailures...)
	failures = append(failures, splitExactFailures...)
	logRecord("split_continuity", len(failures) == 0, map[string]any{"prompt_tokens": 32, "scalar": scalarAlternate, "candidate_whole": whole, "candidate_split": split, "whole_exact": wholeExact, "split_exact": splitExact, "failures": failures})
	if len(failures) != 0 {
		t.Fatal(strings.Join(failures, "; "))
	}
	if os.Getenv("FAK_QWEN35_Q4K_TILED_FP32_SMOKE_ONLY") == "1" {
		return
	}

	// Only reached after smoke and continuity pass: one excluded warmup followed
	// by six paired P128 samples, alternating order to limit order bias.
	timingPrompt := prompt(128, 17, 7919)
	warmScalar, warmCandidate := runPair(timingPrompt)
	logRecord("warmup", true, map[string]any{"prompt_tokens": 128, "included_in_distribution": false, "scalar": warmScalar, "candidate": warmCandidate})
	for pair := 0; pair < 6; pair++ {
		order := []string{"scalar", "candidate"}
		if pair%2 == 1 {
			order[0], order[1] = order[1], order[0]
		}
		arms := make(map[string]q4KTiledF32ModelArm, 2)
		for _, arm := range order {
			got, err := runQ4KTiledF32ModelArm(m, be, arm, timingPrompt, 4)
			if err != nil {
				logRecord("measured_pair_execution_failure", false, map[string]any{"pair": pair + 1, "order": order, "arm": arm, "prompt_tokens": len(timingPrompt), "result": got, "error": err.Error()})
				t.Fatal(err)
			}
			arms[arm] = got
		}
		candidateArm, scalarArm := arms["candidate"], arms["scalar"]
		failures := standaloneCompareArms(t, &candidateArm.standaloneSequenceArm, scalarArm.standaloneSequenceArm)
		exact, exactFailures := compareQ4KTiledF32Exact(candidateArm.standaloneSequenceArm, scalarArm.standaloneSequenceArm)
		candidateArm.ExactLogitBitMismatches = exact.LogitBitMismatches
		candidateArm.ExactTokenIDMismatches = exact.TokenIDMismatches
		failures = append(failures, exactFailures...)
		logRecord("measured_pair", len(failures) == 0, map[string]any{"pair": pair + 1, "order": order, "prompt_tokens": 128, "scalar": scalarArm, "candidate": candidateArm, "failures": failures})
		if len(failures) != 0 {
			t.Fatal(strings.Join(failures, "; "))
		}
		t.Logf("q4k_tiled_fp32_pair=%d order=%v scalar_ns=%d candidate_ns=%d candidate_prefill_dispatches=%d",
			pair+1, order, arms["scalar"].PrefillNanoseconds, arms["candidate"].PrefillNanoseconds, arms["candidate"].PrefillDispatches)
	}
}

func runQ4KTiledF32ModelArm(m *model.Model, be compute.Backend, arm string, prompt []int, continuation int) (q4KTiledF32ModelArm, error) {
	result := q4KTiledF32ModelArm{standaloneSequenceArm: standaloneSequenceArm{Name: arm, ExecutionMode: "whole-sequence-prefill"}}
	oldArm, hadArm := os.LookupEnv("FAK_VULKAN_Q4K_ARM")
	oldCandidate, hadCandidate := os.LookupEnv("FAK_VULKAN_Q4K_TILED_FP32")
	defer standaloneRestoreEnv("FAK_VULKAN_Q4K_ARM", oldArm, hadArm)
	defer standaloneRestoreEnv("FAK_VULKAN_Q4K_TILED_FP32", oldCandidate, hadCandidate)
	if arm == "scalar" {
		_ = os.Setenv("FAK_VULKAN_Q4K_ARM", "scalar")
	} else if arm == "candidate" {
		_ = os.Unsetenv("FAK_VULKAN_Q4K_ARM")
		_ = os.Setenv("FAK_VULKAN_Q4K_TILED_FP32", "candidate")
	} else if arm == "default" {
		_ = os.Unsetenv("FAK_VULKAN_Q4K_ARM")
		_ = os.Unsetenv("FAK_VULKAN_Q4K_TILED_FP32")
	} else {
		return result, fmt.Errorf("unsupported Q4_K tiled FP32 arm %q", arm)
	}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		return result, err
	}
	s.Quant, s.Q4K = true, true
	defer s.Close()
	before, _ := standaloneQ4Count(be)
	transfer := be.(q4KTiledF32TransferWitness)
	h2dBefore, d2hBefore := transfer.VulkanDebugTransferBytes()
	started := time.Now()
	logits := append([]float32(nil), s.Prefill(prompt)...)
	result.PrefillNanoseconds = time.Since(started).Nanoseconds()
	afterPrefill, _ := standaloneQ4Count(be)
	h2dAfterPrefill, d2hAfterPrefill := transfer.VulkanDebugTransferBytes()
	if afterPrefill < before {
		return result, fmt.Errorf("%s tiled FP32 counter regressed during prefill", arm)
	}
	if h2dAfterPrefill < h2dBefore || d2hAfterPrefill < d2hBefore {
		return result, fmt.Errorf("%s transfer counter regressed during prefill", arm)
	}
	result.PrefillDispatches = afterPrefill - before
	result.PrefillH2DBytes = h2dAfterPrefill - h2dBefore
	result.PrefillD2HBytes = d2hAfterPrefill - d2hBefore
	result.logits = [][]float32{logits}
	if err := standaloneFinite(arm+" prefill", logits); err != nil {
		return result, err
	}
	result.Route, _ = s.Qwen35SequencePrefillRouteStatus()
	if result.Route.EffectivePath != compute.Qwen35SequencePrefillPath || result.Route.FallbackActive || !result.Route.NativePerformanceQualifying {
		return result, fmt.Errorf("%s route=%+v", arm, result.Route)
	}
	for step := 0; step < continuation; step++ {
		id := standaloneArgmax(logits)
		result.GeneratedTokenIDs = append(result.GeneratedTokenIDs, id)
		started = time.Now()
		logits = append([]float32(nil), s.Step(id)...)
		result.ContinuationNS = append(result.ContinuationNS, time.Since(started).Nanoseconds())
		if err := standaloneFinite(fmt.Sprintf("%s continuation %d", arm, step), logits); err != nil {
			return result, err
		}
		result.logits = append(result.logits, logits)
	}
	result.GeneratedTokenIDs = append(result.GeneratedTokenIDs, standaloneArgmax(logits))
	afterContinuation, _ := standaloneQ4Count(be)
	h2dAfterContinuation, d2hAfterContinuation := transfer.VulkanDebugTransferBytes()
	if afterContinuation < afterPrefill {
		return result, fmt.Errorf("%s tiled FP32 counter regressed during continuation", arm)
	}
	if h2dAfterContinuation < h2dAfterPrefill || d2hAfterContinuation < d2hAfterPrefill {
		return result, fmt.Errorf("%s transfer counter regressed during continuation", arm)
	}
	result.ContinuationDispatches = afterContinuation - afterPrefill
	result.ContinuationH2DBytes = h2dAfterContinuation - h2dAfterPrefill
	result.ContinuationD2HBytes = d2hAfterContinuation - d2hAfterPrefill
	if (arm == "scalar" || arm == "default") && (result.PrefillDispatches != 0 || result.ContinuationDispatches != 0) {
		return result, fmt.Errorf("%s arm used tiled FP32 Q4_K: prefill=%d continuation=%d", arm, result.PrefillDispatches, result.ContinuationDispatches)
	}
	if arm == "candidate" && result.PrefillDispatches == 0 {
		return result, fmt.Errorf("%s arm produced no tiled FP32 Q4_K prefill dispatch", arm)
	}
	if result.ContinuationDispatches != 0 {
		return result, fmt.Errorf("%s single-token continuation used %d tiled FP32 Q4_K dispatches", arm, result.ContinuationDispatches)
	}
	delta := afterContinuation - before
	result.TiledF32Dispatches = &delta
	return result, nil
}
