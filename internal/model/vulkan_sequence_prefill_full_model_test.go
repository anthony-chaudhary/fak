//go:build vulkan && (windows || linux) && cgo

package model_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type standaloneQ4DispatchWitness interface {
	VulkanDebugQ4KCoopMatDispatches() uint64
}

type standaloneSequenceArm struct {
	Name                 string                                 `json:"name"`
	PrefillNanoseconds   int64                                  `json:"prefill_nanoseconds"`
	ContinuationNS       []int64                                `json:"continuation_nanoseconds"`
	GeneratedTokenIDs    []int                                  `json:"generated_token_ids"`
	CoopMatDispatches    *uint64                                `json:"q4_cooperative_dispatches"`
	Route                model.Qwen35SequencePrefillRouteStatus `json:"route"`
	ExecutionMode        string                                 `json:"execution_mode"`
	PrefillCosine        float64                                `json:"prefill_cosine,omitempty"`
	PrefillRelativeL2    float64                                `json:"prefill_relative_l2,omitempty"`
	ContinuationCosine   []float64                              `json:"continuation_cosine,omitempty"`
	ContinuationRelative []float64                              `json:"continuation_relative_l2,omitempty"`
	logits               [][]float32
}

func TestVulkanFullModelSequencePrefillStandalone(t *testing.T) {
	if os.Getenv("FAK_QWEN35_SEQUENCE_FULL_MODEL_WITNESS") != "1" {
		t.Skip("set FAK_QWEN35_SEQUENCE_FULL_MODEL_WITNESS=1 for the bounded hardware witness")
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
	q8, ok := be.(interface {
		Q8MatMul2DDispatchGrid(int, int) (int, int, int)
	})
	if !ok {
		t.Fatal("backend lacks native Q8 route capability inspection")
	}
	qx, qy, qz := q8.Q8MatMul2DDispatchGrid(5120, 32)
	t.Setenv("FAK_VULKAN_Q4K_ARM", "scalar")
	t.Setenv("FAK_VULKAN_Q4K_COOPMAT", "")
	digest := standaloneHashArtifact(t, path)
	if expected := os.Getenv("FAK_QWEN38_GGUF_SHA256"); expected != "" && !strings.EqualFold(digest, expected) {
		t.Fatal("checkpoint SHA-256 mismatch")
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
	_, counterAvailable := be.(standaloneQ4DispatchWitness)
	provenance := map[string]any{
		"source_archive_sha256":                 os.Getenv("FAK_SOURCE_ARCHIVE_SHA256"),
		"binary_sha256":                         os.Getenv("FAK_BINARY_SHA256"),
		"shader_bundle_sha256":                  os.Getenv("FAK_SHADER_BUNDLE_SHA256"),
		"artifact_sha256":                       digest,
		"device":                                be.Tier(),
		"layers":                                m.Cfg.NumLayers,
		"q8_native_dispatch_grid_out5120_p32":   [3]int{qx, qy, qz},
		"q8_native_cooperative_route_available": qx == 160 && qy == 1 && qz == 1,
		"quality_gate":                          map[string]float64{"cosine_min": .99999, "relative_l2_max": 1e-4},
		"q4_cooperative_candidate":              "disabled",
		"q4_dispatch_counter_available":         counterAvailable,
	}
	logRecord := func(phase string, record map[string]any) {
		t.Helper()
		record["schema"] = "fak.vulkan-sequence-full-model-default/v2"
		record["phase"], record["observed_utc"], record["provenance"] = phase, time.Now().UTC().Format(time.RFC3339Nano), provenance
		data, err := json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		t.Log(string(data))
	}
	run := func(arm string, prompt []int) standaloneSequenceArm {
		t.Helper()
		var result standaloneSequenceArm
		var err error
		if arm == "split_16_16" {
			result, err = standaloneRunSplit(m, be, prompt)
		} else {
			result, err = standaloneRunArm(m, be, arm, prompt, 4)
		}
		if err == nil && result.CoopMatDispatches != nil && *result.CoopMatDispatches != 0 {
			err = fmt.Errorf("%s unexpectedly used %d Q4 cooperative dispatches", arm, *result.CoopMatDispatches)
		}
		if err != nil {
			logRecord("execution_failure", map[string]any{"passed": false, "arm": result, "error": err.Error()})
			t.Fatal(err)
		}
		return result
	}
	prompt := func(n, offset, stride int) []int {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (offset + i*stride) % m.Cfg.VocabSize
		}
		return ids
	}

	continuityPrompt := prompt(32, 53, 3571)
	token := run("token_scalar", continuityPrompt)
	whole := run("scalar", continuityPrompt)
	split := run("split_16_16", continuityPrompt)
	failures := standaloneCompareArms(t, &whole, token)
	failures = append(failures, standaloneCompareArms(t, &split, token)...)
	logRecord("continuity", map[string]any{
		"passed": len(failures) == 0, "failures": failures, "prompt_token_ids": continuityPrompt,
		"token_reference": token, "whole_sequence": whole, "split_sequence": split,
		"scope": "P32 correctness; 16 PrefillNoLogits plus 16 Prefill then four Steps; timing excluded from distribution",
	})
	if len(failures) != 0 {
		t.Fatal(strings.Join(failures, "; "))
	}
	if os.Getenv("FAK_QWEN35_SEQUENCE_CONTINUITY_ONLY") == "1" {
		logRecord("continuity_only_complete", map[string]any{
			"passed": true, "prompt_tokens": 32, "continuation_steps_per_arm": 4,
			"scope":                        "exact clean production snapshot continuity only; no P128 warmup or N=6 timing distribution",
			"timing_distribution_complete": false,
		})
		return
	}

	timingPrompt := prompt(128, 17, 7919)
	token, whole = run("token_scalar", timingPrompt), run("scalar", timingPrompt)
	failures = standaloneCompareArms(t, &whole, token)
	logRecord("warmup", map[string]any{
		"passed": len(failures) == 0, "failures": failures, "prompt_token_ids": timingPrompt,
		"token_reference": token, "whole_sequence": whole, "included_in_distribution": false,
	})
	if len(failures) != 0 {
		t.Fatal(strings.Join(failures, "; "))
	}
	for pair := 0; pair < 6; pair++ {
		order := []string{"token_scalar", "scalar"}
		if pair%2 == 1 {
			order[0], order[1] = order[1], order[0]
		}
		arms := make(map[string]standaloneSequenceArm, 2)
		for _, name := range order {
			arms[name] = run(name, timingPrompt)
		}
		token, whole = arms["token_scalar"], arms["scalar"]
		failures = standaloneCompareArms(t, &whole, token)
		logRecord("measured_pair", map[string]any{
			"pair": pair + 1, "order": order, "passed": len(failures) == 0, "failures": failures,
			"prompt_tokens": len(timingPrompt), "token_reference": token, "whole_sequence": whole,
			"timing_scope": "engine prefill through final logits; excludes model loading, tokenization and transport",
		})
		if len(failures) != 0 {
			t.Fatal(strings.Join(failures, "; "))
		}
	}
	logRecord("complete", map[string]any{
		"passed": true, "samples_per_arm": 6, "prompt_tokens": 128, "warmups_per_arm_excluded": 1,
		"order": "TS,ST,TS,ST,TS,ST", "continuation_steps_per_arm": 4,
		"distribution_reporting": "compute P50/P90 and seeded bootstrap 95% confidence intervals from all six raw observations",
	})
}

func standaloneRunArm(m *model.Model, be compute.Backend, arm string, prompt []int, continuation int) (standaloneSequenceArm, error) {
	result := standaloneSequenceArm{Name: arm}
	oldArm, hadArm := os.LookupEnv("FAK_VULKAN_Q4K_ARM")
	oldCandidate, hadCandidate := os.LookupEnv("FAK_VULKAN_Q4K_COOPMAT")
	_ = os.Setenv("FAK_VULKAN_Q4K_ARM", "scalar")
	_ = os.Unsetenv("FAK_VULKAN_Q4K_COOPMAT")
	defer standaloneRestoreEnv("FAK_VULKAN_Q4K_ARM", oldArm, hadArm)
	defer standaloneRestoreEnv("FAK_VULKAN_Q4K_COOPMAT", oldCandidate, hadCandidate)

	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		return result, fmt.Errorf("%s session: %w", arm, err)
	}
	s.Quant, s.Q4K = true, true
	defer s.Close()
	before, measured := standaloneQ4Count(be)
	started := time.Now()
	var logits []float32
	if arm == "token_scalar" {
		result.ExecutionMode = "single-token-prefill-no-logits-then-final-logits"
		for _, id := range prompt[:len(prompt)-1] {
			s.PrefillNoLogits([]int{id})
		}
		logits = append([]float32(nil), s.Prefill(prompt[len(prompt)-1:])...)
	} else if arm == "scalar" {
		result.ExecutionMode = "whole-sequence-prefill"
		logits = append([]float32(nil), s.Prefill(prompt)...)
	} else {
		return result, fmt.Errorf("unsupported standalone arm %q", arm)
	}
	result.PrefillNanoseconds, result.logits = time.Since(started).Nanoseconds(), [][]float32{logits}
	if err := standaloneFinite(arm+" prefill", logits); err != nil {
		return result, err
	}
	route, present := s.Qwen35SequencePrefillRouteStatus()
	result.Route = route
	if arm == "scalar" && (!present || route.EffectivePath != compute.Qwen35SequencePrefillPath || route.FallbackActive || !route.NativePerformanceQualifying) {
		return result, fmt.Errorf("%s route=%+v present=%t", arm, route, present)
	}
	if arm == "token_scalar" && present && route.NativePerformanceQualifying {
		return result, fmt.Errorf("token reference unexpectedly admitted sequence route: %+v", route)
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
	if measured {
		after, _ := standaloneQ4Count(be)
		if after < before {
			return result, fmt.Errorf("%s Q4 cooperative dispatch counter regressed", arm)
		}
		delta := after - before
		result.CoopMatDispatches = &delta
	}
	return result, nil
}

func standaloneRunSplit(m *model.Model, be compute.Backend, prompt []int) (standaloneSequenceArm, error) {
	result := standaloneSequenceArm{Name: "split_16_16", ExecutionMode: "16-prefill-no-logits-then-16-prefill"}
	if len(prompt) != 32 { //boundarylint:ignore CHANGE_DETECTOR_TEST -- fixed hardware continuity fixture: two 16-token sequence panels
		return result, fmt.Errorf("split witness requires exactly 32 tokens")
	}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		return result, err
	}
	s.Quant, s.Q4K = true, true
	defer s.Close()
	before, measured := standaloneQ4Count(be)
	started := time.Now()
	s.PrefillNoLogits(prompt[:16])
	checkRoute := func() error {
		route, present := s.Qwen35SequencePrefillRouteStatus()
		result.Route = route
		if !present || route.EffectivePath != compute.Qwen35SequencePrefillPath || route.FallbackActive || !route.NativePerformanceQualifying {
			return fmt.Errorf("split sequence route=%+v present=%t", route, present)
		}
		return nil
	}
	if err := checkRoute(); err != nil {
		return result, err
	}
	logits := append([]float32(nil), s.Prefill(prompt[16:])...)
	result.PrefillNanoseconds = time.Since(started).Nanoseconds()
	if err := checkRoute(); err != nil {
		return result, err
	}
	if err := standaloneFinite("split prefill", logits); err != nil {
		return result, err
	}
	result.logits = [][]float32{logits}
	for step := 0; step < 4; step++ {
		id := standaloneArgmax(logits)
		result.GeneratedTokenIDs = append(result.GeneratedTokenIDs, id)
		started = time.Now()
		logits = append([]float32(nil), s.Step(id)...)
		result.ContinuationNS = append(result.ContinuationNS, time.Since(started).Nanoseconds())
		if err := standaloneFinite("split continuation", logits); err != nil {
			return result, err
		}
		result.logits = append(result.logits, logits)
	}
	result.GeneratedTokenIDs = append(result.GeneratedTokenIDs, standaloneArgmax(logits))
	if measured {
		after, _ := standaloneQ4Count(be)
		if after < before {
			return result, fmt.Errorf("split Q4 cooperative dispatch counter regressed")
		}
		delta := after - before
		result.CoopMatDispatches = &delta
	}
	return result, nil
}

func standaloneQ4Count(be compute.Backend) (uint64, bool) {
	w, ok := be.(standaloneQ4DispatchWitness)
	if !ok {
		return 0, false
	}
	return w.VulkanDebugQ4KCoopMatDispatches(), true
}

func standaloneCompareArms(t *testing.T, got *standaloneSequenceArm, want standaloneSequenceArm) []string {
	t.Helper()
	label := got.Name + "_vs_" + want.Name
	if len(got.logits) != len(want.logits) || len(got.GeneratedTokenIDs) != len(want.GeneratedTokenIDs) {
		t.Fatalf("%s vector/token counts differ", label)
	}
	var failures []string
	for step := range want.logits {
		cosine, relativeL2, passed := standaloneCompareLogits(t, label, got.logits[step], want.logits[step])
		if step == 0 {
			got.PrefillCosine, got.PrefillRelativeL2 = cosine, relativeL2
		} else {
			got.ContinuationCosine = append(got.ContinuationCosine, cosine)
			got.ContinuationRelative = append(got.ContinuationRelative, relativeL2)
		}
		if !passed {
			failures = append(failures, fmt.Sprintf("%s_step_%d cosine=%g relative_l2=%g", label, step, cosine, relativeL2))
		}
		if want.GeneratedTokenIDs[step] != got.GeneratedTokenIDs[step] {
			failures = append(failures, fmt.Sprintf("%s_step_%d greedy want=%d got=%d", label, step, want.GeneratedTokenIDs[step], got.GeneratedTokenIDs[step]))
		}
	}
	return failures
}

func standaloneCompareLogits(t *testing.T, label string, got, want []float32) (float64, float64, bool) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("%s logit lengths got=%d want=%d", label, len(got), len(want))
	}
	var dot, gotNorm, wantNorm, errorNorm float64
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
		d := g - w
		errorNorm += d * d
	}
	cosine := dot / math.Sqrt(gotNorm*wantNorm)
	relativeL2 := math.Sqrt(errorNorm / wantNorm)
	passed := !math.IsNaN(cosine) && !math.IsInf(cosine, 0) && cosine >= .99999 &&
		!math.IsNaN(relativeL2) && !math.IsInf(relativeL2, 0) && relativeL2 <= 1e-4
	return cosine, relativeL2, passed
}

func standaloneFinite(label string, values []float32) error {
	if len(values) == 0 {
		return fmt.Errorf("%s returned no logits", label)
	}
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s logit %d is non-finite: %g", label, i, value)
		}
	}
	return nil
}

func standaloneArgmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

func standaloneHashArtifact(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func standaloneRestoreEnv(key, value string, present bool) {
	if present {
		_ = os.Setenv(key, value)
	} else {
		_ = os.Unsetenv(key)
	}
}
