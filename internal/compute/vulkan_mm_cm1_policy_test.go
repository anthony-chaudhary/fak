package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=5ms
func TestVulkanMMCM1ConfigFromEnv(t *testing.T) {
	for _, tc := range []struct {
		env  map[string]string
		want VulkanMMCM1Config
	}{
		{nil, VulkanMMCM1Config{Enabled: true, MinTokens: vulkanMMCM1DefaultMinTokens}},
		{map[string]string{vulkanMMCM1EnvEnable: "0"}, VulkanMMCM1Config{Enabled: false, MinTokens: vulkanMMCM1DefaultMinTokens}},
		{map[string]string{vulkanMMCM1EnvEnable: "1"}, VulkanMMCM1Config{Enabled: true, MinTokens: vulkanMMCM1DefaultMinTokens}},
		{map[string]string{vulkanMMCM1EnvEnable: "0", vulkanMMCM1EnvMinTokens: "16"}, VulkanMMCM1Config{Enabled: false, MinTokens: 16}},
		{map[string]string{vulkanMMCM1EnvMinTokens: "16"}, VulkanMMCM1Config{Enabled: true, MinTokens: 16}},
		{map[string]string{vulkanMMCM1EnvMinTokens: "0"}, VulkanMMCM1Config{Enabled: true, MinTokens: vulkanMMCM1DefaultMinTokens}},
		{map[string]string{vulkanMMCM1EnvMinTokens: "x"}, VulkanMMCM1Config{Enabled: true, MinTokens: vulkanMMCM1DefaultMinTokens}},
	} {
		got := vulkanMMCM1ConfigFromEnv(func(k string) string { return tc.env[k] })
		if got != tc.want {
			t.Errorf("env %v: got %+v want %+v", tc.env, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanMMCM1DecideNamesEveryFallback(t *testing.T) {
	on := VulkanMMCM1Config{Enabled: true, MinTokens: 8}
	for _, tc := range []struct {
		cfg    VulkanMMCM1Config
		dt     Dtype
		tokens int
		want   error
	}{
		{VulkanMMCM1Config{MinTokens: 8}, Q2_K, 512, ErrVulkanMMCM1Disabled},
		{on, Q2_K, 8, ErrVulkanMMCM1BelowThreshold},
		{on, IQ2_XXS, 1, ErrVulkanMMCM1BelowThreshold},
		{on, Q6_K, 512, ErrVulkanMMCM1UnsupportedDtype},
		{on, Q2_K, 9, nil},
		{on, IQ2_XXS, 512, nil},
	} {
		err := vulkanMMCM1Decide(tc.cfg, tc.dt, tc.tokens)
		if tc.want == nil {
			if err != nil {
				t.Errorf("%v tokens=%d: unexpected refusal %v", tc.dt, tc.tokens, err)
			}
			continue
		}
		if !errors.Is(err, tc.want) {
			t.Errorf("%v tokens=%d: got %v want %v", tc.dt, tc.tokens, err, tc.want)
		}
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanMMCM1StatusErrorIsClosed(t *testing.T) {
	for code, want := range map[int]error{
		vulkanMMCM1StatusNotInitialized: ErrVulkanMMCM1Unavailable,
		vulkanMMCM1StatusNoCoopMat:      ErrVulkanMMCM1Unavailable,
		vulkanMMCM1StatusNoShaderF16:    ErrVulkanMMCM1Unavailable,
		vulkanMMCM1StatusNoF16Shape:     ErrVulkanMMCM1Unavailable,
		vulkanMMCM1StatusNoModule:       ErrVulkanMMCM1Unavailable,
		vulkanMMCM1StatusBadFormat:      ErrVulkanMMCM1UnsupportedDtype,
		vulkanMMCM1StatusBadShape:       ErrVulkanMMCM1Shape,
		vulkanMMCM1StatusDispatchFailed: ErrVulkanMMCM1Dispatch,
		99:                              ErrVulkanMMCM1Unavailable,
	} {
		if err := vulkanMMCM1StatusError(code); !errors.Is(err, want) {
			t.Errorf("status %d: got %v want %v", code, err, want)
		}
	}
	if err := vulkanMMCM1StatusError(vulkanMMCM1StatusOK); err != nil {
		t.Errorf("status OK: got %v", err)
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanMMCM1RecordCountsByReason(t *testing.T) {
	before := VulkanMMCM1Snapshot()
	recordVulkanMMCM1(Q2_K, nil, "q2k_matmul")
	recordVulkanMMCM1(Q2_K, ErrVulkanMMCM1Disabled, "q2k_matmul")
	recordVulkanMMCM1(Q2_K, ErrVulkanMMCM1BelowThreshold, "q2k_matmul")
	recordVulkanMMCM1(Q6_K, vulkanMMCM1Decide(VulkanMMCM1Config{Enabled: true, MinTokens: 1}, Q6_K, 4), "q6k_matmul")
	recordVulkanMMCM1(Q2_K, vulkanMMCM1StatusError(vulkanMMCM1StatusNoModule), "q2k_matmul")
	recordVulkanMMCM1(Q2_K, vulkanMMCM1StatusError(vulkanMMCM1StatusBadShape), "q2k_matmul")
	recordVulkanMMCM1(Q2_K, vulkanMMCM1StatusError(vulkanMMCM1StatusDispatchFailed), "q2k_matmul")
	after := VulkanMMCM1Snapshot()
	delta := VulkanMMCM1Stats{
		GEMM: after.GEMM - before.GEMM, Disabled: after.Disabled - before.Disabled,
		BelowThreshold: after.BelowThreshold - before.BelowThreshold, UnsupportedDtype: after.UnsupportedDtype - before.UnsupportedDtype,
		Unavailable: after.Unavailable - before.Unavailable, Shape: after.Shape - before.Shape, Dispatch: after.Dispatch - before.Dispatch,
	}
	if delta != (VulkanMMCM1Stats{1, 1, 1, 1, 1, 1, 1}) {
		t.Fatalf("counter deltas %+v, want one per reason", delta)
	}
}
