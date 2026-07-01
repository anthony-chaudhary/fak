package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestVLLMDeterminismParseNormalizesAndFailsClosed pins that unknown / empty
// determinism declarations fall back to "unavailable" rather than silently
// asserting reproducibility, and that the documented aliases resolve.
func TestVLLMDeterminismParseNormalizesAndFailsClosed(t *testing.T) {
	cases := map[string]VLLMDeterminism{
		"":                                DeterminismUnavailable,
		"   ":                             DeterminismUnavailable,
		"nonsense":                        DeterminismUnavailable,
		"unavailable":                     DeterminismUnavailable,
		"batch_invariance":                DeterminismBatchInvariant,
		"batch-invariant":                 DeterminismBatchInvariant,
		"INVARIANT":                       DeterminismBatchInvariant,
		"deterministic_offline_scheduler": DeterminismDeterministicOffline,
		"deterministic-offline":           DeterminismDeterministicOffline,
		"offline":                         DeterminismDeterministicOffline,
	}
	for in, want := range cases {
		if got := ParseVLLMDeterminism(in); got != want {
			t.Errorf("ParseVLLMDeterminism(%q) = %q, want %q", in, got, want)
		}
	}
	// A hand-built zero value normalizes to the fail-closed default.
	if got := VLLMDeterminism("").Normalize(); got != DeterminismUnavailable {
		t.Errorf("empty.Normalize() = %q, want %q", got, DeterminismUnavailable)
	}
}

// TestVLLMDeterminismCapabilityTokens pins the three negotiable capability tokens
// (acceptance criterion 1 of #1734).
func TestVLLMDeterminismCapabilityTokens(t *testing.T) {
	want := map[VLLMDeterminism]abi.Capability{
		DeterminismUnavailable:          "engine.vllm.determinism.unavailable",
		DeterminismBatchInvariant:       "engine.vllm.determinism.batch_invariance",
		DeterminismDeterministicOffline: "engine.vllm.determinism.deterministic_offline_scheduler",
	}
	for mode, tok := range want {
		if got := mode.Capability(); got != tok {
			t.Errorf("%q.Capability() = %q, want %q", mode, got, tok)
		}
	}
	// An unrecognized mode still advertises the honest "unavailable" token.
	if got := VLLMDeterminism("garbage").Capability(); got != "engine.vllm.determinism.unavailable" {
		t.Errorf("garbage.Capability() = %q, want unavailable token", got)
	}
}

// TestVLLMDeterminismTemperatureZeroGuard is the core witness for acceptance
// criterion 3: a replay/witness claim cannot cite temperature 0 alone as
// determinism when the engine reports dynamic batching without batch invariance.
func TestVLLMDeterminismTemperatureZeroGuard(t *testing.T) {
	// Unavailable: temperature 0 does NOT yield determinism.
	if DeterminismUnavailable.TemperatureZeroYieldsDeterminism() {
		t.Fatal("unavailable mode must NOT let temperature 0 imply determinism (criterion 3)")
	}
	if DeterminismUnavailable.GuaranteesReproducibleOutput() {
		t.Fatal("unavailable mode must not guarantee reproducible output")
	}
	// Both real guarantees DO make temperature 0 reproducible.
	for _, mode := range []VLLMDeterminism{DeterminismBatchInvariant, DeterminismDeterministicOffline} {
		if !mode.TemperatureZeroYieldsDeterminism() {
			t.Errorf("%q must let temperature 0 imply determinism", mode)
		}
		if !mode.GuaranteesReproducibleOutput() {
			t.Errorf("%q must guarantee reproducible output", mode)
		}
	}
}

// TestVLLMEngineDeterminismAccessorReadsEnv pins that the EngineDriver-level
// accessor reports the launch-time determinism posture from FAK_VLLM_DETERMINISM,
// defaulting to unavailable when unset.
func TestVLLMEngineDeterminismAccessorReadsEnv(t *testing.T) {
	eng := NewVLLMEngine(VLLMConfig{})

	t.Setenv("FAK_VLLM_DETERMINISM", "")
	if got := eng.Determinism(); got != DeterminismUnavailable {
		t.Errorf("unset env: Determinism() = %q, want %q", got, DeterminismUnavailable)
	}
	if got := eng.DeterminismCapability(); got != "engine.vllm.determinism.unavailable" {
		t.Errorf("unset env: DeterminismCapability() = %q, want unavailable token", got)
	}

	t.Setenv("FAK_VLLM_DETERMINISM", "batch-invariant")
	if got := eng.Determinism(); got != DeterminismBatchInvariant {
		t.Errorf("batch-invariant env: Determinism() = %q, want %q", got, DeterminismBatchInvariant)
	}
	if got := eng.DeterminismCapability(); got != "engine.vllm.determinism.batch_invariance" {
		t.Errorf("batch-invariant env: DeterminismCapability() = %q, want batch_invariance token", got)
	}
}
