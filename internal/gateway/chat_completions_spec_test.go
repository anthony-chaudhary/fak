package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine/specutil"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// qualifiedVulkanMTPConfig builds a Config that passes every capability check
// in shouldEnableVulkanMTP (the same envelope as admittedVulkanMTPConfig) but
// WITHOUT the FAK_SPECULATIVE / SPECULATIVE env vars, so the default-on path is
// what admits it.
func qualifiedVulkanMTPConfig(t *testing.T) Config {
	t.Helper()
	t.Setenv("FAK_SPECULATIVE", "")
	t.Setenv("SPECULATIVE", "")
	return Config{
		InKernelModel: model.NewSyntheticQwen38MTP(),
		InKernelQ4K:   true,
		Backend: vulkanMTPGatewayBackend{
			Backend:       compute.Default(),
			name:          "vulkan",
			allLogitsPath: compute.Qwen35SequenceAllLogitsPath,
			rawHiddenPath: compute.Qwen35SequenceRawHiddenPath,
		},
	}
}

// withQualificationMatrix swaps the package-level qualification matrix for the
// duration of the test and restores it on return.
func withQualificationMatrix(t *testing.T, m *specutil.Matrix) {
	t.Helper()
	old := specQualificationMatrix
	specQualificationMatrix = m
	t.Cleanup(func() { specQualificationMatrix = old })
}

// hwWitnessedMatrix returns a matrix with a single HW-WITNESSED qualified pair
// for the synthetic Qwen3.8 MTP model on the Vulkan backend.
func hwWitnessedMatrix() *specutil.Matrix {
	m := specutil.NewMatrix()
	m.Record(specutil.Qualification{
		Key:           specutil.QualificationKey{Backend: "vulkan", Model: model.NewSyntheticQwen38MTP().Cfg.Name},
		Qualified:     true,
		Parity:        specutil.ParityResult{Pass: true, DivergenceIndex: -1},
		EvidenceClass: "HW-WITNESSED",
	})
	return m
}

func TestShouldEnableVulkanMTP_DefaultOnFlip(t *testing.T) {
	t.Run("escape hatch disables a qualified pair", func(t *testing.T) {
		withQualificationMatrix(t, hwWitnessedMatrix())
		cfg := qualifiedVulkanMTPConfig(t)
		cfg.DisableVulkanMTP = true
		if shouldEnableVulkanMTP(cfg) {
			t.Fatal("DisableVulkanMTP did not force ordinary decode for a qualified pair")
		}
	})

	t.Run("qualified pair is default-on without env", func(t *testing.T) {
		withQualificationMatrix(t, hwWitnessedMatrix())
		cfg := qualifiedVulkanMTPConfig(t)
		if !shouldEnableVulkanMTP(cfg) {
			t.Fatal("HW-WITNESSED qualified pair was not admitted by default (no env var set)")
		}
	})

	t.Run("unqualified pair is not default-on without env", func(t *testing.T) {
		withQualificationMatrix(t, specutil.NewMatrix())
		cfg := qualifiedVulkanMTPConfig(t)
		if shouldEnableVulkanMTP(cfg) {
			t.Fatal("unqualified pair was admitted by default without an env var")
		}
	})

	t.Run("env override still admits an unqualified pair", func(t *testing.T) {
		withQualificationMatrix(t, specutil.NewMatrix())
		cfg := qualifiedVulkanMTPConfig(t) // clears FAK_SPECULATIVE / SPECULATIVE
		t.Setenv("FAK_SPECULATIVE", "mtp") // operator override, set after the config
		t.Setenv("SPECULATIVE", "")
		if !shouldEnableVulkanMTP(cfg) {
			t.Fatal("FAK_SPECULATIVE=mtp did not admit an unqualified pair (operator override broken)")
		}
	})

	t.Run("MOCK pair does not default-on", func(t *testing.T) {
		// The built-in matrix carries only a MOCK pair; it must NOT default-on.
		withQualificationMatrix(t, specutil.DefaultMatrix())
		cfg := qualifiedVulkanMTPConfig(t)
		if shouldEnableVulkanMTP(cfg) {
			t.Fatal("MOCK pair qualified for default-on; only HW-WITNESSED pairs may")
		}
	})
}
