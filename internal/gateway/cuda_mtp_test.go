package gateway

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// cudaSequenceOnlyGatewayBackend models the resident CUDA surface that exists
// before the MTP paths: sequence prefill and all-row logits, but no draft fuse,
// raw-hidden capture or prefix replay.
type cudaSequenceOnlyGatewayBackend struct {
	compute.Backend
}

func (cudaSequenceOnlyGatewayBackend) Name() string { return "cuda" }

func (cudaSequenceOnlyGatewayBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (cudaSequenceOnlyGatewayBackend) Qwen35SequencePrefill(compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	return compute.Qwen35SequencePrefillResult{}, errors.New("selector-only backend must not execute")
}

func (cudaSequenceOnlyGatewayBackend) Qwen35SequenceAllLogitsPath() string {
	return compute.Qwen35SequenceAllLogitsPath
}

// cudaMTPCapabilityGatewayBackend advertises every resident MTP path with a
// configurable identity so each capability can be made stale in turn.
type cudaMTPCapabilityGatewayBackend struct {
	cudaSequenceOnlyGatewayBackend
	prefill, allLogits, rawHidden, draft, replay string
}

func cudaMTPFullCapabilities() cudaMTPCapabilityGatewayBackend {
	return cudaMTPCapabilityGatewayBackend{
		cudaSequenceOnlyGatewayBackend: cudaSequenceOnlyGatewayBackend{Backend: compute.Default()},
		prefill:                        compute.Qwen35SequencePrefillPath,
		allLogits:                      compute.Qwen35SequenceAllLogitsPath,
		rawHidden:                      compute.Qwen35SequenceRawHiddenPath,
		draft:                          compute.Qwen35MTPDraftPath,
		replay:                         compute.Qwen35SequencePrefixReplayPath,
	}
}

func (b cudaMTPCapabilityGatewayBackend) Qwen35SequencePrefillPath() string { return b.prefill }
func (b cudaMTPCapabilityGatewayBackend) Qwen35SequenceAllLogitsPath() string {
	return b.allLogits
}
func (b cudaMTPCapabilityGatewayBackend) Qwen35SequenceRawHiddenPath() string {
	return b.rawHidden
}
func (b cudaMTPCapabilityGatewayBackend) Qwen35MTPDraftPath() string { return b.draft }
func (cudaMTPCapabilityGatewayBackend) Qwen35MTPFuse(compute.Qwen35MTPFuseRequest) (compute.Tensor, error) {
	return compute.Tensor{}, errors.New("selector-only backend must not execute")
}
func (b cudaMTPCapabilityGatewayBackend) Qwen35SequencePrefixReplayPath() string {
	return b.replay
}

// TestCUDAMTPAdmittedOnlyWithEveryResidentPath pins the CUDA arm of the
// resident MTP gate: a CUDA backend is admitted only when it advertises the
// draft fuse, sequence prefill, all-row logits, raw-hidden and prefix-replay
// paths; the CUDA surface without the MTP paths stays on ordinary decode.
//
// fak-test:runtime fast est=10ms lane=default
func TestCUDAMTPAdmittedOnlyWithEveryResidentPath(t *testing.T) {
	cfg := admittedVulkanMTPConfig(t)
	cfg.Backend = cudaMTPFullCapabilities()
	if !shouldEnableVulkanMTP(cfg) {
		t.Fatal("CUDA backend advertising every resident MTP path was not admitted")
	}
	planner := newInKernelChatPlanner(cfg, "qwen38-cuda", t.Logf).(*agent.InKernelPlanner)
	if !planner.VulkanMTPEnabled() {
		t.Fatal("admitted CUDA route did not configure request-bound resident MTP")
	}

	refused := map[string]compute.Backend{
		"cuda without MTP paths": cudaSequenceOnlyGatewayBackend{Backend: compute.Default()},
	}
	for name, edit := range map[string]func(*cudaMTPCapabilityGatewayBackend){
		"stale prefill":    func(b *cudaMTPCapabilityGatewayBackend) { b.prefill = "stale-contract" },
		"stale all-logits": func(b *cudaMTPCapabilityGatewayBackend) { b.allLogits = "" },
		"stale raw-hidden": func(b *cudaMTPCapabilityGatewayBackend) { b.rawHidden = "" },
		"stale draft":      func(b *cudaMTPCapabilityGatewayBackend) { b.draft = "stale-contract" },
		"stale replay":     func(b *cudaMTPCapabilityGatewayBackend) { b.replay = "" },
	} {
		backend := cudaMTPFullCapabilities()
		edit(&backend)
		refused[name] = backend
	}
	for name, backend := range refused {
		t.Run(name, func(t *testing.T) {
			cfg := admittedVulkanMTPConfig(t)
			cfg.Backend = backend
			if shouldEnableVulkanMTP(cfg) {
				t.Fatal("CUDA backend missing a resident MTP path was admitted")
			}
			planner := newInKernelChatPlanner(cfg, "qwen38-cuda", t.Logf).(*agent.InKernelPlanner)
			if planner.VulkanMTPEnabled() {
				t.Fatal("refused CUDA route configured resident MTP")
			}
		})
	}
}
