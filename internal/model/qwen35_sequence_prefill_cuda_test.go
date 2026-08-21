//go:build cuda

package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type cudaSequenceEntryWitness struct {
	cudaGDNParityBackend
	requests int
}

func (*cudaSequenceEntryWitness) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (b *cudaSequenceEntryWitness) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	b.requests++
	if req.Path != compute.Qwen35SequencePrefillPath || req.KV == nil || len(req.States) != len(req.Layers) {
		return compute.Qwen35SequencePrefillResult{}, errors.New("incomplete resident sequence request")
	}
	return compute.Qwen35SequencePrefillResult{}, errors.New("witness stop before unimplemented full-layer execution")
}

func TestCUDAQwen35SequencePrefillRealContractEntryPoint(t *testing.T) {
	be := &cudaSequenceEntryWitness{cudaGDNParityBackend: requiredCUDAGDNParityBackend(t)}
	m := NewSynthetic(qwen35HybridTestCfg())
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("partial CUDA sequence implementation did not fail closed")
		}
		if be.requests != 1 || !s.halClosed || s.halFailure == nil {
			t.Fatalf("requests=%d closed=%v failure=%v", be.requests, s.halClosed, s.halFailure)
		}
		if be.Qwen35GDNOperationCount() != 0 {
			t.Fatalf("contract refusal replayed %d scalar CUDA GDN operations", be.Qwen35GDNOperationCount())
		}
	}()
	be.ResetQwen35GDNOperationCount()
	s.Prefill([]int{3, 7})
}
