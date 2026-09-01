//go:build darwin && arm64 && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestQwen38MTPQ4KForwardExecutesResidentMetalWeights(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := qwen38MTPQ4KTestModels(t)
	forward, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(forward.Close)
	if !forward.draft.MetalQ4K || forward.tensorFormat != Qwen38MTPFormatQ4K {
		t.Fatalf("draft mechanism format=%q metal=%v", forward.tensorFormat, forward.draft.MetalQ4K)
	}

	prior, embedding := qwen38MTPInputs(m.Cfg.HiddenSize, 0)
	if _, err := forward.Forward(0, prior, embedding); err != nil {
		t.Fatalf("execute Metal Q4_K MTP forward: %v", err)
	}

	metalQ4KMu.Lock()
	resident := metalQ4KW[forward.draft.M]
	_, fc := resident["mtp.fc.weight"]
	_, q := resident["model.layers.0.self_attn.q_proj.weight"]
	_, gate := resident["model.layers.0.mlp.gate_proj.weight"]
	metalQ4KMu.Unlock()
	if !fc || !q || !gate {
		t.Fatalf("Metal MTP residency fc=%v q_proj=%v gate_proj=%v keys=%v", fc, q, gate, resident)
	}

	receipt := EvaluateQwen38MTPEligibility(Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact: true,
		MTPBackendReady:   true,
		Backend:           Qwen38MTPBackendMetal,
		Model:             m,
		Greedy:            true,
		Depth:             3,
		FreshSession:      true,
		MemoryHeadroomOK:  true,
		OperatorEnabled:   true,
	})
	if receipt.Engine != Qwen38EngineMTP ||
		receipt.Backend != Qwen38MTPBackendMetal ||
		receipt.MTPTensorFormat != Qwen38MTPFormatQ4K ||
		receipt.RequestedDepth != 3 ||
		!receipt.TargetEquivalent {
		t.Fatalf("Metal mechanism receipt=%+v", receipt)
	}
}
