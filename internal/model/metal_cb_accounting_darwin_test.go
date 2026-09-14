//go:build darwin && arm64 && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestQwen35FusedDecodeBlockOneCommandBufferPerLayer is the command-buffer
// regression for the W1 keystone: the fused linear-attention decode block — the
// GDN in-projections, convolution, recurrence, gated norm, output projection, and
// the whole SwiGLU MLP — must reach Metal as EXACTLY ONE command buffer, and its
// hidden output must stay numerically equivalent to the historical host
// blockStep/linearAttnStep decomposition within the documented GPU float band.
//
// The historical per-layer path issues its projections, GDN recurrence, and MLP as
// separate submit/sync round trips (the ~5-8 command buffers per layer the ~225/260
// per-token count was built from). Asserting CommandBuffers==1 here is what pins
// the collapse: if any sub-operation regresses to its own commit/wait, this fires.
func TestQwen35FusedDecodeBlockOneCommandBufferPerLayer(t *testing.T) {
	f := newQwen35DecodeBlockFixture(t)
	defer f.close()

	state, err := metalgemm.NewGDNState(f.geom)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	cpu := f.m.NewSession()
	cpu.Q4K = true
	seedQwen35DecodeBlockStates(t, f, state, cpu)
	defer cpu.Close()

	input := randomVecF(f.m.Cfg.HiddenSize, 94900)
	want := qwen35DecodeBlockCPU(f, cpu, input)
	got, receipt, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{
		Input: input, Weights: f.weights, State: state, Panel: f.panel, Block: &f.block,
	})
	if runErr != nil || !accepted {
		t.Fatalf("fused block accepted=%v err=%v", accepted, runErr)
	}
	if receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 {
		t.Fatalf("fused block command buffers=%d commits=%d waits=%d, want 1/1/1", receipt.CommandBuffers, receipt.Commits, receipt.CompletionWaits)
	}
	if receipt.Encoders <= 1 {
		t.Fatalf("fused block encoders=%d, want many dispatches inside one command buffer", receipt.Encoders)
	}
	// The whole-layer fusion must still reproduce the host decomposition.
	assertQwen35DecodeBlockParity(t, "fused block hidden", want, got)
}

// TestQwen35MetalCommandBufferAccountingCountsGraph verifies the Session hook the
// decode lever is measured through records a fused decode graph as one command
// buffer, so a test can assert a per-token total from real execution rather than
// from a self-report.
func TestQwen35MetalCommandBufferAccountingCountsGraph(t *testing.T) {
	s := &Session{}
	s.ResetMetalCommandBuffers()
	s.countMetalGraphCommandBuffer(1)
	s.countMetalCommandBuffer(1) // the LM head GEMV that follows the graph
	if got, want := s.MetalCommandBuffers(), 2; got != want {
		t.Fatalf("per-token Metal command buffers=%d, want %d", got, want)
	}
	if got, want := s.MetalGraphCommandBuffers(), 1; got != want {
		t.Fatalf("graph command buffers=%d, want %d", got, want)
	}
	if got, want := s.MetalDispatchCommandBuffers(), 1; got != want {
		t.Fatalf("dispatch command buffers=%d, want %d", got, want)
	}
}
