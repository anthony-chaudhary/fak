package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func TestServeArtifactResidentQ4KUsesArtifactNotArmLabel(t *testing.T) {
	t.Setenv("FAK_Q4K", "1")
	backend := serveCapBackend{Backend: compute.Default(), uploadDtype: true}
	q8 := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ8_0},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})
	if serveArtifactResidentQ4K(backend, q8) {
		t.Fatal("all-Q8_0 artifact must not select resident Q4_K even when FAK_Q4K=1")
	}
	q4k := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ4_K},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})
	if !serveArtifactResidentQ4K(backend, q4k) {
		t.Fatal("Q4_K artifact should preserve resident Q4_K when backend and environment enable it")
	}
	t.Setenv("FAK_Q4K", "0")
	if serveArtifactResidentQ4K(backend, q4k) {
		t.Fatal("FAK_Q4K=0 must retain the Q8 staging rollback")
	}
}

func TestServeQuantProvenanceDistinguishesArtifactResidentAndSession(t *testing.T) {
	q8 := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ8_0},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})
	got := serveQuantProvenance(q8, false)
	for _, want := range []string{"artifact_quant=Q8_0", "artifact_inventory=mixed(F32+Q8_0)", "resident_quant=Q8_0", "session_quant=Q8_0"} {
		if !strings.Contains(got.Text, want) {
			t.Fatalf("Q8 provenance %q missing %q", got.Text, want)
		}
	}
	if strings.Contains(got.Text, "resident_quant=Q4_K") || strings.Contains(got.Text, "session_quant=Q4_K") {
		t.Fatalf("Q8 artifact emitted Q4_K provenance: %q", got.Text)
	}
}
