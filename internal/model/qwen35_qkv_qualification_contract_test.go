package model

import (
	"os"
	"strings"
	"testing"
)

// Binding/source regression only. CUDA numerical/full-forward witnesses remain
// separate; this test does not simulate or claim their execution.
func TestQwen35QKVQualificationBinding(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	off := read("qwen35_qkv_qualification_off.go")
	if !strings.HasPrefix(off, "//go:build !cuda || !cgo || !fak_cuda_q4k_qkv_qualification || !fak_cuda_v41_index_select") ||
		!strings.Contains(off, "false, nil") || strings.Contains(off, "matWeightHAL") || strings.Contains(off, "s.Backend") {
		t.Fatal("ordinary build must decline without staging/probing weights/backend")
	}
	on := read("qwen35_qkv_qualification_cuda.go")
	if !strings.HasPrefix(on, "//go:build cuda && cgo && fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select") {
		t.Fatal("qualification requires both explicit candidate and existing checked-stream tags")
	}
	for _, text := range []string{
		"s.Backend.(qwen35QKVQualificationBackend)", "s.Backend.(qwen35QueryGateSplitBackend)",
		`s.matWeightHAL(layerName(layer, "self_attn.q_proj.weight"))`,
		`s.matWeightHAL(layerName(layer, "self_attn.k_proj.weight"))`,
		`s.matWeightHAL(layerName(layer, "self_attn.v_proj.weight")), x)`,
	} {
		if !strings.Contains(on, text) {
			t.Fatalf("candidate operand binding lost: %s", text)
		}
	}
	body := read("qwen35_hal.go")
	start := strings.Index(body, "func (s *Session) qwen35FullAttentionHAL(")
	if start < 0 {
		t.Fatal("missing full-attention operation")
	}
	end := strings.Index(body[start:], "\nfunc ")
	if end < 0 {
		t.Fatal("missing full-attention operation")
	}
	body = body[start : start+end]
	for _, text := range []string{
		"qProjection, kRaw, v, sharedQKV, qkvErr := s.qwen35QKVQualification(layer, xn)",
		"if qkvErr != nil {\n\t\ts.failBackendForward(", "qg := qProjection",
		"q = qProjection", "if !sharedQKV {", "splitter.SplitQwen35QueryGate(qg, nH, hd)",
	} {
		if !strings.Contains(body, text) {
			t.Fatalf("candidate dispatch/failure binding lost: %s", text)
		}
	}
	if strings.Count(body, "if !sharedQKV {") != 3 {
		t.Fatal("existing Q/QG and K/V matmuls must all be gated on pre-enqueue decline")
	}
	if strings.Index(body, "if qkvErr != nil") > strings.Index(body, "be.MatMul(") {
		t.Fatal("selected error must stop before fallback GEMV")
	}
}
