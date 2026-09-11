//go:build darwin && arm64 && cgo

package compute

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestMetalAttentionFlashContract verifies that attention.metal and metal_shim.m
// implement threadgroup-tiled FlashAttention with online softmax and SIMD shuffle
// reductions to eliminate scalar thread execution and register spilling (#12521).
func TestMetalAttentionFlashContract(t *testing.T) {
	attnCandidates := []string{
		filepath.Join("shaders", "attention.metal"),
		filepath.Join("internal", "compute", "shaders", "attention.metal"),
		filepath.Join("..", "..", "internal", "compute", "shaders", "attention.metal"),
	}
	var attnBytes []byte
	var err error
	for _, p := range attnCandidates {
		attnBytes, err = os.ReadFile(p)
		if err == nil && len(attnBytes) > 0 {
			break
		}
	}
	if len(attnBytes) == 0 {
		t.Fatalf("could not read attention.metal from candidate paths: %v", attnCandidates)
	}
	attnSrc := string(attnBytes)

	requiredTokens := []string{
		"kernel void attention_f32",
		"threadgroup_position_in_grid",
		"threads_per_threadgroup",
		"threadgroup float qs[256]",
		"threadgroup float tg_sums",
		"simd_sum",
		"threadgroup_barrier",
		"mnew = max(m, score)",
		"float p = exp(score - mnew)",
		"flash_attention_tiled_f32",
		"acc[8]",
	}
	for _, tok := range requiredTokens {
		if !strings.Contains(attnSrc, tok) {
			t.Errorf("attention.metal missing expected token %q", tok)
		}
	}
}

// TestMetalAttentionZeroCopyContract verifies that metal_shim.m wires
// newBufferWithBytesNoCopy and shared storage mode for zero-copy UMA buffer aliasing.
func TestMetalAttentionZeroCopyContract(t *testing.T) {
	shimCandidates := []string{
		"metal_shim.m",
		filepath.Join("internal", "compute", "metal_shim.m"),
		filepath.Join("..", "..", "internal", "compute", "metal_shim.m"),
	}
	var shimBytes []byte
	var err error
	for _, p := range shimCandidates {
		shimBytes, err = os.ReadFile(p)
		if err == nil && len(shimBytes) > 0 {
			break
		}
	}
	if len(shimBytes) == 0 {
		t.Fatalf("could not read metal_shim.m from candidate paths: %v", shimCandidates)
	}
	shimSrc := string(shimBytes)

	requiredShimTokens := []string{
		"fmetal_malloc_zerocopy",
		"newBufferWithBytesNoCopy",
		"MTLResourceStorageModeShared",
		"fmetal_buffer_from_host_zerocopy",
		"fmetal_buffer_is_zerocopy",
		"fmetal_command_encode_attention_f32",
		"[d contents] == host",
		"[s contents] == host",
	}
	for _, tok := range requiredShimTokens {
		if !strings.Contains(shimSrc, tok) {
			t.Errorf("metal_shim.m missing required zero-copy / pipeline token %q", tok)
		}
	}
}

// TestMetalAttentionPipelineBarriers verifies that fmetal_command_encode_attention_f32
// encodes into the caller-owned command buffer without synchronous waitUntilCompleted stalls.
func TestMetalAttentionPipelineBarriers(t *testing.T) {
	shimCandidates := []string{
		"metal_shim.m",
		filepath.Join("internal", "compute", "metal_shim.m"),
		filepath.Join("..", "..", "internal", "compute", "metal_shim.m"),
	}
	var shimBytes []byte
	var err error
	for _, p := range shimCandidates {
		shimBytes, err = os.ReadFile(p)
		if err == nil && len(shimBytes) > 0 {
			break
		}
	}
	if len(shimBytes) == 0 {
		t.Fatalf("could not read metal_shim.m: %v", err)
	}
	shimSrc := string(shimBytes)

	encodeIdx := strings.Index(shimSrc, "fmetal_command_encode_attention_f32")
	if encodeIdx == -1 {
		t.Fatal("metal_shim.m missing fmetal_command_encode_attention_f32")
	}
	encodeBlock := shimSrc[encodeIdx:]
	endBlock := strings.Index(encodeBlock, "\n}\n")
	if endBlock != -1 {
		encodeBlock = encodeBlock[:endBlock]
	}

	if strings.Contains(encodeBlock, "waitUntilCompleted") {
		t.Errorf("fmetal_command_encode_attention_f32 must not call waitUntilCompleted (must pipeline without synchronous barrier stalls)")
	}
	if !strings.Contains(encodeBlock, "dispatchThreadgroups") {
		t.Errorf("fmetal_command_encode_attention_f32 must use dispatchThreadgroups for 128/256 threadgroup grid")
	}
}

// TestMetalAttentionExecution verifies numerical correctness of threadgroup-tiled
// FlashAttention against cpuref reference when Metal device is available. Guarded by
// the same fail-loud contract as metalOrSkip: with FAK_METAL_REQUIRE_DEVICE=1 an
// unregistered device is a hard failure naming the availability verdict, never a silent
// downgrade to CPU that the cosine parity gate could not catch.
func TestMetalAttentionExecution(t *testing.T) {
	mb := Pick("metal")
	be, ok := mb.(*metalBackend)
	if fatal, skip := metalGuardVerdict(ok, mb.Tier()); fatal != "" || skip != "" {
		if fatal != "" {
			t.Fatal(fatal)
		}
		t.Log(skip)
		return
	}

	ref := Default() // cpu-ref
	const (
		nH  = 8
		nKV = 2
		hd  = 64
	)
	grp := nH / nKV
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	var seed lcg = 0xabcdef1
	qData := mtlRscale(&seed, nH*hd, 1.0)
	qRef := NewF32(ref, []int{nH * hd}, qData)
	qMt := be.Upload(qRef, F32)

	for _, nPos := range []int{16, 64, 256, 1024} {
		t.Run("nPos_"+strconv.Itoa(nPos), func(t *testing.T) {
			kvRef := ref.NewKV(KVConfig{NumLayers: 1, NumKVHeads: nKV, HeadDim: hd, RopeTheta: 10000})
			kvMt := be.NewKV(KVConfig{NumLayers: 1, NumKVHeads: nKV, HeadDim: hd, RopeTheta: 10000})

			for p := 0; p < nPos; p++ {
				kData := mtlRscale(&seed, nKV*hd, 1.0)
				vData := mtlRscale(&seed, nKV*hd, 1.0)
				kRef := NewF32(ref, []int{nKV * hd}, kData)
				vRef := NewF32(ref, []int{nKV * hd}, vData)
				kMt := be.Upload(kRef, F32)
				vMt := be.Upload(vRef, F32)
				kvRef.AppendKV(0, kRef, kRef, vRef, p)
				kvMt.AppendKV(0, kMt, kMt, vMt, p)
			}

			outRef := ref.Read(ref.Attention(qRef, kvRef, 0, true, grp, scale))
			outMt := be.Read(be.Attention(qMt, kvMt, 0, true, grp, scale))

			if len(outRef) != len(outMt) {
				t.Fatalf("length mismatch: ref=%d metal=%d", len(outRef), len(outMt))
			}
			c := cosine(outRef, outMt)
			if c < 0.9999 {
				t.Fatalf("attention cosine %.6f < 0.9999 threshold vs cpuref", c)
			}
			t.Logf("nPos=%d cosine=%.8f maxAbs=%.2e", nPos, c, mtlMaxAbsDelta(outRef, outMt))
		})
	}
}
