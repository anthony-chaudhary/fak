// fak-test:runtime fast est=1s lane=default
package compute

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// makeDeterministicQ4KRaw produces valid deterministic GGUF Q4_K bytes [outDim, inDim].
func makeDeterministicQ4KRaw(outDim, inDim int, seed uint64) []byte {
	nblk := (outDim * inDim) / q4kSuper
	raw := make([]byte, nblk*q4kSuperBlock)
	state := seed
	nextByte := func() byte {
		state = state*6364136223846793005 + 1442695040888963407
		return byte(state >> 56)
	}
	// Keep f16 d (~0.0156) and dmin (~0.0078) clean and finite: 0x2000 = 2^-7, 0x1c00 = 2^-8.
	for b := 0; b < nblk; b++ {
		blk := raw[b*q4kSuperBlock : (b+1)*q4kSuperBlock]
		binary.LittleEndian.PutUint16(blk[0:], 0x2000+uint16(b%7))
		binary.LittleEndian.PutUint16(blk[2:], 0x1c00+uint16((b*3)%5))
		for i := 4; i < q4kSuperBlock; i++ {
			blk[i] = nextByte()
		}
	}
	return raw
}

func TestCudaWeightPipelineCachePolicyAndSchedule(t *testing.T) {
	desc, err := BuildWeightPipelineDescriptor(Q4_K, 16, 1024, 256, 3)
	if err != nil {
		t.Fatalf("BuildWeightPipelineDescriptor(Q4_K): %v", err)
	}
	if desc.PipelineHintForRole(PipelineRoleWeight) != L2CacheEvictFirst {
		t.Fatalf("expected weight hint evict_first, got %s", desc.PipelineHintForRole(PipelineRoleWeight))
	}
	if desc.PipelineHintForRole(PipelineRoleActivation) == L2CacheEvictFirst {
		t.Fatalf("activation load must not be tagged evict_first")
	}
	if desc.PipelineHintForRole(PipelineRoleScale) == L2CacheEvictFirst {
		t.Fatalf("scale load must not be tagged evict_first")
	}

	// Verify wait_group depth contract: prologue emits prefetches before first compute, steady-state keeps stages-2 in flight, epilogue drains to 0.
	dmaCount := 0
	computeCount := 0
	for _, st := range desc.Schedule {
		if st.IssueWeightDMA {
			dmaCount++
			if st.WeightL2Hint != L2CacheEvictFirst {
				t.Fatalf("step weight DMA missing evict_first: %+v", st)
			}
		}
		if st.ActL2Hint == L2CacheEvictFirst {
			t.Fatalf("step tagged activation evict_first: %+v", st)
		}
		if st.ComputeSlot >= 0 {
			computeCount++
		}
	}
	if dmaCount != 4 || computeCount != 4 {
		t.Fatalf("expected 4 DMAs and 4 compute steps for 1024/256, got dma=%d compute=%d", dmaCount, computeCount)
	}

	// Refusal checks on bad cache policy or stage count < 2.
	badDesc := *desc
	badDesc.WeightL2Hint = L2CacheNormal
	if err := badDesc.Validate(); err == nil {
		t.Fatal("expected Validate error when weight hint is not evict_first")
	}
	badAct := *desc
	badAct.ActivationL2Hint = L2CacheEvictFirst
	if err := badAct.Validate(); err == nil {
		t.Fatal("expected Validate error when activation hint is evict_first")
	}
	if _, err := BuildWeightPipelineDescriptor(I4, 16, 64, 16, 1); err == nil {
		t.Fatal("expected error for single-stage (non-double-buffered) pipeline")
	}
}

func TestCudaWeightPipelineMatchesCPURefQ4KAndAWQ(t *testing.T) {
	be := Default()

	t.Run("Q4_K_multi_stage_cpuref_parity", func(t *testing.T) {
		const (
			outDim = 16
			inDim  = 512
			tileK  = 256
			stages = 2
			rows   = 3
		)
		raw := makeDeterministicQ4KRaw(outDim, inDim, 777)
		rawSnapshot := append([]byte(nil), raw...)

		desc, err := BuildWeightPipelineDescriptor(Q4_K, outDim, inDim, tileK, stages)
		if err != nil {
			t.Fatalf("BuildWeightPipelineDescriptor: %v", err)
		}
		rng := lcg(1234)
		xData := randVec(&rng, rows*inDim)

		got, err := RunWeightPipelineQ4K(desc, raw, xData, rows)
		if err != nil {
			t.Fatalf("RunWeightPipelineQ4K: %v", err)
		}
		if !bytes.Equal(raw, rawSnapshot) {
			t.Fatal("pipeline execution mutated original packed Q4_K bytes")
		}

		wTensor := NewQ4K(be, []int{outDim, inDim}, raw)
		xTensor := NewF32(be, []int{rows, inDim}, xData)
		want := be.Read(be.BatchedMatMul(wTensor, xTensor, rows))

		var maxDiff float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-4 {
			t.Fatalf("Q4_K pipeline vs cpuref maxDiff=%g (> 1e-4)", maxDiff)
		}
		if cos := cosine(got, want); cos < 0.99999 {
			t.Fatalf("Q4_K pipeline cosine=%f", cos)
		}
	})

	t.Run("AWQ_triple_buffer_cpuref_parity", func(t *testing.T) {
		const (
			outDim = 20
			inDim  = 96
			tileK  = 32
			stages = 3
			rows   = 4
		)
		packed, scales, _, _ := makeAWQTestFixture(outDim, inDim, 1, 555)
		packedSnapshot := append([]byte(nil), packed...)

		desc, err := BuildWeightPipelineDescriptor(I4, outDim, inDim, tileK, stages)
		if err != nil {
			t.Fatalf("BuildWeightPipelineDescriptor: %v", err)
		}
		rng := lcg(5678)
		xData := randVec(&rng, rows*inDim)

		got, err := RunWeightPipelineAWQ(desc, packed, scales, xData, rows)
		if err != nil {
			t.Fatalf("RunWeightPipelineAWQ: %v", err)
		}
		if !bytes.Equal(packed, packedSnapshot) {
			t.Fatal("pipeline execution mutated original packed AWQ bytes")
		}

		denseW := dequantAWQWithGIdxOracle(packed, scales, nil, outDim, inDim)
		wTensor := NewF32(be, []int{outDim, inDim}, denseW)
		xTensor := NewF32(be, []int{rows, inDim}, xData)
		want := be.Read(be.BatchedMatMul(wTensor, xTensor, rows))

		var maxDiff float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-4 {
			t.Fatalf("AWQ pipeline vs cpuref maxDiff=%g (> 1e-4)", maxDiff)
		}
		if cos := cosine(got, want); cos < 0.99999 {
			t.Fatalf("AWQ pipeline cosine=%f", cos)
		}
	})
}
