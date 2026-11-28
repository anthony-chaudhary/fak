package compute

import (
	"math"
	"testing"
)

// Closed in-memory oracle for Aleph-Alpha/Alpha-MoE W8A8 up+down+acc fusion (ADAPT).
// [SW-VERIFIED] structure and oracle only: no hardware, no tuning, no speed claim.
func TestAlphaMoeFused(t *testing.T) {
	t.Parallel()
	if cuhDefine(t, "alpha_moe_fused.cuh", "FAK_ALPHA_MOE_STAGE_UP") != 1 ||
		cuhDefine(t, "alpha_moe_fused.cuh", "FAK_ALPHA_MOE_STAGE_ACT") != 2 ||
		cuhDefine(t, "alpha_moe_fused.cuh", "FAK_ALPHA_MOE_STAGE_DOWN") != 3 ||
		cuhDefine(t, "alpha_moe_fused.cuh", "FAK_ALPHA_MOE_STAGE_ACC") != 4 ||
		cuhDefine(t, "alpha_moe_fused.cuh", "FAK_ALPHA_MOE_INTERMEDIATE_IN_SMEM") != 1 {
		t.Fatal("alpha_moe_fused.cuh stage constants drifted")
	}

	// 2 experts, hidden=3, intermediate=2; int8 weights with per-layer scales.
	x := []int8{2, -3, 4}
	actScale := float32(0.5)
	wUp := [2][2][3]int8{
		{{1, 2, -1}, {0, 1, 2}},
		{{-1, 1, 1}, {2, 0, -1}},
	}
	wDown := [2][3][2]int8{
		{{1, -1}, {2, 1}, {0, 2}},
		{{2, 1}, {-1, 2}, {1, 0}},
	}
	upScale := [2]float32{0.25, 0.5}
	downScale := [2]float32{0.5, 0.25}
	routerW := [2]float32{0.6, 0.4}

	relu := func(v float32) float32 {
		if v < 0 {
			return 0
		}
		return v
	}

	// Fused path: up -> act -> down -> weighted acc with zero global intermediate writes.
	globalIntermediateWrites := 0
	fusedOut := make([]float32, 3)
	for e := 0; e < 2; e++ {
		var smemInter [2]float32
		for i := 0; i < 2; i++ {
			var acc int32
			for k := 0; k < 3; k++ {
				acc += int32(x[k]) * int32(wUp[e][i][k])
			}
			smemInter[i] = relu(float32(acc) * actScale * upScale[e])
		}
		for h := 0; h < 3; h++ {
			var down float32
			for i := 0; i < 2; i++ {
				down += smemInter[i] * float32(wDown[e][h][i]) * downScale[e]
			}
			fusedOut[h] += routerW[e] * down
		}
	}

	// Two-kernel reference path: writes intermediate to global memory then reads it back.
	refOut := make([]float32, 3)
	for e := 0; e < 2; e++ {
		globalInter := make([]float32, 2)
		for i := 0; i < 2; i++ {
			var acc int32
			for k := 0; k < 3; k++ {
				acc += int32(x[k]) * int32(wUp[e][i][k])
			}
			globalInter[i] = relu(float32(acc) * actScale * upScale[e])
			globalIntermediateWrites++
		}
		for h := 0; h < 3; h++ {
			var down float32
			for i := 0; i < 2; i++ {
				down += globalInter[i] * float32(wDown[e][h][i]) * downScale[e]
			}
			refOut[h] += routerW[e] * down
		}
	}

	if globalIntermediateWrites == 0 {
		t.Fatal("reference path did not model global intermediate writes")
	}
	for h := range fusedOut {
		if math.Abs(float64(fusedOut[h]-refOut[h])) > 1e-6 {
			t.Fatalf("out[%d]: fused=%v ref=%v", h, fusedOut[h], refOut[h])
		}
	}
}
