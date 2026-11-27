package metalgemm

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestRoutedMoeDispatchAndScratchContract(t *testing.T) {
	t.Run("dual dispatch and exact scratch byte sizing", func(t *testing.T) {
		for _, tokens := range []int{1, 4, RoutedMoeMaxMVTokens} {
			cfg := RoutedMoeConfig{
				Tokens: tokens, TopK: 4, Slots: 1, Experts: 8, N: 64, K: 64, WeightDType: RoutedMoeDTypeQ4_0,
			}
			d, err := SelectRoutedMoeDispatch(cfg)
			if err != nil {
				t.Fatalf("tokens=%d SelectRoutedMoeDispatch failed: %v", tokens, err)
			}
			if d.Mode != RoutedMoeDispatchVector {
				t.Fatalf("tokens=%d mode=%q, want %q", tokens, d.Mode, RoutedMoeDispatchVector)
			}
			if d.Scratch != (RoutedMoeScratch{}) {
				t.Fatalf("tokens=%d scratch=%+v, want zero scratch", tokens, d.Scratch)
			}
			sc, err := RoutedMoeScratchSizes(cfg)
			if err != nil || sc.TotalBytes != 0 {
				t.Fatalf("tokens=%d RoutedMoeScratchSizes=(%+v, %v), want zero", tokens, sc, err)
			}
		}

		for _, tc := range []struct {
			name    string
			tokens  int
			topk    int
			experts int
			n, k    int
			dtype   RoutedMoeDType
		}{
			{name: "tokens=9 matrix boundary", tokens: 9, topk: 2, experts: 8, n: 64, k: 64, dtype: RoutedMoeDTypeQ4_K},
			{name: "tokens=33 multi-tile", tokens: 33, topk: 4, experts: 16, n: 65, k: 256, dtype: RoutedMoeDTypeQ6_K},
			{name: "tokens<=8 ragged N falls back to matrix", tokens: 4, topk: 2, experts: 5, n: 17, k: 64, dtype: RoutedMoeDTypeQ4_0},
			{name: "tokens<=8 short K<64 for Q4_0 falls back to matrix", tokens: 2, topk: 2, experts: 4, n: 64, k: 32, dtype: RoutedMoeDTypeQ4_0},
		} {
			t.Run(tc.name, func(t *testing.T) {
				cfg := RoutedMoeConfig{
					Tokens: tc.tokens, TopK: tc.topk, Slots: 1, Experts: tc.experts, N: tc.n, K: tc.k, WeightDType: tc.dtype,
				}
				d, err := SelectRoutedMoeDispatch(cfg)
				if err != nil {
					t.Fatalf("SelectRoutedMoeDispatch failed: %v", err)
				}
				if d.Mode != RoutedMoeDispatchMatrix {
					t.Fatalf("mode=%q, want %q", d.Mode, RoutedMoeDispatchMatrix)
				}
				pairs := uint64(tc.tokens * tc.topk)
				wantCounts := uint64(tc.experts+1) * 129 * 4
				wantRoutes := pairs * 4
				ceilPairs := (pairs + 31) / 32
				wantTaskCap := pairs
				if ceilPairs+uint64(tc.experts) < wantTaskCap {
					wantTaskCap = ceilPairs + uint64(tc.experts)
				}
				wantTasks := (wantTaskCap + 1) * 16
				wantTotal := wantCounts + wantRoutes + wantTasks
				if d.Scratch.CountsBytes != wantCounts ||
					d.Scratch.RoutesBytes != wantRoutes ||
					d.Scratch.TasksBytes != wantTasks ||
					d.Scratch.TotalBytes != wantTotal ||
					uint64(d.Scratch.TaskCap) != wantTaskCap {
					t.Fatalf("scratch=%+v, want counts=%d routes=%d tasks=%d total=%d cap=%d",
						d.Scratch, wantCounts, wantRoutes, wantTasks, wantTotal, wantTaskCap)
				}
			})
		}
	})

	t.Run("per-dtype threadgroup geometry table", func(t *testing.T) {
		want := []struct {
			dtype RoutedMoeDType
			x, y  uint32
			div   uint32
		}{
			{RoutedMoeDTypeQ4_0, 8, 8, 8},
			{RoutedMoeDTypeQ4_1, 8, 8, 8},
			{RoutedMoeDTypeQ5_0, 8, 8, 8},
			{RoutedMoeDTypeQ5_1, 8, 8, 8},
			{RoutedMoeDTypeQ8_0, 8, 8, 8},
			{RoutedMoeDTypeQ2_K, 2, 32, 8},
			{RoutedMoeDTypeQ3_K, 2, 32, 4},
			{RoutedMoeDTypeQ4_K, 4, 8, 4},
			{RoutedMoeDTypeQ5_K, 2, 32, 4},
			{RoutedMoeDTypeQ6_K, 2, 32, 2},
			{RoutedMoeDTypeF16, 32, 1, 1},
			{RoutedMoeDTypeF32, 32, 1, 1},
		}
		for _, tc := range want {
			g, err := RoutedMoeVectorGeometry(tc.dtype)
			if err != nil {
				t.Fatalf("dtype %s failed: %v", tc.dtype, err)
			}
			if g.ThreadgroupX != tc.x || g.ThreadgroupY != tc.y || g.NDiv != tc.div || g.ThreadsPerTG != tc.x*tc.y {
				t.Fatalf("dtype %s geometry=%+v, want (%d,%d) div %d", tc.dtype, g, tc.x, tc.y, tc.div)
			}
		}
		if g, err := RoutedMoeVectorGeometry("q4_k"); err != nil || g.DType != RoutedMoeDTypeQ4_K {
			t.Fatalf("lowercase normalization failed: (%+v, %v)", g, err)
		}
	})

	t.Run("closed refusals for dtype, slots, K alignment, and overflow", func(t *testing.T) {
		base := RoutedMoeConfig{
			Tokens: 16, TopK: 4, Slots: 4, Experts: 8, N: 64, K: 64, WeightDType: RoutedMoeDTypeQ4_K,
		}
		if err := base.Validate(); err != nil {
			t.Fatalf("valid base failed: %v", err)
		}

		for _, badDType := range []RoutedMoeDType{RoutedMoeDTypeQ8_1, RoutedMoeDTypeQ8_K, RoutedMoeDTypeBF16, "", "IQ4_XS", "Q4_K_M"} {
			cfg := base
			cfg.WeightDType = badDType
			if err := cfg.Validate(); !errors.Is(err, ErrRoutedMoeUnsupportedDType) {
				t.Fatalf("dtype %q err=%v, want %v", badDType, err, ErrRoutedMoeUnsupportedDType)
			}
			if _, err := SelectRoutedMoeDispatch(cfg); !errors.Is(err, ErrRoutedMoeUnsupportedDType) {
				t.Fatalf("SelectRoutedMoeDispatch dtype %q err=%v, want %v", badDType, err, ErrRoutedMoeUnsupportedDType)
			}
		}

		for _, badSlots := range []int{0, -1, 2, 3, 5} {
			cfg := base
			cfg.Slots = badSlots
			if err := cfg.Validate(); !errors.Is(err, ErrRoutedMoeInvalidConfig) {
				t.Fatalf("slots=%d err=%v, want %v", badSlots, err, ErrRoutedMoeInvalidConfig)
			}
		}

		for _, badK := range []int{0, -32, 16, 31, 33, 96 - 1} {
			cfg := base
			cfg.K = badK
			if err := cfg.Validate(); !errors.Is(err, ErrRoutedMoeInvalidConfig) {
				t.Fatalf("K=%d err=%v, want %v", badK, err, ErrRoutedMoeInvalidConfig)
			}
		}

		for _, badDim := range []RoutedMoeConfig{
			{Tokens: 0, TopK: 2, Slots: 1, Experts: 4, N: 32, K: 32, WeightDType: RoutedMoeDTypeF16},
			{Tokens: 4, TopK: 0, Slots: 1, Experts: 4, N: 32, K: 32, WeightDType: RoutedMoeDTypeF16},
			{Tokens: 4, TopK: 2, Slots: 1, Experts: 0, N: 32, K: 32, WeightDType: RoutedMoeDTypeF16},
			{Tokens: 4, TopK: 2, Slots: 1, Experts: 4, N: 0, K: 32, WeightDType: RoutedMoeDTypeF16},
		} {
			if err := badDim.Validate(); !errors.Is(err, ErrRoutedMoeInvalidConfig) {
				t.Fatalf("badDim=%+v err=%v, want %v", badDim, err, ErrRoutedMoeInvalidConfig)
			}
		}

		overflowCases := []RoutedMoeConfig{
			{Tokens: math.MaxInt, TopK: 2, Slots: 1, Experts: 4, N: 32, K: 32, WeightDType: RoutedMoeDTypeF32},
			{Tokens: 1 << 30, TopK: 8, Slots: 1, Experts: 4, N: 32, K: 32, WeightDType: RoutedMoeDTypeQ4_0},
			{Tokens: 16, TopK: 2, Slots: 1, Experts: math.MaxInt, N: 32, K: 32, WeightDType: RoutedMoeDTypeQ4_0},
			{Tokens: 16, TopK: 2, Slots: 1, Experts: int(math.MaxUint32), N: 32, K: 32, WeightDType: RoutedMoeDTypeQ4_0},
			{Tokens: 16, TopK: 2, Slots: 1, Experts: 1 << 20, N: 1 << 25, K: 1 << 25, WeightDType: RoutedMoeDTypeQ4_K},
			{Tokens: 1 << 29, TopK: 4, Slots: 1, Experts: 4, N: 1 << 31, K: 32, WeightDType: RoutedMoeDTypeF32},
			{Tokens: 1 << 31, TopK: 1, Slots: 1, Experts: 4, N: 32, K: 1 << 31, WeightDType: RoutedMoeDTypeF32},
		}
		for _, cfg := range overflowCases {
			if err := cfg.Validate(); !errors.Is(err, ErrRoutedMoeOverflow) {
				t.Fatalf("overflow cfg=%+v err=%v, want %v", cfg, err, ErrRoutedMoeOverflow)
			}
			if _, err := RoutedMoeScratchSizes(cfg); !errors.Is(err, ErrRoutedMoeOverflow) {
				t.Fatalf("RoutedMoeScratchSizes overflow cfg=%+v err=%v, want %v", cfg, err, ErrRoutedMoeOverflow)
			}
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestRoutedMoeAtomicFreeCountsCompactAndTiles(t *testing.T) {
	t.Run("determinism, counts invariants, order preservation, 32-row tiling, and sentinel clamp", func(t *testing.T) {
		cfg := RoutedMoeConfig{
			Tokens: 160, TopK: 2, Slots: 1, Experts: 6, N: 64, K: 64, WeightDType: RoutedMoeDTypeQ4_K,
		}
		pairs := cfg.Tokens * cfg.TopK
		ids := make([]int32, pairs)
		// Craft skewed counts across experts + invalid IDs so we hit 0, 1, 32, 33, 65-row buckets
		// and negative/OOB sentinel IDs in non-trivial interleaved positions.
		for p := 0; p < pairs; p++ {
			switch {
			case p%11 == 0:
				ids[p] = -1
			case p%13 == 0:
				ids[p] = int32(cfg.Experts)
			case p%17 == 0:
				ids[p] = int32(cfg.Experts + 99)
			case p%19 == 0:
				ids[p] = math.MinInt32
			case p%23 == 0:
				ids[p] = math.MaxInt32
			case p < 180:
				ids[p] = 0 // > 64 hits -> multiple 32-row tiles + partial tail
			case p < 245:
				ids[p] = 2 // leaving expert 1 empty!
			case p == 245:
				ids[p] = 3 // single-hit expert
			default:
				ids[p] = 4 // leaving expert 5 empty
			}
		}

		planA, err := BuildRoutedMoePlan(cfg, ids)
		if err != nil {
			t.Fatalf("BuildRoutedMoePlan failed: %v", err)
		}
		planB, err := BuildRoutedMoePlan(cfg, ids)
		if err != nil {
			t.Fatalf("second BuildRoutedMoePlan failed: %v", err)
		}

		// Determinism across repeated runs.
		if len(planA.Counts) != len(planB.Counts) || len(planA.Routes) != len(planB.Routes) || len(planA.Tasks) != len(planB.Tasks) {
			t.Fatalf("non-deterministic plan lengths")
		}
		for i := range planA.Counts {
			if planA.Counts[i] != planB.Counts[i] {
				t.Fatalf("Counts[%d] differed: %d vs %d", i, planA.Counts[i], planB.Counts[i])
			}
		}
		for i := range planA.Routes {
			if planA.Routes[i] != planB.Routes[i] {
				t.Fatalf("Routes[%d] differed: %d vs %d", i, planA.Routes[i], planB.Routes[i])
			}
		}
		for i := range planA.Tasks {
			if planA.Tasks[i] != planB.Tasks[i] {
				t.Fatalf("Tasks[%d] differed: %+v vs %+v", i, planA.Tasks[i], planB.Tasks[i])
			}
		}

		// Verify Counts[(experts+1)*129] and bucket totals.
		if len(planA.Counts) != (cfg.Experts+1)*RoutedMoeCountsStride {
			t.Fatalf("len(Counts)=%d, want %d", len(planA.Counts), (cfg.Experts+1)*RoutedMoeCountsStride)
		}
		var sumTotals uint32
		for e := 0; e <= cfg.Experts; e++ {
			base := e * RoutedMoeCountsStride
			var laneSum uint32
			for lane := 0; lane < RoutedMoeRouteThreads; lane++ {
				laneSum += planA.Counts[base+lane]
			}
			total := planA.Counts[base+RoutedMoeRouteThreads]
			if laneSum != total || total != planA.ExpertCounts[e] {
				t.Fatalf("bucket %d: laneSum=%d countsTotal=%d expertCount=%d", e, laneSum, total, planA.ExpertCounts[e])
			}
			sumTotals += total
		}
		if sumTotals != uint32(pairs) {
			t.Fatalf("sumTotals=%d, want %d", sumTotals, pairs)
		}
		if planA.ExpertCounts[1] != 0 || planA.ExpertCounts[5] != 0 {
			t.Fatalf("expected empty experts 1 and 5, got counts=%v", planA.ExpertCounts)
		}
		if planA.ExpertCounts[cfg.Experts] == 0 {
			t.Fatal("expected non-zero sentinel bucket for invalid IDs")
		}

		// Verify order preservation inside every bucket and global route bijection.
		seen := make([]bool, pairs)
		for e := 0; e <= cfg.Experts; e++ {
			start := int(planA.ExpertOffsets[e])
			cnt := int(planA.ExpertCounts[e])
			var wantRoutes []uint32
			for p := 0; p < pairs; p++ {
				b := cfg.Experts
				if ids[p] >= 0 && int(ids[p]) < cfg.Experts {
					b = int(ids[p])
				}
				if b == e {
					wantRoutes = append(wantRoutes, uint32(p))
				}
			}
			if cnt != len(wantRoutes) {
				t.Fatalf("bucket %d count=%d, want %d", e, cnt, len(wantRoutes))
			}
			gotSlice := planA.Routes[start : start+cnt]
			for i, r := range gotSlice {
				if r != wantRoutes[i] {
					t.Fatalf("bucket %d pos %d: route=%d, want %d (order not preserved)", e, i, r, wantRoutes[i])
				}
				if i > 0 && gotSlice[i-1] >= r {
					t.Fatalf("bucket %d not strictly ascending at %d: %d >= %d", e, i, gotSlice[i-1], r)
				}
				if seen[r] {
					t.Fatalf("duplicate route %d", r)
				}
				seen[r] = true
			}
		}
		for p, ok := range seen {
			if !ok {
				t.Fatalf("pair %d missing from compacted Routes", p)
			}
		}

		// Verify 32-row tile tasks, sentinel task emission, and task capacity bound.
		if uint64(len(planA.Tasks)) > uint64(planA.Dispatch.Scratch.TaskCap) {
			t.Fatalf("len(Tasks)=%d exceeds TaskCap=%d", len(planA.Tasks), planA.Dispatch.Scratch.TaskCap)
		}
		if planA.TotalTasks != uint32(len(planA.Tasks)) {
			t.Fatalf("TotalTasks=%d, len(Tasks)=%d", planA.TotalTasks, len(planA.Tasks))
		}
		var cursor uint32
		hasSentinelTask := false
		for i, task := range planA.Tasks {
			if task.Rows == 0 || task.Rows > RoutedMoeTaskTileRows {
				t.Fatalf("task[%d] invalid rows=%d", i, task.Rows)
			}
			if task.RouteStart != cursor {
				t.Fatalf("task[%d] RouteStart=%d, want contiguous %d", i, task.RouteStart, cursor)
			}
			if task.Expert == planA.SentinelID {
				hasSentinelTask = true
			}
			for r := uint32(0); r < task.Rows; r++ {
				route := planA.Routes[task.RouteStart+r]
				b := clampExpertBucket(ids[route], cfg.Experts)
				if uint32(b) != task.Expert {
					t.Fatalf("task[%d] row %d route=%d mapped to bucket %d, task.Expert=%d", i, r, route, b, task.Expert)
				}
			}
			cursor += task.Rows
		}
		if cursor != uint32(pairs) {
			t.Fatalf("task row sum=%d, want %d", cursor, pairs)
		}
		if !hasSentinelTask {
			t.Fatal("expected sentinel task for invalid-id rows")
		}
	})

	t.Run("exact 32/33/64 boundaries and large >u16 token count preserve task bound", func(t *testing.T) {
		for _, count := range []int{31, 32, 33, 64, 65} {
			cfg := RoutedMoeConfig{
				Tokens: count, TopK: 1, Slots: 1, Experts: 2, N: 32, K: 32, WeightDType: RoutedMoeDTypeF32,
			}
			ids := make([]int32, count) // all route to expert 0
			plan, err := BuildRoutedMoePlan(cfg, ids)
			if err != nil {
				t.Fatalf("count=%d BuildRoutedMoePlan failed: %v", count, err)
			}
			wantTiles := (count + 31) / 32
			if len(plan.Tasks) != wantTiles {
				t.Fatalf("count=%d tasks=%d, want %d", count, len(plan.Tasks), wantTiles)
			}
		}

		// Large token count > u16::MAX (65537) + all buckets non-empty including sentinel.
		cfg := RoutedMoeConfig{
			Tokens: 65537, TopK: 1, Slots: 1, Experts: 4, N: 32, K: 32, WeightDType: RoutedMoeDTypeQ4_0,
		}
		ids := make([]int32, cfg.Tokens)
		for i := range ids {
			switch i % 5 {
			case 1:
				ids[i] = -1
			case 0:
				ids[i] = 0
			default:
				ids[i] = int32((i % 5) - 1)
			}
		}
		plan, err := BuildRoutedMoePlan(cfg, ids)
		if err != nil {
			t.Fatalf("large token plan failed: %v", err)
		}
		if uint64(len(plan.Tasks)) > uint64(plan.Dispatch.Scratch.TaskCap) {
			t.Fatalf("large token len(Tasks)=%d exceeded TaskCap=%d", len(plan.Tasks), plan.Dispatch.Scratch.TaskCap)
		}
		lastRoute := plan.Routes[len(plan.Routes)-1]
		if lastRoute <= math.MaxUint16 {
			t.Fatalf("expected >u16 route index in sentinel bucket, got %d", lastRoute)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestRoutedMoeHostOracleParityAndInvalidZeroRows(t *testing.T) {
	bruteForceRef := func(cfg RoutedMoeConfig, inputF32, weightsF32 []float32, ids []int32, topkWeights []float32) ([]float32, []float32) {
		pairs := cfg.Tokens * cfg.TopK
		perRoute := make([]float32, pairs*cfg.N)
		for tok := 0; tok < cfg.Tokens; tok++ {
			for kIdx := 0; kIdx < cfg.TopK; kIdx++ {
				p := tok*cfg.TopK + kIdx
				id := ids[p]
				outBase := p * cfg.N
				if id < 0 || int(id) >= cfg.Experts {
					continue // zero row
				}
				slot := 0
				if cfg.Slots == cfg.TopK {
					slot = kIdx
				}
				xBase := (tok*cfg.Slots + slot) * cfg.K
				wExpertBase := int(id) * cfg.N * cfg.K
				for n := 0; n < cfg.N; n++ {
					var acc float64
					for k := 0; k < cfg.K; k++ {
						acc += float64(inputF32[xBase+k]) * float64(weightsF32[wExpertBase+n*cfg.K+k])
					}
					perRoute[outBase+n] = float32(acc)
				}
			}
		}
		if len(topkWeights) == 0 {
			return perRoute, nil
		}
		reduced := make([]float32, cfg.Tokens*cfg.N)
		for tok := 0; tok < cfg.Tokens; tok++ {
			for n := 0; n < cfg.N; n++ {
				var acc float64
				for kIdx := 0; kIdx < cfg.TopK; kIdx++ {
					p := tok*cfg.TopK + kIdx
					acc += float64(topkWeights[p]) * float64(perRoute[p*cfg.N+n])
				}
				reduced[tok*cfg.N+n] = float32(acc)
			}
		}
		return perRoute, reduced
	}

	cases := []struct {
		name      string
		cfg       RoutedMoeConfig
		wantMode  RoutedMoeDispatchMode
		allBadTok int // token index whose slots are all invalid IDs
	}{
		{
			name: "vector path slots=1",
			cfg: RoutedMoeConfig{
				Tokens: 4, TopK: 3, Slots: 1, Experts: 5, N: 64, K: 64, WeightDType: RoutedMoeDTypeQ4_0,
			},
			wantMode:  RoutedMoeDispatchVector,
			allBadTok: 2,
		},
		{
			name: "vector path slots=topk",
			cfg: RoutedMoeConfig{
				Tokens: 8, TopK: 3, Slots: 3, Experts: 5, N: 32, K: 64, WeightDType: RoutedMoeDTypeQ4_K,
			},
			wantMode:  RoutedMoeDispatchVector,
			allBadTok: 5,
		},
		{
			name: "matrix path slots=1 ragged N",
			cfg: RoutedMoeConfig{
				Tokens: 13, TopK: 3, Slots: 1, Experts: 4, N: 37, K: 64, WeightDType: RoutedMoeDTypeQ6_K,
			},
			wantMode:  RoutedMoeDispatchMatrix,
			allBadTok: 7,
		},
		{
			name: "matrix path slots=topk multi-tile",
			cfg: RoutedMoeConfig{
				Tokens: 37, TopK: 4, Slots: 4, Experts: 3, N: 65, K: 32, WeightDType: RoutedMoeDTypeF16,
			},
			wantMode:  RoutedMoeDispatchMatrix,
			allBadTok: 19,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			input := make([]float32, cfg.Tokens*cfg.Slots*cfg.K)
			for i := range input {
				input[i] = float32((i%19)-9)*0.125 + 0.0625
			}
			weights := make([]float32, cfg.Experts*cfg.N*cfg.K)
			for i := range weights {
				weights[i] = float32((i%23)-11)*0.0625 + 0.03125
			}
			pairs := cfg.Tokens * cfg.TopK
			ids := make([]int32, pairs)
			topkWeights := make([]float32, pairs)
			for tok := 0; tok < cfg.Tokens; tok++ {
				for s := 0; s < cfg.TopK; s++ {
					p := tok*cfg.TopK + s
					topkWeights[p] = float32(s+1) * 0.25
					switch {
					case tok == tc.allBadTok:
						if s%2 == 0 {
							ids[p] = -1
						} else {
							ids[p] = int32(cfg.Experts + s)
						}
					case p%7 == 0:
						ids[p] = -3
					case p%11 == 0:
						ids[p] = int32(cfg.Experts)
					default:
						ids[p] = int32((tok*3 + s*2) % cfg.Experts)
					}
				}
			}

			out, err := EvalRoutedMoeHostOracle(cfg, input, weights, ids, topkWeights)
			if err != nil {
				t.Fatalf("EvalRoutedMoeHostOracle failed: %v", err)
			}
			if out.Plan.Dispatch.Mode != tc.wantMode {
				t.Fatalf("dispatch mode=%q, want %q", out.Plan.Dispatch.Mode, tc.wantMode)
			}

			wantRoute, wantReduced := bruteForceRef(cfg, input, weights, ids, topkWeights)
			if len(out.PerRoute) != len(wantRoute) {
				t.Fatalf("PerRoute len=%d, want %d", len(out.PerRoute), len(wantRoute))
			}
			for i := range wantRoute {
				tok := (i / cfg.N) / cfg.TopK
				slot := (i / cfg.N) % cfg.TopK
				id := ids[tok*cfg.TopK+slot]
				if id < 0 || int(id) >= cfg.Experts {
					if math.Float32bits(out.PerRoute[i]) != 0 {
						t.Fatalf("invalid id=%d at tok=%d slot=%d produced non-zero bit pattern %08x (%g)",
							id, tok, slot, math.Float32bits(out.PerRoute[i]), out.PerRoute[i])
					}
					continue
				}
				if diff := math.Abs(float64(out.PerRoute[i] - wantRoute[i])); diff > 1e-6 {
					t.Fatalf("PerRoute[%d] = %g, want %g (diff %g)", i, out.PerRoute[i], wantRoute[i], diff)
				}
			}

			if len(out.Reduced) != len(wantReduced) {
				t.Fatalf("Reduced len=%d, want %d", len(out.Reduced), len(wantReduced))
			}
			for tok := 0; tok < cfg.Tokens; tok++ {
				for n := 0; n < cfg.N; n++ {
					idx := tok*cfg.N + n
					if tok == tc.allBadTok && math.Float32bits(out.Reduced[idx]) != 0 {
						t.Fatalf("all-invalid token %d col %d reduced=%g, want exact 0", tok, n, out.Reduced[idx])
					}
					if diff := math.Abs(float64(out.Reduced[idx] - wantReduced[idx])); diff > 1e-6 {
						t.Fatalf("Reduced[%d] = %g, want %g", idx, out.Reduced[idx], wantReduced[idx])
					}
				}
			}

			// Optional topkWeights == nil omits reduction cleanly.
			noReduce, err := EvalRoutedMoeHostOracle(cfg, input, weights, ids, nil)
			if err != nil {
				t.Fatalf("EvalRoutedMoeHostOracle(nil weights) failed: %v", err)
			}
			if noReduce.Reduced != nil {
				t.Fatalf("expected nil Reduced when topkWeights is nil, got len=%d", len(noReduce.Reduced))
			}
		})
	}

	t.Run("buffer shape mismatches fail closed", func(t *testing.T) {
		cfg := RoutedMoeConfig{Tokens: 2, TopK: 2, Slots: 1, Experts: 3, N: 8, K: 32, WeightDType: RoutedMoeDTypeF32}
		in := make([]float32, 2*1*32)
		w := make([]float32, 3*8*32)
		ids := make([]int32, 4)

		if _, err := EvalRoutedMoeHostOracle(cfg, in[:len(in)-1], w, ids, nil); !errors.Is(err, ErrRoutedMoeBufferShape) {
			t.Fatalf("short input err=%v, want %v", err, ErrRoutedMoeBufferShape)
		}
		if _, err := EvalRoutedMoeHostOracle(cfg, in, w[:len(w)-1], ids, nil); !errors.Is(err, ErrRoutedMoeBufferShape) {
			t.Fatalf("short weights err=%v, want %v", err, ErrRoutedMoeBufferShape)
		}
		if _, err := EvalRoutedMoeHostOracle(cfg, in, w, ids[:3], nil); !errors.Is(err, ErrRoutedMoeBufferShape) {
			t.Fatalf("short ids err=%v, want %v", err, ErrRoutedMoeBufferShape)
		}
		if _, err := EvalRoutedMoeHostOracle(cfg, in, w, ids, []float32{1, 2}); !errors.Is(err, ErrRoutedMoeBufferShape) {
			t.Fatalf("short topkWeights err=%v, want %v", err, ErrRoutedMoeBufferShape)
		}
	})
}
