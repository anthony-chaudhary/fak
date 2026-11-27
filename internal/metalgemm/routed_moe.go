// [SW-VERIFIED] Pure-Go closed host oracle and dispatch/scratch/routing contract
// for GPU-side routed quantized MoE projections. Compiles and executes without
// cgo or Apple Silicon hardware; carries no Metal execution or device speed claim
// on non-Apple hosts.
//
// Provenance: adapted from huggingface/candle PR #4068 at revision
// cfcbd7186d13271843ce7eb41d25659bd0835850 (Apache-2.0 / MIT).

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	// RoutedMoeMaxMVTokens is the token threshold below which aligned shapes
	// take the zero-scratch indexed vector kernel (MAX_MV_TOKENS = 8).
	RoutedMoeMaxMVTokens = 8

	// RoutedMoeRouteThreads is the threadgroup width of the two-stage routing
	// kernels (MOE_ROUTE_THREADS = 128, 4 simdgroups of 32 lanes).
	RoutedMoeRouteThreads = 128

	// RoutedMoeRouteSimds is the simdgroup count per routing threadgroup.
	RoutedMoeRouteSimds = 4

	// RoutedMoeCountsStride is the uint32 stride per expert bucket in Counts:
	// 128 per-lane counts plus 1 bucket total at offset 128.
	RoutedMoeCountsStride = RoutedMoeRouteThreads + 1

	// RoutedMoeTaskTileRows is the maximum route rows grouped into one
	// routed GEMM task tile (MOE_ROUTE_TILE = 32).
	RoutedMoeTaskTileRows = 32

	// RoutedMoeOutputTileCols is the output feature tile width for the routed
	// matrix kernel (OUTPUT_TILE = 64).
	RoutedMoeOutputTileCols = 64

	// RoutedMoeTaskBytes is the byte width of one task descriptor (uint4).
	RoutedMoeTaskBytes = 16
)

var (
	// ErrRoutedMoeInvalidConfig reports invalid shape, slot, or K-alignment parameters.
	ErrRoutedMoeInvalidConfig = errors.New("metalgemm: invalid routed MoE config")
	// ErrRoutedMoeUnsupportedDType reports an unsupported or unrecognized weight dtype.
	ErrRoutedMoeUnsupportedDType = errors.New("metalgemm: unsupported routed MoE weight dtype")
	// ErrRoutedMoeOverflow reports checked integer overflow in tensor or scratch sizing.
	ErrRoutedMoeOverflow = errors.New("metalgemm: routed MoE workspace or tensor size overflow")
	// ErrRoutedMoeBufferShape reports slice length mismatches against the declared config.
	ErrRoutedMoeBufferShape = errors.New("metalgemm: routed MoE buffer length mismatch")
)

// RoutedMoeDType identifies a GGML weight format for routed MoE projections.
type RoutedMoeDType string

const (
	RoutedMoeDTypeF32  RoutedMoeDType = "F32"
	RoutedMoeDTypeF16  RoutedMoeDType = "F16"
	RoutedMoeDTypeQ4_0 RoutedMoeDType = "Q4_0"
	RoutedMoeDTypeQ4_1 RoutedMoeDType = "Q4_1"
	RoutedMoeDTypeQ5_0 RoutedMoeDType = "Q5_0"
	RoutedMoeDTypeQ5_1 RoutedMoeDType = "Q5_1"
	RoutedMoeDTypeQ8_0 RoutedMoeDType = "Q8_0"
	RoutedMoeDTypeQ2_K RoutedMoeDType = "Q2_K"
	RoutedMoeDTypeQ3_K RoutedMoeDType = "Q3_K"
	RoutedMoeDTypeQ4_K RoutedMoeDType = "Q4_K"
	RoutedMoeDTypeQ5_K RoutedMoeDType = "Q5_K"
	RoutedMoeDTypeQ6_K RoutedMoeDType = "Q6_K"

	// Explicitly refused dtypes matching candle indexed_moe_forward.
	RoutedMoeDTypeQ8_1 RoutedMoeDType = "Q8_1"
	RoutedMoeDTypeQ8_K RoutedMoeDType = "Q8_K"
	RoutedMoeDTypeBF16 RoutedMoeDType = "BF16"
)

// RoutedMoeThreadgroupGeometry records the per-dtype Metal threadgroup shape,
// output divisor, and kernel entry names for dual vector/matrix dispatch.
type RoutedMoeThreadgroupGeometry struct {
	DType        RoutedMoeDType
	ThreadgroupX uint32
	ThreadgroupY uint32
	NDiv         uint32
	ThreadsPerTG uint32
	VectorKernel string
	MatrixKernel string

	blockSize uint64
	typeBytes uint64
}

// Per-dtype threadgroup geometry from candle quantized_moe.rs@cfcbd718.
var routedMoeGeometryTable = map[RoutedMoeDType]RoutedMoeThreadgroupGeometry{
	RoutedMoeDTypeQ4_0: {
		DType: RoutedMoeDTypeQ4_0, ThreadgroupX: 8, ThreadgroupY: 8, NDiv: 8, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q4_0_f32", MatrixKernel: "kernel_mul_mm_routed_q4_0_f32",
		blockSize: 32, typeBytes: 18,
	},
	RoutedMoeDTypeQ4_1: {
		DType: RoutedMoeDTypeQ4_1, ThreadgroupX: 8, ThreadgroupY: 8, NDiv: 8, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q4_1_f32", MatrixKernel: "kernel_mul_mm_routed_q4_1_f32",
		blockSize: 32, typeBytes: 20,
	},
	RoutedMoeDTypeQ5_0: {
		DType: RoutedMoeDTypeQ5_0, ThreadgroupX: 8, ThreadgroupY: 8, NDiv: 8, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q5_0_f32", MatrixKernel: "kernel_mul_mm_routed_q5_0_f32",
		blockSize: 32, typeBytes: 22,
	},
	RoutedMoeDTypeQ5_1: {
		DType: RoutedMoeDTypeQ5_1, ThreadgroupX: 8, ThreadgroupY: 8, NDiv: 8, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q5_1_f32", MatrixKernel: "kernel_mul_mm_routed_q5_1_f32",
		blockSize: 32, typeBytes: 24,
	},
	RoutedMoeDTypeQ8_0: {
		DType: RoutedMoeDTypeQ8_0, ThreadgroupX: 8, ThreadgroupY: 8, NDiv: 8, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q8_0_f32", MatrixKernel: "kernel_mul_mm_routed_q8_0_f32",
		blockSize: 32, typeBytes: 34,
	},
	RoutedMoeDTypeQ2_K: {
		DType: RoutedMoeDTypeQ2_K, ThreadgroupX: 2, ThreadgroupY: 32, NDiv: 8, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q2_K_f32", MatrixKernel: "kernel_mul_mm_routed_q2_K_f32",
		blockSize: 256, typeBytes: 84,
	},
	RoutedMoeDTypeQ3_K: {
		DType: RoutedMoeDTypeQ3_K, ThreadgroupX: 2, ThreadgroupY: 32, NDiv: 4, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q3_K_f32", MatrixKernel: "kernel_mul_mm_routed_q3_K_f32",
		blockSize: 256, typeBytes: 110,
	},
	RoutedMoeDTypeQ4_K: {
		DType: RoutedMoeDTypeQ4_K, ThreadgroupX: 4, ThreadgroupY: 8, NDiv: 4, ThreadsPerTG: 32,
		VectorKernel: "kernel_mul_mv_id_q4_K_f32", MatrixKernel: "kernel_mul_mm_routed_q4_K_f32",
		blockSize: 256, typeBytes: 144,
	},
	RoutedMoeDTypeQ5_K: {
		DType: RoutedMoeDTypeQ5_K, ThreadgroupX: 2, ThreadgroupY: 32, NDiv: 4, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q5_K_f32", MatrixKernel: "kernel_mul_mm_routed_q5_K_f32",
		blockSize: 256, typeBytes: 176,
	},
	RoutedMoeDTypeQ6_K: {
		DType: RoutedMoeDTypeQ6_K, ThreadgroupX: 2, ThreadgroupY: 32, NDiv: 2, ThreadsPerTG: 64,
		VectorKernel: "kernel_mul_mv_id_q6_K_f32", MatrixKernel: "kernel_mul_mm_routed_q6_K_f32",
		blockSize: 256, typeBytes: 210,
	},
	RoutedMoeDTypeF16: {
		DType: RoutedMoeDTypeF16, ThreadgroupX: 32, ThreadgroupY: 1, NDiv: 1, ThreadsPerTG: 32,
		VectorKernel: "kernel_mul_mv_id_f16_f32", MatrixKernel: "kernel_mul_mm_routed_f16_f32",
		blockSize: 1, typeBytes: 2,
	},
	RoutedMoeDTypeF32: {
		DType: RoutedMoeDTypeF32, ThreadgroupX: 32, ThreadgroupY: 1, NDiv: 1, ThreadsPerTG: 32,
		VectorKernel: "kernel_mul_mv_id_f32_f32", MatrixKernel: "kernel_mul_mm_routed_f32_f32",
		blockSize: 1, typeBytes: 4,
	},
}

// RoutedMoeVectorGeometry resolves the Metal threadgroup geometry for an
// admitted weight dtype and refuses unsupported dtypes such as Q8_1, Q8_K, BF16,
// or unknown/empty names.
func RoutedMoeVectorGeometry(dtype RoutedMoeDType) (RoutedMoeThreadgroupGeometry, error) {
	norm := RoutedMoeDType(strings.ToUpper(strings.TrimSpace(string(dtype))))
	if norm == "" {
		return RoutedMoeThreadgroupGeometry{}, fmt.Errorf("%w: empty weight dtype", ErrRoutedMoeUnsupportedDType)
	}
	if geom, ok := routedMoeGeometryTable[norm]; ok {
		return geom, nil
	}
	return RoutedMoeThreadgroupGeometry{}, fmt.Errorf("%w: %q", ErrRoutedMoeUnsupportedDType, dtype)
}

// RoutedMoeConfig defines the dimensions and weight format of a routed MoE
// projection: weights [Experts, N, K], input [Tokens, Slots, K], ids [Tokens, TopK]
// producing [Tokens, TopK, N].
type RoutedMoeConfig struct {
	Tokens      int
	TopK        int
	Slots       int
	Experts     int
	N           int
	K           int
	WeightDType RoutedMoeDType
}

type routedMoeCheckedLayout struct {
	geom              RoutedMoeThreadgroupGeometry
	pairs             uint64
	taskCap           uint64
	countsBytes       uint64
	routesBytes       uint64
	tasksBytes        uint64
	totalScratchBytes uint64
	weightExpertBytes uint64
}

func checkedMulU64(a, b uint64) (uint64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > math.MaxUint64/b {
		return 0, false
	}
	return a * b, true
}

func checkedAddU64(a, b uint64) (uint64, bool) {
	if a > math.MaxUint64-b {
		return 0, false
	}
	return a + b, true
}

func fitsHostInt(v uint64) bool {
	return v <= uint64(math.MaxInt)
}

func (cfg RoutedMoeConfig) validateLayout() (routedMoeCheckedLayout, error) {
	if cfg.Tokens <= 0 || cfg.TopK <= 0 || cfg.Experts <= 0 || cfg.N <= 0 || cfg.K <= 0 {
		return routedMoeCheckedLayout{}, fmt.Errorf(
			"%w: dimensions must be positive (tokens=%d topk=%d experts=%d n=%d k=%d)",
			ErrRoutedMoeInvalidConfig, cfg.Tokens, cfg.TopK, cfg.Experts, cfg.N, cfg.K,
		)
	}
	if cfg.K%32 != 0 {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: K=%d must be divisible by 32", ErrRoutedMoeInvalidConfig, cfg.K)
	}
	if cfg.Slots != 1 && cfg.Slots != cfg.TopK {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: slots=%d must be 1 or topk=%d", ErrRoutedMoeInvalidConfig, cfg.Slots, cfg.TopK)
	}
	geom, err := RoutedMoeVectorGeometry(cfg.WeightDType)
	if err != nil {
		return routedMoeCheckedLayout{}, err
	}

	// Metal kernel constants index experts in [0, u32::MAX) so sentinel = experts fits in uint32.
	if uint64(cfg.Experts) >= math.MaxUint32 ||
		uint64(cfg.Tokens) > math.MaxUint32 ||
		uint64(cfg.TopK) > math.MaxUint32 ||
		uint64(cfg.Slots) > math.MaxUint32 ||
		uint64(cfg.N) > math.MaxUint32 ||
		uint64(cfg.K) > math.MaxUint32 {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: dimension exceeds uint32 kernel index range", ErrRoutedMoeOverflow)
	}

	pairs, ok := checkedMulU64(uint64(cfg.Tokens), uint64(cfg.TopK))
	if !ok || pairs > math.MaxUint32 || !fitsHostInt(pairs) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: token*topk pairs overflows kernel index range", ErrRoutedMoeOverflow)
	}

	// Check F32 input [Tokens, Slots, K], F32 output [Tokens, TopK, N], and weights [Experts, N, K].
	inElems, ok := checkedMulU64(uint64(cfg.Tokens), uint64(cfg.Slots))
	if !ok {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: input token*slots overflow", ErrRoutedMoeOverflow)
	}
	inElems, ok = checkedMulU64(inElems, uint64(cfg.K))
	if !ok || !fitsHostInt(inElems) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: input element count overflow", ErrRoutedMoeOverflow)
	}
	if inBytes, ok := checkedMulU64(inElems, 4); !ok || !fitsHostInt(inBytes) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: input byte size overflow", ErrRoutedMoeOverflow)
	}

	outElems, ok := checkedMulU64(pairs, uint64(cfg.N))
	if !ok || !fitsHostInt(outElems) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: output element count overflow", ErrRoutedMoeOverflow)
	}
	if outBytes, ok := checkedMulU64(outElems, 4); !ok || !fitsHostInt(outBytes) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: output byte size overflow", ErrRoutedMoeOverflow)
	}

	wExpertElems, ok := checkedMulU64(uint64(cfg.N), uint64(cfg.K))
	if !ok {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: weight expert element count overflow", ErrRoutedMoeOverflow)
	}
	if wTotalElems, ok := checkedMulU64(uint64(cfg.Experts), wExpertElems); !ok || !fitsHostInt(wTotalElems) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: weight element count overflow", ErrRoutedMoeOverflow)
	}

	var weightRowBytes uint64
	if uint64(cfg.K)%geom.blockSize == 0 {
		weightRowBytes = (uint64(cfg.K) / geom.blockSize) * geom.typeBytes
	} else {
		// Allow K % 32 == 0 host-oracle configs while conservatively upper-bounding row bytes.
		blocks := (uint64(cfg.K) + geom.blockSize - 1) / geom.blockSize
		weightRowBytes = blocks * geom.typeBytes
	}
	weightExpertBytes, ok := checkedMulU64(uint64(cfg.N), weightRowBytes)
	if !ok {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: expert weight byte stride overflow", ErrRoutedMoeOverflow)
	}
	if weightBytes, ok := checkedMulU64(uint64(cfg.Experts), weightExpertBytes); !ok || !fitsHostInt(weightBytes) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: total weight byte size overflow", ErrRoutedMoeOverflow)
	}

	// Matrix scratch sizes:
	//   counts = (experts + 1) * 129 * 4
	//   routes = pairs * 4
	//   tasks  = (min(pairs, ceil(pairs/32) + experts) + 1) * 16
	buckets, ok := checkedAddU64(uint64(cfg.Experts), 1)
	if !ok {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: experts+1 overflow", ErrRoutedMoeOverflow)
	}
	countsElems, ok := checkedMulU64(buckets, RoutedMoeCountsStride)
	if !ok || !fitsHostInt(countsElems) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: counts element count overflow", ErrRoutedMoeOverflow)
	}
	countsBytes, ok := checkedMulU64(countsElems, 4)
	if !ok || !fitsHostInt(countsBytes) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: counts byte size overflow", ErrRoutedMoeOverflow)
	}
	routesBytes, ok := checkedMulU64(pairs, 4)
	if !ok || !fitsHostInt(routesBytes) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: routes byte size overflow", ErrRoutedMoeOverflow)
	}

	ceilPairs := 1 + (pairs-1)/RoutedMoeTaskTileRows
	ceilPlusExperts, ok := checkedAddU64(ceilPairs, uint64(cfg.Experts))
	if !ok {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: task capacity overflow", ErrRoutedMoeOverflow)
	}
	taskCap := pairs
	if ceilPlusExperts < taskCap {
		taskCap = ceilPlusExperts
	}
	taskSlots, ok := checkedAddU64(taskCap, 1)
	if !ok || !fitsHostInt(taskSlots) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: task header slot overflow", ErrRoutedMoeOverflow)
	}
	tasksBytes, ok := checkedMulU64(taskSlots, RoutedMoeTaskBytes)
	if !ok || !fitsHostInt(tasksBytes) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: tasks byte size overflow", ErrRoutedMoeOverflow)
	}

	totalScratch, ok := checkedAddU64(countsBytes, routesBytes)
	if !ok {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: total scratch overflow", ErrRoutedMoeOverflow)
	}
	totalScratch, ok = checkedAddU64(totalScratch, tasksBytes)
	if !ok || !fitsHostInt(totalScratch) {
		return routedMoeCheckedLayout{}, fmt.Errorf("%w: total scratch overflow", ErrRoutedMoeOverflow)
	}

	return routedMoeCheckedLayout{
		geom:              geom,
		pairs:             pairs,
		taskCap:           taskCap,
		countsBytes:       countsBytes,
		routesBytes:       routesBytes,
		tasksBytes:        tasksBytes,
		totalScratchBytes: totalScratch,
		weightExpertBytes: weightExpertBytes,
	}, nil
}

// Validate checks dimensions, slot width, K alignment, weight dtype support,
// and checked integer overflow bounds across tensor and workspace sizes.
func (cfg RoutedMoeConfig) Validate() error {
	_, err := cfg.validateLayout()
	return err
}

// RoutedMoeDispatchMode selects between the zero-scratch indexed vector kernel
// and the three-stage atomic-free matrix pipeline.
type RoutedMoeDispatchMode string

const (
	RoutedMoeDispatchVector RoutedMoeDispatchMode = "vector"
	RoutedMoeDispatchMatrix RoutedMoeDispatchMode = "matrix"
)

// RoutedMoeScratch holds the overflow-checked routing buffer sizes in bytes.
type RoutedMoeScratch struct {
	CountsBytes uint64
	RoutesBytes uint64
	TasksBytes  uint64
	TotalBytes  uint64
	TaskCap     uint32
}

// RoutedMoeDispatch describes the selected Metal dispatch path, threadgroup
// geometry, and routing scratch requirements for a validated config.
type RoutedMoeDispatch struct {
	Mode     RoutedMoeDispatchMode
	Geometry RoutedMoeThreadgroupGeometry
	Scratch  RoutedMoeScratch
	Reason   string
}

// SelectRoutedMoeDispatch applies Candle's dual vector/matrix dispatch
// heuristic and computes overflow-checked routing workspace sizes.
func SelectRoutedMoeDispatch(cfg RoutedMoeConfig) (RoutedMoeDispatch, error) {
	layout, err := cfg.validateLayout()
	if err != nil {
		return RoutedMoeDispatch{}, err
	}

	reason := ""
	switch {
	case cfg.Tokens > RoutedMoeMaxMVTokens:
		reason = "tokens_gt_max_mv_tokens"
	case cfg.N%int(layout.geom.NDiv) != 0:
		reason = "n_not_divisible_by_output_divisor"
	case cfg.K < int(layout.geom.ThreadsPerTG):
		reason = "k_below_vector_threadgroup_width"
	case cfg.K > math.MaxInt32 || cfg.Experts > math.MaxInt32 || cfg.TopK > math.MaxInt32:
		reason = "dim_exceeds_int32_mv_abi"
	case layout.weightExpertBytes > math.MaxUint32:
		reason = "weight_expert_bytes_exceeds_u32"
	}

	if reason == "" {
		return RoutedMoeDispatch{
			Mode:     RoutedMoeDispatchVector,
			Geometry: layout.geom,
			Scratch:  RoutedMoeScratch{},
			Reason:   "tokens_le_8_aligned_vector",
		}, nil
	}

	return RoutedMoeDispatch{
		Mode:     RoutedMoeDispatchMatrix,
		Geometry: layout.geom,
		Scratch: RoutedMoeScratch{
			CountsBytes: layout.countsBytes,
			RoutesBytes: layout.routesBytes,
			TasksBytes:  layout.tasksBytes,
			TotalBytes:  layout.totalScratchBytes,
			TaskCap:     uint32(layout.taskCap),
		},
		Reason: reason,
	}, nil
}

// RoutedMoeScratchSizes returns the routing scratch byte requirements for cfg
// (all zeros when SelectRoutedMoeDispatch chooses the vector path).
func RoutedMoeScratchSizes(cfg RoutedMoeConfig) (RoutedMoeScratch, error) {
	dispatch, err := SelectRoutedMoeDispatch(cfg)
	if err != nil {
		return RoutedMoeScratch{}, err
	}
	return dispatch.Scratch, nil
}

// RoutedMoeTask represents one 32-row tile task emitted by Stage 2 compact
// (`tasks[1+idx] = uint4(Expert, RouteStart, Rows, 0)`). When Expert equals
// SentinelID (Config.Experts), Stage 3 zero-fills the referenced output rows.
type RoutedMoeTask struct {
	Expert     uint32
	RouteStart uint32
	Rows       uint32
}

// RoutedMoePlan captures the deterministic atomic-free Stage 1 + Stage 2
// routing tables and tile work list over fixed expert IDs.
type RoutedMoePlan struct {
	Config        RoutedMoeConfig
	Dispatch      RoutedMoeDispatch
	Pairs         uint32
	SentinelID    uint32
	Counts        []uint32
	ExpertCounts  []uint32
	ExpertOffsets []uint32
	Routes        []uint32
	Tasks         []RoutedMoeTask
	TotalTasks    uint32
}

func clampExpertBucket(id int32, experts int) int {
	if id < 0 || int(id) >= experts {
		return experts
	}
	return int(id)
}

// BuildRoutedMoePlan runs the atomic-free two-stage counts->compact planner
// over ids [Tokens * TopK]. Negative or out-of-range expert IDs clamp into the
// sentinel bucket Experts, routes preserve token/slot scan order within each
// bucket without atomics, and tasks split each non-empty bucket into <=32-row tiles.
func BuildRoutedMoePlan(cfg RoutedMoeConfig, ids []int32) (RoutedMoePlan, error) {
	dispatch, err := SelectRoutedMoeDispatch(cfg)
	if err != nil {
		return RoutedMoePlan{}, err
	}
	pairs := cfg.Tokens * cfg.TopK
	if len(ids) != pairs {
		return RoutedMoePlan{}, fmt.Errorf("%w: ids length=%d, want %d", ErrRoutedMoeBufferShape, len(ids), pairs)
	}

	buckets := cfg.Experts + 1
	counts := make([]uint32, buckets*RoutedMoeCountsStride)
	chunk := (pairs + RoutedMoeRouteThreads - 1) / RoutedMoeRouteThreads

	// Stage 1 (kernel_moe_route_counts): each of the 128 lanes scans a contiguous
	// slice of [0, pairs) and increments only its own disjoint lane counter.
	for tid := 0; tid < RoutedMoeRouteThreads; tid++ {
		begin := tid * chunk
		if begin > pairs {
			begin = pairs
		}
		end := begin + chunk
		if end > pairs {
			end = pairs
		}
		for p := begin; p < end; p++ {
			b := clampExpertBucket(ids[p], cfg.Experts)
			counts[b*RoutedMoeCountsStride+tid]++
		}
	}

	// Reduce 4 simdgroups (32 lanes each) into counts[b*129 + 128].
	for b := 0; b < buckets; b++ {
		base := b * RoutedMoeCountsStride
		var simdTotals [RoutedMoeRouteSimds]uint32
		for simd := 0; simd < RoutedMoeRouteSimds; simd++ {
			var s uint32
			for lane := 0; lane < 32; lane++ {
				s += counts[base+simd*32+lane]
			}
			simdTotals[simd] = s
		}
		var total uint32
		for simd := 0; simd < RoutedMoeRouteSimds; simd++ {
			total += simdTotals[simd]
		}
		counts[base+RoutedMoeRouteThreads] = total
	}

	// Stage 2 (kernel_moe_route_compact): prefix sums across 0..Experts and per-lane
	// exclusive offsets inside each bucket.
	expertCounts := make([]uint32, buckets)
	expertOffsets := make([]uint32, buckets)
	taskOffsets := make([]uint32, buckets)
	laneWrite := make([]uint32, buckets*RoutedMoeRouteThreads)

	var routeAcc, taskAcc uint32
	for b := 0; b < buckets; b++ {
		base := b * RoutedMoeCountsStride
		c := counts[base+RoutedMoeRouteThreads]
		expertCounts[b] = c
		expertOffsets[b] = routeAcc
		taskOffsets[b] = taskAcc

		laneCursor := routeAcc
		for tid := 0; tid < RoutedMoeRouteThreads; tid++ {
			laneWrite[b*RoutedMoeRouteThreads+tid] = laneCursor
			laneCursor += counts[base+tid]
		}

		routeAcc += c
		if c > 0 {
			taskAcc += (c + RoutedMoeTaskTileRows - 1) / RoutedMoeTaskTileRows
		}
	}

	// Order-preserving atomic-free compaction into Routes: every (b, tid) writes
	// strictly to its own disjoint sub-range.
	routes := make([]uint32, pairs)
	for tid := 0; tid < RoutedMoeRouteThreads; tid++ {
		begin := tid * chunk
		if begin > pairs {
			begin = pairs
		}
		end := begin + chunk
		if end > pairs {
			end = pairs
		}
		for p := begin; p < end; p++ {
			b := clampExpertBucket(ids[p], cfg.Experts)
			idx := b*RoutedMoeRouteThreads + tid
			routes[laneWrite[idx]] = uint32(p)
			laneWrite[idx]++
		}
	}

	tasks := make([]RoutedMoeTask, taskAcc)
	for b := 0; b < buckets; b++ {
		total := expertCounts[b]
		if total == 0 {
			continue
		}
		tiles := (total + RoutedMoeTaskTileRows - 1) / RoutedMoeTaskTileRows
		baseRoute := expertOffsets[b]
		baseTask := taskOffsets[b]
		for tile := uint32(0); tile < tiles; tile++ {
			start := tile * RoutedMoeTaskTileRows
			rows := total - start
			if rows > RoutedMoeTaskTileRows {
				rows = RoutedMoeTaskTileRows
			}
			tasks[baseTask+tile] = RoutedMoeTask{
				Expert:     uint32(b),
				RouteStart: baseRoute + start,
				Rows:       rows,
			}
		}
	}

	return RoutedMoePlan{
		Config:        cfg,
		Dispatch:      dispatch,
		Pairs:         uint32(pairs),
		SentinelID:    uint32(cfg.Experts),
		Counts:        counts,
		ExpertCounts:  expertCounts,
		ExpertOffsets: expertOffsets,
		Routes:        routes,
		Tasks:         tasks,
		TotalTasks:    taskAcc,
	}, nil
}

// RoutedMoeOracleOutput holds the compiled routing plan, per-route projection
// [Tokens, TopK, N], and optional weighted-sum reduction [Tokens, N].
type RoutedMoeOracleOutput struct {
	Plan     RoutedMoePlan
	PerRoute []float32
	Reduced  []float32
}

// EvalRoutedMoeHostOracle executes the pure-Go host oracle for the routed MoE
// projection using the compiled atomic-free plan:
//   - inputF32: [Tokens, Slots, K] (Slots == 1 or TopK)
//   - weightsF32: [Experts, N, K]
//   - ids: [Tokens, TopK]
//   - topkWeights: optional [Tokens, TopK]; when non-empty, populates Reduced [Tokens, N].
//
// Sentinel tasks (task.Expert == Experts) explicitly zero-fill invalid-id rows in PerRoute.
func EvalRoutedMoeHostOracle(
	cfg RoutedMoeConfig,
	inputF32 []float32,
	weightsF32 []float32,
	ids []int32,
	topkWeights []float32,
) (RoutedMoeOracleOutput, error) {
	plan, err := BuildRoutedMoePlan(cfg, ids)
	if err != nil {
		return RoutedMoeOracleOutput{}, err
	}

	wantIn := cfg.Tokens * cfg.Slots * cfg.K
	if len(inputF32) != wantIn {
		return RoutedMoeOracleOutput{}, fmt.Errorf("%w: inputF32 length=%d, want %d", ErrRoutedMoeBufferShape, len(inputF32), wantIn)
	}
	wantW := cfg.Experts * cfg.N * cfg.K
	if len(weightsF32) != wantW {
		return RoutedMoeOracleOutput{}, fmt.Errorf("%w: weightsF32 length=%d, want %d", ErrRoutedMoeBufferShape, len(weightsF32), wantW)
	}
	pairs := cfg.Tokens * cfg.TopK
	if len(topkWeights) != 0 && len(topkWeights) != pairs {
		return RoutedMoeOracleOutput{}, fmt.Errorf("%w: topkWeights length=%d, want %d", ErrRoutedMoeBufferShape, len(topkWeights), pairs)
	}

	perRoute := make([]float32, pairs*cfg.N)
	// Stage 3 (kernel_mul_mm_routed / vector mv_id oracle): rowid indirection.
	for _, task := range plan.Tasks {
		if task.Expert == plan.SentinelID {
			for r := uint32(0); r < task.Rows; r++ {
				route := int(plan.Routes[task.RouteStart+r])
				outBase := route * cfg.N
				for n := 0; n < cfg.N; n++ {
					perRoute[outBase+n] = 0
				}
			}
			continue
		}

		wExpertBase := int(task.Expert) * cfg.N * cfg.K
		for r := uint32(0); r < task.Rows; r++ {
			route := int(plan.Routes[task.RouteStart+r])
			batch := route / cfg.TopK
			rowID := route % cfg.TopK
			slot := 0
			if cfg.Slots == cfg.TopK {
				slot = rowID
			}
			xBase := (batch*cfg.Slots + slot) * cfg.K
			outBase := route * cfg.N
			for n := 0; n < cfg.N; n++ {
				wBase := wExpertBase + n*cfg.K
				var acc float64
				for k := 0; k < cfg.K; k++ {
					acc += float64(inputF32[xBase+k]) * float64(weightsF32[wBase+k])
				}
				perRoute[outBase+n] = float32(acc)
			}
		}
	}

	var reduced []float32
	if len(topkWeights) > 0 {
		reduced, err = ReduceRoutedMoeWeightedSum(cfg, perRoute, topkWeights)
		if err != nil {
			return RoutedMoeOracleOutput{}, err
		}
	}

	return RoutedMoeOracleOutput{
		Plan:     plan,
		PerRoute: perRoute,
		Reduced:  reduced,
	}, nil
}

// ReduceRoutedMoeWeightedSum contracts perRoute [Tokens, TopK, N] by
// topkWeights [Tokens, TopK] into [Tokens, N].
func ReduceRoutedMoeWeightedSum(cfg RoutedMoeConfig, perRoute, topkWeights []float32) ([]float32, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	pairs := cfg.Tokens * cfg.TopK
	if len(perRoute) != pairs*cfg.N {
		return nil, fmt.Errorf("%w: perRoute length=%d, want %d", ErrRoutedMoeBufferShape, len(perRoute), pairs*cfg.N)
	}
	if len(topkWeights) != pairs {
		return nil, fmt.Errorf("%w: topkWeights length=%d, want %d", ErrRoutedMoeBufferShape, len(topkWeights), pairs)
	}

	out := make([]float32, cfg.Tokens*cfg.N)
	for t := 0; t < cfg.Tokens; t++ {
		for n := 0; n < cfg.N; n++ {
			var acc float64
			for s := 0; s < cfg.TopK; s++ {
				p := t*cfg.TopK + s
				acc += float64(topkWeights[p]) * float64(perRoute[p*cfg.N+n])
			}
			out[t*cfg.N+n] = float32(acc)
		}
	}
	return out, nil
}
