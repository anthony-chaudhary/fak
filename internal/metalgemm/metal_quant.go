//go:build darwin && arm64 && cgo

package metalgemm

// metal_quant.go — Split-K workgroup saturation for low-batch quantized GEMM on Apple Silicon.
//
// Study Provenance: EricLBuehler/mistral.rs:mistralrs-quant:afq_qmm_splitk
//
// Solves GPU wave starvation on Apple Silicon during autoregressive decode (M=1) and small batch
// inference. Under standard GEMV dispatch, a matrix multiplication with M=1 and N=2048 yields only
// ceil(M/32)*ceil(N/32) = 64 threadgroups. Apple Silicon GPUs have wide execution arrays designed
// for 512+ concurrent threadgroups. By partitioning the K dimension across split_k threadgroups,
// workgroup occupancy reaches 100% saturation (512 active threadgroups). Intermediate partial
// results are stored in a scratch buffer [split_k, M, N] and reduced along the split_k axis.

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation
#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <string.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

// Static pipeline state handles for Split-K stages.
static id<MTLComputePipelineState> gSplitKGemvQ8PSO = nil;
static id<MTLComputePipelineState> gSplitKGemvQ4PSO = nil;
static id<MTLComputePipelineState> gSplitKGemmQ8PSO = nil;
static id<MTLComputePipelineState> gSplitKGemmQ4PSO = nil;
static id<MTLComputePipelineState> gSplitKReducePSO = nil;
static dispatch_once_t gSplitKOnce;
static int gSplitKInitSuccess = 0;

static int mg_splitk_compile_pipelines(const char* source) {
    dispatch_once(&gSplitKOnce, ^{
        if (!mg_init()) {
            return;
        }
        if (gDev == nil) {
            return;
        }
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [gDev newLibraryWithSource:src options:nil error:&error];
        if (lib == nil) {
            NSLog(@"splitk: library compilation failed: %@", error);
            return;
        }

        id<MTLFunction> fnGemvQ8 = [lib newFunctionWithName:@"splitk_gemv_q8_0"];
        id<MTLFunction> fnGemvQ4 = [lib newFunctionWithName:@"splitk_gemv_q4_0"];
        id<MTLFunction> fnGemmQ8 = [lib newFunctionWithName:@"splitk_gemm_q8_0"];
        id<MTLFunction> fnGemmQ4 = [lib newFunctionWithName:@"splitk_gemm_q4_0"];
        id<MTLFunction> fnReduce = [lib newFunctionWithName:@"splitk_reduce"];

        if (!fnGemvQ8 || !fnGemvQ4 || !fnGemmQ8 || !fnGemmQ4 || !fnReduce) {
            NSLog(@"splitk: failed to resolve one or more shader functions");
            return;
        }

        gSplitKGemvQ8PSO = [gDev newComputePipelineStateWithFunction:fnGemvQ8 error:&error];
        gSplitKGemvQ4PSO = [gDev newComputePipelineStateWithFunction:fnGemvQ4 error:&error];
        gSplitKGemmQ8PSO = [gDev newComputePipelineStateWithFunction:fnGemmQ8 error:&error];
        gSplitKGemmQ4PSO = [gDev newComputePipelineStateWithFunction:fnGemmQ4 error:&error];
        gSplitKReducePSO = [gDev newComputePipelineStateWithFunction:fnReduce error:&error];

        if (!gSplitKGemvQ8PSO || !gSplitKGemvQ4PSO || !gSplitKGemmQ8PSO || !gSplitKGemmQ4PSO || !gSplitKReducePSO) {
            NSLog(@"splitk: compute pipeline state compilation failed: %@", error);
            return;
        }
        gSplitKInitSuccess = 1;
    });
    return gSplitKInitSuccess;
}

static int mg_splitk_gemm_execute(
    const float* x,
    const void* w,
    int format, // 0 = Q8_0, 1 = Q4_0
    float* out,
    int M,
    int N,
    int K,
    int split_k,
    int k_partition_size
) {
    if (!gSplitKInitSuccess || gDev == nil || gQueue == nil) {
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0 || split_k <= 0 || k_partition_size <= 0) {
        return -2;
    }

    size_t x_bytes = (size_t)M * K * sizeof(float);
    size_t blk_bytes = (format == 0) ? 34 : 18; // Q8_0=34, Q4_0=18
    size_t w_bytes = (size_t)N * (K / 32) * blk_bytes;
    size_t out_bytes = (size_t)M * N * sizeof(float);
    size_t scratch_bytes = (size_t)split_k * M * N * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> bufX = [gDev newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufW = [gDev newBufferWithBytes:w length:w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [gDev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        if (!bufX || !bufW || !bufOut) {
            return -3;
        }

        id<MTLBuffer> bufScratch = nil;
        if (split_k >= 2) {
            bufScratch = [gDev newBufferWithLength:scratch_bytes options:MTLResourceStorageModeShared];
            if (!bufScratch) {
                return -4;
            }
        }

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return -5;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) return -6;

        id<MTLComputePipelineState> pso = nil;
        if (format == 0) {
            pso = (M == 1) ? gSplitKGemvQ8PSO : gSplitKGemmQ8PSO;
        } else {
            pso = (M == 1) ? gSplitKGemvQ4PSO : gSplitKGemmQ4PSO;
        }

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufX offset:0 atIndex:0];
        [enc setBuffer:bufW offset:0 atIndex:1];
        if (split_k >= 2) {
            [enc setBuffer:bufScratch offset:0 atIndex:2];
        } else {
            [enc setBuffer:bufOut offset:0 atIndex:2]; // Direct write to final_out in single-pass
        }
        [enc setBytes:&M length:sizeof(int) atIndex:3];
        [enc setBytes:&N length:sizeof(int) atIndex:4];
        [enc setBytes:&K length:sizeof(int) atIndex:5];
        [enc setBytes:&k_partition_size length:sizeof(int) atIndex:6];
        [enc setBytes:&split_k length:sizeof(int) atIndex:7];

        if (M == 1) {
            MTLSize threadsPerTG = MTLSizeMake(32, 1, 1);
            MTLSize threadgroups = MTLSizeMake((N + 31) / 32, 1, split_k);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
        } else {
            int bm = 32;
            int bn = 32;
            int actM = (M < bm) ? M : bm;
            MTLSize threadsPerTG = MTLSizeMake(bn, actM, 1);
            MTLSize threadgroups = MTLSizeMake((N + bn - 1) / bn, (M + bm - 1) / bm, split_k);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
        }

        if (split_k >= 2) {
            // Synchronize buffer writes from Stage 1 before Stage 2 reduction reads.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Stage 2: Reduction
            [enc setComputePipelineState:gSplitKReducePSO];
            [enc setBuffer:bufScratch offset:0 atIndex:0];
            [enc setBuffer:bufOut offset:0 atIndex:1];
            [enc setBytes:&M length:sizeof(int) atIndex:2];
            [enc setBytes:&N length:sizeof(int) atIndex:3];
            [enc setBytes:&split_k length:sizeof(int) atIndex:4];

            if (M == 1) {
                int rThreads = (N < 256) ? N : 256;
                MTLSize rThreadsPerTG = MTLSizeMake(rThreads, 1, 1);
                MTLSize rThreadgroups = MTLSizeMake((N + rThreads - 1) / rThreads, 1, 1);
                [enc dispatchThreadgroups:rThreadgroups threadsPerThreadgroup:rThreadsPerTG];
            } else {
                MTLSize rThreadsPerTG = MTLSizeMake(16, 16, 1);
                MTLSize rThreadgroups = MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1);
                [enc dispatchThreadgroups:rThreadgroups threadsPerThreadgroup:rThreadsPerTG];
            }
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            NSLog(@"splitk: command buffer execution failed: %@", cb.error);
            return -7;
        }

        memcpy(out, [bufOut contents], out_bytes);
    }
    return 0;
}
*/
import "C"

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"
)

// Prior-art: EricLBuehler/mistral.rs:mistralrs-quant:afq_qmm_splitk (route=borrow)
// Prior-art: llama.cpp Metal / MLX (route=borrow)

// SplitKShaderSource embeds the Metal Shading Language source for Stage 1 partitioned
// dot products and Stage 2 intermediate scratch reduction kernels.
//
//go:embed splitk_gemm.metal
var SplitKShaderSource string

// TargetThreadgroups specifies the target active threadgroup count (512) for saturating Apple Silicon execution units.
const TargetThreadgroups = 512

// Tile sizes along M and N dimensions matching mistral.rs / Apple Silicon occupancy tuning.
const (
	SplitK_BM = 32
	SplitK_BN = 32
)

// LargeMNThreshold is the M*N element count above which problem dimensions naturally saturate
// the GPU threadgroup grid without requiring Split-K partitioning.
const LargeMNThreshold = 512 * 512

// SplitKConfig captures the partitioning parameters computed by CalculateSplitK.
type SplitKConfig struct {
	SplitK         int
	KPartitionSize int
	MTiles         int
	NTiles         int
	CurrentTGs     int
	TotalTGs       int
}

// CalculateSplitK calculates the dynamic Split-K factor following mistral.rs exact logic:
// - mTiles = ceil(m, BM), nTiles = ceil(n, BN)
// - currentTgs = mTiles * nTiles
// - splitK = max(1, TargetThreadgroups / max(1, currentTgs))
// - if k == 0 || groupSize == 0 { splitK = 1 } else { splitK = min(splitK, k / groupSize); while splitK > 1 && k % (splitK * groupSize) != 0 { splitK-- } }
// - kPartitionSize = k / splitK
// - totalTgs = currentTgs * splitK
func CalculateSplitK(m, n, k, groupSize int) SplitKConfig {
	bm := SplitK_BM
	bn := SplitK_BN

	mTiles := 1
	if m > 0 {
		mTiles = (m + bm - 1) / bm
	}
	nTiles := 1
	if n > 0 {
		nTiles = (n + bn - 1) / bn
	}

	currentTgs := mTiles * nTiles

	denom := currentTgs
	if denom < 1 {
		denom = 1
	}
	splitK := TargetThreadgroups / denom
	if splitK < 1 {
		splitK = 1
	}

	if k == 0 || groupSize == 0 {
		splitK = 1
	} else {
		maxSplits := k / groupSize
		if splitK > maxSplits {
			splitK = maxSplits
		}
		if splitK < 1 {
			splitK = 1
		}
		for splitK > 1 && k%(splitK*groupSize) != 0 {
			splitK--
		}
	}

	kPartitionSize := k
	if splitK > 0 {
		kPartitionSize = k / splitK
	}
	totalTgs := currentTgs * splitK

	return SplitKConfig{
		SplitK:         splitK,
		KPartitionSize: kPartitionSize,
		MTiles:         mTiles,
		NTiles:         nTiles,
		CurrentTGs:     currentTgs,
		TotalTGs:       totalTgs,
	}
}

// QuantFormat designates the weight quantization format (Q8_0 or Q4_0).
type QuantFormat int

const (
	QuantFormatQ8_0 QuantFormat = 0
	QuantFormatQ4_0 QuantFormat = 1
)

func (q QuantFormat) String() string {
	switch q {
	case QuantFormatQ8_0:
		return "Q8_0"
	case QuantFormatQ4_0:
		return "Q4_0"
	default:
		return "unknown"
	}
}

// QuantizedMatrix encapsulates quantized weight storage [N, K] in row-major block format.
type QuantizedMatrix struct {
	Format QuantFormat
	N      int // Output channels (number of rows)
	K      int // Input channels (reduction dimension)
	Data   []byte
}

// Float32ToFloat16Bits converts a float32 into its IEEE 754 half-precision 16-bit representation.
func Float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x007fffff

	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant = (mant | 0x00800000) >> uint(1-exp)
		return sign | uint16(mant>>13)
	} else if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}

// Float16BitsToFloat32 converts a 16-bit IEEE 754 half-precision word into a float32.
func Float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7c00) >> 10
	mant := uint32(h & 0x03ff)

	var fBits uint32
	if exp == 0 {
		if mant == 0 {
			fBits = sign
		} else {
			val := float32(mant) / 1024.0 * float32(math.Pow(2, -14))
			if (h & 0x8000) != 0 {
				return -val
			}
			return val
		}
	} else if exp == 31 {
		fBits = sign | 0x7f800000 | (mant << 13)
	} else {
		fBits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(fBits)
}

// QuantizeQ8_0 encodes an [N, K] float32 weight matrix into Q8_0 blocks.
// Each 32-weight block contains a 16-bit float scale followed by 32 int8 codes (34 bytes total).
func QuantizeQ8_0(w []float32, n, k int) (*QuantizedMatrix, error) {
	if n <= 0 || k <= 0 {
		return nil, errors.New("splitk: dimensions must be positive")
	}
	if k%32 != 0 {
		return nil, fmt.Errorf("splitk: K dimension (%d) must be a multiple of 32", k)
	}
	if len(w) < n*k {
		return nil, fmt.Errorf("splitk: weight slice too small (got %d, want %d)", len(w), n*k)
	}

	const blkSize = 32
	const blkBytes = 34
	numBlocksPerRow := k / blkSize
	totalBytes := n * numBlocksPerRow * blkBytes
	data := make([]byte, totalBytes)

	for row := 0; row < n; row++ {
		rowSrc := w[row*k : (row+1)*k]
		rowDst := data[row*numBlocksPerRow*blkBytes : (row+1)*numBlocksPerRow*blkBytes]

		for b := 0; b < numBlocksPerRow; b++ {
			srcOff := b * blkSize
			dstOff := b * blkBytes

			var maxAbs float32
			for i := 0; i < blkSize; i++ {
				abs := float32(math.Abs(float64(rowSrc[srcOff+i])))
				if abs > maxAbs {
					maxAbs = abs
				}
			}
			scale := maxAbs / 127.0
			if scale < 1e-8 {
				scale = 1e-8
			}
			f16Scale := Float32ToFloat16Bits(scale)
			binary.LittleEndian.PutUint16(rowDst[dstOff:dstOff+2], f16Scale)
			unpackedScale := Float16BitsToFloat32(f16Scale)
			invScale := float32(1.0) / unpackedScale

			for i := 0; i < blkSize; i++ {
				val := rowSrc[srcOff+i] * invScale
				code := int(math.Round(float64(val)))
				if code > 127 {
					code = 127
				} else if code < -127 {
					code = -127
				}
				rowDst[dstOff+2+i] = byte(int8(code))
			}
		}
	}

	return &QuantizedMatrix{
		Format: QuantFormatQ8_0,
		N:      n,
		K:      k,
		Data:   data,
	}, nil
}

// QuantizeQ4_0 encodes an [N, K] float32 weight matrix into Q4_0 blocks.
// Each 32-weight block contains a 16-bit float scale followed by 16 bytes of 4-bit nibbles (18 bytes total).
func QuantizeQ4_0(w []float32, n, k int) (*QuantizedMatrix, error) {
	if n <= 0 || k <= 0 {
		return nil, errors.New("splitk: dimensions must be positive")
	}
	if k%32 != 0 {
		return nil, fmt.Errorf("splitk: K dimension (%d) must be a multiple of 32", k)
	}
	if len(w) < n*k {
		return nil, fmt.Errorf("splitk: weight slice too small (got %d, want %d)", len(w), n*k)
	}

	const blkSize = 32
	const blkBytes = 18
	numBlocksPerRow := k / blkSize
	totalBytes := n * numBlocksPerRow * blkBytes
	data := make([]byte, totalBytes)

	for row := 0; row < n; row++ {
		rowSrc := w[row*k : (row+1)*k]
		rowDst := data[row*numBlocksPerRow*blkBytes : (row+1)*numBlocksPerRow*blkBytes]

		for b := 0; b < numBlocksPerRow; b++ {
			srcOff := b * blkSize
			dstOff := b * blkBytes

			var maxAbs float32
			for i := 0; i < blkSize; i++ {
				abs := float32(math.Abs(float64(rowSrc[srcOff+i])))
				if abs > maxAbs {
					maxAbs = abs
				}
			}
			scale := maxAbs / 7.0
			if scale < 1e-8 {
				scale = 1e-8
			}
			f16Scale := Float32ToFloat16Bits(scale)
			binary.LittleEndian.PutUint16(rowDst[dstOff:dstOff+2], f16Scale)
			unpackedScale := Float16BitsToFloat32(f16Scale)
			invScale := float32(1.0) / unpackedScale

			for j := 0; j < 16; j++ {
				v0 := int(math.Round(float64(rowSrc[srcOff+2*j]*invScale))) + 8
				if v0 < 0 {
					v0 = 0
				} else if v0 > 15 {
					v0 = 15
				}
				v1 := int(math.Round(float64(rowSrc[srcOff+2*j+1]*invScale))) + 8
				if v1 < 0 {
					v1 = 0
				} else if v1 > 15 {
					v1 = 15
				}
				rowDst[dstOff+2+j] = byte((v1 << 4) | (v0 & 0x0f))
			}
		}
	}

	return &QuantizedMatrix{
		Format: QuantFormatQ4_0,
		N:      n,
		K:      k,
		Data:   data,
	}, nil
}

// DequantizeQ8_0 dequantizes a Q8_0 payload into float32.
func DequantizeQ8_0(dst []float32, data []byte, n, k int) error {
	if n <= 0 || k <= 0 || k%32 != 0 {
		return errors.New("splitk: invalid dimensions for dequantization")
	}
	const blkSize = 32
	const blkBytes = 34
	numBlocks := (n * k) / blkSize
	neededBytes := numBlocks * blkBytes
	if len(data) < neededBytes || len(dst) < n*k {
		return errors.New("splitk: buffer too small for dequantization")
	}

	for b := 0; b < numBlocks; b++ {
		srcOff := b * blkBytes
		dstOff := b * blkSize
		scaleBits := binary.LittleEndian.Uint16(data[srcOff : srcOff+2])
		scale := Float16BitsToFloat32(scaleBits)

		for i := 0; i < blkSize; i++ {
			qs := int8(data[srcOff+2+i])
			dst[dstOff+i] = scale * float32(qs)
		}
	}
	return nil
}

// DequantizeQ4_0 dequantizes a Q4_0 payload into float32.
func DequantizeQ4_0(dst []float32, data []byte, n, k int) error {
	if n <= 0 || k <= 0 || k%32 != 0 {
		return errors.New("splitk: invalid dimensions for dequantization")
	}
	const blkSize = 32
	const blkBytes = 18
	numBlocks := (n * k) / blkSize
	neededBytes := numBlocks * blkBytes
	if len(data) < neededBytes || len(dst) < n*k {
		return errors.New("splitk: buffer too small for dequantization")
	}

	for b := 0; b < numBlocks; b++ {
		srcOff := b * blkBytes
		dstOff := b * blkSize
		scaleBits := binary.LittleEndian.Uint16(data[srcOff : srcOff+2])
		scale := Float16BitsToFloat32(scaleBits)

		for j := 0; j < 16; j++ {
			byteVal := data[srcOff+2+j]
			q0 := int(byteVal&0x0f) - 8
			q1 := int(byteVal>>4) - 8
			dst[dstOff+2*j] = scale * float32(q0)
			dst[dstOff+2*j+1] = scale * float32(q1)
		}
	}
	return nil
}

// ScratchBufferManager manages thread-safe allocation and reuse of scratch reduction memory.
type ScratchBufferManager struct {
	pool sync.Pool
}

// NewScratchBufferManager constructs a memory pool that recycles float32 slices up to 4MB
// to avoid heap churn during multi-slice reduction passes.
func NewScratchBufferManager() *ScratchBufferManager {
	return &ScratchBufferManager{
		pool: sync.Pool{
			New: func() any {
				return make([]float32, 0, 4096)
			},
		},
	}
}

// GetCPUBuffer retrieves or allocates a CPU scratch buffer to satisfy neededFloats.
// Slices returned are safe for concurrent use across goroutines.
func (m *ScratchBufferManager) GetCPUBuffer(neededFloats int) []float32 {
	v := m.pool.Get()
	if v != nil {
		buf := v.([]float32)
		if cap(buf) >= neededFloats {
			res := buf[:neededFloats]
			for i := range res {
				res[i] = 0
			}
			return res
		}
	}
	return make([]float32, neededFloats)
}

// PutCPUBuffer returns a scratch buffer to the pool for reuse if within reasonable capacity.
func (m *ScratchBufferManager) PutCPUBuffer(buf []float32) {
	if cap(buf) > 0 && cap(buf) <= 4*1024*1024 {
		m.pool.Put(buf[:0])
	}
}

var defaultScratchManager = NewScratchBufferManager()

// MetalSplitKAvailable reports whether the Metal device and Split-K pipelines are initialized and ready.
func MetalSplitKAvailable() bool {
	if !Available() {
		return false
	}
	cStr := C.CString(SplitKShaderSource)
	defer C.free(unsafe.Pointer(cStr))
	return C.mg_splitk_compile_pipelines(cStr) == 1
}

// SplitKExecutionStats records dispatch telemetry and saturation metrics.
type SplitKExecutionStats struct {
	SplitK         int
	KPartitionSize int
	CurrentTGs     int
	TotalTGs       int
	GPUDispatched  bool
	SinglePass     bool
	Duration       time.Duration
}

// ExecuteSplitK performs dynamic dispatch of quantized matrix multiplication:
// - When split_k < 2 or m*n is large (>= LargeMNThreshold), fall back to single-pass dispatch.
// - When split_k >= 2, dispatch Stage 1 to scratch reduction buffer, followed by Stage 2 reduction.
// - Routes to GPU execution when Metal is available, otherwise falling back to pure-Go reference.
func ExecuteSplitK(X []float32, W *QuantizedMatrix, M, N, K int, out []float32) (SplitKExecutionStats, error) {
	start := time.Now()
	if len(X) < M*K {
		return SplitKExecutionStats{}, fmt.Errorf("splitk: X buffer too small (got %d, want %d)", len(X), M*K)
	}
	if len(out) < M*N {
		return SplitKExecutionStats{}, fmt.Errorf("splitk: out buffer too small (got %d, want %d)", len(out), M*N)
	}
	if W == nil || W.N != N || W.K != K {
		return SplitKExecutionStats{}, errors.New("splitk: invalid weight matrix dimensions")
	}

	cfg := CalculateSplitK(M, N, K, 32)
	singlePass := false
	if M*N >= LargeMNThreshold || cfg.SplitK < 2 {
		singlePass = true
		cfg.SplitK = 1
		cfg.KPartitionSize = K
		cfg.TotalTGs = cfg.CurrentTGs
	}

	stats := SplitKExecutionStats{
		SplitK:         cfg.SplitK,
		KPartitionSize: cfg.KPartitionSize,
		CurrentTGs:     cfg.CurrentTGs,
		TotalTGs:       cfg.TotalTGs,
		SinglePass:     singlePass,
	}

	// Try GPU path if available
	if MetalSplitKAvailable() {
		formatCode := C.int(0)
		if W.Format == QuantFormatQ4_0 {
			formatCode = C.int(1)
		}
		res := C.mg_splitk_gemm_execute(
			(*C.float)(unsafe.Pointer(&X[0])),
			unsafe.Pointer(&W.Data[0]),
			formatCode,
			(*C.float)(unsafe.Pointer(&out[0])),
			C.int(M),
			C.int(N),
			C.int(K),
			C.int(cfg.SplitK),
			C.int(cfg.KPartitionSize),
		)
		if res == 0 {
			stats.GPUDispatched = true
			stats.Duration = time.Since(start)
			return stats, nil
		}
	}

	// Fallback to pure-Go numerical reference path
	err := ExecuteSplitKRefWithConfig(X, W, M, N, K, cfg, out)
	stats.GPUDispatched = false
	stats.Duration = time.Since(start)
	return stats, err
}

// ExecuteSplitKRef executes matrix multiplication using the pure-Go reference implementation.
func ExecuteSplitKRef(X []float32, W *QuantizedMatrix, M, N, K int, out []float32) error {
	cfg := CalculateSplitK(M, N, K, 32)
	return ExecuteSplitKRefWithConfig(X, W, M, N, K, cfg, out)
}

// ExecuteSplitKRefWithConfig executes matrix multiplication with a pre-computed SplitKConfig.
func ExecuteSplitKRefWithConfig(X []float32, W *QuantizedMatrix, M, N, K int, cfg SplitKConfig, out []float32) error {
	if len(X) < M*K {
		return fmt.Errorf("splitk: X buffer too small (got %d, want %d)", len(X), M*K)
	}
	if len(out) < M*N {
		return fmt.Errorf("splitk: out buffer too small (got %d, want %d)", len(out), M*N)
	}
	if W == nil || W.N != N || W.K != K {
		return errors.New("splitk: invalid weight matrix dimensions")
	}

	if cfg.SplitK >= 2 {
		scratch := defaultScratchManager.GetCPUBuffer(cfg.SplitK * M * N)
		defer defaultScratchManager.PutCPUBuffer(scratch)

		// Stage 1: Partitioned K GEMM into scratch [split_k, M, N]
		for s := 0; s < cfg.SplitK; s++ {
			kStart := s * cfg.KPartitionSize
			kEnd := kStart + cfg.KPartitionSize
			if kEnd > K {
				kEnd = K
			}
			bStart := kStart / 32
			numBlocks := (kEnd - kStart) / 32
			sliceOffset := s * (M * N)

			for m := 0; m < M; m++ {
				xRow := X[m*K : (m+1)*K]
				rowOffset := sliceOffset + m*N

				for n := 0; n < N; n++ {
					var acc float32
					if W.Format == QuantFormatQ8_0 {
						const blkBytes = 34
						blocksPerRow := K / 32
						wRowBytes := W.Data[n*blocksPerRow*blkBytes : (n+1)*blocksPerRow*blkBytes]
						for b := 0; b < numBlocks; b++ {
							blkOff := (bStart + b) * blkBytes
							scaleBits := binary.LittleEndian.Uint16(wRowBytes[blkOff : blkOff+2])
							scale := Float16BitsToFloat32(scaleBits)
							xBlk := xRow[kStart+b*32 : kStart+(b+1)*32]
							var bsum float32
							for i := 0; i < 32; i++ {
								qs := int8(wRowBytes[blkOff+2+i])
								bsum += float32(qs) * xBlk[i]
							}
							acc += scale * bsum
						}
					} else if W.Format == QuantFormatQ4_0 {
						const blkBytes = 18
						blocksPerRow := K / 32
						wRowBytes := W.Data[n*blocksPerRow*blkBytes : (n+1)*blocksPerRow*blkBytes]
						for b := 0; b < numBlocks; b++ {
							blkOff := (bStart + b) * blkBytes
							scaleBits := binary.LittleEndian.Uint16(wRowBytes[blkOff : blkOff+2])
							scale := Float16BitsToFloat32(scaleBits)
							xBlk := xRow[kStart+b*32 : kStart+(b+1)*32]
							var bsum float32
							for j := 0; j < 16; j++ {
								byteVal := wRowBytes[blkOff+2+j]
								q0 := int(byteVal&0x0f) - 8
								q1 := int(byteVal>>4) - 8
								bsum += float32(q0)*xBlk[2*j] + float32(q1)*xBlk[2*j+1]
							}
							acc += scale * bsum
						}
					}
					scratch[rowOffset+n] = acc
				}
			}
		}

		// Stage 2: Reduction along split_k axis
		mn := M * N
		for idx := 0; idx < mn; idx++ {
			var sum float32
			for s := 0; s < cfg.SplitK; s++ {
				sum += scratch[s*mn+idx]
			}
			out[idx] = sum
		}
	} else {
		// Single-pass direct execution
		numBlocks := K / 32
		for m := 0; m < M; m++ {
			xRow := X[m*K : (m+1)*K]
			for n := 0; n < N; n++ {
				var acc float32
				if W.Format == QuantFormatQ8_0 {
					const blkBytes = 34
					blocksPerRow := K / 32
					wRowBytes := W.Data[n*blocksPerRow*blkBytes : (n+1)*blocksPerRow*blkBytes]
					for b := 0; b < numBlocks; b++ {
						blkOff := b * blkBytes
						scaleBits := binary.LittleEndian.Uint16(wRowBytes[blkOff : blkOff+2])
						scale := Float16BitsToFloat32(scaleBits)
						xBlk := xRow[b*32 : (b+1)*32]
						var bsum float32
						for i := 0; i < 32; i++ {
							qs := int8(wRowBytes[blkOff+2+i])
							bsum += float32(qs) * xBlk[i]
						}
						acc += scale * bsum
					}
				} else if W.Format == QuantFormatQ4_0 {
					const blkBytes = 18
					blocksPerRow := K / 32
					wRowBytes := W.Data[n*blocksPerRow*blkBytes : (n+1)*blocksPerRow*blkBytes]
					for b := 0; b < numBlocks; b++ {
						blkOff := b * blkBytes
						scaleBits := binary.LittleEndian.Uint16(wRowBytes[blkOff : blkOff+2])
						scale := Float16BitsToFloat32(scaleBits)
						xBlk := xRow[b*32 : (b+1)*32]
						var bsum float32
						for j := 0; j < 16; j++ {
							byteVal := wRowBytes[blkOff+2+j]
							q0 := int(byteVal&0x0f) - 8
							q1 := int(byteVal>>4) - 8
							bsum += float32(q0)*xBlk[2*j] + float32(q1)*xBlk[2*j+1]
						}
						acc += scale * bsum
					}
				}
				out[m*N+n] = acc
			}
		}
	}
	return nil
}
