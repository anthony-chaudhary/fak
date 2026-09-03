// Prior-art: mtplx / MLX blocked-sequential GDN prefill kernel (mtplx/kernels/gdn_blocked_prefill.py).

//go:build darwin && arm64 && cgo

package metalgemm

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <string.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

static id<MTLComputePipelineState> gGDNBlockedPrefillPSO = nil;

static int mg_gdn_blocked_prefill_init_pso(const char *src_str) {
    if (gGDNBlockedPrefillPSO != nil) return 1;
    if (!mg_init()) return 0;
    @autoreleasepool {
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:src_str];
        id<MTLLibrary> lib = [gDev newLibraryWithSource:src options:nil error:&error];
        if (lib == nil) {
            NSLog(@"mg_gdn_blocked_prefill: library compile error: %@", error);
            return 0;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gdn_blocked_sequential_prefill"];
        if (fn == nil) {
            NSLog(@"mg_gdn_blocked_prefill: kernel function not found");
            return 0;
        }
        gGDNBlockedPrefillPSO = [gDev newComputePipelineStateWithFunction:fn error:&error];
        if (gGDNBlockedPrefillPSO == nil) {
            NSLog(@"mg_gdn_blocked_prefill: PSO creation error: %@", error);
            return 0;
        }
        return 1;
    }
}

static int mg_gdn_blocked_prefill_run(
    const float *q, const float *k, const float *v, const float *g, const float *beta,
    const float *stateIn, float *stateOut, float *out,
    int batchSize, int numTokens, int numKeyHeads, int numValueHeads,
    int keyHeadDim, int valueHeadDim,
    int hasStateIn, int hasStateOut,
    float *gpuTimeMs)
{
    if (gGDNBlockedPrefillPSO == nil) return -1;
    if (q == NULL || k == NULL || v == NULL || g == NULL || beta == NULL || out == NULL) return -2;
    if (keyHeadDim != 128 || valueHeadDim % 32 != 0 || numTokens <= 0) return -3;

    @autoreleasepool {
        size_t qElems = (size_t)batchSize * numTokens * numKeyHeads * keyHeadDim;
        size_t kElems = qElems;
        size_t vElems = (size_t)batchSize * numTokens * numValueHeads * valueHeadDim;
        size_t gbElems = (size_t)batchSize * numTokens * numValueHeads;
        size_t stateElems = (size_t)batchSize * numValueHeads * valueHeadDim * keyHeadDim;
        size_t outElems = vElems;

        id<MTLBuffer> qBuf = [gDev newBufferWithBytes:q length:qElems * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kBuf = [gDev newBufferWithBytes:k length:kElems * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> vBuf = [gDev newBufferWithBytes:v length:vElems * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gBuf = [gDev newBufferWithBytes:g length:gbElems * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> betaBuf = [gDev newBufferWithBytes:beta length:gbElems * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [gDev newBufferWithLength:outElems * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLBuffer> sInBuf = nil;
        if (hasStateIn && stateIn != NULL) {
            sInBuf = [gDev newBufferWithBytes:stateIn length:stateElems * sizeof(float) options:MTLResourceStorageModeShared];
        } else {
            sInBuf = [gDev newBufferWithLength:sizeof(float) * 4 options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> sOutBuf = nil;
        if (hasStateOut && stateOut != NULL) {
            sOutBuf = [gDev newBufferWithLength:stateElems * sizeof(float) options:MTLResourceStorageModeShared];
        } else {
            sOutBuf = [gDev newBufferWithLength:sizeof(float) * 4 options:MTLResourceStorageModeShared];
        }

        if (!qBuf || !kBuf || !vBuf || !gBuf || !betaBuf || !outBuf || !sInBuf || !sOutBuf) {
            return -4;
        }

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -5;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil) return -6;

        [enc setComputePipelineState:gGDNBlockedPrefillPSO];
        [enc setBuffer:qBuf offset:0 atIndex:0];
        [enc setBuffer:kBuf offset:0 atIndex:1];
        [enc setBuffer:vBuf offset:0 atIndex:2];
        [enc setBuffer:gBuf offset:0 atIndex:3];
        [enc setBuffer:betaBuf offset:0 atIndex:4];
        [enc setBuffer:sInBuf offset:0 atIndex:5];
        [enc setBuffer:sOutBuf offset:0 atIndex:6];
        [enc setBuffer:outBuf offset:0 atIndex:7];

        [enc setBytes:&numTokens length:sizeof(int) atIndex:8];
        [enc setBytes:&numKeyHeads length:sizeof(int) atIndex:9];
        [enc setBytes:&numValueHeads length:sizeof(int) atIndex:10];
        [enc setBytes:&keyHeadDim length:sizeof(int) atIndex:11];
        [enc setBytes:&valueHeadDim length:sizeof(int) atIndex:12];
        [enc setBytes:&hasStateIn length:sizeof(int) atIndex:13];
        [enc setBytes:&hasStateOut length:sizeof(int) atIndex:14];

        MTLSize threadsPerTG = MTLSizeMake(256, 1, 1);
        MTLSize tgGrid = MTLSizeMake(valueHeadDim / 32, numValueHeads, batchSize);
        [enc dispatchThreadgroups:tgGrid threadsPerThreadgroup:threadsPerTG];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status != MTLCommandBufferStatusCompleted) {
            if (cmd.error) {
                NSLog(@"mg_gdn_blocked_prefill execution failed: %@", cmd.error);
            }
            return -7;
        }

        memcpy(out, outBuf.contents, outElems * sizeof(float));
        if (hasStateOut && stateOut != NULL) {
            memcpy(stateOut, sOutBuf.contents, stateElems * sizeof(float));
        }

        if (gpuTimeMs != NULL) {
            *gpuTimeMs = (float)((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0);
        }
        return 0;
    }
}
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

// Blocked-sequential staging parameters.
const (
	// GDNPrefillTokenBlockSize is the token block size TB = 32 for coalesced threadgroup staging.
	GDNPrefillTokenBlockSize = 32

	// GDNPrefillDimBlockSize is the value-head dimension block size DB = 32.
	GDNPrefillDimBlockSize = 32

	// GDNPrefillKeyHeadDim is the key head dimension Dk = 128 (Qwen3.8 standard).
	GDNPrefillKeyHeadDim = 128
)

// GDNBlockedPrefillMSL is the complete Metal Shading Language source for the
// blocked-sequential Gated DeltaNet prefill kernel.
//
// Key architectural mechanisms:
//  1. Stages k, q, v, g, beta into threadgroup memory in coalesced 32-token blocks (TB = 32).
//  2. Splits each value head into 32-row blocks (DB = 32), cutting redundant k/q memory reads by 32x.
//  3. Stores recurrent state fragments in registers (float4 st[4] per thread across 8 threads per dv row).
//  4. Reduces linear recurrence contractions via SIMD shuffles (simd_shuffle_down and simd_shuffle),
//     completely eliminating inner threadgroup barriers from the sequential token recurrence loop.
const GDNBlockedPrefillMSL = `#include <metal_stdlib>
using namespace metal;

constant int TB = 32;
constant int DB = 32;
constant int DK = 128;

kernel void gdn_blocked_sequential_prefill(
    device const float *q_dev [[buffer(0)]],
    device const float *k_dev [[buffer(1)]],
    device const float *v_dev [[buffer(2)]],
    device const float *g_dev [[buffer(3)]],
    device const float *beta_dev [[buffer(4)]],
    device const float *state_in [[buffer(5)]],
    device float *state_out [[buffer(6)]],
    device float *out_dev [[buffer(7)]],
    constant int &numTokens [[buffer(8)]],
    constant int &numKeyHeads [[buffer(9)]],
    constant int &numValueHeads [[buffer(10)]],
    constant int &keyHeadDim [[buffer(11)]],
    constant int &valueHeadDim [[buffer(12)]],
    constant int &hasStateIn [[buffer(13)]],
    constant int &hasStateOut [[buffer(14)]],
    uint3 groupPos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    int dv_block = groupPos.x;
    int hv = groupPos.y;
    int b = groupPos.z;
    int dv0 = dv_block * DB;
    int repeat = numValueHeads / numKeyHeads;
    int hk = hv / repeat;

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;

    int dv_local = lane_id / 8;
    int dv_idx = simd_id * 4 + dv_local;
    int k_thread_idx = lane_id % 8;
    int d0 = k_thread_idx * 16;
    int dv_global = dv0 + dv_idx;
    bool active_thread = (dv_global < valueHeadDim);

    // Staging buffers in threadgroup memory:
    // Total footprint: 2048*4 + 2048*4 + 1024*4 + 32*4 + 32*4 = 20,736 bytes (< 32 KB limit).
    threadgroup float sh_k[16 * 128];
    threadgroup float sh_q[16 * 128];
    threadgroup float sh_v[32 * 32];
    threadgroup float sh_g[32];
    threadgroup float sh_beta[32];

    // State fragment in registers: 16 floats per thread (4 x float4).
    // 8 threads x 16 floats = 128 floats (entire Dk row for dv_global).
    float4 st[4];
    if (hasStateIn != 0 && active_thread) {
        long state_offset = (((long)b * numValueHeads + hv) * valueHeadDim + dv_global) * keyHeadDim + d0;
        const device float4 *s_ptr = (const device float4 *)(state_in + state_offset);
        st[0] = s_ptr[0];
        st[1] = s_ptr[1];
        st[2] = s_ptr[2];
        st[3] = s_ptr[3];
    } else {
        st[0] = float4(0.0f);
        st[1] = float4(0.0f);
        st[2] = float4(0.0f);
        st[3] = float4(0.0f);
    }

    // Process sequence in coalesced 32-token blocks (TB = 32)
    for (int t0 = 0; t0 < numTokens; t0 += TB) {
        int cur_block = min(TB, numTokens - t0);

        // 1. Coalesced staging of v, g, beta for up to 32 tokens
        // 256 threads cooperatively load 32 x 32 = 1024 floats (256 float4s) of v
        if (tid < (uint)(cur_block * 8)) {
            int tok = tid / 8;
            int dv4 = tid % 8;
            int t = t0 + tok;
            long v_offset = (((long)b * numTokens + t) * numValueHeads + hv) * valueHeadDim + dv0 + dv4 * 4;
            threadgroup float4 *sh_v4 = (threadgroup float4 *)sh_v;
            sh_v4[tid] = *((const device float4 *)(v_dev + v_offset));
        }
        if (tid < (uint)cur_block) {
            int t = t0 + tid;
            long gb_offset = ((long)b * numTokens + t) * numValueHeads + hv;
            sh_g[tid] = g_dev[gb_offset];
            sh_beta[tid] = beta_dev[gb_offset];
        }

        // Process in two 16-token sub-blocks to fit within Apple Silicon 32 KB threadgroup memory
        for (int sub = 0; sub < cur_block; sub += 16) {
            int sub_tokens = min(16, cur_block - sub);

            // Coalesced stage k and q for sub_tokens (up to 16 x 128 = 2048 floats = 512 float4s)
            threadgroup float4 *sh_k4 = (threadgroup float4 *)sh_k;
            threadgroup float4 *sh_q4 = (threadgroup float4 *)sh_q;
            for (int i = 0; i < 2; ++i) {
                int load_idx = tid + i * 256;
                if (load_idx < sub_tokens * 32) {
                    int tok = load_idx / 32;
                    int dk4 = load_idx % 32;
                    int t = t0 + sub + tok;
                    long kq_offset = (((long)b * numTokens + t) * numKeyHeads + hk) * keyHeadDim + dk4 * 4;
                    sh_k4[load_idx] = *((const device float4 *)(k_dev + kq_offset));
                    sh_q4[load_idx] = *((const device float4 *)(q_dev + kq_offset));
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Sequential linear recurrence across sub_tokens.
            // ZERO threadgroup barriers inside this loop: all communication uses SIMD shuffle!
            for (int tok = 0; tok < sub_tokens; ++tok) {
                int block_tok = sub + tok;
                float g_val = sh_g[block_tok];
                float beta_val = sh_beta[block_tok];

                // 1. Decay state
                for (int i = 0; i < 4; ++i) {
                    st[i] *= g_val;
                }

                // 2. kvmem = dot(S, k)
                threadgroup const float4 *k_tok = (threadgroup const float4 *)(sh_k + tok * 128);
                float4 k_vec[4];
                k_vec[0] = k_tok[k_thread_idx * 4 + 0];
                k_vec[1] = k_tok[k_thread_idx * 4 + 1];
                k_vec[2] = k_tok[k_thread_idx * 4 + 2];
                k_vec[3] = k_tok[k_thread_idx * 4 + 3];

                float local_kv = dot(st[0], k_vec[0]) + dot(st[1], k_vec[1]) +
                                 dot(st[2], k_vec[2]) + dot(st[3], k_vec[3]);

                // Reduce across 8 threads of dv_local within the simdgroup
                float kv = local_kv;
                kv += simd_shuffle_down(kv, 4);
                kv += simd_shuffle_down(kv, 2);
                kv += simd_shuffle_down(kv, 1);

                // 3. Delta = (v - kvmem) * beta (computed by lane 0 of the 8-lane group)
                float delta = 0.0f;
                if ((lane_id % 8) == 0) {
                    float v_val = sh_v[block_tok * 32 + dv_idx];
                    delta = (v_val - kv) * beta_val;
                }
                // Broadcast delta to all 8 lanes in the group
                ushort base_lane = (lane_id / 8) * 8;
                delta = simd_shuffle(delta, base_lane);

                // 4. Update state: S += outer(k, delta)
                for (int i = 0; i < 4; ++i) {
                    st[i] += k_vec[i] * delta;
                }

                // 5. Readout: out = dot(S_updated, q)
                threadgroup const float4 *q_tok = (threadgroup const float4 *)(sh_q + tok * 128);
                float4 q_vec[4];
                q_vec[0] = q_tok[k_thread_idx * 4 + 0];
                q_vec[1] = q_tok[k_thread_idx * 4 + 1];
                q_vec[2] = q_tok[k_thread_idx * 4 + 2];
                q_vec[3] = q_tok[k_thread_idx * 4 + 3];

                float local_out = dot(st[0], q_vec[0]) + dot(st[1], q_vec[1]) +
                                  dot(st[2], q_vec[2]) + dot(st[3], q_vec[3]);

                float out_val = local_out;
                out_val += simd_shuffle_down(out_val, 4);
                out_val += simd_shuffle_down(out_val, 2);
                out_val += simd_shuffle_down(out_val, 1);

                // Lane 0 writes output into sh_v (reused in-place as output staging buffer)
                if ((lane_id % 8) == 0 && active_thread) {
                    sh_v[block_tok * 32 + dv_idx] = out_val;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Coalesced write of output (from sh_v) to device memory
        if (tid < (uint)(cur_block * 8)) {
            int tok = tid / 8;
            int dv4 = tid % 8;
            int t = t0 + tok;
            long out_offset = (((long)b * numTokens + t) * numValueHeads + hv) * valueHeadDim + dv0 + dv4 * 4;
            threadgroup const float4 *sh_out4 = (threadgroup const float4 *)sh_v;
            *((device float4 *)(out_dev + out_offset)) = sh_out4[tid];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final state writeback
    if (hasStateOut != 0 && active_thread) {
        long state_offset = (((long)b * numValueHeads + hv) * valueHeadDim + dv_global) * keyHeadDim + d0;
        device float4 *s_out_ptr = (device float4 *)(state_out + state_offset);
        s_out_ptr[0] = st[0];
        s_out_ptr[1] = st[1];
        s_out_ptr[2] = st[2];
        s_out_ptr[3] = st[3];
    }
}
`

var (
	gdnPrefillInitOnce sync.Once
	gdnPrefillInitErr  error
)

func ensureGDNBlockedPrefillPipeline() error {
	gdnPrefillInitOnce.Do(func() {
		if !Available() {
			gdnPrefillInitErr = errors.New("metalgemm: Metal is not available on this platform")
			return
		}
		cSrc := C.CString(GDNBlockedPrefillMSL)
		defer C.free(unsafe.Pointer(cSrc))
		if C.mg_gdn_blocked_prefill_init_pso(cSrc) != 1 {
			gdnPrefillInitErr = errors.New("metalgemm: failed to compile blocked-sequential GDN prefill Metal pipeline")
		}
	})
	return gdnPrefillInitErr
}

// GDNBlockedPrefillConfig defines the dimensions and blocked staging parameters.
type GDNBlockedPrefillConfig struct {
	BatchSize      int // B (defaults to 1)
	NumTokens      int // T (prompt length)
	NumKeyHeads    int // Hk
	NumValueHeads  int // Hv
	KeyHeadDim     int // Dk (standard 128)
	ValueHeadDim   int // Dv (standard 128, multiple of 32)
	TokenBlockSize int // TB = 32
	DimBlockSize   int // DB = 32
}

// DefaultGDNBlockedPrefillConfig constructs a valid configuration with standard Qwen3.8 parameters.
func DefaultGDNBlockedPrefillConfig(tokens, numKeyHeads, numValueHeads int) GDNBlockedPrefillConfig {
	return GDNBlockedPrefillConfig{
		BatchSize:      1,
		NumTokens:      tokens,
		NumKeyHeads:    numKeyHeads,
		NumValueHeads:  numValueHeads,
		KeyHeadDim:     GDNPrefillKeyHeadDim,
		ValueHeadDim:   128,
		TokenBlockSize: GDNPrefillTokenBlockSize,
		DimBlockSize:   GDNPrefillDimBlockSize,
	}
}

// Validate checks configuration invariants.
func (c *GDNBlockedPrefillConfig) Validate() error {
	if c.BatchSize <= 0 {
		c.BatchSize = 1
	}
	if c.NumTokens <= 0 {
		return errors.New("gdn_blocked_prefill: NumTokens must be positive")
	}
	if c.NumKeyHeads <= 0 || c.NumValueHeads <= 0 {
		return errors.New("gdn_blocked_prefill: head counts must be positive")
	}
	if c.NumValueHeads%c.NumKeyHeads != 0 {
		return fmt.Errorf("gdn_blocked_prefill: NumValueHeads (%d) must be a multiple of NumKeyHeads (%d)",
			c.NumValueHeads, c.NumKeyHeads)
	}
	if c.KeyHeadDim != GDNPrefillKeyHeadDim {
		return fmt.Errorf("gdn_blocked_prefill: KeyHeadDim must be %d, got %d",
			GDNPrefillKeyHeadDim, c.KeyHeadDim)
	}
	if c.ValueHeadDim <= 0 || c.ValueHeadDim%GDNPrefillDimBlockSize != 0 {
		return fmt.Errorf("gdn_blocked_prefill: ValueHeadDim (%d) must be a positive multiple of DB=%d",
			c.ValueHeadDim, GDNPrefillDimBlockSize)
	}
	if c.TokenBlockSize <= 0 {
		c.TokenBlockSize = GDNPrefillTokenBlockSize
	}
	if c.DimBlockSize <= 0 {
		c.DimBlockSize = GDNPrefillDimBlockSize
	}
	return nil
}

// GDNBlockedPrefillStats captures execution statistics and memory traffic metrics.
type GDNBlockedPrefillStats struct {
	Tokens               int
	TokenBlockSize       int
	DimBlockSize         int
	DRAMReadCutFactor    float64
	RedundantReadSavings int64   // bytes of k/q redundant memory reads eliminated
	GPUExecuteMs         float64 // Metal on-GPU execution time in milliseconds
}

// GDNBlockedPrefillDispatchGrid defines the grid layout for command dispatch.
type GDNBlockedPrefillDispatchGrid struct {
	ThreadgroupsPerGrid   [3]int
	ThreadsPerThreadgroup [3]int
}

// BuildGDNBlockedPrefillDispatchGrid derives the Metal grid layout for blocked prefill.
func BuildGDNBlockedPrefillDispatchGrid(cfg GDNBlockedPrefillConfig) GDNBlockedPrefillDispatchGrid {
	return GDNBlockedPrefillDispatchGrid{
		ThreadgroupsPerGrid:   [3]int{cfg.ValueHeadDim / cfg.DimBlockSize, cfg.NumValueHeads, cfg.BatchSize},
		ThreadsPerThreadgroup: [3]int{256, 1, 1},
	}
}

// GDNBlockedPrefill executes the blocked-sequential Gated DeltaNet prefill kernel on Metal.
func GDNBlockedPrefill(
	cfg GDNBlockedPrefillConfig,
	q, k, v, g, beta, initialState []float32,
) (output []float32, finalState []float32, stats GDNBlockedPrefillStats, err error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, GDNBlockedPrefillStats{}, err
	}
	if err := ensureGDNBlockedPrefillPipeline(); err != nil {
		return nil, nil, GDNBlockedPrefillStats{}, err
	}

	expectedQ := cfg.BatchSize * cfg.NumTokens * cfg.NumKeyHeads * cfg.KeyHeadDim
	expectedV := cfg.BatchSize * cfg.NumTokens * cfg.NumValueHeads * cfg.ValueHeadDim
	expectedGB := cfg.BatchSize * cfg.NumTokens * cfg.NumValueHeads
	stateElems := cfg.BatchSize * cfg.NumValueHeads * cfg.ValueHeadDim * cfg.KeyHeadDim

	if len(q) < expectedQ || len(k) < expectedQ {
		return nil, nil, GDNBlockedPrefillStats{}, fmt.Errorf("q/k length (%d/%d) smaller than expected %d", len(q), len(k), expectedQ)
	}
	if len(v) < expectedV {
		return nil, nil, GDNBlockedPrefillStats{}, fmt.Errorf("v length (%d) smaller than expected %d", len(v), expectedV)
	}
	if len(g) < expectedGB || len(beta) < expectedGB {
		return nil, nil, GDNBlockedPrefillStats{}, fmt.Errorf("g/beta length (%d/%d) smaller than expected %d", len(g), len(beta), expectedGB)
	}

	output = make([]float32, expectedV)
	finalState = make([]float32, stateElems)

	hasStateIn := C.int(0)
	var stateInPtr *C.float
	if len(initialState) >= stateElems {
		hasStateIn = C.int(1)
		stateInPtr = (*C.float)(unsafe.Pointer(&initialState[0]))
	}

	var gpuMs C.float
	status := C.mg_gdn_blocked_prefill_run(
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		(*C.float)(unsafe.Pointer(&g[0])),
		(*C.float)(unsafe.Pointer(&beta[0])),
		stateInPtr,
		(*C.float)(unsafe.Pointer(&finalState[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(cfg.BatchSize),
		C.int(cfg.NumTokens),
		C.int(cfg.NumKeyHeads),
		C.int(cfg.NumValueHeads),
		C.int(cfg.KeyHeadDim),
		C.int(cfg.ValueHeadDim),
		hasStateIn,
		C.int(1),
		&gpuMs,
	)

	if status != 0 {
		return nil, nil, GDNBlockedPrefillStats{}, fmt.Errorf("metalgemm: GDN blocked prefill kernel execution failed (code %d)", int(status))
	}

	// Memory savings calculation:
	// Unblocked kernel reads k/q once per Dv-slice (Dv reads total).
	// Blocked kernel reads k/q once per DB block (Dv/DB reads total).
	// Cut factor = DB = 32x.
	cutFactor := float64(cfg.DimBlockSize)
	scalarReads := int64(cfg.ValueHeadDim) * int64(cfg.BatchSize*cfg.NumTokens*cfg.NumKeyHeads*cfg.KeyHeadDim) * 4 * 2
	blockedReads := int64(cfg.ValueHeadDim/cfg.DimBlockSize) * int64(cfg.BatchSize*cfg.NumTokens*cfg.NumKeyHeads*cfg.KeyHeadDim) * 4 * 2
	savings := scalarReads - blockedReads

	stats = GDNBlockedPrefillStats{
		Tokens:               cfg.NumTokens,
		TokenBlockSize:       cfg.TokenBlockSize,
		DimBlockSize:         cfg.DimBlockSize,
		DRAMReadCutFactor:    cutFactor,
		RedundantReadSavings: savings,
		GPUExecuteMs:         float64(gpuMs),
	}

	return output, finalState, stats, nil
}

// GDNBlockedPrefillCPU provides the exact ground-truth reference implementation
// of the Gated DeltaNet sequential recurrence for parity comparison.
func GDNBlockedPrefillCPU(
	cfg GDNBlockedPrefillConfig,
	q, k, v, g, beta, initialState []float32,
) (output []float32, finalState []float32, err error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}

	expectedQ := cfg.BatchSize * cfg.NumTokens * cfg.NumKeyHeads * cfg.KeyHeadDim
	expectedV := cfg.BatchSize * cfg.NumTokens * cfg.NumValueHeads * cfg.ValueHeadDim
	expectedGB := cfg.BatchSize * cfg.NumTokens * cfg.NumValueHeads
	stateElems := cfg.BatchSize * cfg.NumValueHeads * cfg.ValueHeadDim * cfg.KeyHeadDim

	if len(q) < expectedQ || len(k) < expectedQ {
		return nil, nil, fmt.Errorf("q/k length smaller than expected %d", expectedQ)
	}
	if len(v) < expectedV {
		return nil, nil, fmt.Errorf("v length smaller than expected %d", expectedV)
	}
	if len(g) < expectedGB || len(beta) < expectedGB {
		return nil, nil, fmt.Errorf("g/beta length smaller than expected %d", expectedGB)
	}

	output = make([]float32, expectedV)
	finalState = make([]float32, stateElems)
	if len(initialState) >= stateElems {
		copy(finalState, initialState[:stateElems])
	}

	repeat := cfg.NumValueHeads / cfg.NumKeyHeads

	for b := 0; b < cfg.BatchSize; b++ {
		for hv := 0; hv < cfg.NumValueHeads; hv++ {
			hk := hv / repeat
			for t := 0; t < cfg.NumTokens; t++ {
				gVal := g[(b*cfg.NumTokens+t)*cfg.NumValueHeads+hv]
				betaVal := beta[(b*cfg.NumTokens+t)*cfg.NumValueHeads+hv]

				kOffset := ((b*cfg.NumTokens+t)*cfg.NumKeyHeads + hk) * cfg.KeyHeadDim
				qOffset := kOffset
				kt := k[kOffset : kOffset+cfg.KeyHeadDim]
				qt := q[qOffset : qOffset+cfg.KeyHeadDim]

				vOffset := ((b*cfg.NumTokens+t)*cfg.NumValueHeads + hv) * cfg.ValueHeadDim
				vt := v[vOffset : vOffset+cfg.ValueHeadDim]
				outOffset := vOffset
				ot := output[outOffset : outOffset+cfg.ValueHeadDim]

				for dv := 0; dv < cfg.ValueHeadDim; dv++ {
					stOffset := ((b*cfg.NumValueHeads+hv)*cfg.ValueHeadDim + dv) * cfg.KeyHeadDim
					st := finalState[stOffset : stOffset+cfg.KeyHeadDim]

					// 1. Decay
					for i := 0; i < cfg.KeyHeadDim; i++ {
						st[i] *= gVal
					}

					// 2. kvmem
					var kvmem float32
					for i := 0; i < cfg.KeyHeadDim; i++ {
						kvmem += st[i] * kt[i]
					}

					// 3. Delta
					delta := (vt[dv] - kvmem) * betaVal

					// 4. Update state
					for i := 0; i < cfg.KeyHeadDim; i++ {
						st[i] += delta * kt[i]
					}

					// 5. Readout
					var outVal float32
					for i := 0; i < cfg.KeyHeadDim; i++ {
						outVal += st[i] * qt[i]
					}
					ot[dv] = outVal
				}
			}
		}
	}

	return output, finalState, nil
}

// GDNCosineSimilarity computes the cosine similarity between two float32 slices.
func GDNCosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, sumA, sumB float64
	for i := range a {
		va, vb := float64(a[i]), float64(b[i])
		dot += va * vb
		sumA += va * va
		sumB += vb * vb
	}
	denom := math.Sqrt(sumA * sumB)
	if denom == 0 {
		return 1.0
	}
	return dot / denom
}

// gdnArgmaxMatch checks that the argmax along the value head dimension matches
// between test and reference outputs for every token and head.
func gdnArgmaxMatch(cfg GDNBlockedPrefillConfig, got, want []float32) (mismatches int, total int) {
	totalTokensHeads := cfg.BatchSize * cfg.NumTokens * cfg.NumValueHeads
	for th := 0; th < totalTokensHeads; th++ {
		offset := th * cfg.ValueHeadDim
		gotRow := got[offset : offset+cfg.ValueHeadDim]
		wantRow := want[offset : offset+cfg.ValueHeadDim]

		gotMaxIdx, wantMaxIdx := 0, 0
		gotMaxVal, wantMaxVal := gotRow[0], wantRow[0]
		for d := 1; d < cfg.ValueHeadDim; d++ {
			if gotRow[d] > gotMaxVal {
				gotMaxVal = gotRow[d]
				gotMaxIdx = d
			}
			if wantRow[d] > wantMaxVal {
				wantMaxVal = wantRow[d]
				wantMaxIdx = d
			}
		}

		if gotMaxIdx != wantMaxIdx {
			mismatches++
		}
		total++
	}
	return mismatches, total
}
