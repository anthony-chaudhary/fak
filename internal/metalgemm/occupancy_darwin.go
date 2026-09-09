//go:build darwin && arm64 && cgo

package metalgemm

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation -framework IOKit

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

#ifndef kIOMainPortDefault
#define kIOMainPortDefault 0
#endif

static int mg_discover_gpu_cores(void) {
    int cores = 0;
    const char *serviceNames[] = {"AGXAccelerator", "IOAccelerator", NULL};
    for (int i = 0; serviceNames[i] != NULL && cores <= 0; i++) {
        CFMutableDictionaryRef matching = IOServiceMatching(serviceNames[i]);
        if (!matching) continue;

        io_iterator_t iterator = IO_OBJECT_NULL;
        kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
        if (kr != KERN_SUCCESS || !iterator) continue;

        io_service_t service;
        while ((service = IOIteratorNext(iterator)) != IO_OBJECT_NULL) {
            CFTypeRef prop = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
            if (!prop) {
                prop = IORegistryEntryCreateCFProperty(service, CFSTR("core-count"), kCFAllocatorDefault, 0);
            }
            if (prop) {
                if (CFGetTypeID(prop) == CFNumberGetTypeID()) {
                    int val = 0;
                    if (CFNumberGetValue((CFNumberRef)prop, kCFNumberIntType, &val) && val > 0) {
                        cores = val;
                    }
                }
                CFRelease(prop);
            }
            IOObjectRelease(service);
            if (cores > 0) break;
        }
        IOObjectRelease(iterator);
    }
    return cores;
}

typedef struct {
    uint32_t batch;
    uint32_t q_tokens;
    uint32_t kv_tokens;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t num_splits;
    uint32_t chunk_size;
    float scale;
    int32_t causal;
    int32_t sliding_window;
} SplitKVParams;

static id<MTLComputePipelineState> gSplitKVStage1PSO = nil;
static id<MTLComputePipelineState> gSplitKVStage2PSO = nil;
static dispatch_once_t gSplitKVOnce;

static int mg_split_kv_init_pipeline(const char *src_str) {
    if (gSplitKVStage1PSO != nil && gSplitKVStage2PSO != nil) return 1;
    if (!mg_init()) return 0;

    dispatch_once(&gSplitKVOnce, ^{
        @autoreleasepool {
            NSError *error = nil;
            NSString *src = [NSString stringWithUTF8String:src_str];
            id<MTLLibrary> lib = [gDev newLibraryWithSource:src options:nil error:&error];
            if (lib == nil) {
                NSLog(@"mg_split_kv: library compile error: %@", error);
                return;
            }
            id<MTLFunction> fnStage1 = [lib newFunctionWithName:@"split_kv_decode_stage1"];
            if (fnStage1 == nil) {
                NSLog(@"mg_split_kv: split_kv_decode_stage1 function not found");
                return;
            }
            id<MTLFunction> fnStage2 = [lib newFunctionWithName:@"split_kv_decode_stage2"];
            if (fnStage2 == nil) {
                NSLog(@"mg_split_kv: split_kv_decode_stage2 function not found");
                return;
            }
            gSplitKVStage1PSO = [gDev newComputePipelineStateWithFunction:fnStage1 error:&error];
            if (gSplitKVStage1PSO == nil) {
                NSLog(@"mg_split_kv: stage1 PSO error: %@", error);
                return;
            }
            gSplitKVStage2PSO = [gDev newComputePipelineStateWithFunction:fnStage2 error:&error];
            if (gSplitKVStage2PSO == nil) {
                NSLog(@"mg_split_kv: stage2 PSO error: %@", error);
                return;
            }
        }
    });
    return (gSplitKVStage1PSO != nil && gSplitKVStage2PSO != nil) ? 1 : 0;
}

static int mg_split_kv_execute(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    int batch,
    int q_tokens,
    int kv_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int num_splits,
    int chunk_size,
    float scale,
    int causal,
    int sliding_window
) {
    if (q == NULL || k == NULL || v == NULL || out == NULL) return -1;
    if (batch <= 0 || q_tokens <= 0 || kv_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return -2;
    if (num_splits <= 0 || chunk_size <= 0) return -3;
    if (head_dim > 256) return -4;
    if (gSplitKVStage1PSO == nil || gSplitKVStage2PSO == nil) return -5;

    SplitKVParams params;
    params.batch = (uint32_t)batch;
    params.q_tokens = (uint32_t)q_tokens;
    params.kv_tokens = (uint32_t)kv_tokens;
    params.num_heads = (uint32_t)num_heads;
    params.num_kv_heads = (uint32_t)num_kv_heads;
    params.head_dim = (uint32_t)head_dim;
    params.num_splits = (uint32_t)num_splits;
    params.chunk_size = (uint32_t)chunk_size;
    params.scale = scale;
    params.causal = causal ? 1 : 0;
    params.sliding_window = sliding_window;

    size_t total_bq = (size_t)batch * q_tokens;
    size_t q_bytes = total_bq * num_heads * head_dim * sizeof(float);
    size_t kv_bytes = (size_t)batch * kv_tokens * num_kv_heads * head_dim * sizeof(float);
    size_t part_out_bytes = total_bq * num_heads * num_splits * head_dim * sizeof(float);
    size_t meta_bytes = total_bq * num_heads * num_splits * sizeof(float);
    size_t out_bytes = total_bq * num_heads * head_dim * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> bufQ = [gDev newBufferWithBytes:q length:q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [gDev newBufferWithBytes:k length:kv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [gDev newBufferWithBytes:v length:kv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPartOut = [gDev newBufferWithLength:part_out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPartMax = [gDev newBufferWithLength:meta_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPartSum = [gDev newBufferWithLength:meta_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [gDev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];

        if (!bufQ || !bufK || !bufV || !bufPartOut || !bufPartMax || !bufPartSum || !bufOut) {
            return -6;
        }

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return -7;

        // Stage 1: split_kv_decode_stage1
        id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];
        if (enc1 == nil) return -8;

        [enc1 setComputePipelineState:gSplitKVStage1PSO];
        [enc1 setBuffer:bufQ offset:0 atIndex:0];
        [enc1 setBuffer:bufK offset:0 atIndex:1];
        [enc1 setBuffer:bufV offset:0 atIndex:2];
        [enc1 setBuffer:bufPartOut offset:0 atIndex:3];
        [enc1 setBuffer:bufPartMax offset:0 atIndex:4];
        [enc1 setBuffer:bufPartSum offset:0 atIndex:5];
        [enc1 setBytes:&params length:sizeof(params) atIndex:6];

        MTLSize threads1 = MTLSizeMake(32, 1, 1);
        MTLSize threadgroups1 = MTLSizeMake(num_heads, total_bq, num_splits);
        [enc1 dispatchThreadgroups:threadgroups1 threadsPerThreadgroup:threads1];
        [enc1 endEncoding];

        // Stage 2: split_kv_decode_stage2
        id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
        if (enc2 == nil) return -9;

        [enc2 setComputePipelineState:gSplitKVStage2PSO];
        [enc2 setBuffer:bufPartOut offset:0 atIndex:0];
        [enc2 setBuffer:bufPartMax offset:0 atIndex:1];
        [enc2 setBuffer:bufPartSum offset:0 atIndex:2];
        [enc2 setBuffer:bufOut offset:0 atIndex:3];
        [enc2 setBytes:&params length:sizeof(params) atIndex:4];

        MTLSize threads2 = MTLSizeMake(32, 1, 1);
        MTLSize threadgroups2 = MTLSizeMake(num_heads, total_bq, 1);
        [enc2 dispatchThreadgroups:threadgroups2 threadsPerThreadgroup:threads2];
        [enc2 endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            NSLog(@"mg_split_kv: command buffer failed: %@", cb.error);
            return -10;
        }

        memcpy(out, [bufOut contents], out_bytes);
    }
    return 0;
}
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"
)

//go:embed flash_decode_split.metal
var flashDecodeSplitSource string

const (
	// MinSplitTokens is the minimum KV context length threshold for activating Split-KV.
	MinSplitTokens = 512
	// DefaultChunkSize is the default token chunk size for sequence partitioning.
	DefaultChunkSize = 512
	// MaxSplits is the upper clamp on the number of sequence splits.
	MaxSplits = 64
	// MinSplits is the lower clamp on the number of sequence splits.
	MinSplits = 1
	// DefaultGPUCores is the fallback core count if IORegistry is unavailable.
	DefaultGPUCores = 8
)

// SplitKVConfig defines configuration parameters for Split-KV FlashDecoding.
type SplitKVConfig struct {
	NumQueryHeads int     `json:"num_query_heads"`
	NumKVHeads    int     `json:"num_kv_heads"`
	HeadDim       int     `json:"head_dim"`
	Scale         float32 `json:"scale"`
	Causal        bool    `json:"causal"`
	SlidingWindow int     `json:"sliding_window,omitempty"`
	Batch         int     `json:"batch,omitempty"`
	ChunkSize     int     `json:"chunk_size,omitempty"`
	NumSplits     int     `json:"num_splits,omitempty"`
}

// SplitKVResult encapsulates the computed attention tensor and execution metadata.
type SplitKVResult struct {
	Output      []float32 `json:"output"`
	NumSplits   int       `json:"num_splits"`
	ChunkSize   int       `json:"chunk_size"`
	SplitActive bool      `json:"split_active"`
}

var (
	discoveredCoresOnce sync.Once
	discoveredCores     int
	testCoresOverride   int64 = 0

	splitKVPipelineOnce sync.Once
	splitKVPipelineOK   bool
)

// DiscoverGPUCoreCount queries the Darwin IORegistry (AGXAccelerator / IOAccelerator)
// for the hardware "gpu-core-count" property, returning the discovered count or fallback.
func DiscoverGPUCoreCount() int {
	discoveredCoresOnce.Do(func() {
		c := int(C.mg_discover_gpu_cores())
		if c <= 0 {
			c = DefaultGPUCores
		}
		discoveredCores = c
	})
	return discoveredCores
}

// GPUCoreCount returns the active GPU core count, honoring test overrides if set.
func GPUCoreCount() int {
	if ov := atomic.LoadInt64(&testCoresOverride); ov > 0 {
		return int(ov)
	}
	return DiscoverGPUCoreCount()
}

// SetGPUCoreCountForTesting overrides the GPU core count for testing, or resets when n <= 0.
func SetGPUCoreCountForTesting(n int) {
	atomic.StoreInt64(&testCoresOverride, int64(n))
}

// ShouldActivateSplitKV determines whether dynamic GPU occupancy gating activates Split-KV.
// Base grid: Grid_base = num_query_heads * batch.
// Rule: Grid_base < core_count * 8 and kv_tokens >= MinSplitTokens (512).
func ShouldActivateSplitKV(cores, numHeads, batch, kvTokens int) bool {
	if cores <= 0 {
		cores = GPUCoreCount()
	}
	if numHeads <= 0 {
		numHeads = 1
	}
	if batch <= 0 {
		batch = 1
	}
	gridBase := numHeads * batch
	return gridBase < cores*8 && kvTokens >= MinSplitTokens
}

// CalculateSplitCount calculates the split count S along the sequence dimension,
// chunked in chunkSize (default 512) tokens and clamped between 1 and 64.
func CalculateSplitCount(cores, numHeads, batch, kvTokens, chunkSize int) int {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	if kvTokens <= 0 {
		return MinSplits
	}
	s := (kvTokens + chunkSize - 1) / chunkSize
	if s < MinSplits {
		s = MinSplits
	}
	if s > MaxSplits {
		s = MaxSplits
	}
	return s
}

// SplitKVPipelineAvailable returns true if the Metal Split-KV pipelines (stage 1 and stage 2)
// are successfully compiled and available on the device.
func SplitKVPipelineAvailable() bool {
	if !Available() {
		return false
	}
	splitKVPipelineOnce.Do(func() {
		cSource := C.CString(flashDecodeSplitSource)
		defer C.free(unsafe.Pointer(cSource))
		splitKVPipelineOK = C.mg_split_kv_init_pipeline(cSource) == 1
	})
	return splitKVPipelineOK
}

// ExecuteSplitKVFlashDecode executes the two-stage Split-KV FlashDecoding kernel on Metal GPU.
func ExecuteSplitKVFlashDecode(cfg SplitKVConfig, q, k, v []float32, qTokens, kvTokens int) (SplitKVResult, error) {
	if !SplitKVPipelineAvailable() {
		return SplitKVResult{}, errors.New("metalgemm: Metal Split-KV FlashDecoding pipeline is unavailable")
	}

	batch := cfg.Batch
	if batch <= 0 {
		batch = 1
	}
	if qTokens <= 0 {
		qTokens = 1
	}
	if kvTokens <= 0 {
		return SplitKVResult{}, errors.New("metalgemm: kvTokens must be positive")
	}
	if cfg.NumQueryHeads <= 0 {
		return SplitKVResult{}, errors.New("metalgemm: NumQueryHeads must be positive")
	}
	if cfg.NumKVHeads <= 0 {
		cfg.NumKVHeads = cfg.NumQueryHeads
	}
	if cfg.NumQueryHeads%cfg.NumKVHeads != 0 {
		return SplitKVResult{}, fmt.Errorf("metalgemm: NumQueryHeads (%d) not divisible by NumKVHeads (%d)",
			cfg.NumQueryHeads, cfg.NumKVHeads)
	}
	if cfg.HeadDim <= 0 || cfg.HeadDim > 256 {
		return SplitKVResult{}, fmt.Errorf("metalgemm: HeadDim %d must be in range 1..256", cfg.HeadDim)
	}

	expectedQ := batch * qTokens * cfg.NumQueryHeads * cfg.HeadDim
	expectedKV := batch * kvTokens * cfg.NumKVHeads * cfg.HeadDim
	if len(q) < expectedQ {
		return SplitKVResult{}, fmt.Errorf("metalgemm: Q slice length %d smaller than expected %d", len(q), expectedQ)
	}
	if len(k) < expectedKV {
		return SplitKVResult{}, fmt.Errorf("metalgemm: K slice length %d smaller than expected %d", len(k), expectedKV)
	}
	if len(v) < expectedKV {
		return SplitKVResult{}, fmt.Errorf("metalgemm: V slice length %d smaller than expected %d", len(v), expectedKV)
	}

	chunkSize := cfg.ChunkSize
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	numSplits := cfg.NumSplits
	if numSplits <= 0 {
		numSplits = CalculateSplitCount(GPUCoreCount(), cfg.NumQueryHeads, batch, kvTokens, chunkSize)
	}

	scale := cfg.Scale
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	}

	out := make([]float32, expectedQ)
	var causalInt C.int
	if cfg.Causal {
		causalInt = 1
	}

	rc := C.mg_split_kv_execute(
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(batch),
		C.int(qTokens),
		C.int(kvTokens),
		C.int(cfg.NumQueryHeads),
		C.int(cfg.NumKVHeads),
		C.int(cfg.HeadDim),
		C.int(numSplits),
		C.int(chunkSize),
		C.float(scale),
		causalInt,
		C.int(cfg.SlidingWindow),
	)
	if rc != 0 {
		return SplitKVResult{}, fmt.Errorf("metalgemm: mg_split_kv_execute failed with code %d", int(rc))
	}

	occupancy := ShouldActivateSplitKV(GPUCoreCount(), cfg.NumQueryHeads, batch, kvTokens)
	return SplitKVResult{
		Output:      out,
		NumSplits:   numSplits,
		ChunkSize:   chunkSize,
		SplitActive: occupancy,
	}, nil
}

// ReferenceSplitKVDecode computes reference Split-KV FlashDecoding on CPU for mathematical parity validation.
func ReferenceSplitKVDecode(cfg SplitKVConfig, q, k, v []float32, qTokens, kvTokens int) (SplitKVResult, error) {
	batch := cfg.Batch
	if batch <= 0 {
		batch = 1
	}
	if qTokens <= 0 {
		qTokens = 1
	}
	if kvTokens <= 0 {
		return SplitKVResult{}, errors.New("metalgemm: kvTokens must be positive")
	}
	if cfg.NumQueryHeads <= 0 {
		return SplitKVResult{}, errors.New("metalgemm: NumQueryHeads must be positive")
	}
	if cfg.NumKVHeads <= 0 {
		cfg.NumKVHeads = cfg.NumQueryHeads
	}
	if cfg.NumQueryHeads%cfg.NumKVHeads != 0 {
		return SplitKVResult{}, fmt.Errorf("metalgemm: NumQueryHeads (%d) not divisible by NumKVHeads (%d)",
			cfg.NumQueryHeads, cfg.NumKVHeads)
	}
	if cfg.HeadDim <= 0 {
		return SplitKVResult{}, errors.New("metalgemm: HeadDim must be positive")
	}

	nH := cfg.NumQueryHeads
	nKV := cfg.NumKVHeads
	hd := cfg.HeadDim
	scale := cfg.Scale
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(hd)))
	}
	grp := nH / nKV

	expectedQ := batch * qTokens * nH * hd
	expectedKV := batch * kvTokens * nKV * hd
	if len(q) < expectedQ || len(k) < expectedKV || len(v) < expectedKV {
		return SplitKVResult{}, errors.New("metalgemm: slice bounds insufficient for ReferenceSplitKVDecode")
	}

	chunkSize := cfg.ChunkSize
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	numSplits := cfg.NumSplits
	if numSplits <= 0 {
		numSplits = CalculateSplitCount(GPUCoreCount(), nH, batch, kvTokens, chunkSize)
	}

	out := make([]float32, expectedQ)
	occupancy := ShouldActivateSplitKV(GPUCoreCount(), nH, batch, kvTokens)

	for b := 0; b < batch; b++ {
		for qi := 0; qi < qTokens; qi++ {
			globalQPos := qi
			if kvTokens >= qTokens {
				globalQPos = (kvTokens - qTokens) + qi
			}

			for h := 0; h < nH; h++ {
				kvh := h / grp
				qOffset := (b*qTokens+qi)*(nH*hd) + h*hd
				qRow := q[qOffset : qOffset+hd]

				// Stage 1: Partitioned local softmax per split
				partialMax := make([]float32, numSplits)
				partialSum := make([]float32, numSplits)
				partialAcc := make([][]float32, numSplits)

				for s := 0; s < numSplits; s++ {
					kStart := s * chunkSize
					kEnd := (s + 1) * chunkSize
					if kEnd > kvTokens {
						kEnd = kvTokens
					}
					if kStart >= kvTokens {
						partialMax[s] = float32(math.Inf(-1))
						partialSum[s] = 0
						partialAcc[s] = make([]float32, hd)
						continue
					}

					m_s := float32(math.Inf(-1))
					l_s := float32(0)
					acc_s := make([]float32, hd)

					for kj := kStart; kj < kEnd; kj++ {
						if cfg.Causal && kj > globalQPos {
							continue
						}
						if cfg.SlidingWindow > 0 && kj+cfg.SlidingWindow <= globalQPos {
							continue
						}

						kOffset := (b*kvTokens+kj)*(nKV*hd) + kvh*hd
						kRow := k[kOffset : kOffset+hd]
						var dot float32
						for d := 0; d < hd; d++ {
							dot += qRow[d] * kRow[d]
						}
						score := dot * scale

						if score > m_s {
							var alpha float32
							if m_s > float32(math.Inf(-1)) {
								alpha = float32(math.Exp(float64(m_s - score)))
							}
							pVal := float32(1.0)
							l_s = l_s*alpha + pVal
							m_s = score

							vOffset := (b*kvTokens+kj)*(nKV*hd) + kvh*hd
							vRow := v[vOffset : vOffset+hd]
							for d := 0; d < hd; d++ {
								acc_s[d] = acc_s[d]*alpha + pVal*vRow[d]
							}
						} else {
							pVal := float32(math.Exp(float64(score - m_s)))
							l_s += pVal

							vOffset := (b*kvTokens+kj)*(nKV*hd) + kvh*hd
							vRow := v[vOffset : vOffset+hd]
							for d := 0; d < hd; d++ {
								acc_s[d] += pVal * vRow[d]
							}
						}
					}
					partialMax[s] = m_s
					partialSum[s] = l_s
					partialAcc[s] = acc_s
				}

				// Stage 2: Reduction across S splits
				globalM := float32(math.Inf(-1))
				for s := 0; s < numSplits; s++ {
					if partialMax[s] > globalM {
						globalM = partialMax[s]
					}
				}

				var globalL float32
				if globalM > float32(math.Inf(-1)) {
					for s := 0; s < numSplits; s++ {
						if partialMax[s] > float32(math.Inf(-1)) && partialSum[s] > 0 {
							globalL += partialSum[s] * float32(math.Exp(float64(partialMax[s]-globalM)))
						}
					}
				}

				outOffset := (b*qTokens+qi)*(nH*hd) + h*hd
				if globalL > 0 {
					invL := 1.0 / globalL
					for s := 0; s < numSplits; s++ {
						if partialMax[s] > float32(math.Inf(-1)) {
							factor := float32(math.Exp(float64(partialMax[s]-globalM))) * invL
							for d := 0; d < hd; d++ {
								out[outOffset+d] += partialAcc[s][d] * factor
							}
						}
					}
				}
			}
		}
	}

	return SplitKVResult{
		Output:      out,
		NumSplits:   numSplits,
		ChunkSize:   chunkSize,
		SplitActive: occupancy,
	}, nil
}
