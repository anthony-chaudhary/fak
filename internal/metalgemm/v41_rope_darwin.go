//go:build darwin && arm64 && cgo

package metalgemm

// The frequency construction and dispatch geometry below are adapted from
// antirez/ds4 ds4_metal.m@bd66c402070042bf0a79ad6ece8242de4c93680c
// under the MIT notice reproduced in v41_rope.metal.

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

typedef struct {
    uint32_t width, heads, rows, start, inverse, stride;
    float frequencies[32];
} fak_v41_rope_args;

static id<MTLComputePipelineState> fakV41RoPEPipeline = nil;

static int fak_v41_rope_run(const char *source, float *values, uint32_t width,
        uint32_t heads, uint32_t rows, uint32_t start, uint32_t stride,
        int compressed, int inverse) {
    if (!source || !values || width < 64 || !heads || !rows || !stride ||
        rows > 1048576 || (uint64_t)start + (uint64_t)(rows - 1) * stride >= 1048576 ||
        (uint64_t)width * heads * rows > SIZE_MAX / sizeof(float)) return 0;
    if (!mg_init() || !gDev || !gQueue) return 0;

    @autoreleasepool {
        if (!fakV41RoPEPipeline) {
            NSError *error = nil;
            NSString *src = [NSString stringWithUTF8String:source];
            id<MTLLibrary> library = [gDev newLibraryWithSource:src options:nil error:&error];
            if (!library) return 0;
            id<MTLFunction> function = [library newFunctionWithName:@"fak_v41_rope"];
            if (!function) return 0;
            fakV41RoPEPipeline = [gDev newComputePipelineStateWithFunction:function error:&error];
            if (!fakV41RoPEPipeline) return 0;
        }

        fak_v41_rope_args args = {
            .width = width, .heads = heads, .rows = rows, .start = start,
            .inverse = inverse != 0, .stride = stride,
        };
        const float base = compressed ? 160000.0f : 10000.0f;
        const float low = floorf(64.0f * logf(65536.0f / (32.0f * 2.0f * (float)M_PI)) / (2.0f * logf(base)));
        const float high = ceilf(64.0f * logf(65536.0f / (2.0f * (float)M_PI)) / (2.0f * logf(base)));
        for (int i = 0; i < 32; i++) {
            const float denominator = powf(base, (float)i / 32.0f);
            float f = 1.0f / denominator;
            if (compressed) {
                const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                const float smooth = 1.0f - ramp;
                f = (f / 16.0f) * (1.0f - smooth) + f * smooth;
            }
            args.frequencies[i] = f;
        }

        const NSUInteger bytes = (NSUInteger)width * heads * rows * sizeof(float);
        id<MTLBuffer> buffer = [gDev newBufferWithBytes:values length:bytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!buffer || !command || !encoder) return 0;
        [encoder setComputePipelineState:fakV41RoPEPipeline];
        [encoder setBytes:&args length:sizeof(args) atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(heads, rows, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(values, buffer.contents, bytes);
        return 1;
    }
}
*/
import "C"

import (
	_ "embed"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

//go:embed v41_rope.metal
var deepSeekV41RoPESource string

var deepSeekV41RoPEMu sync.Mutex

// DeepSeekV41RoPEReceipt binds this component witness to fak's native Metal path.
type DeepSeekV41RoPEReceipt struct {
	Engine         string `json:"engine"`
	Backend        string `json:"backend"`
	CommandBuffers int    `json:"command_buffers"`
	CPUFallbacks   int    `json:"cpu_fallbacks"`
}

// DeepSeekV41RoPE applies V4.1's unit-magnitude RoPE to the final 64 dimensions
// of each head. Outputs are rounded to BF16, matching the published checkpoint.
func DeepSeekV41RoPE(values []float32, width, heads, rows, start, stride int, compressed, inverse bool) (DeepSeekV41RoPEReceipt, error) {
	receipt := DeepSeekV41RoPEReceipt{Engine: "fak-native", Backend: "metal"}
	if width < 64 || width > math.MaxUint32 || heads <= 0 || heads > math.MaxUint32 ||
		rows <= 0 || rows > 1048576 || start < 0 || start > math.MaxUint32 ||
		stride <= 0 || stride > math.MaxUint32 {
		return receipt, fmt.Errorf("metalgemm: invalid DeepSeek V4.1 RoPE geometry")
	}
	last := uint64(start) + uint64(rows-1)*uint64(stride)
	if last >= 1048576 || heads > math.MaxInt/width || rows > math.MaxInt/(width*heads) || len(values) != width*heads*rows {
		return receipt, fmt.Errorf("metalgemm: invalid DeepSeek V4.1 RoPE geometry")
	}
	deepSeekV41RoPEMu.Lock()
	defer deepSeekV41RoPEMu.Unlock()
	source := C.CString(deepSeekV41RoPESource)
	defer C.free(unsafe.Pointer(source))
	compressedInt, inverseInt := 0, 0
	if compressed {
		compressedInt = 1
	}
	if inverse {
		inverseInt = 1
	}
	if C.fak_v41_rope_run(source, (*C.float)(unsafe.Pointer(&values[0])), C.uint32_t(width), C.uint32_t(heads), C.uint32_t(rows), C.uint32_t(start), C.uint32_t(stride), C.int(compressedInt), C.int(inverseInt)) == 0 {
		return receipt, fmt.Errorf("metalgemm: DeepSeek V4.1 Metal RoPE dispatch failed")
	}
	receipt.CommandBuffers = 1
	return receipt, nil
}
