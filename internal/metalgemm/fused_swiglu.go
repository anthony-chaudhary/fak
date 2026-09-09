// Prior-art: mlc-ai/mlc-llm:compiler_pass:FuseDequantizeMatmulEwise, llama.cpp Metal
//go:build darwin && arm64 && cgo

// Package metalgemm provides Metal acceleration kernels for quantized matrix operations.
// fused_swiglu.go implements a fused Dequantize-GEMV-SwiGLU compute pipeline for Qwen MLP blocks.
// Study Provenance: mlc-ai/mlc-llm:compiler_pass:FuseDequantizeMatmulEwise
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

// External helper from q4k.m for Q4_K inspection via command encoder interception
int mg_issue8833_q4k_encode_gemv(void* command, int wid, void* x, void* y);
void *mg_graph_encode_q6k_from(void *opaque, int wid, void *input, int elems);

// External helpers from q8.m
id<MTLBuffer> mg_q8_codes_buf(int wid);
id<MTLBuffer> mg_q8_scales_buf(int wid);
void mg_q8_dims(int wid, int* out, int* in, int* nblk);

// Fallback GEMV if needed
void mg_q6k_gemv(int wid, const float* x, float* y, void* event);

@interface MGQ4KSpyEncoder : NSObject
@property (nonatomic, strong) id<MTLBuffer> buf;
@property (nonatomic, assign) NSUInteger offset;
@property (nonatomic, assign) int nblk;
@property (nonatomic, assign) int out;
@end

@implementation MGQ4KSpyEncoder
- (void)setComputePipelineState:(id)pso {}
- (void)setBuffer:(id)buf offset:(NSUInteger)offset atIndex:(NSUInteger)idx {
    if (idx == 0) {
        self.buf = buf;
        self.offset = offset;
    }
}
- (void)setBytes:(const void *)bytes length:(NSUInteger)len atIndex:(NSUInteger)idx {
    if (idx == 3 && len >= sizeof(int)) self.nblk = *(const int *)bytes;
    if (idx == 4 && len >= sizeof(int)) self.out = *(const int *)bytes;
}
- (void)dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt {}
- (void)endEncoding {}
@end

@interface MGQ4KSpyCmdBuf : NSObject
@property (nonatomic, strong) MGQ4KSpyEncoder *encoder;
@end

@implementation MGQ4KSpyCmdBuf
- (id)computeCommandEncoder {
    return self.encoder;
}
@end

static id<MTLBuffer> sDummyBuf = nil;

static int extract_q4k(int wid, id<MTLBuffer> *outBuf, NSUInteger *outOffset, int *outNblk, int *outOut) {
    if (sDummyBuf == nil) {
        sDummyBuf = [gDev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    }
    MGQ4KSpyEncoder *enc = [[MGQ4KSpyEncoder alloc] init];
    MGQ4KSpyCmdBuf *cmd = [[MGQ4KSpyCmdBuf alloc] init];
    cmd.encoder = enc;

    int rc = mg_issue8833_q4k_encode_gemv((__bridge void*)cmd, wid, (__bridge void*)sDummyBuf, (__bridge void*)sDummyBuf);
    if (rc && enc.buf != nil) {
        *outBuf = enc.buf;
        *outOffset = enc.offset;
        *outNblk = enc.nblk;
        *outOut = enc.out;
        return 1;
    }
    return 0;
}

typedef struct {
    id<MTLCommandBuffer> cb;
    id<MTLBuffer> xf, xq, xd;
    NSMutableArray *results;
    int P, in, encoders, committed, readbacks, buffers;
    double gpu_ms, wait_ms;
} FusedProjectionGraph;

static int extract_q6k(int wid, int in, id<MTLBuffer> *outBuf, int *outNblk, int *outOut) {
    MGQ4KSpyEncoder *enc = [[MGQ4KSpyEncoder alloc] init];
    MGQ4KSpyCmdBuf *cmd = [[MGQ4KSpyCmdBuf alloc] init];
    cmd.encoder = enc;

    FusedProjectionGraph g;
    memset((void*)&g, 0, sizeof(g));
    g.P = 1;
    g.in = in;
    g.cb = (id<MTLCommandBuffer>)cmd;
    id<MTLBuffer> dummy = [gDev newBufferWithLength:(NSUInteger)in * sizeof(float) options:MTLResourceStorageModeShared];
    g.results = [NSMutableArray arrayWithObject:dummy];

    void *res = mg_graph_encode_q6k_from(&g, wid, (__bridge void*)dummy, in);
    if (res != NULL && enc.buf != nil) {
        *outBuf = enc.buf;
        *outNblk = enc.nblk;
        *outOut = enc.out;
        return 1;
    }
    return 0;
}

static id<MTLComputePipelineState> psoQ4KFusedGemvSwiGLU = nil;
static id<MTLComputePipelineState> psoQ8FusedGemvSwiGLU = nil;
static id<MTLComputePipelineState> psoQ8FusedGemvSwiGLUQuant = nil;
static id<MTLComputePipelineState> psoQ6KGemv = nil;
static id<MTLComputePipelineState> psoQ4KGemv = nil;
static int gFusedSwiGLUReady = 0;

int mg_fused_swiglu_init_pipeline(const char *src_str) {
    if (gFusedSwiGLUReady && psoQ4KFusedGemvSwiGLU != nil && psoQ8FusedGemvSwiGLU != nil) {
        return 1;
    }
    if (!mg_init() || gDev == nil || gQueue == nil) {
        return 0;
    }
    @autoreleasepool {
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:src_str];
        id<MTLLibrary> lib = [gDev newLibraryWithSource:src options:nil error:&err];
        if (lib == nil) {
            NSLog(@"fused_swiglu: library compile failed: %@", err);
            return 0;
        }
        id<MTLFunction> fnQ4K = [lib newFunctionWithName:@"q4k_fused_gemv_swiglu"];
        id<MTLFunction> fnQ8  = [lib newFunctionWithName:@"q8_fused_gemv_swiglu"];
        id<MTLFunction> fnQ8Q = [lib newFunctionWithName:@"q8_fused_gemv_swiglu_quant"];
        id<MTLFunction> fnQ6  = [lib newFunctionWithName:@"q6k_gemv"];
        id<MTLFunction> fnQ4  = [lib newFunctionWithName:@"q4k_gemv"];
        if (fnQ4K == nil || fnQ8 == nil) {
            NSLog(@"fused_swiglu: required kernel function not found in library");
            return 0;
        }
        psoQ4KFusedGemvSwiGLU = [gDev newComputePipelineStateWithFunction:fnQ4K error:&err];
        psoQ8FusedGemvSwiGLU  = [gDev newComputePipelineStateWithFunction:fnQ8  error:&err];
        if (fnQ8Q) psoQ8FusedGemvSwiGLUQuant = [gDev newComputePipelineStateWithFunction:fnQ8Q error:nil];
        if (fnQ6)  psoQ6KGemv                 = [gDev newComputePipelineStateWithFunction:fnQ6  error:nil];
        if (fnQ4)  psoQ4KGemv                 = [gDev newComputePipelineStateWithFunction:fnQ4  error:nil];

        if (psoQ4KFusedGemvSwiGLU == nil || psoQ8FusedGemvSwiGLU == nil) {
            NSLog(@"fused_swiglu: compute pipeline creation failed: %@", err);
            return 0;
        }
        gFusedSwiGLUReady = 1;
        return 1;
    }
}

static id<MTLBuffer> gFusedXBuf = nil;
static long gFusedXCap = 0;
static id<MTLBuffer> gFusedInterBuf = nil;
static long gFusedInterCap = 0;
static id<MTLBuffer> gFusedYBuf = nil;
static long gFusedYCap = 0;

static void fused_grow_scratch(long xElems, long interElems, long yElems) {
    if (gFusedXBuf == nil || gFusedXCap < xElems) {
        gFusedXBuf = [gDev newBufferWithLength:(NSUInteger)(xElems * sizeof(float)) options:MTLResourceStorageModeShared];
        gFusedXCap = xElems;
    }
    if (gFusedInterBuf == nil || gFusedInterCap < interElems) {
        gFusedInterBuf = [gDev newBufferWithLength:(NSUInteger)(interElems * sizeof(float)) options:MTLResourceStorageModeShared];
        gFusedInterCap = interElems;
    }
    if (yElems > 0 && (gFusedYBuf == nil || gFusedYCap < yElems)) {
        gFusedYBuf = [gDev newBufferWithLength:(NSUInteger)(yElems * sizeof(float)) options:MTLResourceStorageModeShared];
        gFusedYCap = yElems;
    }
}

int mg_q4k_fused_gemv_swiglu(int gate_wid, int up_wid, const float* x, float* inter) {
    if (gate_wid < 0 || up_wid < 0 || x == NULL || inter == NULL) return 0;
    if (!gFusedSwiGLUReady || psoQ4KFusedGemvSwiGLU == nil) return 0;

    @autoreleasepool {
        id<MTLBuffer> gBuf = nil, uBuf = nil;
        NSUInteger gOffset = 0, uOffset = 0;
        int gNblk = 0, uNblk = 0, gOut = 0, uOut = 0;

        if (!extract_q4k(gate_wid, &gBuf, &gOffset, &gNblk, &gOut)) return 0;
        if (!extract_q4k(up_wid, &uBuf, &uOffset, &uNblk, &uOut)) return 0;
        if (gNblk != uNblk || gOut != uOut || gOut <= 0 || gNblk <= 0) return 0;

        long inElems = (long)gNblk * 256;
        long outElems = (long)gOut;
        fused_grow_scratch(inElems, outElems, 0);

        memcpy(gFusedXBuf.contents, x, (size_t)inElems * sizeof(float));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return 0;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) return 0;

        [enc setComputePipelineState:psoQ4KFusedGemvSwiGLU];
        [enc setBuffer:gBuf offset:gOffset atIndex:0];
        [enc setBuffer:uBuf offset:uOffset atIndex:1];
        [enc setBuffer:gFusedXBuf offset:0 atIndex:2];
        [enc setBuffer:gFusedInterBuf offset:0 atIndex:3];
        [enc setBytes:&gNblk length:sizeof(int) atIndex:4];
        [enc setBytes:&gOut length:sizeof(int) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)gOut, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        memcpy(inter, gFusedInterBuf.contents, (size_t)outElems * sizeof(float));
        return 1;
    }
}

int mg_q8_fused_gemv_swiglu(int gate_wid, int up_wid, const float* x, const float* xd, float* inter) {
    if (gate_wid < 0 || up_wid < 0 || x == NULL || inter == NULL) return 0;
    if (!gFusedSwiGLUReady || psoQ8FusedGemvSwiGLU == nil) return 0;

    @autoreleasepool {
        int gOut = 0, gIn = 0, gNblk = 0;
        int uOut = 0, uIn = 0, uNblk = 0;
        mg_q8_dims(gate_wid, &gOut, &gIn, &gNblk);
        mg_q8_dims(up_wid, &uOut, &uIn, &uNblk);
        if (gOut <= 0 || gIn <= 0 || gNblk <= 0 || gOut != uOut || gIn != uIn || gNblk != uNblk) return 0;

        id<MTLBuffer> gCodes = mg_q8_codes_buf(gate_wid);
        id<MTLBuffer> gScales = mg_q8_scales_buf(gate_wid);
        id<MTLBuffer> uCodes = mg_q8_codes_buf(up_wid);
        id<MTLBuffer> uScales = mg_q8_scales_buf(up_wid);
        if (gCodes == nil || gScales == nil || uCodes == nil || uScales == nil) return 0;

        long inElems = (long)gIn;
        long outElems = (long)gOut;
        fused_grow_scratch(inElems, outElems, 0);

        memcpy(gFusedXBuf.contents, x, (size_t)inElems * sizeof(float));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return 0;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) return 0;

        [enc setComputePipelineState:psoQ8FusedGemvSwiGLU];
        [enc setBuffer:gCodes offset:0 atIndex:0];
        [enc setBuffer:gScales offset:0 atIndex:1];
        [enc setBuffer:uCodes offset:0 atIndex:2];
        [enc setBuffer:uScales offset:0 atIndex:3];
        [enc setBuffer:gFusedXBuf offset:0 atIndex:4];
        [enc setBuffer:gFusedInterBuf offset:0 atIndex:5];
        [enc setBytes:&gNblk length:sizeof(int) atIndex:6];
        [enc setBytes:&gOut length:sizeof(int) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)gOut, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        memcpy(inter, gFusedInterBuf.contents, (size_t)outElems * sizeof(float));
        return 1;
    }
}

int mg_q4k_fused_mlp_q6down(int gate_wid, int up_wid, int down_wid, const float* x, float* y) {
    if (gate_wid < 0 || up_wid < 0 || down_wid < 0 || x == NULL || y == NULL) return 0;
    if (!gFusedSwiGLUReady || psoQ4KFusedGemvSwiGLU == nil) return 0;

    @autoreleasepool {
        id<MTLBuffer> gBuf = nil, uBuf = nil;
        NSUInteger gOffset = 0, uOffset = 0;
        int gNblk = 0, uNblk = 0, gOut = 0, uOut = 0;

        if (!extract_q4k(gate_wid, &gBuf, &gOffset, &gNblk, &gOut)) return 0;
        if (!extract_q4k(up_wid, &uBuf, &uOffset, &uNblk, &uOut)) return 0;
        if (gNblk != uNblk || gOut != uOut || gOut <= 0 || gNblk <= 0) return 0;

        long inElems = (long)gNblk * 256;
        long interElems = (long)gOut;

        id<MTLBuffer> dBuf = nil;
        int dNblk = 0, dOut = 0;
        int hasFusedDown = extract_q6k(down_wid, (int)interElems, &dBuf, &dNblk, &dOut);

        if (hasFusedDown && dBuf != nil && psoQ6KGemv != nil) {
            long yElems = (long)dOut;
            fused_grow_scratch(inElems, interElems, yElems);
            memcpy(gFusedXBuf.contents, x, (size_t)inElems * sizeof(float));

            id<MTLCommandBuffer> cb = [gQueue commandBuffer];
            if (cb == nil) return 0;

            // Stage 1: Fused Gate+Up GEMV + SwiGLU in registers -> gFusedInterBuf
            id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
            [e1 setComputePipelineState:psoQ4KFusedGemvSwiGLU];
            [e1 setBuffer:gBuf offset:gOffset atIndex:0];
            [e1 setBuffer:uBuf offset:uOffset atIndex:1];
            [e1 setBuffer:gFusedXBuf offset:0 atIndex:2];
            [e1 setBuffer:gFusedInterBuf offset:0 atIndex:3];
            [e1 setBytes:&gNblk length:sizeof(int) atIndex:4];
            [e1 setBytes:&gOut length:sizeof(int) atIndex:5];
            [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)gOut, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e1 endEncoding];

            // Stage 2: Q6_K Down projection GEMV: gFusedInterBuf -> gFusedYBuf
            id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
            [e2 setComputePipelineState:psoQ6KGemv];
            [e2 setBuffer:dBuf offset:0 atIndex:0];
            [e2 setBuffer:gFusedInterBuf offset:0 atIndex:1];
            [e2 setBuffer:gFusedYBuf offset:0 atIndex:2];
            [e2 setBytes:&dNblk length:sizeof(int) atIndex:3];
            [e2 setBytes:&dOut length:sizeof(int) atIndex:4];
            [e2 dispatchThreadgroups:MTLSizeMake((NSUInteger)dOut, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e2 endEncoding];

            [cb commit];
            [cb waitUntilCompleted];

            memcpy(y, gFusedYBuf.contents, (size_t)yElems * sizeof(float));
            return 1;
        }

        // Fallback if Q6_K down buffer extraction was declined: compute inter via fused swiglu,
        // then dispatch mg_q6k_gemv (still eliminates gMlpGate and gMlpUp!).
        fused_grow_scratch(inElems, interElems, 0);
        memcpy(gFusedXBuf.contents, x, (size_t)inElems * sizeof(float));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return 0;
        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        [e1 setComputePipelineState:psoQ4KFusedGemvSwiGLU];
        [e1 setBuffer:gBuf offset:gOffset atIndex:0];
        [e1 setBuffer:uBuf offset:uOffset atIndex:1];
        [e1 setBuffer:gFusedXBuf offset:0 atIndex:2];
        [e1 setBuffer:gFusedInterBuf offset:0 atIndex:3];
        [e1 setBytes:&gNblk length:sizeof(int) atIndex:4];
        [e1 setBytes:&gOut length:sizeof(int) atIndex:5];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)gOut, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e1 endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        mg_q6k_gemv(down_wid, (const float*)gFusedInterBuf.contents, y, NULL);
        return 1;
    }
}

int mg_q4k_fused_mlp(int gate_wid, int up_wid, int down_wid, const float* x, float* y) {
    return mg_q4k_fused_mlp_q6down(gate_wid, up_wid, down_wid, x, y);
}
*/
import "C"

import (
	_ "embed"
	"errors"
	"sync"
	"unsafe"
)

//go:embed fused_swiglu.metal
var fusedSwiGLUSource string

var (
	fusedInitOnce sync.Once
	fusedReady    bool
)

// EnsureFusedSwiGLUPipeline compiles and caches the fused Dequantize-GEMV-SwiGLU compute
// pipelines from embedded MSL source. Caches compute pipeline states and returns nil on success.
func EnsureFusedSwiGLUPipeline() error {
	if !Available() {
		return errors.New("metalgemm: metal device not available")
	}
	fusedInitOnce.Do(func() {
		cSrc := C.CString(fusedSwiGLUSource)
		defer C.free(unsafe.Pointer(cSrc))
		if C.mg_fused_swiglu_init_pipeline(cSrc) == 1 {
			fusedReady = true
		}
	})
	if !fusedReady {
		return errors.New("metalgemm: failed to initialize fused swiglu pipeline")
	}
	return nil
}

// FusedDequantGEMVSwiGLU evaluates fused Gate and Up Q4_K dequant-GEMVs and computes SwiGLU:
// inter[o] = silu(W_g*x) * (W_up*x)
// entirely in registers, writing directly to inter without global DRAM round-trips for separate
// Gate and Up activation tensors.
func FusedDequantGEMVSwiGLU(gate, up *Q4KWeight, x, inter []float32) bool {
	if !Available() {
		return false
	}
	if gate == nil || up == nil || gate.id < 0 || up.id < 0 {
		return false
	}
	if gate.In != up.In || gate.Out != up.Out || gate.In <= 0 || gate.Out <= 0 {
		return false
	}
	if len(x) < gate.In || len(inter) < gate.Out {
		return false
	}
	if err := EnsureFusedSwiGLUPipeline(); err != nil {
		return false
	}
	q4kExecutionMu.Lock()
	defer q4kExecutionMu.Unlock()

	rc := C.mg_q4k_fused_gemv_swiglu(
		gate.id,
		up.id,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&inter[0])),
	)
	return rc == 1
}

// FusedDequantGEMVSwiGLUQ8 evaluates fused Gate and Up Q8_0 dequant-GEMVs and computes SwiGLU:
// inter[o] = silu(W_g*x) * (W_up*x)
// entirely in registers. Accepts float32 activations with optional Q8 block scales xd.
func FusedDequantGEMVSwiGLUQ8(gate, up *Q8Weight, x, xd []float32, inter []float32) bool {
	if !Available() {
		return false
	}
	if gate == nil || up == nil || gate.id < 0 || up.id < 0 {
		return false
	}
	if gate.In != up.In || gate.Out != up.Out || gate.In <= 0 || gate.Out <= 0 {
		return false
	}
	if len(x) < gate.In || len(inter) < gate.Out {
		return false
	}
	if err := EnsureFusedSwiGLUPipeline(); err != nil {
		return false
	}
	q4kExecutionMu.Lock()
	defer q4kExecutionMu.Unlock()

	var xdPtr *C.float
	if len(xd) >= gate.Nblk {
		xdPtr = (*C.float)(unsafe.Pointer(&xd[0]))
	}

	rc := C.mg_q8_fused_gemv_swiglu(
		gate.id,
		up.id,
		(*C.float)(unsafe.Pointer(&x[0])),
		xdPtr,
		(*C.float)(unsafe.Pointer(&inter[0])),
	)
	return rc == 1
}

// FusedMLPQ6DownFast runs an end-to-end fused MLP block (Q4_K gate/up, Q6_K down) for one decode token:
// y = down( silu(gate*x) * (up*x) )
// dispatches the fused Gate+Up SwiGLU kernel directly followed by Down projection GEMV, skipping
// the creation and DRAM write/read of gMlpGate and gMlpUp activation tensors.
func FusedMLPQ6DownFast(gate, up *Q4KWeight, down *Q6KWeight, x, y []float32) bool {
	if !Available() {
		return false
	}
	q6kRegistryMu.RLock()
	defer q6kRegistryMu.RUnlock()
	if gate == nil || up == nil || down == nil || gate.id < 0 || up.id < 0 || !q6kWeightValidLocked(down) {
		return false
	}
	if gate.In != up.In || gate.Out != up.Out || down.In != gate.Out || down.Out != gate.In {
		return false
	}
	if len(x) < gate.In || len(y) < down.Out {
		return false
	}
	if err := EnsureFusedSwiGLUPipeline(); err != nil {
		return false
	}
	q4kExecutionMu.Lock()
	defer q4kExecutionMu.Unlock()

	rc := C.mg_q4k_fused_mlp_q6down(
		gate.id,
		up.id,
		down.id,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])),
	)
	return rc == 1
}

// FusedMLPFast is an alias to FusedMLPQ6DownFast for end-to-end fused decode MLP evaluation.
func FusedMLPFast(gate, up *Q4KWeight, down *Q6KWeight, x, y []float32) bool {
	return FusedMLPQ6DownFast(gate, up, down, x, y)
}

// FusedSwiGLUBandwidthSavingsBytes returns the count of global DRAM bytes eliminated
// by fusing weight dequantization and dual-projection GEMV into a single SwiGLU kernel.
// Eliminates writing separate Gate (I*4 bytes) and Up (I*4 bytes) activations to memory.
func FusedSwiGLUBandwidthSavingsBytes(intermediateDim int) int {
	return 2 * intermediateDim * 4
}
