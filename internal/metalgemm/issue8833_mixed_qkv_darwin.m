//go:build darwin && arm64 && cgo

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreFoundation/CoreFoundation.h>
#include <string.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_issue8833_q8_encode_gemv(void*, int, void*, void*, void*);
int mg_issue8833_q4k_encode_gemv(void*, int, void*, void*);

typedef struct {
    uintptr_t command_buffer;
    int committed, completed_wait, host_readback, encoders;
    double gpu_milliseconds, wait_milliseconds;
    int timing_available;
} mg_execution_event;

typedef struct {
    mg_execution_event events[2];
    int event_count;
    int submitted;
} mg_issue8833_result;

static void issue8833_finish_event(mg_execution_event* e, id<MTLCommandBuffer> cb,
                                   int encoders, CFAbsoluteTime waitStart) {
    e->command_buffer = (uintptr_t)(__bridge void*)cb;
    e->committed = 1;
    e->completed_wait = 1;
    e->encoders = encoders;
    e->wait_milliseconds = (CFAbsoluteTimeGetCurrent() - waitStart) * 1000.0;
    if (cb.GPUStartTime > 0 && cb.GPUEndTime >= cb.GPUStartTime) {
        e->timing_available = 1;
        e->gpu_milliseconds = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    }
}

// Returns 0 on success, 1 for a pre-encoding decline, and 2 for a post-submit failure.
// Every Metal object below is call-owned and is released by the autorelease pool on every exit.
int mg_issue8833_mixed_qkv(int selector, int qwid, int kwid, int vwid,
                           const signed char* xq, const float* xd, const float* xf,
                           int hidden, int qout, int kout, int vout,
                           float* q, float* k, float* v, int inject_setup, int inject_post,
                           mg_issue8833_result* result) {
    memset(result, 0, sizeof(*result));
    if (inject_setup || gDev == nil || gQueue == nil || xq == NULL || xd == NULL || xf == NULL ||
        hidden != 4096 || qout != 8192 || kout != 1024 || vout != 1024 ||
        (selector != 1 && selector != 2)) return 1;
    @autoreleasepool {
        id<MTLBuffer> xbq = [gDev newBufferWithBytes:xq length:(NSUInteger)hidden options:MTLResourceStorageModeShared];
        id<MTLBuffer> xbd = [gDev newBufferWithBytes:xd length:(NSUInteger)(hidden/32)*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> xbf = [gDev newBufferWithBytes:xf length:(NSUInteger)hidden*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> qb = [gDev newBufferWithLength:(NSUInteger)qout*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> kb = [gDev newBufferWithLength:(NSUInteger)kout*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> vb = [gDev newBufferWithLength:(NSUInteger)vout*4 options:MTLResourceStorageModeShared];
        if (!xbq || !xbd || !xbf || !qb || !kb || !vb) return 1;

        id<MTLCommandBuffer> first = [gQueue commandBuffer];
        if (!first) return 1;
        if (!mg_issue8833_q8_encode_gemv((__bridge void*)first, qwid, (__bridge void*)xbq,
                                         (__bridge void*)xbd, (__bridge void*)qb) ||
            !mg_issue8833_q8_encode_gemv((__bridge void*)first, kwid, (__bridge void*)xbq,
                                         (__bridge void*)xbd, (__bridge void*)kb)) return 1;
        int firstEncoders = 2;
        if (selector == 2 && !mg_issue8833_q4k_encode_gemv((__bridge void*)first, vwid,
                                                            (__bridge void*)xbf, (__bridge void*)vb)) return 1;
        if (selector == 2) firstEncoders++;
        [first commit];
        result->submitted = 1;
        CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
        [first waitUntilCompleted];
        issue8833_finish_event(&result->events[0], first, firstEncoders, started);
        result->event_count = 1;
        if (first.status != MTLCommandBufferStatusCompleted || inject_post) return 2;

        if (selector == 1) {
            id<MTLCommandBuffer> second = [gQueue commandBuffer];
            // Failure here is post-submit: the owner has already submitted Q+K and must not decline.
            if (!second || !mg_issue8833_q4k_encode_gemv((__bridge void*)second, vwid,
                                                          (__bridge void*)xbf, (__bridge void*)vb)) return 2;
            [second commit];
            started = CFAbsoluteTimeGetCurrent();
            [second waitUntilCompleted];
            issue8833_finish_event(&result->events[1], second, 1, started);
            result->event_count = 2;
            if (second.status != MTLCommandBufferStatusCompleted) return 2;
        }
        memcpy(q, qb.contents, (size_t)qout*4);
        memcpy(k, kb.contents, (size_t)kout*4);
        memcpy(v, vb.contents, (size_t)vout*4);
        result->events[result->event_count-1].host_readback = 1;
        return 0;
    }
}
