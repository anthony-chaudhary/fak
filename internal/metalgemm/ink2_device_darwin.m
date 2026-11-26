//go:build darwin && arm64 && cgo

#import <Metal/Metal.h>
#include "ink2_device.h"

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

// The open command buffer and its single compute encoder collect every
// dispatch between two fences, so one token is one submission.
@interface MGInk2Device : NSObject {
@public
    NSMutableArray<id<MTLLibrary>> *libraries;
    NSMutableArray<id<MTLComputePipelineState>> *pipelines;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    NSString *error;
    mg_ink2_stats stats;
}
@end

@implementation MGInk2Device
@end

void *mg_ink2_create(void) {
    @autoreleasepool {
        if (!mg_init() || gDev == nil || gQueue == nil) return NULL;
        MGInk2Device *d = [[MGInk2Device alloc] init];
        d->libraries = [NSMutableArray array];
        d->pipelines = [NSMutableArray array];
        return (void *)CFBridgingRetain(d);
    }
}

void mg_ink2_close(void *dev) {
    if (!dev) return;
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        // Metal asserts when an encoder is released before endEncoding, even
        // if its command buffer is never committed.
        [d->encoder endEncoding];
        d->encoder = nil;
        d->command = nil;
    }
    CFRelease(dev);
}

void *mg_ink2_alloc(size_t bytes) {
    @autoreleasepool {
        if (!mg_init() || gDev == nil || bytes == 0) return NULL;
        id<MTLBuffer> b = [gDev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        return b ? (void *)CFBridgingRetain(b) : NULL;
    }
}

void mg_ink2_release(void *buffer) {
    if (buffer) CFRelease(buffer);
}

void *mg_ink2_span(void *buffer, size_t *bytes) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffer;
    *bytes = b.length;
    return b.contents;
}

static int ink2_fail(MGInk2Device *d, NSString *reason) {
    d->error = reason;
    return 0;
}

int mg_ink2_library(void *dev, const char *source, int precise) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        MTLCompileOptions *options = nil;
        if (precise) {
            options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            // mathMode replaces this on newer SDKs only; this spelling builds on all of them.
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [gDev newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                options:options error:&err];
        if (!lib) {
            ink2_fail(d, [NSString stringWithFormat:@"shader compile: %@", err.localizedDescription]);
            return -1;
        }
        [d->libraries addObject:lib];
        return (int)d->libraries.count - 1;
    }
}

int mg_ink2_pipeline(void *dev, int library, const char *function, uint32_t *width, uint32_t *max_threads) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        if (library < 0 || (NSUInteger)library >= d->libraries.count) {
            ink2_fail(d, @"unknown shader library");
            return -1;
        }
        NSString *name = [NSString stringWithUTF8String:function];
        id<MTLFunction> fn = [d->libraries[library] newFunctionWithName:name];
        if (!fn) {
            ink2_fail(d, [NSString stringWithFormat:@"no kernel named %@", name]);
            return -1;
        }
        NSError *err = nil;
        id<MTLComputePipelineState> p = [gDev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) {
            ink2_fail(d, [NSString stringWithFormat:@"pipeline %@: %@", name, err.localizedDescription]);
            return -1;
        }
        *width = (uint32_t)p.threadExecutionWidth;
        *max_threads = (uint32_t)p.maxTotalThreadsPerThreadgroup;
        [d->pipelines addObject:p];
        return (int)d->pipelines.count - 1;
    }
}

int mg_ink2_dispatch(void *dev, int pipeline, const mg_ink2_arg *args, int count,
                     const void *constants, const uint32_t *size, int exact) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        if (pipeline < 0 || (NSUInteger)pipeline >= d->pipelines.count) return ink2_fail(d, @"unknown pipeline");
        // Without the validation layer an out-of-range binding is silent
        // device memory corruption, so every extent is checked here.
        for (int i = 0; i < count; i++) {
            if (!args[i].buffer) continue;
            size_t length = ((__bridge id<MTLBuffer>)(void *)args[i].buffer).length;
            if (args[i].offset > length || args[i].length > length - args[i].offset)
                return ink2_fail(d, [NSString stringWithFormat:@"argument %d: %zu bytes at offset %zu exceed a %zu-byte buffer",
                                                               i, args[i].length, args[i].offset, length]);
        }
        if (!d->encoder) {
            d->command = [gQueue commandBuffer];
            d->encoder = [d->command computeCommandEncoder];
            if (!d->encoder) {
                d->command = nil;
                return ink2_fail(d, @"no compute encoder");
            }
        }
        id<MTLComputeCommandEncoder> enc = d->encoder;
        [enc setComputePipelineState:d->pipelines[pipeline]];
        for (int i = 0; i < count; i++) {
            if (args[i].buffer)
                [enc setBuffer:(__bridge id<MTLBuffer>)(void *)args[i].buffer offset:args[i].offset atIndex:i];
            else
                [enc setBytes:(const char *)constants + args[i].offset length:args[i].length atIndex:i];
        }
        MTLSize grid = MTLSizeMake(size[0], size[1], size[2]);
        MTLSize group = MTLSizeMake(size[3], size[4], size[5]);
        if (exact) [enc dispatchThreads:grid threadsPerThreadgroup:group];
        else [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
        d->stats.pending++;
        d->stats.dispatches++;
        return 1;
    }
}

void mg_ink2_barrier(void *dev) {
    MGInk2Device *d = (__bridge MGInk2Device *)dev;
    [d->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

int mg_ink2_flush(void *dev) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        if (!d->encoder) return 1;
        id<MTLCommandBuffer> cb = d->command;
        [d->encoder endEncoding];
        d->encoder = nil;
        d->command = nil;
        d->stats.pending = 0;
        d->stats.submissions++;
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
            return ink2_fail(d, [NSString stringWithFormat:@"command buffer: %@", cb.error.localizedDescription]);
        CFTimeInterval gpu = cb.GPUEndTime - cb.GPUStartTime;
        if (gpu > 0) d->stats.gpu_seconds += gpu;
        return 1;
    }
}

void mg_ink2_get_stats(void *dev, mg_ink2_stats *out) {
    *out = ((__bridge MGInk2Device *)dev)->stats;
}

size_t mg_ink2_error(void *dev, char *dst, size_t cap) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        if (!d->error || cap == 0) return 0;
        NSUInteger used = 0;
        [d->error getBytes:dst maxLength:cap usedLength:&used encoding:NSUTF8StringEncoding
                   options:NSStringEncodingConversionAllowLossy range:NSMakeRange(0, d->error.length) remainingRange:NULL];
        return used;
    }
}
