//go:build darwin && arm64 && cgo

#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include "ink2_device.h"

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

// A byte range of one buffer that dispatches since the last barrier touched.
typedef struct {
    uintptr_t buffer;
    size_t lo, hi;
} ink2_span;

typedef struct {
    ink2_span *at;
    size_t n, cap;
} ink2_spans;

// The open command buffer and its single compute encoder collect every
// dispatch between two fences, so one token is one submission. A concurrent
// encoder gets a buffer barrier only before a dispatch that touches bytes an
// earlier, unbarriered dispatch wrote, or writes bytes one read.
@interface MGInk2Device : NSObject {
@public
    NSMutableArray<id<MTLLibrary>> *libraries;
    NSMutableArray<id<MTLComputePipelineState>> *pipelines;
    // Bit i of a pipeline's mask is set when the kernel may write buffer(i).
    uint64_t *writeMasks;
    size_t maskCap;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    BOOL concurrent;
    uintptr_t untracked;
    ink2_spans reads, writes;
    // Command buffers committed before the fence, oldest first, each waiting
    // on the previous one's event value.
    NSUInteger commitEvery, encoded;
    NSMutableArray<id<MTLCommandBuffer>> *inflight;
    id<MTLEvent> chain;
    uint64_t chainValue;
    double firstCommit;
    NSString *error;
    mg_ink2_stats stats;
    // Pass profiling: one compute pass per dispatch, two samples per pass.
    BOOL passProfile;
    id<MTLCounterSet> timestamps;
    NSMutableArray<id<MTLCounterSampleBuffer>> *samples;
    NSUInteger passes, flushedPasses;
    uint64_t gpu0;
    double host0;
}
@end

// Passes per sample buffer; a command buffer takes as many buffers as it needs.
enum { kInk2PassesPerBuffer = 512 };

static double ink2_now(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

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
        // Committed buffers must finish before the buffers they bind are freed.
        for (id<MTLCommandBuffer> cb in d->inflight) [cb waitUntilCompleted];
        d->inflight = nil;
        free(d->writeMasks);
        free(d->reads.at);
        free(d->writes.at);
        d->writeMasks = NULL;
        d->reads = d->writes = (ink2_spans){0};
    }
    CFRelease(dev);
}

void mg_ink2_configure(void *dev, int concurrent, int commit_every, void *untracked) {
    MGInk2Device *d = (__bridge MGInk2Device *)dev;
    d->concurrent = concurrent != 0;
    d->commitEvery = commit_every > 0 ? (NSUInteger)commit_every : 0;
    d->untracked = (uintptr_t)untracked;
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
        id<MTLComputePipelineState> p = nil;
        // Without reflection every binding counts as written, which barriers
        // like the serial encoder does.
        uint64_t mask = ~(uint64_t)0;
        if (@available(macOS 13.0, *)) {
            MTLComputePipelineReflection *refl = nil;
            p = [gDev newComputePipelineStateWithFunction:fn options:MTLPipelineOptionBindingInfo reflection:&refl error:&err];
            if (p && refl) {
                mask = 0;
                for (id<MTLBinding> b in refl.bindings) {
                    if (b.type != MTLBindingTypeBuffer || b.access == MTLBindingAccessReadOnly) continue;
                    mask |= b.index < 64 ? (uint64_t)1 << b.index : ~(uint64_t)0;
                }
            }
        } else {
            p = [gDev newComputePipelineStateWithFunction:fn error:&err];
        }
        if (!p) {
            ink2_fail(d, [NSString stringWithFormat:@"pipeline %@: %@", name, err.localizedDescription]);
            return -1;
        }
        size_t at = d->pipelines.count;
        if (at >= d->maskCap) {
            size_t cap = d->maskCap ? 2 * d->maskCap : 64;
            uint64_t *grown = realloc(d->writeMasks, cap * sizeof(*grown));
            if (!grown) {
                ink2_fail(d, @"pipeline table allocation");
                return -1;
            }
            d->writeMasks = grown;
            d->maskCap = cap;
        }
        d->writeMasks[at] = mask;
        *width = (uint32_t)p.threadExecutionWidth;
        *max_threads = (uint32_t)p.maxTotalThreadsPerThreadgroup;
        [d->pipelines addObject:p];
        return (int)at;
    }
}

int mg_ink2_profile_passes(void *dev) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        if (d->command) return 0;
        if (@available(macOS 11.0, *)) {
            if (![gDev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return 0;
            for (id<MTLCounterSet> set in gDev.counterSets)
                if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) d->timestamps = set;
            if (!d->timestamps) return 0;
            d->samples = [NSMutableArray array];
            uint64_t cpu = 0;
            [gDev sampleTimestamps:&cpu gpuTimestamp:&d->gpu0];
            d->host0 = ink2_now();
            d->passProfile = YES;
            return 1;
        }
        return 0;
    }
}

// The encoder of the next pass, sampling its start and end into the pooled
// sample buffers.
static id<MTLComputeCommandEncoder> ink2_pass(MGInk2Device *d) API_AVAILABLE(macos(11.0)) {
    NSUInteger buffer = d->passes / kInk2PassesPerBuffer, slot = d->passes % kInk2PassesPerBuffer;
    while (d->samples.count <= buffer) {
        MTLCounterSampleBufferDescriptor *desc = [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet = d->timestamps;
        desc.storageMode = MTLStorageModeShared;
        desc.sampleCount = 2 * kInk2PassesPerBuffer;
        NSError *err = nil;
        id<MTLCounterSampleBuffer> sb = [gDev newCounterSampleBufferWithDescriptor:desc error:&err];
        if (!sb) {
            ink2_fail(d, [NSString stringWithFormat:@"counter sample buffer: %@", err.localizedDescription]);
            return nil;
        }
        [d->samples addObject:sb];
    }
    MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
    pass.dispatchType = MTLDispatchTypeSerial;
    MTLComputePassSampleBufferAttachmentDescriptor *at = pass.sampleBufferAttachments[0];
    at.sampleBuffer = d->samples[buffer];
    at.startOfEncoderSampleIndex = 2 * slot;
    at.endOfEncoderSampleIndex = 2 * slot + 1;
    return [d->command computeCommandEncoderWithDescriptor:pass];
}

int mg_ink2_profile_take(void *dev, double *out, int cap) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        NSUInteger n = d->flushedPasses;
        d->flushedPasses = 0;
        if (n == 0 || (NSUInteger)cap < n) return (int)n;
        if (@available(macOS 11.0, *)) {
            // Timestamps are in a device unit, measured here against the host
            // clock since profiling began; nanoseconds until that is known.
            uint64_t cpu = 0, gpu = 0;
            [gDev sampleTimestamps:&cpu gpuTimestamp:&gpu];
            double elapsed = ink2_now() - d->host0, tick = 1e-9;
            if (gpu > d->gpu0 && elapsed > 1e-3) tick = elapsed / (double)(gpu - d->gpu0);
            for (NSUInteger done = 0; done < n; done += kInk2PassesPerBuffer) {
                NSUInteger count = MIN(n - done, (NSUInteger)kInk2PassesPerBuffer);
                NSData *data = [d->samples[done / kInk2PassesPerBuffer] resolveCounterRange:NSMakeRange(0, 2 * count)];
                const MTLCounterResultTimestamp *ts = data.bytes;
                for (NSUInteger i = 0; i < count; i++) {
                    BOOL ok = data.length >= 2 * count * sizeof(*ts) && ts[2 * i].timestamp != MTLCounterErrorValue &&
                              ts[2 * i + 1].timestamp != MTLCounterErrorValue && ts[2 * i + 1].timestamp >= ts[2 * i].timestamp;
                    out[done + i] = ok ? (double)(ts[2 * i + 1].timestamp - ts[2 * i].timestamp) * tick : -1;
                }
            }
        }
        return (int)n;
    }
}

static BOOL ink2_overlaps(const ink2_spans *set, ink2_span s) {
    for (size_t i = 0; i < set->n; i++)
        if (set->at[i].buffer == s.buffer && set->at[i].lo < s.hi && s.lo < set->at[i].hi) return YES;
    return NO;
}

static BOOL ink2_note(ink2_spans *set, ink2_span s) {
    for (size_t i = 0; i < set->n; i++) {
        ink2_span *o = &set->at[i];
        if (o->buffer == s.buffer && o->lo <= s.hi && s.lo <= o->hi) {
            o->lo = MIN(o->lo, s.lo);
            o->hi = MAX(o->hi, s.hi);
            return YES;
        }
    }
    if (set->n == set->cap) {
        size_t cap = set->cap ? 2 * set->cap : 64;
        ink2_span *grown = realloc(set->at, cap * sizeof(*grown));
        if (!grown) return NO;
        set->at = grown;
        set->cap = cap;
    }
    set->at[set->n++] = s;
    return YES;
}

static void ink2_forget(MGInk2Device *d) {
    d->reads.n = 0;
    d->writes.n = 0;
}

static void ink2_barrier(MGInk2Device *d) {
    [d->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    d->stats.barriers++;
    ink2_forget(d);
}

// The extent a binding may touch: its declared bytes, or the rest of the
// buffer when it declared none.
static ink2_span ink2_extent(const mg_ink2_arg *a) {
    size_t length = ((__bridge id<MTLBuffer>)(void *)a->buffer).length;
    size_t hi = a->length ? a->offset + a->length : length;
    return (ink2_span){a->buffer, a->offset, MAX(hi, a->offset + 1)};
}

// Orders this dispatch after every unbarriered one it conflicts with, then
// records what it touches. Returns 0 when the record cannot grow.
static int ink2_track(MGInk2Device *d, uint64_t mask, const mg_ink2_arg *args, int count) {
    BOOL hazard = NO;
    for (int i = 0; i < count && !hazard; i++) {
        if (!args[i].buffer || args[i].buffer == d->untracked) continue;
        ink2_span s = ink2_extent(&args[i]);
        BOOL writes = i >= 64 || (mask >> i & 1);
        hazard = ink2_overlaps(&d->writes, s) || (writes && ink2_overlaps(&d->reads, s));
    }
    if (hazard) ink2_barrier(d);
    for (int i = 0; i < count; i++) {
        if (!args[i].buffer || args[i].buffer == d->untracked) continue;
        ink2_span s = ink2_extent(&args[i]);
        BOOL writes = i >= 64 || (mask >> i & 1);
        if (!ink2_note(writes ? &d->writes : &d->reads, s)) return ink2_fail(d, @"hazard record allocation");
    }
    return 1;
}

// Commits the open command buffer without waiting, chaining the next one
// after it so a token's dispatches still run in order.
static void ink2_commit_early(MGInk2Device *d) {
    [d->encoder endEncoding];
    d->encoder = nil;
    ink2_forget(d);
    if (!d->chain) d->chain = [gDev newEvent];
    [d->command encodeSignalEvent:d->chain value:++d->chainValue];
    if (!d->inflight) d->inflight = [NSMutableArray array];
    if (d->inflight.count == 0) d->firstCommit = ink2_now();
    [d->command commit];
    [d->inflight addObject:d->command];
    d->command = nil;
    d->encoded = 0;
    d->stats.commits++;
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
        if (!d->command) {
            d->command = [gQueue commandBuffer];
            if (d->chainValue) [d->command encodeWaitForEvent:d->chain value:d->chainValue];
        }
        if (d->passProfile) {
            if (@available(macOS 11.0, *)) d->encoder = ink2_pass(d);
            if (!d->encoder) return 0;
        } else if (!d->encoder) {
            d->encoder = d->concurrent ? [d->command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                       : [d->command computeCommandEncoder];
            if (!d->encoder) {
                d->command = nil;
                return ink2_fail(d, @"no compute encoder");
            }
        }
        if (d->concurrent && !d->passProfile && !ink2_track(d, d->writeMasks[pipeline], args, count)) return 0;
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
        if (d->passProfile) {
            [enc endEncoding];
            d->encoder = nil;
            d->passes++;
        }
        d->stats.pending++;
        d->stats.dispatches++;
        if (d->commitEvery && !d->passProfile && ++d->encoded >= d->commitEvery) ink2_commit_early(d);
        return 1;
    }
}

void mg_ink2_barrier(void *dev) {
    MGInk2Device *d = (__bridge MGInk2Device *)dev;
    if (d->encoder) ink2_barrier(d);
}

int mg_ink2_flush(void *dev) {
    @autoreleasepool {
        MGInk2Device *d = (__bridge MGInk2Device *)dev;
        if (!d->command && d->inflight.count == 0) return 1;
        NSMutableArray<id<MTLCommandBuffer>> *buffers = d->inflight ? d->inflight : [NSMutableArray array];
        d->inflight = nil;
        double committed = buffers.count ? d->firstCommit : ink2_now();
        if (d->command) {
            [d->encoder endEncoding];
            d->encoder = nil;
            ink2_forget(d);
            [d->command commit];
            [buffers addObject:d->command];
            d->command = nil;
            d->stats.commits++;
        }
        d->encoded = 0;
        d->stats.pending = 0;
        d->stats.submissions++;
        d->stats.last_gpu_seconds = 0;
        NSUInteger passes = d->passes;
        d->passes = 0;
        NSString *failed = nil;
        for (id<MTLCommandBuffer> cb in buffers) {
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted && !failed) failed = cb.error.localizedDescription ?: @"not completed";
        }
        double resumed = ink2_now();
        if (failed) return ink2_fail(d, [NSString stringWithFormat:@"command buffer: %@", failed]);
        d->flushedPasses = passes;
        double gpu = 0;
        for (id<MTLCommandBuffer> cb in buffers) {
            if (cb.GPUEndTime > cb.GPUStartTime) gpu += cb.GPUEndTime - cb.GPUStartTime;
            CFTimeInterval kernel = cb.kernelEndTime - cb.kernelStartTime;
            if (kernel > 0) d->stats.kernel_seconds += kernel;
        }
        id<MTLCommandBuffer> first = buffers.firstObject, last = buffers.lastObject;
        if (gpu > 0) {
            d->stats.gpu_seconds += gpu;
            d->stats.last_gpu_seconds = gpu;
            // The GPU times share the host clock of ink2_now.
            if (first.GPUStartTime > committed) d->stats.queue_seconds += first.GPUStartTime - committed;
            if (resumed > last.GPUEndTime) d->stats.tail_seconds += resumed - last.GPUEndTime;
        }
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
