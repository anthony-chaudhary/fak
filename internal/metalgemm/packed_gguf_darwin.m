//go:build darwin && arm64 && cgo

#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#include "packed_gguf.h"

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;

@interface MGPackedStore : NSObject {
@public
    id<MTLComputePipelineState> pipelines[41], panelPipelines[41], routedPipelines[41];
    id<MTLBuffer> inputs, outputs;
    id<MTLComputePipelineState> situ, expertSum, route;
}
@end

@implementation MGPackedStore
@end

static NSString * const packedNames[41] = {
    [0]=@"f32", [1]=@"f16", [8]=@"q8_0", [11]=@"q3_K", [12]=@"q4_K", [13]=@"q5_K",
    [14]=@"q6_K", [17]=@"iq2_xs", [18]=@"iq3_xxs", [20]=@"iq4_nl", [21]=@"iq3_s",
    [22]=@"iq2_s", [30]=@"bf16", [40]=@"nvfp4"
};

void *mg_packed_create(const char *shader) {
    @autoreleasepool {
        NSError *err = nil;
        id<MTLLibrary> lib = [gDev newLibraryWithSource:[NSString stringWithUTF8String:shader]
                                                options:nil error:&err];
        if (!lib) { NSLog(@"packed GGUF shader: %@", err); return NULL; }
        MGPackedStore *store = [[MGPackedStore alloc] init];
        for (unsigned i = 0; i < 41; i++) {
            if (!packedNames[i]) continue;
            NSString *name = [@"mv_" stringByAppendingString:packedNames[i]];
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) return NULL;
            NSString *panelName = [@"panel_" stringByAppendingString:packedNames[i]];
            id<MTLFunction> panel = [lib newFunctionWithName:panelName];
            if (!panel) return NULL;
            store->panelPipelines[i] = [gDev newComputePipelineStateWithFunction:panel error:&err];
            if (!store->panelPipelines[i] || store->panelPipelines[i].threadExecutionWidth != 32 ||
                store->panelPipelines[i].maxTotalThreadsPerThreadgroup < 128) return NULL;
            NSString *routedName = [@"routed_" stringByAppendingString:packedNames[i]];
            id<MTLFunction> routed = [lib newFunctionWithName:routedName];
            store->routedPipelines[i] = [gDev newComputePipelineStateWithFunction:routed error:&err];
            if (!store->routedPipelines[i] || store->routedPipelines[i].threadExecutionWidth != 32 ||
                store->routedPipelines[i].maxTotalThreadsPerThreadgroup < 128) return NULL;
            store->pipelines[i] = [gDev newComputePipelineStateWithFunction:fn error:&err];
            if (!store->pipelines[i] || store->pipelines[i].threadExecutionWidth != 32 ||
                store->pipelines[i].maxTotalThreadsPerThreadgroup < 128) {
                NSLog(@"packed GGUF pipeline %@: %@", name, err);
                return NULL;
            }
        }
        store->situ = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"packed_situ"] error:&err];
        store->expertSum = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"packed_expert_sum"] error:&err];
        store->route = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"packed_route"] error:&err];
        if (!store->situ || !store->expertSum || !store->route) {
            NSLog(@"packed GGUF expert pipelines: %@", err);
            return NULL;
        }
        return (void *)CFBridgingRetain(store);
    }
}

size_t mg_packed_max_buffer(void) { return gDev.maxBufferLength; }

static id<MTLBuffer> packed_scratch(id<MTLBuffer> buffer, size_t bytes) {
    if (buffer.length >= bytes) return buffer;
    return [gDev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

int mg_packed_prepare(void *ctx, size_t nx, size_t ny, float **x, float **y) {
    @autoreleasepool {
        MGPackedStore *s = (__bridge MGPackedStore *)ctx;
        s->inputs = packed_scratch(s->inputs, nx*sizeof(float));
        s->outputs = packed_scratch(s->outputs, ny*sizeof(float));
        *x = s->inputs.contents;
        *y = s->outputs.contents;
        return *x && *y;
    }
}

static void packed_encode(MGPackedStore *s, id<MTLComputeCommandEncoder> enc, const mg_packed_job *j) {
    [enc setComputePipelineState:s->pipelines[j->type]];
    [enc setBuffer:(__bridge id<MTLBuffer>)(void *)j->weight offset:(j->byte_offset & ~(size_t)3) atIndex:0];
    [enc setBuffer:s->inputs offset:j->input_offset*sizeof(float) atIndex:1];
    [enc setBuffer:s->outputs offset:j->output_offset*sizeof(float) atIndex:2];
    uint32_t shape[] = {j->rows, j->cols, (uint32_t)(j->byte_offset & 3)};
    [enc setBytes:shape length:sizeof(shape) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((j->rows+3u)/4u,1,1)
        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}

static int packed_finish(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSLog(@"packed GGUF command: %@", cb.error);
        return 0;
    }
    return 1;
}

static void packed_encode_panel(MGPackedStore *s, id<MTLComputeCommandEncoder> enc,
                                const mg_packed_job *jobs, const mg_packed_panel *panel) {
    const mg_packed_job *j = &jobs[panel->job[0]];
    if (panel->count == 1) { packed_encode(s, enc, j); return; }
    struct { uint32_t count, input[4], output[4]; } offsets = {.count=panel->count};
    for (uint32_t i = 0; i < panel->count; i++) {
        offsets.input[i] = (uint32_t)jobs[panel->job[i]].input_offset;
        offsets.output[i] = (uint32_t)jobs[panel->job[i]].output_offset;
    }
    uint32_t shape[] = {j->rows, j->cols, (uint32_t)(j->byte_offset & 3)};
    [enc setComputePipelineState:s->panelPipelines[j->type]];
    [enc setBuffer:(__bridge id<MTLBuffer>)(void *)j->weight offset:(j->byte_offset & ~(size_t)3) atIndex:0];
    [enc setBuffer:s->inputs offset:0 atIndex:1];
    [enc setBuffer:s->outputs offset:0 atIndex:2];
    [enc setBytes:shape length:sizeof(shape) atIndex:3];
    [enc setBytes:&offsets length:sizeof(offsets) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((j->rows+3u)/4u,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}

static bool same_packed_matrix(const mg_packed_job *a, const mg_packed_job *b) {
    return a->weight == b->weight && a->byte_offset == b->byte_offset &&
           a->rows == b->rows && a->cols == b->cols;
}

// Prompt rows repeat the same sequence of controller projections. The packed
// activation and result arrays have a fixed stride even when Q/K/V interleave.
static bool packed_f32_panel(MGPackedStore *s, const mg_packed_job *jobs, size_t count) {
    size_t period = 1;
    while (period < count && !same_packed_matrix(jobs, &jobs[period])) ++period;
    if (period == count || count % period != 0) return false;
    size_t tokens = count / period;
    if (tokens > INT_MAX) return false;
    size_t dx = jobs[period].input_offset - jobs[0].input_offset;
    size_t dy = jobs[period].output_offset - jobs[0].output_offset;
    if (dx > INT_MAX || dy > INT_MAX) return false;
    for (size_t i = 0; i < count; ++i) {
        const mg_packed_job *first = &jobs[i % period], *j = &jobs[i];
        if (!same_packed_matrix(first, j) || j->cols > dx || j->rows > dy ||
            first->input_offset + (i/period)*dx != j->input_offset ||
            first->output_offset + (i/period)*dy != j->output_offset) return false;
        // Short panels of skinny-output GEMVs are faster as independent dots.
        if (tokens < 16 && j->rows <= j->cols) return false;
    }
    const float *x = s->inputs.contents;
    float *y = s->outputs.contents;
    for (size_t i = 0; i < period; ++i) {
        const mg_packed_job *j = &jobs[i];
        id<MTLBuffer> w = (__bridge id<MTLBuffer>)(void *)j->weight;
        const float *weight = (const float *)((const char *)w.contents + j->byte_offset);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)tokens, (int)j->rows,
                    (int)j->cols, 1.f, x + j->input_offset, (int)dx, weight, (int)j->cols,
                    0.f, y + j->output_offset, (int)dy);
    }
    return true;
}

// The router and recurrent controls are skinny FP32 matrices with CPU-visible
// inputs and outputs. Avoid a GPU round trip for these small projections.
static bool packed_host_f32(MGPackedStore *s, const mg_packed_job *jobs, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const mg_packed_job *j = &jobs[i];
        if (j->type != 0 || (j->rows > 128 && j->cols > 128) ||
            j->rows > INT_MAX || j->cols > INT_MAX || (j->byte_offset % sizeof(float)) != 0) return false;
    }
    if (packed_f32_panel(s, jobs, count)) return true;
    const float *x = s->inputs.contents;
    float *y = s->outputs.contents;
    for (size_t i = 0; i < count; ++i) {
        const mg_packed_job *j = &jobs[i];
        id<MTLBuffer> w = (__bridge id<MTLBuffer>)(void *)j->weight;
        const float *weights = (const float *)((const char *)w.contents + j->byte_offset);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)j->rows, (int)j->cols,
                    1.0f, weights, (int)j->cols, x+j->input_offset, 1, 0.0f, y+j->output_offset, 1);
    }
    return true;
}

int mg_packed_run(void *ctx, const mg_packed_job *jobs, size_t count, const mg_packed_panel *panels) {
    @autoreleasepool {
        MGPackedStore *s = (__bridge MGPackedStore *)ctx;
        size_t jobCount = count;
        if (panels) {
            jobCount = 0;
            for (size_t i = 0; i < count; ++i) jobCount += panels[i].count;
        }
        if (packed_host_f32(s, jobs, jobCount)) return 1;
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;
        for (size_t i = 0; i < count; ++i) {
            if (panels) packed_encode_panel(s, enc, jobs, &panels[i]);
            else packed_encode(s, enc, &jobs[i]);
        }
        return packed_finish(cb, enc);
    }
}

static void packed_situ(MGPackedStore *s, id<MTLComputeCommandEncoder> enc,
                        mg_packed_moe_shape shape, size_t gateAt, size_t actAt) {
    size_t k = shape.slots;
    [enc setComputePipelineState:s->situ];
    [enc setBuffer:s->outputs offset:gateAt*sizeof(float) atIndex:0];
    [enc setBuffer:s->inputs offset:shape.hidden*sizeof(float) atIndex:1];
    [enc setBuffer:s->inputs offset:actAt*sizeof(float) atIndex:2];
    [enc setBytes:&shape length:sizeof(shape) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(k*shape.width, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void packed_sum(MGPackedStore *s, id<MTLComputeCommandEncoder> enc,
                       mg_packed_moe_shape shape, size_t downs) {
    size_t k = shape.slots;
    [enc setComputePipelineState:s->expertSum];
    [enc setBuffer:s->outputs offset:downs*sizeof(float) atIndex:0];
    [enc setBuffer:s->inputs offset:shape.hidden*sizeof(float) atIndex:1];
    [enc setBuffer:s->outputs offset:(downs + k*shape.hidden)*sizeof(float) atIndex:2];
    [enc setBytes:&shape length:sizeof(shape) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(shape.hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

int mg_packed_moe(void *ctx, const mg_packed_job *jobs, mg_packed_moe_shape shape) {
    @autoreleasepool {
        MGPackedStore *s = (__bridge MGPackedStore *)ctx;
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;
        size_t k = shape.slots;
        for (size_t i = 0; i < k; ++i) packed_encode(s, enc, &jobs[i]);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        packed_situ(s, enc, shape, 0, jobs[k].input_offset);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        for (size_t i = 0; i < k; ++i) packed_encode(s, enc, &jobs[k+i]);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        packed_sum(s, enc, shape, jobs[k].output_offset);
        return packed_finish(cb, enc);
    }
}

static void packed_indexed(MGPackedStore *s, id<MTLComputeCommandEncoder> enc,
                           const mg_packed_job *j, size_t idsAt, uint32_t k, uint32_t xStride) {
    [enc setComputePipelineState:s->routedPipelines[j->type]];
    [enc setBuffer:(__bridge id<MTLBuffer>)(void *)j->weight offset:(j->byte_offset & ~(size_t)3) atIndex:0];
    [enc setBuffer:s->inputs offset:j->input_offset*sizeof(float) atIndex:1];
    [enc setBuffer:s->outputs offset:j->output_offset*sizeof(float) atIndex:2];
    uint32_t dims[] = {j->rows, j->cols, (uint32_t)(j->byte_offset & 3)};
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc setBuffer:s->inputs offset:idsAt*sizeof(float) atIndex:4];
    [enc setBytes:&xStride length:sizeof(xStride) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((j->rows+3u)/4u,k,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}

int mg_packed_routed_ffn(void *ctx, const mg_packed_job *jobs, mg_packed_moe_shape shape,
                         mg_packed_route_shape route, size_t idsAt, size_t tableAt) {
    @autoreleasepool {
        MGPackedStore *s = (__bridge MGPackedStore *)ctx;
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;
        packed_encode(s, enc, jobs);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setComputePipelineState:s->route];
        [enc setBuffer:s->outputs offset:0 atIndex:0];
        [enc setBuffer:s->inputs offset:tableAt*sizeof(float) atIndex:1];
        [enc setBuffer:s->inputs offset:idsAt*sizeof(float) atIndex:2];
        [enc setBuffer:s->inputs offset:shape.hidden*sizeof(float) atIndex:3];
        [enc setBytes:&route length:sizeof(route) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        packed_indexed(s, enc, jobs+1, idsAt, shape.slots, 0);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        packed_situ(s, enc, shape, jobs[1].output_offset, jobs[2].input_offset);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        packed_indexed(s, enc, jobs+2, idsAt, shape.slots, shape.activation_stride);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        packed_sum(s, enc, shape, jobs[2].output_offset);
        return packed_finish(cb, enc);
    }
}

void mg_packed_close(void *ctx) {
    if (ctx) CFRelease(ctx);
}
