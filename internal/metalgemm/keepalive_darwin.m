//go:build darwin && arm64 && cgo
/*
MIT License
Copyright (c) 2026 Carlos Galarza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MSL kernel copied directly from carloslfu/slotstream
 a855c49090330df88b2464814bcc003c8a7c5bfe
 Sources/Slotstream/GPUKeepAlive.swift.
Lifecycle adapted to bounded completion callbacks: no host/GPU waits, no idle
resubmission, two buffers maximum, sticky failure. Model data is never accessed.
*/
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOKit/ps/IOPowerSources.h>
#import <IOKit/ps/IOPSKeys.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

extern id<MTLDevice> gDev;
static NSObject *kaLock;
static id<MTLCommandQueue> kaQueue;
static id<MTLComputePipelineState> kaPipeline;
static id<MTLBuffer> kaFlag;
static BOOL kaActive, kaFailed;
static int kaInFlight;
static unsigned long long kaSubmitted;
static const uint32_t kaIterations=2048;
static double kaMaxCompletedGPUMilliseconds;
static long long kaFailureCode;
static char kaFailureDomain[96], kaFailureMessage[192];
// Preserve the first typed refusal with bounded default-readable evidence.
static void kaFail(NSError *error, long long code, const char *reason) {
 if (!kaFailed) {
  kaFailureCode=error ? (long long)error.code : code;
  snprintf(kaFailureDomain,sizeof(kaFailureDomain),"%s",error ? error.domain.UTF8String : "fak.metalgemm.keepalive");
  snprintf(kaFailureMessage,sizeof(kaFailureMessage),"%s",error ? error.localizedDescription.UTF8String : reason);
 }
 kaFailed=YES; kaActive=NO;
 if (kaFlag) atomic_store_explicit((_Atomic uint32_t*)[kaFlag contents],1,memory_order_relaxed);
}

static void kaEnsureLock(void) {
 static dispatch_once_t once;
 dispatch_once(&once, ^{ kaLock=[NSObject new]; });
}

// Called with kaLock held. Completion callbacks share the same lock, so a
// balanced end prevents any further submission and an overlapping start cannot
// exceed the two-buffer limit while prior finite buffers drain.
static void kaSubmit(void) {
 while (kaActive && !kaFailed && kaInFlight<2) {
  id<MTLCommandBuffer> buffer=[kaQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[buffer computeCommandEncoder];
  if (!buffer || !encoder) { kaFail(nil,1,"command buffer or encoder creation refused"); return; }
  uint32_t count=kaIterations;
  [encoder setComputePipelineState:kaPipeline];
  [encoder setBuffer:kaFlag offset:0 atIndex:0];
  [encoder setBytes:&count length:sizeof(count) atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [encoder endEncoding];
  kaInFlight++; kaSubmitted++;
  [buffer addCompletedHandler:^(id<MTLCommandBuffer> done) {
   @autoreleasepool { @synchronized(kaLock) {
    kaInFlight--;
    if (done.status==MTLCommandBufferStatusCompleted && done.GPUStartTime>0 && done.GPUEndTime>=done.GPUStartTime) {
     double elapsed=1000*(done.GPUEndTime-done.GPUStartTime);
     if (elapsed>kaMaxCompletedGPUMilliseconds) kaMaxCompletedGPUMilliseconds=elapsed;
    }
    if (done.status!=MTLCommandBufferStatusCompleted) {
     kaFail(done.error,2,"command buffer did not complete");
    }
    kaSubmit();
   } }
  }];
  [buffer commit];
 }
}

int mg_keepalive_begin(void) {
 @autoreleasepool {
  kaEnsureLock();
  @synchronized(kaLock) {
   if (kaFailed) return 0;
   if (!kaQueue) {
    id<MTLDevice> device=gDev;
    if (!device) return 0;
    NSString *source=@"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void slotstream_keepalive(device atomic_uint* flag [[buffer(0)]],\n"
     "                                 constant uint& iterations [[buffer(1)]],\n"
     "                                 uint i [[thread_position_in_grid]]) {\n"
     "  for (uint j = 0; j < iterations; j++) {\n"
     "    if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) break;\n"
     "  }\n"
     "}\n";
    NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> function=[library newFunctionWithName:@"slotstream_keepalive"];
    if (!function) { kaFail(error,3,"keepalive shader library or function creation refused"); return 0; }
    kaPipeline=[device newComputePipelineStateWithFunction:function error:&error];
    kaQueue=[device newCommandQueue];
    kaFlag=[device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    if (!kaPipeline || !kaQueue || !kaFlag) { kaFail(error,4,"keepalive pipeline, queue or flag allocation refused"); return 0; }
   }
   atomic_store_explicit((_Atomic uint32_t*)[kaFlag contents],0,memory_order_relaxed);
   kaActive=YES; kaSubmit();
   return !kaFailed;
  }
 }
}
void mg_keepalive_end(void) {
 kaEnsureLock();
 @synchronized(kaLock) {
  kaActive=NO;
  if (kaFlag) atomic_store_explicit((_Atomic uint32_t*)[kaFlag contents],1,memory_order_relaxed);
 }
}
int mg_keepalive_power(void) {
 @autoreleasepool {
  CFTypeRef info=IOPSCopyPowerSourcesInfo();
  if (!info) return 0;
  CFStringRef source=IOPSGetProvidingPowerSourceType(info);
  int bits=0;
  if (source && (CFEqual(source,CFSTR(kIOPSACPowerValue)) || CFEqual(source,CFSTR(kIOPSBatteryPowerValue)))) {
   bits=1;
   if(CFEqual(source,CFSTR(kIOPSBatteryPowerValue))) bits|=2;
  }
  CFRelease(info);
  if (@available(macOS 12.0,*)) { if ([NSProcessInfo processInfo].lowPowerModeEnabled) bits|=4; }
  else return 0;
  return bits;
 }
}
void mg_keepalive_state(unsigned long long *submitted,int *inflight,int *failed,long long *code,char *domain,char *message,unsigned int *iterations,double *max_gpu_ms) {
 kaEnsureLock();
 @synchronized(kaLock) { *submitted=kaSubmitted; *inflight=kaInFlight; *failed=kaFailed; *code=kaFailureCode; *iterations=kaIterations; *max_gpu_ms=kaMaxCompletedGPUMilliseconds; memcpy(domain,kaFailureDomain,sizeof(kaFailureDomain)); memcpy(message,kaFailureMessage,sizeof(kaFailureMessage)); }
}
