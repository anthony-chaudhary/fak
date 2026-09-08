//go:build darwin && arm64 && cgo

// zerocopy_darwin.m — zero-copy GGUF tensor memory mapping into Metal shared buffers on Apple Silicon.
// Directly wraps page-aligned memory-mapped file pages into MTLBuffer instances using
// newBufferWithBytesNoCopy:length:options:deallocator: with MTLResourceStorageModeShared.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>

extern id<MTLDevice>       gDev;
extern id<MTLCommandQueue> gQueue;
extern int                 mg_init(void);
extern int                 mg_q4k_upload_span(const unsigned char* raw, size_t nbytes, size_t offset, int out, int in);

// mg_zerocopy_create_shared_buffer wraps an existing page-aligned memory span into an MTLBuffer
// configured with MTLResourceStorageModeShared. Returns a retained CFTypeRef or NULL on failure.
void* mg_zerocopy_create_shared_buffer(void* ptr, size_t length) {
    if (ptr == NULL || length == 0) return NULL;
    if (gDev == nil) {
        if (mg_init() != 1 || gDev == nil) return NULL;
    }
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    if (((uintptr_t)ptr % page) != 0 || (length % page) != 0) {
        return NULL;
    }
    id<MTLBuffer> buf = [gDev newBufferWithBytesNoCopy:ptr
                                                length:(NSUInteger)length
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    if (buf == nil) return NULL;
    return (void*)CFBridgingRetain(buf);
}

// mg_zerocopy_buffer_is_shared verifies that the underlying buffer uses MTLResourceStorageModeShared.
int mg_zerocopy_buffer_is_shared(void* handle) {
    if (handle == NULL) return 0;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
    return (buf.storageMode == MTLStorageModeShared) ? 1 : 0;
}

// mg_zerocopy_buffer_length returns the length in bytes of the Metal shared buffer.
size_t mg_zerocopy_buffer_length(void* handle) {
    if (handle == NULL) return 0;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
    return (size_t)buf.length;
}

// mg_zerocopy_buffer_contents returns the CPU-accessible memory pointer of the Metal shared buffer.
void* mg_zerocopy_buffer_contents(void* handle) {
    if (handle == NULL) return NULL;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
    return buf.contents;
}

// mg_zerocopy_buffer_free releases the retained Metal buffer handle.
void mg_zerocopy_buffer_free(void* handle) {
    if (handle == NULL) return;
    CFRelease((CFTypeRef)handle);
}

// mg_zerocopy_upload_span wraps an mmap span directly into a Metal shared buffer and registers
// it with the Q4_K weight table via mg_q4k_upload_span.
int mg_zerocopy_upload_span(const unsigned char* raw, size_t nbytes, size_t offset, int out, int in) {
    return mg_q4k_upload_span(raw, nbytes, offset, out, in);
}
