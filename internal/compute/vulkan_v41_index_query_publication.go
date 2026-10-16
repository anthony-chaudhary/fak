//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
// Return status by value: a Go int passed by address would escape on every MISS.
typedef struct { int selected; int status; } FVKIndexQueryResult;
static FVKIndexQueryResult fvk_index_query_call(float* raw, float* published, const void* source, const void* weight, const void* input, int probe) {
 FVKIndexQueryResult result;
 result.status = 0;
 result.selected = fvk_v41_try_index_query_publication(raw,published,source,weight,input,probe,&result.status);
 return result;
}
*/
import "C"
import "unsafe"

func (v *vulkanBackend) v41TryIndexQueryPublication(original Backend, projection Tensor, weight Buffer, input Tensor) (*V41IndexQueryPublication, bool, error) {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil || projection.Backend() != v || input.Backend() != v || projection.Dtype != F32 || projection.Layout != RowMajor || projection.Quant != nil || len(projection.Shape) != 1 || projection.Shape[0] != v41IndexQueryPublicationCount || input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil || len(input.Shape) != 1 || input.Shape[0] <= 0 {
		return nil, false, nil
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	requireVulkanSubmissionHealthyLocked("index-query publication transport")
	simple := func(buf Buffer) *vulkanBuf {
		b, ok := buf.(*vulkanBuf)
		if !ok || b == nil || b.ptr == nil || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return nil
		}
		return b
	}
	y, w, x := simple(projection.Buf()), simple(weight), simple(input.Buf())
	if y == nil || w == nil || x == nil {
		return nil, false, nil
	}
	var status C.int
	invoke := func(raw, published []float32, probe C.int) int {
		var rp, pp *C.float
		if len(raw) > 0 {
			rp = (*C.float)(unsafe.Pointer(&raw[0]))
		}
		if len(published) > 0 {
			pp = (*C.float)(unsafe.Pointer(&published[0]))
		}
		result := C.fvk_index_query_call(rp, pp, y.ptr, w.ptr, x.ptr, probe)
		status = result.status
		return int(result.selected)
	}
	failure := func() (*V41IndexQueryPublication, bool, error) {
		err := vulkanReadError(int(status))
		err.Site = "index-query publication transport"
		return nil, true, err
	}
	probe := invoke(nil, nil, 1)
	if probe == 0 && status == 0 {
		return nil, false, nil
	}
	if probe != 1 || status != 0 {
		return failure()
	}
	// Explicit per-call payload: 32,768 raw + 32,784 published bytes. This is
	// not a global reservation. Early raw errors allocate 32,784 bytes earlier.
	raw := make([]float32, v41IndexQueryPublicationCount)
	published := make([]float32, v41IndexQueryPublicationCount+4)
	selected := invoke(raw, published, 0)
	if selected == 0 && status == 0 {
		return nil, false, nil
	}
	if selected != 1 || status != 0 {
		return failure()
	}
	return &V41IndexQueryPublication{raw: raw, publication: published}, true, nil
}
