//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
typedef struct { int selected; int status; } FVKKVNormResult;
static FVKKVNormResult fvk_kv_norm_call(const float* host,float* image,const void* gain,float eps,int probe){
 FVKKVNormResult r; r.status=0;
 r.selected=fvk_v41_try_kv_norm_publication(host,image,gain,eps,probe,&r.status);
 return r;
}
*/
import "C"
import "unsafe"

func (v *vulkanBackend) v41TryKVNormPublication(original Backend, gain Tensor, input []float32, eps float32) (*V41KVNormPublication, bool, error) {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil || len(input) != v41KVNormWidth || gain.Backend() != v || gain.Dtype != F32 || gain.Layout != RowMajor || gain.Quant != nil || len(gain.Shape) != 1 || gain.Shape[0] != v41KVNormWidth {
		return nil, false, nil
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	requireVulkanSubmissionHealthyLocked("KV norm chain transport")
	w, ok := gain.Buf().(*vulkanBuf)
	if !ok || w == nil || w.ptr == nil || w.scalePtr != nil || len(w.q8Chunks) != 0 {
		return nil, false, nil
	}
	var status C.int
	invoke := func(image []float32, probe C.int) int {
		var out *C.float
		if len(image) > 0 {
			out = (*C.float)(unsafe.Pointer(&image[0]))
		}
		r := C.fvk_kv_norm_call((*C.float)(unsafe.Pointer(&input[0])), out, w.ptr, C.float(eps), probe)
		status = r.status
		return int(r.selected)
	}
	fail := func() (*V41KVNormPublication, bool, error) {
		err := vulkanReadError(int(status))
		err.Site = "KV norm chain transport"
		return nil, true, err
	}
	probe := invoke(nil, 1)
	if probe == 0 && status == 0 {
		return nil, false, nil
	}
	if probe != 1 || status != 0 {
		return fail()
	}
	// Root-accepted per-call payload, not a process-wide reservation. Allocation
	// panic/Goexit unlocks before any native owner has been claimed.
	image := make([]float32, v41KVNormImageWords)
	selected := invoke(image, 0)
	if selected != 1 || status != 0 {
		return fail()
	}
	return &V41KVNormPublication{image: image}, true, nil
}
