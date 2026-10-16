//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"
import (
	"encoding/binary"
	"errors"
	"math"
	"unsafe"
)

// The explicit per-selected-chunk host cost is bounded independently of model
// geometry. This is not a memory reservation or a process-wide residency cap.
const v41FFNPublicationHostBytes = 1310736

func (v *vulkanBackend) v41TryFFNNormPublication(original Backend, gain Tensor, owned []float32, rows, width int, eps float32) (*V41FFNNormPublication, bool, error) {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil {
		return nil, false, nil
	}
	if rows < 2 || rows > 64 || width <= 0 || width > (v41FFNPublicationHostBytes-16)/(4*rows) || len(owned) != rows*width {
		return nil, false, nil
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock() // includes Go allocation panic/Goexit; no native claim yet
	requireVulkanSubmissionHealthyLocked("FFN publication panel transport")
	b, ok := gain.buf.(*vulkanBuf)
	if !ok || b == nil || b.ptr == nil || gain.Backend() != v || gain.Dtype != F32 || len(gain.Shape) != 1 || gain.Shape[0] != width || gain.Numel() != width {
		return nil, false, nil
	}
	var status C.int
	probe := int(C.fvk_v41_try_ffn_norm_publication((*C.float)(unsafe.Pointer(&owned[0])), nil, b.ptr, C.int(rows), C.int(width), C.float(eps), 1, &status))
	if probe == 0 && status == 0 {
		return nil, false, nil
	}
	if probe != 1 || status != 0 {
		err := vulkanReadError(int(status))
		err.Site = "FFN publication panel transport"
		return nil, true, err
	}
	// Native state cannot change between probe and commit under vulkanMu. One
	// bounded allocation replaces successful scalar output rows. Early failure
	// may temporarily retain this full slab until return, unlike scalar prefixes.
	slab := make([]float32, rows*width+4)
	selected := int(C.fvk_v41_try_ffn_norm_publication((*C.float)(unsafe.Pointer(&owned[0])), (*C.float)(unsafe.Pointer(&slab[0])), b.ptr, C.int(rows), C.int(width), C.float(eps), 0, &status))
	if selected != 1 || status != 0 {
		// A defensive commit MISS is an invariant failure, never a second route.
		err := vulkanReadError(int(status))
		err.Site = "FFN publication panel transport"
		return nil, true, err
	}
	var raw [16]byte
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(slab[i]))
	}
	verdict := decodeV41BF16PublishStatus(raw[:], rows*width, V41BF16PublishFFNNormOutput)
	bad := -1
	if verdict != nil {
		var arithmetic *V41BF16CheckedArithmeticError
		if !errors.As(verdict, &arithmetic) {
			return nil, true, verdict
		}
		bad = arithmetic.Index
	}
	return &V41FFNNormPublication{values: slab[4:], rows: rows, width: width, firstBad: bad}, true, nil
}
