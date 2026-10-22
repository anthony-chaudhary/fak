//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"

int fvk_debug_select_qwen35_value_head_tile(int enabled);
*/
import "C"

import "fmt"

var _ Qwen35GDNTiledOutBackend = (*vulkanBackend)(nil)

// Qwen35GDNDecodeTiledOut is Qwen35GDNDecode for an out_proj kept in the GGUF tiled input-column
// order: the core is tiled on the device before the output projection, with no host readback.
func (v *vulkanBackend) Qwen35GDNDecodeTiledOut(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState Tensor, err error) {
	return v.qwen35GDNDecode(normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj, convState, recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon, true)
}

// VulkanDebugQwen35ValueHeadTileAvailable reports whether tiling takes the kernel route (its
// module loaded and is selected); otherwise it records one device copy per (token, head).
func (v *vulkanBackend) VulkanDebugQwen35ValueHeadTileAvailable() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_qwen35_value_head_tile() != 0
}

// VulkanDebugSelectQwen35ValueHeadTile selects the tiling kernel (when its module loaded) or the
// copy route for parity witnesses, returning the previous selection.
func (v *vulkanBackend) VulkanDebugSelectQwen35ValueHeadTile(enabled bool) bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	flag := C.int(0)
	if enabled {
		flag = 1
	}
	return C.fvk_debug_select_qwen35_value_head_tile(flag) != 0
}

// VulkanDebugQwen35ValueHeadTileSnapshot returns how many tiling passes took the kernel and the
// copy route since the backend started.
func (v *vulkanBackend) VulkanDebugQwen35ValueHeadTileSnapshot() (kernelCalls, copyCalls int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.qwen35TileKernelCalls, v.qwen35TileCopyCalls
}

// VulkanDebugQwen35TileValueHeads runs one tiling pass on a resident [tokens, nV*vHd] core panel.
func (v *vulkanBackend) VulkanDebugQwen35TileValueHeads(core Tensor, tokens, numKeyHeads, numValueHeads, valueHeadDim int) (Tensor, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.qwen35TileValueHeadsLocked(core, tokens, numKeyHeads, numValueHeads, valueHeadDim)
}

func (v *vulkanBackend) qwen35TileValueHeads(core Tensor, tokens, numKeyHeads, numValueHeads, valueHeadDim int) (Tensor, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.qwen35TileValueHeadsLocked(core, tokens, numKeyHeads, numValueHeads, valueHeadDim)
}

// qwen35TileValueHeadsLocked records the tiling into a fresh transient panel of the core's shape.
// The caller holds vulkanMu; inside an open batch the pass is ordered after the core's producer.
func (v *vulkanBackend) qwen35TileValueHeadsLocked(core Tensor, tokens, numKeyHeads, numValueHeads, valueHeadDim int) (Tensor, error) {
	if _, err := Qwen35TiledValueHeadSource(numKeyHeads, numValueHeads); err != nil {
		return Tensor{}, err
	}
	b, ok := core.buf.(*vulkanBuf)
	if !ok || b == nil || b.ptr == nil || core.be != v || core.Dtype != F32 || tokens <= 0 || valueHeadDim <= 0 ||
		core.Numel() != tokens*numValueHeads*valueHeadDim {
		return Tensor{}, fmt.Errorf("%w: core must be a resident F32 [%d, %d] panel", ErrQwen35ValueHeadTiling, tokens, numValueHeads*valueHeadDim)
	}
	out, _ := v.devTr(append([]int(nil), core.Shape...), F32)
	status := int(C.fvk_qwen35_value_head_tile_f32(v.vp(core), v.vp(out), C.int(tokens), C.int(numKeyHeads), C.int(numValueHeads), C.int(valueHeadDim)))
	if status != 0 {
		return Tensor{}, fmt.Errorf("compute: vulkan Qwen GDN value-head tiling failed closed (code %d)", status)
	}
	if C.fvk_have_qwen35_value_head_tile() != 0 {
		v.qwen35TileKernelCalls++
	} else {
		v.qwen35TileCopyCalls++
	}
	return out, nil
}
