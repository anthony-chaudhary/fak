//go:build cuda && cgo

package ggufload

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// OpenInkling2LoopCUDA maps an inkling2_loop GGUF and uploads its packed
// weights to be once: NVFP4 experts repacked for compute.NewNVFP4 (the
// per-expert .scale stays a host multiply in the forward), K-quant, Q8_0 and
// i-quant tensors verbatim, large dense tensors widened to F32. The token
// embedding and small dense matrices stay host-side.
func OpenInkling2LoopCUDA(path string, be compute.Backend) (*Inkling2LoopModel, error) {
	if be == nil || be.Name() != "cuda" {
		return nil, fmt.Errorf("inkling2_loop: OpenInkling2LoopCUDA needs the cuda backend")
	}
	return OpenInkling2LoopDevice(path, be)
}
