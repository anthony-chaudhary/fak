package model

import (
	"encoding/json"
	"fmt"
	"math"
)

// estimate.go — the load-time device-fit pre-check for the safetensors loader (issue #709;
// capacity-bridge Plank 5, docs/explainers/hardware-limits-and-capacity.md). LoadSafetensors
// decodes every source tensor into one resident f32 buffer, so a model too big for the box
// OOM-panics mid-decode (the make/append in appendSafetensorsFileInto). EstimateSafetensors
// LoadBytes lifts that resident footprint OFF THE HEADER ALONE (no tensor read, no full load),
// so a caller can pair it with compute.RefuseIfTooBig to refuse an oversize model BEFORE the
// decode allocation. RefuseIfTooBig fails open on unknown capacity (cpu-ref), so this never
// gates the portable floor; the estimator itself is pure arithmetic over the header.

// EstimateSafetensorsLoadBytes reports the resident f32 weight footprint a safetensors
// checkpoint will occupy once LoadSafetensors decodes it, computed from the file HEADER alone
// (no tensor data is read, no model is built). Every float tensor (F32/BF16/F16) contributes
// its shape product * 4 bytes — the decoded f32 it becomes, the currency ResidentReport tallies
// after load. A non-float tensor (e.g. a U8 MXFP4 _blocks/_scales pair) contributes its on-disk
// span so a block layout is counted by what the loader reads, not mis-sized as f32. It is an
// overestimate for a model that drops vision/MTP tensors at load (those never resident) — the
// safe direction for a fit check. Pair with compute.RefuseIfTooBig(be, est, headroom) to refuse
// before the decode: that returns a typed *compute.FitError only on a known-too-small device
// and nil on the cpu-ref floor (unknown capacity), so the load path there is unchanged.
func EstimateSafetensorsLoadBytes(path string) (int64, error) {
	sf, err := openSafetensorsFile(path)
	if err != nil {
		return 0, err
	}
	defer sf.Close()
	var total int64
	for _, name := range safetensorsTensorNames(sf.hdr) {
		var e stEntry
		if err := json.Unmarshal(sf.hdr[name], &e); err != nil {
			return 0, fmt.Errorf("safetensors: estimate %s: %w", name, err)
		}
		switch e.Dtype {
		case "F32", "BF16", "F16":
			elems, ok := checkedShapeProduct(e.Shape...)
			if !ok {
				return 0, fmt.Errorf("safetensors: estimate %s shape %v overflows element count", name, e.Shape)
			}
			n := int64(elems) * 4
			if n/4 != int64(elems) || total > math.MaxInt64-n {
				return 0, fmt.Errorf("safetensors: estimate %s f32 byte size overflows int64", name)
			}
			total += n
		default:
			// Non-float (e.g. U8 MXFP4 blocks/scales): contribute the on-disk span so a block
			// layout is counted by the bytes the loader reads, not mis-sized as decoded f32.
			if len(e.DataOffsets) != 2 || int64(e.DataOffsets[1]) < int64(e.DataOffsets[0]) {
				return 0, fmt.Errorf("safetensors: estimate %s malformed data_offsets", name)
			}
			span := int64(e.DataOffsets[1]) - int64(e.DataOffsets[0])
			if total > math.MaxInt64-span {
				return 0, fmt.Errorf("safetensors: estimate %s byte size overflows int64", name)
			}
			total += span
		}
	}
	return total, nil
}
