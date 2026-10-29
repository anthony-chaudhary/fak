package model

import "github.com/anthony-chaudhary/fak/internal/stepobs"

// v41ColdCompressorHostPairs is a ratio-two, host-only composer. The caller
// already established full geometry and at least two input rows. Admission is
// metadata-only: malformed shapes keep the original scalar first-error site.
// Arbitrary projection, pooling, normalization and observation callbacks keep
// their scalar calls. No device route or adapter arithmetic enters this path.
func (m *Model) v41ColdCompressorHostPairs(l int, inputs [][]float32, scratch *v41ProjScratch) ([][]float32, bool, error) {
	if m == nil || scratch == nil || len(inputs) < 2 || m.lora != nil || m.prism != nil || tensorLookupProbe != nil ||
		scratch.denseProjection() != nil || scratch.compressorNorm != nil || scratch.gatherOps.compressorPoolFunc() != nil || scratch.activationTrace != nil {
		return nil, false, nil
	}
	observation := stepobs.CaptureHostInputObservation()
	if !observation.Unchanged() || m.v41CompressorHostCustomClock() {
		return nil, false, nil
	}
	H, width := m.Cfg.HiddenSize, v41CompressorWidth(m.Cfg)
	elements, ok := checkedMulInt(width, H)
	bytes, bytesOK := checkedMulInt(elements, 4)
	_, pairOK := checkedMulInt(2, H)
	_, resultOK := checkedMulInt(2, width)
	if H <= 0 || width <= 0 || !ok || !bytesOK || !pairOK || !resultOK {
		return nil, false, nil
	}
	for _, row := range inputs {
		if len(row) != H {
			return nil, false, nil
		}
	}
	for _, leaf := range v41ColdCompressorLeaves {
		meta, present := m.manifest[layerName(l, leaf)]
		if !present || (meta.Dtype != "f32" && meta.Dtype != "F32") || len(meta.Shape) != 2 || meta.Shape[0] != width || meta.Shape[1] != H ||
			meta.Nbytes != bytes || meta.Offset < 0 || meta.Offset > len(m.raw) || bytes > len(m.raw)-meta.Offset || meta.Offset%4 != 0 {
			return nil, false, nil
		}
	}
	// Preserve the scalar composer's norm lookup before pool construction and
	// projection. Pool validation still happens after that row's two projections.
	gain, eps := m.tensor(layerName(l, "attn.compressor.norm.weight")), float32(m.Cfg.RMSNormEps)
	pool, err := NewV41CompressorPool(2, width)
	if err != nil {
		return nil, true, v41StageErr(v41StageCompress, l, err)
	}
	weights := [2][]float32{m.tensor(layerName(l, v41ColdCompressorLeaves[0])), m.tensor(layerName(l, v41ColdCompressorLeaves[1]))}
	panel := make([]float32, 2*H)
	var result [][]float32
	for pos := 0; pos < len(inputs); {
		rows := 1
		var projected [2][]float32
		if pos+1 < len(inputs) && observation.Unchanged() && !m.v41CompressorHostCustomClock() && tensorLookupProbe == nil && m.lora == nil && m.prism == nil {
			rows = 2
			copy(panel[:H], inputs[pos])
			copy(panel[H:], inputs[pos+1])
			for leaf := range projected {
				opened := m.v41NowNanos()
				// matRows and matMulBatch both call mathx.FDot for every output. The
				// latter visits a weight row once around both input rows without changing
				// either dot reduction. Only independent projections are speculative;
				// their finite verdict remains in the original row-major pool order.
				projected[leaf] = matMulBatch(weights[leaf], panel, width, H, 2)
				m.v41NoteCompressorProjection(0, 1, 0, 2, 0, 0, opened)
			}
		} else {
			// Odd tails and a revoked observation retain scalar calls, without replay
			// of any completed group or resetting the partially filled pool.
			for leaf := range projected {
				projected[leaf], err = m.v41ProjMatRowsWithProjection(l, v41ColdCompressorLeaves[leaf], inputs[pos], width, H, nil)
				if err != nil {
					return nil, true, err
				}
			}
		}
		for row := 0; row < rows; row++ {
			pooled, emitted, err := pool.PushNormalized(pos+row, projected[0][row*width:(row+1)*width], projected[1][row*width:(row+1)*width], gain, eps)
			if err != nil {
				return nil, true, v41StageErr(v41StageCompress, l, err)
			}
			if emitted {
				result = append(result, pooled)
			}
		}
		pos += rows
	}
	return result, true, nil
}

func (m *Model) v41CompressorHostCustomClock() bool {
	if ledger := v41ExpertFaultLedgerOf(m, false); ledger != nil {
		ledger.mu.Lock()
		custom := ledger.nowNanos != nil
		ledger.mu.Unlock()
		return custom
	}
	return false
}
