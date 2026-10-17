package model

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This opt-in diagnostic adapter keeps target execution on its admitted HAL
// route. It deliberately fences and copies selected residual rows to the host;
// it is not a device-resident DFlash graph or a throughput-qualified route.
// The existing caller-exclusive Session and backend execution owner contracts
// apply through every transaction, shared-head operation, and cleanup.
func validateQwen35TargetFeatureSession(s *Session) error {
	if s == nil || s.M == nil || s.Cache == nil || !s.modelWeightsHeld {
		return errors.New("model: target features require a live model-owning session")
	}
	cfg := s.M.Cfg
	if !cfg.IsQwen35Hybrid() || cfg.IsDeepSeekV41() || cfg.usesMLAMoELayout() || cfg.isMiniMaxSparseAttn() || cfg.isGemma4() {
		return errors.New("model: target features require a Qwen hybrid architecture")
	}
	if cfg.HiddenSize < 1 || cfg.HiddenSize > 5120 || cfg.VocabSize < 1 || cfg.VocabSize > 248320 {
		return errors.New("model: target feature diagnostic row budget exceeded")
	}
	if s.DenseGPULayers != 0 || s.GPULayers != 0 || s.executionPolicy == ExecutionPolicyDeviceOnly || s.M.epCoord != nil || s.M.epRanks > 1 || s.M.epRankSet || s.Cache.Precision() != KVPrecisionFP32 || s.PrecisionPolicy != nil || s.PrecisionSchedule != nil {
		return errors.New("model: target features require local, unsplit diagnostic execution")
	}
	if s.Backend == nil {
		if s.Quant || s.Q4 || s.Q4K || s.GPTQ || s.F16 || s.Metal || s.MetalQ4K || s.qwen35HAL != nil {
			return errors.New("model: host target features require native f32 state")
		}
	} else {
		if s.halClosed || s.halFailure != nil || s.halKV == nil || s.qwen35HAL == nil || s.Q4 || s.GPTQ || s.F16 || s.Metal || s.MetalQ4K {
			return errors.New("model: target features require a live admitted Qwen HAL session")
		}
		if (s.Quant && !s.useHALQ8Weights()) || (s.Q4K && !s.useHALQ4KWeights()) {
			return errors.New("model: quantized target features require native packed HAL weights")
		}
	}
	_, _, err := qwen35TargetFeaturePrefix(s)
	return err
}

// HAL KV owns position even when the legacy host cache remains positionless.
// Return the matching lineage together so a caller cannot mix authorities.
func qwen35TargetFeaturePrefix(s *Session) (int, *tokenLineage, error) {
	if s == nil || s.Cache == nil {
		return 0, nil, errors.New("model: target feature prefix is unavailable")
	}
	position, lineage := s.Cache.Len(), &s.Cache.lineage
	if s.Backend != nil {
		if s.halKV == nil || s.halClosed || !s.modelWeightsHeld {
			return 0, nil, errors.New("model: target feature HAL prefix is unavailable")
		}
		position, lineage = s.halKV.Len(), &s.halLineage
	}
	if lineage.fault != "" || len(lineage.ids) != position {
		return 0, nil, errors.New("model: target features require exact prefix token lineage")
	}
	return position, lineage, nil
}

func (c *qwen35TargetLayerCapture) wants(token, position, layer int) bool {
	if c == nil {
		return false
	}
	next := len(c.row.Hidden)
	return token == c.row.Token && position == c.row.Position && next < len(c.row.Layers) && layer == c.row.Layers[next]
}

// A backend Read may borrow storage. Copy before any later residual mutation,
// Free, or Recycle. Shape checks bound every read to one activation/logits row.
func qwen35TargetFeatureRead(be compute.Backend, tensor compute.Tensor, width int) []float32 {
	if width < 1 || width > 248320 || tensor.Buf() == nil || len(tensor.Shape) != 1 || tensor.Shape[0] != width || tensor.Dtype != compute.F32 {
		panic("model: malformed target feature diagnostic row")
	}
	row := be.Read(tensor)
	if len(row) != width {
		panic(fmt.Sprintf("model: target feature diagnostic read width %d, want %d", len(row), width))
	}
	for _, value := range row {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			panic("model: nonfinite target feature diagnostic row")
		}
	}
	return append([]float32(nil), row...)
}

func (s *Session) captureQwen35TargetLayerHAL(id, pos, layer int, residual compute.Tensor, batch batchBackend) {
	capture := s.qwen35TargetLayerCapture
	if !capture.wants(id, pos, layer) {
		return
	}
	if batch != nil {
		batch.FlushBatch()
	}
	row := qwen35TargetFeatureRead(s.Backend, residual, s.M.Cfg.HiddenSize)
	capture.row.Hidden = append(capture.row.Hidden, row)
	if batch != nil {
		batch.BeginBatch()
	}
}

// Called only while the existing transaction owns backend execution. Replay
// also takes this path, even though it does not repeat the discarded taps.
func (s *Session) tokenHALTargetFeatures(id, pos int) []float32 {
	defer s.recycleHALToken()
	logits, _ := s.tokenHALOutput(id, pos, halFullLogits)
	s.admitHostStage(HostStageLogitsReadback)
	return qwen35TargetFeatureRead(s.Backend, logits, s.M.Cfg.VocabSize)
}

// Draft output rows share the admitted target head without expanding the target
// checkpoint. Only one hidden row and one vocabulary row cross the host boundary.
func (s *Session) dflash2TargetHead(ctx context.Context, hidden []float32) (out []float32, err error) {
	if ctx == nil || len(hidden) != s.M.Cfg.HiddenSize {
		return nil, errors.New("dflash2: invalid shared-head row or context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.Backend == nil {
		return append([]float32(nil), s.headResident(hidden)...), nil
	}
	scope := s.beginBackendExecution()
	defer scope.finishError(&err)
	defer func() {
		if p := recover(); p != nil {
			out, err = nil, fmt.Errorf("dflash2: shared HAL head: %v", p)
		}
	}()
	defer s.recycleHALToken()
	s.ensureOpenBackendSession()
	if s.executionPolicy == ExecutionPolicyDeviceOnly || !s.qwen35SequenceOutputHeadFits() {
		return nil, errors.New("dflash2: diagnostic shared head is outside the admitted host/buffer budget")
	}
	// Stage immutable weights before opening this operation's batch. The returned
	// head is borrowed from the target/model and must never be freed here.
	head := s.lmHeadMatHAL()
	x := s.uploadHostF32([]int{len(hidden)}, hidden, compute.MemoryActivation, "dflash2-shared-head-input")
	defer s.Backend.Free(x)
	var batch batchBackend
	if b, ok := s.Backend.(batchBackend); ok {
		batch = b
		batch.BeginBatch()
		defer batch.FlushBatch()
	}
	logits := s.Backend.MatMul(head, x)
	defer s.Backend.Free(logits)
	if batch != nil {
		batch.FlushBatch()
	}
	out = qwen35TargetFeatureRead(s.Backend, logits, s.M.Cfg.VocabSize)
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
