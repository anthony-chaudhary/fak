package ggufload

import (
	"fmt"
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// streamedDenseMaterializeWindowBytes mirrors model.q4kMaterializeWindowBytes: a lazy dense
// tensor faulted without a mapping allocates its page-aligned payload once and fills it through
// one reusable read window of this size.
const streamedDenseMaterializeWindowBytes int64 = 64 << 20

// EstimateStreamedExpertHostLoadPeak predicts the host anonymous peak of the streamed-expert load
// (LoadModelQ4KStreamedExperts) whose dense side is held as a bounded streamed working set
// (WithStreamedDenseQ4KWorkingSet). It is the MoE route EstimateQ4KHostLoadPeak does not qualify.
//
// Routed-expert slabs the tier owns and V4.1 Engram tables are never materialized, so they are not
// charged; the tier's own host retention budget is. Dense tensors the loader holds as range
// descriptors are charged at min(eligible, declared working set), and the largest of them is
// charged again as the one-tensor staging transient its device upload materializes. Every other
// tensor is an owned copy: native Q8 for a 2-D quant weight, F32 otherwise, and F32 for a name
// with no canonical mapping, so an unrecognized tensor overcharges rather than escaping the gate.
func (s *WeightSource) EstimateStreamedExpertHostLoadPeak(opts ...Q4KLoadOption) (HostLoadPeak, error) {
	if s == nil || s.File == nil {
		return HostLoadPeak{}, fmt.Errorf("gguf: streamed-expert host peak requires a weight source")
	}
	cfg, err := s.File.Config()
	if err != nil {
		return HostLoadPeak{}, err
	}
	o, err := resolveQ4KLoadOptions(cfg, opts)
	if err != nil {
		return HostLoadPeak{}, err
	}
	if !o.streamedExperts || !o.streamedDenseBounded || o.expertShardSet {
		return HostLoadPeak{}, fmt.Errorf("%w: streamed-expert host peak requires streamed experts with a bounded streamed-dense working set", ErrQ4KLoadEstimateUnsupported)
	}
	shards, err := s.FusedExpertTensors()
	if err != nil {
		return HostLoadPeak{}, err
	}
	streamed := map[string]bool{}
	for _, sh := range shards {
		for _, f := range sh.Fused {
			streamed[f.Name] = true
		}
	}

	var out HostLoadPeak
	byType := map[TensorType]*HostLoadTypeRow{}
	var windows []int64
	var eligible, largestLazy int64
	add := func(dst *int64, n int64) error {
		if n < 0 || *dst > math.MaxInt64-n {
			return fmt.Errorf("gguf: streamed-expert host peak estimate overflows int64")
		}
		*dst += n
		return nil
	}
	for _, info := range s.File.Tensors {
		if streamed[info.Name] ||
			(archIsDeepSeek41(cfg.ModelType) && deepseek41EngramTableTensor(info.Name)) ||
			(archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(info.Name)) {
			continue
		}
		payloadU, err := tensorPayloadBytes(info)
		if err != nil {
			return HostLoadPeak{}, err
		}
		elemsU, err := tensorElems(info)
		if err != nil {
			return HostLoadPeak{}, err
		}
		payload, err := checkedEstimateUint64(payloadU, "payload", info.Name)
		if err != nil {
			return HostLoadPeak{}, err
		}
		f32, err := checkedEstimateMulUint64(elemsU, 4, "f32 bytes", info.Name)
		if err != nil {
			return HostLoadPeak{}, err
		}
		canon, mapped := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if mapped {
			var keep bool
			if canon, keep = model.QuantSourceTensorName(cfg, canon); !keep {
				continue
			}
		}
		row := byType[info.Type]
		if row == nil {
			row = &HostLoadTypeRow{Type: info.Type.String()}
			byType[info.Type] = row
		}
		row.Tensors++
		if err := add(&row.PayloadBytes, payload); err != nil {
			return HostLoadPeak{}, err
		}
		if mapped && streamedDenseLazy(cfg, o, info.Type, canon) {
			if err := add(&row.StreamBytes, payload); err != nil {
				return HostLoadPeak{}, err
			}
			if err := add(&eligible, payload); err != nil {
				return HostLoadPeak{}, err
			}
			largestLazy = max(largestLazy, payload)
			windows = append(windows, 0)
			continue
		}
		shape, shapeErr := modelShapeFromGGUFDims(info.Name, info.Dims)
		if mapped && shapeErr == nil && len(shape) == 2 && shape[1]%32 == 0 && model.IsQuantWeight(canon) {
			q8U, err := estimateNativeQ8LogicalBytes(elemsU)
			if err != nil {
				return HostLoadPeak{}, err
			}
			q8, err := checkedEstimateUint64(q8U, "q8 bytes", info.Name)
			if err != nil {
				return HostLoadPeak{}, err
			}
			if err := add(&row.Q8Bytes, q8); err != nil {
				return HostLoadPeak{}, err
			}
			window, err := sumWindow(payload, f32, f32)
			if err != nil {
				return HostLoadPeak{}, err
			}
			windows = append(windows, window)
			continue
		}
		steady := max(payload, f32)
		if err := add(&row.F32Bytes, steady); err != nil {
			return HostLoadPeak{}, err
		}
		window, err := sumWindow(payload, f32)
		if err != nil {
			return HostLoadPeak{}, err
		}
		windows = append(windows, window)
	}

	for _, r := range byType {
		out.Rows = append(out.Rows, *r)
		if err := add(&out.SteadyAnonBytes, r.AnonBytes()); err != nil {
			return HostLoadPeak{}, err
		}
	}
	sort.Slice(out.Rows, func(i, j int) bool {
		if a, b := out.Rows[i].AnonBytes(), out.Rows[j].AnonBytes(); a != b {
			return a > b
		}
		return out.Rows[i].Type < out.Rows[j].Type
	})
	if err := add(&out.SteadyAnonBytes, min(eligible, max(o.streamedDenseBytes, 0))); err != nil {
		return HostLoadPeak{}, err
	}
	if err := add(&out.SteadyAnonBytes, max(o.streamedExpertBytes, 0)); err != nil {
		return HostLoadPeak{}, err
	}
	out.Workers = loadWorkers()
	if out.WorkerWindowBytes, err = maxContiguousWindowSum(windows, out.Workers); err != nil {
		return HostLoadPeak{}, err
	}
	staging := int64(0)
	if largestLazy > 0 {
		staging, err = sumWindow(largestLazy, min(largestLazy, streamedDenseMaterializeWindowBytes))
		if err != nil {
			return HostLoadPeak{}, err
		}
	}
	out.TransientBytes = max(out.WorkerWindowBytes, staging)
	out.PeakAnonBytes = out.SteadyAnonBytes
	if err := add(&out.PeakAnonBytes, out.TransientBytes); err != nil {
		return HostLoadPeak{}, err
	}
	return out, nil
}

// streamedDenseLazy is the loader's streamed-dense dispatch (computeQ4KTensorWork): an eligible
// dense Q4_K, or a retained eligible Q2_K/Q3_K/Q5_K/Q6_K, becomes a range descriptor.
func streamedDenseLazy(cfg model.Config, o q4kLoadOptions, t TensorType, canon string) bool {
	if !o.streamedDenseQ4K {
		return false
	}
	if t == TensorQ4_K {
		return model.ResidentQ4KEligible(cfg, canon)
	}
	return denseKQuantRetained(o, t) && lazyDenseKQuantBoundedEligible(cfg, t, canon)
}

// StreamedDenseWorkingSetFromHeadroom sizes a bounded streamed-dense working set from the host
// memory left after a load's fixed peak (margin included), holding back reserve of that room. A
// host that cannot hold the fixed peak gets a one-byte bound rather than 0, because a zero working
// set declares stream-through with unbounded memoization; AdmitHostLoadPeak then refuses the load.
func StreamedDenseWorkingSetFromHeadroom(fixed HostLoadPeak, availBytes, marginBytes int64, reserve float64) int64 {
	room := availBytes - max(marginBytes, 0) - fixed.PeakAnonBytes
	return max(int64(float64(room)*(1-reserve)), 1)
}
