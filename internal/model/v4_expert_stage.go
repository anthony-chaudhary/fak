package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var ErrV4ExpertNotSelected = errors.New("v4 expert tensor was not admitted by the selected batch plan")

// ErrV4ExpertDecodeDtype is the fail-closed refusal for a decoder that hands the
// stager a resident tensor whose dtype is not the F32 the ring budget is keyed on.
// Admitting it would silently mis-key the bounded residency accounting.
var ErrV4ExpertDecodeDtype = errors.New("v4 expert decoder returned a non-F32 decoded tensor")

// v4ExpertDecode turns one exact safetensors tensor range into a host tensor for
// the real pagedRing. Official V4 FP4/FP8 decoding is intentionally supplied by
// a separate implementation; the stager never labels fixture F32 as V4 quant.
type v4ExpertDecode func(v4ExpertTensor) (compute.Tensor, error)

// v4ExpertTensorSource is the narrow selective-read contract shared by the
// single-file and indexed-shard sources. Bounds are resolved during planning;
// the stager only needs exact tensor bytes after admission.
type v4ExpertTensorSource interface {
	read(name string) (v4ExpertTensor, error)
	entry(name string) (stEntry, bool)
}

type v4ExpertStageStats struct {
	SourceReads       int64
	SourceBytes       int64
	PageIn            int
	Hits              int
	Evictions         int
	PeakResidentBytes int64
}

type v4ExpertStager struct {
	source   v4ExpertTensorSource
	ring     *pagedRing
	selected map[string]int64
	decode   v4ExpertDecode
	dtype    compute.Dtype
	stats    v4ExpertStageStats
}

func newV4ExpertStager(source *v4ExpertSource, ring *pagedRing, plan v4ExpertBatchPlan, dtype compute.Dtype, decode v4ExpertDecode) (*v4ExpertStager, error) {
	if source == nil || ring == nil || decode == nil {
		return nil, fmt.Errorf("v4 expert stager: nil source, ring, or decoder")
	}
	selected := make(map[string]int64, plan.TensorCount)
	for _, group := range plan.Groups {
		for _, name := range group.TensorNames {
			entry, ok := source.entries[name]
			if !ok {
				return nil, fmt.Errorf("%w: %s", ErrV4ExpertNotSelected, name)
			}
			start, end, err := safetensorsDataBounds(source.file.dataBase, source.file.size, entry)
			if err != nil {
				return nil, err
			}
			selected[name] = end - start
		}
	}
	return &v4ExpertStager{source: source, ring: ring, selected: selected, decode: decode, dtype: dtype}, nil
}

func newV4ShardedExpertStager(source *v4ShardedExpertSource, ring *pagedRing, plan v4ExpertBatchPlan, dtype compute.Dtype, decode v4ExpertDecode) (*v4ExpertStager, error) {
	if source == nil || ring == nil || decode == nil {
		return nil, fmt.Errorf("v4 expert stager: nil source, ring, or decoder")
	}
	selected := make(map[string]int64, plan.TensorCount)
	for _, group := range plan.Groups {
		for _, name := range group.TensorNames {
			shard, ok := source.weightMap[name]
			if !ok {
				return nil, fmt.Errorf("%w: %s", ErrV4ExpertNotSelected, name)
			}
			src, err := source.shard(shard)
			if err != nil {
				return nil, err
			}
			entry, ok := src.entries[name]
			if !ok {
				return nil, fmt.Errorf("%w: %s", ErrV4ExpertNotSelected, name)
			}
			start, end, err := safetensorsDataBounds(src.file.dataBase, src.file.size, entry)
			if err != nil {
				return nil, err
			}
			selected[name] = end - start
		}
	}
	return &v4ExpertStager{source: source, ring: ring, selected: selected, decode: decode, dtype: dtype}, nil
}

// matMul executes one pre-admitted expert tensor. A resident hit reaches
// matMulStaged with a closure that is never invoked, so it performs no source IO
// or decode. A miss reads and decodes before admission; an error therefore cannot
// leave a corrupt resident behind.
//
// The decoder must hand back an F32 tensor: that is the dtype the ring budget is
// keyed on, so a non-F32 resident is refused fail-closed rather than admitted
// against a budget it does not match. On a miss the ring is charged the DECODED
// F32 footprint (f32TensorBytes of the decoded shape), not the packed source
// span, so a bounded ring admits exactly as many resident experts as its budget
// can actually hold.
func (s *v4ExpertStager) matMul(name string, x compute.Tensor) ([]float32, error) {
	weightBytes, ok := s.selected[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrV4ExpertNotSelected, name)
	}
	var decoded compute.Tensor
	residentBytes := weightBytes
	if !s.ring.isResident(name) {
		tensor, err := s.source.read(name)
		if err != nil {
			return nil, err
		}
		s.stats.SourceReads++
		s.stats.SourceBytes += int64(len(tensor.Bytes))
		decoded, err = s.decode(tensor)
		if err != nil {
			return nil, err
		}
		if decoded.Dtype != compute.F32 {
			return nil, fmt.Errorf("%w: %s decoded as %s", ErrV4ExpertDecodeDtype, name, decoded.Dtype)
		}
		decodedBytes, ok := f32TensorBytes(decoded.Shape)
		if !ok {
			return nil, fmt.Errorf("v4 expert stager: %s has an unmaterializable decoded shape %v", name, decoded.Shape)
		}
		residentBytes = decodedBytes
	}
	beforePageIn, beforeHit, beforeEvict := s.ring.pageIn, s.ring.hit, s.ring.evict
	got := s.ring.matMulStaged(name, func() compute.Tensor { return decoded }, s.dtype, x, residentBytes, false)
	if got == nil {
		return nil, fmt.Errorf("v4 expert stager: %s cannot fit ring budget", name)
	}
	s.stats.PageIn += s.ring.pageIn - beforePageIn
	s.stats.Hits += s.ring.hit - beforeHit
	s.stats.Evictions += s.ring.evict - beforeEvict
	if s.ring.used() > s.stats.PeakResidentBytes {
		s.stats.PeakResidentBytes = s.ring.used()
	}
	return got, nil
}

// stageInFlight issues the device transfer for one projection ahead of the GEMM that
// demands it, so the host->device copy overlaps the arithmetic that runs in between
// (#13044). It is the V4 stager's single-weight analogue of the R3 activated-set
// prefetch (expert_ring_prefetch.go): the projection order is known before the first
// GEMM, so a transfer can be put on the wire a GEMM early instead of being awaited
// immediately behind the weight before it.
//
// It resolves the projection through the same resident-first rule matMul uses: an already
// resident weight is skipped (no transfer), and a weight that cannot fit the budget, or
// whose source read/decode fails, is simply not staged — the caller's matMul then falls
// back to its own paging path, which surfaces the error exactly as it would without the
// hint, so a hint can never mask a failure. On a synchronous backend every transfer has
// already landed, so this is a pure reordering of the same uploads and the same resident
// set; the ring's LRU admission remains free to evict unpinned weights as before.
func (s *v4ExpertStager) stageInFlight(name string) {
	if s == nil || s.ring == nil || name == "" {
		return
	}
	if _, ok := s.selected[name]; !ok {
		return
	}
	if s.ring.isResident(name) {
		return
	}
	tensor, err := s.source.read(name)
	if err != nil {
		return
	}
	s.stats.SourceReads++
	s.stats.SourceBytes += int64(len(tensor.Bytes))
	decoded, err := s.decode(tensor)
	if err != nil || decoded.Dtype != compute.F32 {
		return
	}
	decodedBytes, ok := f32TensorBytes(decoded.Shape)
	if !ok {
		return
	}
	beforePageIn, beforeEvict := s.ring.pageIn, s.ring.evict
	if _, ok := s.ring.stageInFlight(name, func() compute.Tensor { return decoded }, s.dtype, decodedBytes, false); !ok {
		return
	}
	s.stats.PageIn += s.ring.pageIn - beforePageIn
	s.stats.Evictions += s.ring.evict - beforeEvict
	if s.ring.used() > s.stats.PeakResidentBytes {
		s.stats.PeakResidentBytes = s.ring.used()
	}
}

func (s *v4ExpertStager) Stats() v4ExpertStageStats { return s.stats }
