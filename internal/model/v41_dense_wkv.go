package model

import (
	"fmt"
	"math"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// V41DenseWKVPayloadBytes bounds the exact E4M3 [512,5120] codes and
// row-major UE8M0 [16,160] tile scales; no F32 weight expansion is retained.
const V41DenseWKVPayloadBytes = 512*5120 + 16*160

// V41DenseWKV owns an immutable pair, independent of the originating shard.
// This explicit intake does not register a model or admit native1017 serving.
type V41DenseWKV struct{ codes, scales []byte }

// ReadV41DenseWKV reads one exact pair. The caller must serialize shard Close
// against this method. A positive combined bound is mandatory, including when
// the underlying generic reader permits a legacy unbounded call.
func (s *V41PackedShard) ReadV41DenseWKV(layer int, bound int64) (*V41DenseWKV, error) {
	if s == nil || s.file == nil || layer < 0 || bound < V41DenseWKVPayloadBytes {
		return nil, fmt.Errorf("%w: dense WKV requires live shard, layer and positive pair bound", ErrV41PackedUnsupported)
	}
	name := layerName(layer, "attn.wkv.weight")
	t, err := s.Reference(name)
	if err != nil {
		return nil, err
	}
	r := t.Ref
	if r.Dtype != "F8_E4M3" || r.PackedWeight || !sameShape(r.Shape, []int{512, 5120}) || !sameShape(r.LogicalShape, []int{512, 5120}) || r.ScaleName != layerName(layer, "attn.wkv.scale") || r.ScaleDtype != "F8_E8M0" || !sameShape(r.ScaleShape, []int{16, 160}) || t.ByteLen() != 512*5120 || t.ScaleEnd-t.ScaleStart != 16*160 {
		return nil, fmt.Errorf("%w: incompatible dense WKV pair", ErrV41PackedUnsupported)
	}
	codes, err := s.Read(t, 512*5120)
	if err != nil {
		return nil, err
	}
	scales, err := s.ReadScale(t, 16*160)
	if err != nil {
		return nil, err
	}
	if err := validateV41DenseWKV(codes, scales); err != nil {
		return nil, err
	}
	// ReaderAt already returns owned bytes; mmap views require a bounded copy.
	if s.file.data != nil {
		codes = append([]byte(nil), codes...)
		scales = append([]byte(nil), scales...)
	}
	return &V41DenseWKV{codes: codes, scales: scales}, nil
}

func validateV41DenseWKV(codes, scales []byte) error {
	if len(codes) != 512*5120 || len(scales) != 16*160 {
		return fmt.Errorf("%w: dense WKV payload length", ErrV41PackedUnsupported)
	}
	var factors [16 * 160]float32
	for i, v := range scales {
		if v == 255 {
			return fmt.Errorf("%w: nonfinite dense WKV scale", ErrV41PackedUnsupported)
		}
		factors[i] = float32(math.Ldexp(1, int(v)-127))
	}
	for i, v := range codes {
		f := fp8E4M3ToF32(v) * factors[(i/5120/32)*160+(i%5120)/32]
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return fmt.Errorf("%w: nonfinite dense WKV value at %d", ErrV41PackedUnsupported, i)
		}
	}
	return nil
}

// Upload requires qualified production capability. The backend consumes the
// read-only slices synchronously and must own any retained device allocations.
func (w *V41DenseWKV) Upload(b compute.V41FP8WKVBackend) (compute.V41FP8WKVWeights, error) {
	if b == nil || !b.V41FP8WKVQualified() {
		return nil, fmt.Errorf("%w: dense WKV is not qualified", ErrV41PackedUnsupported)
	}
	return w.UploadForWitness(b)
}

// UploadForWitness is an explicit physical-validation seam, not serving admission.
func (w *V41DenseWKV) UploadForWitness(b compute.V41FP8WKVBackend) (compute.V41FP8WKVWeights, error) {
	if w == nil || len(w.codes) != 512*5120 || len(w.scales) != 16*160 || b == nil || !b.V41FP8WKVSupported() {
		return nil, fmt.Errorf("%w: dense WKV unsupported or absent", ErrV41PackedUnsupported)
	}
	return b.UploadV41FP8WKV(w.codes, w.scales)
}

// V41DenseWKVProjection retains one upload across repeated projections. The
// mutex orders launch/retirement; errors never trigger a host fallback or upload.
type V41DenseWKVProjection struct {
	mu      sync.Mutex
	backend compute.V41FP8WKVBackend
	weights compute.V41FP8WKVWeights
}

func (w *V41DenseWKV) Bind(b compute.V41FP8WKVBackend) (*V41DenseWKVProjection, error) {
	weights, err := w.Upload(b)
	if err != nil {
		return nil, err
	}
	if weights == nil {
		return nil, fmt.Errorf("%w: missing uploaded WKV owner", ErrV41PackedUnsupported)
	}
	return &V41DenseWKVProjection{backend: b, weights: weights}, nil
}

func (w *V41DenseWKV) BindForWitness(b compute.V41FP8WKVBackend) (*V41DenseWKVProjection, error) {
	weights, err := w.UploadForWitness(b)
	if err != nil {
		return nil, err
	}
	if weights == nil {
		return nil, fmt.Errorf("%w: missing uploaded WKV owner", ErrV41PackedUnsupported)
	}
	return &V41DenseWKVProjection{backend: b, weights: weights}, nil
}

func (p *V41DenseWKVProjection) Project(input compute.Tensor, rows int) (compute.Tensor, error) {
	if p == nil {
		return compute.Tensor{}, fmt.Errorf("%w: nil WKV projection", ErrV41PackedUnsupported)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.weights == nil || p.backend == nil {
		return compute.Tensor{}, fmt.Errorf("%w: released WKV projection", ErrV41PackedUnsupported)
	}
	return p.backend.V41FP8WKV(p.weights, input, rows)
}

// Release retains the handle on failure so native quarantine/recovery remains
// observable. Only a successful release transitions this owner to closed.
func (p *V41DenseWKVProjection) Release() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.weights == nil {
		return nil
	}
	if err := p.weights.Release(); err != nil {
		return err
	}
	p.weights = nil
	p.backend = nil
	return nil
}
