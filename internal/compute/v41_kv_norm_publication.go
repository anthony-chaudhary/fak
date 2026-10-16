package compute

import (
	"encoding/binary"
	"math"
)

const v41KVNormWidth = 512
const v41KVNormImageWords = 1032

type v41KVNormPublicationBackend interface {
	v41TryKVNormPublication(Backend, Tensor, []float32, float32) (*V41KVNormPublication, bool, error)
}

// V41KVNormPublication owns one bounded transient image. Its owner must consume
// input verdict, then validate the complete Raw row, then request Published.
// The model copies the final row in place; no image remains in session state.
type V41KVNormPublication struct{ image []float32 }

func (p *V41KVNormPublication) valid() bool { return p != nil && len(p.image) == v41KVNormImageWords }
func (p *V41KVNormPublication) verdict(offset int, site uint32) error {
	if !p.valid() {
		return &V41BF16CheckedProtocolError{Message: "invalid KV norm image"}
	}
	var raw [16]byte
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(p.image[offset+i]))
	}
	return v41DeferredVerdict("vulkan", V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: v41KVNormWidth, Site: site}, raw[:])
}
func (p *V41KVNormPublication) InputError() error { return p.verdict(0, V41BF16PublishKVNormInput) }
func (p *V41KVNormPublication) Raw() []float32 {
	if !p.valid() {
		return nil
	}
	return p.image[4:516:516]
}
func (p *V41KVNormPublication) Published() ([]float32, error) {
	if err := p.verdict(516, V41BF16PublishKVNormOutput); err != nil {
		return nil, err
	}
	return p.image[520:1032:1032], nil
}

// TryV41KVNormPublication borrows host input and gain. A native pool MISS has no
// side effects; selected transport failure never permits scalar replay.
func TryV41KVNormPublication(be Backend, gain Tensor, input []float32, eps float32) (*V41KVNormPublication, bool, error) {
	if b, ok := be.(v41KVNormPublicationBackend); ok {
		return b.v41TryKVNormPublication(be, gain, input, eps)
	}
	return nil, false, nil
}
