package compute

import (
	"encoding/binary"
	"math"
)

const v41IndexQueryPublicationCount = 8192

type v41IndexQueryPublicationBackend interface {
	v41TryIndexQueryPublication(Backend, Tensor, Buffer, Tensor) (*V41IndexQueryPublication, bool, error)
}

// V41IndexQueryPublication keeps raw projection evidence independent of the
// published allocation. Publication status is deliberately not decoded by the
// transport: callers must finish validating the entire raw projection first.
type V41IndexQueryPublication struct {
	raw         []float32
	publication []float32
}

func (p *V41IndexQueryPublication) Raw() []float32 {
	if p == nil {
		return nil
	}
	return p.raw[:len(p.raw):len(p.raw)]
}

// Published decodes the checked envelope only when explicitly requested after
// raw validation. Its returned storage retains a 16-byte status prefix, not raw.
func (p *V41IndexQueryPublication) Published() ([]float32, error) {
	if p == nil || len(p.raw) != v41IndexQueryPublicationCount || len(p.publication) != v41IndexQueryPublicationCount+4 {
		return nil, &V41BF16CheckedProtocolError{Message: "invalid resident index-query publication"}
	}
	var status [16]byte
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(status[4*i:], math.Float32bits(p.publication[i]))
	}
	if err := v41DeferredVerdict("vulkan", V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: v41IndexQueryPublicationCount, Site: V41BF16PublishIndexerProjection}, status[:]); err != nil {
		return nil, err
	}
	return p.publication[4:len(p.publication):len(p.publication)], nil
}

// TryV41IndexQueryPublication borrows a live projection and its protected
// operands. Only an exact canonical backend can select. A native capacity miss
// leaves the same projection available for ordinary readback. Selected faults
// never authorize replay. A defensive post-probe miss may allocate host memory.
func TryV41IndexQueryPublication(be Backend, projection Tensor, weight Buffer, input Tensor) (*V41IndexQueryPublication, bool, error) {
	if b, ok := be.(v41IndexQueryPublicationBackend); ok {
		return b.v41TryIndexQueryPublication(be, projection, weight, input)
	}
	return nil, false, nil
}
