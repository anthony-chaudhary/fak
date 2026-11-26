//go:build darwin && arm64 && cgo

package ggufload

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// embedding gives the token table a device view when a decode forward can read
// it in place; FAK_INK2_LEGACY=pipeline keeps it host-only, which leaves decode
// unpipelined.
func (s *inkling2MetalDeviceSet) embedding(q inkling2QuantEmbedding) (inkling2DeviceEmbedding, bool) {
	if metalgemm.Ink2Legacy("pipeline") || !metalgemm.Ink2CanEmbed(uint32(q.m.typ)) {
		return inkling2DeviceEmbedding{}, false
	}
	pm, ok := s.packed(q.m.typ, q.m.raw, q.m.rows, q.m.cols, 1)
	if !ok {
		return inkling2DeviceEmbedding{}, false
	}
	t, err := compute.MetalInkling2Weight(pm, 0)
	if err != nil {
		return inkling2DeviceEmbedding{}, false
	}
	return inkling2DeviceEmbedding{inkling2QuantEmbedding: q, table: t}, true
}
