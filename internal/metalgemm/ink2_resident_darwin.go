//go:build darwin && arm64 && cgo

package metalgemm

import "math"

// The touch kernel reads nothing at run time, but it declares a read of every
// binding, which is what makes Metal wire the buffers for the command buffer;
// a wired buffer stays resident for later ones.
const ink2ResidentTouchSource = `
#include <metal_stdlib>
using namespace metal;
kernel void ink2_resident_touch(
    device const uchar *b0 [[buffer(0)]], device const uchar *b1 [[buffer(1)]],
    device const uchar *b2 [[buffer(2)]], device const uchar *b3 [[buffer(3)]],
    device const uchar *b4 [[buffer(4)]], device const uchar *b5 [[buffer(5)]],
    device const uchar *b6 [[buffer(6)]], device const uchar *b7 [[buffer(7)]],
    device const uchar *b8 [[buffer(8)]], device const uchar *b9 [[buffer(9)]],
    device const uchar *b10 [[buffer(10)]], device const uchar *b11 [[buffer(11)]],
    device const uchar *b12 [[buffer(12)]], device const uchar *b13 [[buffer(13)]],
    device const uchar *b14 [[buffer(14)]], device const uchar *b15 [[buffer(15)]],
    device uint *sink [[buffer(16)]], constant uint &never [[buffer(17)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid == never) {
        sink[0] = b0[0] + b1[0] + b2[0] + b3[0] + b4[0] + b5[0] + b6[0] + b7[0] +
                  b8[0] + b9[0] + b10[0] + b11[0] + b12[0] + b13[0] + b14[0] + b15[0];
    }
}
`

const ink2ResidentTouchArgs = 16

// MakeResident wires bufs for the GPU now and waits, so the first command
// buffer that reads them does not pay the residency (for a mapped GGUF, the
// page-in) before it can start.
func (d *Ink2Device) MakeResident(bufs []Ink2Buf) error {
	var live []Ink2Buf
	for _, b := range bufs {
		if b.Valid() {
			live = append(live, b)
		}
	}
	if len(live) == 0 {
		return nil
	}
	p, err := d.Pipeline(ink2ResidentTouchSource, "ink2_resident_touch")
	if err != nil {
		return err
	}
	sink, err := d.Alloc(1)
	if err != nil {
		return err
	}
	defer d.Free(sink)
	args := make([]Ink2Arg, ink2ResidentTouchArgs+2)
	args[ink2ResidentTouchArgs] = Ink2BufferAt(sink, 0, 4)
	args[ink2ResidentTouchArgs+1] = Ink2Const(uint32(math.MaxUint32))
	for at := 0; at < len(live); at += ink2ResidentTouchArgs {
		group := live[at:min(len(live), at+ink2ResidentTouchArgs)]
		for i := range ink2ResidentTouchArgs {
			args[i] = Ink2BufferAt(group[min(i, len(group)-1)], 0, 0)
		}
		if err := d.Dispatch(p, [3]int{1, 1, 1}, [3]int{1, 1, 1}, args...); err != nil {
			return err
		}
	}
	return d.Flush()
}

// Ink2Buffers is every Metal view the store's matrices alias.
func (s *PackedGGUFStore) Ink2Buffers() []Ink2Buf {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Ink2Buf, len(s.buffers))
	for i, h := range s.buffers {
		out[i] = Ink2Buf{h: h}
	}
	return out
}

// Ink2Buffers is every Metal view of m.
func (m *PackedGGUFMatrix) Ink2Buffers() []Ink2Buf {
	out := make([]Ink2Buf, len(m.views))
	for i, v := range m.views {
		out[i] = Ink2Buf{h: v.handle}
	}
	return out
}
