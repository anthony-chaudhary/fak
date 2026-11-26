package metalgemm

import (
	"errors"
	"fmt"
	"unsafe"
)

var (
	ErrInk2Closed = errors.New("metalgemm: inkling2 device closed")
	ErrInk2Shape  = errors.New("metalgemm: invalid inkling2 device operand")
	// ErrInk2GraphUnsupported is returned by every graph-capture entry: the
	// deferred encoder already submits a token as one command buffer, so the
	// caller's eager path is the Metal path.
	ErrInk2GraphUnsupported = fmt.Errorf("metalgemm: inkling2 graph capture: %w", errors.ErrUnsupported)
)

// Ink2Buf is a shared-storage MTLBuffer handle. A buffer from Ink2Device.Alloc
// is released with Ink2Device.Free; an adopted handle stays owned by whoever
// allocated it.
type Ink2Buf struct{ h unsafe.Pointer }

// Ink2AdoptBuffer wraps an id<MTLBuffer> allocated elsewhere on the shared
// Metal device.
func Ink2AdoptBuffer(handle unsafe.Pointer) Ink2Buf { return Ink2Buf{h: handle} }

func (b Ink2Buf) Valid() bool { return b.h != nil }

// Ink2Ref addresses float32 element Off of a device buffer. The zero Ref is an
// absent optional operand.
type Ink2Ref struct {
	Buf Ink2Buf
	Off int
}

func (r Ink2Ref) Valid() bool { return r.Buf.h != nil }

// At returns the reference off elements past r.
func (r Ink2Ref) At(off int) Ink2Ref { return Ink2Ref{Buf: r.Buf, Off: r.Off + off} }

// Ink2Weight is a GGUF matrix the packed kernels read in place: Experts
// consecutive Rows x Cols matrices of GGML type Type, ByteOff bytes into Buf.
type Ink2Weight struct {
	Buf                 Ink2Buf
	ByteOff             int
	Type                uint32
	Rows, Cols, Experts int
}

// Ink2DenseF32 views rows x cols float32 values at r as a weight.
func Ink2DenseF32(r Ink2Ref, rows, cols int) Ink2Weight {
	return Ink2Weight{Buf: r.Buf, ByteOff: 4 * r.Off, Rows: rows, Cols: cols, Experts: 1}
}

// Ink2Stats counts device work since the device was opened.
type Ink2Stats struct {
	Submissions, Dispatches uint64
	// Pending is the number of dispatches encoded but not yet submitted.
	Pending    int
	GPUSeconds float64
	// EncodeSeconds is host time spent encoding dispatches and WaitSeconds host
	// time blocked in fences, so a token's wall time is these two plus the
	// caller's own. A wait is QueueSeconds from commit to the GPU starting,
	// GPUSeconds, then TailSeconds until the waiter resumes.
	EncodeSeconds, WaitSeconds float64
	QueueSeconds, TailSeconds  float64
	// SchedSeconds is the driver's scheduling time for the command buffers.
	SchedSeconds float64
}

// The types below mirror the operand structs of compute.Inkling2Device field
// for field, so that package converts them without a per-field copy.

type Ink2AttnParams struct {
	Heads, KVHeads, HeadDim, RelLen int
	QKClip, RelClip, Eps            float32
	Global                          bool
	Alpha, Floor, Scale             float32
}

type Ink2Situ struct {
	Width            int
	Beta, LinearBeta float64
}

type Ink2Controls struct {
	Cur, Msg, Down, Gain, Forget []Ink2Ref
}

type Ink2KVConv struct {
	KW, VW, KDyn, VDyn, KHist, VHist Ink2Ref
	Kn                               int
	KNorm, VNorm                     float32
}
