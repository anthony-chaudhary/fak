//go:build darwin && arm64 && cgo

package metalgemm

/*
int  mg_q5k_ready(void);
int  mg_q5k_upload(const unsigned char* raw, int out, int in, int nocopy);
void mg_q5k_release(int wid);
int  mg_q5k_live_count(void);
void *mg_graph_encode_q5k_from(void *opaque, int wid, void *input, int elems);
*/
import "C"

import (
	"errors"
	"os"
	"runtime"
	"sync"
	"unsafe"
)

// Q5KBlockBytes is the size of one 256-weight Q5_K super-block.
const Q5KBlockBytes = 176

// Q5KWeight is a raw Q5_K matrix [Out, In] resident on the GPU, consumed only by the projection
// graph (q5k_mul_mv at P=1, q5k_mul_mm for a panel). It keeps a Qwen3.5 linear-attention band
// that ships as Q5_K at 5.5 bits/weight instead of a requantized 8.5-bit Q8_0 copy.
type Q5KWeight struct {
	mu      sync.RWMutex
	id      C.int
	Out, In int
	pin     *runtime.Pinner
}

// Q5KAvailable reports whether the Q5_K GEMV and GEMM pipelines compiled on this device.
func Q5KAvailable() bool { return Available() && C.mg_q5k_ready() != 0 }

// UploadQ5KGoOwned makes a row-major Q5_K payload (verbatim GGUF super-blocks, length
// out*(in/256)*176) resident. A page-aligned Go-heap payload whose capacity covers the
// page-rounded bytes is aliased without a copy and pinned until Release; it MUST NOT be mutated
// while the handle is live. Anything else is copied. Returns nil when the kernels are unavailable,
// the shape is invalid, or the payload is short.
func UploadQ5KGoOwned(raw []byte, out, in int) *Q5KWeight {
	if !Q5KAvailable() || in <= 0 || in%256 != 0 || out <= 0 {
		return nil
	}
	need := out * (in / 256) * Q5KBlockBytes
	if len(raw) < need {
		return nil
	}
	page := os.Getpagesize()
	rounded := need
	if page > 1 && need%page != 0 {
		rounded += page - need%page
	}
	w := &Q5KWeight{id: -1, Out: out, In: in}
	if (page <= 1 || uintptr(unsafe.Pointer(&raw[0]))%uintptr(page) == 0) && cap(raw) >= rounded {
		w.pin = new(runtime.Pinner)
		w.pin.Pin(&raw[0])
		w.id = C.mg_q5k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in), 1)
		if w.id < 0 {
			w.pin.Unpin()
			w.pin = nil
		}
	}
	if w.id < 0 {
		w.id = C.mg_q5k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in), 0)
	}
	runtime.KeepAlive(raw)
	if w.id < 0 {
		return nil
	}
	return w
}

// ID returns the backend handle, or -1 once released.
func (w *Q5KWeight) ID() int {
	if w == nil {
		return -1
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return int(w.id)
}

// NoCopy reports whether the handle aliases its Go owner instead of a device copy.
func (w *Q5KWeight) NoCopy() bool {
	if w == nil {
		return false
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.pin != nil
}

// Release drops the native buffer before unpinning an aliased owner. Idempotent.
func (w *Q5KWeight) Release() {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.id >= 0 {
		C.mg_q5k_release(w.id)
		w.id = -1
	}
	if w.pin != nil {
		w.pin.Unpin()
		w.pin = nil
	}
}

// LiveQ5KWeights counts resident native Q5_K slots.
func LiveQ5KWeights() int { return int(C.mg_q5k_live_count()) }

// EncodeQ5KFrom encodes a resident Q5_K projection of a graph-owned f32 panel without
// committing, waiting, or reading it back. The caller keeps w alive until the graph finishes.
func (g *ProjectionGraph) EncodeQ5KFrom(w *Q5KWeight, input *GraphResult) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q5_K weight")
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.id < 0 || input == nil || input.graph != g || input.ptr == nil || input.p != g.p || input.out != w.In {
		return nil, errors.New("metalgemm: invalid graph Q5_K projection input")
	}
	return g.add(C.mg_graph_encode_q5k_from(g.ptr, C.int(w.id), input.ptr, C.int(input.p*input.out)), w.Out)
}
