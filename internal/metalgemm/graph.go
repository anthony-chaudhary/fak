//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
typedef struct {
    int committed;
    int completed_wait;
    int encoders;
    int host_readbacks;
    double gpu_milliseconds;
    double wait_milliseconds;
    int timing_available;
} mg_graph_receipt;
void *mg_graph_begin(const float *xf, const signed char *xq, const float *xd, int P, int in);
void *mg_graph_encode_q4k(void *graph, int wid);
void *mg_graph_encode_q6k(void *graph, int wid);
void *mg_graph_encode_q8(void *graph, int wid);
int mg_graph_finish(void *graph, mg_graph_receipt *receipt);
int mg_graph_read(void *graph, void *result, float *dst, int n);
void mg_graph_free(void *graph);
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

var (
	errGraphTerminal = errors.New("metalgemm: graph is terminal")
	errGraphEmpty    = errors.New("metalgemm: graph has no encoded projections")
)

type GraphReceipt struct {
	Committed, CompletedWait, TimingAvailable bool
	Encoders, HostReadbacks                   int
	GPUMilliseconds, WaitMilliseconds         float64
}

type GraphResult struct {
	ptr   unsafe.Pointer
	out   int
	p     int
	graph *ProjectionGraph
}

type ProjectionGraph struct {
	ptr       unsafe.Pointer
	p         int
	encoders  int
	finished  bool
	freed     bool
	readbacks int
}

// BeginProjectionGraph uploads one activation panel for all projections in the graph.
// xf is required by Q4_K/Q6_K; xq/xd are required by Q8. Supplying both permits mixed graphs.
func BeginProjectionGraph(xf []float32, xq []int8, xd []float32, P, in int) (*ProjectionGraph, error) {
	if P <= 0 || in <= 0 || len(xf) != 0 && len(xf) != P*in || len(xq) != 0 && len(xq) != P*in || len(xd) != 0 && len(xd) != P*(in/32) {
		return nil, fmt.Errorf("metalgemm: invalid projection graph panel P=%d in=%d xf=%d xq=%d xd=%d", P, in, len(xf), len(xq), len(xd))
	}
	var pf *C.float
	var pq *C.schar
	var pd *C.float
	if len(xf) > 0 {
		pf = (*C.float)(unsafe.Pointer(&xf[0]))
	}
	if len(xq) > 0 {
		pq = (*C.schar)(unsafe.Pointer(&xq[0]))
	}
	if len(xd) > 0 {
		pd = (*C.float)(unsafe.Pointer(&xd[0]))
	}
	p := C.mg_graph_begin(pf, pq, pd, C.int(P), C.int(in))
	if p == nil {
		return nil, errors.New("metalgemm: graph begin failed")
	}
	return &ProjectionGraph{ptr: p, p: P}, nil
}

func (g *ProjectionGraph) open() error {
	if g == nil || g.ptr == nil || g.finished || g.freed {
		return errGraphTerminal
	}
	return nil
}
func (g *ProjectionGraph) add(ptr unsafe.Pointer, out int) (*GraphResult, error) {
	if ptr == nil {
		return nil, errors.New("metalgemm: graph projection encode failed")
	}
	g.encoders++
	return &GraphResult{ptr: ptr, out: out, p: g.p, graph: g}, nil
}
func (g *ProjectionGraph) EncodeQ4K(w *Q4KWeight) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q4_K weight")
	}
	return g.add(C.mg_graph_encode_q4k(g.ptr, C.int(w.id)), w.Out)
}
func (g *ProjectionGraph) EncodeQ6K(w *Q6KWeight) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q6_K weight")
	}
	return g.add(C.mg_graph_encode_q6k(g.ptr, C.int(w.id)), w.Out)
}
func (g *ProjectionGraph) EncodeQ8(w *Q8Weight) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q8 weight")
	}
	return g.add(C.mg_graph_encode_q8(g.ptr, C.int(w.id)), w.Out)
}
func (g *ProjectionGraph) Finish() (GraphReceipt, error) {
	if err := g.open(); err != nil {
		return GraphReceipt{}, err
	}
	if g.encoders == 0 {
		return GraphReceipt{}, errGraphEmpty
	}
	var r C.mg_graph_receipt
	// Submission consumes the owner even when the device reports an error; never permit a
	// second commit of the same native command buffer.
	g.finished = true
	if C.mg_graph_finish(g.ptr, &r) == 0 {
		return GraphReceipt{}, errors.New("metalgemm: graph submit failed")
	}
	return GraphReceipt{Committed: r.committed != 0, CompletedWait: r.completed_wait != 0, TimingAvailable: r.timing_available != 0, Encoders: int(r.encoders), HostReadbacks: int(r.host_readbacks), GPUMilliseconds: float64(r.gpu_milliseconds), WaitMilliseconds: float64(r.wait_milliseconds)}, nil
}
func (g *ProjectionGraph) Read(r *GraphResult) ([]float32, error) {
	if g == nil || !g.finished || g.freed {
		return nil, errGraphTerminal
	}
	if r == nil || r.graph != g || r.ptr == nil {
		return nil, errors.New("metalgemm: result does not belong to graph")
	}
	out := make([]float32, r.p*r.out)
	if len(out) > 0 && C.mg_graph_read(g.ptr, r.ptr, (*C.float)(unsafe.Pointer(&out[0])), C.int(len(out))) == 0 {
		return nil, errors.New("metalgemm: graph read failed")
	}
	g.readbacks++
	return out, nil
}
func (g *ProjectionGraph) Free() {
	if g != nil && g.ptr != nil && !g.freed {
		C.mg_graph_free(g.ptr)
		g.ptr = nil
		g.freed = true
	}
}
