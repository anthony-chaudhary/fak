//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdlib.h>
#include "ink2_device.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"time"
	"unsafe"
)

// Ink2Device runs the resident inkling2_loop ops on the shared Metal device.
// Every op only encodes onto one open command buffer; Read, ReadInts, Flush
// and a host write into device memory are the fences that submit it and wait.
//
// Ops live as methods in sibling files and build on four calls: Pipeline (or
// PipelinePrecise) to compile and cache a kernel, Ink2Ref.Span / Ink2BufferAt /
// Ink2Const to bind its arguments, and Dispatch (or DispatchThreads) to encode
// a launch.
type Ink2Device struct {
	mu      sync.Mutex
	ctx     unsafe.Pointer
	null    Ink2Buf
	pending int
	libs    map[ink2Source]C.int
	pipes   map[ink2Kernel]Ink2Pipeline
	args    []C.mg_ink2_arg
	consts  []byte
	names   []string
	encode  time.Duration
	wait    time.Duration
	prof    ink2Profiler
}

type ink2Source struct {
	text    *byte
	n       int
	precise bool
}

type ink2Kernel struct {
	src      ink2Source
	function string
}

// Ink2Pipeline is a compiled kernel of one device.
type Ink2Pipeline struct {
	id                int32
	width, maxThreads uint32
}

// ThreadWidth is the SIMD-group width the kernel executes with.
func (p Ink2Pipeline) ThreadWidth() int { return int(p.width) }

// MaxThreads is the largest threadgroup the kernel accepts.
func (p Ink2Pipeline) MaxThreads() int { return int(p.maxThreads) }

// Ink2Arg is one kernel argument. Arguments bind in order: the i-th argument
// of a dispatch is [[buffer(i)]] of the kernel.
type Ink2Arg struct {
	buf       Ink2Buf
	off, need int
	bytes     []byte
}

// Span binds r and requires n float32 elements to be in range.
func (r Ink2Ref) Span(n int) Ink2Arg { return Ink2Arg{buf: r.Buf, off: 4 * r.Off, need: 4 * n} }

// Ink2BufferAt binds b at a byte offset and requires need bytes to be in range.
func Ink2BufferAt(b Ink2Buf, byteOff, need int) Ink2Arg {
	return Ink2Arg{buf: b, off: byteOff, need: need}
}

// Ink2Const binds a copy of v as a constant struct. T must lay out exactly as
// the kernel's `constant T &` argument does.
func Ink2Const[T any](v T) Ink2Arg {
	return Ink2Arg{bytes: append([]byte(nil), unsafe.Slice((*byte)(unsafe.Pointer(&v)), unsafe.Sizeof(v))...)}
}

// NewInk2Device opens a device in the profile mode FAK_INK2_PROFILE names.
func NewInk2Device() (*Ink2Device, error) {
	mode, err := parseInk2ProfileMode(os.Getenv(Ink2ProfileEnv))
	if err != nil {
		return nil, err
	}
	return newInk2Device(mode)
}

func openInk2Device() (*Ink2Device, error) {
	if !Available() {
		return nil, ErrMetalUnavailable
	}
	ctx := C.mg_ink2_create()
	if ctx == nil {
		return nil, ErrMetalUnavailable
	}
	d := &Ink2Device{ctx: ctx, libs: map[ink2Source]C.int{}, pipes: map[ink2Kernel]Ink2Pipeline{}}
	// A kernel cannot take a null pointer, so an absent operand binds this.
	null, err := d.Alloc(16)
	if err != nil {
		d.Close()
		return nil, err
	}
	d.null = null
	// Early commits measured no gain, so decode commits once per fence; the
	// schedule tests still drive them.
	d.setSchedule(!Ink2Legacy("serial"), 0)
	return d, nil
}

// Ink2LegacyEnv lists, comma-separated, the device behaviours to run the
// previous way: "serial" encodes every dispatch on a serial encoder;
// "pipeline" reads each decode token back before encoding the next; the
// routed GEMV ("routed", "scale", "nvfp4w", "routed2"), lean op ("mix",
// "ctrl", "rms", "route", "ctrlmix") and matvec ("mvlat") families parse their
// own names from the same list, and "tinyfuse" keeps the general loops of the
// decode attend and qkv post.
const Ink2LegacyEnv = "FAK_INK2_LEGACY"

// Ink2Legacy reports whether FAK_INK2_LEGACY names name.
func Ink2Legacy(name string) bool {
	for _, f := range strings.Split(os.Getenv(Ink2LegacyEnv), ",") {
		if strings.EqualFold(strings.TrimSpace(f), name) {
			return true
		}
	}
	return false
}

// setSchedule picks the encoder: concurrent places a buffer barrier only
// where a dispatch's bindings overlap an unbarriered earlier dispatch's writes
// (or it writes what one read); commitEvery > 0 commits early. Either keeps
// every dispatch's inputs exactly what the serial encoder gives it.
func (d *Ink2Device) setSchedule(concurrent bool, commitEvery int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	C.mg_ink2_configure(d.ctx, C.int(btoi(concurrent)), C.int(commitEvery), d.null.h)
}

// Close drops any dispatches that were never fenced.
func (d *Ink2Device) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx == nil {
		return
	}
	C.mg_ink2_release(d.null.h)
	C.mg_ink2_close(d.ctx)
	d.ctx, d.null, d.pending = nil, Ink2Buf{}, 0
}

func (d *Ink2Device) lastError(op string) error {
	var text [512]C.char
	n := C.mg_ink2_error(d.ctx, &text[0], C.size_t(len(text)))
	return fmt.Errorf("metalgemm: inkling2 %s: %s", op, C.GoStringN(&text[0], C.int(n)))
}

// Alloc returns a zeroed buffer of n float32 elements.
func (d *Ink2Device) Alloc(n int) (Ink2Buf, error) {
	if n <= 0 || n > math.MaxInt/4 {
		return Ink2Buf{}, fmt.Errorf("%w: alloc of %d floats", ErrInk2Shape, n)
	}
	h := C.mg_ink2_alloc(C.size_t(4 * n))
	if h == nil {
		return Ink2Buf{}, fmt.Errorf("metalgemm: inkling2 alloc of %d floats failed", n)
	}
	return Ink2Buf{h: h}, nil
}

// Free releases a buffer from Alloc. Dispatches already encoded keep it alive.
func (d *Ink2Device) Free(b Ink2Buf) {
	if b.h != nil {
		C.mg_ink2_release(b.h)
	}
}

func (d *Ink2Device) flushLocked() error {
	if d.ctx == nil {
		return ErrInk2Closed
	}
	if d.pending == 0 {
		return nil
	}
	d.pending = 0
	start := time.Now()
	ok := C.mg_ink2_flush(d.ctx) != 0
	d.wait += time.Since(start)
	d.profileFlushed(ok)
	if !ok {
		return d.lastError("flush")
	}
	return nil
}

// Flush submits every pending dispatch as one command buffer and waits for it.
func (d *Ink2Device) Flush() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.flushLocked()
}

// host fences and returns the host view of n elements at r.
func (d *Ink2Device) host(r Ink2Ref, n int) ([]byte, error) {
	if err := d.flushLocked(); err != nil {
		return nil, err
	}
	return d.view(r, n)
}

// view is the host view of n elements at r, without a fence.
func (d *Ink2Device) view(r Ink2Ref, n int) ([]byte, error) {
	if !r.Valid() || r.Off < 0 || n < 0 {
		return nil, fmt.Errorf("%w: %d floats at offset %d", ErrInk2Shape, n, r.Off)
	}
	var bytes C.size_t
	base := C.mg_ink2_span(r.Buf.h, &bytes)
	if base == nil || r.Off+n > int(bytes)/4 {
		return nil, fmt.Errorf("%w: %d floats at offset %d of a %d-byte buffer", ErrInk2Shape, n, r.Off, int(bytes))
	}
	return unsafe.Slice((*byte)(unsafe.Add(base, 4*r.Off)), 4*n), nil
}

// Write copies src into device memory. Pending dispatches are fenced first so
// that the ones already encoded still see the old contents.
func (d *Ink2Device) Write(dst Ink2Ref, src []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	host, err := d.host(dst, len(src))
	if err != nil || len(src) == 0 {
		return err
	}
	copy(host, unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), 4*len(src)))
	return nil
}

// Zero clears n elements at dst, with the fence of Write.
func (d *Ink2Device) Zero(dst Ink2Ref, n int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	host, err := d.host(dst, n)
	clear(host)
	return err
}

func (d *Ink2Device) Read(src Ink2Ref, n int) ([]float32, error) {
	out := make([]float32, n)
	d.mu.Lock()
	defer d.mu.Unlock()
	host, err := d.host(src, n)
	if err != nil || n == 0 {
		return out, err
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), 4*n), host)
	return out, nil
}

func (d *Ink2Device) ReadInts(src Ink2Ref, n int) ([]int32, error) {
	out := make([]int32, n)
	d.mu.Lock()
	defer d.mu.Unlock()
	host, err := d.host(src, n)
	if err != nil || n == 0 {
		return out, err
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), 4*n), host)
	return out, nil
}

func (d *Ink2Device) Stats() Ink2Stats {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx == nil {
		return Ink2Stats{}
	}
	var s C.mg_ink2_stats
	C.mg_ink2_get_stats(d.ctx, &s)
	return Ink2Stats{Submissions: uint64(s.submissions), Dispatches: uint64(s.dispatches),
		Barriers: uint64(s.barriers), Commits: uint64(s.commits),
		Pending: int(s.pending), GPUSeconds: float64(s.gpu_seconds),
		EncodeSeconds: d.encode.Seconds(), WaitSeconds: d.wait.Seconds(),
		QueueSeconds: float64(s.queue_seconds), TailSeconds: float64(s.tail_seconds), SchedSeconds: float64(s.kernel_seconds)}
}

// Pipeline returns the kernel named function of an MSL source, compiling the
// source once per device. Pass the same string value on every call (an
// embedded file or a package-level variable): the cache keys on its storage,
// not its text.
func (d *Ink2Device) Pipeline(source, function string) (Ink2Pipeline, error) {
	return d.pipeline(source, function, false)
}

// PipelinePrecise is Pipeline with fast math off, for a kernel whose result
// depends on evaluation order or on infinities and NaNs surviving.
func (d *Ink2Device) PipelinePrecise(source, function string) (Ink2Pipeline, error) {
	return d.pipeline(source, function, true)
}

func (d *Ink2Device) pipeline(source, function string, precise bool) (Ink2Pipeline, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx == nil {
		return Ink2Pipeline{}, ErrInk2Closed
	}
	key := ink2Kernel{src: ink2Source{text: unsafe.StringData(source), n: len(source), precise: precise}, function: function}
	if p, ok := d.pipes[key]; ok {
		return p, nil
	}
	lib, ok := d.libs[key.src]
	if !ok {
		text := C.CString(source)
		lib = C.mg_ink2_library(d.ctx, text, C.int(btoi(precise)))
		C.free(unsafe.Pointer(text))
		if lib < 0 {
			return Ink2Pipeline{}, d.lastError("library")
		}
		d.libs[key.src] = lib
	}
	name := C.CString(function)
	defer C.free(unsafe.Pointer(name))
	var width, maxThreads C.uint32_t
	id := C.mg_ink2_pipeline(d.ctx, lib, name, &width, &maxThreads)
	if id < 0 {
		return Ink2Pipeline{}, d.lastError("pipeline")
	}
	p := Ink2Pipeline{id: int32(id) + 1, width: uint32(width), maxThreads: uint32(maxThreads)}
	for len(d.names) <= int(id) {
		d.names = append(d.names, "")
	}
	d.names[id] = function
	d.pipes[key] = p
	return p, nil
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Dispatch encodes one launch of p over groups threadgroups of perGroup
// threads each. A later launch reads what an earlier one wrote: launches run
// as if in encode order. Ordering follows the bound extents and the kernel's
// const qualifiers, so an argument must declare every byte the kernel may
// touch (a zero length means the rest of the buffer), and a kernel must not
// write through a pointer it declares device const.
func (d *Ink2Device) Dispatch(p Ink2Pipeline, groups, perGroup [3]int, args ...Ink2Arg) error {
	return d.dispatch(p, groups, perGroup, false, args)
}

// DispatchThreads encodes one launch of p over exactly threads grid threads,
// split into perGroup-sized threadgroups with smaller ones at the edges.
func (d *Ink2Device) DispatchThreads(p Ink2Pipeline, threads, perGroup [3]int, args ...Ink2Arg) error {
	return d.dispatch(p, threads, perGroup, true, args)
}

func (d *Ink2Device) dispatch(p Ink2Pipeline, size, perGroup [3]int, exact bool, args []Ink2Arg) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx == nil {
		return ErrInk2Closed
	}
	if p.id == 0 || len(args) == 0 {
		return fmt.Errorf("%w: dispatch without a pipeline or arguments", ErrInk2Shape)
	}
	start := time.Now()
	var dims [6]C.uint32_t
	group := 1
	for i := range 3 {
		if size[i] < 1 || size[i] > math.MaxUint32 || perGroup[i] < 1 || perGroup[i] > int(p.maxThreads) {
			return fmt.Errorf("%w: dispatch of %v by %v", ErrInk2Shape, size, perGroup)
		}
		dims[i], dims[3+i] = C.uint32_t(size[i]), C.uint32_t(perGroup[i])
		group *= perGroup[i]
	}
	if group > int(p.maxThreads) {
		return fmt.Errorf("%w: threadgroup %v exceeds %d threads", ErrInk2Shape, perGroup, p.maxThreads)
	}
	d.args, d.consts = d.args[:0], d.consts[:0]
	for i, a := range args {
		switch {
		case len(a.bytes) > 0:
			d.args = append(d.args, C.mg_ink2_arg{offset: C.size_t(len(d.consts)), length: C.size_t(len(a.bytes))})
			d.consts = append(d.consts, a.bytes...)
		case a.bytes != nil || a.off < 0 || a.need < 0:
			return fmt.Errorf("%w: dispatch argument %d", ErrInk2Shape, i)
		case a.buf.h == nil:
			d.args = append(d.args, C.mg_ink2_arg{buffer: C.uintptr_t(uintptr(d.null.h))})
		default:
			d.args = append(d.args, C.mg_ink2_arg{buffer: C.uintptr_t(uintptr(a.buf.h)), offset: C.size_t(a.off), length: C.size_t(a.need)})
		}
	}
	if C.mg_ink2_dispatch(d.ctx, C.int(p.id-1), &d.args[0], C.int(len(d.args)),
		unsafe.Pointer(unsafe.SliceData(d.consts)), &dims[0], C.int(btoi(exact))) == 0 {
		return d.lastError("dispatch")
	}
	d.pending++
	encode := time.Since(start)
	d.encode += encode
	if d.prof.mode != Ink2ProfileOff {
		return d.profileDispatch(d.names[p.id-1], size, perGroup, exact, encode)
	}
	return nil
}

// Barrier orders the launches encoded so far before the ones that follow,
// whether or not their bindings overlap.
func (d *Ink2Device) Barrier() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx != nil && d.pending > 0 {
		C.mg_ink2_barrier(d.ctx)
	}
}

// The store's own source minus the panel and fused-MoE kernels, which no
// inkling2 op launches. Built once so the pipeline cache sees one string.
var ink2PackedSource = sync.OnceValue(func() string { return packedCodebooks + packedDequant + packedGEMV })

var ink2MVKernel, ink2RoutedKernel = ink2PackedKernels("mv_"), ink2PackedKernels("routed_")

// ink2v_pair_rows_, the rows kernel of a chunk of up to ink2VerifyPairMax rows.
var ink2VerifyPairRowsKernel = ink2PackedKernels("ink2v_pair_rows_")

// ink2VerifyExpertsKernel has the block types only: no expert stack is scalar.
var ink2VerifyExpertsKernel = func() [41]string {
	k := ink2PackedKernels("ink2v_experts_")
	k[0], k[1], k[30] = "", "", ""
	return k
}()

func ink2PackedKernels(prefix string) (names [41]string) {
	for typ, name := range map[int]string{0: "f32", 1: "f16", 8: "q8_0", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
		17: "iq2_xs", 18: "iq3_xxs", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 30: "bf16", 40: "nvfp4"} {
		names[typ] = prefix + name
	}
	return names
}

const ink2ExpertScaleSource = `#include <metal_stdlib>
using namespace metal;

kernel void ink2_expert_scale(device float *y [[buffer(0)]],
                              device const uint *ids [[buffer(1)]],
                              device const float *scale [[buffer(2)]],
                              constant uint &rows [[buffer(3)]],
                              uint2 at [[thread_position_in_grid]]) {
    y[at.y * rows + at.x] *= scale[ids[at.y]];
}
`

type ink2PackedShape struct{ rows, cols, byteOffset uint32 }

// packedShape validates a weight for the packed kernels and returns the bytes
// one expert occupies. Slot t reads its activation at x + t*xStride.
func (d *Ink2Device) packedShape(w Ink2Weight, x Ink2Ref, xStride int) (int, error) {
	if !w.Buf.Valid() || w.ByteOff < 0 || w.Rows < 1 || w.Rows > math.MaxInt32 || w.Cols < 1 || w.Cols > math.MaxInt32 ||
		w.Experts < 1 || int(w.Type) >= len(ink2MVKernel) || ink2MVKernel[w.Type] == "" {
		return 0, fmt.Errorf("%w: weight type %d [%d %d] x%d", ErrInk2Shape, w.Type, w.Rows, w.Cols, w.Experts)
	}
	block, blockBytes, err := packedBlock(w.Type)
	if err != nil {
		return 0, err
	}
	// The block kernels load the activation as float4 vectors.
	if w.Cols%block != 0 || (block > 1 && (x.Off%4 != 0 || xStride%4 != 0)) {
		return 0, fmt.Errorf("%w: type %d weight of %d columns with the activation at offset %d, stride %d",
			ErrInk2Shape, w.Type, w.Cols, x.Off, xStride)
	}
	return w.Rows * (w.Cols / block * blockBytes), nil
}

// packed is packedShape plus the weight type's pipeline from kernels.
func (d *Ink2Device) packed(w Ink2Weight, kernels *[41]string, x Ink2Ref, xStride int) (Ink2Pipeline, int, error) {
	if int(w.Type) >= len(kernels) || kernels[w.Type] == "" {
		return Ink2Pipeline{}, 0, fmt.Errorf("%w: weight type %d [%d %d] x%d", ErrInk2Shape, w.Type, w.Rows, w.Cols, w.Experts)
	}
	bytes, err := d.packedShape(w, x, xStride)
	if err != nil {
		return Ink2Pipeline{}, 0, err
	}
	p, err := d.Pipeline(ink2PackedSource(), kernels[w.Type])
	if err != nil {
		return Ink2Pipeline{}, 0, err
	}
	if p.width != 32 || p.maxThreads < 128 {
		return Ink2Pipeline{}, 0, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of 128 threads", kernels[w.Type])
	}
	return p, bytes, nil
}

func (w Ink2Weight) args(x Ink2Arg, y Ink2Arg, bytes int) []Ink2Arg {
	// A Metal binding offset is 4-byte aligned; the kernel adds the remainder.
	return []Ink2Arg{Ink2BufferAt(w.Buf, w.ByteOff&^3, bytes+w.ByteOff&3), x, y,
		Ink2Const(ink2PackedShape{uint32(w.Rows), uint32(w.Cols), uint32(w.ByteOff & 3)})}
}

// MatVec encodes y = W x for the first expert of w.
func (d *Ink2Device) MatVec(w Ink2Weight, x, y Ink2Ref) error {
	p, bytes, err := d.packed(w, &ink2MVKernel, x, 0)
	if err != nil {
		return err
	}
	if !x.Valid() || !y.Valid() {
		return fmt.Errorf("%w: matvec without an input or output", ErrInk2Shape)
	}
	d.profileDetail("%dx%d", w.Rows, w.Cols)
	if pl := ink2MVPick(w.Type, w.Cols); pl.kernel != "" {
		return d.matVecDec(pl, w, x, y)
	}
	return d.Dispatch(p, [3]int{(w.Rows + 3) / 4, 1, 1}, [3]int{128, 1, 1}, w.args(x.Span(w.Cols), y.Span(w.Rows), bytes)...)
}

// MatVecQ8 declines: without a shared-codes kernel the caller's plain MatVec is
// the Q8_0 path.
func (d *Ink2Device) MatVecQ8(w Ink2Weight, x, gate, q Ink2Ref, quantize bool, y Ink2Ref) (bool, error) {
	return false, nil
}

// ExpertMatVec encodes, for each of k routed slots t, y[t] = W[ids[t]] x[t]
// with x[t] at x + t*xStride and y[t] at y + t*w.Rows, times the optional
// per-expert scale. It declines (false) a fused SITU input: the caller's
// separate Situ op feeds the same product.
func (d *Ink2Device) ExpertMatVec(w Ink2Weight, ids, x Ink2Ref, xStride int, y Ink2Ref, k int, scale Ink2Ref, situ *Ink2Situ) (bool, error) {
	if situ != nil {
		return false, nil
	}
	if pl := ink2DecRoutedPick(w.Type, w.Cols); pl.kernel != "" {
		if _, err := d.packedShape(w, x, xStride); err != nil {
			return false, err
		}
		d.profileDetail("%dx%d k=%d of %d", w.Rows, w.Cols, k, w.Experts)
		if !ink2DecLegacy.scale || !scale.Valid() {
			return true, d.routedDec(pl, w, ids, x, xStride, y, k, scale)
		}
		if err := d.routedDec(pl, w, ids, x, xStride, y, k, Ink2Ref{}); err != nil {
			return false, err
		}
		return true, d.ExpertScale(y, ids, scale, k, w.Rows)
	}
	p, bytes, err := d.packed(w, &ink2RoutedKernel, x, xStride)
	if err != nil {
		return false, err
	}
	if !ids.Valid() || !x.Valid() || !y.Valid() || k < 1 || xStride < 0 || xStride > math.MaxUint32 {
		return false, fmt.Errorf("%w: expert matvec of %d slots, input stride %d", ErrInk2Shape, k, xStride)
	}
	d.profileDetail("%dx%d k=%d of %d", w.Rows, w.Cols, k, w.Experts)
	args := append(w.args(x.Span((k-1)*xStride+w.Cols), y.Span(k*w.Rows), w.Experts*bytes), ids.Span(k), Ink2Const(uint32(xStride)))
	if err := d.Dispatch(p, [3]int{(w.Rows + 3) / 4, k, 1}, [3]int{128, 1, 1}, args...); err != nil {
		return false, err
	}
	if !scale.Valid() {
		return true, nil
	}
	return true, d.ExpertScale(y, ids, scale, k, w.Rows)
}

// ExpertScale multiplies each routed slot's rows values by scale[ids[slot]].
func (d *Ink2Device) ExpertScale(y, ids, scale Ink2Ref, k, rows int) error {
	if !y.Valid() || !ids.Valid() || !scale.Valid() || k < 1 || rows < 1 || rows > math.MaxUint32 {
		return fmt.Errorf("%w: expert scale of %d slots by %d rows", ErrInk2Shape, k, rows)
	}
	p, err := d.Pipeline(ink2ExpertScaleSource, "ink2_expert_scale")
	if err != nil {
		return err
	}
	// The scale table's length is the expert count, which only the ids know.
	return d.DispatchThreads(p, [3]int{rows, k, 1}, [3]int{min(rows, p.MaxThreads()), 1, 1},
		y.Span(k*rows), ids.Span(k), scale.Span(0), Ink2Const(uint32(rows)))
}

// Ink2Weight returns one expert of m as a weight view.
func (m *PackedGGUFMatrix) Ink2Weight(expert int) (Ink2Weight, error) {
	if expert < 0 || expert >= m.experts {
		return Ink2Weight{}, fmt.Errorf("%w: expert %d of %d", ErrPackedGGUFShape, expert, m.experts)
	}
	v := m.views[expert/m.expertsPerView]
	return Ink2Weight{Buf: Ink2Buf{h: v.handle}, ByteOff: v.offset + (expert%m.expertsPerView)*m.stride,
		Type: m.typ, Rows: m.rows, Cols: m.cols, Experts: 1}, nil
}

// Ink2Stack returns the whole expert stack of m, or false when the stack is
// split across Metal buffers and so cannot be indexed by a device-side id.
func (m *PackedGGUFMatrix) Ink2Stack() (Ink2Weight, bool) {
	if len(m.views) != 1 {
		return Ink2Weight{}, false
	}
	return Ink2Weight{Buf: Ink2Buf{h: m.views[0].handle}, ByteOff: m.views[0].offset,
		Type: m.typ, Rows: m.rows, Cols: m.cols, Experts: m.experts}, true
}

// Dims reports the GGML type and the shape of m.
func (m *PackedGGUFMatrix) Dims() (ggmlType uint32, rows, cols, experts int) {
	return m.typ, m.rows, m.cols, m.experts
}
