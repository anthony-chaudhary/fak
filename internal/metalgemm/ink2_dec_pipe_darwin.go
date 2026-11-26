//go:build darwin && arm64 && cgo

package metalgemm

/*
#include "ink2_device.h"
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"time"
	"unsafe"
)

//go:embed ink2_dec_pipe.metal
var ink2PipeSource string

// ErrInk2CommitProfiled refuses an early commit while a per-dispatch profile
// expects every dispatch of a fence in one submission.
var ErrInk2CommitProfiled = fmt.Errorf("metalgemm: inkling2 commit under a dispatch profile: %w", errors.ErrUnsupported)

// ink2EmbedTypes are the GGML types EmbedRow reads.
var ink2EmbedTypes = map[uint32]string{13: "ink2_dec_embed_q5_K"}

// Ink2CanEmbed reports whether EmbedRow reads a table of GGML type typ.
func Ink2CanEmbed(typ uint32) bool { return ink2EmbedTypes[typ] != "" }

// Commit submits the open command buffer without waiting and returns the
// ticket ReadIntsAfter waits on. Later dispatches go to a new command buffer
// that runs after this one.
func (d *Ink2Device) Commit() (uint64, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx == nil {
		return 0, ErrInk2Closed
	}
	if d.prof.mode != Ink2ProfileOff {
		return 0, ErrInk2CommitProfiled
	}
	return uint64(C.mg_ink2_commit(d.ctx)), nil
}

// ReadIntsAfter waits for the command buffers committed up to ticket and reads n
// int32 at src, leaving later command buffers running. The caller guarantees no
// later dispatch writes src.
func (d *Ink2Device) ReadIntsAfter(ticket uint64, src Ink2Ref, n int) ([]int32, error) {
	out := make([]int32, n)
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.ctx == nil {
		return nil, ErrInk2Closed
	}
	start := time.Now()
	left := C.mg_ink2_wait(d.ctx, C.uint64_t(ticket))
	d.wait += time.Since(start)
	if left < 0 {
		return nil, d.lastError("wait")
	}
	if left == 0 {
		d.pending = 0
	}
	host, err := d.view(src, n)
	if err != nil || n == 0 {
		return out, err
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), 4*n), host)
	return out, nil
}

// EmbedRow encodes out[0:w.Cols] = row idx[0] of the table w, dequantized bit
// for bit as the host dequantizes it, and out[w.Cols] = pos as int32 bits. An
// index past the table reads its last row.
func (d *Ink2Device) EmbedRow(w Ink2Weight, idx Ink2Ref, pos int, out Ink2Ref) error {
	name := ink2EmbedTypes[w.Type]
	if name == "" || w.Experts != 1 || !idx.Valid() || !out.Valid() || pos < 0 || pos > math.MaxInt32 {
		return fmt.Errorf("%w: embed row of type %d [%d %d] at %d", ErrInk2Shape, w.Type, w.Rows, w.Cols, pos)
	}
	bytes, err := d.packedShape(w, Ink2Ref{}, 0)
	if err != nil {
		return err
	}
	p, err := d.PipelinePrecise(ink2PipeSource, name)
	if err != nil {
		return err
	}
	shape := struct{ rows, cols, byteOffset, pos uint32 }{uint32(w.Rows), uint32(w.Cols), uint32(w.ByteOff & 3), uint32(pos)}
	return d.DispatchThreads(p, [3]int{w.Cols + 1, 1, 1}, [3]int{min(w.Cols+1, 256, p.MaxThreads()), 1, 1},
		Ink2BufferAt(w.Buf, w.ByteOff&^3, bytes+w.ByteOff&3), idx.Span(1), out.Span(w.Cols+1), Ink2Const(shape))
}

// Ink2Span is n float32 elements at a reference.
type Ink2Span struct {
	At Ink2Ref
	N  int
}

// CopySpans encodes dst[i] = src[i] for each pair, word for word.
func (d *Ink2Device) CopySpans(src, dst []Ink2Span) error {
	if len(src) != len(dst) {
		return fmt.Errorf("%w: copy of %d spans into %d", ErrInk2Shape, len(src), len(dst))
	}
	p, err := d.PipelinePrecise(ink2PipeSource, "ink2_dec_copy8")
	if err != nil {
		return err
	}
	for lo := 0; lo < len(src); lo += 8 {
		var args [17]Ink2Arg
		var shape struct{ n [8]uint32 }
		widest := 0
		for k := range 8 {
			args[k], args[8+k] = Ink2Ref{}.Span(0), Ink2Ref{}.Span(0)
			if lo+k >= len(src) {
				continue
			}
			s, t := src[lo+k], dst[lo+k]
			if s.N != t.N || s.N < 1 || s.N > math.MaxInt32 || !s.At.Valid() || !t.At.Valid() {
				return fmt.Errorf("%w: copy of %d floats into %d", ErrInk2Shape, s.N, t.N)
			}
			args[k], args[8+k] = s.At.Span(s.N), t.At.Span(t.N)
			shape.n[k] = uint32(s.N)
			widest = max(widest, s.N)
		}
		args[16] = Ink2Const(shape)
		if err := d.DispatchThreads(p, [3]int{widest, min(len(src)-lo, 8), 1}, [3]int{min(widest, 256, p.MaxThreads()), 1, 1}, args[:]...); err != nil {
			return err
		}
	}
	return nil
}
