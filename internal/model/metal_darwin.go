//go:build darwin && arm64 && cgo

package model

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var qwen35MetalBatchMu sync.Mutex

type metalQwen35ProjectionBatcher struct {
	mode Qwen35MetalBatchMode
	held bool
}

func newQwen35ProjectionBatcher(mode Qwen35MetalBatchMode) qwen35ProjectionBatcher {
	if mode != Qwen35MetalBatchControl && mode != Qwen35MetalBatchMixed {
		return nil
	}
	return &metalQwen35ProjectionBatcher{mode: mode}
}

func (b *metalQwen35ProjectionBatcher) Begin(s *Session) bool {
	if b.held || s == nil || !s.MetalQ4K || !metalgemm.Available() {
		return false
	}
	qwen35MetalBatchMu.Lock()
	b.held = true
	return true
}

func (b *metalQwen35ProjectionBatcher) Finish(*Session) { b.release() }
func (b *metalQwen35ProjectionBatcher) Abort(*Session)  { b.release() }
func (b *metalQwen35ProjectionBatcher) Close(*Session)  { b.release() }

func (b *metalQwen35ProjectionBatcher) release() {
	if !b.held {
		return
	}
	b.held = false
	qwen35MetalBatchMu.Unlock()
}

func nativeEventsDelta(before metalgemm.NativeEvents) Qwen35NativeEvents {
	after := metalgemm.QuantNativeEvents()
	return Qwen35NativeEvents{
		CommandBuffers: after.CommandBuffers - before.CommandBuffers,
		Commits:        after.Commits - before.Commits,
		Waits:          after.Waits - before.Waits,
	}
}

func (b *metalQwen35ProjectionBatcher) control(s *Session, names []string, x []float32, outs []int) ([][]float32, Qwen35NativeEvents, bool) {
	before := metalgemm.QuantNativeEvents()
	out := s.q4kGroupDispatch(names, x, outs)
	if out == nil {
		return nil, Qwen35NativeEvents{}, false
	}
	return out, nativeEventsDelta(before), true
}

func (b *metalQwen35ProjectionBatcher) MulGroup(s *Session, _ int, names []string, x []float32, outs []int) ([][]float32, Qwen35NativeEvents, bool) {
	if len(names) != 3 || len(outs) != 3 || !b.held {
		return nil, Qwen35NativeEvents{}, false
	}
	if b.mode == Qwen35MetalBatchControl {
		return b.control(s, names, x, outs)
	}

	q4ws := make([]*metalgemm.Q4KWeight, 0, 1)
	q4pos := make([]int, 0, 1)
	q8ws := make([]*metalgemm.Q8Weight, 0, 2)
	q8pos := make([]int, 0, 2)
	for i, name := range names {
		if qt := s.M.q4kw[name]; qt != nil {
			w := s.M.metalQ4KWeight(name, qt)
			if w == nil {
				return b.control(s, names, x, outs)
			}
			q4ws = append(q4ws, w)
			q4pos = append(q4pos, i)
			continue
		}
		if s.M.kqw[name] != nil {
			return b.control(s, names, x, outs)
		}
		qt := s.M.q8w[name]
		if qt == nil {
			return b.control(s, names, x, outs)
		}
		w := s.M.metalQ8Weight(name, qt)
		if w == nil {
			return b.control(s, names, x, outs)
		}
		q8ws = append(q8ws, w)
		q8pos = append(q8pos, i)
	}
	if len(q4ws) == 0 || len(q8ws) == 0 {
		return b.control(s, names, x, outs)
	}

	qv := s.quantizeVecQ8(x)
	before := metalgemm.QuantNativeEvents()
	q4out, q8out, status := metalgemm.GEMVGroupMixedQ4KQ8(q4ws, q8ws, x, qv.q, qv.d)
	if status == 0 {
		return b.control(s, names, x, outs)
	}
	if status < 0 {
		panic("model: mixed Q8/Q4_K Metal command buffer failed after submission")
	}
	out := make([][]float32, len(names))
	for i, pos := range q4pos {
		out[pos] = q4out[i]
	}
	for i, pos := range q8pos {
		out[pos] = q8out[i]
	}
	return out, nativeEventsDelta(before), true
}
