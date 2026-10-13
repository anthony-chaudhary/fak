package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Fail-closed refusals of the session's device KV mirror (#13821). Each is
// surfaced through the selected shared-attention operation error; none permits
// host replay.
var (
	errV41ResidentKVDesync   = errors.New("model: V4.1 resident KV mirror is out of step with the host ring")
	errV41ResidentKVBackend  = errors.New("model: V4.1 resident KV backend operation failed")
	errV41ResidentKVCapacity = errors.New("model: V4.1 resident KV request exceeds the mirror capacity")
)

// v41ResidentKVDisabled keeps every decode step on the per-call upload path.
// Tests flip it to compare the two paths; production leaves it false.
var v41ResidentKVDisabled bool

// v41ResidentKVView names where a decode step's flat KV rows live. Flat rows
// [0, windowRows) are the window keys firstKey..pos, the last being this step's
// new row; the remaining compressedRows rows are compressed groups 0..n-1. The
// host ring stays the source of truth: the mirror only ever receives rows from
// the request's flat KV or synchronously borrowed row view.
type v41ResidentKVView struct {
	owner          *V41AttentionState
	pos            int
	ringRows       int
	firstKey       int
	windowRows     int
	compressedRows int
	sinkKey        string
}

// v41ResidentKVLayer mirrors one layer: a key-addressed ring of ringRows+1
// slots (slot = key % (ringRows+1), so the new row never evicts a visible key)
// and an append-only compressed log. nextPos is the position the mirror expects
// next; any other position, or a different host state, reseeds from the flat KV.
type v41ResidentKVLayer struct {
	owner    *V41AttentionState
	nextPos  int
	ring     compute.Tensor
	ringKeys []int
	log      compute.Tensor
	logRows  int
	logCap   int
}

type v41ResidentKV struct {
	s      *Session
	ops    compute.V41RowLogBackend
	layers map[int]*v41ResidentKVLayer
}

func (s *Session) newV41ResidentKV() *v41ResidentKV {
	ops, ok := s.Backend.(compute.V41RowLogBackend)
	if !ok || !ops.V41RowLogAvailable() {
		return nil
	}
	return &v41ResidentKV{s: s, ops: ops, layers: map[int]*v41ResidentKVLayer{}}
}

func v41ResidentKVKey(layer int, part string) string {
	return fmt.Sprintf("v41-resident-kv/%d/%s", layer, part)
}

// adopt registers a mirror tensor with the session so Close frees it, freeing
// any tensor it replaces.
func (r *v41ResidentKV) adopt(key string, t compute.Tensor) {
	if old, ok := r.s.halW[key]; ok && old.Buf() != t.Buf() {
		if _, borrowed := r.s.borrowedHALW[key]; !borrowed {
			r.s.Backend.Free(old)
		}
	}
	r.s.halW[key] = t
}

func (r *v41ResidentKV) current(key string, t compute.Tensor) bool {
	held, ok := r.s.halW[key]
	return ok && t.Buf() != nil && held.Buf() == t.Buf()
}

func (r *v41ResidentKV) alloc(rows, width int) (compute.Tensor, error) {
	t, err := r.ops.V41RowLogAlloc(rows, width)
	if err != nil {
		return compute.Tensor{}, fmt.Errorf("%w: alloc: %w", errV41ResidentKVBackend, err)
	}
	return t, nil
}

func (r *v41ResidentKV) copyRows(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	if err := r.ops.V41RowLogCopy(dst, dstRow, src, srcRow, rows); err != nil {
		return fmt.Errorf("%w: copy: %w", errV41ResidentKVBackend, err)
	}
	return nil
}

// write uploads flat rows [from, from+rows) once and copies them to dst at
// dstRow. The staging tensor is released before returning.
func (r *v41ResidentKV) write(dst compute.Tensor, dstRow int, kv []float32, rowKV *v41SharedAttentionRows, width, from, rows int) error {
	var values []float32
	if rowKV == nil {
		values = kv[from*width : (from+rows)*width]
	} else {
		// Pack only the missing span, never the full retained history.
		values = make([]float32, 0, rows*width)
		for i := from; i < from+rows; i++ {
			values = append(values, rowKV.row(i)...)
		}
	}
	staged := r.s.uploadHostF32([]int{rows, width}, values, compute.MemoryActivation, "V4.1 resident KV rows")
	defer r.s.Backend.Free(staged)
	return r.copyRows(dst, dstRow, staged, 0, rows)
}

// panel returns a fresh caller-owned [len(sourceRows), width] device panel of
// the selected flat rows, uploading only rows the mirror does not yet hold.
func (r *v41ResidentKV) panel(layer int, view *v41ResidentKVView, kv []float32, rowKV *v41SharedAttentionRows, width int, sourceRows []int) (compute.Tensor, error) {
	flatRows := len(kv) / width
	shapeOK := len(kv) == flatRows*width
	if rowKV != nil {
		flatRows = rowKV.len()
		shapeOK = kv == nil && len(rowKV.window) == view.windowRows && len(rowKV.compressed) == view.compressedRows
	}
	if view.owner == nil || view.windowRows < 1 || view.compressedRows < 0 || view.firstKey < 0 ||
		view.firstKey+view.windowRows-1 != view.pos || view.windowRows+view.compressedRows != flatRows || !shapeOK {
		return compute.Tensor{}, fmt.Errorf("%w: layer %d view window=[%d,+%d) compressed=%d does not describe %d flat rows at position %d",
			errV41ResidentKVDesync, layer, view.firstKey, view.windowRows, view.compressedRows, flatRows, view.pos)
	}
	slots := view.ringRows + 1
	if view.ringRows < 1 || view.windowRows > slots {
		return compute.Tensor{}, fmt.Errorf("%w: layer %d needs %d window rows from a %d-row ring", errV41ResidentKVCapacity, layer, view.windowRows, view.ringRows)
	}
	ringKey, logKey := v41ResidentKVKey(layer, "ring"), v41ResidentKVKey(layer, "log")
	m := r.layers[layer]
	if m == nil || m.owner != view.owner || m.nextPos != view.pos || len(m.ringKeys) != slots ||
		!r.current(ringKey, m.ring) || (m.logCap > 0 && !r.current(logKey, m.log)) {
		ring, err := r.alloc(slots, width)
		if err != nil {
			return compute.Tensor{}, err
		}
		r.adopt(ringKey, ring)
		m = &v41ResidentKVLayer{owner: view.owner, nextPos: view.pos, ring: ring, ringKeys: make([]int, slots)}
		for i := range m.ringKeys {
			m.ringKeys[i] = -1
		}
		if held, ok := r.s.halW[logKey]; ok {
			delete(r.s.halW, logKey)
			r.s.Backend.Free(held)
		}
		r.layers[layer] = m
	}
	// The mirror now matches the host up to pos. Mark it unusable until this
	// step's writes land, so a failure below can never leave a stale ring live.
	m.nextPos = -1
	for at := 0; at < view.windowRows; {
		key := view.firstKey + at
		slot := key % slots
		if m.ringKeys[slot] == key && key != view.pos {
			at++
			continue
		}
		run := 1
		for at+run < view.windowRows && slot+run < slots && (m.ringKeys[slot+run] != key+run || key+run == view.pos) {
			run++
		}
		if err := r.write(m.ring, slot, kv, rowKV, width, at, run); err != nil {
			return compute.Tensor{}, err
		}
		for i := 0; i < run; i++ {
			m.ringKeys[slot+i] = key + i
		}
		at += run
	}
	if view.compressedRows < m.logRows {
		return compute.Tensor{}, fmt.Errorf("%w: layer %d compressed rows fell from %d to %d without a position rollback", errV41ResidentKVDesync, layer, m.logRows, view.compressedRows)
	}
	if grow := view.compressedRows - m.logRows; grow > 0 {
		if view.compressedRows > m.logCap {
			capacity := max(2*m.logCap, view.compressedRows, 16)
			next, err := r.alloc(capacity, width)
			if err != nil {
				return compute.Tensor{}, err
			}
			if m.logRows > 0 {
				if err := r.copyRows(next, 0, m.log, 0, m.logRows); err != nil {
					r.s.Backend.Free(next)
					return compute.Tensor{}, err
				}
			}
			r.adopt(logKey, next)
			m.log, m.logCap = next, capacity
		}
		if err := r.write(m.log, m.logRows, kv, rowKV, width, view.windowRows+m.logRows, grow); err != nil {
			return compute.Tensor{}, err
		}
		m.logRows = view.compressedRows
	}
	spans := make([]compute.V41RowSpan, 0, 4)
	for _, row := range sourceRows {
		var span compute.V41RowSpan
		switch {
		case row < 0 || row >= flatRows:
			return compute.Tensor{}, fmt.Errorf("%w: layer %d selected flat row %d outside %d", errV41ResidentKVDesync, layer, row, flatRows)
		case row < view.windowRows:
			key := view.firstKey + row
			if m.ringKeys[key%slots] != key {
				return compute.Tensor{}, fmt.Errorf("%w: layer %d ring slot %d holds key %d, want %d", errV41ResidentKVDesync, layer, key%slots, m.ringKeys[key%slots], key)
			}
			span = compute.V41RowSpan{Src: m.ring, Row: key % slots, Rows: 1}
		default:
			span = compute.V41RowSpan{Src: m.log, Row: row - view.windowRows, Rows: 1}
		}
		if n := len(spans); n > 0 && spans[n-1].Src.Buf() == span.Src.Buf() && spans[n-1].Row+spans[n-1].Rows == span.Row {
			spans[n-1].Rows++
			continue
		}
		spans = append(spans, span)
	}
	out, err := r.ops.V41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, fmt.Errorf("%w: gather: %w", errV41ResidentKVBackend, err)
	}
	m.nextPos = view.pos + 1
	return out, nil
}

// sink returns the session's resident copy of a checkpoint attention sink.
func (r *v41ResidentKV) sink(key string, values []float32) compute.Tensor {
	residency := "v41-attn-sink:" + key
	return r.s.cachedImmutableWeight(residency, residency, func() compute.Tensor {
		return r.s.uploadHostF32([]int{len(values)}, values, compute.MemoryWeights, "V4.1 resident attention sink "+key)
	})
}
