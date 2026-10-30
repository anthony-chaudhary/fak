package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41EngramGateFunc applies one Engram layer to the resident mHC panel (Refs
// #13668, #13811): it decodes the gathered rows into the projection input
// [positions, cols*dim] (on the device when the backend binds the row decode),
// projects it through engram_kv.weight, runs the gate/add over the panel and
// replaces it with the published successor. handled=false is a decline before
// any device work (an unsupported projection weight or geometry); after
// selection a failure closes the session and is never replayed on the host.
type v41EngramGateFunc func(layer int, streams *v41DeviceStreams, rows *v41EngramGatheredRows, positions int, geometry v41EngramGeometry, eps float32) (handled bool, err error)

var errV41EngramGateResult = errors.New("model: invalid selected V4.1 device Engram gate result")

// v41EngramGateFunc binds the resident Engram gate/add when the backend
// exposes the dedicated operation. The projection shares the Engram projection
// seam's immutable weight cache key, so both arms stage one device weight. With
// the device row decode bound, the stored row bytes are the only Engram data
// the host touches; a decode fault poisons the projection so the gate's status
// refuses the layer, and only then is the decode status read back.
func (s *Session) v41EngramGateFunc() v41EngramGateFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	return s.v41EngramGateFuncWith(s.v41EngramRowDecoder())
}

// v41EngramGateFuncWith binds the gate with an explicit row decoder; nil keeps
// the host decode arm, which device-only refuses.
func (s *Session) v41EngramGateFuncWith(decoder compute.V41EngramRowDecodeBackend) v41EngramGateFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	op, ok := s.Backend.(compute.V41EngramGateBackend)
	if !ok || !op.SupportsV41EngramGate() {
		return nil
	}
	return func(l int, streams *v41DeviceStreams, gathered *v41EngramGatheredRows, positions int, g v41EngramGeometry, eps float32) (bool, error) {
		if decoder == nil && s.v41DeviceOnlyPolicy() {
			return true, s.v41DeviceOnlyDeclined(l, "engram-row-decode")
		}
		in, okIn := checkedMulInt(g.cols, g.dim)
		out, okOut := checkedMulInt(g.hc+1, g.H)
		count, okCount := checkedMulInt(positions, g.cols)
		if !okIn || !okOut || !okCount || g.hc != compute.V41EngramGateStreams || positions <= 0 || streams == nil ||
			streams.Rows() != positions || streams.Hidden() != g.H || gathered == nil || len(gathered.rows) != count || gathered.dim != g.dim ||
			len(g.qNorm) != g.hc*g.H || len(g.kNorm) != g.hc*g.H || s.M.v41EngramProjectionShape(l, out, in) != nil {
			return false, nil
		}
		name := layerName(l, "engram_kv.weight")
		descriptor, supported, _ := s.v41GroupedWeight(name, in)
		if !supported {
			return false, nil
		}
		encoding, decodable := gathered.encoding()
		decodable = decodable && compute.V41EngramRowDecodeSupportsEncoding(decoder, encoding)
		var stored []byte
		if decodable && decoder != nil {
			stored, decodable = gathered.stored()
		}
		var rows []float32
		if decoder == nil || !decodable {
			if s.v41DeviceOnlyPolicy() {
				return true, s.v41DeviceOnlyDeclined(l, "engram-row-decode")
			}
			var err error
			if rows, err = gathered.decodeHost(); err != nil {
				return true, err
			}
			for _, v := range rows {
				if !finite32(v) {
					return false, nil
				}
			}
		}
		opened := s.M.v41NowNanos()
		completed, calls := 0, 0
		var upload, readback int64
		defer func() { s.M.v41NoteEngramProjection(1, 0, completed, 0, calls, upload, readback, 0, opened) }()
		_, err := v41SelectedDeviceCall(s, l, "v41-engram-gate", func(stage *string) (struct{}, error) {
			key := fmt.Sprintf("v41-engram:%s:%v:%d:%d", name, descriptor.dtype, out, in)
			var source compute.Tensor
			if _, cached := s.lookupImmutableWeight(key, key); !cached {
				var err error
				if source, err = descriptor.host(0, out, in); err != nil {
					return struct{}{}, err
				}
			}
			*stage = "weight upload"
			weight := s.cachedImmutableWeight(key, key, func() compute.Tensor { return s.Backend.Upload(source, descriptor.dtype) })
			*stage = "norms upload"
			normsKey := "v41-engram-gate-norms:" + layerName(l, "engram_q_norm.weight") + ":" + layerName(l, "engram_k_norm.weight")
			norms := s.cachedImmutableWeight(normsKey, normsKey, func() compute.Tensor {
				values := append(append(make([]float32, 0, 2*len(g.qNorm)), g.qNorm...), g.kNorm...)
				return s.uploadHostF32([]int{len(values)}, values, compute.MemoryWeights, "V4.1 Engram gate norms")
			})
			shape := []int{positions, in}
			if positions == 1 {
				shape = []int{in}
			}
			var x, decodeStatus compute.Tensor
			if stored != nil {
				*stage = "row decode"
				panel, status, err := decoder.V41EngramRowDecode(stored, encoding, count, g.dim)
				upload += int64(len(stored))
				inputs := []compute.Tensor{streams.Tensor(), weight, norms}
				for _, t := range []compute.Tensor{panel, status} {
					if t.Buf() != nil && !v41EngramGateAliases(t, inputs) {
						defer s.Backend.Free(t)
					}
				}
				if err != nil {
					return struct{}{}, err
				}
				if !v41MHCFreshOutput(s, panel, []int{count, g.dim}, inputs...) || !v41MHCFreshOutput(s, status, []int{count}, inputs...) || panel.Buf() == status.Buf() {
					return struct{}{}, errV41EngramGateResult
				}
				x, decodeStatus = panel, status
				x.Shape = shape
			} else {
				*stage = "activation upload"
				x = s.uploadHostF32(shape, rows, compute.MemoryActivation, "V4.1 Engram gate rows")
				defer s.Backend.Free(x)
				upload += int64(len(rows)) * 4
			}
			*stage = "projection"
			calls++
			var projected compute.Tensor
			if positions == 1 {
				projected = s.Backend.MatMul(weight, x)
			} else {
				projected = s.Backend.BatchedMatMul(weight, x, positions)
			}
			if projected.Buf() == nil || projected.Buf() == x.Buf() || projected.Buf() == weight.Buf() || projected.Dtype != compute.F32 || projected.Numel() != positions*out {
				if projected.Buf() != nil && projected.Buf() != x.Buf() && projected.Buf() != weight.Buf() {
					s.Backend.Free(projected)
				}
				return struct{}{}, errV41EngramGateResult
			}
			defer s.Backend.Free(projected)
			if stored != nil {
				// A faulting row turns row 0's first value element into NaN, which
				// the gate's status refuses before any stream publishes.
				*stage = "row decode poison"
				if err := decoder.V41EngramRowDecodePoison(projected, decodeStatus, count, g.hc*g.H); err != nil {
					return struct{}{}, err
				}
			}
			*stage = "gate"
			next, status, err := op.V41EngramGate(streams.Tensor(), projected, norms, positions, g.H, eps)
			inputs := []compute.Tensor{streams.Tensor(), projected, norms}
			published := false
			defer func() {
				// Release every fresh output the panel did not take; never an input.
				if next.Buf() != nil && !published && !v41EngramGateAliases(next, inputs) {
					s.Backend.Free(next)
				}
				if status.Buf() != nil && status.Buf() != next.Buf() && !v41EngramGateAliases(status, inputs) {
					s.Backend.Free(status)
				}
			}()
			if err != nil {
				return struct{}{}, err
			}
			if !v41MHCFreshOutput(s, next, []int{positions, compute.V41MHCStreams, g.H}, inputs...) ||
				!v41MHCFreshOutput(s, status, []int{positions, compute.V41MHCStreams}, inputs...) || next.Buf() == status.Buf() {
				return struct{}{}, errV41EngramGateResult
			}
			*stage = "status readback"
			flags := s.Backend.Read(status)
			readback += int64(len(flags)) * 4
			if len(flags) != positions*compute.V41MHCStreams {
				return struct{}{}, errV41EngramGateResult
			}
			for _, flag := range flags {
				if flag != 0 {
					if stored != nil {
						*stage = "row decode status readback"
						words := append([]float32(nil), s.Backend.Read(decodeStatus)...)
						readback += int64(len(words)) * 4
						if len(words) != count {
							return struct{}{}, errV41EngramGateResult
						}
						if fault := gathered.deviceFault(words); fault != nil {
							return struct{}{}, fault
						}
					}
					// The host projection refuses a non-finite BF16 key or value
					// before any stream publishes; the panel is left unchanged.
					return struct{}{}, errV41ProjectionResult
				}
			}
			*stage = "publish"
			if err := streams.Replace(next); err != nil {
				return struct{}{}, err
			}
			published = true
			return struct{}{}, nil
		})
		if err != nil {
			return true, v41EngramProjectionErr(l, err)
		}
		completed = positions
		return true, nil
	}
}

func v41EngramGateAliases(t compute.Tensor, inputs []compute.Tensor) bool {
	for _, in := range inputs {
		if t.Buf() == in.Buf() {
			return true
		}
	}
	return false
}

// v41EngramInjectResident applies one Engram layer to the resident panel. The
// host gathers the stored rows (retrieval, like a weight read); the decode, when
// the backend binds it, the projection, gate and stream publication run on the
// device seam. handled is false only when the seam declined before any device
// work.
func (m *Model) v41EngramInjectResident(l int, streams *v41DeviceStreams, positions int, layerRows []uint32, eps float32, geometry v41EngramGeometry, gate v41EngramGateFunc) (gatheredRows int, handled bool, err error) {
	gathered, err := m.v41EngramGatherRows(l, positions, layerRows, eps, geometry)
	if err != nil {
		return 0, true, err
	}
	handled, err = gate(l, streams, gathered, positions, geometry, eps)
	return len(gathered.rows), handled, err
}
