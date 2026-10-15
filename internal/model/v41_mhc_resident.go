package model

import (
	"errors"
	"fmt"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41MHCResident is one forward's device-resident mHC state (Refs #13668): the
// [rows,4,H] stream panel and the carried coefficient panel [rows,24] (the
// previous FFN mix, one-hot pre for the first layer). Between ops the streams
// and every mix stay on the backend. The host receives only the collapsed rows
// its sublayers consume and returns only their outputs; the mHC projection
// reads the stream panel directly and its flatten-RMS scale runs inside the
// device split.
type v41MHCResident struct {
	d       *v41MHCDevice
	streams *v41DeviceStreams
	carry   compute.Tensor
	layer   v41MHCResidentLayer
	rows    int
	hidden  int
}

// resident reports whether this seam dispatches on a backend operation (as
// opposed to the host reference or a device-only refusal).
func (d *v41MHCDevice) resident() bool { return d != nil && d.op != nil && d.refusal == nil }

// guard runs one selected device step with the seam's fail-closed contract:
// every backend failure closes the session and returns V41MHCOperationError.
func (d *v41MHCDevice) guard(layer int, op string, body func(stage *string) error) (cause error) {
	if d.refusal != nil {
		return d.fail(layer, op, "admission", d.refusal)
	}
	d.s.ensureOpenBackendSession()
	stage := "payload"
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		if err, ok := r.(error); ok {
			var closed *BackendForwardOperationError
			if errors.As(err, &closed) {
				panic(r)
			}
			var backend *compute.BackendError
			if errors.As(err, &backend) {
				cause = d.fail(layer, op, stage, err)
				return
			}
		}
		if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
			if original, ok := r.(error); ok {
				err = original
			}
			cause = d.fail(layer, op, stage, err)
			return
		}
		err, ok := r.(error)
		if !ok {
			err = fmt.Errorf("unclassified backend panic: %v", r)
		}
		d.fail(layer, op, stage, err)
		panic(r)
	}()
	if err := body(&stage); err != nil {
		return d.fail(layer, op, stage, err)
	}
	return nil
}

func v41MHCFreshOutput(s *Session, out compute.Tensor, shape []int, inputs ...compute.Tensor) bool {
	if out.Buf() == nil || out.Backend() != s.Backend || out.Dtype != compute.F32 || out.Layout != compute.RowMajor || out.Quant != nil || !slices.Equal(out.Shape, shape) {
		return false
	}
	for _, in := range inputs {
		if out.Buf() == in.Buf() {
			return false
		}
	}
	return true
}

func v41MHCRowsFlat(rows [][]float32, H int) ([]float32, bool) {
	flat := make([]float32, 0, len(rows)*H)
	for _, row := range rows {
		if len(row) != H {
			return nil, false
		}
		for _, v := range row {
			if !finite32(v) {
				return nil, false
			}
		}
		flat = append(flat, row...)
	}
	return flat, true
}

// newResident uploads the embedding rows, publishes the four BF16 stream
// copies on the device and seeds the one-hot carry. It returns the BF16 stream
// 0 rows (one readback) so the caller's hidden rows match the host path.
func (d *v41MHCDevice) newResident(x [][]float32) (*v41MHCResident, error) {
	if len(x) == 0 || len(x[0]) == 0 {
		return nil, d.fail(-1, "initial-streams", "admission", errV41MHCResult)
	}
	H := len(x[0])
	flat, ok := v41MHCRowsFlat(x, H)
	if !ok {
		for _, row := range x {
			// The host refuses the same input before any selection.
			if _, bad := v41BF16PublishFault(row); bad {
				return nil, v41MHCBF16Refusal(-1, row)
			}
		}
		return nil, d.fail(-1, "initial-streams", "admission", errV41MHCResult)
	}
	r := &v41MHCResident{d: d, rows: len(x), hidden: H}
	s := d.s
	err := d.guard(-1, "initial-streams", func(stage *string) error {
		*stage = "x upload"
		xt := s.uploadHostF32([]int{r.rows, H}, flat, compute.MemoryActivation, "V4.1 mHC initial-streams x")
		defer s.Backend.Free(xt)
		*stage = "initial-streams"
		out, err := d.op.V41MHCInitialStreams(xt, r.rows, H)
		if err != nil {
			if out.Buf() != nil && out.Buf() != xt.Buf() {
				s.Backend.Free(out)
			}
			return err
		}
		if !v41MHCFreshOutput(s, out, []int{r.rows, compute.V41MHCStreams, H}, xt) {
			return errV41MHCResult
		}
		if r.streams, err = newV41DeviceStreams(s.Backend, out); err != nil {
			s.Backend.Free(out)
			return err
		}
		*stage = "carry upload"
		carry := make([]float32, r.rows*compute.V41MHCCoeffWidth)
		for row := 0; row < r.rows; row++ {
			carry[row*compute.V41MHCCoeffWidth] = 1
		}
		r.carry = s.uploadHostF32([]int{r.rows, compute.V41MHCCoeffWidth}, carry, compute.MemoryActivation, "V4.1 mHC carry")
		return nil
	})
	if err != nil {
		r.free()
		return nil, err
	}
	return r, nil
}

func (r *v41MHCResident) free() {
	if r == nil {
		return
	}
	r.abort()
	r.streams.Free()
	r.streams = nil
	r.dropTensor(&r.carry)
}

func (r *v41MHCResident) dropTensor(t *compute.Tensor) {
	if t != nil && t.Buf() != nil {
		r.d.s.Backend.Free(*t)
		*t = compute.Tensor{}
	}
}

// streamZero reads stream 0 back, the BF16 hidden row each token carries.
func (r *v41MHCResident) streamZero() ([][]float32, error) {
	var host [][][]float32
	err := r.d.guard(-1, "stream-readback", func(stage *string) error {
		*stage = "readback"
		var err error
		host, err = r.streams.ReadHost()
		return err
	})
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(host))
	for t := range host {
		out[t] = host[t][0]
	}
	return out, nil
}

// hostStreams reads the panel back for the Engram host injection of a session
// that binds no Engram gate; setHostStreams publishes its BF16 result.
// Device-only never reaches this round trip.
func (r *v41MHCResident) hostStreams(layer int) ([][][]float32, error) {
	var host [][][]float32
	err := r.d.guard(layer, "stream-readback", func(stage *string) error {
		*stage = "readback"
		var err error
		host, err = r.streams.ReadHost()
		return err
	})
	return host, err
}

func (r *v41MHCResident) setHostStreams(layer int, host [][][]float32) error {
	flat := make([]float32, 0, r.rows*compute.V41MHCStreams*r.hidden)
	if len(host) != r.rows {
		return r.d.fail(layer, "stream-upload", "admission", errV41MHCResult)
	}
	for _, streams := range host {
		row, H, err := v41MHCStreamsRow(streams)
		if err != nil || H != r.hidden {
			return r.d.fail(layer, "stream-upload", "admission", errV41MHCResult)
		}
		flat = append(flat, row...)
	}
	s := r.d.s
	return r.d.guard(layer, "stream-upload", func(stage *string) error {
		*stage = "upload"
		next := s.uploadHostF32([]int{r.rows, compute.V41MHCStreams, r.hidden}, flat, compute.MemoryActivation, "V4.1 mHC streams")
		if err := r.streams.Replace(next); err != nil {
			s.Backend.Free(next)
			return err
		}
		return nil
	})
}

// mixes projects the flattened streams through leaf on the device and splits
// the RMS-scaled result: the attention mix (mhc.*) or the FFN mix (mhc.ffn_*).
func (r *v41MHCResident) mixes(layer int, streams *v41DeviceStreams, leaf, scaleLeaf, baseLeaf string, iters int, hcEps, rmsEps float32) (compute.Tensor, error) {
	m := r.d.s.M
	scale, base := m.tensor(layerName(layer, scaleLeaf)), m.tensor(layerName(layer, baseLeaf))
	if len(scale) != 3 || len(base) != compute.V41MHCCoeffWidth || iters < 1 || !finite32(hcEps) || hcEps <= 0 || !finite32(rmsEps) || rmsEps <= 0 {
		return compute.Tensor{}, v41StageErr(v41StageMHC, layer, fmt.Errorf("model: invalid V41 mHC geometry iters=%d scale=%d base=%d", iters, len(scale), len(base)))
	}
	for i, v := range scale {
		if !finite32(v) {
			return compute.Tensor{}, v41StageErr(v41StageMHC, layer, fmt.Errorf("model: non-finite V41 mHC coefficient at %d", i))
		}
	}
	for i, v := range base {
		if !finite32(v) {
			return compute.Tensor{}, v41StageErr(v41StageMHC, layer, fmt.Errorf("model: non-finite V41 mHC coefficient at %d", len(scale)+i))
		}
	}
	_, transposed, ok := m.v41MHCWeightLayoutNamed(layer, leaf)
	if !ok {
		return compute.Tensor{}, v41StageErr(v41StageMHC, layer, fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	s := r.d.s
	rows, width := streams.Rows(), compute.V41MHCStreams*streams.Hidden()
	var coeffs compute.Tensor
	err := r.d.guard(layer, "projection", func(stage *string) error {
		*stage = "weight"
		weight, err := s.v41MHCMixWeight(layer, leaf, streams.Hidden(), transposed)
		if err != nil {
			return err
		}
		*stage = "projection"
		var raw compute.Tensor
		if rows == 1 {
			input := streams.Flat()
			input.Shape = []int{width}
			raw = s.Backend.MatMul(weight, input)
		} else {
			raw = s.Backend.BatchedMatMul(weight, streams.Flat(), rows)
		}
		if raw.Buf() == nil || raw.Buf() == streams.Tensor().Buf() || raw.Buf() == weight.Buf() || raw.Dtype != compute.F32 || raw.Numel() != rows*compute.V41MHCCoeffWidth {
			if raw.Buf() != nil && raw.Buf() != streams.Tensor().Buf() && raw.Buf() != weight.Buf() {
				s.Backend.Free(raw)
			}
			return errV41MHCResult
		}
		defer s.Backend.Free(raw)
		view := raw
		view.Shape = []int{rows, compute.V41MHCCoeffWidth}
		*stage = "params upload"
		// The split coefficients are immutable checkpoint data, just like the
		// mix matrix above. The model/backend owner separates loaded model
		// generations and backend instances; these leaf names separate layer
		// and attention/FFN role. Forward-local cleanup must not free them.
		paramsKey := "v41-mhc-split-params:" + layerName(layer, scaleLeaf) + ":" + layerName(layer, baseLeaf)
		pt := s.cachedImmutableWeight(paramsKey, paramsKey, func() compute.Tensor {
			params := append(append(make([]float32, 0, compute.V41MHCParamWidth), scale...), base...)
			return s.uploadHostF32([]int{compute.V41MHCParamWidth}, params, compute.MemoryWeights, "V4.1 mHC split params")
		})
		*stage = "rms-split"
		out, err := r.d.op.V41MHCRMSSplit(view, streams.Tensor(), pt, rows, streams.Hidden(), iters, hcEps, rmsEps)
		if err != nil {
			if out.Buf() != nil && out.Buf() != raw.Buf() && out.Buf() != pt.Buf() && out.Buf() != streams.Tensor().Buf() {
				s.Backend.Free(out)
			}
			return err
		}
		if !v41MHCFreshOutput(s, out, []int{rows, compute.V41MHCCoeffWidth}, raw, pt, streams.Tensor()) {
			return errV41MHCResult
		}
		coeffs = out
		return nil
	})
	return coeffs, err
}

// collapse returns the BF16 hc_pre rows sum_h pre[h]*stream[h] of streams under
// coeffs, owned on the host for the sublayer that consumes them.
func (r *v41MHCResident) collapse(layer int, streams *v41DeviceStreams, coeffs compute.Tensor) ([][]float32, error) {
	return r.collapseWithRetention(layer, streams, coeffs, true)
}

// collapseWithRetention keeps the selected dispatch, readback and validation
// boundary even when no host consumer needs an independently owned copy.
func (r *v41MHCResident) collapseWithRetention(layer int, streams *v41DeviceStreams, coeffs compute.Tensor, retain bool) ([][]float32, error) {
	s := r.d.s
	rows, H := streams.Rows(), streams.Hidden()
	var out [][]float32
	err := r.d.guard(layer, "pre-collapse", func(stage *string) error {
		*stage = "pre-collapse"
		t, err := r.d.op.V41MHCPre(streams.Tensor(), coeffs, rows, H)
		fresh := v41MHCFreshOutput(s, t, []int{rows, H}, streams.Tensor(), coeffs)
		if fresh {
			defer s.Backend.Free(t)
		}
		if err != nil {
			return err
		}
		if !fresh {
			return errV41MHCResult
		}
		*stage = "readback"
		values := s.Backend.Read(t)
		if len(values) != rows*H {
			return errV41MHCResult
		}
		if !retain {
			// Read may lend backend scratch. Inspect it before Free, then drop
			// it; no borrowed value escapes or substitutes for a later read.
			for _, v := range values {
				if !finite32(v) {
					return errV41MHCResult
				}
			}
			return nil
		}
		flat := make([]float32, len(values))
		for i, v := range values {
			if !finite32(v) {
				return errV41MHCResult
			}
			flat[i] = v
		}
		out = make([][]float32, rows)
		for row := range out {
			out[row] = flat[row*H : (row+1)*H : (row+1)*H]
		}
		return nil
	})
	return out, err
}

// post uploads the sublayer rows and returns the fresh BF16 post-mixed panel
// sum_src comb[src,dst]*stream[src] + post[dst]*BF16(x). The caller owns it.
func (r *v41MHCResident) post(layer int, x [][]float32, streams *v41DeviceStreams, coeffs compute.Tensor) (compute.Tensor, error) {
	s := r.d.s
	rows, H := streams.Rows(), streams.Hidden()
	flat, ok := v41MHCRowsFlat(x, H)
	if !ok || len(x) != rows {
		for _, row := range x {
			// A non-finite sublayer row is the host's BF16 publication refusal.
			if _, bad := v41BF16PublishFault(row); bad {
				return compute.Tensor{}, v41MHCBF16Refusal(layer, row)
			}
		}
		return compute.Tensor{}, v41StageErr(v41StageMHC, layer, fmt.Errorf("model: invalid V41 mHC post shape"))
	}
	var next compute.Tensor
	err := r.d.guard(layer, "post-mix", func(stage *string) error {
		*stage = "x upload"
		xt := s.uploadHostF32([]int{rows, H}, flat, compute.MemoryActivation, "V4.1 mHC post-mix x")
		defer s.Backend.Free(xt)
		*stage = "post-mix"
		var err error
		next, err = r.recordPost(xt, streams, coeffs)
		return err
	})
	return next, err
}

// recordPost preserves the existing operand order and fresh-output contract.
// Its caller owns the guard and every input; only the returned panel is new.
func (r *v41MHCResident) recordPost(x compute.Tensor, streams *v41DeviceStreams, coeffs compute.Tensor) (compute.Tensor, error) {
	s := r.d.s
	out, err := r.d.op.V41MHCPost(x, streams.Tensor(), coeffs, streams.Rows(), streams.Hidden())
	if err != nil {
		if out.Buf() != nil && out.Buf() != x.Buf() && out.Buf() != streams.Tensor().Buf() && out.Buf() != coeffs.Buf() {
			s.Backend.Free(out)
		}
		return compute.Tensor{}, err
	}
	if !v41MHCFreshOutput(s, out, []int{streams.Rows(), compute.V41MHCStreams, streams.Hidden()}, x, streams.Tensor(), coeffs) {
		return compute.Tensor{}, errV41MHCResult
	}
	return out, nil
}

// finishLayerChecked borrows the decode chain's still-owned [H] sum after its
// finite-only readback. The shape view adds no BF16 publication or device allocation.
func (r *v41MHCResident) finishLayerChecked(layer int, sum compute.Tensor) error {
	err := r.d.guard(layer, "post-mix", func(stage *string) error {
		if r.rows != 1 || sum.Buf() == nil || sum.Backend() != r.d.s.Backend || sum.Dtype != compute.F32 || sum.Layout != compute.RowMajor || sum.Quant != nil || len(sum.Shape) != 1 || sum.Shape[0] != r.hidden {
			return errV41MHCResult
		}
		sum.Shape = []int{1, r.hidden}
		*stage = "post-mix"
		var err error
		r.layer.next, err = r.recordPost(sum, r.layer.updated, r.layer.ffn)
		return err
	})
	if err != nil {
		r.abort()
	}
	return err
}

// v41MHCMixWeight stages (or borrows the cached) device mHC mix weight in its
// logical [24, 4H] orientation, sharing the projection seam's cache key.
func (s *Session) v41MHCMixWeight(l int, leaf string, H int, transposed bool) (weight compute.Tensor, err error) {
	width, ok := checkedMulInt(4, H)
	elems, ok2 := checkedMulInt(v41MHCMixWidth, width)
	if !ok || !ok2 || H <= 0 || !v41MHCMixLeaf(leaf) {
		return compute.Tensor{}, errV41MHCDeviceUnavailable
	}
	name := layerName(l, leaf)
	out, in, present := s.M.residentShape(name)
	if !present {
		return compute.Tensor{}, errV41MHCDeviceUnavailable
	}
	if transposed {
		if out != width || in != v41MHCMixWidth || !s.M.has(name) {
			return compute.Tensor{}, errV41MHCDeviceUnavailable
		}
	} else if out != v41MHCMixWidth || in != width {
		return compute.Tensor{}, errV41MHCDeviceUnavailable
	}
	w, supported, _ := s.v41GroupedWeight(name, width)
	if !supported || w.elems != elems {
		return compute.Tensor{}, errV41MHCDeviceUnavailable
	}
	var payload error
	defer func() {
		if payload == nil {
			return
		}
		if r := recover(); r != nil {
			if e, ok := r.(error); ok && e == payload {
				weight, err = compute.Tensor{}, payload
				return
			}
			panic(r)
		}
	}()
	layout := "logical"
	if transposed {
		layout = "transposed-f32"
	}
	key := fmt.Sprintf("v41-mhc:%s:%v:%d:%d:%s", name, w.dtype, v41MHCMixWidth, width, layout)
	weight = s.cachedImmutableWeight(key, key, func() compute.Tensor {
		rows, cols := v41MHCMixWidth, width
		if transposed {
			rows, cols = width, v41MHCMixWidth
			meta := s.M.manifest[name]
			nbytes, ok := checkedMulInt(elems, 4)
			if !ok || meta.Nbytes != nbytes || meta.Offset < 0 || meta.Offset > len(s.M.raw) || nbytes > len(s.M.raw)-meta.Offset {
				payload = errV41ProjectionResult
				panic(payload)
			}
		}
		source, err := w.host(0, rows, cols)
		if err != nil {
			payload = err
			panic(payload)
		}
		if transposed {
			host, ok := source.Buf().(compute.HostBuffer)
			if !ok || len(host.F32()) != elems {
				payload = errV41ProjectionResult
				panic(payload)
			}
			stored := host.F32()
			logical := make([]float32, elems)
			for o := 0; o < v41MHCMixWidth; o++ {
				for i := 0; i < width; i++ {
					logical[o*width+i] = stored[i*v41MHCMixWidth+o]
				}
			}
			source = compute.NewF32(compute.Default(), []int{v41MHCMixWidth, width}, logical)
		}
		return s.Backend.Upload(source, w.dtype)
	})
	return weight, nil
}

// layer is the per-layer state machine over the resident panel. beginLayer
// computes the attention mix and the attention collapse; ffnInput post-mixes
// the attention output and computes the FFN mix and collapse; finishLayer
// post-mixes the MoE output into a pending panel; publish commits it (streams
// and carry) and abort discards every pending tensor. A failed layer therefore
// never advances the carried streams.
type v41MHCResidentLayer struct {
	attn, ffn compute.Tensor
	updated   *v41DeviceStreams
	next      compute.Tensor
}

func (r *v41MHCResident) beginLayer(layer, iters int, hcEps, rmsEps float32) ([][]float32, error) {
	r.abort()
	attn, err := r.mixes(layer, r.streams, "mhc.mixes.weight", "mhc.scale", "mhc.base", iters, hcEps, rmsEps)
	if err != nil {
		return nil, err
	}
	r.layer.attn = attn
	rows, err := r.collapse(layer, r.streams, r.carry)
	if err != nil {
		r.abort()
	}
	return rows, err
}

func (r *v41MHCResident) ffnInput(layer int, attnOut [][]float32, iters int, hcEps, rmsEps float32) ([][]float32, error) {
	updated, err := r.post(layer, attnOut, r.streams, r.layer.attn)
	if err != nil {
		r.abort()
		return nil, err
	}
	if r.layer.updated, err = newV41DeviceStreams(r.d.s.Backend, updated); err != nil {
		r.d.s.Backend.Free(updated)
		r.abort()
		return nil, r.d.fail(layer, "post-mix", "result", err)
	}
	if r.layer.ffn, err = r.mixes(layer, r.layer.updated, "mhc.ffn_mixes.weight", "mhc.ffn_scale", "mhc.ffn_base", iters, hcEps, rmsEps); err != nil {
		r.abort()
		return nil, err
	}
	rows, err := r.collapse(layer, r.layer.updated, r.layer.attn)
	if err != nil {
		r.abort()
	}
	return rows, err
}

func (r *v41MHCResident) finishLayer(layer int, moe [][]float32) error {
	next, err := r.post(layer, moe, r.layer.updated, r.layer.ffn)
	if err != nil {
		r.abort()
		return err
	}
	r.layer.next = next
	return nil
}

func (r *v41MHCResident) publish(layer int) error {
	if r.layer.next.Buf() == nil || r.layer.ffn.Buf() == nil {
		r.abort()
		return r.d.fail(layer, "publish", "admission", errV41MHCResult)
	}
	if err := r.streams.Replace(r.layer.next); err != nil {
		r.abort()
		return r.d.fail(layer, "publish", "result", err)
	}
	r.layer.next = compute.Tensor{}
	r.dropTensor(&r.carry)
	r.carry, r.layer.ffn = r.layer.ffn, compute.Tensor{}
	r.abort()
	return nil
}

func (r *v41MHCResident) abort() {
	if r == nil {
		return
	}
	r.dropTensor(&r.layer.attn)
	r.dropTensor(&r.layer.ffn)
	r.dropTensor(&r.layer.next)
	r.layer.updated.Free()
	r.layer.updated = nil
}

// hiddenRows is the per-layer hidden row a host consumer reads: hc_pre of the
// published streams under the carried coefficients.
func (r *v41MHCResident) hiddenRows(layer int) ([][]float32, error) {
	return r.collapse(layer, r.streams, r.carry)
}

// checkHiddenRows preserves the per-layer hidden collapse's failure boundary
// without materializing a host panel that serving will immediately discard.
func (r *v41MHCResident) checkHiddenRows(layer int) error {
	_, err := r.collapseWithRetention(layer, r.streams, r.carry, false)
	return err
}
