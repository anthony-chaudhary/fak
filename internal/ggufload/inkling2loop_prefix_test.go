package ggufload

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type prefixTestDevice struct {
	compute.Backend
	stepCount int
}

func newPrefixTestDevice(t *testing.T) *prefixTestDevice {
	t.Helper()
	be := compute.Default()
	if be == nil {
		t.Fatal("compute.Default() returned nil")
	}
	return &prefixTestDevice{Backend: be}
}

func (d *prefixTestDevice) slice(ref compute.Ink2Ref) []float32 {
	if ref.T.Buf() == nil {
		return nil
	}
	s, ok := d.Backend.Host(ref.T)
	if !ok {
		return nil
	}
	return s[ref.Off:]
}

func (d *prefixTestDevice) Ink2Alloc(n int) (compute.Tensor, error) {
	return compute.NewF32(d.Backend, []int{n}, make([]float32, n)), nil
}

func (d *prefixTestDevice) Ink2Write(dst compute.Ink2Ref, src []float32) error {
	copy(d.slice(dst), src)
	return nil
}

func (d *prefixTestDevice) Ink2Read(src compute.Ink2Ref, n int) ([]float32, error) {
	s := d.slice(src)
	out := make([]float32, n)
	copy(out, s[:n])
	return out, nil
}

func (d *prefixTestDevice) Ink2ReadInts(src compute.Ink2Ref, n int) ([]int32, error) {
	s := d.slice(src)
	out := make([]int32, n)
	for i := 0; i < n; i++ {
		out[i] = int32(math.Float32bits(s[i]))
	}
	return out, nil
}

func (d *prefixTestDevice) Ink2MatVec(w compute.Tensor, x, y compute.Ink2Ref) error {
	mat, ok := d.Backend.Host(w)
	if !ok {
		return errors.New("non-host weight")
	}
	rows, cols := w.Shape[0], w.Shape[1]
	xv, yv := d.slice(x), d.slice(y)
	for r := 0; r < rows; r++ {
		var sum float32
		row := mat[r*cols : (r+1)*cols]
		for c := 0; c < cols; c++ {
			sum += row[c] * xv[c]
		}
		yv[r] = sum
	}
	return nil
}

func (d *prefixTestDevice) Ink2MatVecQ8(compute.Tensor, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, bool, compute.Ink2Ref) (bool, error) {
	return false, nil
}

func (d *prefixTestDevice) Ink2ExpertMatVec(compute.Tensor, int, compute.Ink2Ref, compute.Ink2Ref, int, compute.Ink2Ref, int, compute.Ink2Ref, *compute.Ink2Situ) (bool, error) {
	return false, nil
}

func prefixRMS(x, w []float32, n int, eps float32) []float32 {
	var ss float64
	for i := 0; i < n; i++ {
		v := float64(x[i])
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = x[i] * inv
		if w != nil {
			out[i] *= w[i]
		}
	}
	return out
}

func (d *prefixTestDevice) Ink2RMS(x, w, y compute.Ink2Ref, rows, n int, eps float32) error {
	xv, yv := d.slice(x), d.slice(y)
	wv := d.slice(w)
	for r := 0; r < rows; r++ {
		copy(yv[r*n:(r+1)*n], prefixRMS(xv[r*n:(r+1)*n], wv, n, eps))
	}
	return nil
}

func (d *prefixTestDevice) Ink2CtrlPair(cur, msg, pair compute.Ink2Ref, n int) error {
	pv := d.slice(pair)
	copy(pv[:n], prefixRMS(d.slice(cur), nil, n, 1e-6))
	copy(pv[n:2*n], prefixRMS(d.slice(msg), nil, n, 1e-6))
	return nil
}

func (d *prefixTestDevice) Ink2Mix(states []compute.Ink2Ref, queries compute.Ink2Ref, nq int, norms []compute.Ink2Ref, eps float32, out compute.Ink2Ref, n int) error {
	normStates := make([][]float32, len(states))
	for i, st := range states {
		normStates[i] = prefixRMS(d.slice(st), nil, n, 1e-6)
	}
	qv := d.slice(queries)
	ov := d.slice(out)
	inv := 1.0 / math.Sqrt(float64(n))
	for qi := 0; qi < nq; qi++ {
		q := qv[qi*n : (qi+1)*n]
		scores := make([]float64, len(normStates))
		maxS := math.Inf(-1)
		for si, st := range normStates {
			var dot float64
			for j := 0; j < n; j++ {
				dot += float64(st[j]) * float64(q[j])
			}
			scores[si] = dot * inv
			maxS = math.Max(maxS, scores[si])
		}
		var denom float64
		for si := range scores {
			scores[si] = math.Exp(scores[si] - maxS)
			denom += scores[si]
		}
		mixed := make([]float32, n)
		for si, st := range normStates {
			wt := float32(scores[si] / denom)
			for j := 0; j < n; j++ {
				mixed[j] += wt * st[j]
			}
		}
		if qi < len(norms) && norms[qi].T.Buf() != nil {
			mixed = prefixRMS(mixed, d.slice(norms[qi]), n, eps)
		}
		copy(ov[qi*n:(qi+1)*n], mixed)
	}
	return nil
}

func (d *prefixTestDevice) Ink2Control(ctl compute.Ink2Controls, g, out compute.Ink2Ref, n, rank int, norms []compute.Ink2Ref, eps float32) error {
	gv, ov := d.slice(g), d.slice(out)
	for j := range ctl.Cur {
		cur := d.slice(ctl.Cur[j])
		msg := d.slice(ctl.Msg[j])
		pair := append(prefixRMS(cur, nil, n, 1e-6), prefixRMS(msg, nil, n, 1e-6)...)
		dw, _, _ := d.matInfo(ctl.Down[j].T)
		gw, _, _ := d.matInfo(ctl.Gain[j].T)
		fw, _, _ := d.matInfo(ctl.Forget[j].T)
		gRow := gv[j*rank : (j+1)*rank]
		for r := 0; r < rank; r++ {
			var sum float32
			for c := 0; c < 2*n; c++ {
				sum += dw[r*2*n+c] * pair[c]
			}
			gRow[r] = sum
		}
		res := make([]float32, n)
		for r := 0; r < n; r++ {
			var gSum, fSum float64
			for c := 0; c < rank; c++ {
				gSum += float64(gw[r*rank+c] * gRow[c])
				fSum += float64(fw[r*rank+c] * gRow[c])
			}
			fg := float32(math.Tanh(fSum))
			gg := float32(math.Tanh(gSum))
			res[r] = cur[r] - fg*cur[r] + gg*msg[r]
		}
		if j < len(norms) && norms[j].T.Buf() != nil {
			res = prefixRMS(res, d.slice(norms[j]), n, eps)
		}
		copy(ov[j*n:(j+1)*n], res)
	}
	return nil
}

func (d *prefixTestDevice) matInfo(t compute.Tensor) ([]float32, int, int) {
	s, _ := d.Backend.Host(t)
	return s, t.Shape[0], t.Shape[1]
}

func prefixConvolveSlice(cur, weight, dyn, hist, out, resid []float32, n, heads, kn int, norm float32) {
	past := kn - 1
	perHead := 0
	if dyn != nil && heads > 0 {
		perHead = n / heads
	}
	for c := 0; c < n; c++ {
		hc := hist[c*past : (c+1)*past]
		w := weight[c*kn : (c+1)*kn]
		var acc float32
		for t := 0; t < past; t++ {
			acc += hc[t] * w[t]
		}
		acc += cur[c] * w[past]
		if dyn != nil && perHead > 0 {
			hd := c / perHead
			for lag := 0; lag < kn; lag++ {
				xv := cur[c]
				if lag > 0 {
					xv = hc[past-lag]
				}
				acc += dyn[hd*kn+lag] * xv
			}
		}
		y := (cur[c] + acc) * norm
		if resid != nil {
			y += resid[c]
		}
		out[c] = y
		copy(hc, hc[1:])
		hc[past-1] = cur[c]
	}
}

func (d *prefixTestDevice) Ink2Conv(x, w, dyn, hist, out, resid compute.Ink2Ref, n, heads, kn int, norm float32) error {
	prefixConvolveSlice(d.slice(x), d.slice(w), d.slice(dyn), d.slice(hist), d.slice(out), d.slice(resid), n, heads, kn, norm)
	return nil
}

func (d *prefixTestDevice) Ink2CombineConv(down, wts compute.Ink2Ref, k int, w, hist, out, resid compute.Ink2Ref, n, kn int, norm float32) error {
	comb := make([]float32, n)
	dv, wv := d.slice(down), d.slice(wts)
	for i := 0; i < k; i++ {
		wt := wv[i]
		for j := 0; j < n; j++ {
			comb[j] += float32(wt * dv[i*n+j])
		}
	}
	prefixConvolveSlice(comb, d.slice(w), nil, d.slice(hist), d.slice(out), d.slice(resid), n, 0, kn, norm)
	return nil
}

func prefixClipVec(v, clip []float32, capVal float32) {
	var ss float64
	for _, x := range v {
		ss += float64(x) * float64(x)
	}
	norm := float32(math.Sqrt(ss + 1e-12))
	if clip != nil {
		for i := range v {
			v[i] *= clip[i]
		}
	}
	if capVal > 0 && norm > capVal {
		s := capVal / norm
		for i := range v {
			v[i] *= s
		}
	}
}

func (d *prefixTestDevice) Ink2QKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv) error {
	kvDim := p.KVHeads * p.HeadDim
	qv := d.slice(q)
	rv := d.slice(r)
	kv := make([]float32, kvDim)
	vv := make([]float32, kvDim)
	if cv != nil {
		prefixConvolveSlice(d.slice(k), d.slice(cv.KW), d.slice(cv.KDyn), d.slice(cv.KHist), kv, nil, kvDim, p.KVHeads, cv.Kn, cv.KNorm)
		prefixConvolveSlice(d.slice(v), d.slice(cv.VW), d.slice(cv.VDyn), d.slice(cv.VHist), vv, nil, kvDim, p.KVHeads, cv.Kn, cv.VNorm)
	} else {
		copy(kv, d.slice(k)[:kvDim])
		copy(vv, d.slice(v)[:kvDim])
	}
	qcv, rcv := d.slice(qclip), d.slice(rclip)
	for hd := 0; hd < p.Heads; hd++ {
		prefixClipVec(qv[hd*p.HeadDim:(hd+1)*p.HeadDim], qcv[hd*p.HeadDim:(hd+1)*p.HeadDim], p.QKClip)
		prefixClipVec(rv[hd*p.RelLen:(hd+1)*p.RelLen], rcv[hd*p.RelLen:(hd+1)*p.RelLen], p.RelClip)
	}
	for hd := 0; hd < p.KVHeads; hd++ {
		prefixClipVec(kv[hd*p.HeadDim:(hd+1)*p.HeadDim], nil, p.QKClip)
		copy(vv[hd*p.HeadDim:(hd+1)*p.HeadDim], prefixRMS(vv[hd*p.HeadDim:(hd+1)*p.HeadDim], nil, p.HeadDim, p.Eps))
	}
	pos := int(int32(math.Float32bits(d.slice(meta)[0])))
	if p.Global {
		tau := float32(1 + float64(p.Alpha)*math.Log(math.Max(1, float64(pos+1)/float64(p.Floor))))
		for i := range qv[:p.Heads*p.HeadDim] {
			qv[i] *= tau
		}
		for i := range rv[:p.Heads*p.RelLen] {
			rv[i] *= tau
		}
	}
	slot := pos
	if !p.Global {
		slot = pos % cap
	}
	copy(d.slice(kcache)[slot*kvDim:(slot+1)*kvDim], kv)
	copy(d.slice(vcache)[slot*kvDim:(slot+1)*kvDim], vv)
	return nil
}

func (d *prefixTestDevice) Ink2Attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int) error {
	pos := int(int32(math.Float32bits(d.slice(meta)[0])))
	qv, rv, sv, rb := d.slice(q), d.slice(r), d.slice(sinks), d.slice(relBias)
	kc, vc, ov := d.slice(kcache), d.slice(vcache), d.slice(out)
	group := p.Heads / p.KVHeads
	kvDim := p.KVHeads * p.HeadDim
	start := 0
	if !p.Global && window > 0 && pos-window+1 > 0 {
		start = pos - window + 1
	}
	for hd := 0; hd < p.Heads; hd++ {
		kvh := hd / group
		qh := qv[hd*p.HeadDim : (hd+1)*p.HeadDim]
		rh := rv[hd*p.RelLen : (hd+1)*p.RelLen]
		sc := make([]float64, pos-start+1)
		maxS := math.Inf(-1)
		for j := start; j <= pos; j++ {
			idx := j
			if !p.Global {
				idx = j % cap
			}
			kj := kc[idx*kvDim+kvh*p.HeadDim : idx*kvDim+(kvh+1)*p.HeadDim]
			var qk float64
			for i := 0; i < p.HeadDim; i++ {
				qk += float64(qh[i] * kj[i])
			}
			bias := -float64(sv[hd])
			dist := pos - j
			if dist < extent {
				rbRow := rb[dist*p.RelLen : (dist+1)*p.RelLen]
				for i := 0; i < p.RelLen; i++ {
					bias += float64(rbRow[i] * rh[i])
				}
			}
			s := qk*float64(p.Scale) + bias
			sc[j-start] = s
			maxS = math.Max(maxS, s)
		}
		if maxS < 0 {
			maxS = 0
		}
		denom := math.Exp(-maxS)
		for i := range sc {
			sc[i] = math.Exp(sc[i] - maxS)
			denom += sc[i]
		}
		oh := ov[hd*p.HeadDim : (hd+1)*p.HeadDim]
		clear(oh)
		for j := start; j <= pos; j++ {
			idx := j
			if !p.Global {
				idx = j % cap
			}
			vj := vc[idx*kvDim+kvh*p.HeadDim : idx*kvDim+(kvh+1)*p.HeadDim]
			wt := float32(sc[j-start] / denom)
			for i := 0; i < p.HeadDim; i++ {
				oh[i] += wt * vj[i]
			}
		}
	}
	return nil
}

func (d *prefixTestDevice) Ink2SigmoidMul(x, gate compute.Ink2Ref, n int) error {
	xv, gv := d.slice(x), d.slice(gate)
	for i := 0; i < n; i++ {
		xv[i] *= float32(1.0 / (1.0 + math.Exp(-float64(gv[i]))))
	}
	return nil
}

func (d *prefixTestDevice) Ink2Route(logits, bias, ids, wts compute.Ink2Ref, nexp, k int, scale float32) error {
	lv, bv := d.slice(logits), d.slice(bias)
	probs := make([]float32, nexp)
	sel := make([]float32, nexp)
	for i := 0; i < nexp; i++ {
		probs[i] = float32(1.0 / (1.0 + math.Exp(-float64(lv[i]))))
		sel[i] = probs[i]
		if bv != nil {
			sel[i] += bv[i]
		}
	}
	used := make([]bool, nexp)
	idOut, wtOut := d.slice(ids), d.slice(wts)
	var sum float32
	for j := 0; j < k; j++ {
		best := -1
		for i := 0; i < nexp; i++ {
			if !used[i] && (best < 0 || sel[i] > sel[best]) {
				best = i
			}
		}
		used[best] = true
		idOut[j] = math.Float32frombits(uint32(int32(best)))
		wtOut[j] = probs[best]
		sum += probs[best]
	}
	for j := 0; j < k; j++ {
		wtOut[j] /= (sum + 1e-20)
		if scale != 0 {
			wtOut[j] *= scale
		}
	}
	return nil
}

func (d *prefixTestDevice) Ink2ExpertScale(y, ids, scale compute.Ink2Ref, k, rows int) error {
	yv, idv, scv := d.slice(y), d.slice(ids), d.slice(scale)
	for t := 0; t < k; t++ {
		id := int(int32(math.Float32bits(idv[t])))
		s := scv[id]
		for r := 0; r < rows; r++ {
			yv[t*rows+r] *= s
		}
	}
	return nil
}

func (d *prefixTestDevice) Ink2Situ(gu, act compute.Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	gv, av := d.slice(gu), d.slice(act)
	for t := 0; t < k; t++ {
		gRow := gv[t*2*width : (t+1)*2*width]
		aRow := av[t*actStride : t*actStride+width]
		for j := 0; j < width; j++ {
			g, u := float64(gRow[j]), float64(gRow[width+j])
			a := beta * math.Tanh(g/beta) * (1.0 / (1.0 + math.Exp(-g)))
			if linearBeta > 0 {
				u = linearBeta * math.Tanh(u/linearBeta)
			}
			aRow[j] = float32(a * u)
		}
	}
	return nil
}

func (d *prefixTestDevice) Ink2Combine(down, wts, out compute.Ink2Ref, k, n int) error {
	dv, wv, ov := d.slice(down), d.slice(wts), d.slice(out)
	clear(ov[:n])
	for i := 0; i < k; i++ {
		wt := wv[i]
		for j := 0; j < n; j++ {
			ov[j] += float32(wt * dv[i*n+j])
		}
	}
	return nil
}

func (d *prefixTestDevice) Ink2Add(a, b, out compute.Ink2Ref, n int) error {
	av, bv, ov := d.slice(a), d.slice(b), d.slice(out)
	for i := 0; i < n; i++ {
		ov[i] = av[i] + bv[i]
	}
	return nil
}

func (d *prefixTestDevice) Ink2SoftcapArgmax(logits compute.Ink2Ref, n int, softcap float32, outIdx compute.Ink2Ref) error {
	d.stepCount++
	lv := d.slice(logits)[:n]
	if softcap != 0 {
		for i, v := range lv {
			lv[i] = softcap * float32(math.Tanh(float64(v/softcap)))
		}
	}
	best := inkling2Argmax(lv)
	d.slice(outIdx)[0] = math.Float32frombits(uint32(int32(best)))
	return nil
}

func (d *prefixTestDevice) Ink2GraphBegin() error  { return errors.New("eager only") }
func (d *prefixTestDevice) Ink2GraphEnd() error    { return nil }
func (d *prefixTestDevice) Ink2GraphLaunch() error { return nil }
func (d *prefixTestDevice) Ink2GraphReset()        {}

// half and fp8 device extensions support compressed KV dtype parity and mismatch tests.
func prefixF32ToF16(f float32) uint16 {
	u := math.Float32bits(f)
	sign := uint16((u >> 16) & 0x8000)
	exp := int((u>>23)&0xff) - 127
	mant := u & 0x7fffff
	if exp <= -15 {
		return sign
	}
	if exp >= 16 {
		return sign | 0x7c00
	}
	return sign | uint16(exp+15)<<10 | uint16(mant>>13)
}

func prefixF16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1f)
	mant := uint32(h & 0x03ff)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03ff
	}
	return math.Float32frombits(sign | uint32(exp-15+127)<<23 | (mant << 13))
}

func (d *prefixTestDevice) Ink2QKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv) error {
	kvDim := p.KVHeads * p.HeadDim
	tmpK, _ := d.Ink2Alloc(cap * kvDim)
	tmpV, _ := d.Ink2Alloc(cap * kvDim)
	if err := d.Ink2QKVPost(q, r, k, v, qclip, rclip, compute.Ink2At(tmpK, 0), compute.Ink2At(tmpV, 0), meta, p, cap, cv); err != nil {
		return err
	}
	pos := int(int32(math.Float32bits(d.slice(meta)[0])))
	slot := pos
	if !p.Global {
		slot = pos % cap
	}
	kf, _ := d.Backend.Host(tmpK)
	vf, _ := d.Backend.Host(tmpV)
	kh, vh := d.slice(kcache), d.slice(vcache)
	posW := kvDim / 2
	for i := 0; i < posW; i++ {
		loK := uint32(prefixF32ToF16(kf[slot*kvDim+2*i]))
		hiK := uint32(prefixF32ToF16(kf[slot*kvDim+2*i+1]))
		kh[slot*posW+i] = math.Float32frombits(loK | (hiK << 16))
		loV := uint32(prefixF32ToF16(vf[slot*kvDim+2*i]))
		hiV := uint32(prefixF32ToF16(vf[slot*kvDim+2*i+1]))
		vh[slot*posW+i] = math.Float32frombits(loV | (hiV << 16))
	}
	return nil
}

func (d *prefixTestDevice) Ink2AttendHalf(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int) error {
	pos := int(int32(math.Float32bits(d.slice(meta)[0])))
	kvDim := p.KVHeads * p.HeadDim
	posW := kvDim / 2
	tmpK, _ := d.Ink2Alloc(cap * kvDim)
	tmpV, _ := d.Ink2Alloc(cap * kvDim)
	kf, _ := d.Backend.Host(tmpK)
	vf, _ := d.Backend.Host(tmpV)
	kh, vh := d.slice(kcache), d.slice(vcache)
	start := 0
	if !p.Global && window > 0 && pos-window+1 > 0 {
		start = pos - window + 1
	}
	for j := start; j <= pos; j++ {
		idx := j
		if !p.Global {
			idx = j % cap
		}
		for i := 0; i < posW; i++ {
			wk := math.Float32bits(kh[idx*posW+i])
			kf[idx*kvDim+2*i] = prefixF16ToF32(uint16(wk & 0xffff))
			kf[idx*kvDim+2*i+1] = prefixF16ToF32(uint16(wk >> 16))
			wv := math.Float32bits(vh[idx*posW+i])
			vf[idx*kvDim+2*i] = prefixF16ToF32(uint16(wv & 0xffff))
			vf[idx*kvDim+2*i+1] = prefixF16ToF32(uint16(wv >> 16))
		}
	}
	return d.Ink2Attend(q, r, sinks, relBias, compute.Ink2At(tmpK, 0), compute.Ink2At(tmpV, 0), scores, out, meta, p, extent, window, cap)
}

func (d *prefixTestDevice) Ink2PHalfKVReady() error { return nil }
func (d *prefixTestDevice) Ink2PQKVPostHalf(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}
func (d *prefixTestDevice) Ink2PAttendHalf(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int, int) error {
	return nil
}

func (d *prefixTestDevice) Ink2PFP8KVReady() error { return nil }
func (d *prefixTestDevice) Ink2QKVPostFP8(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv) error {
	kvDim := p.KVHeads * p.HeadDim
	tmpK, _ := d.Ink2Alloc(cap * kvDim)
	tmpV, _ := d.Ink2Alloc(cap * kvDim)
	if err := d.Ink2QKVPost(q, r, k, v, qclip, rclip, compute.Ink2At(tmpK, 0), compute.Ink2At(tmpV, 0), meta, p, cap, cv); err != nil {
		return err
	}
	pos := int(int32(math.Float32bits(d.slice(meta)[0])))
	slot := pos
	if !p.Global {
		slot = pos % cap
	}
	kf, _ := d.Backend.Host(tmpK)
	vf, _ := d.Backend.Host(tmpV)
	kc, vc := d.slice(kcache), d.slice(vcache)
	ks, vs := d.slice(kscale), d.slice(vscale)
	encodeRow := func(src, dstCodes, dstScales []float32) {
		for kvh := 0; kvh < p.KVHeads; kvh++ {
			var amax float32
			for i := 0; i < p.HeadDim; i++ {
				if a := float32(math.Abs(float64(src[slot*kvDim+kvh*p.HeadDim+i]))); a > amax {
					amax = a
				}
			}
			sc := amax / 127.0
			if sc < 1e-8 {
				sc = 1e-8
			}
			dstScales[slot*p.KVHeads+kvh] = sc
			for w := 0; w < p.HeadDim/4; w++ {
				var packed uint32
				for b := 0; b < 4; b++ {
					q := int(math.Round(float64(src[slot*kvDim+kvh*p.HeadDim+w*4+b] / sc)))
					if q < -127 {
						q = -127
					} else if q > 127 {
						q = 127
					}
					packed |= uint32(byte(int8(q))) << (8 * b)
				}
				dstCodes[(slot*kvDim+kvh*p.HeadDim)/4+w] = math.Float32frombits(packed)
			}
		}
	}
	encodeRow(kf, kc, ks)
	encodeRow(vf, vc, vs)
	return nil
}

func (d *prefixTestDevice) Ink2AttendFP8(q, r, sinks, relBias, kcache, vcache, kscale, vscale, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int) error {
	pos := int(int32(math.Float32bits(d.slice(meta)[0])))
	kvDim := p.KVHeads * p.HeadDim
	tmpK, _ := d.Ink2Alloc(cap * kvDim)
	tmpV, _ := d.Ink2Alloc(cap * kvDim)
	kf, _ := d.Backend.Host(tmpK)
	vf, _ := d.Backend.Host(tmpV)
	kc, vc := d.slice(kcache), d.slice(vcache)
	ks, vs := d.slice(kscale), d.slice(vscale)
	start := 0
	if !p.Global && window > 0 && pos-window+1 > 0 {
		start = pos - window + 1
	}
	for j := start; j <= pos; j++ {
		idx := j
		if !p.Global {
			idx = j % cap
		}
		for kvh := 0; kvh < p.KVHeads; kvh++ {
			sk, sv := ks[idx*p.KVHeads+kvh], vs[idx*p.KVHeads+kvh]
			for w := 0; w < p.HeadDim/4; w++ {
				pk := math.Float32bits(kc[(idx*kvDim+kvh*p.HeadDim)/4+w])
				pv := math.Float32bits(vc[(idx*kvDim+kvh*p.HeadDim)/4+w])
				for b := 0; b < 4; b++ {
					kf[idx*kvDim+kvh*p.HeadDim+w*4+b] = float32(int8(byte(pk>>(8*b)))) * sk
					vf[idx*kvDim+kvh*p.HeadDim+w*4+b] = float32(int8(byte(pv>>(8*b)))) * sv
				}
			}
		}
	}
	return d.Ink2Attend(q, r, sinks, relBias, compute.Ink2At(tmpK, 0), compute.Ink2At(tmpV, 0), scores, out, meta, p, extent, window, cap)
}

func (d *prefixTestDevice) Ink2PQKVPostFP8(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}

func (d *prefixTestDevice) Ink2PAttendFP8(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int, int) error {
	return nil
}

type prefixTestEmbedding struct {
	rows, cols int
	data       []float32
}

func (e prefixTestEmbedding) Row(token int, dst []float32) {
	t := ((token % e.rows) + e.rows) % e.rows
	copy(dst, e.data[t*e.cols:(t+1)*e.cols])
}

func testDetFloats(n int, seed, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = scale * float32(math.Sin(float64(seed)*1.19+float64(i)*0.53))
	}
	return out
}

func testOnes(n int, v float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func newPrefixTestModel(t *testing.T, be *prefixTestDevice) *Inkling2LoopModel {
	t.Helper()
	hp := model.Inkling2LoopHParams{
		Hidden:             8,
		Layers:             4,
		Heads:              2,
		KVHeads:            1,
		HeadDim:            4,
		VocabUnpadded:      12,
		SlidingWindow:      4,
		RelLength:          2,
		RelExtent:          4,
		ConvKernel:         3,
		AttnResBlock:       2,
		Experts:            2,
		ExpertsUsed:        1,
		ExpertWidth:        4,
		CoreStart:          2,
		CoreEnd:            4,
		Passes:             2,
		GateRank:           4,
		LocalLayer:         []bool{true, false, true, false},
		NormEps:            1e-6,
		QKClip:             8,
		RelClip:            8,
		LogScaleFloor:      8,
		LogScaleAlpha:      0.1,
		SituBeta:           1.0,
		SituLinearBeta:     1.0,
		ExpertWeightsScale: 1.0,
		FinalLogitSoftcap:  30.0,
	}
	kvDim := hp.KVHeads * hp.HeadDim
	qDim := hp.Heads * hp.HeadDim
	denseMat := func(rows, cols int, seed, scale float32) *model.Inkling2LoopDenseMatrix {
		return &model.Inkling2LoopDenseMatrix{R: rows, C: cols, Data: testDetFloats(rows*cols, seed, scale)}
	}
	devExperts := func(rows, cols int, seed, scale float32) *inkling2DeviceExperts {
		views := make([]compute.Tensor, hp.Experts)
		for e := 0; e < hp.Experts; e++ {
			views[e] = compute.NewF32(be, []int{rows, cols}, testDetFloats(rows*cols, seed+float32(e)*0.9, scale))
		}
		return &inkling2DeviceExperts{w: views, rows: rows, padK: cols}
	}

	layers := make([]model.Inkling2LoopLayerWeights, hp.Layers)
	for i := 0; i < hp.Layers; i++ {
		seed := float32(i + 1)
		local := hp.LocalLayer[i]
		extent := hp.RelExtent
		if local {
			extent = hp.SlidingWindow
		}
		var attnRes [][]float32
		if i > 0 {
			if local {
				attnRes = [][]float32{testDetFloats(hp.Hidden, seed+0.1, 0.2)}
			} else {
				attnRes = [][]float32{
					testDetFloats(hp.Hidden, seed+0.1, 0.2),
					testDetFloats(hp.Hidden, seed+0.2, 0.2),
					testDetFloats(hp.Hidden, seed+0.3, 0.2),
				}
			}
		}
		lw := model.Inkling2LoopLayerWeights{
			Wq:           denseMat(qDim, hp.Hidden, seed+1, 0.2),
			Wk:           denseMat(kvDim, hp.Hidden, seed+2, 0.2),
			Wv:           denseMat(kvDim, hp.Hidden, seed+3, 0.2),
			Wo:           denseMat(hp.Hidden, qDim, seed+4, 0.2),
			Gate:         denseMat(qDim, hp.Hidden, seed+5, 0.2),
			Rel:          denseMat(hp.Heads*hp.RelLength, hp.Hidden, seed+6, 0.2),
			SinkProj:     denseMat(hp.Heads, hp.Hidden, seed+7, 0.1),
			AttnNorm:     testOnes(hp.Hidden, 1),
			QInpNorm:     testOnes(hp.Hidden, 1),
			KInpNorm:     testOnes(hp.Hidden, 1),
			VInpNorm:     testOnes(hp.Hidden, 1),
			QClip:        testOnes(qDim, 2),
			RelBias:      testDetFloats(extent*hp.RelLength, seed+8, 0.1),
			RelClip:      testOnes(hp.Heads*hp.RelLength, 2),
			KConv:        testDetFloats(kvDim*hp.ConvKernel, seed+9, 0.15),
			VConv:        testDetFloats(kvDim*hp.ConvKernel, seed+10, 0.15),
			AttnPostConv: testDetFloats(hp.Hidden*hp.ConvKernel, seed+11, 0.15),
			FFNPostConv:  testDetFloats(hp.Hidden*hp.ConvKernel, seed+12, 0.15),
			AttnResQuery: attnRes,
			FFNResQuery:  testDetFloats(hp.Hidden, seed+13, 0.2),
			FFNNorm:      testOnes(hp.Hidden, 1),
			ExpProbsBias: testDetFloats(hp.Experts, seed+14, 0.05),
			Router:       denseMat(hp.Experts, hp.Hidden, seed+15, 0.2),
			GateUp:       devExperts(2*hp.ExpertWidth, hp.Hidden, seed+16, 0.2),
			Down:         devExperts(hp.Hidden, hp.ExpertWidth, seed+17, 0.2),
			GateUpScale:  testOnes(hp.Experts, 1),
			DownScale:    testOnes(hp.Experts, 1),
		}
		if !local {
			lw.KConvProj = denseMat(hp.KVHeads*hp.ConvKernel, hp.Hidden, seed+18, 0.1)
			lw.VConvProj = denseMat(hp.KVHeads*hp.ConvKernel, hp.Hidden, seed+19, 0.1)
		}
		layers[i] = lw
	}
	core := hp.CoreEnd - hp.CoreStart
	ctrls := make([]model.Inkling2LoopControllerWeights, core)
	for c := 0; c < core; c++ {
		seed := float32(100 + c)
		for ch := 0; ch < 4; ch++ {
			ctrls[c].MsgQuery[ch] = testDetFloats(hp.Hidden, seed+float32(ch)*0.2, 0.2)
			ctrls[c].Down[ch] = denseMat(hp.GateRank, 2*hp.Hidden, seed+float32(ch)*0.3, 0.15)
			ctrls[c].Gain[ch] = denseMat(hp.Hidden, hp.GateRank, seed+float32(ch)*0.4, 0.15)
			ctrls[c].Forget[ch] = denseMat(hp.Hidden, hp.GateRank, seed+float32(ch)*0.5, 0.15)
		}
	}
	return &Inkling2LoopModel{
		HP: hp,
		Weights: &model.Inkling2LoopWeights{
			TokEmbd:        prefixTestEmbedding{rows: hp.VocabUnpadded, cols: hp.Hidden, data: testDetFloats(hp.VocabUnpadded*hp.Hidden, 200, 0.5)},
			TokNorm:        testOnes(hp.Hidden, 1),
			Output:         denseMat(hp.VocabUnpadded, hp.Hidden, 201, 0.35),
			OutputNorm:     testOnes(hp.Hidden, 1),
			OutputResQuery: testDetFloats(hp.Hidden, 202, 0.2),
			Layers:         layers,
			Controllers:    ctrls,
		},
	}
}

func greedyRunTokensAndLogits(t *testing.T, sess *inkling2ResidentSession, prompt []int, steps int) ([]int, [][]float32) {
	t.Helper()
	next, err := sess.Prefill(context.Background(), prompt)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	toks := make([]int, 0, steps+1)
	logits := make([][]float32, 0, steps+1)
	l0, err := sess.LastLogits()
	if err != nil {
		t.Fatalf("LastLogits after prefill: %v", err)
	}
	toks = append(toks, next)
	logits = append(logits, append([]float32(nil), l0...))
	for i := 0; i < steps; i++ {
		next, err = sess.Step(next)
		if err != nil {
			t.Fatalf("Step %d: %v", i, err)
		}
		li, err := sess.LastLogits()
		if err != nil {
			t.Fatalf("LastLogits step %d: %v", i, err)
		}
		toks = append(toks, next)
		logits = append(logits, append([]float32(nil), li...))
	}
	return toks, logits
}

func assertFloat32BitsEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len mismatch: got %d want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] bits mismatch: got %g (%08x) want %g (%08x)",
				label, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func prefixEnvLookup(kv, localKV string) func(string) (string, bool) {
	return func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_KV":
			return kv, kv != ""
		case "FAK_INKLING2_LOCAL_SWA_KV":
			return localKV, localKV != ""
		default:
			return "", false
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestInkling2PrefixCheckpointRestoreMatchesCold(t *testing.T) {
	t.Parallel()
	lookupF32 := prefixEnvLookup("f32", "f32")

	// Prefix length 7 wraps SlidingWindow=4 on ring cap=4, testing both global KV,
	// wrapped local SWA rings, and short-conv rings.
	prefix := []int{1, 4, 2, 7, 3, 9, 5}
	suffix := []int{6, 0, 8, 11}
	fullPrompt := append(append([]int(nil), prefix...), suffix...)

	beCold := newPrefixTestDevice(t)
	mCold := newPrefixTestModel(t, beCold)
	rCold, err := newInkling2ResidentRingForLookup(mCold, beCold, 32, 8192, 1, 1, false, lookupF32)
	if err != nil {
		t.Fatalf("newInkling2ResidentRing cold: %v", err)
	}
	defer rCold.close()
	sessCold := &inkling2ResidentSession{s: rCold}

	// 1. Capture state at prefix boundary on cold session, then continue cold decode.
	if _, err := sessCold.Prefill(context.Background(), prefix); err != nil {
		t.Fatalf("cold prefill prefix: %v", err)
	}
	prefixCk, err := sessCold.CheckpointPrefix(prefix, "unit-f32")
	if err != nil {
		t.Fatalf("CheckpointPrefix: %v", err)
	}
	if prefixCk.Pos != len(prefix) {
		t.Fatalf("checkpoint Pos = %d, want %d", prefixCk.Pos, len(prefix))
	}
	// Finish full prompt cold from position 0 on a completely fresh cold session.
	beColdFull := newPrefixTestDevice(t)
	mColdFull := newPrefixTestModel(t, beColdFull)
	rColdFull, err := newInkling2ResidentRingForLookup(mColdFull, beColdFull, 32, 8192, 1, 1, false, lookupF32)
	if err != nil {
		t.Fatalf("newInkling2ResidentRing cold full: %v", err)
	}
	defer rColdFull.close()
	sessColdFull := &inkling2ResidentSession{s: rColdFull}
	wantToks, wantLogits := greedyRunTokensAndLogits(t, sessColdFull, fullPrompt, 4)

	// 2. Dirty a third session with an unrelated sequence, restore prefixCk via cache,
	// verify byte-exact state restore and identical suffix + decode continuation.
	cache, err := NewInkling2ResidentPrefixCache(DefaultInkling2ResidentPrefixConfig())
	if err != nil {
		t.Fatalf("NewInkling2ResidentPrefixCache: %v", err)
	}
	if err := cache.Put(prefixCk); err != nil {
		t.Fatalf("cache.Put: %v", err)
	}

	beWarm := newPrefixTestDevice(t)
	mWarm := newPrefixTestModel(t, beWarm)
	rWarm, err := newInkling2ResidentRingForLookup(mWarm, beWarm, 32, 8192, 1, 1, false, lookupF32)
	if err != nil {
		t.Fatalf("newInkling2ResidentRing warm: %v", err)
	}
	defer rWarm.close()
	sessWarm := &inkling2ResidentSession{s: rWarm}
	if _, err := sessWarm.Prefill(context.Background(), []int{11, 10, 9, 8, 7, 6, 5, 4, 3}); err != nil {
		t.Fatalf("dirty prefill: %v", err)
	}
	if err := sessWarm.RestorePrefix(prefixCk); err != nil {
		t.Fatalf("RestorePrefix: %v", err)
	}
	// Byte-exact state check against prefixCk captured on cold session.
	restoredCk, err := sessWarm.CheckpointPrefix(prefix, "unit-f32")
	if err != nil {
		t.Fatalf("CheckpointPrefix after restore: %v", err)
	}
	assertFloat32BitsEqual(t, "restored.LastLogits", restoredCk.LastLogits, prefixCk.LastLogits)
	for slot := range prefixCk.Slots {
		assertFloat32BitsEqual(t, "slot.Conv", restoredCk.Slots[slot].Conv, prefixCk.Slots[slot].Conv)
		assertFloat32BitsEqual(t, "slot.K", restoredCk.Slots[slot].K, prefixCk.Slots[slot].K)
		assertFloat32BitsEqual(t, "slot.V", restoredCk.Slots[slot].V, prefixCk.Slots[slot].V)
	}

	// Verify PrefillWithPrefixCache matches longest prefix and executes ONLY suffix steps.
	beWarm.stepCount = 0
	nextWarm, matched, err := sessWarm.PrefillWithPrefixCache(context.Background(), cache, "unit-f32", fullPrompt)
	if err != nil {
		t.Fatalf("PrefillWithPrefixCache: %v", err)
	}
	if matched != len(prefix) {
		t.Fatalf("matched prefix = %d, want %d", matched, len(prefix))
	}
	if beWarm.stepCount != len(suffix) {
		t.Fatalf("PrefillWithPrefixCache executed %d steps, want suffix-only %d", beWarm.stepCount, len(suffix))
	}
	lWarm0, err := sessWarm.LastLogits()
	if err != nil {
		t.Fatalf("LastLogits warm: %v", err)
	}
	gotToks := []int{nextWarm}
	gotLogits := [][]float32{append([]float32(nil), lWarm0...)}
	curTok := nextWarm
	for i := 0; i < 4; i++ {
		curTok, err = sessWarm.Step(curTok)
		if err != nil {
			t.Fatalf("warm Step %d: %v", i, err)
		}
		li, err := sessWarm.LastLogits()
		if err != nil {
			t.Fatalf("warm LastLogits %d: %v", i, err)
		}
		gotToks = append(gotToks, curTok)
		gotLogits = append(gotLogits, append([]float32(nil), li...))
	}
	if !slices.Equal(gotToks, wantToks) {
		t.Fatalf("continuation tokens mismatch: got %v want %v", gotToks, wantToks)
	}
	for i := range wantLogits {
		assertFloat32BitsEqual(t, "continuation_logits", gotLogits[i], wantLogits[i])
	}

	// 3. Exact full-prompt cache hit preserves LastLogits cleanly with 0 recomputed steps.
	fullCk, err := sessColdFull.CheckpointPrefix(append(slices.Clone(fullPrompt), wantToks[:4]...), "unit-f32")
	if err != nil {
		t.Fatalf("CheckpointPrefix full: %v", err)
	}
	if err := cache.Put(fullCk); err != nil {
		t.Fatalf("cache.Put full: %v", err)
	}
	beWarm.stepCount = 0
	exactNext, exactMatched, err := sessWarm.PrefillWithPrefixCache(context.Background(), cache, "unit-f32", append(slices.Clone(fullPrompt), wantToks[:4]...))
	if err != nil {
		t.Fatalf("exact hit PrefillWithPrefixCache: %v", err)
	}
	if exactMatched != len(fullPrompt)+4 || beWarm.stepCount != 0 || exactNext != wantToks[4] {
		t.Fatalf("exact hit mismatch: matched=%d steps=%d next=%d want %d", exactMatched, beWarm.stepCount, exactNext, wantToks[4])
	}

	// 4. Default newInkling2Resident behavior stays identical when prefix cache is unused.
	beDefaultA := newPrefixTestDevice(t)
	rDefaultA, err := newInkling2Resident(newPrefixTestModel(t, beDefaultA), beDefaultA)
	if err != nil {
		t.Fatalf("newInkling2Resident default A: %v", err)
	}
	defer rDefaultA.close()
	beDefaultB := newPrefixTestDevice(t)
	rDefaultB, err := newInkling2Resident(newPrefixTestModel(t, beDefaultB), beDefaultB)
	if err != nil {
		t.Fatalf("newInkling2Resident default B: %v", err)
	}
	defer rDefaultB.close()
	if (&inkling2ResidentSession{s: rDefaultA}).HeldBytes() != (&inkling2ResidentSession{s: rDefaultB}).HeldBytes() || rDefaultA.pos != 0 {
		t.Fatal("default newInkling2Resident baseline invariants drifted")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestInkling2PrefixCacheBoundedLRUEviction(t *testing.T) {
	t.Parallel()
	if _, err := NewInkling2ResidentPrefixCache(Inkling2ResidentPrefixConfig{}); !errors.Is(err, ErrInkling2PrefixConfigInvalid) {
		t.Fatalf("expected ErrInkling2PrefixConfigInvalid for unbounded 0/0 config, got %v", err)
	}
	if _, err := NewInkling2ResidentPrefixCache(Inkling2ResidentPrefixConfig{MaxEntries: -1, MaxBytes: 100}); !errors.Is(err, ErrInkling2PrefixConfigInvalid) {
		t.Fatalf("expected ErrInkling2PrefixConfigInvalid for negative MaxEntries, got %v", err)
	}

	mkCk := func(scope string, toks []int, payloadFloats int) *Inkling2ResidentPrefixCheckpoint {
		ck := &Inkling2ResidentPrefixCheckpoint{
			Scope:        scope,
			PrefixTokens: slices.Clone(toks),
			HParamsJSON:  `{"Hidden":8}`,
			Pos:          len(toks),
			KVDType:      Inkling2KVF32,
			LocalKVDType: Inkling2KVF32,
			Slots: []Inkling2PrefixSlotState{{
				Conv: make([]float32, 8),
				K:    make([]float32, payloadFloats),
				V:    make([]float32, payloadFloats),
			}},
		}
		ck.byteSize = ck.computeByteSize()
		return ck
	}

	// MaxEntries LRU eviction: A, B inserted, touch A, insert C -> B evicted, A & C kept.
	cEntries, err := NewInkling2ResidentPrefixCache(Inkling2ResidentPrefixConfig{MaxEntries: 2})
	if err != nil {
		t.Fatalf("NewInkling2ResidentPrefixCache: %v", err)
	}
	ckA := mkCk("s", []int{1, 2}, 4)
	ckB := mkCk("s", []int{3, 4}, 4)
	ckC := mkCk("s", []int{5, 6}, 4)
	if err := cEntries.Put(ckA); err != nil {
		t.Fatal(err)
	}
	if err := cEntries.Put(ckB); err != nil {
		t.Fatal(err)
	}
	if _, ok := cEntries.GetExact("s", Inkling2KVF32, Inkling2KVF32, []int{1, 2}); !ok {
		t.Fatal("expected hit on A before C insertion")
	}
	if err := cEntries.Put(ckC); err != nil {
		t.Fatal(err)
	}
	if _, ok := cEntries.GetExact("s", Inkling2KVF32, Inkling2KVF32, []int{3, 4}); ok {
		t.Fatal("expected B (LRU) to be evicted after C insertion")
	}
	if st := cEntries.Stats(); st.Entries != 2 || st.Evictions != 1 {
		t.Fatalf("unexpected cache stats after entry eviction: %+v", st)
	}

	// MaxBytes LRU eviction + single-entry budget rejection without evicting existing entries.
	singleBytes := ckA.ByteSize()
	cBytes, err := NewInkling2ResidentPrefixCache(Inkling2ResidentPrefixConfig{
		MaxEntries: 16,
		MaxBytes:   singleBytes * 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := cBytes.Put(ckA); err != nil {
		t.Fatal(err)
	}
	if err := cBytes.Put(ckB); err != nil {
		t.Fatal(err)
	}
	huge := mkCk("s", []int{9, 10}, 512)
	if err := cBytes.Put(huge); !errors.Is(err, ErrInkling2PrefixTooLarge) {
		t.Fatalf("expected ErrInkling2PrefixTooLarge, got %v", err)
	}
	if st := cBytes.Stats(); st.Entries != 2 || st.Rejections != 1 || st.Bytes > singleBytes*2 {
		t.Fatalf("oversized Put corrupted cache or missed rejection accounting: %+v", st)
	}
	// Inserting ckC under MaxBytes budget triggers byte-driven eviction of A.
	if err := cBytes.Put(ckC); err != nil {
		t.Fatal(err)
	}
	if st := cBytes.Stats(); st.Bytes > singleBytes*2 || st.Evictions != 1 {
		t.Fatalf("MaxBytes eviction invariant failed: %+v (limit=%d)", st, singleBytes*2)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestInkling2PrefixCheckpointCompressedKVAndMismatchRefusal(t *testing.T) {
	t.Parallel()
	prefix := []int{2, 5, 1, 7, 4, 8}
	suffix := []int{3, 9, 6}
	full := append(slices.Clone(prefix), suffix...)

	for _, tc := range []struct {
		name        string
		kvEnv       string
		localKV     string
		wantKV      Inkling2KVDType
		wantLocalKV Inkling2KVDType
	}{
		{name: "f16_global_f16_local", kvEnv: "f16", localKV: "f16", wantKV: Inkling2KVF16, wantLocalKV: Inkling2KVF16},
		{name: "fp8_global_fp8_local", kvEnv: "fp8", localKV: "fp8", wantKV: Inkling2KVFP8, wantLocalKV: Inkling2KVFP8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			lookupKV := prefixEnvLookup(tc.kvEnv, tc.localKV)
			lookupF32 := prefixEnvLookup("f32", "f32")

			beCold := newPrefixTestDevice(t)
			rCold, err := newInkling2ResidentRingForLookup(newPrefixTestModel(t, beCold), beCold, 32, 8192, 1, 1, false, lookupKV)
			if err != nil {
				t.Fatalf("newInkling2ResidentRing cold: %v", err)
			}
			defer rCold.close()
			if rCold.kvDType() != tc.wantKV || rCold.localKVDtype != tc.wantLocalKV {
				t.Fatalf("resolved KV=(%s,%s), want (%s,%s)", rCold.kvDType(), rCold.localKVDtype, tc.wantKV, tc.wantLocalKV)
			}
			sessCold := &inkling2ResidentSession{s: rCold}
			wantToks, wantLogits := greedyRunTokensAndLogits(t, sessCold, full, 3)

			bePref := newPrefixTestDevice(t)
			rPref, err := newInkling2ResidentRingForLookup(newPrefixTestModel(t, bePref), bePref, 32, 8192, 1, 1, false, lookupKV)
			if err != nil {
				t.Fatalf("newInkling2ResidentRing prefix: %v", err)
			}
			defer rPref.close()
			sessPref := &inkling2ResidentSession{s: rPref}
			if _, err := sessPref.Prefill(context.Background(), prefix); err != nil {
				t.Fatalf("prefix prefill: %v", err)
			}
			ck, err := sessPref.CheckpointPrefix(prefix, tc.name)
			if err != nil {
				t.Fatalf("CheckpointPrefix: %v", err)
			}

			beWarm := newPrefixTestDevice(t)
			rWarm, err := newInkling2ResidentRingForLookup(newPrefixTestModel(t, beWarm), beWarm, 32, 8192, 1, 1, false, lookupKV)
			if err != nil {
				t.Fatalf("newInkling2ResidentRing warm: %v", err)
			}
			defer rWarm.close()
			sessWarm := &inkling2ResidentSession{s: rWarm}
			if err := sessWarm.RestorePrefix(ck); err != nil {
				t.Fatalf("RestorePrefix: %v", err)
			}
			next, err := sessWarm.Prefill(context.Background(), suffix)
			if err != nil {
				t.Fatalf("suffix prefill: %v", err)
			}
			l0, err := sessWarm.LastLogits()
			if err != nil {
				t.Fatalf("LastLogits: %v", err)
			}
			gotToks := []int{next}
			gotLogits := [][]float32{append([]float32(nil), l0...)}
			cur := next
			for i := 0; i < 3; i++ {
				cur, err = sessWarm.Step(cur)
				if err != nil {
					t.Fatalf("Step %d: %v", i, err)
				}
				li, err := sessWarm.LastLogits()
				if err != nil {
					t.Fatalf("LastLogits %d: %v", i, err)
				}
				gotToks = append(gotToks, cur)
				gotLogits = append(gotLogits, append([]float32(nil), li...))
			}
			if !slices.Equal(gotToks, wantToks) {
				t.Fatalf("%s tokens mismatch: got %v want %v", tc.name, gotToks, wantToks)
			}
			for i := range wantLogits {
				assertFloat32BitsEqual(t, tc.name, gotLogits[i], wantLogits[i])
			}

			// Mismatched KV dtype session refuses ck without mutating session state.
			beF32 := newPrefixTestDevice(t)
			rF32, err := newInkling2ResidentRingForLookup(newPrefixTestModel(t, beF32), beF32, 32, 8192, 1, 1, false, lookupF32)
			if err != nil {
				t.Fatalf("newInkling2ResidentRing f32: %v", err)
			}
			defer rF32.close()
			sessF32 := &inkling2ResidentSession{s: rF32}
			if err := sessF32.RestorePrefix(ck); err == nil {
				t.Fatalf("expected dtype mismatch error restoring %s checkpoint into f32 session", tc.name)
			}
			if sessF32.Pos() != 0 {
				t.Fatalf("failed RestorePrefix mutated session Pos to %d", sessF32.Pos())
			}
		})
	}
}
