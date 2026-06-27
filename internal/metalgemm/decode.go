//go:build darwin && cgo && fakmetal

// decode.go — Go side of the GPU-resident Q8 decode forward (decode.m, issue #67). The decode
// twin of the Prefill binding in metalgemm.go: DecodeConfig/DecodeLayer register the model once
// (Q8 projection weight ids from the q8.m table + f16 norm/bias ids from the gW table), then
// DecodeStep runs one token through the whole model on the GPU in one command buffer.

package metalgemm

/*
void mg_decode_config(int nLayers, int H, int hd, int nH, int nKV, int Im, float eps, float theta, float scale, int attnBias);
void mg_decode_layer(int layer, int q, int k, int v, int o, int gate, int up, int down, int inNorm, int postNorm, int qb, int kb, int vb);
void mg_decode_head(int finalNormID, int headWid, int vocab);
void mg_decode_reset(void);
int  mg_decode_step(const float* xEmbed, const float* Kctx, const float* Vctx, int L,
                    float* lastPre, float* newKraw, float* newKpost, float* newV, float* logits, int seedFlag);
*/
import "C"

import "unsafe"

// DecodeConfig records the model geometry the resident decode forward needs. scale is the attention
// score multiplier (cfg.attnScale()); attnBias is whether q/k/v carry a bias. Call once per model
// before DecodeLayer.
func DecodeConfig(nLayers, H, hd, nH, nKV, I int, eps, theta, scale float32, attnBias bool) {
	b := C.int(0)
	if attnBias {
		b = 1
	}
	C.mg_decode_config(C.int(nLayers), C.int(H), C.int(hd), C.int(nH), C.int(nKV), C.int(I),
		C.float(eps), C.float(theta), C.float(scale), b)
}

// DecodeLayer records one layer's resident weight ids: the seven Q8 projection handles (q/k/v/o,
// gate/up/down — q8.m wids) and the f16 norm/bias ids (gW table; qb/kb/vb == -1 when no bias).
func DecodeLayer(layer, q, k, v, o, gate, up, down, inNorm, postNorm, qb, kb, vb int) {
	C.mg_decode_layer(C.int(layer), C.int(q), C.int(k), C.int(v), C.int(o), C.int(gate),
		C.int(up), C.int(down), C.int(inNorm), C.int(postNorm), C.int(qb), C.int(kb), C.int(vb))
}

// DecodeHead registers the final RMSNorm vector (gW id), the Q8 LM-head weight (q8.m wid) and the
// vocab size, so DecodeStep (when given a logits buffer) runs the final norm + vocab projection on
// the GPU and returns logits directly — no CPU head, no post-forward round-trip. Optional.
func DecodeHead(finalNormID, headWid, vocab int) {
	C.mg_decode_head(C.int(finalNormID), C.int(headWid), C.int(vocab))
}

// DecodeReset clears the per-model decode topology (geometry + per-layer weight-id table). The
// compiled pipelines are model-independent and kept.
func DecodeReset() { C.mg_decode_reset() }

// DecodeStep runs one decode token through the whole model on the GPU in ONE command buffer.
// xEmbed is the new token's f32 embedding [H]. Kctx/Vctx are the per-layer post-RoPE K and V from
// the CPU cache, laid out [nLayers*L*w] (w = nKV*hd); pass nil-safe empty slices when L == 0. L is
// the number of cached positions (== the new token's absolute position). Returns the pre-final-norm
// hidden [H] (caller applies final norm + head) and the new token's per-layer pre-RoPE K, post-RoPE
// K and V [nLayers*w] (caller appends to its cache). ok is false if the backend declined.
// vocab > 0 (with DecodeHead registered) makes the forward also run the final norm + LM head on the
// GPU and return logits [vocab]; vocab == 0 returns logits == nil and the caller applies the head.
// seed==true uploads the L-row Kctx/Vctx context into the resident KV (a new/desynced sequence);
// seed==false appends onto the resident KV with no re-upload (the steady decode path) and ignores
// Kctx/Vctx — returns ok==false if the resident length disagrees, so the caller re-seeds.
func DecodeStep(xEmbed, Kctx, Vctx []float32, L, nLayers, w, H, vocab int, seed bool) (lastPre, newKpost, newV, logits []float32, ok bool) {
	if !Available() || len(xEmbed) < H {
		return nil, nil, nil, nil, false
	}
	lastPre = make([]float32, H)
	newKraw := make([]float32, nLayers*w) // unused (the fast Q8 decode keeps post-RoPE K/V only)
	newKpost = make([]float32, nLayers*w)
	newV = make([]float32, nLayers*w)
	var lp *C.float
	if vocab > 0 {
		logits = make([]float32, vocab)
		lp = (*C.float)(unsafe.Pointer(&logits[0]))
	}
	var kp, vp *C.float
	if seed && L > 0 {
		if len(Kctx) < nLayers*L*w || len(Vctx) < nLayers*L*w {
			return nil, nil, nil, nil, false
		}
		kp = (*C.float)(unsafe.Pointer(&Kctx[0]))
		vp = (*C.float)(unsafe.Pointer(&Vctx[0]))
	}
	seedF := C.int(0)
	if seed {
		seedF = 1
	}
	r := C.mg_decode_step((*C.float)(unsafe.Pointer(&xEmbed[0])), kp, vp, C.int(L),
		(*C.float)(unsafe.Pointer(&lastPre[0])), (*C.float)(unsafe.Pointer(&newKraw[0])),
		(*C.float)(unsafe.Pointer(&newKpost[0])), (*C.float)(unsafe.Pointer(&newV[0])), lp, seedF)
	if r != 1 {
		return nil, nil, nil, nil, false
	}
	return lastPre, newKpost, newV, logits, true
}
