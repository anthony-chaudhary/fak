package model

import (
	"math"
	"os"
	"sync"
)

// attnDecodeBatch computes per-user causal attention for ONE batched decode step. For each
// user b, the query Q[b] attends over user b's OWN full KV cache (caches[b].K[l]/V[l]); the
// result [nH*hd] is written into attnOut[b] (which the caller has zeroed). Users are fully
// independent — own cache, own length — so the work parallelises across the flat
// (user,kv-head) unit index. Each unit computes the grp query heads sharing one V head, then
// streams each value vector once while updating those grp output heads. For every individual
// output element, the j=0..nPos-1 accumulation order is unchanged, so StepBatch stays
// bit-identical to serial Step. This is the decode analogue of attnPrefillInto.
//
// W is the per-layer sliding-window bound (cfg.windowForLayer): W<0 (the default) is full
// causal attention and reduces the loops byte-for-byte to the pre-SWA path; W>=0 masks the
// score/V loops to the contiguous visible suffix (windowLoContig). This SWA mask layers on
// top of the GQA-fuse + saxpy3-SIMD perf branches — they coexist.
func attnDecodeBatch(attnOut, Q []float32, caches []*KVCache, l, B, nH, hd, w, grp, W int, scale float32, scoreDot func(a, b []float32) float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32), scoreScratch [][]float32, obs AttnObserver) [][]float32 {
	nKV := nH / grp
	units := B * nKV
	maxPos := 0
	for b := 0; b < B; b++ {
		// Precision-aware position count (#12981): kvLen on f32 equals len(K[l])/w
		// (byte-identical); on q8 it reads the packed rows, so the scratch is sized
		// from the real prefix instead of an always-zero nil K slice.
		if n := caches[b].kvLen(l); n > maxPos {
			maxPos = n
		}
	}
	useSaxpy3SIMD := B >= attnSaxpy3SIMDMinBatch
	nw := currentWorkerCount()
	if nw > units {
		nw = units
	}
	if nw <= 0 {
		return scoreScratch
	}
	scoreScratch = grow2D(scoreScratch, nw*grp, maxPos)
	work := func(wkr, nw int) {
		for u := wkr; u < units; u += nw {
			b := u / nKV
			kvh := u % nKV
			c := caches[b]
			// Precision-aware read (#12981): on f32 attentionRows returns the cache's own
			// K/V slices (zero copy, byte-identical); on q8 it dequantizes the layer so this
			// batch-decode lane can never read a nil K/V slice. kvLen is the representation-
			// independent position count (len(K[l])/w on f32, the packed row count on q8).
			Kl, Vl := c.attentionRows(l)
			nPos := c.kvLen(l)
			// SWA read-time mask: this user's query (its just-appended K row, at absolute
			// position nPos-1 since the cache is contiguous and was appended at Cache.Len())
			// attends only keys in the window. j0=0 (full causal) when W<0.
			j0 := windowLoContig(nPos, nPos-1, W)
			visible := nPos - j0
			if attnGQAFuse && grp == 3 && scoreDot3 != nil {
				h0 := kvh * grp
				q0, q1, q2 := packedHead3(Q, b, nH*hd, h0, hd)
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, visible)
				fillSoftmaxAttentionScores3(sc0, sc1, sc2, q0, q1, q2, Kl, j0, nPos, w, kvh, hd, scale, scoreDot3)
				if obs != nil { // #852: query is the just-appended row at abs pos nPos-1
					emitAttnRow(obs, l, nPos-1, h0+0, j0, sc0)
					emitAttnRow(obs, l, nPos-1, h0+1, j0, sc1)
					emitAttnRow(obs, l, nPos-1, h0+2, j0, sc2)
				}
			} else {
				for g := 0; g < grp; g++ {
					h := kvh*grp + g
					qh := packedHead(Q, b, nH*hd, h, hd)
					sc := scoreScratchHead(scoreScratch, wkr, grp, g, visible)
					fillSoftmaxAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
					if obs != nil { // #852: query is the just-appended row at abs pos nPos-1
						emitAttnRow(obs, l, nPos-1, h, j0, sc)
					}
				}
			}
			// Clear this kv-head's query-head output rows before the value accumulation
			// below adds into them. Both score branches above leave attnOut untouched, and
			// the fused branch's h0 is kvh*grp, so the rows to clear are the same set either
			// way — one loop, not one per branch.
			zeroPackedHeads(attnOut, b, nH*hd, kvh*grp, grp, hd)
			if grp == 3 {
				h0 := kvh * grp
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, visible)
				accumulatePackedAttentionValues3(attnOut, b, nH*hd, h0, hd, Vl, sc0, sc1, sc2, j0, nPos, w, kvh,
					useSaxpy3SIMD && visible >= attnSaxpy3SIMDMinPos)
				continue
			}
			accumulateAttentionGroup(attnOut, b, nH*hd, kvh*grp, grp, hd, Vl, scoreScratch, wkr*grp, j0, nPos, w, kvh)
		}
	}

	if nw <= 1 {
		work(0, 1)
		return scoreScratch
	}
	parFor(nw, nw, func(lo, hi int) {
		for k := lo; k < hi; k++ {
			work(k, nw)
		}
	})
	return scoreScratch
}

// attnPrefillMultiInto computes causal GQA attention for a rectangular multi-sequence
// prefill panel. Rows are laid out [user][token], with P new tokens per user. Each row
// attends only to that user's own cache prefix plus earlier rows from the same user; other
// users' K/V rows are never visible. This is the multi-agent analogue of attnPrefillInto.
//
// W is the per-layer sliding-window bound (cfg.windowForLayer): W<0 is full causal and the
// score/V loops reduce byte-for-byte to the pre-SWA path; W>=0 masks each query to the
// contiguous visible suffix (windowLoContig). scoreScratch is a reusable per-worker softmax
// scratch (one row per worker); it is grown as needed and returned so the caller can pool it
// across layers/calls (pass nil to allocate fresh). attnOut is written per (row,head) output
// slice via saxpy, which `+=`-accumulates, so the caller MUST zero any region attnOut covers
// before calling — load-bearing when attnOut is a reused buffer.
func attnPrefillMultiInto(attnOut, Q []float32, caches []*KVCache, baseB []int, layer, P, nH, hd, w, grp, W int, scale float32, scoreDot func(a, b []float32) float32, scoreScratch [][]float32) [][]float32 {
	B := len(caches)
	units := B * P * nH
	maxPos := 0
	for b := 0; b < B; b++ {
		if n := baseB[b] + P; n > maxPos {
			maxPos = n
		}
	}
	nw := currentWorkerCount()
	if nw > units {
		nw = units
	}
	if nw <= 0 {
		return scoreScratch
	}
	scoreScratch = grow2D(scoreScratch, nw, maxPos)

	work := func(wkr, nw int) {
		scores := scoreScratch[wkr][:maxPos]
		for u := wkr; u < units; u += nw {
			row := u / nH
			h := u % nH
			b := row / P
			t := row % P
			c := caches[b]
			// Precision-aware read (#12981): on f32 attentionRows returns the cache's own
			// K/V slices (zero copy, byte-identical); on q8 it dequantizes the layer so the
			// lane can never read nil.
			Kl, Vl := c.attentionRows(layer)
			nPos := baseB[b] + t + 1
			// SWA read-time mask: query (absolute position baseB[b]+t) over the contiguous
			// prefill cache. j0=0 (full causal) when W<0.
			j0 := windowLoContig(nPos, baseB[b]+t, W)
			kvh := h / grp
			qh := packedHead(Q, row, nH*hd, h, hd)
			sc := scores[:nPos-j0]
			fillAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
			softmaxInPlace(sc)
			out := packedHead(attnOut, row, nH*hd, h, hd)
			accumulateAttentionValues(out, Vl, sc, j0, nPos, w, kvh, hd)
		}
	}

	if nw == 1 {
		work(0, 1)
		return scoreScratch
	}
	var wg sync.WaitGroup
	for k := 0; k < nw; k++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			work(k, nw)
		}(k)
	}
	wg.Wait()
	return scoreScratch
}

// attnPrefillMultiGQAInto is the GQA-fused prefill analogue of attnDecodeBatch's fast path.
// It accepts the same sliding-window W bound as attnPrefillMultiInto; W<0 is full causal.
func attnPrefillMultiGQAInto(attnOut, Q []float32, caches []*KVCache, baseB []int, layer, P, nH, hd, w, grp, W int, scale float32, scoreDot func(a, b []float32) float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32), scoreScratch [][]float32) [][]float32 {
	B := len(caches)
	nKV := nH / grp
	units := B * P * nKV
	maxPos := 0
	for b := 0; b < B; b++ {
		if n := baseB[b] + P; n > maxPos {
			maxPos = n
		}
	}
	nw := currentWorkerCount()
	if nw > units {
		nw = units
	}
	if nw <= 0 {
		return scoreScratch
	}
	scoreScratch = grow2D(scoreScratch, nw*grp, maxPos)
	useSaxpy3SIMD := B >= attnSaxpy3SIMDMinBatch

	work := func(wkr, nw int) {
		for u := wkr; u < units; u += nw {
			row := u / nKV
			kvh := u % nKV
			b := row / P
			t := row % P
			c := caches[b]
			// Precision-aware read (#12981): on f32 attentionRows returns the cache's own
			// K/V slices (zero copy, byte-identical); on q8 it dequantizes the layer so the
			// lane can never read nil.
			Kl, Vl := c.attentionRows(layer)
			nPos := baseB[b] + t + 1
			j0 := windowLoContig(nPos, baseB[b]+t, W)
			span := nPos - j0
			if attnGQAFuse && grp == 3 && scoreDot3 != nil {
				h0 := kvh * grp
				q0, q1, q2 := packedHead3(Q, row, nH*hd, h0, hd)
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, span)
				fillSoftmaxAttentionScores3(sc0, sc1, sc2, q0, q1, q2, Kl, j0, nPos, w, kvh, hd, scale, scoreDot3)
			} else {
				for g := 0; g < grp; g++ {
					h := kvh*grp + g
					qh := packedHead(Q, row, nH*hd, h, hd)
					sc := scoreScratchHead(scoreScratch, wkr, grp, g, span)
					fillSoftmaxAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
				}
			}
			if grp == 3 {
				h0 := kvh * grp
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, span)
				accumulatePackedAttentionValues3(attnOut, row, nH*hd, h0, hd, Vl, sc0, sc1, sc2, j0, nPos, w, kvh,
					useSaxpy3SIMD && span >= attnSaxpy3SIMDMinPos)
				continue
			}
			accumulateAttentionGroup(attnOut, row, nH*hd, kvh*grp, grp, hd, Vl, scoreScratch, wkr*grp, j0, nPos, w, kvh)
		}
	}

	if nw == 1 {
		work(0, 1)
		return scoreScratch
	}
	var wg sync.WaitGroup
	for k := 0; k < nw; k++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			work(k, nw)
		}(k)
	}
	wg.Wait()
	return scoreScratch
}

// decodeSplitKEnabled gates the split-K (flash-decode) regime of attnDecodeStep. Off
// (FAK_DECODE_SPLITK=0) keeps every span on the exact per-group path, which is
// bit-identical to the serial reference loop — the parity reference for #13693.
var decodeSplitKEnabled = initDecodeSplitK()

func initDecodeSplitK() bool {
	switch os.Getenv("FAK_DECODE_SPLITK") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

// decodeSplitKMinPos is the visible-span floor at which attnDecodeStep switches from
// the exact per-group units to split-K chunks. Below it the per-head work is small,
// nKV-wide parallelism is enough, and the output stays bit-identical to the serial loop.
var decodeSplitKMinPos = envIntMin("FAK_DECODE_SPLITK_MINPOS", 1, 256)

// decodeSplitKMinChunk bounds how finely one head's span is cut: a chunk shorter than
// this spends more on its (max, sum, partial) merge record than on the attend itself.
var decodeSplitKMinChunk = envIntMin("FAK_DECODE_SPLITK_MINCHUNK", 8, 64)

// decodeSplitKOversub is units per worker: P and E cores finish chunks at different
// rates, so a few units per worker lets the parFor steal loop balance them.
const decodeSplitKOversub = 4

// decodeKVRows reads one layer's cached K/V rows for single-session decode attention
// in either representation. head returns kv-head kvh of row j: the f32 cache slice
// itself (zero copy), or on q8 that head's elements dequantized into dst.
type decodeKVRows struct {
	c      *KVCache
	l      int
	w, hd  int
	q8     bool
	kf, vf []float32
}

func newDecodeKVRows(c *KVCache, l, w, hd int) decodeKVRows {
	r := decodeKVRows{c: c, l: l, w: w, hd: hd, q8: c.quantized()}
	if !r.q8 {
		r.kf, r.vf = c.K[l], c.V[l]
	}
	return r
}

func (r decodeKVRows) head(j, kvh int, k bool, dst []float32) []float32 {
	if !r.q8 {
		src := r.vf
		if k {
			src = r.kf
		}
		off := j*r.w + kvh*r.hd
		return src[off : off+r.hd]
	}
	rows := &r.c.vQ8[r.l]
	if k {
		rows = &r.c.kQ8[r.l]
	}
	rows.decodeRangeInto(dst, j, kvh*r.hd, r.hd)
	return dst[:r.hd]
}

// decodeAttnStepArgs is the per-layer shape and score policy of one decode attend.
type decodeAttnStepArgs struct {
	l, lo, nPos, qpos int
	nH, hd, grp       int
	scale, softcap    float32
	// cfg supplies the ALiBi bias (a no-op unless cfg.Alibi); sinks is the layer's
	// per-head attention-sink logits, nil when the layer has none.
	cfg   *Config
	sinks []float32
	obs   AttnObserver
	nw    int
}

// splitKDecodeScratch is attnDecodeStep's reusable per-session scratch.
type splitKDecodeScratch struct {
	part    []float32 // [unit][g][hd] unnormalized partial outputs (split-K only)
	mx, sum []float32 // [unit][g] chunk max / exp-sum (split-K only)
	rows    []float32 // [unit][2*hd] q8 dequant rows (K then V)
}

// attnDecodeStep is blockStep's full-attention decode for ONE session (#13693): the
// query q [nH*hd] attends rows [lo, nPos) of layer l and the result is written into
// attnOut [nH*hd], which the caller has zeroed. scores is the caller's reusable score
// scratch; the grown slice is returned.
//
// Two regimes share one kernel:
//
//   - exact: one unit per kv head. Each unit reads (and on q8 dequantizes) a K/V row
//     once for its grp query heads and runs the serial reference arithmetic per head —
//     scalar dot, softcap, sink-aware softmax, saxpy in j order — so every output
//     element sees the identical operation sequence and the result is bit-identical
//     to the serial per-head loop, whatever the worker count.
//   - split-K: for a visible span >= decodeSplitKMinPos, each kv head's span is cut
//     into chunks and units are (kv head, chunk), so all cores take part even at
//     nKV=4. A unit records its chunk's (max, exp-sum, unnormalized partial) per query
//     head; a log-sum-exp merge then rescales and sums the partials. ALiBi and an
//     attention observer (which needs whole post-softmax rows) stay on the exact path.
func attnDecodeStep(attnOut, q, scores []float32, rows decodeKVRows, a decodeAttnStepArgs, sc *splitKDecodeScratch) []float32 {
	nKV := a.nH / a.grp
	span := a.nPos - a.lo
	if span <= 0 || nKV <= 0 {
		return scores
	}
	nw := a.nw
	if nw < 1 || a.obs != nil || a.nH*span*a.hd < parThreshold {
		// Observer callbacks are not required to be goroutine-safe; keep emission serial.
		// A tiny attend is cheaper inline than a parFor dispatch, as for the GEMV kernels.
		nw = 1
	}
	if decodeSplitKEnabled && nw > 1 && a.obs == nil && !a.cfg.Alibi && span >= decodeSplitKMinPos {
		chunksPerHead := (nw*decodeSplitKOversub + nKV - 1) / nKV
		chunk := (span + chunksPerHead - 1) / chunksPerHead
		if chunk < decodeSplitKMinChunk {
			chunk = decodeSplitKMinChunk
		}
		if nChunks := (span + chunk - 1) / chunk; nChunks > 1 {
			return attnDecodeSplitK(attnOut, q, scores, rows, a, sc, nw, chunk, nChunks)
		}
	}
	return attnDecodeExact(attnOut, q, scores, rows, a, sc, nw)
}

func attnDecodeExact(attnOut, q, scores []float32, rows decodeKVRows, a decodeAttnStepArgs, sc *splitKDecodeScratch, nw int) []float32 {
	nKV, grp, hd := a.nH/a.grp, a.grp, a.hd
	span := a.nPos - a.lo
	scores = grow(scores, nKV*grp*span)
	if rows.q8 {
		sc.rows = grow(sc.rows, nKV*2*hd)
	}
	unit := func(kvh int) {
		var kbuf, vbuf []float32
		if rows.q8 {
			kbuf = sc.rows[kvh*2*hd : kvh*2*hd+hd]
			vbuf = sc.rows[kvh*2*hd+hd : (kvh+1)*2*hd]
		}
		base := kvh * grp * span
		for j := a.lo; j < a.nPos; j++ {
			kh := rows.head(j, kvh, true, kbuf)
			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				scores[base+g*span+j-a.lo] = dot(q[h*hd:(h+1)*hd], kh)*a.scale + a.cfg.alibiScoreBias(h, j, a.nPos)
			}
		}
		for g := 0; g < grp; g++ {
			h := kvh*grp + g
			s := scores[base+g*span : base+(g+1)*span]
			softcapInPlace(s, a.softcap)
			softmaxWithSink(s, a.sinks, h)
			if a.obs != nil { // #852: emit the post-softmax row (copy-out, math untouched)
				emitAttnRow(a.obs, a.l, a.qpos, h, a.lo, s)
			}
		}
		for j := a.lo; j < a.nPos; j++ {
			vh := rows.head(j, kvh, false, vbuf)
			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				saxpy(attnOut[h*hd:(h+1)*hd], vh, scores[base+g*span+j-a.lo])
			}
		}
	}
	runDecodeUnits(nKV, nw, unit)
	return scores
}

func attnDecodeSplitK(attnOut, q, scores []float32, rows decodeKVRows, a decodeAttnStepArgs, sc *splitKDecodeScratch, nw, chunk, nChunks int) []float32 {
	nKV, grp, hd := a.nH/a.grp, a.grp, a.hd
	units := nKV * nChunks
	scores = grow(scores, units*grp*chunk)
	sc.part = grow(sc.part, units*grp*hd)
	sc.mx = grow(sc.mx, units*grp)
	sc.sum = grow(sc.sum, units*grp)
	if rows.q8 {
		sc.rows = grow(sc.rows, units*2*hd)
	}
	unit := func(u int) {
		kvh, c := u/nChunks, u%nChunks
		j0 := a.lo + c*chunk
		j1 := j0 + chunk
		if j1 > a.nPos {
			j1 = a.nPos
		}
		n := j1 - j0
		var kbuf, vbuf []float32
		if rows.q8 {
			kbuf = sc.rows[u*2*hd : u*2*hd+hd]
			vbuf = sc.rows[u*2*hd+hd : (u+1)*2*hd]
		}
		base := u * grp * chunk
		for j := j0; j < j1; j++ {
			kh := rows.head(j, kvh, true, kbuf)
			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				scores[base+g*chunk+j-j0] = fdot(q[h*hd:(h+1)*hd], kh) * a.scale
			}
		}
		for g := 0; g < grp; g++ {
			s := scores[base+g*chunk : base+g*chunk+n]
			softcapInPlace(s, a.softcap)
			mx := s[0]
			for _, v := range s {
				if v > mx {
					mx = v
				}
			}
			var sum float32
			for i, v := range s {
				e := float32(math.Exp(float64(v - mx)))
				s[i] = e
				sum += e
			}
			sc.mx[u*grp+g], sc.sum[u*grp+g] = mx, sum
			clear(sc.part[(u*grp+g)*hd : (u*grp+g+1)*hd])
		}
		for j := j0; j < j1; j++ {
			vh := rows.head(j, kvh, false, vbuf)
			for g := 0; g < grp; g++ {
				saxpy(sc.part[(u*grp+g)*hd:(u*grp+g+1)*hd], vh, scores[base+g*chunk+j-j0])
			}
		}
	}
	runDecodeUnits(units, nw, unit)

	// Log-sum-exp merge: softmax over the whole span is exp(s-M)/D with M the global
	// max and D = sum_c exp(m_c-M)*sum_c (+ exp(sink-M) for a sink layer), so each
	// chunk's unnormalized partial is scaled by exp(m_c-M)/D and summed.
	for h := 0; h < a.nH; h++ {
		kvh, g := h/grp, h%grp
		mxAll := float32(math.Inf(-1))
		sink, hasSink := sinkFor(a.sinks, h)
		if hasSink {
			mxAll = sink
		}
		for c := 0; c < nChunks; c++ {
			if m := sc.mx[(kvh*nChunks+c)*grp+g]; m > mxAll {
				mxAll = m
			}
		}
		var den float32
		if hasSink {
			den = float32(math.Exp(float64(sink - mxAll)))
		}
		for c := 0; c < nChunks; c++ {
			i := (kvh*nChunks+c)*grp + g
			den += float32(math.Exp(float64(sc.mx[i]-mxAll))) * sc.sum[i]
		}
		out := attnOut[h*hd : (h+1)*hd]
		for c := 0; c < nChunks; c++ {
			i := (kvh*nChunks+c)*grp + g
			wc := float32(math.Exp(float64(sc.mx[i]-mxAll))) / den
			saxpy(out, sc.part[i*hd:(i+1)*hd], wc)
		}
	}
	return scores
}

// runDecodeUnits runs unit(0..n-1) once each, across up to nw parFor workers.
func runDecodeUnits(n, nw int, unit func(int)) {
	body := func(lo, hi int) {
		for u := lo; u < hi; u++ {
			unit(u)
		}
	}
	if nw <= 1 || n <= 1 {
		body(0, n)
		return
	}
	parFor(n, nw, body)
}

func sinkFor(sinks []float32, h int) (float32, bool) {
	if h < 0 || h >= len(sinks) {
		return 0, false
	}
	return sinks[h], true
}

// softmaxWithSink is Model.softmaxAttentionScores with the layer's sink tensor
// resolved once: the identical softmaxInPlace / softmaxDropSinkInPlace per head.
func softmaxWithSink(s, sinks []float32, h int) {
	if sink, ok := sinkFor(sinks, h); ok {
		softmaxDropSinkInPlace(s, sink)
		return
	}
	softmaxInPlace(s)
}
