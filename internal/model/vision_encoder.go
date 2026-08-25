package model

// vision_encoder.go — the CLIP/ViT image-tower forward (#4030).
//
// #4029 gave the vision weights a resident home (Model.Vision, a VisionTower) and a
// name resolver (ResolveVisionTensorNames) that proves a manifest carries the canonical
// v.*/mm.* set. This file is the forward that reads them: patch-embed -> N bidirectional
// ViT blocks -> optional post-LN -> spatial patch-merge -> the mm.* projector, producing
// one decoder-hidden-width vector per merged patch. Those vectors are exactly what
// VisionEmbedding.Vectors carries and ForwardMultimodal splices into the prompt sequence
// (multimodal.go), so the image tower and the text decoder meet only at hidden-size rows.
//
// Architecture is bound to #4029's visionSpec CONTRACT, not to any one upstream release:
// the substrate models LayerNorm (ln1/ln2, with bias), a 2-matrix MLP (ffn_up -> GELU ->
// ffn_down, HF mlp.fc1/fc2), split q/k/v/out attention, an OPTIONAL additive position
// embedding, and an mm.0 -> GELU -> mm.2 projector (Qwen2-VL's PatchMerger). The forward
// implements precisely that. Two geometry facts that upstreams vary (the patch-embed
// in-features — still-image 3*P^2 vs Qwen2-VL temporal-doubled 6*P^2 — and the projector
// inner/out dims) are DERIVED FROM THE TENSORS themselves rather than hardcoded, so the
// encoder conforms to whatever the loaded tower actually carries.
//
// Witness boundary (honest): the numeric primitives (layernorm, matRows, geluErf,
// softmaxInPlace) are the same the decoder is oracle-checked on, and the shapes/grid
// math/byte-accounting are unit-tested here. What is NOT witnessed in-tree is bitwise
// parity against a real llama.cpp CLIP mmproj — that needs a cached Qwen2-VL mmproj GGUF
// fixture (asset-gated, like the rest of the vision epic). The activation choice
// (GELU-erf) and LN epsilon default are the modeled Qwen2-VL values, isolated behind
// vgelu/visionLNEps so a real fixture can pin them with a one-line change.

import (
	"fmt"
	"math"
)

// visionChannels is the RGB channel count the patch-embed unfolds; alpha is dropped.
const visionChannels = 3

// defaultVisionLNEps is the ViT LayerNorm epsilon when the tower does not carry one
// (Qwen2-VL's vision tower uses 1e-6, distinct from the decoder's 1e-5/1e-6).
const defaultVisionLNEps = 1e-6

// vgelu is the ViT MLP + projector activation. The modeled Qwen2-VL vision tower uses
// GELU; geluErf is the exact form. Isolated here so a real-fixture parity check can swap
// it (e.g. to QuickGELU) in one place without touching the forward.
func vgelu(x float32) float32 { return geluErf(x) }

// visionEncoder is a resolved, validated view over a VisionTower ready to run the ViT
// forward. It holds the canonical->source name map (resolved once at construction) and
// the derived geometry the forward reuses per call.
type visionEncoder struct {
	tower         *VisionTower
	res           *Resolution
	decoderHidden int // projector output width must equal this (the splice contract)

	// derived geometry, validated once in newVisionEncoder
	hidden   int     // ViT hidden width
	layers   int     // ViT block count
	heads    int     // attention heads
	headDim  int     // hidden/heads
	patch    int     // conv patch edge
	merge    int     // spatial merge factor (>=1)
	temporal int     // patch-embed temporal replication (1 still-image, 2 Qwen2-VL)
	patchIn  int     // patch-embed in-features = channels*temporal*patch*patch
	ffn      int     // MLP inner width
	projMid  int     // projector hidden width (mm.0 out)
	projOut  int     // projector output width (== decoderHidden)
	lnEps    float32 // ViT LayerNorm epsilon
}

// NewVisionEncoder builds the image tower's forward over a model's retained vision
// weights. It errors (rather than returning a nil encoder) when the model carries no
// vision tower, when a required canonical tensor is missing, or when the tower geometry
// is internally inconsistent — so a caller never runs a half-resolved tower.
func (m *Model) NewVisionEncoder() (VisionEncoder, error) {
	if m.Vision == nil {
		return nil, fmt.Errorf("model: no vision tower retained (load with RetainVision / --mmproj)")
	}
	return newVisionEncoder(m.Vision, m.Cfg.HiddenSize)
}

// newVisionEncoder resolves and validates a tower against decoderHidden. Split from
// NewVisionEncoder so tests can drive a hand-built VisionTower without a full Model.
func newVisionEncoder(tower *VisionTower, decoderHidden int) (*visionEncoder, error) {
	if tower == nil {
		return nil, fmt.Errorf("model: vision encoder: nil tower")
	}
	cfg := tower.Cfg
	if decoderHidden <= 0 {
		return nil, fmt.Errorf("model: vision encoder: decoder hidden size %d is not positive", decoderHidden)
	}
	if cfg.HiddenSize <= 0 || cfg.NumLayers <= 0 || cfg.NumHeads <= 0 || cfg.PatchSize <= 0 {
		return nil, fmt.Errorf("model: vision encoder: incomplete geometry (hidden=%d layers=%d heads=%d patch=%d); pin it from the mmproj clip.* metadata",
			cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.PatchSize)
	}
	if cfg.HiddenSize%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("model: vision encoder: hidden %d not divisible by heads %d", cfg.HiddenSize, cfg.NumHeads)
	}
	res, err := ResolveVisionTensorNames(cfg, tower.manifest)
	if err != nil {
		return nil, err
	}
	e := &visionEncoder{
		tower:         tower,
		res:           res,
		decoderHidden: decoderHidden,
		hidden:        cfg.HiddenSize,
		layers:        cfg.NumLayers,
		heads:         cfg.NumHeads,
		headDim:       cfg.HiddenSize / cfg.NumHeads,
		patch:         cfg.PatchSize,
		merge:         cfg.MergeSize,
		lnEps:         cfg.LNEps,
	}
	if e.merge < 1 {
		e.merge = 1
	}
	if e.lnEps <= 0 {
		e.lnEps = defaultVisionLNEps
	}

	// patch-embed in-features -> temporal factor. Weight is [hidden, channels*temporal*P*P].
	patchW := e.w("v.patch_embd.weight")
	if len(patchW)%e.hidden != 0 {
		return nil, fmt.Errorf("model: vision encoder: patch_embd.weight len %d not divisible by hidden %d", len(patchW), e.hidden)
	}
	e.patchIn = len(patchW) / e.hidden
	perFrame := visionChannels * e.patch * e.patch
	if perFrame == 0 || e.patchIn%perFrame != 0 {
		return nil, fmt.Errorf("model: vision encoder: patch_embd in-features %d not a multiple of channels*patch^2 %d", e.patchIn, perFrame)
	}
	e.temporal = e.patchIn / perFrame
	if e.temporal < 1 {
		return nil, fmt.Errorf("model: vision encoder: derived temporal factor %d < 1", e.temporal)
	}

	// MLP inner width from ffn_up (block 0 is representative; the resolver proved every layer).
	ffnUp := e.w("v.blk.0.ffn_up.weight")
	if len(ffnUp)%e.hidden != 0 {
		return nil, fmt.Errorf("model: vision encoder: ffn_up.weight len %d not divisible by hidden %d", len(ffnUp), e.hidden)
	}
	e.ffn = len(ffnUp) / e.hidden

	// projector dims: mm.0 is [projMid, hidden*merge^2]; optional mm.2 is [projOut, projMid].
	mm0 := e.w("mm.0.weight")
	inMerge := e.hidden * e.merge * e.merge
	if inMerge == 0 || len(mm0)%inMerge != 0 {
		return nil, fmt.Errorf("model: vision encoder: mm.0.weight len %d not divisible by hidden*merge^2 %d", len(mm0), inMerge)
	}
	e.projMid = len(mm0) / inMerge
	if mm2 := e.wOpt("mm.2.weight"); mm2 != nil {
		if e.projMid == 0 || len(mm2)%e.projMid != 0 {
			return nil, fmt.Errorf("model: vision encoder: mm.2.weight len %d not divisible by projector mid %d", len(mm2), e.projMid)
		}
		e.projOut = len(mm2) / e.projMid
	} else {
		e.projOut = e.projMid
	}
	if e.projOut != e.decoderHidden {
		return nil, fmt.Errorf("model: vision encoder: projector output width %d != decoder hidden %d (image vectors would not splice)", e.projOut, e.decoderHidden)
	}
	return e, nil
}

// w reads a required canonical tensor via the resolved source name.
func (e *visionEncoder) w(canonical string) []float32 {
	src, ok := e.res.Resolved[canonical]
	if !ok {
		panic("model: vision encoder: unresolved required tensor " + canonical)
	}
	return e.tower.tensor(src)
}

// wOpt reads an optional canonical tensor, returning nil when the tower omits it.
func (e *visionEncoder) wOpt(canonical string) []float32 {
	src, ok := e.res.Resolved[canonical]
	if !ok {
		return nil
	}
	return e.tower.tensor(src)
}

// GridForImage returns the patch grid (rows, cols) a width x height image reduces to,
// AFTER spatial merge — i.e. the token count is rows*cols. Dimensions must already be a
// positive multiple of patch*merge; ok is false otherwise (the caller preprocesses).
func (e *visionEncoder) GridForImage(w, h int) (rows, cols int, ok bool) {
	unit := e.patch * e.merge
	if w <= 0 || h <= 0 || w%unit != 0 || h%unit != 0 {
		return 0, 0, false
	}
	return h / unit, w / unit, true
}

// NumImageTokens is the number of hidden-size vectors EncodeImage yields for a width x
// height image (rows*cols merged patches), or 0 when the dimensions are not a clean
// multiple of patch*merge. This is what the serve seam (#4032) reconciles against the
// count of <|image_pad|> placeholders.
func (e *visionEncoder) NumImageTokens(w, h int) int {
	rows, cols, ok := e.GridForImage(w, h)
	if !ok {
		return 0
	}
	return rows * cols
}

// forwardPixels runs the ViT tower over a normalized pixel plane and returns one
// decoder-hidden-width vector per merged patch. px is CHW f32 (channel-major, then row,
// then col) of shape [visionChannels, gridRows*patch, gridCols*patch]. It is the
// witnessed core: EncodeImage is a thin decode/preprocess wrapper around it.
func (e *visionEncoder) forwardPixels(px []float32, gridRows, gridCols int) ([][]float32, error) {
	if gridRows <= 0 || gridCols <= 0 {
		return nil, fmt.Errorf("model: vision encoder: grid %dx%d must be positive", gridRows, gridCols)
	}
	imgH := gridRows * e.patch
	imgW := gridCols * e.patch
	if want := visionChannels * imgH * imgW; len(px) != want {
		return nil, fmt.Errorf("model: vision encoder: pixel plane len %d != channels*H*W %d", len(px), want)
	}
	if gridRows%e.merge != 0 || gridCols%e.merge != 0 {
		return nil, fmt.Errorf("model: vision encoder: grid %dx%d not divisible by merge %d", gridRows, gridCols, e.merge)
	}
	seq := gridRows * gridCols

	// ---- patch embed: each patch -> one hidden vector -------------------------------
	patchW := e.w("v.patch_embd.weight")
	patchB := e.wOpt("v.patch_embd.bias")
	x := make([][]float32, seq)
	unfold := make([]float32, e.patchIn)
	for pr := 0; pr < gridRows; pr++ {
		for pc := 0; pc < gridCols; pc++ {
			e.unfoldPatch(px, imgH, imgW, pr, pc, unfold)
			row := matRows(patchW, unfold, e.hidden, e.patchIn)
			addBiasVec(row, patchB)
			x[pr*gridCols+pc] = row
		}
	}

	// optional additive position embedding ([numPos, hidden]); added per patch in order.
	if pos := e.wOpt("v.position_embd.weight"); pos != nil {
		numPos := len(pos) / e.hidden
		for t := 0; t < seq && t < numPos; t++ {
			addVec(x[t], pos[t*e.hidden:(t+1)*e.hidden])
		}
	}

	// ---- bidirectional ViT blocks ---------------------------------------------------
	for l := 0; l < e.layers; l++ {
		e.block(l, x)
	}

	// optional post layernorm applied to every token.
	if pw := e.wOpt("v.post_ln.weight"); pw != nil {
		pb := e.wOpt("v.post_ln.bias")
		for t := 0; t < seq; t++ {
			x[t] = layernorm(x[t], pw, pb, e.lnEps)
		}
	}

	// ---- spatial merge + projector --------------------------------------------------
	return e.mergeAndProject(x, gridRows, gridCols), nil
}

// unfoldPatch fills dst (len patchIn) with the pixels of patch (pr,pc) in the modeled
// (channel, temporal, row, col) nesting; a still image replicates the single frame
// across the temporal axis so a temporal-doubled patch_embd (Qwen2-VL) sees each frame.
func (e *visionEncoder) unfoldPatch(px []float32, imgH, imgW, pr, pc int, dst []float32) {
	plane := imgH * imgW
	idx := 0
	for c := 0; c < visionChannels; c++ {
		base := c * plane
		for tf := 0; tf < e.temporal; tf++ {
			for i := 0; i < e.patch; i++ {
				rowBase := base + (pr*e.patch+i)*imgW + pc*e.patch
				for j := 0; j < e.patch; j++ {
					dst[idx] = px[rowBase+j]
					idx++
				}
			}
		}
	}
}

// block runs one bidirectional pre-norm ViT block in place on x.
func (e *visionEncoder) block(l int, x [][]float32) {
	p := func(s string) string { return "v.blk." + itoa(l) + "." + s }
	ln1w, ln1b := e.w(p("ln1.weight")), e.wOpt(p("ln1.bias"))
	ln2w, ln2b := e.w(p("ln2.weight")), e.wOpt(p("ln2.bias"))
	qw, qb := e.w(p("attn_q.weight")), e.wOpt(p("attn_q.bias"))
	kw, kb := e.w(p("attn_k.weight")), e.wOpt(p("attn_k.bias"))
	vw, vb := e.w(p("attn_v.weight")), e.wOpt(p("attn_v.bias"))
	ow, ob := e.w(p("attn_out.weight")), e.wOpt(p("attn_out.bias"))
	upW, upB := e.w(p("ffn_up.weight")), e.wOpt(p("ffn_up.bias"))
	dnW, dnB := e.w(p("ffn_down.weight")), e.wOpt(p("ffn_down.bias"))

	seq := len(x)
	H := e.hidden

	// attention: project all tokens after ln1, then full (bidirectional) SDPA.
	q := make([][]float32, seq)
	k := make([][]float32, seq)
	v := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xn := layernorm(x[t], ln1w, ln1b, e.lnEps)
		q[t] = matRows(qw, xn, H, H)
		addBiasVec(q[t], qb)
		k[t] = matRows(kw, xn, H, H)
		addBiasVec(k[t], kb)
		v[t] = matRows(vw, xn, H, H)
		addBiasVec(v[t], vb)
	}
	attn := e.attention(q, k, v)
	for t := 0; t < seq; t++ {
		o := matRows(ow, attn[t], H, H)
		addBiasVec(o, ob)
		addVec(x[t], o) // residual
	}

	// MLP: ln2 -> ffn_up -> GELU -> ffn_down, residual.
	for t := 0; t < seq; t++ {
		xn := layernorm(x[t], ln2w, ln2b, e.lnEps)
		up := matRows(upW, xn, e.ffn, H)
		addBiasVec(up, upB)
		for i := range up {
			up[i] = vgelu(up[i])
		}
		dn := matRows(dnW, up, H, e.ffn)
		addBiasVec(dn, dnB)
		addVec(x[t], dn) // residual
	}
}

// attention computes full bidirectional multi-head SDPA. q/k/v are [seq][hidden]; the
// return is [seq][hidden] (heads concatenated). Scale is 1/sqrt(headDim).
func (e *visionEncoder) attention(q, k, v [][]float32) [][]float32 {
	seq := len(q)
	H, hd, nH := e.hidden, e.headDim, e.heads
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		out[t] = make([]float32, H)
	}
	scores := make([]float32, seq)
	for h := 0; h < nH; h++ {
		off := h * hd
		for t := 0; t < seq; t++ {
			qt := q[t][off : off+hd]
			for j := 0; j < seq; j++ {
				scores[j] = dot(qt, k[j][off:off+hd]) * scale
			}
			softmaxInPlace(scores)
			acc := out[t][off : off+hd]
			for j := 0; j < seq; j++ {
				w := scores[j]
				vj := v[j][off : off+hd]
				saxpy(acc, vj, w)
			}
		}
	}
	return out
}

// mergeAndProject folds each merge x merge block of patches into one vector of length
// hidden*merge^2 (row-major sub-order), then runs the mm.0 -> GELU -> mm.2 projector,
// yielding one decoder-hidden-width vector per merged block in row-major order.
func (e *visionEncoder) mergeAndProject(x [][]float32, gridRows, gridCols int) [][]float32 {
	m := e.merge
	mRows, mCols := gridRows/m, gridCols/m
	H := e.hidden
	inMerge := H * m * m
	mm0w, mm0b := e.w("mm.0.weight"), e.wOpt("mm.0.bias")
	mm2w, mm2b := e.wOpt("mm.2.weight"), e.wOpt("mm.2.bias")

	out := make([][]float32, 0, mRows*mCols)
	merged := make([]float32, inMerge)
	for br := 0; br < mRows; br++ {
		for bc := 0; bc < mCols; bc++ {
			idx := 0
			for di := 0; di < m; di++ {
				for dj := 0; dj < m; dj++ {
					patch := (br*m+di)*gridCols + (bc*m + dj)
					copy(merged[idx:idx+H], x[patch])
					idx += H
				}
			}
			h0 := matRows(mm0w, merged, e.projMid, inMerge)
			addBiasVec(h0, mm0b)
			if mm2w == nil {
				out = append(out, h0)
				continue
			}
			for i := range h0 {
				h0[i] = vgelu(h0[i])
			}
			o := matRows(mm2w, h0, e.projOut, e.projMid)
			addBiasVec(o, mm2b)
			out = append(out, o)
		}
	}
	return out
}

// addBiasVec adds bias into v in place when bias is present (nil bias is a no-op, so an
// optional projector/attention bias is handled uniformly).
func addBiasVec(v, bias []float32) {
	if bias == nil {
		return
	}
	for i := range v {
		v[i] += bias[i]
	}
}

// addVec adds src into dst in place (residual / position-embed add).
func addVec(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
