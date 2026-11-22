package ggufload

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TensorNVFP4 is ggml's NVFP4: 64-value blocks of four UE4M3 sub-block scales
// then 32 bytes of packed E2M1 values. Inkling-2-Mini GGUFs keep the released
// NVFP4 experts in it, with a per-expert F32 ".scale" tensor beside.
const TensorNVFP4 TensorType = 40

const (
	qkNVFP4         = 64
	qkNVFP4Sub      = 16
	blockNVFP4Bytes = qkNVFP4/qkNVFP4Sub + qkNVFP4/2
)

// kvaluesFP4 is ggml's E2M1 table doubled; ue4m3ToF32 halves the scale to match.
var kvaluesFP4 = [16]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

func ue4m3ToF32(x uint8) float32 {
	if x == 0 || x == 0x7F {
		return 0
	}
	exp := int(x>>3) & 0xF
	man := float64(x & 0x7)
	var raw float64
	if exp == 0 {
		raw = math.Ldexp(man, -9)
	} else {
		raw = math.Ldexp(1+man/8, exp-7)
	}
	return float32(raw * 0.5)
}

func dequantNVFP4(out []float32, raw []byte) {
	for b := 0; b < len(out)/qkNVFP4; b++ {
		blk := raw[b*blockNVFP4Bytes : (b+1)*blockNVFP4Bytes]
		qs := blk[qkNVFP4/qkNVFP4Sub:]
		for s := 0; s < qkNVFP4/qkNVFP4Sub; s++ {
			d := ue4m3ToF32(blk[s])
			y := out[b*qkNVFP4+s*qkNVFP4Sub:]
			for j := 0; j < qkNVFP4Sub/2; j++ {
				q := qs[s*(qkNVFP4Sub/2)+j]
				y[j] = kvaluesFP4[q&0x0F] * d
				y[j+qkNVFP4Sub/2] = kvaluesFP4[q>>4] * d
			}
		}
	}
}

func inkling2RowBytes(t TensorType, cols int) (int, error) {
	if t == TensorNVFP4 {
		if cols%qkNVFP4 != 0 {
			return 0, fmt.Errorf("inkling2_loop: NVFP4 row of %d values", cols)
		}
		return cols / qkNVFP4 * blockNVFP4Bytes, nil
	}
	tr, ok := tensorTypeTraits(t)
	if !ok || (tr.IsQuantized && !inkling2HostType(t)) {
		return 0, &UnsupportedInkling2LoopTensorTypeError{Type: t}
	}
	if uint64(cols)%tr.BlockSize != 0 {
		return 0, fmt.Errorf("inkling2_loop: %s row of %d values is not block aligned", t, cols)
	}
	return int(uint64(cols) / tr.BlockSize * tr.TypeSize), nil
}

// inkling2HostType lists the encodings of the Unsloth UD mainline GGUFs this
// host reference decodes; anything else refuses instead of guessing.
func inkling2HostType(t TensorType) bool {
	switch t {
	case TensorF32, TensorF16, TensorBF16, TensorQ8_0, TensorQ3_K, TensorQ4_K, TensorQ5_K, TensorQ6_K,
		TensorIQ2_XS, TensorIQ2_S, TensorIQ3_XXS, TensorIQ3_S, TensorIQ4_NL, TensorIQ4_XS, TensorNVFP4:
		return true
	}
	return false
}

type UnsupportedInkling2LoopTensorTypeError struct{ Type TensorType }

func (e *UnsupportedInkling2LoopTensorTypeError) Error() string {
	return fmt.Sprintf("inkling2_loop host reference: unsupported tensor type %s", e.Type)
}

func inkling2DequantRow(t TensorType, raw []byte, out []float32) error {
	if t == TensorNVFP4 {
		dequantNVFP4(out, raw)
		return nil
	}
	got, err := dequantF32IntoLimited(out, TensorInfo{Name: "inkling2-row", Dims: []uint64{uint64(len(out))}, Type: t}, raw, 1)
	if err != nil {
		return err
	}
	if &got[0] != &out[0] {
		copy(out, got)
	}
	return nil
}

// inkling2QuantMatrix keeps a GGUF matrix packed and dequantizes one row at a
// time inside the dot product.
type inkling2QuantMatrix struct {
	name       string
	typ        TensorType
	raw        []byte
	rows, cols int
	rowBytes   int
}

func (m *inkling2QuantMatrix) Rows() int { return m.rows }
func (m *inkling2QuantMatrix) Cols() int { return m.cols }
func (m *inkling2QuantMatrix) MatVec(dst, x []float32) {
	inkling2RowsMatVec(m.typ, m.raw, m.rows, m.cols, m.rowBytes, dst, x)
}

func inkling2RowsMatVec(typ TensorType, raw []byte, rows, cols, rowBytes int, dst, x []float32) {
	workers := runtime.GOMAXPROCS(0)
	if rows < 64 {
		workers = 1
	}
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < rows; start += chunk {
		end := min(start+chunk, rows)
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			row := make([]float32, cols)
			for r := a; r < b; r++ {
				if err := inkling2DequantRow(typ, raw[r*rowBytes:(r+1)*rowBytes], row); err != nil {
					panic(err)
				}
				var s float32
				for i, v := range row {
					s += v * x[i]
				}
				dst[r] = s
			}
		}(start, end)
	}
	wg.Wait()
}

type inkling2QuantExperts struct {
	typ                         TensorType
	raw                         []byte
	experts, rows, cols, rowLen int
}

func (e *inkling2QuantExperts) Experts() int { return e.experts }
func (e *inkling2QuantExperts) Rows() int    { return e.rows }
func (e *inkling2QuantExperts) Cols() int    { return e.cols }
func (e *inkling2QuantExperts) ExpertMatVec(expert int, dst, x []float32) {
	stride := e.rows * e.rowLen
	inkling2RowsMatVec(e.typ, e.raw[expert*stride:(expert+1)*stride], e.rows, e.cols, e.rowLen, dst, x)
}

type inkling2QuantEmbedding struct{ m *inkling2QuantMatrix }

func (e inkling2QuantEmbedding) Row(token int, dst []float32) {
	if err := inkling2DequantRow(e.m.typ, e.m.raw[token*e.m.rowBytes:(token+1)*e.m.rowBytes], dst); err != nil {
		panic(err)
	}
}

// Inkling2LoopModel is an inkling2_loop GGUF bound to the host reference.
type Inkling2LoopModel struct {
	HP      model.Inkling2LoopHParams
	Weights *model.Inkling2LoopWeights
	closer  io.Closer
}

func (m *Inkling2LoopModel) Close() error {
	if m.closer == nil {
		return nil
	}
	return m.closer.Close()
}

// OpenInkling2Loop maps an inkling2_loop GGUF (mainline llama.cpp layout).
func OpenInkling2Loop(path string) (*Inkling2LoopModel, error) {
	data, closer, _, err := model.MmapOpen(path)
	if err != nil {
		return nil, err
	}
	m, err := bindInkling2Loop(data)
	if err != nil {
		closer.Close()
		return nil, err
	}
	m.closer = closer
	return m, nil
}

type inkling2Binder struct {
	f       *File
	data    []byte
	tensors map[string]TensorInfo
	err     error
}

func bindInkling2Loop(data []byte) (*Inkling2LoopModel, error) {
	f, err := ReadSize(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	if a, _ := f.String("general.architecture"); a != "inkling2_loop" {
		return nil, fmt.Errorf("inkling2_loop: general.architecture %q", a)
	}
	b := &inkling2Binder{f: f, data: data, tensors: map[string]TensorInfo{}}
	for _, t := range f.Tensors {
		b.tensors[t.Name] = t
	}
	hp, err := b.hparams()
	if err != nil {
		return nil, err
	}
	w := &model.Inkling2LoopWeights{
		TokNorm:        b.vec("token_embd_norm.weight"),
		OutputNorm:     b.vec("output_norm.weight"),
		OutputResQuery: b.vec("output_res_query.weight"),
		Output:         b.mat("output.weight"),
	}
	if emb := b.mat("token_embd.weight"); emb != nil {
		w.TokEmbd = inkling2QuantEmbedding{emb.(*inkling2QuantMatrix)}
	}
	for il := 0; il < hp.Layers; il++ {
		p := fmt.Sprintf("blk.%d.", il)
		lw := model.Inkling2LoopLayerWeights{
			Wq: b.mat(p + "attn_q.weight"), Wk: b.mat(p + "attn_k.weight"), Wv: b.mat(p + "attn_v.weight"),
			Wo: b.mat(p + "attn_output.weight"), Gate: b.mat(p + "attn_gate.weight"), Rel: b.mat(p + "attn_rel.weight"),
			SinkProj: b.mat(p + "attn_sink_proj.weight"),
			QClip:    b.vec(p + "attn_q_clip.weight"), RelBias: b.vec(p + "attn_rel_bias.weight"), RelClip: b.vec(p + "attn_rel_clip.weight"),
			KConv: b.vec(p + "attn_k_conv.weight"), VConv: b.vec(p + "attn_v_conv.weight"),
			AttnPostConv: b.vec(p + "attn_post_conv.weight"), FFNPostConv: b.vec(p + "ffn_post_conv.weight"),
			FFNResQuery: b.vec(p + "ffn_res_query.weight"), FFNNorm: b.vec(p + "ffn_norm.weight"),
			ExpProbsBias: b.optVec(p + "exp_probs_b.bias"),
			Router:       b.mat(p + "ffn_gate_inp.weight"),
			GateUp:       b.experts(p + "ffn_gate_up_exps.weight"), Down: b.experts(p + "ffn_down_exps.weight"),
			GateUpScale: b.optVec(p + "ffn_gate_up_exps.scale"), DownScale: b.optVec(p + "ffn_down_exps.scale"),
		}
		if hp.LocalLayer[il] {
			lw.AttnNorm = b.vec(p + "attn_norm.weight")
		} else {
			lw.QInpNorm = b.vec(p + "attn_q_inp_norm.weight")
			lw.KInpNorm = b.vec(p + "attn_k_inp_norm.weight")
			lw.VInpNorm = b.vec(p + "attn_v_inp_norm.weight")
			lw.KConvProj = b.mat(p + "attn_k_conv_proj.weight")
			lw.VConvProj = b.mat(p + "attn_v_conv_proj.weight")
		}
		if il > 0 {
			lw.AttnResQuery = b.rowsOf(p+"attn_res_query.weight", hp.Hidden)
		}
		w.Layers = append(w.Layers, lw)
	}
	ctrlNames := [4]string{"q", "k", "v", "ffn"}
	for il := hp.CoreStart; il < hp.CoreEnd; il++ {
		p := fmt.Sprintf("blk.%d.", il)
		var cw model.Inkling2LoopControllerWeights
		msg := b.rowsOf(p+"loop_msg_query.weight", hp.Hidden)
		for c, n := range ctrlNames {
			if len(msg) == 4 {
				cw.MsgQuery[c] = msg[c]
			}
			cw.Down[c] = b.mat(p + "loop_" + n + "_ctrl_down.weight")
			cw.Gain[c] = b.mat(p + "loop_" + n + "_ctrl_gain.weight")
			cw.Forget[c] = b.mat(p + "loop_" + n + "_ctrl_forget.weight")
		}
		if b.err == nil && len(msg) != 4 {
			b.err = fmt.Errorf("inkling2_loop: %sloop_msg_query has %d rows, want 4", p, len(msg))
		}
		w.Controllers = append(w.Controllers, cw)
	}
	if b.err != nil {
		return nil, b.err
	}
	return &Inkling2LoopModel{HP: hp, Weights: w}, nil
}

func (b *inkling2Binder) hparams() (model.Inkling2LoopHParams, error) {
	const p = "inkling2_loop."
	var hp model.Inkling2LoopHParams
	u := func(key string, dst *int) {
		if v, ok := b.f.Uint64(p + key); ok {
			*dst = int(v)
		} else if b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: missing %s%s", p, key)
		}
	}
	fl := func(key string, dst *float32, required bool) {
		if v, ok := b.f.Float64(p + key); ok {
			*dst = float32(v)
		} else if required && b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: missing %s%s", p, key)
		}
	}
	u("embedding_length", &hp.Hidden)
	u("block_count", &hp.Layers)
	u("attention.key_length", &hp.HeadDim)
	u("vocab_size_unpadded", &hp.VocabUnpadded)
	u("attention.sliding_window", &hp.SlidingWindow)
	u("attention.relative_length", &hp.RelLength)
	u("attention.relative_extent", &hp.RelExtent)
	u("shortconv.l_cache", &hp.ConvKernel)
	u("attn_res.block_size", &hp.AttnResBlock)
	u("expert_count", &hp.Experts)
	u("expert_used_count", &hp.ExpertsUsed)
	u("expert_feed_forward_length", &hp.ExpertWidth)
	u("loop.start", &hp.CoreStart)
	u("loop.end", &hp.CoreEnd)
	u("loop.count", &hp.Passes)
	u("loop.gate_rank", &hp.GateRank)
	var floor int
	u("attention.log_scale_floor", &floor)
	fl("attention.layer_norm_rms_epsilon", &hp.NormEps, true)
	fl("attention.qk_clip", &hp.QKClip, true)
	fl("attention.relative_clip", &hp.RelClip, true)
	fl("attention.log_scale_alpha", &hp.LogScaleAlpha, true)
	fl("activation.situ_beta", &hp.SituBeta, true)
	fl("activation.situ_linear_beta", &hp.SituLinearBeta, true)
	fl("expert_weights_scale", &hp.ExpertWeightsScale, false)
	fl("final_logit_softcapping", &hp.FinalLogitSoftcap, false)
	hp.LogScaleFloor = float32(floor)
	heads, _ := b.f.IntArray(p + "attention.head_count")
	kv, _ := b.f.IntArray(p + "attention.head_count_kv")
	if len(heads) == 0 || len(kv) == 0 {
		if v, ok := b.f.Uint64(p + "attention.head_count"); ok {
			heads = []int{int(v)}
		}
		if v, ok := b.f.Uint64(p + "attention.head_count_kv"); ok {
			kv = []int{int(v)}
		}
	}
	for _, h := range heads {
		if h != heads[0] {
			return hp, fmt.Errorf("inkling2_loop: per-layer head counts %v are not uniform", heads)
		}
	}
	for _, h := range kv {
		if h != kv[0] {
			return hp, fmt.Errorf("inkling2_loop: per-layer kv head counts %v are not uniform", kv)
		}
	}
	if len(heads) > 0 && len(kv) > 0 {
		hp.Heads, hp.KVHeads = heads[0], kv[0]
	}
	hp.LocalLayer, _ = b.f.BoolArray(p + "attention.sliding_window_pattern")
	if b.err != nil {
		return hp, b.err
	}
	if len(hp.LocalLayer) != hp.Layers || hp.Heads == 0 || hp.KVHeads == 0 || hp.Heads%hp.KVHeads != 0 {
		return hp, fmt.Errorf("inkling2_loop: invalid heads %d/%d or %d-entry layer pattern for %d layers", hp.Heads, hp.KVHeads, len(hp.LocalLayer), hp.Layers)
	}
	return hp, nil
}

func (b *inkling2Binder) info(name string) (TensorInfo, []byte, bool) {
	t, ok := b.tensors[name]
	if !ok {
		if b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: missing tensor %s", name)
		}
		return t, nil, false
	}
	var cols, rest uint64 = t.Dims[0], 1
	for _, d := range t.Dims[1:] {
		rest *= d
	}
	rb, err := inkling2RowBytes(t.Type, int(cols))
	if err != nil {
		if b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: %s: %w", name, err)
		}
		return t, nil, false
	}
	size := int64(rb) * int64(rest)
	if t.FileOffset < 0 || t.FileOffset+size > int64(len(b.data)) {
		if b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: tensor %s runs past the file", name)
		}
		return t, nil, false
	}
	return t, b.data[t.FileOffset : t.FileOffset+size], true
}

func (b *inkling2Binder) vec(name string) []float32 {
	t, raw, ok := b.info(name)
	if !ok {
		return nil
	}
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	cols := int(t.Dims[0])
	rb, _ := inkling2RowBytes(t.Type, cols)
	out := make([]float32, n)
	for r := 0; r < n/cols; r++ {
		if err := inkling2DequantRow(t.Type, raw[r*rb:(r+1)*rb], out[r*cols:(r+1)*cols]); err != nil && b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: %s: %w", name, err)
		}
	}
	return out
}

func (b *inkling2Binder) optVec(name string) []float32 {
	if _, ok := b.tensors[name]; !ok {
		return nil
	}
	return b.vec(name)
}

func (b *inkling2Binder) rowsOf(name string, width int) [][]float32 {
	v := b.vec(name)
	var rows [][]float32
	for i := 0; i+width <= len(v); i += width {
		rows = append(rows, v[i:i+width])
	}
	return rows
}

func (b *inkling2Binder) mat(name string) model.Inkling2LoopMatrix {
	t, raw, ok := b.info(name)
	if !ok {
		return nil
	}
	if len(t.Dims) != 2 {
		if b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: %s has %d dims, want a matrix", name, len(t.Dims))
		}
		return nil
	}
	cols, rows := int(t.Dims[0]), int(t.Dims[1])
	rb, _ := inkling2RowBytes(t.Type, cols)
	return &inkling2QuantMatrix{name: name, typ: t.Type, raw: raw, rows: rows, cols: cols, rowBytes: rb}
}

func (b *inkling2Binder) experts(name string) model.Inkling2LoopExperts {
	t, raw, ok := b.info(name)
	if !ok {
		return nil
	}
	if len(t.Dims) != 3 {
		if b.err == nil {
			b.err = fmt.Errorf("inkling2_loop: %s has %d dims, want experts", name, len(t.Dims))
		}
		return nil
	}
	cols := int(t.Dims[0])
	rb, _ := inkling2RowBytes(t.Type, cols)
	return &inkling2QuantExperts{typ: t.Type, raw: raw, experts: int(t.Dims[2]), rows: int(t.Dims[1]), cols: cols, rowLen: rb}
}

// Inkling2LoopGreedy runs the host reference over prompt token IDs and returns
// n greedily decoded token IDs.
func Inkling2LoopGreedy(m *Inkling2LoopModel, prompt []int, n int) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("inkling2_loop: empty prompt")
	}
	h, err := model.NewInkling2LoopHost(m.HP, m.Weights)
	if err != nil {
		return nil, err
	}
	var logits []float32
	for _, t := range prompt {
		logits = h.Step(t)
	}
	out := make([]int, 0, n)
	for len(out) < n {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		out = append(out, best)
		if len(out) == n {
			break
		}
		logits = h.Step(best)
	}
	return out, nil
}
