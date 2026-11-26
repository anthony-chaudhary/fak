package ggufload

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

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
	default:
		// Every other encoding is refused by the caller rather than decoded.
		return false
	}
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

const (
	inkling2ArchLoop = "inkling2_loop"
	inkling2ArchBase = "inkling2"

	// Published Inkling-2-Small GGUFs carry a stale log_scale_floor; the
	// checkpoint was trained with 8192.
	inkling2StaleLogScaleFloor   = 128000
	inkling2CorrectLogScaleFloor = 8192
)

// Inkling2LoopModel is an inkling2 or inkling2_loop GGUF bound to the host
// reference. An inkling2 model is the same graph with an empty loop core.
type Inkling2LoopModel struct {
	HP      model.Inkling2LoopHParams
	Weights *model.Inkling2LoopWeights
	// Arch is the GGUF general.architecture that was bound.
	Arch string
	// LogScaleFloorOverridden reports that the stale published floor was
	// replaced with the correct value.
	LogScaleFloorOverridden bool
	closer                  io.Closer
	// path is the GGUF the model was opened from, where adjacent drafts are looked up
	path string
	// readahead is the map-time page-cache reader awaiting the device bind; the
	// closer chain stops it before the unmap.
	readahead *inkling2UploadReadahead
	// SpecDefault is the greedy speculation run while FAK_INKLING2_SPEC is unset.
	SpecDefault Inkling2SpecDefault
}

func (m *Inkling2LoopModel) Close() error {
	if m.closer == nil {
		return nil
	}
	return m.closer.Close()
}

// Inkling2NonUniformHeadsError refuses per-layer head counts that differ: the
// host reference has a single head geometry.
type Inkling2NonUniformHeadsError struct {
	Key    string
	Counts []int
}

func (e *Inkling2NonUniformHeadsError) Error() string {
	return fmt.Sprintf("inkling2: per-layer %s %v is not uniform", e.Key, e.Counts)
}

type inkling2Closers []io.Closer

func (c inkling2Closers) Close() error {
	var first error
	for _, x := range c {
		if err := x.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// OpenInkling2Loop maps an inkling2 or inkling2_loop GGUF (mainline llama.cpp
// layout). A split checkpoint may be opened through any of its shards.
func OpenInkling2Loop(path string) (*Inkling2LoopModel, error) {
	return openInkling2Loop(path, nil)
}

// openInkling2Loop is OpenInkling2Loop with a hook that runs once the shards
// are mapped, before they are bound; f is nil for a split checkpoint. The
// closer it returns, if any, runs before the unmap.
func openInkling2Loop(path string, mapped func(shards []inkling2MappedShard, f *File) io.Closer) (*Inkling2LoopModel, error) {
	data, closer, err := inkling2MapFile(path)
	if err != nil {
		return nil, err
	}
	closers := inkling2Closers{closer}
	f, err := ReadSize(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		closers.Close()
		return nil, err
	}
	shards := [][]byte{data}
	paths := []string{path}
	if count, ok := f.Uint64("split.count"); ok && count > 1 {
		closers.Close()
		f = nil
		shards, closers, paths, err = inkling2MapShards(path, int(count))
		if err != nil {
			return nil, err
		}
	}
	if mapped != nil {
		mappedShards := make([]inkling2MappedShard, len(shards))
		for i := range shards {
			mappedShards[i] = inkling2MappedShard{data: shards[i], path: paths[i]}
		}
		if c := mapped(mappedShards, f); c != nil {
			closers = append(inkling2Closers{c}, closers...)
		}
	}
	m, err := bindInkling2Shards(shards, nil)
	if err != nil {
		closers.Close()
		return nil, err
	}
	m.closer, m.path = closers, path
	return m, nil
}

type inkling2PackedWeight interface {
	model.Inkling2LoopMatrix
	model.Inkling2LoopExperts
}

// inkling2Pack rebinds a quantized matrix or expert stack onto an accelerator.
type inkling2Pack func(t TensorInfo) (inkling2PackedWeight, error)

func inkling2MapFile(path string) ([]byte, io.Closer, error) {
	data, closer, ok, err := model.MmapOpen(path)
	if err != nil {
		return nil, nil, err
	}
	if !ok {
		if data, err = os.ReadFile(path); err != nil {
			return nil, nil, err
		}
		closer = io.NopCloser(nil)
	}
	return data, closer, nil
}

// inkling2MapShards maps every shard of a -N-of-M.gguf set in order.
func inkling2MapShards(path string, count int) ([][]byte, inkling2Closers, []string, error) {
	named, ok := shardPathCount(path)
	if !ok {
		return nil, nil, nil, fmt.Errorf("inkling2: %s declares split.count=%d but its name is not a -N-of-M.gguf shard path", path, count)
	}
	if named != count {
		return nil, nil, nil, fmt.Errorf("inkling2: %s declares split.count=%d, but its filename names a %d-shard set", path, count, named)
	}
	first, err := firstShardPath(path)
	if err != nil {
		return nil, nil, nil, err
	}
	paths, err := shardPaths(first, count)
	if err != nil {
		return nil, nil, nil, err
	}
	var closers inkling2Closers
	shards := make([][]byte, 0, count)
	for _, p := range paths {
		data, c, err := inkling2MapFile(p)
		if err != nil {
			closers.Close()
			return nil, nil, nil, fmt.Errorf("inkling2: open shard %s: %w", p, err)
		}
		closers = append(closers, c)
		shards = append(shards, data)
	}
	return shards, closers, paths, nil
}

type inkling2Binder struct {
	pack    inkling2Pack
	f       *File
	arch    string
	prefix  string
	shards  [][]byte
	tensors map[string]TensorInfo
	shardOf map[string]int
	err     error
}

// A pack resolves tensors by FileOffset, so it only binds a single-file GGUF.
func bindInkling2LoopWith(data []byte, pack inkling2Pack) (*Inkling2LoopModel, error) {
	return bindInkling2Shards([][]byte{data}, pack)
}

// bindInkling2Shards binds shards in filename order. Shard 1 must be the config
// carrier (the one holding general.architecture); split.no is checked per shard
// so a mislabelled set cannot bind tensors under the wrong config.
func bindInkling2Shards(shards [][]byte, pack inkling2Pack) (*Inkling2LoopModel, error) {
	if pack != nil && len(shards) != 1 {
		return nil, fmt.Errorf("inkling2: packed bind needs a single-file GGUF, have %d shards", len(shards))
	}
	b := &inkling2Binder{pack: pack, shards: shards, tensors: map[string]TensorInfo{}, shardOf: map[string]int{}}
	for i, data := range shards {
		f, err := ReadSize(bytes.NewReader(data), int64(len(data)))
		if err != nil {
			return nil, fmt.Errorf("inkling2: shard %d: %w", i+1, err)
		}
		if i == 0 {
			b.f = f
		}
		if c, ok := f.Uint64("split.count"); ok && int(c) != len(shards) && (c > 1 || len(shards) > 1) {
			return nil, fmt.Errorf("inkling2: shard %d declares split.count=%d, have %d shards", i+1, c, len(shards))
		}
		if len(shards) > 1 {
			no, present := f.Uint64("split.no")
			if !validShardNo(no, present, i+1) {
				return nil, fmt.Errorf("inkling2: shard %d of %d declares split.no=%d", i+1, len(shards), no)
			}
		}
		for _, t := range f.Tensors {
			if _, dup := b.tensors[t.Name]; dup {
				return nil, fmt.Errorf("inkling2: duplicate tensor %s in shard %d", t.Name, i+1)
			}
			b.tensors[t.Name] = t
			b.shardOf[t.Name] = i
		}
	}
	arch, hasArch := b.f.String("general.architecture")
	if !hasArch && len(shards) > 1 {
		return nil, fmt.Errorf("inkling2: split shard 1 carries no config (general.architecture missing)")
	}
	if arch != inkling2ArchLoop && arch != inkling2ArchBase {
		return nil, fmt.Errorf("inkling2: unsupported general.architecture %q", arch)
	}
	b.arch = arch
	b.prefix = arch + "."
	hp, overridden, err := b.hparams(arch)
	if err != nil {
		return nil, err
	}
	w := &model.Inkling2LoopWeights{
		TokNorm:        b.vec("token_embd_norm.weight"),
		OutputNorm:     b.vec("output_norm.weight"),
		OutputResQuery: b.vec("output_res_query.weight"),
		Output:         b.mat("output.weight"),
	}
	if emb := b.matrix("token_embd.weight", false); emb != nil {
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
			b.err = fmt.Errorf("%s: %sloop_msg_query has %d rows, want 4", b.arch, p, len(msg))
		}
		w.Controllers = append(w.Controllers, cw)
	}
	if b.err != nil {
		return nil, b.err
	}
	return &Inkling2LoopModel{HP: hp, Weights: w, Arch: arch, LogScaleFloorOverridden: overridden}, nil
}

func (b *inkling2Binder) hparams(arch string) (model.Inkling2LoopHParams, bool, error) {
	hp, err := b.readHParams(arch)
	if err != nil {
		return hp, false, err
	}
	if arch == inkling2ArchBase && hp.LogScaleFloor == inkling2StaleLogScaleFloor {
		hp.LogScaleFloor = inkling2CorrectLogScaleFloor
		return hp, true, nil
	}
	return hp, false, nil
}

func (b *inkling2Binder) readHParams(arch string) (model.Inkling2LoopHParams, error) {
	p := b.prefix
	var hp model.Inkling2LoopHParams
	u := func(key string, dst *int) {
		if v, ok := b.f.Uint64(p + key); ok {
			*dst = int(v)
		} else if b.err == nil {
			b.err = fmt.Errorf("%s: missing %s%s", arch, p, key)
		}
	}
	fl := func(key string, dst *float32, required bool) {
		if v, ok := b.f.Float64(p + key); ok {
			*dst = float32(v)
		} else if required && b.err == nil {
			b.err = fmt.Errorf("%s: missing %s%s", arch, p, key)
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
	if arch == inkling2ArchLoop {
		u("loop.start", &hp.CoreStart)
		u("loop.end", &hp.CoreEnd)
		u("loop.count", &hp.Passes)
		u("loop.gate_rank", &hp.GateRank)
	} else {
		// inkling2 has no loop core; stray loop.* keys must not add passes.
		hp.Passes = 1
	}
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
	heads := b.headCounts(p + "attention.head_count")
	kv := b.headCounts(p + "attention.head_count_kv")
	for _, h := range heads {
		if h != heads[0] {
			return hp, &Inkling2NonUniformHeadsError{Key: "attention.head_count", Counts: heads}
		}
	}
	for _, h := range kv {
		if h != kv[0] {
			return hp, &Inkling2NonUniformHeadsError{Key: "attention.head_count_kv", Counts: kv}
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
		return hp, fmt.Errorf("%s: invalid heads %d/%d or %d-entry layer pattern for %d layers", arch, hp.Heads, hp.KVHeads, len(hp.LocalLayer), hp.Layers)
	}
	if hp.CoreStart < 0 || hp.CoreEnd < hp.CoreStart || hp.CoreEnd > hp.Layers || hp.Passes < 1 {
		return hp, fmt.Errorf("%s: invalid loop core [%d,%d) x%d for %d layers", arch, hp.CoreStart, hp.CoreEnd, hp.Passes, hp.Layers)
	}
	return hp, nil
}

// headCounts reads a head count stored either as a scalar or as a per-layer array.
func (b *inkling2Binder) headCounts(key string) []int {
	if v, ok := b.f.IntArray(key); ok && len(v) > 0 {
		return v
	}
	if v, ok := b.f.Uint64(key); ok {
		return []int{int(v)}
	}
	return nil
}

func (b *inkling2Binder) info(name string) (TensorInfo, []byte, bool) {
	t, ok := b.tensors[name]
	if !ok {
		if b.err == nil {
			b.err = fmt.Errorf("%s: missing tensor %s", b.arch, name)
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
			b.err = fmt.Errorf("%s: %s: %w", b.arch, name, err)
		}
		return t, nil, false
	}
	size := int64(rb) * int64(rest)
	data := b.shards[b.shardOf[name]]
	if t.FileOffset < 0 || t.FileOffset+size > int64(len(data)) {
		if b.err == nil {
			b.err = fmt.Errorf("%s: tensor %s runs past the file", b.arch, name)
		}
		return t, nil, false
	}
	return t, data[t.FileOffset : t.FileOffset+size], true
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
			b.err = fmt.Errorf("%s: %s: %w", b.arch, name, err)
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
	return b.matrix(name, true)
}

func (b *inkling2Binder) packed(t TensorInfo) inkling2PackedWeight {
	if b.err != nil {
		return nil
	}
	w, err := b.pack(t)
	if err != nil {
		b.err = fmt.Errorf("inkling2_loop: bind %s: %w", t.Name, err)
		return nil
	}
	return w
}

func (b *inkling2Binder) matrix(name string, usePacked bool) model.Inkling2LoopMatrix {
	t, raw, ok := b.info(name)
	if !ok {
		return nil
	}
	if len(t.Dims) != 2 {
		if b.err == nil {
			b.err = fmt.Errorf("%s: %s has %d dims, want a matrix", b.arch, name, len(t.Dims))
		}
		return nil
	}
	if usePacked && b.pack != nil {
		return b.packed(t)
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
			b.err = fmt.Errorf("%s: %s has %d dims, want experts", b.arch, name, len(t.Dims))
		}
		return nil
	}
	if b.pack != nil {
		return b.packed(t)
	}
	cols := int(t.Dims[0])
	rb, _ := inkling2RowBytes(t.Type, cols)
	return &inkling2QuantExperts{typ: t.Type, raw: raw, experts: int(t.Dims[2]), rows: int(t.Dims[1]), cols: cols, rowLen: rb}
}

// The matrix interface predates fallible device submissions. Unwind just that
// failure at the token boundary so the caller can discard the partial session.
type inkling2DispatchError struct{ cause error }

func inkling2RecoverDispatch(err *error) {
	if r := recover(); r != nil {
		if failure, ok := r.(inkling2DispatchError); ok {
			*err = failure.cause
		} else {
			panic(r)
		}
	}
}

func Inkling2LoopStep(h *model.Inkling2LoopHost, token int) (logits []float32, err error) {
	defer inkling2RecoverDispatch(&err)
	return h.Step(token), nil
}

func Inkling2LoopPrefillToken(h *model.Inkling2LoopHost, token int, final bool) (logits []float32, err error) {
	defer inkling2RecoverDispatch(&err)
	return h.PrefillToken(token, final), nil
}

func Inkling2LoopPrefill(ctx context.Context, h *model.Inkling2LoopHost, tokens []int) (logits []float32, err error) {
	defer inkling2RecoverDispatch(&err)
	return h.Prefill(ctx, tokens)
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
	logits, err := Inkling2LoopPrefill(context.Background(), h, prompt)
	if err != nil {
		return nil, err
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
		logits, err = Inkling2LoopStep(h, best)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
