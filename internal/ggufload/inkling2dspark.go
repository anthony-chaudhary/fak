package ggufload

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The DSpark draft ships only as the HF checkpoint's BF16 mtp.safetensors
// (no GGUF carries it); its geometry is config.json's mtp_config.
const (
	inkling2DSparkWeightsFile = "mtp.safetensors"
	inkling2DSparkConfigFile  = "config.json"
)

type inkling2DSparkConfig struct {
	TextConfig struct {
		UnpaddedVocabSize int `json:"unpadded_vocab_size"`
	} `json:"text_config"`
	MTPConfig *struct {
		HiddenSize        int     `json:"hidden_size"`
		NumHiddenLayers   int     `json:"num_hidden_layers"`
		NumAttentionHeads int     `json:"num_attention_heads"`
		NumKeyValueHeads  int     `json:"num_key_value_heads"`
		HeadDim           int     `json:"head_dim"`
		IntermediateSize  int     `json:"intermediate_size"`
		RMSNormEps        float32 `json:"rms_norm_eps"`
		RopeTheta         float64 `json:"rope_theta"`
		LocalLayerIDs     []int   `json:"local_layer_ids"`
		SlidingWindowSize int     `json:"sliding_window_size"`
		BlockSize         int     `json:"block_size"`
		MarkovRank        int     `json:"markov_rank"`
		MarkovHeadType    string  `json:"markov_head_type"`
		TargetLayerIDs    []int   `json:"target_layer_ids"`
		QBias             bool    `json:"q_bias"`
		OBias             bool    `json:"o_bias"`
	} `json:"mtp_config"`
}

// Inkling2DSparkModel is the DSpark draft bound to a loaded target, whose
// token embedding, embedding norm and LM head it shares. TargetVisits are the
// target layer visits to tap (Inkling2LoopHost.FeatureVisits).
type Inkling2DSparkModel struct {
	HP           model.Inkling2DSparkHParams
	Weights      *model.Inkling2DSparkWeights
	TargetVisits []int
	closer       io.Closer
}

func (m *Inkling2DSparkModel) Close() error {
	if m.closer == nil {
		return nil
	}
	return m.closer.Close()
}

// OpenInkling2DSpark binds dir/mtp.safetensors with dir/config.json, where dir
// is the HF snapshot of the target checkpoint.
func OpenInkling2DSpark(dir string, target *Inkling2LoopModel) (*Inkling2DSparkModel, error) {
	raw, err := os.ReadFile(filepath.Join(dir, inkling2DSparkConfigFile))
	if err != nil {
		return nil, err
	}
	var cfg inkling2DSparkConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("inkling2 dspark: %s: %w", inkling2DSparkConfigFile, err)
	}
	data, closer, _, err := model.MmapOpen(filepath.Join(dir, inkling2DSparkWeightsFile))
	if err != nil {
		return nil, err
	}
	m, err := bindInkling2DSpark(&cfg, data, target)
	if err != nil {
		closer.Close()
		return nil, err
	}
	m.closer = closer
	return m, nil
}

func bindInkling2DSpark(cfg *inkling2DSparkConfig, data []byte, target *Inkling2LoopModel) (*Inkling2DSparkModel, error) {
	mc := cfg.MTPConfig
	if mc == nil {
		return nil, fmt.Errorf("inkling2 dspark: config has no mtp_config")
	}
	if mc.MarkovHeadType != "rnn_hidden" || mc.QBias || mc.OBias {
		return nil, fmt.Errorf("inkling2 dspark: unsupported mtp_config (markov_head_type %q, q_bias %v, o_bias %v)", mc.MarkovHeadType, mc.QBias, mc.OBias)
	}
	thp := target.HP
	visits := thp.Layers + (thp.Passes-1)*(thp.CoreEnd-thp.CoreStart)
	tl := mc.TargetLayerIDs
	if len(tl) < 2 || tl[len(tl)-1] != visits || mc.HiddenSize != thp.Hidden {
		return nil, fmt.Errorf("inkling2 dspark: target_layer_ids %v / hidden %d do not fit a %d-visit, %d-wide target", tl, mc.HiddenSize, visits, thp.Hidden)
	}
	for i := 1; i < len(tl); i++ {
		if tl[i] <= tl[i-1] {
			return nil, fmt.Errorf("inkling2 dspark: target_layer_ids %v not strictly increasing", tl)
		}
	}
	hp := model.Inkling2DSparkHParams{
		Hidden: mc.HiddenSize, Layers: mc.NumHiddenLayers, Heads: mc.NumAttentionHeads,
		KVHeads: mc.NumKeyValueHeads, HeadDim: mc.HeadDim, Intermediate: mc.IntermediateSize,
		Block: mc.BlockSize, MarkovRank: mc.MarkovRank, SlidingWindow: mc.SlidingWindowSize,
		VocabUnpadded: thp.VocabUnpadded, Taps: len(tl), LocalLayer: make([]bool, mc.NumHiddenLayers),
		NormEps: mc.RMSNormEps, RopeTheta: mc.RopeTheta,
	}
	for _, id := range mc.LocalLayerIDs {
		if id < 0 || id >= hp.Layers {
			return nil, fmt.Errorf("inkling2 dspark: local layer %d of %d", id, hp.Layers)
		}
		hp.LocalLayer[id] = true
	}
	st, err := parseInkling2Safetensors(data)
	if err != nil {
		return nil, err
	}
	H, r, kv, q := hp.Hidden, hp.MarkovRank, hp.KVHeads*hp.HeadDim, hp.Heads*hp.HeadDim
	w := &model.Inkling2DSparkWeights{
		Norm:          st.vec("norm.weight", H),
		MaskEmbed:     st.vec("mask_embed", H),
		MarkovW1:      inkling2BF16Embedding{st.mat("markov_head.markov_w1.weight", -1, r)},
		RnnProj:       st.mat("markov_head.rnn_proj.weight", 3*r, 2*r+2*H),
		RnnBias:       st.vec("markov_head.rnn_proj.bias", 3*r),
		HiddenOut:     st.mat("markov_head.hidden_out.weight", H, r),
		HiddenOutBias: st.vec("markov_head.hidden_out.bias", H),
		MarkovNorm:    st.vec("markov_head.dflare_norm.weight", H),
		TokEmbd:       target.Weights.TokEmbd,
		TokNorm:       target.Weights.TokNorm,
		Output:        target.Weights.Output,
	}
	fuse := st.vec("fuse_weight", (hp.Layers+1)*hp.Taps)
	for l := 0; l <= hp.Layers && fuse != nil; l++ {
		w.FuseWeight = append(w.FuseWeight, fuse[l*hp.Taps:(l+1)*hp.Taps])
	}
	for l := 0; l < hp.Layers; l++ {
		p := fmt.Sprintf("layers.%d.", l)
		w.Layers = append(w.Layers, model.Inkling2DSparkLayerWeights{
			InputNorm:    st.vec(p+"input_layernorm.weight", H),
			PostAttnNorm: st.vec(p+"post_attention_layernorm.weight", H),
			DFlareNorm:   st.vec(p+"dflare_norm.weight", H),
			QNorm:        st.vec(p+"self_attn.q_norm.weight", hp.HeadDim),
			KNorm:        st.vec(p+"self_attn.k_norm.weight", hp.HeadDim),
			KNormCtx:     st.vec(p+"self_attn.k_norm_ctx.weight", hp.HeadDim),
			Wq:           st.mat(p+"self_attn.q_proj.weight", q, H),
			Wk:           st.mat(p+"self_attn.k_proj.weight", kv, H),
			Wv:           st.mat(p+"self_attn.v_proj.weight", kv, H),
			Wo:           st.mat(p+"self_attn.o_proj.weight", H, q),
			WkCtx:        st.mat(p+"self_attn.k_proj_ctx.weight", kv, H),
			WvCtx:        st.mat(p+"self_attn.v_proj_ctx.weight", kv, H),
			Gate:         st.mat(p+"mlp.gate_proj.weight", hp.Intermediate, H),
			Up:           st.mat(p+"mlp.up_proj.weight", hp.Intermediate, H),
			Down:         st.mat(p+"mlp.down_proj.weight", H, hp.Intermediate),
		})
	}
	if st.err != nil {
		return nil, st.err
	}
	if len(st.used) != len(st.entries) {
		for name := range st.entries {
			if !st.used[name] {
				return nil, fmt.Errorf("inkling2 dspark: unbound checkpoint tensor %s", name)
			}
		}
	}
	return &Inkling2DSparkModel{HP: hp, Weights: w, TargetVisits: append([]int(nil), tl...)}, nil
}

type inkling2STEntry struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

// inkling2Safetensors binds BF16 tensors of one mapped safetensors file and
// records the first binding error, so a checkpoint missing several tensors
// fails with the first one named.
type inkling2Safetensors struct {
	data    []byte
	entries map[string]inkling2STEntry
	used    map[string]bool
	err     error
}

func parseInkling2Safetensors(data []byte) (*inkling2Safetensors, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("inkling2 dspark: safetensors file of %d bytes", len(data))
	}
	n := binary.LittleEndian.Uint64(data)
	if n > uint64(len(data)-8) {
		return nil, fmt.Errorf("inkling2 dspark: safetensors header of %d bytes exceeds the file", n)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+n], &hdr); err != nil {
		return nil, fmt.Errorf("inkling2 dspark: safetensors header: %w", err)
	}
	st := &inkling2Safetensors{data: data[8+n:], entries: map[string]inkling2STEntry{}, used: map[string]bool{}}
	for name, raw := range hdr {
		if name == "__metadata__" {
			continue
		}
		var e inkling2STEntry
		if err := json.Unmarshal(raw, &e); err != nil {
			return nil, fmt.Errorf("inkling2 dspark: tensor %s: %w", name, err)
		}
		st.entries[name] = e
	}
	return st, nil
}

// bf16 returns the raw BF16 payload of name, checking its element count.
func (st *inkling2Safetensors) bf16(name string, count int) []byte {
	if st.err != nil {
		return nil
	}
	e, ok := st.entries[name]
	if !ok {
		st.err = fmt.Errorf("inkling2 dspark: missing tensor %s", name)
		return nil
	}
	n := 1
	for _, d := range e.Shape {
		n *= d
	}
	if e.DType != "BF16" || (count >= 0 && n != count) || e.Offsets[1]-e.Offsets[0] != 2*n ||
		e.Offsets[0] < 0 || e.Offsets[1] > len(st.data) {
		st.err = fmt.Errorf("inkling2 dspark: tensor %s is %s%v, want BF16 of %d values", name, e.DType, e.Shape, count)
		return nil
	}
	st.used[name] = true
	return st.data[e.Offsets[0]:e.Offsets[1]]
}

func (st *inkling2Safetensors) vec(name string, count int) []float32 {
	raw := st.bf16(name, count)
	if raw == nil {
		return nil
	}
	out := make([]float32, len(raw)/2)
	for i := range out {
		out[i] = bf16ToF32(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return out
}

// mat binds a [rows][cols] BF16 matrix; rows < 0 takes the row count from the file.
func (st *inkling2Safetensors) mat(name string, rows, cols int) *inkling2BF16Matrix {
	if rows < 0 {
		if e, ok := st.entries[name]; ok && len(e.Shape) == 2 {
			rows = e.Shape[0]
		}
	}
	raw := st.bf16(name, rows*cols)
	if raw == nil {
		return &inkling2BF16Matrix{}
	}
	return &inkling2BF16Matrix{rows: rows, cols: cols, raw: raw}
}

func bf16ToF32(u uint16) float32 { return math.Float32frombits(uint32(u) << 16) }

type inkling2BF16Matrix struct {
	rows, cols int
	raw        []byte
}

func (m *inkling2BF16Matrix) Rows() int { return m.rows }
func (m *inkling2BF16Matrix) Cols() int { return m.cols }
func (m *inkling2BF16Matrix) MatVec(dst, x []float32) {
	model.Inkling2LoopParallelRows(m.rows, func(r int) {
		row := m.raw[2*r*m.cols : 2*(r+1)*m.cols]
		var s float32
		for c := range x {
			s += bf16ToF32(binary.LittleEndian.Uint16(row[2*c:])) * x[c]
		}
		dst[r] = s
	})
}

type inkling2BF16Embedding struct{ m *inkling2BF16Matrix }

func (e inkling2BF16Embedding) Row(token int, dst []float32) {
	row := e.m.raw[2*token*e.m.cols : 2*(token+1)*e.m.cols]
	for c := range dst {
		dst[c] = bf16ToF32(binary.LittleEndian.Uint16(row[2*c:]))
	}
}

// Inkling2DSparkTeacherForced runs the target host over tokens (a prompt of
// promptLen tokens and its greedy continuation) and, at every anchor position
// p >= promptLen with a full block of known continuation, drafts Block-1
// tokens and returns how many leading drafts match tokens[p+1:]. Under greedy
// decoding this is the accepted length speculative decode would realize.
func Inkling2DSparkTeacherForced(target *Inkling2LoopModel, draft *Inkling2DSparkModel, tokens []int, promptLen int) ([]int, error) {
	h, err := model.NewInkling2LoopHost(target.HP, target.Weights)
	if err != nil {
		return nil, err
	}
	h.FeatureVisits = draft.TargetVisits
	d, err := model.NewInkling2DSparkDraft(draft.HP, draft.Weights)
	if err != nil {
		return nil, err
	}
	gamma := draft.HP.Block - 1
	var accepted []int
	for p, tok := range tokens {
		if p >= promptLen && p+gamma < len(tokens) {
			proposal, err := d.Draft(tok, p)
			if err != nil {
				return nil, err
			}
			n := 0
			for n < gamma && proposal[n] == tokens[p+1+n] {
				n++
			}
			accepted = append(accepted, n)
		}
		h.Step(tok)
		if err := d.Commit(p, h.Features); err != nil {
			return nil, err
		}
	}
	return accepted, nil
}
