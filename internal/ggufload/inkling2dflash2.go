// Inkling-2-Mini DFlash2 draft: strict loader and pure-Go host reference
// forward. The draft semantics (dynamic short convolutions, context K/V from
// target taps, non-causal sliding-window block attention, low-rank candidate
// selector) are ported from incoai/splash@2afc7337 (Apache-2.0):
// model/DFlashDraft.cpp, model/DraftCheckpoint.cpp, ops/DraftSelector.cpp and
// metal/kernels/decode/{draft,sampling}.metal.
package ggufload

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// Inkling2DFlash2DirName is the draft checkpoint directory that sits next
	// to the target GGUF.
	Inkling2DFlash2DirName = "Inkling-2-Mini-DFlash2"
	inkling2DFlash2Weights = "model.safetensors"
	inkling2DFlash2Config  = "config.json"
	// The selector keeps this many candidates per proposal position.
	inkling2DFlash2Candidates = 16
	inkling2DFlash2Taps       = 2
	inkling2DFlash2Stages     = 2
)

// Inkling2DFlash2Config is config.json with every field the draft reads.
type Inkling2DFlash2Config struct {
	Architectures []string `json:"architectures"`
	Layers        int      `json:"num_hidden_layers"`
	Intermediate  int      `json:"intermediate_size"`
	Hidden        int      `json:"hidden_size"`
	Vocab         int      `json:"vocab_size"`
	HeadDim       int      `json:"head_dim"`
	Heads         int      `json:"num_attention_heads"`
	KVHeads       int      `json:"num_key_value_heads"`
	Window        int      `json:"sliding_window"`
	Causal        bool     `json:"is_causal"`
	AttnBias      bool     `json:"attention_bias"`
	HiddenAct     string   `json:"hidden_act"`
	Eps           float64  `json:"rms_norm_eps"`
	TieEmbed      bool     `json:"tie_word_embeddings"`
	TargetLayers  int      `json:"num_target_layers"`
	Rope          struct {
		Theta float64 `json:"rope_theta"`
		Type  string  `json:"rope_type"`
	} `json:"rope_parameters"`
	DFlash struct {
		Taps        []int   `json:"target_layer_ids"`
		Mask        int     `json:"mask_token_id"`
		Block       int     `json:"block_size"`
		Rank        int     `json:"selector_rank"`
		TopK        int     `json:"selector_top_k"`
		Group       int     `json:"conv_group_size"`
		Softcap     float64 `json:"final_logit_softcapping"`
		SoftcapMode string  `json:"selector_softcap_mode"`
	} `json:"dflash_config"`
}

// validate pins the one geometry the reference forward implements; any drift
// is a different draft and is refused rather than run approximately.
// The conv tap count is pinned by the base_kernel tensor shapes instead.
func (c *Inkling2DFlash2Config) validate() error {
	d := c.DFlash
	ok := slices.Contains(c.Architectures, "DFlash2DraftModel") && c.Layers == 5 && c.Intermediate == 6912 &&
		c.Hidden == 2304 && c.Vocab == 201024 && c.HeadDim == 128 && c.Heads == 32 && c.KVHeads == 8 &&
		c.Window == 2048 && !c.Causal && !c.AttnBias && c.HiddenAct == "silu" && c.Eps == 1e-6 && !c.TieEmbed &&
		c.TargetLayers == 48 && c.Rope.Theta == 10000 && c.Rope.Type == "default" &&
		d.Block == 8 && d.Mask == 200056 && d.Group == 16 && d.Rank == 256 &&
		d.TopK == inkling2DFlash2Candidates && d.Softcap == 30 && d.SoftcapMode == "after_correction" &&
		slices.Equal(d.Taps, []int{5, 23, 41, 48})
	if !ok {
		return fmt.Errorf("inkling2 dflash2: incompatible configuration %+v", *c)
	}
	return nil
}

func (c *Inkling2DFlash2Config) dynamicWidth() int {
	return inkling2DFlash2Stages * inkling2DFlash2Taps * (c.Hidden / c.DFlash.Group)
}

// Inkling2DFlash2TensorShapes is every tensor model.safetensors must hold, and
// nothing else.
func Inkling2DFlash2TensorShapes(c *Inkling2DFlash2Config) map[string][]int {
	H, q, kv, dyn := c.Hidden, c.Heads*c.HeadDim, c.KVHeads*c.HeadDim, c.dynamicWidth()
	s := map[string][]int{
		"fc.weight":          {H, H * len(c.DFlash.Taps)},
		"hidden_norm.weight": {H},
		"norm.weight":        {H},
		"candidate_selector.hidden_projection.weight": {c.DFlash.Rank, H},
		"candidate_selector.predecessor_codebook":     {c.Vocab, c.DFlash.Rank},
		"candidate_selector.successor_codebook":       {c.Vocab, c.DFlash.Rank},
	}
	for l := 0; l < c.Layers; l++ {
		p := fmt.Sprintf("layers.%d.", l)
		s[p+"input_layernorm.weight"] = []int{H}
		s[p+"post_attention_layernorm.weight"] = []int{H}
		for _, conv := range []string{"attention_conv", "mlp_conv"} {
			s[p+conv+".base_kernel"] = []int{inkling2DFlash2Stages, inkling2DFlash2Taps, H}
			s[p+conv+".kernel_projection.weight"] = []int{dyn, H}
		}
		s[p+"self_attn.q_proj.weight"] = []int{q, H}
		s[p+"self_attn.k_proj.weight"] = []int{kv, H}
		s[p+"self_attn.v_proj.weight"] = []int{kv, H}
		s[p+"self_attn.o_proj.weight"] = []int{H, q}
		s[p+"self_attn.q_norm.weight"] = []int{c.HeadDim}
		s[p+"self_attn.k_norm.weight"] = []int{c.HeadDim}
		s[p+"mlp.gate_proj.weight"] = []int{c.Intermediate, H}
		s[p+"mlp.up_proj.weight"] = []int{c.Intermediate, H}
		s[p+"mlp.down_proj.weight"] = []int{H, c.Intermediate}
	}
	return s
}

type inkling2DFlash2Layer struct {
	inputNorm, postNorm []float32
	attnBase, mlpBase   []float32 // [stage][tap][hidden]
	attnDyn, mlpDyn     *inkling2BF16Matrix
	wq, wk, wv, wo      *inkling2BF16Matrix
	qNorm, kNorm        []float32
	gate, up, down      *inkling2BF16Matrix
}

// Inkling2DFlash2Model is the bound draft checkpoint.
type Inkling2DFlash2Model struct {
	Config     Inkling2DFlash2Config
	Dir        string
	Tensors    int
	layers     []inkling2DFlash2Layer
	fc         *inkling2BF16Matrix
	hiddenNorm []float32
	norm       []float32
	selProj    *inkling2BF16Matrix
	predCB     *inkling2BF16Matrix
	succCB     *inkling2BF16Matrix
	// mapped is the whole checkpoint mapping every matrix payload lies in
	mapped []byte
	closer io.Closer
}

func (m *Inkling2DFlash2Model) Close() error {
	if m.closer == nil {
		return nil
	}
	return m.closer.Close()
}

// Inkling2DFlash2AdjacentDir is the draft directory beside a target GGUF, or
// "" when it is absent.
func Inkling2DFlash2AdjacentDir(ggufPath string) (string, error) {
	dir := filepath.Join(filepath.Dir(ggufPath), Inkling2DFlash2DirName)
	if st, err := os.Stat(dir); errors.Is(err, os.ErrNotExist) {
		return "", nil
	} else if err != nil {
		return "", err
	} else if !st.IsDir() {
		return "", fmt.Errorf("inkling2 dflash2: %s is not a directory", dir)
	}
	return dir, nil
}

// OpenInkling2DFlash2 reads dir/config.json and maps dir/model.safetensors,
// refusing any geometry, dtype, shape, missing, extra or overlapping tensor.
func OpenInkling2DFlash2(dir string) (*Inkling2DFlash2Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, inkling2DFlash2Config))
	if err != nil {
		return nil, err
	}
	var cfg Inkling2DFlash2Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("inkling2 dflash2: %s: %w", inkling2DFlash2Config, err)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	data, closer, _, err := model.MmapOpen(filepath.Join(dir, inkling2DFlash2Weights))
	if err != nil {
		return nil, err
	}
	m, err := bindInkling2DFlash2(&cfg, data)
	if err != nil {
		closer.Close()
		return nil, err
	}
	m.Dir, m.closer, m.mapped = dir, closer, data
	return m, nil
}

func bindInkling2DFlash2(cfg *Inkling2DFlash2Config, data []byte) (*Inkling2DFlash2Model, error) {
	st, err := parseInkling2Safetensors(data)
	if err != nil {
		return nil, fmt.Errorf("inkling2 dflash2: %w", err)
	}
	want := Inkling2DFlash2TensorShapes(cfg)
	if err := checkInkling2DFlash2Tensors(st, want); err != nil {
		return nil, err
	}
	H, q, kv, dyn := cfg.Hidden, cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim, cfg.dynamicWidth()
	m := &Inkling2DFlash2Model{
		Config:     *cfg,
		Tensors:    len(st.entries),
		fc:         st.mat("fc.weight", H, H*len(cfg.DFlash.Taps)),
		hiddenNorm: st.vec("hidden_norm.weight", H),
		norm:       st.vec("norm.weight", H),
		selProj:    st.mat("candidate_selector.hidden_projection.weight", cfg.DFlash.Rank, H),
		predCB:     st.mat("candidate_selector.predecessor_codebook", cfg.Vocab, cfg.DFlash.Rank),
		succCB:     st.mat("candidate_selector.successor_codebook", cfg.Vocab, cfg.DFlash.Rank),
	}
	for l := 0; l < cfg.Layers; l++ {
		p := fmt.Sprintf("layers.%d.", l)
		m.layers = append(m.layers, inkling2DFlash2Layer{
			inputNorm: st.vec(p+"input_layernorm.weight", H),
			postNorm:  st.vec(p+"post_attention_layernorm.weight", H),
			attnBase:  st.vec(p+"attention_conv.base_kernel", inkling2DFlash2Stages*inkling2DFlash2Taps*H),
			mlpBase:   st.vec(p+"mlp_conv.base_kernel", inkling2DFlash2Stages*inkling2DFlash2Taps*H),
			attnDyn:   st.mat(p+"attention_conv.kernel_projection.weight", dyn, H),
			mlpDyn:    st.mat(p+"mlp_conv.kernel_projection.weight", dyn, H),
			wq:        st.mat(p+"self_attn.q_proj.weight", q, H),
			wk:        st.mat(p+"self_attn.k_proj.weight", kv, H),
			wv:        st.mat(p+"self_attn.v_proj.weight", kv, H),
			wo:        st.mat(p+"self_attn.o_proj.weight", H, q),
			qNorm:     st.vec(p+"self_attn.q_norm.weight", cfg.HeadDim),
			kNorm:     st.vec(p+"self_attn.k_norm.weight", cfg.HeadDim),
			gate:      st.mat(p+"mlp.gate_proj.weight", cfg.Intermediate, H),
			up:        st.mat(p+"mlp.up_proj.weight", cfg.Intermediate, H),
			down:      st.mat(p+"mlp.down_proj.weight", H, cfg.Intermediate),
		})
	}
	if st.err != nil {
		return nil, fmt.Errorf("inkling2 dflash2: %w", st.err)
	}
	if len(st.used) != len(want) {
		return nil, fmt.Errorf("inkling2 dflash2: bound %d of %d tensors", len(st.used), len(want))
	}
	return m, nil
}

// checkInkling2DFlash2Tensors requires exactly the wanted BF16 tensors at
// their exact shapes, tiling the payload with no gap, overlap or tail.
func checkInkling2DFlash2Tensors(st *inkling2Safetensors, want map[string][]int) error {
	for name := range st.entries {
		if _, ok := want[name]; !ok {
			return fmt.Errorf("inkling2 dflash2: unexpected checkpoint tensor %s", name)
		}
	}
	type span struct {
		name       string
		start, end int
	}
	var spans []span
	for name, shape := range want {
		e, ok := st.entries[name]
		if !ok {
			return fmt.Errorf("inkling2 dflash2: missing tensor %s", name)
		}
		if e.DType != "BF16" || !slices.Equal(e.Shape, shape) {
			return fmt.Errorf("inkling2 dflash2: tensor %s is %s%v, want BF16%v", name, e.DType, e.Shape, shape)
		}
		n := 2
		for _, d := range shape {
			n *= d
		}
		if e.Offsets[1]-e.Offsets[0] != n {
			return fmt.Errorf("inkling2 dflash2: tensor %s spans %d bytes, want %d", name, e.Offsets[1]-e.Offsets[0], n)
		}
		spans = append(spans, span{name, e.Offsets[0], e.Offsets[1]})
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	at := 0
	for _, s := range spans {
		if s.start != at {
			return fmt.Errorf("inkling2 dflash2: tensor %s starts at %d, want %d", s.name, s.start, at)
		}
		at = s.end
	}
	if at != len(st.data) {
		return fmt.Errorf("inkling2 dflash2: tensors cover %d of %d payload bytes", at, len(st.data))
	}
	return nil
}

// Inkling2DFlash2Target is what the draft borrows from its target: the token
// table with its norm (draft rows of tokens are the target's first-layer
// input rows), and the LM head over the first Vocab (unpadded) tokens.
type Inkling2DFlash2Target struct {
	TokEmbd interface {
		Row(token int, dst []float32)
	}
	TokNorm []float32
	Output  model.Inkling2LoopMatrix
	Vocab   int
	NormEps float32
}

// Inkling2DFlash2TargetOf borrows the host target's table, norm and head.
func Inkling2DFlash2TargetOf(t *Inkling2LoopModel) Inkling2DFlash2Target {
	return Inkling2DFlash2Target{TokEmbd: t.Weights.TokEmbd, TokNorm: t.Weights.TokNorm, Output: t.Weights.Output,
		Vocab: t.HP.VocabUnpadded, NormEps: t.HP.NormEps}
}

// Inkling2DFlash2Draft is the host reference forward of one lane: the
// committed context rows' per-layer keys and values, and greedy block
// proposals over them.
type Inkling2DFlash2Draft struct {
	M       *Inkling2DFlash2Model
	T       Inkling2DFlash2Target
	keys    [][][]float32 // [layer][position - base][kvHeads*headDim], normed and rotated
	values  [][][]float32
	base    int
	n       int
	invFreq []float64
}

func NewInkling2DFlash2Draft(m *Inkling2DFlash2Model, t Inkling2DFlash2Target) (*Inkling2DFlash2Draft, error) {
	c := &m.Config
	if t.TokEmbd == nil || t.Output == nil || len(t.TokNorm) != c.Hidden || t.Output.Cols() != c.Hidden ||
		t.Vocab <= c.DFlash.Mask || t.Vocab > t.Output.Rows() || t.Output.Rows() > c.Vocab {
		return nil, fmt.Errorf("inkling2 dflash2: target table/head do not fit a %d-wide, %d-row draft", c.Hidden, c.Vocab)
	}
	d := &Inkling2DFlash2Draft{M: m, T: t, keys: make([][][]float32, c.Layers), values: make([][][]float32, c.Layers)}
	pairs := c.HeadDim / 2
	for i := 0; i < pairs; i++ {
		d.invFreq = append(d.invFreq, math.Pow(c.Rope.Theta, -float64(2*i)/float64(c.HeadDim)))
	}
	return d, nil
}

// Len is the number of committed context rows; the next anchor sits there.
func (d *Inkling2DFlash2Draft) Len() int { return d.n }

// Commit appends the context row of position pos from the target's taps, one
// row of Hidden values per target_layer_ids entry in order.
func (d *Inkling2DFlash2Draft) Commit(pos int, taps [][]float32) error {
	c := &d.M.Config
	if pos != d.n {
		return fmt.Errorf("inkling2 dflash2: commit of position %d after %d rows", pos, d.n)
	}
	if len(taps) != len(c.DFlash.Taps) {
		return fmt.Errorf("inkling2 dflash2: %d taps, want %d", len(taps), len(c.DFlash.Taps))
	}
	cat := make([]float32, 0, c.Hidden*len(taps))
	for i, t := range taps {
		if len(t) != c.Hidden {
			return fmt.Errorf("inkling2 dflash2: tap %d has %d values, want %d", i, len(t), c.Hidden)
		}
		cat = append(cat, t...)
	}
	h := inkling2DFlash2RMS(d.M.fc.mulRows([][]float32{cat})[0], d.M.hiddenNorm, float32(c.Eps))
	kvw := c.KVHeads * c.HeadDim
	for l := range d.M.layers {
		L := &d.M.layers[l]
		k := L.wk.mulRows([][]float32{h})[0]
		v := L.wv.mulRows([][]float32{h})[0]
		for hd := 0; hd < c.KVHeads; hd++ {
			d.normRope(k[hd*c.HeadDim:(hd+1)*c.HeadDim], L.kNorm, pos)
		}
		d.keys[l] = append(d.keys[l], k[:kvw])
		d.values[l] = append(d.values[l], v[:kvw])
	}
	d.n++
	// A query sees at most Window-1 context rows; older ones are dead.
	if drop := d.n - d.base - (c.Window - 1); drop > 0 {
		for l := range d.keys {
			d.keys[l] = d.keys[l][drop:]
			d.values[l] = d.values[l][drop:]
		}
		d.base += drop
	}
	return nil
}

// normRope RMS-normalizes one head in place and rotates its halves at pos.
func (d *Inkling2DFlash2Draft) normRope(x, w []float32, pos int) {
	n := inkling2DFlash2RMS(x, w, float32(d.M.Config.Eps))
	half := len(x) / 2
	for i := 0; i < half; i++ {
		s, co := math.Sincos(float64(pos) * d.invFreq[i])
		a, b := float64(n[i]), float64(n[i+half])
		x[i] = float32(a*co - b*s)
		x[i+half] = float32(b*co + a*s)
	}
}

// Inkling2DFlash2Block is one draft block's output: the greedy chain and,
// per proposal position, the selector's candidates and corrected scores.
type Inkling2DFlash2Block struct {
	Tokens     []int
	Candidates [][]int
	Scores     [][]float32
}

// Propose drafts depth (1..Block-1) tokens after anchor, which sits at
// position Len(): the selector's greedy chain, one child a row and no host
// chain (DraftTreeLane{depth, children: 1}).
func (d *Inkling2DFlash2Draft) Propose(anchor, depth int) (*Inkling2DFlash2Block, error) {
	c := &d.M.Config
	rows := c.DFlash.Block
	if depth < 1 || depth >= rows || anchor < 0 || anchor >= d.T.Vocab {
		return nil, fmt.Errorf("inkling2 dflash2: invalid proposal (anchor %d, depth %d)", anchor, depth)
	}
	eps := float32(c.Eps)
	x := make([][]float32, rows)
	for r := range x {
		tok := c.DFlash.Mask
		if r == 0 {
			tok = anchor
		}
		e := make([]float32, c.Hidden)
		d.T.TokEmbd.Row(tok, e)
		x[r] = inkling2DFlash2RMS(e, d.T.TokNorm, d.T.NormEps)
	}
	for l := range d.M.layers {
		x = d.layer(l, x, eps)
	}
	fin := make([][]float32, rows)
	for r := range x {
		fin[r] = inkling2DFlash2RMS(x[r], d.M.norm, eps)
	}
	sel := d.M.selProj.mulRows(fin[1 : depth+1])
	out := &Inkling2DFlash2Block{}
	prev := anchor
	logits := make([]float32, d.T.Output.Rows())
	for p := 0; p < depth; p++ {
		d.T.Output.MatVec(logits, fin[p+1])
		cand, unary := inkling2DFlash2Top(logits[:d.T.Vocab], inkling2DFlash2Candidates)
		scores := make([]float32, len(cand))
		ctx := make([]float32, c.DFlash.Rank)
		pred := d.M.predCB.row(prev)
		for i := range ctx {
			ctx[i] = pred[i] * sel[p][i]
		}
		best := 0
		for i, tok := range cand {
			succ := d.M.succCB.row(tok)
			var e float32
			for j := range ctx {
				e += ctx[j] * succ[j]
			}
			sum := unary[i] + e
			if cap := float32(c.DFlash.Softcap); cap > 0 && !math.IsInf(float64(sum), -1) {
				sum = cap * float32(math.Tanh(float64(sum/cap)))
			}
			scores[i] = sum
			if sum > scores[best] {
				best = i
			}
		}
		out.Candidates = append(out.Candidates, cand)
		out.Scores = append(out.Scores, scores)
		out.Tokens = append(out.Tokens, cand[best])
		prev = cand[best]
	}
	return out, nil
}

// layer is one draft layer over the block's rows (all of them see each other).
func (d *Inkling2DFlash2Draft) layer(l int, x [][]float32, eps float32) [][]float32 {
	c := &d.M.Config
	L := &d.M.layers[l]
	rows, H := len(x), c.Hidden
	n := make([][]float32, rows)
	for r := range x {
		n[r] = inkling2DFlash2RMS(x[r], L.inputNorm, eps)
	}
	dyn := L.attnDyn.mulRows(n)
	conv := inkling2DFlash2Conv(n, dyn, L.attnBase, nil, 0, H, c.DFlash.Group)
	q, k, v := L.wq.mulRows(conv), L.wk.mulRows(conv), L.wv.mulRows(conv)
	pos := d.n
	for r := 0; r < rows; r++ {
		for hd := 0; hd < c.Heads; hd++ {
			d.normRope(q[r][hd*c.HeadDim:(hd+1)*c.HeadDim], L.qNorm, pos+r)
		}
		for hd := 0; hd < c.KVHeads; hd++ {
			d.normRope(k[r][hd*c.HeadDim:(hd+1)*c.HeadDim], L.kNorm, pos+r)
		}
	}
	attn := d.attend(l, q, k, v)
	o := L.wo.mulRows(attn)
	res := inkling2DFlash2Conv(o, dyn, L.attnBase, x, 1, H, c.DFlash.Group)

	n2 := make([][]float32, rows)
	for r := range res {
		n2[r] = inkling2DFlash2RMS(res[r], L.postNorm, eps)
	}
	dyn2 := L.mlpDyn.mulRows(n2)
	conv2 := inkling2DFlash2Conv(n2, dyn2, L.mlpBase, nil, 0, H, c.DFlash.Group)
	g, u := L.gate.mulRows(conv2), L.up.mulRows(conv2)
	for r := range g {
		for i, gv := range g[r] {
			g[r][i] = gv / (1 + float32(math.Exp(float64(-gv)))) * u[r][i]
		}
	}
	down := L.down.mulRows(g)
	return inkling2DFlash2Conv(down, dyn2, L.mlpBase, res, 1, H, c.DFlash.Group)
}

// attend runs each block row over its window of committed context rows and
// every row of the block, non-causally, grouped-query.
func (d *Inkling2DFlash2Draft) attend(l int, q, k, v [][]float32) [][]float32 {
	c := &d.M.Config
	D, group := c.HeadDim, c.Heads/c.KVHeads
	scale := 1 / math.Sqrt(float64(D))
	ctxK, ctxV := d.keys[l], d.values[l]
	out := make([][]float32, len(q))
	var wg sync.WaitGroup
	for r := range q {
		out[r] = make([]float32, c.Heads*D)
		first := max(0, d.n+r-(c.Window-1)) - d.base
		for hd := 0; hd < c.Heads; hd++ {
			wg.Add(1)
			go func(r, hd, first int) {
				defer wg.Done()
				kh := hd / group
				qh := q[r][hd*D : (hd+1)*D]
				var scores []float64
				var vals [][]float32
				add := func(kr, vr []float32) {
					var s float64
					for i, qv := range qh {
						s += float64(qv) * float64(kr[kh*D+i])
					}
					scores = append(scores, s*scale)
					vals = append(vals, vr[kh*D:(kh+1)*D])
				}
				for p := first; p < len(ctxK); p++ {
					add(ctxK[p], ctxV[p])
				}
				for j := range k {
					add(k[j], v[j])
				}
				mx := math.Inf(-1)
				for _, s := range scores {
					mx = math.Max(mx, s)
				}
				acc := make([]float64, D)
				var total float64
				for i, s := range scores {
					w := math.Exp(s - mx)
					total += w
					for j, vv := range vals[i] {
						acc[j] += w * float64(vv)
					}
				}
				dst := out[r][hd*D : (hd+1)*D]
				for j := range dst {
					dst[j] = float32(acc[j] / total)
				}
			}(r, hd, first)
		}
	}
	wg.Wait()
	return out
}

// inkling2DFlash2Conv is one stage of a dynamic short convolution (kind 0
// before the mixer, 1 after it with the residual): channel ch of row r weighs
// row r and row r-1 of the block, each by its base tap plus the dynamic
// weight of its 16-channel group.
func inkling2DFlash2Conv(in, dyn [][]float32, base []float32, residual [][]float32, kind, H, group int) [][]float32 {
	groups := H / group
	out := make([][]float32, len(in))
	for r := range in {
		out[r] = make([]float32, H)
		dr := dyn[r]
		for ch := 0; ch < H; ch++ {
			g := ch / group
			v := in[r][ch] * (base[(kind*inkling2DFlash2Taps)*H+ch] + dr[(kind*inkling2DFlash2Taps)*groups+g])
			if r > 0 {
				v += in[r-1][ch] * (base[(kind*inkling2DFlash2Taps+1)*H+ch] + dr[(kind*inkling2DFlash2Taps+1)*groups+g])
			}
			if residual != nil {
				v += residual[r][ch]
			}
			out[r][ch] = v
		}
	}
	return out
}

// inkling2DFlash2Top is the k largest logits, value descending then token
// ascending, as the selector ranks them.
func inkling2DFlash2Top(logits []float32, k int) ([]int, []float32) {
	ids := make([]int, 0, k+1)
	vals := make([]float32, 0, k+1)
	for tok, v := range logits {
		if len(ids) == k && !(v > vals[k-1]) {
			continue
		}
		at := len(ids)
		for at > 0 && v > vals[at-1] {
			at--
		}
		ids = slices.Insert(ids, at, tok)
		vals = slices.Insert(vals, at, v)
		if len(ids) > k {
			ids, vals = ids[:k], vals[:k]
		}
	}
	return ids, vals
}

func inkling2DFlash2RMS(x, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	s := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s * w[i]
	}
	return out
}

func (m *inkling2BF16Matrix) row(r int) []float32 {
	raw := m.raw[2*r*m.cols : 2*(r+1)*m.cols]
	out := make([]float32, m.cols)
	for i := range out {
		out[i] = bf16ToF32(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return out
}

// mulRows is x @ m^T for a few rows, decoding each weight row once for all
// of them.
func (m *inkling2BF16Matrix) mulRows(xs [][]float32) [][]float32 {
	out := make([][]float32, len(xs))
	for i := range out {
		out[i] = make([]float32, m.rows)
	}
	const block = 32
	blocks := (m.rows + block - 1) / block
	workers := min(runtime.GOMAXPROCS(0), blocks)
	var wg sync.WaitGroup
	next := make(chan int, blocks)
	for b := 0; b < blocks; b++ {
		next <- b
	}
	close(next)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			wr := make([]float32, m.cols)
			for b := range next {
				for r := b * block; r < min((b+1)*block, m.rows); r++ {
					raw := m.raw[2*r*m.cols : 2*(r+1)*m.cols]
					for i := range wr {
						wr[i] = bf16ToF32(binary.LittleEndian.Uint16(raw[2*i:]))
					}
					for i, x := range xs {
						var s float32
						for j, wv := range wr {
							s += wv * x[j]
						}
						out[i][r] = s
					}
				}
			}
		}()
	}
	wg.Wait()
	return out
}
