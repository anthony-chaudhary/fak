// Package qwen4exp implements the deterministic correctness oracle for the
// Qwen3.8-Flash-Next four-layer cadence. It is deliberately small, scalar, and
// fak-native: it is a parity target, not a production throughput path.
package qwen4exp

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
)

const (
	Engine                  = "fak-native/qwen4exp"
	HiddenSize              = 4
	NumRoutedExperts        = 512
	ExpertsPerToken         = 10
	SparseSelectionCapacity = 2048
	StateLayout             = "layer-major,row-major,float32-le"
	SourceRevision          = "QwenLM/Qwen3.8-Flash-Next@513aa6e18a335296fc13e538232a8735b230877d"
	CheckpointRevision      = "Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c"
)

type LayerKind string

const (
	GatedDelta      LayerKind = "gated-delta"
	SparseAttention LayerKind = "sparse-attention"
)

var Cadence = [...]LayerKind{GatedDelta, GatedDelta, GatedDelta, SparseAttention}

// TensorSpec captures the checkpoint index contract that the oracle protects.
type TensorSpec struct {
	Name  string
	Shape []int
}

// oracleTensorShape returns the exact tensor names and logical shapes used
// by the four-layer oracle. Callers must not transpose or alias these tensors.
func requiredTensorSchema() []TensorSpec {
	// Names are pinned verbatim from model.safetensors.index.json at
	// CheckpointRevision. The compact shapes describe the oracle's synthetic
	// weights; names and ordering protect the production loader seam.
	out := make([]TensorSpec, 0, 64)
	for layer, kind := range Cadence {
		prefix := fmt.Sprintf("model.language_model.layers.%d", layer)
		if kind == GatedDelta {
			for _, name := range []string{"A_log", "conv1d.weight", "dt_bias", "in_proj_a.weight", "in_proj_b.weight", "in_proj_qkv.weight", "in_proj_z.weight", "norm.weight", "out_proj.weight"} {
				out = append(out, TensorSpec{Name: prefix + ".linear_attn." + name, Shape: []int{HiddenSize, HiddenSize}})
			}
		} else {
			for _, name := range []string{"indexer.index_qk_proj.weight", "indexer.k_layernorm.weight", "indexer.q_layernorm.weight", "k_norm.weight", "k_proj.weight", "o_proj.weight", "q_norm.weight", "q_proj.weight", "v_proj.weight"} {
				out = append(out, TensorSpec{Name: prefix + ".self_attn." + name, Shape: []int{HiddenSize, HiddenSize}})
			}
		}
		for _, name := range []string{"experts.down_proj", "experts.gate_up_proj", "gate.weight", "shared_expert_gate.weight", "shared_expert.down_proj.weight", "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight"} {
			out = append(out, TensorSpec{Name: prefix + ".mlp." + name, Shape: []int{HiddenSize, HiddenSize}})
		}
	}
	return out
}

func ValidateTensorLayout(got []TensorSpec) error {
	want := requiredTensorSchema()
	if len(got) != len(want) {
		return fmt.Errorf("qwen4exp tensor layout: got %d entries, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Name != want[i].Name || !sameInts(got[i].Shape, want[i].Shape) {
			return fmt.Errorf("qwen4exp tensor layout[%d]: got %s%v, want %s%v", i, got[i].Name, got[i].Shape, want[i].Name, want[i].Shape)
		}
	}
	return nil
}

type ProjectionTrace struct {
	Q []float32 `json:"q"`
	K []float32 `json:"k"`
	V []float32 `json:"v"`
}

type RouteTrace struct {
	ExpertIDs     []int     `json:"expert_ids"`
	ExpertWeights []float32 `json:"expert_weights"`
	SharedExpert  bool      `json:"shared_expert"`
}

type LayerTrace struct {
	Kind          LayerKind       `json:"kind"`
	Projection    ProjectionTrace `json:"projection"`
	Recurrent     []float32       `json:"recurrent_state,omitempty"`
	Route         RouteTrace      `json:"route"`
	SelectedToken []int           `json:"selected_token_ids,omitempty"`
	Output        []float32       `json:"output"`
}

type TokenTrace struct {
	TokenID int          `json:"token_id"`
	Layers  []LayerTrace `json:"layers"`
}

type Trace struct {
	Engine             string       `json:"engine"`
	SourceRevision     string       `json:"source_revision"`
	CheckpointRevision string       `json:"checkpoint_revision"`
	StateLayout        string       `json:"state_layout"`
	Tokens             []TokenTrace `json:"tokens"`
}

// State contains only FP32 recurrent matrices. Sparse-attention history is
// replayable input context and is intentionally outside the restore blob.
type State struct {
	Recurrent [3][HiddenSize * HiddenSize]float32
}

func (s State) MarshalBinary() []byte {
	b := make([]byte, 0, 3*HiddenSize*HiddenSize*4)
	var word [4]byte
	for _, layer := range s.Recurrent {
		for _, value := range layer {
			binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
			b = append(b, word[:]...)
		}
	}
	return b
}

func (s *State) UnmarshalBinary(b []byte) error {
	if len(b) != 3*HiddenSize*HiddenSize*4 {
		return fmt.Errorf("qwen4exp state: got %d bytes, want %d", len(b), 3*HiddenSize*HiddenSize*4)
	}
	off := 0
	for layer := range s.Recurrent {
		for i := range s.Recurrent[layer] {
			s.Recurrent[layer][i] = math.Float32frombits(binary.LittleEndian.Uint32(b[off:]))
			off += 4
		}
	}
	return nil
}

// Oracle executes the cadence without an external runtime or fallback.
type Oracle struct {
	state   State
	history [][HiddenSize]float32
}

func New() *Oracle                { return &Oracle{} }
func (o *Oracle) State() State    { return o.state }
func (o *Oracle) Restore(s State) { o.state = s }

func (o *Oracle) Run(tokenIDs []int) (Trace, error) {
	if len(tokenIDs) == 0 {
		return Trace{}, errors.New("qwen4exp oracle: empty token sequence")
	}
	trace := Trace{Engine: Engine, SourceRevision: SourceRevision, CheckpointRevision: CheckpointRevision, StateLayout: StateLayout, Tokens: make([]TokenTrace, 0, len(tokenIDs))}
	for _, tokenID := range tokenIDs {
		x := embedding(tokenID)
		tt := TokenTrace{TokenID: tokenID, Layers: make([]LayerTrace, 0, len(Cadence))}
		for layer, kind := range Cadence {
			lt := LayerTrace{Kind: kind}
			if kind == GatedDelta {
				q, k, v := project(x, layer, 1), project(x, layer, 2), project(x, layer, 3)
				lt.Projection = ProjectionTrace{Q: slice(q), K: slice(k), V: slice(v)}
				x = gatedDelta(x, q, k, v, project(x, layer, 4), project(x, layer, 5), project(x, layer, 6), &o.state.Recurrent[layer])
				lt.Recurrent = slice16(o.state.Recurrent[layer])
			} else {
				q, k, v := project(x, layer, 1), project(x, layer, 2), project(x, layer, 3)
				lt.Projection = ProjectionTrace{Q: slice(q), K: slice(k), V: slice(v)}
				o.history = append(o.history, x)
				lt.SelectedToken = selectSparseTokens(o.history)
				x = sparseAttention(x, q, o.history, lt.SelectedToken, layer)
			}
			ids, weights := route(x, layer)
			lt.Route = RouteTrace{ExpertIDs: ids, ExpertWeights: weights, SharedExpert: true}
			x = moe(x, ids, weights, layer)
			lt.Output = slice(x)
			tt.Layers = append(tt.Layers, lt)
		}
		trace.Tokens = append(trace.Tokens, tt)
	}
	return trace, nil
}

func embedding(token int) [HiddenSize]float32 {
	var out [HiddenSize]float32
	for i := range out {
		out[i] = f32(math.Sin(float64((token+1)*(i+3))) * 0.25)
	}
	return out
}

func project(x [HiddenSize]float32, layer, projection int) [HiddenSize]float32 {
	var out [HiddenSize]float32
	for row := range out {
		for col := range x {
			w := f32(math.Sin(float64((layer+1)*101+(projection+1)*37+(row+1)*11+(col+1)*3)) * 0.19)
			out[row] = f32(float64(out[row]) + float64(x[col]*w))
		}
	}
	return out
}

func gatedDelta(residual, q, k, v, z, a, b [HiddenSize]float32, state *[HiddenSize * HiddenSize]float32) [HiddenSize]float32 {
	alpha := sigmoid(mean(a))
	beta := sigmoid(mean(b))
	for row := 0; row < HiddenSize; row++ {
		prediction := float32(0)
		for col := 0; col < HiddenSize; col++ {
			prediction = f32(float64(prediction) + float64(state[row*HiddenSize+col]*k[col]))
		}
		delta := f32(float64(beta * (v[row] - prediction)))
		for col := 0; col < HiddenSize; col++ {
			state[row*HiddenSize+col] = f32(float64(alpha*state[row*HiddenSize+col]) + float64(delta*k[col]))
		}
	}
	out := residual
	for row := 0; row < HiddenSize; row++ {
		y := float32(0)
		for col := 0; col < HiddenSize; col++ {
			y = f32(float64(y) + float64(state[row*HiddenSize+col]*q[col]))
		}
		out[row] = f32(float64(out[row]) + float64(y*sigmoid(z[row])))
	}
	return out
}

type scoredID struct {
	id    int
	score float32
}

func selectSparseTokens(history [][HiddenSize]float32) []int {
	scored := make([]scoredID, len(history))
	for id, x := range history {
		score := float32(0)
		for i, value := range x {
			score = f32(float64(score) + float64(value*f32(math.Cos(float64((i+1)*17))*0.31)))
		}
		scored[id] = scoredID{id, score}
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score == scored[j].score {
			return scored[i].id < scored[j].id
		}
		return scored[i].score > scored[j].score
	})
	if len(scored) > SparseSelectionCapacity {
		scored = scored[:SparseSelectionCapacity]
	}
	ids := make([]int, len(scored))
	for i := range scored {
		ids[i] = scored[i].id
	}
	return ids
}

func sparseAttention(residual, q [HiddenSize]float32, history [][HiddenSize]float32, ids []int, layer int) [HiddenSize]float32 {
	out := residual
	if len(ids) == 0 {
		return out
	}
	for _, id := range ids {
		k := project(history[id], layer, 2)
		v := project(history[id], layer, 3)
		score := dot(q, k) / float32(math.Sqrt(HiddenSize))
		weight := sigmoid(score) / float32(len(ids))
		for i := range out {
			out[i] = f32(float64(out[i]) + float64(weight*v[i]))
		}
	}
	return out
}

func route(x [HiddenSize]float32, layer int) ([]int, []float32) {
	all := make([]scoredID, NumRoutedExperts)
	for expert := range all {
		score := float32(0)
		for i, value := range x {
			w := f32(math.Sin(float64((layer+1)*1009+(expert+1)*17+(i+1)*5)) * 0.23)
			score = f32(float64(score) + float64(value*w))
		}
		all[expert] = scoredID{expert, score}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].score == all[j].score {
			return all[i].id < all[j].id
		}
		return all[i].score > all[j].score
	})
	all = all[:ExpertsPerToken]
	ids := make([]int, ExpertsPerToken)
	weights := make([]float32, ExpertsPerToken)
	max, sum := all[0].score, float32(0)
	for i, item := range all {
		ids[i] = item.id
		weights[i] = f32(math.Exp(float64(item.score - max)))
		sum = f32(float64(sum) + float64(weights[i]))
	}
	for i := range weights {
		weights[i] = f32(float64(weights[i] / sum))
	}
	return ids, weights
}

func moe(x [HiddenSize]float32, ids []int, weights []float32, layer int) [HiddenSize]float32 {
	out := x
	for dim := range out {
		shared := f32(math.Tanh(float64(x[dim])) * 0.1)
		mixed := float32(0)
		for i, id := range ids {
			mixed = f32(float64(mixed) + float64(weights[i]*f32(math.Tanh(float64(x[dim]+float32(id%13)*0.01))*0.07)))
		}
		out[dim] = f32(float64(out[dim]) + float64(shared+mixed) + float64(layer+1)*0.0001)
	}
	return out
}

func dot(a, b [HiddenSize]float32) float32 {
	var s float32
	for i := range a {
		s = f32(float64(s) + float64(a[i]*b[i]))
	}
	return s
}
func mean(a [HiddenSize]float32) float32 {
	var s float32
	for _, v := range a {
		s = f32(float64(s) + float64(v))
	}
	return s / HiddenSize
}
func sigmoid(x float32) float32                            { return f32(1 / (1 + math.Exp(-float64(x)))) }
func f32(v float64) float32                                { return float32(v) }
func slice(a [HiddenSize]float32) []float32                { return append([]float32(nil), a[:]...) }
func slice16(a [HiddenSize * HiddenSize]float32) []float32 { return append([]float32(nil), a[:]...) }
func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
