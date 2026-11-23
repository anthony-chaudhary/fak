package ggufload

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2DSparkTestCheckpoint builds an mtp.safetensors image holding every
// draft tensor at the given geometry, BF16 values 1.0, plus extra tensors.
func inkling2DSparkTestCheckpoint(t *testing.T, H, layers, heads, kvHeads, dim, inter, rank, vocab, taps int, drop string, extra ...string) []byte {
	t.Helper()
	shapes := map[string][]int{
		"norm.weight": {H}, "mask_embed": {H}, "fuse_weight": {layers + 1, taps},
		"markov_head.markov_w1.weight": {vocab, rank}, "markov_head.rnn_proj.weight": {3 * rank, 2*rank + 2*H},
		"markov_head.rnn_proj.bias": {3 * rank}, "markov_head.hidden_out.weight": {H, rank},
		"markov_head.hidden_out.bias": {H}, "markov_head.dflare_norm.weight": {H},
	}
	for l := 0; l < layers; l++ {
		p := fmt.Sprintf("layers.%d.", l)
		for _, n := range []string{"input_layernorm", "post_attention_layernorm", "dflare_norm"} {
			shapes[p+n+".weight"] = []int{H}
		}
		for _, n := range []string{"q_norm", "k_norm", "k_norm_ctx"} {
			shapes[p+"self_attn."+n+".weight"] = []int{dim}
		}
		shapes[p+"self_attn.q_proj.weight"] = []int{heads * dim, H}
		shapes[p+"self_attn.o_proj.weight"] = []int{H, heads * dim}
		for _, n := range []string{"k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"} {
			shapes[p+"self_attn."+n+".weight"] = []int{kvHeads * dim, H}
		}
		shapes[p+"mlp.gate_proj.weight"] = []int{inter, H}
		shapes[p+"mlp.up_proj.weight"] = []int{inter, H}
		shapes[p+"mlp.down_proj.weight"] = []int{H, inter}
	}
	for _, e := range extra {
		shapes[e] = []int{1}
	}
	delete(shapes, drop)
	names := make([]string, 0, len(shapes))
	for n := range shapes {
		names = append(names, n)
	}
	sort.Strings(names)
	hdr := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var payload []byte
	for _, n := range names {
		count := 1
		for _, d := range shapes[n] {
			count *= d
		}
		start := len(payload)
		for i := 0; i < count; i++ {
			payload = binary.LittleEndian.AppendUint16(payload, 0x3F80)
		}
		hdr[n] = map[string]any{"dtype": "BF16", "shape": shapes[n], "data_offsets": []int{start, len(payload)}}
	}
	js, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(js)))
	return append(append(out, js...), payload...)
}

func TestInkling2DSparkBindsCheckpointAndRefusesDrift(t *testing.T) {
	const H, layers, heads, kvHeads, dim, inter, rank, vocab = 8, 3, 2, 1, 4, 12, 2, 10
	var cfg inkling2DSparkConfig
	cfgJSON := `{"text_config":{"unpadded_vocab_size":9},"mtp_config":{"hidden_size":8,"num_hidden_layers":3,
		"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"intermediate_size":12,"rms_norm_eps":1e-6,
		"rope_theta":1000000.0,"local_layer_ids":[0,2],"sliding_window_size":1024,"block_size":8,"markov_rank":2,
		"markov_head_type":"rnn_hidden","target_layer_ids":[1,3,4]}}`
	if err := json.Unmarshal([]byte(cfgJSON), &cfg); err != nil {
		t.Fatal(err)
	}
	target := &Inkling2LoopModel{
		HP:      model.Inkling2LoopHParams{Hidden: H, Layers: 2, CoreStart: 1, CoreEnd: 2, Passes: 3, VocabUnpadded: 9},
		Weights: &model.Inkling2LoopWeights{},
	}
	data := inkling2DSparkTestCheckpoint(t, H, layers, heads, kvHeads, dim, inter, rank, vocab, 3, "")
	m, err := bindInkling2DSpark(&cfg, data, target)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(m.TargetVisits, m.HP.LocalLayer, m.HP.Taps, m.HP.Block, m.HP.VocabUnpadded); got != "[1 3 4] [true false true] 3 8 9" {
		t.Fatalf("bound geometry = %s", got)
	}
	if _, err := model.NewInkling2DSparkDraft(m.HP, m.Weights); err != nil {
		t.Fatalf("bound weights rejected by the draft: %v", err)
	}
	if m.Weights.MarkovW1 == nil || m.Weights.RnnProj.Rows() != 3*rank || len(m.Weights.FuseWeight) != layers+1 {
		t.Fatalf("markov head / fuse rows not bound: %+v", m.Weights)
	}

	for name, data := range map[string][]byte{
		"missing tensor layers.1.self_attn.k_proj_ctx.weight": inkling2DSparkTestCheckpoint(t, H, layers, heads, kvHeads, dim, inter, rank, vocab, 3, "layers.1.self_attn.k_proj_ctx.weight"),
		"unbound checkpoint tensor confidence_head.weight":    inkling2DSparkTestCheckpoint(t, H, layers, heads, kvHeads, dim, inter, rank, vocab, 3, "", "confidence_head.weight"),
	} {
		if _, err := bindInkling2DSpark(&cfg, data, target); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("err = %v, want %q", err, name)
		}
	}
	cfg.MTPConfig.TargetLayerIDs = []int{1, 3}
	if _, err := bindInkling2DSpark(&cfg, data, target); err == nil || !strings.Contains(err.Error(), "4-visit") {
		t.Fatalf("target visit drift: err = %v", err)
	}
}

// The draft's greedy acceptance against the target's greedy continuation on
// the llama.cpp oracle prompts: the speculative-decode prize available to the
// CUDA path at this checkpoint.
func TestInkling2DSparkAcceptanceOnOraclePrompts(t *testing.T) {
	path, dir := os.Getenv("FAK_INKLING2_GGUF"), os.Getenv("FAK_INKLING2_MTP_DIR")
	if path == "" || dir == "" {
		t.Skip("FAK_INKLING2_GGUF and FAK_INKLING2_MTP_DIR (HF snapshot holding mtp.safetensors) not set")
	}
	m, err := OpenInkling2Loop(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	dm, err := OpenInkling2DSpark(dir, m)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	gamma := dm.HP.Block - 1
	var blocks, drafted, steps, emitted int
	depth := make([]int, gamma+1)
	for _, c := range inkling2OracleCases {
		start := time.Now()
		tokens := append(append([]int(nil), c.prompt...), c.greedy...)
		accepted, err := Inkling2DSparkTeacherForced(m, dm, tokens, len(c.prompt))
		if err != nil {
			t.Fatal(err)
		}
		caseSum := 0
		for _, n := range accepted {
			caseSum += n
			depth[n]++
		}
		// Greedy speculative decode commits accepted+1 tokens per verify.
		caseSteps, caseEmitted := 0, 0
		for i := 0; i < len(accepted); i += accepted[i] + 1 {
			caseSteps++
			caseEmitted += accepted[i] + 1
		}
		t.Logf("%s: %d blocks, accepted %v, mean %.2f/%d, greedy walk %d tokens in %d verifies (%.2f tok/verify), %s",
			c.name, len(accepted), accepted, float64(caseSum)/float64(len(accepted)), gamma,
			caseEmitted, caseSteps, float64(caseEmitted)/float64(caseSteps), time.Since(start))
		blocks += len(accepted)
		drafted += caseSum
		steps += caseSteps
		emitted += caseEmitted
	}
	mean := float64(drafted) / float64(blocks)
	t.Logf("all: %d blocks, mean accepted %.3f of %d, accept-length histogram %v, greedy walk %.3f tok/verify",
		blocks, mean, gamma, depth, float64(emitted)/float64(steps))
	// A mis-wired draft (wrong taps, RoPE or Markov order) accepts near zero.
	if mean < inkling2DSparkMinMeanAccepted {
		t.Fatalf("mean accepted %.3f below %.2f", mean, inkling2DSparkMinMeanAccepted)
	}
}

const inkling2DSparkMinMeanAccepted = 1.0
