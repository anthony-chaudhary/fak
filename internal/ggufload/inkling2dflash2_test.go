package ggufload

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const inkling2DFlash2DefaultGGUF = "models/inkling2/Inkling-2-Mini-UD-Q3_K_XL.gguf"

// inkling2DFlash2TestDir is FAK_INKLING2_DFLASH2_DIR, else the draft beside
// FAK_INKLING2_GGUF, else beside the default model path; "" when absent.
func inkling2DFlash2TestDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("FAK_INKLING2_DFLASH2_DIR"); dir != "" {
		return dir
	}
	gguf := os.Getenv("FAK_INKLING2_GGUF")
	if gguf == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		gguf = filepath.Join(home, inkling2DFlash2DefaultGGUF)
	}
	dir, err := Inkling2DFlash2AdjacentDir(gguf)
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestInkling2DFlash2CheckpointShapes(t *testing.T) {
	dir := inkling2DFlash2TestDir(t)
	if dir == "" {
		t.Skip("no Inkling-2-Mini-DFlash2 directory (set FAK_INKLING2_DFLASH2_DIR or FAK_INKLING2_GGUF)")
	}
	m, err := OpenInkling2DFlash2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Tensors != 81 || len(Inkling2DFlash2TensorShapes(&m.Config)) != 81 || len(m.layers) != 5 {
		t.Fatalf("bound %d tensors, %d layers; want 81, 5", m.Tensors, len(m.layers))
	}
	// Norm multipliers trained from 1 stay finite and mostly positive; a
	// mis-sliced payload reads as noise.
	for name, v := range map[string][]float32{"hidden_norm": m.hiddenNorm, "norm": m.norm, "layers.0.k_norm": m.layers[0].kNorm} {
		var sum float64
		for _, x := range v {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("%s holds %v", name, x)
			}
			sum += float64(x)
		}
		if sum/float64(len(v)) <= 0 {
			t.Fatalf("%s mean %.4f not positive", name, sum/float64(len(v)))
		}
	}
	if m.fc.Rows() != 2304 || m.fc.Cols() != 9216 || m.predCB.Rows() != 201024 || m.layers[4].attnDyn.Rows() != 576 {
		t.Fatalf("fc %dx%d, codebook %d rows, dyn %d rows", m.fc.Rows(), m.fc.Cols(), m.predCB.Rows(), m.layers[4].attnDyn.Rows())
	}
}

func TestInkling2DFlash2ConfigRefusesDrift(t *testing.T) {
	good := `{"architectures":["DFlash2DraftModel"],"num_hidden_layers":5,"intermediate_size":6912,"hidden_size":2304,
		"vocab_size":201024,"head_dim":128,"num_attention_heads":32,"num_key_value_heads":8,"sliding_window":2048,
		"is_causal":false,"attention_bias":false,"hidden_act":"silu","rms_norm_eps":1e-6,"tie_word_embeddings":false,
		"num_target_layers":48,"rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
		"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"final_logit_softcapping":30.0,
		"mask_token_id":200056,"selector_rank":256,"selector_softcap_mode":"after_correction","selector_top_k":16,
		"target_layer_ids":[5,23,41,48]}}`
	var c Inkling2DFlash2Config
	if err := json.Unmarshal([]byte(good), &c); err != nil {
		t.Fatal(err)
	}
	if err := c.validate(); err != nil {
		t.Fatal(err)
	}
	for _, drift := range []func(*Inkling2DFlash2Config){
		func(c *Inkling2DFlash2Config) { c.DFlash.Taps = []int{5, 23, 41, 47} },
		func(c *Inkling2DFlash2Config) { c.Causal = true },
		func(c *Inkling2DFlash2Config) { c.DFlash.SoftcapMode = "before_correction" },
		func(c *Inkling2DFlash2Config) { c.Window = 4096 },
		func(c *Inkling2DFlash2Config) { c.Architectures = []string{"DFlashDraftModel"} },
	} {
		d := c
		d.DFlash.Taps = slices.Clone(c.DFlash.Taps)
		drift(&d)
		if err := d.validate(); err == nil {
			t.Fatalf("drifted config accepted: %+v", d)
		}
	}
}

// inkling2DFlash2TinySafetensors lays out BF16 tensors in the given order.
func inkling2DFlash2TinySafetensors(t *testing.T, names []string, shapes map[string][]int, gap int) *inkling2Safetensors {
	t.Helper()
	hdr := map[string]any{}
	at := 0
	for _, n := range names {
		count := 2
		for _, d := range shapes[n] {
			count *= d
		}
		hdr[n] = map[string]any{"dtype": "BF16", "shape": shapes[n], "data_offsets": []int{at, at + count}}
		at += count + gap
	}
	js, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(js)))
	data = append(append(data, js...), make([]byte, at)...)
	st, err := parseInkling2Safetensors(data)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func TestInkling2DFlash2TensorCheckRefusesDrift(t *testing.T) {
	want := map[string][]int{"a": {2, 3}, "b": {4}}
	if err := checkInkling2DFlash2Tensors(inkling2DFlash2TinySafetensors(t, []string{"a", "b"}, want, 0), want); err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		names  []string
		shapes map[string][]int
		gap    int
	}{
		"missing tensor b":      {[]string{"a"}, want, 0},
		"unexpected checkpoint": {[]string{"a", "b", "c"}, map[string][]int{"a": {2, 3}, "b": {4}, "c": {1}}, 0},
		"want BF16[2 3]":        {[]string{"a", "b"}, map[string][]int{"a": {3, 2}, "b": {4}}, 0},
		"starts at":             {[]string{"a", "b"}, want, 2},
	} {
		err := checkInkling2DFlash2Tensors(inkling2DFlash2TinySafetensors(t, tc.names, tc.shapes, tc.gap), want)
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("err = %v, want %q", err, name)
		}
	}
}

func TestInkling2DFlash2ConvAndTop(t *testing.T) {
	const H, group = 4, 2
	in := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	// dyn per row: [stage][tap][group]
	dyn := [][]float32{{0, 0, 0, 0, 0, 0, 0, 0}, {0.5, 0, 1, 0, 0, 0, 0, 0}}
	base := []float32{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}
	got := inkling2DFlash2Conv(in, dyn, base, nil, 0, H, group)
	// row 1 ch 0: 5*(1+0.5) + 1*(2+1) = 10.5; ch 2 (group 1): 7*1 + 3*2 = 13
	if got[0][0] != 1 || got[1][0] != 10.5 || got[1][2] != 13 {
		t.Fatalf("prepare stage = %v", got)
	}
	res := inkling2DFlash2Conv(in, dyn, base, [][]float32{{1, 1, 1, 1}, {1, 1, 1, 1}}, 1, H, group)
	// row 1 ch 0: 5*3 + 1*4 + 1 = 20
	if res[0][0] != 4 || res[1][0] != 20 {
		t.Fatalf("residual stage = %v", res)
	}
	ids, vals := inkling2DFlash2Top([]float32{1, 3, 3, -1, 2, 3}, 3)
	if !slices.Equal(ids, []int{1, 2, 5}) || vals[2] != 3 {
		t.Fatalf("top = %v %v", ids, vals)
	}
	ids, _ = inkling2DFlash2Top([]float32{1, 3, 3, -1, 2, 3}, 5)
	if !slices.Equal(ids, []int{1, 2, 5, 4, 0}) {
		t.Fatalf("top5 = %v", ids)
	}
}

// inkling2DFlash2TeacherForced drafts a full block at every anchor of the
// target's greedy continuation and returns how many leading drafts match it:
// the greedy accepted length speculative decode would realize.
func inkling2DFlash2TeacherForced(target *Inkling2LoopModel, d *Inkling2DFlash2Draft, tokens []int, promptLen int) ([]int, error) {
	h, err := model.NewInkling2LoopHost(target.HP, target.Weights)
	if err != nil {
		return nil, err
	}
	h.FeatureVisits = d.M.Config.DFlash.Taps
	gamma := d.M.Config.DFlash.Block - 1
	var accepted []int
	for p, tok := range tokens {
		if p >= promptLen && p+gamma < len(tokens) {
			b, err := d.Propose(tok, gamma)
			if err != nil {
				return nil, err
			}
			n := 0
			for n < gamma && b.Tokens[n] == tokens[p+1+n] {
				n++
			}
			accepted = append(accepted, n)
		}
		h.PrefillToken(tok, false)
		if err := d.Commit(p, h.Features); err != nil {
			return nil, err
		}
	}
	return accepted, nil
}

// The draft against the target's greedy continuation of the llama.cpp oracle
// prompts. A mis-wired draft (taps, RoPE, conv taps or selector) accepts
// near zero.
func TestInkling2DFlash2AcceptanceOnOraclePrompts(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_GGUF")
	if path == "" {
		t.Skip("FAK_INKLING2_GGUF not set (host target forward, minutes)")
	}
	dir, err := Inkling2DFlash2AdjacentDir(path)
	if err != nil || dir == "" {
		t.Skipf("no %s beside %s (%v)", Inkling2DFlash2DirName, path, err)
	}
	tm, err := OpenInkling2Loop(path)
	if err != nil {
		t.Fatal(err)
	}
	defer tm.Close()
	if v := tm.HP.Layers + (tm.HP.Passes-1)*(tm.HP.CoreEnd-tm.HP.CoreStart); v != 48 {
		t.Fatalf("target has %d visits, the draft reads 48", v)
	}
	dm, err := OpenInkling2DFlash2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	gamma := dm.Config.DFlash.Block - 1
	var blocks, drafted, steps, emitted int
	hist := make([]int, gamma+1)
	for _, c := range inkling2OracleCases[:2] {
		d, err := NewInkling2DFlash2Draft(dm, Inkling2DFlash2TargetOf(tm))
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		tokens := append(append([]int(nil), c.prompt...), c.greedy...)
		accepted, err := inkling2DFlash2TeacherForced(tm, d, tokens, len(c.prompt))
		if err != nil {
			t.Fatal(err)
		}
		sum, cs, ce := 0, 0, 0
		for _, n := range accepted {
			sum += n
			hist[n]++
		}
		for i := 0; i < len(accepted); i += accepted[i] + 1 {
			cs++
			ce += accepted[i] + 1
		}
		t.Logf("%s: %d blocks, accepted %v, mean %.2f/%d, greedy walk %.2f tok/verify, %s",
			c.name, len(accepted), accepted, float64(sum)/float64(len(accepted)), gamma, float64(ce)/float64(cs), time.Since(start))
		blocks += len(accepted)
		drafted += sum
		steps += cs
		emitted += ce
	}
	mean := float64(drafted) / float64(blocks)
	t.Logf("DFlash2 host reference: %d blocks, mean accepted %.3f of %d, histogram %v, greedy walk %.3f tok/verify",
		blocks, mean, gamma, hist, float64(emitted)/float64(steps))
	if mean < 3.0 {
		t.Fatalf("mean accepted %.3f below 3.0", mean)
	}
}
