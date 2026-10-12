package computetrace

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// The golden was written by the converter before the stage map became a
// registry; the deepseek41 map must keep reproducing it byte for byte.
func TestLlamaCppDeepSeek41ByteIdentical(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "llamacpp-deepseek41-golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, convert := range []func(string) (Artifact, error){
		ConvertLlamaCppDump,
		func(dir string) (Artifact, error) { return ConvertLlamaCppDumpArch(dir, LlamaCppArchDeepSeek41) },
	} {
		a, err := convert(filepath.Join("testdata", "llamacpp-deepseek41-dump"))
		if err != nil {
			t.Fatal(err)
		}
		got, _ := json.Marshal(a)
		if !bytes.Equal(append(got, '\n'), want) {
			t.Fatalf("deepseek41 conversion drifted from the golden (%d vs %d bytes)", len(got)+1, len(want))
		}
	}
	if _, err := ConvertLlamaCppDumpArch(filepath.Join("testdata", "llamacpp-deepseek41-dump"), "llama"); !errors.Is(err, ErrUnknownStageMap) {
		t.Fatalf("unknown arch err=%v", err)
	}
}

// qwen4expCbNames is every cb() name in Nathanw1014/llama.cpp@dff6004
// src/models/qwen4exp.cpp plus the GDN base class it calls,
// src/models/delta-net-base.cpp build_recurrent_attn (attn_output, new_state).
var qwen4expCbNames = []string{
	"hc_norm", "hc_gate", "hc_mixed", "hc_inject", "hc_combine", "model.input_embed", "hc_init",
	"ffn_out", "l_last", "h_nextn", "result_norm", "result_output", "linear_attn_qkv_mixed", "z",
	"indexer_k_raw", "indexer_k_pooled", "indexer_k", "indexer_q", "indexer_score",
	"indexer_score_tokens", "indexer_top_k", "kqv_out", "Qcur_full", "Qcur_reshaped",
	"Qcur_normed", "Kcur", "Vcur", "Kcur_normed", "gate_reshaped", "Qcur", "attn_pregate",
	"gate_sigmoid", "attn_gated", "attn_output", "beta", "beta_sigmoid", "alpha", "a_softplus",
	"gate", "state_predelta", "conv_output_raw", "conv_output_silu", "q_conv", "k_conv", "v_conv",
	"q_conv_predelta", "k_conv_predelta", "v_conv_predelta", "final_output", "linear_attn_out",
	"ffn_moe_out", "ffn_shexp", "shared_expert_gate", "shared_expert_gate_sigmoid",
	"ffn_shexp_gated", "conv_state_at", "ple_embd", "ple_gate", "ple_gated_value", "ple_conv_out",
	"mtp_tok_embd", "mtp_hnorm", "mtp_enorm", "mtp_eh_proj", "mtp_ffn_out", "new_state",
}

func TestLlamaCppQwen4ExpCoversEveryCbName(t *testing.T) {
	m, ok := LlamaCppStageMapFor(LlamaCppArchQwen4Exp)
	if !ok {
		t.Fatal("qwen4exp stage map missing")
	}
	want := slices.Clone(qwen4expCbNames)
	slices.Sort(want)
	want = slices.Compact(want)
	if got := m.Names(); !slices.Equal(got, want) {
		t.Fatalf("stage map names differ from dff6004 cb() names:\n got %v\nwant %v", got, want)
	}
	for _, t2 := range []map[string]LlamaCppStage{m.Layered, m.Pre, m.Post} {
		for n := range t2 {
			if _, dup := m.Unmapped[n]; dup {
				t.Fatalf("%s is both mapped and unmapped", n)
			}
		}
	}
	if got := LlamaCppStageMapArches(); !slices.Equal(got, []string{"deepseek41", "qwen4exp"}) {
		t.Fatalf("arches=%v", got)
	}
}

const (
	qwenFixtureLayers = 48
	qwenFixtureEmbd   = 16
	qwenFixtureHC     = 4
)

// writeQwen4ExpDump writes a dff6004-shaped dump: 48 layers on the 3 GDN : 1
// full-attention cadence, PLE every fourth layer, each mix/RoPE name emitted
// twice as the reference does, and the last layer's post-attention stages plus
// the post-stack outputs gathered down to the final token.
func writeQwen4ExpDump(t *testing.T, tokens []int) string {
	t.Helper()
	nt := int64(len(tokens))
	H, hc := int64(qwenFixtureEmbd), int64(qwenFixtureHC)
	var ts []dumpTensor
	add := func(name string, ne ...int64) {
		for len(ne) < 4 {
			ne = append(ne, 1)
		}
		ts = append(ts, dumpTensor{name: name, ne: [4]int64{ne[0], ne[1], ne[2], ne[3]}, dumped: true})
	}
	add("model.input_embed", H, nt)
	add("hc_init", H, hc, nt)
	for il := 0; il < qwenFixtureLayers; il++ {
		n := func(base string) string { return fmt.Sprintf("%s-%d", base, il) }
		if il%4 == 0 {
			add(n("ple_embd"), 8, nt)
			add(n("ple_gate"), 1, hc, nt)
			add(n("ple_gated_value"), H, hc, nt)
			add(n("ple_conv_out"), H, hc, nt)
		}
		add(n("hc_norm"), H*hc, nt)
		add(n("hc_mixed"), H, nt)
		if il%4 == 3 {
			add(n("Qcur_full"), 2*8*2, nt)
			add(n("Qcur_normed"), 8, 2, nt)
			add(n("Kcur_normed"), 8, 1, nt)
			add(n("gate_reshaped"), 8*2, nt)
			add(n("Qcur"), 8, 2, nt)
			add(n("Qcur"), 8, 2, nt)
			add(n("attn_pregate"), 8*2, nt)
			add(n("gate_sigmoid"), 8*2, nt)
			add(n("attn_gated"), 8*2, nt)
			add(n("attn_output"), H, nt)
		} else {
			add(n("linear_attn_qkv_mixed"), 24, nt, 1)
			add(n("z"), 8, nt)
			add(n("attn_output"), 4, 2, nt, 1) // delta-net-base: the raw GDN core, not the projection
			add(n("final_output"), 8, nt, 1)
			add(n("linear_attn_out"), H, nt, 1)
		}
		rows := nt
		if il == qwenFixtureLayers-1 {
			rows = 1
		}
		add(n("hc_combine"), H, hc, rows)
		add(n("hc_mixed"), H, rows)
		add(n("ffn_moe_out"), H, rows)
		add(n("ffn_shexp"), H, rows)
		add(n("shared_expert_gate"), 1, rows)
		add(n("shared_expert_gate_sigmoid"), 1, rows)
		add(n("ffn_shexp_gated"), H, rows)
		add(n("ffn_out"), H, rows)
		add(n("hc_combine"), H, hc, rows)
		add(n("l_last"), H, hc, rows)
	}
	add("result_norm", H, 1)
	add("result_output", 32, 1)
	return writeDump(t, tokens, ts)
}

func TestLlamaCppQwen4ExpMapsAll48Layers(t *testing.T) {
	tokens := []int{11, 12, 13}
	a, err := ConvertLlamaCppDumpArch(writeQwen4ExpDump(t, tokens), LlamaCppArchQwen4Exp)
	if err != nil {
		t.Fatal(err)
	}
	type key struct {
		layer int
		stage string
		token int
	}
	got := map[key]Event{}
	stagesByLayer := map[int]map[string]bool{}
	for _, e := range a.Events {
		k := key{e.Layer, e.Phase, e.Token}
		if _, dup := got[k]; dup {
			t.Fatalf("duplicate identity %+v", k)
		}
		got[k] = e
		if stagesByLayer[e.Layer] == nil {
			stagesByLayer[e.Layer] = map[string]bool{}
		}
		stagesByLayer[e.Layer][e.Phase] = true
	}
	for il := 0; il < qwenFixtureLayers; il++ {
		for _, stage := range []string{"attention_projected", "moe_sum", "residual_streams", "moe_routed", "shared_expert_gated"} {
			if !stagesByLayer[il][stage] {
				t.Fatalf("layer %d has no %s", il, stage)
			}
		}
		attn := "gdn_core_normed"
		if il%4 == 3 {
			attn = "attn_gated"
		}
		if !stagesByLayer[il][attn] {
			t.Fatalf("layer %d has no %s", il, attn)
		}
		if il < qwenFixtureLayers-1 {
			for tok := range tokens {
				if _, ok := got[key{il, "residual_streams", tok}]; !ok {
					t.Fatalf("layer %d residual_streams token %d missing", il, tok)
				}
			}
		}
	}
	if len(stagesByLayer) != qwenFixtureLayers+2 {
		t.Fatalf("layers=%d, want 48 plus the pre and post rows", len(stagesByLayer))
	}
	if e, ok := got[key{-1, "residual_streams", 2}]; !ok || e.Shapes[0][0] != qwenFixtureEmbd*qwenFixtureHC {
		t.Fatalf("hc_init row: %+v", e)
	}
	if _, ok := got[key{-1, "embedding", 0}]; !ok {
		t.Fatal("embedding missing")
	}
	q := got[key{3, "attn_q_normed", 0}]
	if q.Shapes[0][0] != 16 || len(q.ExcludedSampleIndices) != 0 {
		t.Fatalf("attn_q_normed shape=%v excluded=%v", q.Shapes, q.ExcludedSampleIndices)
	}
	// The last layer's gathered rows and the post-stack rows are the final token.
	last := qwenFixtureLayers - 1
	for _, k := range []key{{last, "residual_streams", 2}, {last, "moe_sum", 2}, {qwenFixtureLayers, "logits", 2}, {qwenFixtureLayers, "final_norm", 2}} {
		if _, ok := got[k]; !ok {
			t.Fatalf("tail-aligned row %+v missing", k)
		}
	}
	if _, ok := got[key{last, "residual_streams", 0}]; ok {
		t.Fatal("last-layer gathered row numbered from token 0")
	}
	if _, ok := got[key{last, "attention_projected", 0}]; !ok {
		t.Fatal("last-layer attention runs over every token and must not be tail aligned")
	}
}

func TestLlamaCppQwen4ExpSkipsEmptyGatheredUbatch(t *testing.T) {
	// A two-ubatch prompt with last-token logits: the first ubatch gathers zero
	// output rows on the last layer and after the stack.
	H := int64(qwenFixtureEmbd)
	dir := writeDump(t, []int{1, 2, 3}, []dumpTensor{
		{name: "linear_attn_out-0", ne: [4]int64{H, 2, 1, 1}, dumped: true},
		{name: "ffn_out-0", ne: [4]int64{H, 0, 1, 1}, dumped: true},
		{name: "result_output", ne: [4]int64{32, 0, 1, 1}, dumped: true},
		{name: "linear_attn_out-0", ne: [4]int64{H, 1, 1, 1}, dumped: true},
		{name: "ffn_out-0", ne: [4]int64{H, 1, 1, 1}, dumped: true},
		{name: "result_output", ne: [4]int64{32, 1, 1, 1}, dumped: true},
	})
	a, err := ConvertLlamaCppDumpArch(dir, LlamaCppArchQwen4Exp)
	if err != nil {
		t.Fatal(err)
	}
	type key struct {
		layer int
		stage string
		token int
	}
	got := map[key]bool{}
	for _, e := range a.Events {
		got[key{e.Layer, e.Phase, e.Token}] = true
	}
	want := []key{{0, "attention_projected", 0}, {0, "attention_projected", 1}, {0, "attention_projected", 2}, {0, "moe_sum", 2}, {1, "logits", 2}}
	for _, k := range want {
		if !got[k] {
			t.Fatalf("missing %+v in %v", k, got)
		}
	}
	if len(a.Events) != len(want) {
		t.Fatalf("events=%d, want %d", len(a.Events), len(want))
	}
	// A zero-row tensor on a stage that is not tail aligned is still malformed.
	bad := writeDump(t, []int{1}, []dumpTensor{{name: "linear_attn_out-0", ne: [4]int64{H, 0, 1, 1}, dumped: true}})
	if _, err := ConvertLlamaCppDumpArch(bad, LlamaCppArchQwen4Exp); !errors.Is(err, ErrLlamaCppDump) {
		t.Fatalf("zero-row attention err=%v", err)
	}
	// attn_output without a dumped linear_attn_out is the full-attention projection.
	clash := writeDump(t, []int{1}, []dumpTensor{
		{name: "attn_output-3", ne: [4]int64{H, 1, 1, 1}, dumped: true},
		{name: "linear_attn_out-3", ne: [4]int64{H, 1, 1, 1}, dumped: false},
		{name: "ffn_out-3", ne: [4]int64{H, 1, 1, 1}, dumped: true},
		{name: "ffn_moe_out-3", ne: [4]int64{H, 1, 1, 1}, dumped: true},
	})
	if _, err := ConvertLlamaCppDumpArch(clash, LlamaCppArchQwen4Exp); err != nil {
		t.Fatalf("an undumped superseding name must not suppress attn_output: %v", err)
	}
}
