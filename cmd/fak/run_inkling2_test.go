package main

import (
	"errors"
	"flag"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestRunInkling2RawPreservesPromptBytes(t *testing.T) {
	fs, flags := newRunFlagSet("run", flag.ContinueOnError)
	cmd, err := parseRunArgs(fs, flags, []string{"model.gguf", "--raw", "--timings", "--backend", "cpu", "   def f():\n"})
	if err != nil || cmd.action != runActionChat || !cmd.chatConfig.raw || !cmd.chatConfig.timings || cmd.chatConfig.prompt != "   def f():\n" {
		t.Fatalf("prompt %q raw %v timings %v, err %v", cmd.chatConfig.prompt, cmd.chatConfig.raw, cmd.chatConfig.timings, err)
	}
}

func TestRunInkling2PackedWeightPlan(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(path, make([]byte, 128), 0600); err != nil {
		t.Fatal(err)
	}
	f := &ggufload.File{Tensors: []ggufload.TensorInfo{
		{Name: "blk.0.ffn_gate_up_exps.weight", Dims: []uint64{16, 16, 4}},
		{Name: "output.weight", Dims: []uint64{16, 64}},
		{Name: "blk.0.attn_rel_bias.weight", Dims: []uint64{4, 8}},
		{Name: "blk.6.loop_q_ctrl_down.weight", Dims: []uint64{4608, 64}},
		{Name: "blk.6.loop_k_ctrl_gain.weight", Dims: []uint64{64, 2304}},
		{Name: "blk.6.loop_ffn_ctrl_forget.weight", Dims: []uint64{64, 2304}},
		{Name: "blk.6.loop_msg_query.weight", Dims: []uint64{4, 8}},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{2304, 2304}},
		{Name: "blk.0.attn_k_conv_proj.weight", Dims: []uint64{2304, 24}},
		{Name: "blk.0.ffn_gate_inp.weight", Dims: []uint64{2304, 256}},
		{Name: "blk.0.future.weight", Dims: []uint64{2, 8}},
	}}
	p, err := inkling2RunMetalMemory(path, f)
	if err != nil || p.SteadyBytes != 128+2*4*4*8+2*8*4 || p.StartupPeakBytes < p.SteadyBytes {
		t.Fatalf("plan %+v, err %v", p, err)
	}
	f.Tensors[2].Dims = []uint64{math.MaxInt64, 2}
	if _, err := inkling2RunMetalMemory(path, f); err == nil {
		t.Fatal("overflow accepted")
	}
}

func TestRunInkling2ExplicitPackedMetal(t *testing.T) {
	t.Setenv("FAK_METAL", "")
	t.Setenv("FAK_BACKEND", "metal")
	for _, name := range []string{"", "metal", "cpu"} {
		be, metal, err := inkling2RunBackend(name, false)
		if name != "cpu" && !metalgemm.Available() {
			if !errors.Is(err, metalgemm.ErrMetalUnavailable) {
				t.Fatalf("backend %q: %v", name, err)
			}
			continue
		}
		if err != nil || be != nil || metal != (name != "cpu") {
			t.Fatalf("backend %q: selected=%v metal=%v error=%v", name, be, metal, err)
		}
	}
}
