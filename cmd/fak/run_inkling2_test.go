package main

import (
	"errors"
	"flag"
	"math"
	"os"
	"path/filepath"
	"strings"
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
	if !inkling2RunControllerWeight("blk.6.loop_q_ctrl_down.weight") || inkling2RunControllerWeight("blk.6.loop_msg_query.weight") || inkling2RunControllerWeight("output.weight") {
		t.Fatal("controller projection classification")
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

func TestRunInkling2MetalTargetSelector(t *testing.T) {
	for _, tc := range []struct {
		selector                    string
		loopArch, present, resident bool
		want                        inkling2MetalPath
		errHas                      string
	}{
		{selector: "", loopArch: true, present: true, resident: true, want: inkling2MetalVendored},
		{selector: "", loopArch: true, present: false, resident: true, want: inkling2MetalNative},
		{selector: "", loopArch: false, present: true, want: inkling2MetalNative},
		{selector: " Vendored ", loopArch: true, present: true, want: inkling2MetalVendored},
		{selector: "vendored", loopArch: true, present: false, errHas: "not available"},
		{selector: "vendored", loopArch: false, present: true, errHas: "architecture"},
		{selector: "native", loopArch: true, present: true, resident: true, want: inkling2MetalNative},
		{selector: "NATIVE", loopArch: true, present: false, want: inkling2MetalNative},
		{selector: "resident", loopArch: true, present: true, resident: true, want: inkling2MetalResident},
		{selector: " Resident", loopArch: false, present: false, resident: true, want: inkling2MetalResident},
		{selector: "resident", loopArch: true, present: true, errHas: "device-resident Metal forward is not available"},
		{selector: "splash", loopArch: true, present: true, errHas: `unknown FAK_INKLING_METAL_TARGET="splash" (want vendored, native or resident)`},
	} {
		got, err := inkling2MetalSelect(tc.selector, tc.loopArch, tc.present, tc.resident)
		if tc.errHas != "" {
			if err == nil || !strings.Contains(err.Error(), tc.errHas) {
				t.Errorf("%+v: error %v, want %q", tc, err, tc.errHas)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("%+v: path=%v error=%v", tc, got, err)
		}
	}
}

func TestRunInkling2ResidentCtx(t *testing.T) {
	for _, tc := range []struct {
		v    string
		set  bool
		want int
	}{
		{"", false, 4096}, {"8192", true, 8192}, {" 512 ", true, 512}, {"511", true, 4096},
		{"many", true, 4096}, {"23552", true, 23552}, {"100000", true, 23552},
	} {
		if got := inkling2ResidentCtx(tc.v, tc.set); got != tc.want {
			t.Errorf("ctx(%q, %v) = %d, want %d", tc.v, tc.set, got, tc.want)
		}
	}
}

func TestRunInkling2ResidentSessionBytes(t *testing.T) {
	// Four layers, the middle two revisited once: six visits, of which the
	// local layer 1 is visited twice.
	s := inkling2ResidentShape{Hidden: 8, Heads: 4, KVHeads: 2, HeadDim: 4, SlidingWindow: 16, ConvKernel: 3,
		CoreStart: 1, CoreEnd: 3, Passes: 2, Local: []bool{false, true, false, true}}
	page := func(floats int64) int64 { return (4*floats + 16383) / 16384 * 16384 }
	const ctx = 8192
	global, local := 2*page(ctx*8), 2*page(16*8)
	want := 3*global + 3*local + 6*page((2*8+2*8)*2) + page(2*4*ctx) + page(8+4) + page(16<<20) + page(4)
	if got := inkling2ResidentSessionBytes(s, ctx); got != want {
		t.Fatalf("session bytes %d, want %d", got, want)
	}
	if grown := inkling2ResidentSessionBytes(s, 2*ctx); grown-want != 3*global+page(2*4*ctx) {
		t.Fatalf("doubling the context added %d bytes", grown-want)
	}
}

func TestRunInkling2ResidentCtxError(t *testing.T) {
	if inkling2ResidentCtxError(nil, 4096) != nil {
		t.Fatal("nil error wrapped")
	}
	other := errors.New("inkling2 resident: arena exhausted")
	if got := inkling2ResidentCtxError(other, 4096); got != other {
		t.Fatalf("unrelated error rewritten: %v", got)
	}
	full := errors.New("inkling2 resident: position 4096 exceeds the 4096-token global cache")
	got := inkling2ResidentCtxError(full, 4096)
	for _, want := range []string{"4096-token context", "fak-metal-resident", "FAK_INKLING2_CTX", "default 4096", "max 23552"} {
		if got == nil || !strings.Contains(got.Error(), want) {
			t.Errorf("context error %v lacks %q", got, want)
		}
	}
	if !errors.Is(got, full) {
		t.Error("driver error not wrapped")
	}
}

func TestRunInkling2MetalTargetLabels(t *testing.T) {
	if inkling2MetalTargetVendoredLabel != "vendored-splash" || inkling2MetalTargetNativeLabel != "fak-metalgemm" || inkling2MetalTargetResidentLabel != "fak-metal-resident" {
		t.Fatalf("labels %q %q %q", inkling2MetalTargetVendoredLabel, inkling2MetalTargetNativeLabel, inkling2MetalTargetResidentLabel)
	}
	script, err := os.ReadFile(filepath.Join("..", "..", "scripts", "inkling2-metal-target-ab.sh"))
	if err != nil {
		t.Fatal(err)
	}
	for selector, label := range map[string]string{
		"vendored": inkling2MetalTargetVendoredLabel,
		"native":   inkling2MetalTargetNativeLabel,
		"resident": inkling2MetalTargetResidentLabel,
	} {
		if want := selector + `) echo "` + label + `"`; !strings.Contains(string(script), want) {
			t.Errorf("A/B script lacks %s", want)
		}
	}
}
