package ggufload

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// A Small-shaped toy: global attention every 6th layer, attn_res block 6, no
// loop keys or loop tensors.
const (
	ink2sLayers  = 12
	ink2sHidden  = 16
	ink2sHeads   = 4
	ink2sKV      = 2
	ink2sHeadDim = 4
	ink2sVocab   = 32
	ink2sRelLen  = 4
	ink2sRelExt  = 8
	ink2sWindow  = 6
	ink2sConv    = 3
	ink2sExperts = 8
	ink2sUsed    = 2
	ink2sExpW    = 8
)

type ink2sOpts struct {
	arch      string
	scalar    bool  // scalar head counts instead of per-layer arrays
	kvHeads   []int // overrides the per-layer kv array
	floor     uint64
	loopKeys  bool
	splitNo   int // 0 = single file; otherwise this shard's 1-based index of 2
	declareNo int // nonzero: write split.no as this 1-based index instead of splitNo
	keepNames func(i int) bool
}

func ink2sGlobal(il int) bool { return il%6 == 5 }

func ink2sTensors() []splitTensor {
	var out []splitTensor
	seed := 0
	add := func(name string, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		vs := make([]float32, n)
		for i := range vs {
			seed++
			vs[i] = float32(seed%97)/97 - 0.5
		}
		out = append(out, splitTensor{name: name, dims: dims, typ: TensorF32, data: f32Payload(vs...)})
	}
	const H, Q, KV = ink2sHidden, ink2sHeads * ink2sHeadDim, ink2sKV * ink2sHeadDim
	add("token_embd.weight", H, ink2sVocab)
	add("token_embd_norm.weight", H)
	add("output_norm.weight", H)
	add("output_res_query.weight", H)
	add("output.weight", H, ink2sVocab)
	for il := 0; il < ink2sLayers; il++ {
		p := fmt.Sprintf("blk.%d.", il)
		add(p+"attn_q.weight", H, Q)
		add(p+"attn_k.weight", H, KV)
		add(p+"attn_v.weight", H, KV)
		add(p+"attn_output.weight", Q, H)
		add(p+"attn_gate.weight", H, Q)
		add(p+"attn_rel.weight", H, ink2sHeads*ink2sRelLen)
		add(p+"attn_sink_proj.weight", H, ink2sHeads)
		add(p+"attn_q_clip.weight", ink2sHeadDim, ink2sHeads)
		// local layers bias over the sliding window, global ones over the relative extent
		extent := uint64(ink2sWindow)
		if ink2sGlobal(il) {
			extent = ink2sRelExt
		}
		add(p+"attn_rel_bias.weight", ink2sRelLen, extent)
		add(p+"attn_rel_clip.weight", ink2sRelLen, ink2sHeads)
		add(p+"attn_k_conv.weight", ink2sConv, KV)
		add(p+"attn_v_conv.weight", ink2sConv, KV)
		add(p+"attn_post_conv.weight", ink2sConv, H)
		add(p+"ffn_post_conv.weight", ink2sConv, H)
		add(p+"ffn_res_query.weight", H)
		add(p+"ffn_norm.weight", H)
		add(p+"ffn_gate_inp.weight", H, ink2sExperts)
		add(p+"ffn_gate_up_exps.weight", H, 2*ink2sExpW, ink2sExperts)
		add(p+"ffn_down_exps.weight", ink2sExpW, H, ink2sExperts)
		if ink2sGlobal(il) {
			add(p+"attn_q_inp_norm.weight", H)
			add(p+"attn_k_inp_norm.weight", H)
			add(p+"attn_v_inp_norm.weight", H)
			add(p+"attn_k_conv_proj.weight", H, ink2sKV*ink2sConv)
			add(p+"attn_v_conv_proj.weight", H, ink2sKV*ink2sConv)
		} else {
			add(p+"attn_norm.weight", H)
		}
		if il > 0 {
			rows := uint64(1)
			if ink2sGlobal(il) {
				rows = 3
			}
			add(p+"attn_res_query.weight", H, rows)
		}
	}
	return out
}

func writeKVUint64ArrayForTest(b *bytes.Buffer, key string, values []int) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeUint32))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		_ = binary.Write(b, binary.LittleEndian, uint32(v))
	}
}

func buildInk2sGGUF(o ink2sOpts, tensors []splitTensor) []byte {
	const align = 32
	p := o.arch + "."
	var kvs []func(*bytes.Buffer)
	kv := func(f func(*bytes.Buffer)) { kvs = append(kvs, f) }
	kv(func(b *bytes.Buffer) { writeKVUint32(b, "general.alignment", align) })
	if o.splitNo != 0 {
		kv(func(b *bytes.Buffer) { writeKVUint64(b, "split.count", 2) })
		no := o.splitNo
		if o.declareNo != 0 {
			no = o.declareNo
		}
		kv(func(b *bytes.Buffer) { writeKVUint64(b, "split.no", uint64(no-1)) })
	}
	if o.splitNo <= 1 {
		kv(func(b *bytes.Buffer) { writeKVString(b, "general.architecture", o.arch) })
		for k, v := range map[string]uint64{
			"embedding_length": ink2sHidden, "block_count": ink2sLayers, "attention.key_length": ink2sHeadDim,
			"vocab_size_unpadded": ink2sVocab, "attention.sliding_window": ink2sWindow,
			"attention.relative_length": ink2sRelLen, "attention.relative_extent": ink2sRelExt,
			"shortconv.l_cache": ink2sConv, "attn_res.block_size": 6, "expert_count": ink2sExperts,
			"expert_used_count": ink2sUsed, "expert_feed_forward_length": ink2sExpW,
			"attention.log_scale_floor": o.floor,
		} {
			kv(func(b *bytes.Buffer) { writeKVUint64(b, p+k, v) })
		}
		if o.loopKeys {
			for k, v := range map[string]uint64{"loop.start": 6, "loop.end": 12, "loop.count": 3, "loop.gate_rank": 2} {
				kv(func(b *bytes.Buffer) { writeKVUint64(b, p+k, v) })
			}
		}
		for k, v := range map[string]float32{
			"attention.layer_norm_rms_epsilon": 1e-6, "attention.qk_clip": 30, "attention.relative_clip": 10,
			"attention.log_scale_alpha": 0.1, "activation.situ_beta": 1, "activation.situ_linear_beta": 0.5,
		} {
			kv(func(b *bytes.Buffer) { writeKVFloat32(b, p+k, v) })
		}
		pattern := make([]bool, ink2sLayers)
		heads := make([]int, ink2sLayers)
		kvh := make([]int, ink2sLayers)
		for il := range pattern {
			pattern[il] = !ink2sGlobal(il)
			heads[il], kvh[il] = ink2sHeads, ink2sKV
		}
		if o.kvHeads != nil {
			kvh = o.kvHeads
		}
		kv(func(b *bytes.Buffer) { writeKVBoolArrayForTest(b, p+"attention.sliding_window_pattern", pattern) })
		if o.scalar {
			kv(func(b *bytes.Buffer) { writeKVUint32(b, p+"attention.head_count", ink2sHeads) })
			kv(func(b *bytes.Buffer) { writeKVUint32(b, p+"attention.head_count_kv", ink2sKV) })
		} else {
			kv(func(b *bytes.Buffer) { writeKVUint64ArrayForTest(b, p+"attention.head_count", heads) })
			kv(func(b *bytes.Buffer) { writeKVUint64ArrayForTest(b, p+"attention.head_count_kv", kvh) })
		}
	}
	var keep []splitTensor
	for i, t := range tensors {
		if o.keepNames == nil || o.keepNames(i) {
			keep = append(keep, t)
		}
	}
	offsets := make([]uint64, len(keep))
	off := 0
	for i, t := range keep {
		offsets[i] = uint64(off)
		off = (off + len(t.data) + align - 1) / align * align
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(keep)), uint64(len(kvs)))
	for _, f := range kvs {
		f(&b)
	}
	for i, t := range keep {
		writeTensorInfoForTest(&b, t.name, t.dims, t.typ, offsets[i])
	}
	padToAlignment(&b, align)
	for _, t := range keep {
		b.Write(t.data)
		padToAlignment(&b, align)
	}
	return b.Bytes()
}

func writeInk2sFile(t *testing.T, dir, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func openInk2s(t *testing.T, o ink2sOpts) (*Inkling2LoopModel, error) {
	t.Helper()
	path := writeInk2sFile(t, t.TempDir(), "small.gguf", buildInk2sGGUF(o, ink2sTensors()))
	m, err := OpenInkling2Loop(path)
	if err == nil {
		t.Cleanup(func() { m.Close() })
	}
	return m, err
}

func TestInkling2SmallBindsWithoutLoop(t *testing.T) {
	for _, scalar := range []bool{false, true} {
		t.Run(fmt.Sprintf("scalar=%v", scalar), func(t *testing.T) {
			m, err := openInk2s(t, ink2sOpts{arch: "inkling2", scalar: scalar, floor: 8192})
			if err != nil {
				t.Fatalf("bind: %v", err)
			}
			hp := m.HP
			if m.Arch != "inkling2" || hp.Passes != 1 || hp.CoreStart != hp.CoreEnd || hp.GateRank != 0 {
				t.Fatalf("arch %q passes %d core [%d,%d) gate %d", m.Arch, hp.Passes, hp.CoreStart, hp.CoreEnd, hp.GateRank)
			}
			if hp.Heads != ink2sHeads || hp.KVHeads != ink2sKV || hp.Layers != ink2sLayers || hp.AttnResBlock != 6 {
				t.Fatalf("hparams %+v", hp)
			}
			for il, local := range hp.LocalLayer {
				if local == ink2sGlobal(il) {
					t.Fatalf("layer %d local=%v", il, local)
				}
			}
			if len(m.Weights.Controllers) != 0 || len(m.Weights.Layers) != ink2sLayers {
				t.Fatalf("controllers %d layers %d", len(m.Weights.Controllers), len(m.Weights.Layers))
			}
			if m.LogScaleFloorOverridden || hp.LogScaleFloor != 8192 {
				t.Fatalf("floor %v overridden %v", hp.LogScaleFloor, m.LogScaleFloorOverridden)
			}
			if got := len(m.Weights.Layers[5].AttnResQuery); got != 3 {
				t.Fatalf("global layer attn_res_query rows %d", got)
			}
		})
	}
}

func TestInkling2SmallStaleLogScaleFloorOverridden(t *testing.T) {
	m, err := openInk2s(t, ink2sOpts{arch: "inkling2", floor: 128000})
	if err != nil {
		t.Fatal(err)
	}
	if !m.LogScaleFloorOverridden || m.HP.LogScaleFloor != 8192 {
		t.Fatalf("floor %v overridden %v", m.HP.LogScaleFloor, m.LogScaleFloorOverridden)
	}
}

func TestInkling2SmallStrayLoopKeysIgnored(t *testing.T) {
	m, err := openInk2s(t, ink2sOpts{arch: "inkling2", floor: 8192, loopKeys: true})
	if err != nil {
		t.Fatal(err)
	}
	if m.HP.Passes != 1 || m.HP.CoreStart != m.HP.CoreEnd || m.HP.GateRank != 0 {
		t.Fatalf("hparams %+v", m.HP)
	}
}

func TestInkling2SmallNonUniformHeadsRefuseTyped(t *testing.T) {
	kv := make([]int, ink2sLayers)
	for i := range kv {
		kv[i] = ink2sKV
	}
	kv[3] = 4
	_, err := openInk2s(t, ink2sOpts{arch: "inkling2", floor: 8192, kvHeads: kv})
	var typed *Inkling2NonUniformHeadsError
	if !errors.As(err, &typed) || typed.Key != "attention.head_count_kv" {
		t.Fatalf("err = %v, want typed non-uniform refusal", err)
	}
}

func TestInkling2SmallUnknownArchRefuses(t *testing.T) {
	_, err := openInk2s(t, ink2sOpts{arch: "inkling3", floor: 8192})
	if err == nil || !strings.Contains(err.Error(), "general.architecture") {
		t.Fatalf("err = %v, want architecture refusal", err)
	}
}

func TestInkling2SmallLoopArchStillRequiresLoopKeys(t *testing.T) {
	_, err := openInk2s(t, ink2sOpts{arch: "inkling2_loop", floor: 8192})
	if err == nil || !strings.Contains(err.Error(), "loop.start") {
		t.Fatalf("err = %v, want missing loop key refusal", err)
	}
}

func TestInkling2SmallSplitShardsBindIdentically(t *testing.T) {
	tensors := ink2sTensors()
	half := len(tensors) / 2
	dir := t.TempDir()
	single := writeInk2sFile(t, dir, "single.gguf", buildInk2sGGUF(ink2sOpts{arch: "inkling2", floor: 128000}, tensors))
	writeInk2sFile(t, dir, "Small-00001-of-00002.gguf", buildInk2sGGUF(ink2sOpts{
		arch: "inkling2", floor: 128000, splitNo: 1, keepNames: func(i int) bool { return i < half },
	}, tensors))
	shard2 := writeInk2sFile(t, dir, "Small-00002-of-00002.gguf", buildInk2sGGUF(ink2sOpts{
		arch: "inkling2", splitNo: 2, keepNames: func(i int) bool { return i >= half },
	}, tensors))

	want, err := OpenInkling2Loop(single)
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	// Opening through a later shard must still find the config carrier.
	got, err := OpenInkling2Loop(shard2)
	if err != nil {
		t.Fatalf("split bind: %v", err)
	}
	if n := len(got.closer.(inkling2Closers)); n != 2 {
		t.Fatalf("split model holds %d mappings, want 2", n)
	}
	if !reflect.DeepEqual(got.HP, want.HP) || got.LogScaleFloorOverridden != want.LogScaleFloorOverridden {
		t.Fatalf("hparams differ:\n got %+v\nwant %+v", got.HP, want.HP)
	}
	if !reflect.DeepEqual(got.Weights, want.Weights) {
		t.Fatal("split weights differ from single-file weights")
	}
	if v := got.Weights.Layers[ink2sLayers-1].FFNNorm; len(v) != ink2sHidden || math.IsNaN(float64(v[0])) {
		t.Fatalf("last-layer ffn_norm from shard 2 = %v", v)
	}
	if err := got.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	// Every shard must be unmapped: on Windows a live mapping blocks removal.
	for _, n := range []string{"Small-00001-of-00002.gguf", "Small-00002-of-00002.gguf"} {
		if err := os.Remove(filepath.Join(dir, n)); err != nil {
			t.Fatalf("shard still mapped after Close: %v", err)
		}
	}
}

// writeInk2sSplit writes a two-shard Small set; opts adjust each shard.
func writeInk2sSplit(t *testing.T, dir string, adjust func(shard int, o *ink2sOpts)) {
	t.Helper()
	tensors := ink2sTensors()
	half := len(tensors) / 2
	for shard := 1; shard <= 2; shard++ {
		lo := shard == 1
		o := ink2sOpts{arch: "inkling2", floor: 8192, splitNo: shard, keepNames: func(i int) bool { return (i < half) == lo }}
		if adjust != nil {
			adjust(shard, &o)
		}
		writeInk2sFile(t, dir, fmt.Sprintf("Small-%05d-of-00002.gguf", shard), buildInk2sGGUF(o, tensors))
	}
}

func wantInk2sRefusal(t *testing.T, path, want string) {
	t.Helper()
	m, err := OpenInkling2Loop(path)
	if err == nil {
		m.Close()
		t.Fatalf("bound %s, want refusal containing %q", filepath.Base(path), want)
	}
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("err = %v, want %q", err, want)
	}
}

func TestInkling2SmallSplitCountWithoutShardNameRefuses(t *testing.T) {
	path := writeInk2sFile(t, t.TempDir(), "small.gguf", buildInk2sGGUF(ink2sOpts{arch: "inkling2", floor: 8192, splitNo: 1}, ink2sTensors()))
	wantInk2sRefusal(t, path, "not a -N-of-M.gguf shard path")
}

func TestInkling2SmallMissingSiblingShardRefuses(t *testing.T) {
	dir := t.TempDir()
	writeInk2sSplit(t, dir, nil)
	if err := os.Remove(filepath.Join(dir, "Small-00002-of-00002.gguf")); err != nil {
		t.Fatal(err)
	}
	wantInk2sRefusal(t, filepath.Join(dir, "Small-00001-of-00002.gguf"), "open shard")
}

func TestInkling2SmallSwappedSplitNoRefuses(t *testing.T) {
	dir := t.TempDir()
	writeInk2sSplit(t, dir, func(shard int, o *ink2sOpts) { o.declareNo = 3 - shard })
	wantInk2sRefusal(t, filepath.Join(dir, "Small-00001-of-00002.gguf"), "declares split.no")
}

func TestInkling2SmallSplitShard1WithoutConfigRefuses(t *testing.T) {
	dir := t.TempDir()
	// Config on shard 2 only: split.no still reads 1-based, but shard 1 has no
	// general.architecture.
	tensors := ink2sTensors()
	half := len(tensors) / 2
	writeInk2sFile(t, dir, "Small-00001-of-00002.gguf", buildInk2sGGUF(ink2sOpts{
		arch: "inkling2", splitNo: 2, declareNo: 1, keepNames: func(i int) bool { return i < half },
	}, tensors))
	writeInk2sFile(t, dir, "Small-00002-of-00002.gguf", buildInk2sGGUF(ink2sOpts{
		arch: "inkling2", floor: 8192, splitNo: 1, declareNo: 2, keepNames: func(i int) bool { return i >= half },
	}, tensors))
	wantInk2sRefusal(t, filepath.Join(dir, "Small-00002-of-00002.gguf"), "split shard 1 carries no config")
}

func TestInkling2SmallEmptyCoreHostStepFinite(t *testing.T) {
	m, err := openInk2s(t, ink2sOpts{arch: "inkling2", floor: 128000, loopKeys: true})
	if err != nil {
		t.Fatal(err)
	}
	h, err := model.NewInkling2LoopHost(m.HP, m.Weights)
	if err != nil {
		t.Fatalf("host: %v", err)
	}
	for _, tok := range []int{1, 7} {
		logits := h.Step(tok)
		if len(logits) < ink2sVocab {
			t.Fatalf("step %d: %d logits, want >= %d", tok, len(logits), ink2sVocab)
		}
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("step %d: logit %d = %v", tok, i, v)
			}
		}
	}
}
