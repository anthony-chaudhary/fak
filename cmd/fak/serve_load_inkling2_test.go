package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

// A Small-shaped toy (arch inkling2, no loop keys, global attention every 6th
// layer) small enough to decode on the host in milliseconds.
const (
	ink2RunLayers  = 12
	ink2RunHidden  = 16
	ink2RunHeads   = 4
	ink2RunKV      = 2
	ink2RunHeadDim = 4
	ink2RunVocab   = 32
	ink2RunRelLen  = 4
	ink2RunRelExt  = 8
	ink2RunConv    = 3
	ink2RunExperts = 8
	ink2RunUsed    = 2
	ink2RunExpW    = 8
	ink2RunAlign   = 32
)

type ink2RunTensor struct {
	name string
	dims []uint64
	data []byte
}

func ink2RunGlobal(il int) bool { return il%6 == 5 }

func ink2RunTensors() []ink2RunTensor {
	var out []ink2RunTensor
	seed := 0
	add := func(name string, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		var b bytes.Buffer
		for i := 0; i < n; i++ {
			seed++
			_ = binary.Write(&b, binary.LittleEndian, math.Float32bits(float32(seed%97)/97-0.5))
		}
		out = append(out, ink2RunTensor{name: name, dims: dims, data: b.Bytes()})
	}
	const H, Q, KV = ink2RunHidden, ink2RunHeads * ink2RunHeadDim, ink2RunKV * ink2RunHeadDim
	add("token_embd.weight", H, ink2RunVocab)
	add("token_embd_norm.weight", H)
	add("output_norm.weight", H)
	add("output_res_query.weight", H)
	add("output.weight", H, ink2RunVocab)
	for il := 0; il < ink2RunLayers; il++ {
		p := fmt.Sprintf("blk.%d.", il)
		add(p+"attn_q.weight", H, Q)
		add(p+"attn_k.weight", H, KV)
		add(p+"attn_v.weight", H, KV)
		add(p+"attn_output.weight", Q, H)
		add(p+"attn_gate.weight", H, Q)
		add(p+"attn_rel.weight", H, ink2RunHeads*ink2RunRelLen)
		add(p+"attn_sink_proj.weight", H, ink2RunHeads)
		add(p+"attn_q_clip.weight", ink2RunHeadDim, ink2RunHeads)
		add(p+"attn_rel_bias.weight", ink2RunRelLen, ink2RunRelExt)
		add(p+"attn_rel_clip.weight", ink2RunRelLen, ink2RunHeads)
		add(p+"attn_k_conv.weight", ink2RunConv, KV)
		add(p+"attn_v_conv.weight", ink2RunConv, KV)
		add(p+"attn_post_conv.weight", ink2RunConv, H)
		add(p+"ffn_post_conv.weight", ink2RunConv, H)
		add(p+"ffn_res_query.weight", H)
		add(p+"ffn_norm.weight", H)
		add(p+"ffn_gate_inp.weight", H, ink2RunExperts)
		add(p+"ffn_gate_up_exps.weight", H, 2*ink2RunExpW, ink2RunExperts)
		add(p+"ffn_down_exps.weight", ink2RunExpW, H, ink2RunExperts)
		if ink2RunGlobal(il) {
			add(p+"attn_q_inp_norm.weight", H)
			add(p+"attn_k_inp_norm.weight", H)
			add(p+"attn_v_inp_norm.weight", H)
			add(p+"attn_k_conv_proj.weight", H, KV)
			add(p+"attn_v_conv_proj.weight", H, KV)
		} else {
			add(p+"attn_norm.weight", H)
		}
		if il > 0 {
			rows := uint64(1)
			if ink2RunGlobal(il) {
				rows = 3
			}
			add(p+"attn_res_query.weight", H, rows)
		}
	}
	return out
}

func ink2RunStr(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func ink2RunPutKV(b *bytes.Buffer, key string, typ ggufload.ValueType, v any) {
	ink2RunStr(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(typ))
	if s, ok := v.(string); ok {
		ink2RunStr(b, s)
		return
	}
	_ = binary.Write(b, binary.LittleEndian, v)
}

func ink2RunArray(b *bytes.Buffer, key string, elem ggufload.ValueType, n int, write func(i int)) {
	ink2RunStr(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(elem))
	_ = binary.Write(b, binary.LittleEndian, uint64(n))
	for i := 0; i < n; i++ {
		write(i)
	}
}

func ink2RunPad(b *bytes.Buffer) {
	for b.Len()%ink2RunAlign != 0 {
		b.WriteByte(0)
	}
}

// buildInk2RunShard writes shard no (1-based) of count; only shard 1 carries
// the model config, as gguf-split writes it.
func buildInk2RunShard(no, count int, tensors []ink2RunTensor) []byte {
	const p = "inkling2."
	var kvs []func(*bytes.Buffer)
	kv := func(f func(*bytes.Buffer)) { kvs = append(kvs, f) }
	kv(func(b *bytes.Buffer) { ink2RunPutKV(b, "general.alignment", ggufload.TypeUint32, uint32(ink2RunAlign)) })
	if count > 1 {
		kv(func(b *bytes.Buffer) { ink2RunPutKV(b, "split.count", ggufload.TypeUint64, uint64(count)) })
		kv(func(b *bytes.Buffer) { ink2RunPutKV(b, "split.no", ggufload.TypeUint64, uint64(no-1)) })
	}
	if no == 1 {
		kv(func(b *bytes.Buffer) { ink2RunPutKV(b, "general.architecture", ggufload.TypeString, "inkling2") })
		for k, v := range map[string]uint64{
			"embedding_length": ink2RunHidden, "block_count": ink2RunLayers, "attention.key_length": ink2RunHeadDim,
			"vocab_size_unpadded": ink2RunVocab, "attention.sliding_window": ink2RunRelExt,
			"attention.relative_length": ink2RunRelLen, "attention.relative_extent": ink2RunRelExt,
			"shortconv.l_cache": ink2RunConv, "attn_res.block_size": 6, "expert_count": ink2RunExperts,
			"expert_used_count": ink2RunUsed, "expert_feed_forward_length": ink2RunExpW,
			"attention.log_scale_floor": 8192,
		} {
			kv(func(b *bytes.Buffer) { ink2RunPutKV(b, p+k, ggufload.TypeUint64, v) })
		}
		for k, v := range map[string]float32{
			"attention.layer_norm_rms_epsilon": 1e-6, "attention.qk_clip": 30, "attention.relative_clip": 10,
			"attention.log_scale_alpha": 0.1, "activation.situ_beta": 1, "activation.situ_linear_beta": 0.5,
		} {
			kv(func(b *bytes.Buffer) { ink2RunPutKV(b, p+k, ggufload.TypeFloat32, math.Float32bits(v)) })
		}
		kv(func(b *bytes.Buffer) {
			ink2RunArray(b, p+"attention.sliding_window_pattern", ggufload.TypeBool, ink2RunLayers, func(i int) {
				if ink2RunGlobal(i) {
					b.WriteByte(0)
				} else {
					b.WriteByte(1)
				}
			})
		})
		kv(func(b *bytes.Buffer) {
			ink2RunPutKV(b, p+"attention.head_count", ggufload.TypeUint32, uint32(ink2RunHeads))
		})
		kv(func(b *bytes.Buffer) {
			ink2RunPutKV(b, p+"attention.head_count_kv", ggufload.TypeUint32, uint32(ink2RunKV))
		})
	}
	offsets := make([]uint64, len(tensors))
	off := 0
	for i, t := range tensors {
		offsets[i] = uint64(off)
		off = (off + len(t.data) + ink2RunAlign - 1) / ink2RunAlign * ink2RunAlign
	}
	var b bytes.Buffer
	b.WriteString(ggufload.Magic)
	_ = binary.Write(&b, binary.LittleEndian, uint32(ggufload.Version))
	_ = binary.Write(&b, binary.LittleEndian, uint64(len(tensors)))
	_ = binary.Write(&b, binary.LittleEndian, uint64(len(kvs)))
	for _, f := range kvs {
		f(&b)
	}
	for i, t := range tensors {
		ink2RunStr(&b, t.name)
		_ = binary.Write(&b, binary.LittleEndian, uint32(len(t.dims)))
		for _, d := range t.dims {
			_ = binary.Write(&b, binary.LittleEndian, d)
		}
		_ = binary.Write(&b, binary.LittleEndian, uint32(ggufload.TensorF32))
		_ = binary.Write(&b, binary.LittleEndian, offsets[i])
	}
	ink2RunPad(&b)
	for _, t := range tensors {
		b.Write(t.data)
		ink2RunPad(&b)
	}
	return b.Bytes()
}

// writeInk2RunSplit writes a two-shard Small-shaped set and returns both paths.
func writeInk2RunSplit(t *testing.T) (shard1, shard2 string) {
	t.Helper()
	dir := t.TempDir()
	all := ink2RunTensors()
	half := len(all) / 2
	shard1 = filepath.Join(dir, "Inkling-2-Small-UD-Q2_K_XL-00001-of-00002.gguf")
	shard2 = filepath.Join(dir, "Inkling-2-Small-UD-Q2_K_XL-00002-of-00002.gguf")
	if err := os.WriteFile(shard1, buildInk2RunShard(1, 2, all[:half]), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(shard2, buildInk2RunShard(2, 2, all[half:]), 0o644); err != nil {
		t.Fatal(err)
	}
	return shard1, shard2
}

func TestServeLoadInkling2SelectsArmAndDecodesOnHost(t *testing.T) {
	shard1, shard2 := writeInk2RunSplit(t)
	for _, path := range []string{shard1, shard2} {
		arch, err := serveInkling2Arch(path)
		if err != nil || arch != "inkling2" {
			t.Fatalf("serveInkling2Arch(%s) = %q, %v; want inkling2", filepath.Base(path), arch, err)
		}
	}
	var log bytes.Buffer
	l, err := loadServeInkling2(shard1, nil, &log)
	if err != nil {
		t.Fatalf("loadServeInkling2: %v", err)
	}
	defer l.Close()
	if l.Arch != "inkling2" || l.Backend != nil {
		t.Fatalf("arm = %q backend %v; want inkling2 on the host", l.Arch, l.Backend)
	}
	if !strings.Contains(log.String(), "inkling2 placement: host forward") {
		t.Fatalf("placement line missing from load log %q", log.String())
	}
	out, err := ggufload.Inkling2LoopGreedy(l.Model, []int{1, 2, 3}, 2)
	if err != nil {
		t.Fatalf("Inkling2LoopGreedy: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("greedy produced %d tokens, want 2", len(out))
	}
	for _, id := range out {
		if id < 0 || id >= ink2RunVocab {
			t.Fatalf("token %d outside vocab %d", id, ink2RunVocab)
		}
	}
}

func TestServeLoadInkling2NonInklingGGUFIsNotSelected(t *testing.T) {
	var b bytes.Buffer
	b.WriteString(ggufload.Magic)
	_ = binary.Write(&b, binary.LittleEndian, uint32(ggufload.Version))
	_ = binary.Write(&b, binary.LittleEndian, uint64(0))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1))
	ink2RunPutKV(&b, "general.architecture", ggufload.TypeString, "llama")
	path := filepath.Join(t.TempDir(), "llama.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if arch, err := serveInkling2Arch(path); err != nil || arch != "" {
		t.Fatalf("serveInkling2Arch(llama) = %q, %v; want not selected", arch, err)
	}
}

func TestServeLoadInkling2MissingShardIsTypedDiskRefusal(t *testing.T) {
	shard1, shard2 := writeInk2RunSplit(t)
	if err := os.Remove(shard2); err != nil {
		t.Fatal(err)
	}
	_, err := loadServeInkling2(shard1, nil, nil)
	var refusal *serveInkling2ResourceError
	if !errors.As(err, &refusal) || refusal.Resource != serveInkling2Disk {
		t.Fatalf("load with a missing shard = %v; want a disk serveInkling2ResourceError", err)
	}
	if !strings.Contains(err.Error(), filepath.Base(shard2)) {
		t.Fatalf("refusal %q does not name the missing shard", err)
	}
	var unsupported *fakmodel.UnsupportedInkling2LoopExecutionError
	if errors.As(err, &unsupported) {
		t.Fatalf("missing resource surfaced as the execution refusal: %v", err)
	}
}

func TestServeLoadInkling2DeviceErrorsNameTheResource(t *testing.T) {
	over := &ggufload.Inkling2DenseOverBudgetError{DenseBytes: 8 << 30, BudgetBytes: 4 << 30}
	for _, c := range []struct {
		err  error
		want serveInkling2Resource
	}{
		{fmt.Errorf("bind: %w", over), serveInkling2VRAM},
		{errors.New("cuMemAlloc: out of memory"), serveInkling2VRAM},
		{errors.New("cuda: driver not initialized"), serveInkling2CUDA},
	} {
		var refusal *serveInkling2ResourceError
		if err := serveInkling2ClassifyDeviceError("m.gguf", c.err); !errors.As(err, &refusal) || refusal.Resource != c.want {
			t.Fatalf("classify(%v) = %v; want %s", c.err, err, c.want)
		}
	}
}

func TestServeLoadInkling2AliasesAreCatalogued(t *testing.T) {
	for _, alias := range fakmodel.Inkling2GGUFAliases {
		if _, ok := modelreg.Catalog[alias]; !ok {
			t.Fatalf("refusal names alias %q, which the catalog does not carry", alias)
		}
	}
}

func TestServeLoadInkling2SelectsGGUFFileSuffix(t *testing.T) {
	path := filepath.Join(t.TempDir(), "Inkling-2-Mini-UD-Q3_K_XL.gguf_file")
	if err := os.WriteFile(path, buildInk2RunShard(1, 1, ink2RunTensors()), 0o644); err != nil {
		t.Fatal(err)
	}
	if arch, err := serveInkling2Arch(path); err != nil || arch == "" {
		t.Fatalf("serveInkling2Arch(%s) = %q, %v; want the inkling2 arm", filepath.Base(path), arch, err)
	}
	var log bytes.Buffer
	l, err := loadServeInkling2(path, nil, &log)
	if err != nil {
		t.Fatalf("loadServeInkling2: %v", err)
	}
	defer l.Close()
	if n := strings.Count(log.String(), "inkling2 placement:"); n != 1 {
		t.Fatalf("placement line written %d times, want once: %q", n, log.String())
	}
	if out, err := ggufload.Inkling2LoopGreedy(l.Model, []int{1}, 1); err != nil || len(out) != 1 {
		t.Fatalf("greedy = %v, %v; want one token", out, err)
	}
}

func TestServeLoadInkling2TruncatedShardIsTypedDiskRefusal(t *testing.T) {
	shard1, shard2 := writeInk2RunSplit(t)
	st, err := os.Stat(shard2)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(shard2, st.Size()-1); err != nil {
		t.Fatal(err)
	}
	_, err = loadServeInkling2(shard1, nil, nil)
	var refusal *serveInkling2ResourceError
	if !errors.As(err, &refusal) || refusal.Resource != serveInkling2Disk || !strings.Contains(err.Error(), "truncated") {
		t.Fatalf("load with a truncated shard = %v; want a disk truncation refusal", err)
	}
}
