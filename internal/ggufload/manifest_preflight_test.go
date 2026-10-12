package ggufload

import (
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

func preflightTokens(p ManifestPreflight) []string {
	out := make([]string, 0, len(p.Refusals))
	for _, r := range p.Refusals {
		out = append(out, string(r.Reason)+" "+r.Axis)
	}
	sort.Strings(out)
	return out
}

func syntheticPreflightFile() *File {
	u32 := func(v uint32) Value { return Value{Type: TypeUint32, Value: v} }
	tokens := make([]Value, 4)
	for i := range tokens {
		tokens[i] = Value{Type: TypeString, Value: fmt.Sprintf("t%d", i)}
	}
	return &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "llama"},
			"llama.block_count":                      u32(1),
			"llama.embedding_length":                 u32(8),
			"llama.feed_forward_length":              u32(16),
			"llama.attention.head_count":             u32(2),
			"llama.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
			"llama.unconsumed_axis":                  u32(7),
			"tokenizer.ggml.tokens":                  {Type: TypeArray, Value: tokens},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{8, 4}, Type: TensorF32},
			{Name: "blk.0.attn_q.weight", Dims: []uint64{8, 8}, Type: TensorType(99)},
			{Name: "blk.0.bogus_proj.weight", Dims: []uint64{8, 8}, Type: TensorF32},
		},
	}
}

// fak-test:runtime fast est=10ms
func TestManifestPreflightSyntheticReportsExactlyTheFourSeams(t *testing.T) {
	p := BuildManifestPreflight(ManifestPreflightInput{
		File:     syntheticPreflightFile(),
		HFConfig: []byte(`{"hidden_size": 16, "num_hidden_layers": 1}`),
	})
	got := preflightTokens(p)
	want := []string{
		string(newmodel.RefusalHFMismatch) + " llama.embedding_length",
		string(newmodel.RefusalKeyUnread) + " llama.unconsumed_axis",
		string(newmodel.RefusalTensorOrphan) + " blk.0.bogus_proj.weight",
		string(newmodel.RefusalUnsupportedDtype) + " " + OpHostDequant + "(TensorType(99))",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("refusals = %v, want %v", got, want)
	}
	if p.Verdict != ManifestPreflightRefused || len(p.RefusedTiers) != 1 || p.RefusedTiers[0] != TierHost {
		t.Fatalf("verdict %s tiers %v, want REFUSED [%s]", p.Verdict, p.RefusedTiers, TierHost)
	}
	for _, r := range p.Refusals {
		if r.Schema != newmodel.RefusalSchema || r.Phase != newmodel.PhaseHeaderPreflight {
			t.Fatalf("refusal %+v not in %s/%s", r, newmodel.RefusalSchema, newmodel.PhaseHeaderPreflight)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestManifestPreflightLeavesCallerFileUntraced(t *testing.T) {
	f := syntheticPreflightFile()
	BuildManifestPreflight(ManifestPreflightInput{File: f})
	if f.ReadKeys() != nil || f.keyReads != nil {
		t.Fatal("preflight traced the caller's File")
	}
}

func loadV41HeaderFixture(t *testing.T) *File {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "deepseek41_q2k_header", "*.gguf.gz"))
	if err != nil || len(paths) != 7 { //boundarylint:ignore CHANGE_DETECTOR_TEST pinned deepseek41_q2k_header/provenance.json artifact contains exactly seven shards
		t.Fatalf("V4.1 header fixture: %d shards, err %v", len(paths), err)
	}
	sort.Strings(paths)
	shards := make([]*File, len(paths))
	for i, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		zr, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		hdr, err := io.ReadAll(io.LimitReader(zr, 16<<20))
		if err != nil {
			t.Fatal(err)
		}
		if shards[i], err = Read(bytes.NewReader(hdr)); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
	}
	f, err := MergeShardHeaders(shards[0], shards[1:]...)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// fak-test:runtime fast est=300ms
func TestManifestPreflightV41HeaderReportsZero(t *testing.T) {
	f := loadV41HeaderFixture(t)
	hf, err := os.ReadFile(filepath.Join("..", "model", "testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	p := BuildManifestPreflight(ManifestPreflightInput{File: f, HFConfig: hf})
	if p.Refused() {
		t.Fatalf("V4.1 header refused: %v", preflightTokens(p))
	}
	if p.Tensors.Consumed != 1046 || p.Keys.ArchScoped == 0 || p.Keys.Unread != 0 {
		t.Fatalf("coverage = %+v / %+v, want 1046 consumed tensors and every arch key read", p.Tensors, p.Keys)
	}
	if p.HF.Bindings == 0 || p.HF.Matched != p.HF.Bindings {
		t.Fatalf("HF parity = %+v, want every binding matched", p.HF)
	}
}

// The V4.1 retro's 29-day seam: compress_rope_freq_base sat in the header with no
// reader. The preflight's key trace must keep seeing its reader.
// fak-test:runtime fast est=300ms
func TestManifestPreflightV41CompressRopeBaseHasReader(t *testing.T) {
	f := loadV41HeaderFixture(t)
	f.TraceKeyReads()
	if _, err := f.Config(); err != nil {
		t.Fatal(err)
	}
	read := map[string]bool{}
	for _, k := range f.ReadKeys() {
		read[k] = true
	}
	for _, k := range []string{"deepseek41.attention.compress_rope_freq_base", "deepseek41.expert_gating_func", "deepseek41.hash_layer_count"} {
		if !read[k] {
			t.Fatalf("%s has no reader", k)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestDeepSeek41GGUFGatingAndHashLayersRefuseUnimplemented(t *testing.T) {
	for key, v := range map[string]Value{
		"deepseek41.expert_gating_func": {Type: TypeUint32, Value: uint32(2)},
		"deepseek41.hash_layer_count":   {Type: TypeUint32, Value: uint32(3)},
	} {
		meta := vcruzHParamMeta()
		meta[key] = v
		_, err := (&File{Metadata: meta}).Config()
		if !errors.Is(err, ErrDeepSeek41HParam) {
			t.Fatalf("%s=%v: err %v, want ErrDeepSeek41HParam", key, v.Value, err)
		}
	}
}

// fak-test:runtime fast est=50ms
func TestHostDequantSupportedMatchesDequant(t *testing.T) {
	for typ, tr := range ggufTypeTraits {
		info := TensorInfo{Name: "probe", Dims: []uint64{tr.BlockSize}, Type: typ}
		_, err := dequantF32(info, make([]byte, tr.TypeSize))
		if HostDequantSupported(typ) != (err == nil) {
			t.Fatalf("%s: HostDequantSupported=%v, dequant err %v", typ, HostDequantSupported(typ), err)
		}
	}
	if HostDequantSupported(TensorType(99)) {
		t.Fatal("unknown type reported supported")
	}
}

// fak-test:runtime fast est=1ms
func TestMetaValueEqualIntegersAreExact(t *testing.T) {
	if metaValueEqual(Value{Type: TypeUint32, Value: uint32(1048576)}, float64(1048577)) {
		t.Fatal("integer key matched an HF value one larger")
	}
	if !metaValueEqual(Value{Type: TypeFloat32, Value: float32(1e-6)}, 1e-6) {
		t.Fatal("f32 key did not match its HF float at f32 precision")
	}
}

// fak-test:runtime fast est=10ms
func TestManifestPreflightBlockCountExcludesNextN(t *testing.T) {
	f := syntheticPreflightFile()
	delete(f.Metadata, "llama.unconsumed_axis")
	f.Tensors = f.Tensors[:1]
	f.Metadata["llama.block_count"] = Value{Type: TypeUint32, Value: uint32(2)}
	f.Metadata["llama.nextn_predict_layers"] = Value{Type: TypeUint32, Value: uint32(1)}
	p := BuildManifestPreflight(ManifestPreflightInput{File: f, HFConfig: []byte(`{"num_hidden_layers": 1, "hidden_size": 8}`)})
	for _, r := range p.Refusals {
		if r.Reason == newmodel.RefusalHFMismatch {
			t.Fatalf("block_count with one NextN block refused against HF num_hidden_layers: %+v", r)
		}
	}
	if p.HF.Mismatched != 0 || p.HF.Matched != 2 {
		t.Fatalf("HF parity = %+v", p.HF)
	}
}

// fak-test:runtime fast est=300ms
func TestManifestPreflightConfigErrorReportsNoUnreadKeys(t *testing.T) {
	f := loadV41HeaderFixture(t)
	f.Metadata["deepseek41.hash_layer_count"] = Value{Type: TypeUint32, Value: uint32(3)}
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	if p.Counts[string(newmodel.RefusalHeaderConfigInvalid)] != 1 || p.Counts[string(newmodel.RefusalKeyUnread)] != 0 {
		t.Fatalf("counts = %v, want one HEADER_CONFIG_INVALID and no KEY_UNREAD", p.Counts)
	}
}

// fak-test:runtime fast est=300ms
func TestManifestPreflightQwen38HeaderReadsEveryKey(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "qwen38_ud_q2kxl_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	hdr, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	if err != nil {
		t.Fatal(err)
	}
	f, err := Read(bytes.NewReader(hdr))
	if err != nil {
		t.Fatal(err)
	}
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	if p.Refused() || p.Keys.Unread != 0 {
		t.Fatalf("served Qwen3.8-27B header refused: %v", preflightTokens(p))
	}
}

// fak-test:runtime fast est=10ms
func TestQwen35HybridHeaderRefusesInconsistentFallbackKeys(t *testing.T) {
	for key, v := range map[string]Value{
		"ssm.inner_size":          {Type: TypeUint32, Value: uint32(4096)},
		"attention.value_length":  {Type: TypeUint32, Value: uint32(128)},
		"rope.dimension_sections": {Type: TypeArray, Value: []Value{{Type: TypeUint32, Value: uint32(16)}}},
	} {
		meta := synthQwen38Meta("qwen35")
		meta["qwen35."+key] = v
		if _, err := (&File{Metadata: meta}).Config(); err == nil {
			t.Fatalf("qwen35.%s=%v admitted", key, v.Value)
		}
	}
}
