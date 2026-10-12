package ggufload

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func loadQwen4ExpHeaderFixture(t *testing.T) *File {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "qwen4exp_ud_iq4xs_header", "*.gguf.gz"))
	if err != nil || len(paths) != 3 {
		t.Fatalf("Flash-Next header fixture: %d shards, err %v", len(paths), err)
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

func qwen4ExpHFConfig(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "model", "testdata", "qwen4exp_flash_next_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// fak-test:runtime fast est=400ms
func TestQwen4ExpPinnedHeaderPreflightGreen(t *testing.T) {
	p := BuildManifestPreflight(ManifestPreflightInput{File: loadQwen4ExpHeaderFixture(t), HFConfig: qwen4ExpHFConfig(t)})
	if p.Refused() {
		t.Fatalf("pinned Flash-Next UD-IQ4_XS header refused: %v", preflightTokens(p))
	}
	if p.ModelType != "qwen4exp" || p.Tensors.Mode != TensorCoverageArchSchema || p.Tensors.Consumed != 1224 || p.Tensors.Requested < 1224 {
		t.Fatalf("coverage = %s %+v, want arch-schema with all 1224 tensors consumed", p.ModelType, p.Tensors)
	}
	if p.Keys.Unread != 0 || p.Keys.Read != p.Keys.ArchScoped {
		t.Fatalf("keys = %+v, want every qwen4exp key read", p.Keys)
	}
	if p.HF.Mismatched != 0 || p.HF.Matched != p.HF.Bindings || p.HF.Bindings < 30 {
		t.Fatalf("HF parity against f5d0827 = %+v, want every binding matched", p.HF)
	}
}

// fak-test:runtime fast est=300ms
func TestQwen4ExpGGUFConfigMatchesHF(t *testing.T) {
	cfg, err := loadQwen4ExpHeaderFixture(t).Config()
	if err != nil {
		t.Fatal(err)
	}
	hf, err := model.LoadQwen4ExpFlashNextConfig(filepath.Join("..", "model", "testdata", "qwen4exp_flash_next_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	x := cfg.Qwen4Exp
	if x == nil || x.PLE == nil {
		t.Fatalf("Qwen4Exp hparams = %+v, want hyper-connection, indexer and PLE axes", x)
	}
	if cfg.ModelType != "qwen4exp" || cfg.NumLayers != hf.NumHiddenLayers || cfg.FullAttentionInterval != hf.FullAttentionInterval ||
		cfg.NumExperts != hf.NumExperts || cfg.NumExpertsPerTok != hf.NumExpertsPerToken || x.SharedExpertFFN != hf.SharedExpertIntermediateSize ||
		x.IndexTopK != hf.IndexerBudget || x.PLE.NgramSize != hf.NgramSize || cfg.MaxPositionEmbeddings != hf.MaxPositionEmbeddings {
		t.Fatalf("GGUF config %+v / %+v disagrees with HF %+v", cfg, x, hf)
	}
	for l, lt := range hf.LayerTypes {
		if cfg.LayerTypes[l] != lt {
			t.Fatalf("layer %d = %s, HF %s", l, cfg.LayerTypes[l], lt)
		}
	}
	if !cfg.HasPLEEngram() || x.PLE.Layers[0] != 1 || x.PLE.NumHeads != 16 || x.PLE.EmbedPerLayer != 160 || x.HCCount != 4 || x.HCLowRank != 320 {
		t.Fatalf("PLE/HC axes = %+v / %+v", x, x.PLE)
	}
	if _, err := model.ClassifyForwardPath(cfg, nil); err == nil {
		t.Fatal("qwen4exp classified onto a generic forward; it must refuse until its own forward lands")
	}
}

// fak-test:runtime fast est=10ms
func TestQwen4ExpHeaderRefusesWhatTheReferenceRefuses(t *testing.T) {
	f := loadQwen4ExpHeaderFixture(t)
	for key, v := range map[string]Value{
		"qwen4exp.hyper_connection.count":  {Type: TypeUint32, Value: uint32(1)},
		"qwen4exp.attention.indexer.top_k": {Type: TypeUint32, Value: uint32(0)},
		"qwen4exp.ple.layers":              {Type: TypeArray, Value: []Value{{Type: TypeUint32, Value: uint32(1)}, {Type: TypeUint32, Value: uint32(5)}}},
		"qwen4exp.ple.ngram_size":          {Type: TypeUint32, Value: uint32(1)},
		"qwen4exp.ssm.inner_size":          {Type: TypeUint32, Value: uint32(4096)},
		"qwen4exp.full_attention_interval": {Type: TypeUint32, Value: uint32(0)},
	} {
		g := *f
		g.Metadata = make(map[string]Value, len(f.Metadata))
		for k, mv := range f.Metadata {
			g.Metadata[k] = mv
		}
		g.Metadata[key] = v
		if _, err := g.Config(); err == nil {
			t.Fatalf("%s=%v admitted", key, v.Value)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestQwen4ExpOrphanAndMissingTensorsReported(t *testing.T) {
	f := loadQwen4ExpHeaderFixture(t)
	g := *f
	g.Tensors = nil
	for _, ti := range f.Tensors {
		if ti.Name != "blk.1.ple_conv1d.weight" {
			g.Tensors = append(g.Tensors, ti)
		}
	}
	g.Tensors = append(g.Tensors, TensorInfo{Name: "blk.0.attn_norm.weight", Dims: []uint64{2560}, Type: TensorF32})
	g.Metadata = map[string]Value{}
	for k, v := range f.Metadata {
		g.Metadata[k] = v
	}
	delete(g.Metadata, "split.tensors.count")
	p := BuildManifestPreflight(ManifestPreflightInput{File: &g})
	got := preflightTokens(p)
	want := []string{
		string(newmodel.RefusalTensorMissing) + " blk.1.ple_conv1d.weight",
		string(newmodel.RefusalTensorOrphan) + " blk.0.attn_norm.weight",
	}
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("refusals = %v, want %v", got, want)
	}
}

type qwen4ExpRefIDs struct {
	Prompts []struct {
		ID            string `json:"id"`
		Text          string `json:"text"`
		Rendered      string `json:"rendered"`
		InputTokenIDs []int  `json:"input_token_ids"`
	} `json:"prompts"`
}

// fak-test:runtime fast est=2s
func TestQwen4ExpTokenizerMatchesReferenceIDs(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "qwen4exp_reference_token_ids.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref qwen4ExpRefIDs
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	gt, ok := loadQwen4ExpHeaderFixture(t).GGMLTokenizer()
	if !ok {
		t.Fatal("Flash-Next header carries no BPE tokenizer")
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatal(err)
	}
	if len(ref.Prompts) != 4 {
		t.Fatalf("reference has %d prompts, want 4", len(ref.Prompts))
	}
	for _, p := range ref.Prompts {
		text := p.Text
		if p.Rendered != "" {
			text = p.Rendered
		}
		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatalf("%s: %v", p.ID, err)
		}
		if !equalInts(ids, p.InputTokenIDs) {
			t.Fatalf("%s: fak ids diverge from dff6004 at %d\nfak %v\nref %v", p.ID, firstDiff(ids, p.InputTokenIDs), ids, p.InputTokenIDs)
		}
	}
}

func equalInts(a, b []int) bool {
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

func firstDiff(a, b []int) int {
	for i := range a {
		if i >= len(b) || a[i] != b[i] {
			return i
		}
	}
	return len(a)
}

var _ = errors.New

func withMeta(f *File, set map[string]Value, drop ...string) *File {
	g := *f
	g.Metadata = make(map[string]Value, len(f.Metadata))
	for k, v := range f.Metadata {
		g.Metadata[k] = v
	}
	for k, v := range set {
		g.Metadata[k] = v
	}
	for _, k := range drop {
		delete(g.Metadata, k)
	}
	return &g
}

// Keys the reference treats as optional load the way it does.
// fak-test:runtime fast est=50ms
func TestQwen4ExpOptionalKeysFollowReference(t *testing.T) {
	f := loadQwen4ExpHeaderFixture(t)
	u32 := func(v uint32) Value { return Value{Type: TypeUint32, Value: v} }
	blocks64, _ := f.Uint64("qwen4exp.block_count")
	blocks := int(blocks64)

	cfg, err := withMeta(f, nil, "qwen4exp.full_attention_interval").Config()
	if err != nil || len(cfg.LayerTypes) != blocks || cfg.LayerTypes[3] != "full_attention" || cfg.LayerTypes[2] != "linear_attention" {
		t.Fatalf("missing full_attention_interval: err %v, %d layer types; want the reference's default interval 4", err, len(cfg.LayerTypes))
	}
	cfg, err = withMeta(f, map[string]Value{"qwen4exp.nextn_predict_layers": u32(1)}).Config()
	if err != nil || cfg.NumLayers != blocks || len(cfg.LayerTypes) != blocks {
		t.Fatalf("nextn key without a head tensor: err %v, layers %d; want all %d blocks as trunk", err, cfg.NumLayers, blocks)
	}
	g := withMeta(f, map[string]Value{"qwen4exp.nextn_predict_layers": u32(1)})
	g.Tensors = append(append([]TensorInfo(nil), f.Tensors...), TensorInfo{Name: "blk.47.nextn.eh_proj.weight", Dims: []uint64{5120, 2560}, Type: TensorF32})
	if _, err := g.Config(); err == nil {
		t.Fatal("a file carrying the MTP head was admitted")
	}
	cfg, err = withMeta(f, nil, "qwen4exp.expert_shared_feed_forward_length").Config()
	if err != nil || cfg.Qwen4Exp.SharedExpertFFN != cfg.IntermediateSize {
		t.Fatalf("missing shared FFN length: err %v; want the reference's n_ff fallback", err)
	}
	cfg, err = withMeta(f, map[string]Value{"qwen4exp.ple.layers": {Type: TypeArray, Value: []Value{}}}).Config()
	if err != nil || cfg.Qwen4Exp.PLE != nil || cfg.HasPLEEngram() {
		t.Fatalf("empty ple.layers: err %v; want a model with no PLE", err)
	}
}
