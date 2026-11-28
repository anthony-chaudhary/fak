package model

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// fak-test:runtime fast est=40ms
//
// (draft/MTP) load must open ONLY the shards whose index entries hold a wanted key, while
// the full-load path still opens every shard. The witness is the exact set of shard files
// the loader opens, recorded by the injected opener — so the test fails if the filter
// opens a shard that holds no wanted key.
func TestSafetensorsShardFilter(t *testing.T) {
	t.Parallel()

	const (
		shard1 = "model-00001-of-00004.safetensors"
		shard2 = "model-00002-of-00004.safetensors"
		shard3 = "model-00003-of-00004.safetensors"
		shard4 = "model-00004-of-00004.safetensors"
	)

	// shard1 holds the target/backbone keys, shard2 and shard3 hold the draft (mtp.*)
	// keys, shard4 holds the head/norm. A draft load wants only the mtp.* keys, which
	// live in shards 2 and 3 — shards 1 and 4 must never be opened.
	shardTensors := map[string]map[string]tinySTTensor{
		shard1: {
			"model.embed_tokens.weight":              {dtype: "F32", shape: []int{4, 32}, data: f32TestBytes(sequenceFloats(128, 0.01))},
			"model.layers.0.self_attn.q_proj.weight": {dtype: "BF16", shape: []int{2, 32}, data: bf16TestBytes(sequenceFloats(64, -0.07))},
		},
		shard2: {
			"mtp.0.self_attn.q_proj.weight": {dtype: "BF16", shape: []int{2, 32}, data: bf16TestBytes(sequenceFloats(64, 0.11))},
			"mtp.0.mlp.down_proj.weight":    {dtype: "F32", shape: []int{2, 32}, data: f32TestBytes(sequenceFloats(64, 0.13))},
		},
		shard3: {
			"mtp.1.self_attn.q_proj.weight": {dtype: "BF16", shape: []int{2, 32}, data: bf16TestBytes(sequenceFloats(64, 0.17))},
		},
		shard4: {
			"model.norm.weight": {dtype: "BF16", shape: []int{32}, data: bf16TestBytes(sequenceFloats(32, 0.25))},
			"lm_head.weight":    {dtype: "F32", shape: []int{4, 32}, data: f32TestBytes(sequenceFloats(128, 0.5))},
		},
	}

	dir := t.TempDir()
	weightMap := map[string]string{}
	for shard, tensors := range shardTensors {
		buf := tinySafetensorsBytes(t, tensors)
		if err := os.WriteFile(filepath.Join(dir, shard), buf, 0o644); err != nil {
			t.Fatalf("write shard %s: %v", shard, err)
		}
		for name := range tensors {
			weightMap[name] = shard
		}
	}
	index, err := json.Marshal(struct {
		WeightMap map[string]string `json:"weight_map"`
	}{WeightMap: weightMap})
	if err != nil {
		t.Fatalf("marshal index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), index, 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	// recordingOpener records every shard path the loader opens and serves the fixture
	// bytes; an unexpected path is a hard error so a stray open cannot pass silently.
	newOpener := func() (*[]string, safetensorsFileOpener) {
		var opened []string
		open := func(path string) (*safetensorsFile, error) {
			opened = append(opened, filepath.Base(path))
			buf, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			return newSafetensorsFile(bytes.NewReader(buf), int64(len(buf)), nil)
		}
		return &opened, open
	}

	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 4, TieWordEmbeddings: false}

	t.Run("only-shards-holding-a-wanted-key", func(t *testing.T) {
		opened, open := newOpener()
		// A draft load wants the mtp.0.* keys, which live only in shard2.
		_, err := loadSafetensorsDirFiltered(dir, cfg, open, []string{"mtp.0."})
		if err != nil {
			t.Fatalf("draft load: %v", err)
		}
		if want := []string{shard2}; !reflect.DeepEqual(*opened, want) {
			t.Fatalf("opened shards = %v, want %v (filter opened a shard with no wanted key)", *opened, want)
		}
	})

	t.Run("key-spread-across-shards-selects-all", func(t *testing.T) {
		opened, open := newOpener()
		// The "mtp." prefix matches keys in shard2 AND shard3, so both must be opened.
		_, err := loadSafetensorsDirFiltered(dir, cfg, open, []string{"mtp."})
		if err != nil {
			t.Fatalf("draft load: %v", err)
		}
		want := []string{shard2, shard3}
		sort.Strings(want)
		if !reflect.DeepEqual(*opened, want) {
			t.Fatalf("opened shards = %v, want %v", *opened, want)
		}
	})

	t.Run("empty-wanted-set-selects-all-shards", func(t *testing.T) {
		// The full-load path must be unchanged: a nil wanted set opens every shard.
		for _, wanted := range [][]string{nil, []string{}} {
			opened, open := newOpener()
			if _, err := loadSafetensorsDirFiltered(dir, cfg, open, wanted); err != nil {
				t.Fatalf("full load (wanted=%v): %v", wanted, err)
			}
			want := []string{shard1, shard2, shard3, shard4}
			sort.Strings(want)
			if !reflect.DeepEqual(*opened, want) {
				t.Fatalf("full load (wanted=%v) opened %v, want all %v", wanted, *opened, want)
			}
		}
	})

	t.Run("absent-wanted-key-is-reported", func(t *testing.T) {
		opened, open := newOpener()
		_, err := loadSafetensorsDirFiltered(dir, cfg, open, []string{"mtp.9."})
		if err == nil || !strings.Contains(err.Error(), "mtp.9.") {
			t.Fatalf("absent wanted key not reported: err=%v", err)
		}
		if len(*opened) != 0 {
			t.Fatalf("absent wanted key still opened shards: %v", *opened)
		}
	})

	t.Run("selection-is-pure", func(t *testing.T) {
		// The selection helper is the unit under test; pin its contract directly too.
		sel := selectSafetensorsShards(weightMap, []string{"mtp.0."})
		if !reflect.DeepEqual(sel.Shards, []string{shard2}) || len(sel.Missing) != 0 {
			t.Fatalf("select mtp.0. = %+v, want shards [%s]", sel, shard2)
		}
		sel = selectSafetensorsShards(weightMap, []string{"mtp."})
		if !reflect.DeepEqual(sel.Shards, []string{shard2, shard3}) || len(sel.Missing) != 0 {
			t.Fatalf("select mtp. = %+v, want shards [%s %s]", sel, shard2, shard3)
		}
		sel = selectSafetensorsShards(weightMap, nil)
		if !reflect.DeepEqual(sel.Shards, []string{shard1, shard2, shard3, shard4}) || len(sel.Missing) != 0 {
			t.Fatalf("select nil = %+v, want all shards", sel)
		}
		sel = selectSafetensorsShards(weightMap, []string{"mtp.0.", "nope."})
		if !reflect.DeepEqual(sel.Shards, []string{shard2}) || !reflect.DeepEqual(sel.Missing, []string{"nope."}) {
			t.Fatalf("select mixed = %+v, want shards [%s] missing [nope.]", sel, shard2)
		}
	})
}
