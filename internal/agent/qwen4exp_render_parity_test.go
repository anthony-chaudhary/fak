package agent

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func loadQwen4ExpHeader(t *testing.T) *ggufload.File {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("..", "ggufload", "testdata", "qwen4exp_ud_iq4xs_header", "*.gguf.gz"))
	if err != nil || len(paths) != 3 {
		t.Fatalf("Flash-Next header fixture: %d shards, err %v", len(paths), err)
	}
	sort.Strings(paths)
	shards := make([]*ggufload.File, len(paths))
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
		if shards[i], err = ggufload.Read(bytes.NewReader(hdr)); err != nil {
			t.Fatal(err)
		}
	}
	f, err := ggufload.MergeShardHeaders(shards[0], shards[1:]...)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// The chat prompt's reference ids are dff6004's --jinja render of the GGUF's own
// template; fak's renderer and tokenizer must reproduce both the text and the ids.
// fak-test:runtime fast est=500ms
func TestQwen4ExpChatRenderMatchesReference(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "ggufload", "testdata", "qwen4exp_reference_token_ids.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompts []struct {
			ID       string    `json:"id"`
			Messages []Message `json:"messages"`
			Rendered string    `json:"rendered"`
			IDs      []int     `json:"input_token_ids"`
		} `json:"prompts"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	f := loadQwen4ExpHeader(t)
	cfg, err := f.Config()
	if err != nil {
		t.Fatal(err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatal("no tokenizer in header")
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, p := range ref.Prompts {
		if len(p.Messages) == 0 {
			continue
		}
		checked++
		got := renderInKernelChatMLTools(p.Messages, nil, cfg)
		if got != p.Rendered {
			t.Fatalf("%s: fak render %q, dff6004 --jinja %q", p.ID, got, p.Rendered)
		}
		ids, err := tok.Encode(got)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != len(p.IDs) {
			t.Fatalf("%s: %d ids, reference %d", p.ID, len(ids), len(p.IDs))
		}
		for i := range ids {
			if ids[i] != p.IDs[i] {
				t.Fatalf("%s: id %d = %d, reference %d", p.ID, i, ids[i], p.IDs[i])
			}
		}
	}
	if checked != 1 {
		t.Fatalf("checked %d chat prompts, want 1", checked)
	}
}
