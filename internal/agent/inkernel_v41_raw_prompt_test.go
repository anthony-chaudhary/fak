package agent_test

import (
	"bufio"
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// v41OracleSubsetTokenizer rebuilds the committed slice of the
// vcruz305/DeepSeek-V4.1-Flash-GGUF vocab (internal/tokenizer testdata), which
// tokenizes the reference-oracle prompts exactly as the full vocab does.
func v41OracleSubsetTokenizer(t *testing.T, control map[int]string) *tokenizer.Tokenizer {
	t.Helper()
	f, err := os.Open(filepath.Join("..", "tokenizer", "testdata", "deepseek_v41_vocab_subset.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var (
		pre    string
		tokens []string
		types  []int32
		merges []string
	)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		fields := strings.Split(sc.Text(), "\t")
		switch fields[0] {
		case "pre":
			pre = fields[1]
		case "n_vocab":
			n, err := strconv.Atoi(fields[1])
			if err != nil {
				t.Fatal(err)
			}
			tokens, types = make([]string, n), make([]int32, n)
			for i := range types {
				types[i] = 1
			}
		case "token":
			id, _ := strconv.Atoi(fields[1])
			typ, _ := strconv.Atoi(fields[2])
			text, err := base64.StdEncoding.DecodeString(fields[3])
			if err != nil || id >= len(tokens) {
				t.Fatalf("bad token line %q", sc.Text())
			}
			tokens[id], types[id] = string(text), int32(typ)
		case "merge":
			text, err := base64.StdEncoding.DecodeString(fields[1])
			if err != nil {
				t.Fatalf("bad merge line %q", sc.Text())
			}
			merges = append(merges, string(text))
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	for id, text := range control {
		tokens[id], types[id] = text, 3
	}
	tok, err := tokenizer.FromGGML(tokens, merges, types, pre)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// A raw V4.1 completion is the GGUF BOS followed by the prompt's own tokens, the
// shape of the reference oracle (reference-oracle-20261012-strix3: gold prompt
// [0 671 7658 7380 362 8495 344]).
// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime medium est=2s lane=default
func TestV41RawCompletionPromptIsBOSPlusPromptTokens(t *testing.T) {
	m := v41CompressorHTTPModel(t)
	bos := 0
	m.Cfg.BOSTokenID = &bos
	planner := agent.NewInKernelPlanner(m, v41OracleSubsetTokenizer(t, nil), "native", false, nil, false)
	msgs := []agent.Message{{Role: agent.RoleUser, Content: "The chemical symbol for gold is"}}

	raw, err := planner.EncodePrompt(context.Background(), msgs, nil, agent.WithRawPrompt())
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{0, 671, 7658, 7380, 362, 8495, 344}; !reflect.DeepEqual(raw.TokenIDs, want) {
		t.Fatalf("raw V4.1 prompt ids = %v, want oracle %v", raw.TokenIDs, want)
	}
	if raw.RendererID != "deepseek41-raw-bos-v1" {
		t.Fatalf("raw V4.1 renderer = %q", raw.RendererID)
	}
}

// A V4.1 chat turn renders the GGUF's non-thinking template, so the user text
// sits between the role control tokens the vocab declares (<｜User｜> 128803,
// <｜Assistant｜> 128804, </think> 128822 in the pinned GGUF header).
// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime medium est=2s lane=default
func TestV41ChatPromptUsesDeepSeekTemplateIDs(t *testing.T) {
	m := v41CompressorHTTPModel(t)
	tok := v41OracleSubsetTokenizer(t, map[int]string{128803: "<｜User｜>", 128804: "<｜Assistant｜>", 128822: "</think>"})
	planner := agent.NewInKernelPlanner(m, tok, "native", false, nil, false)
	chat, err := planner.EncodePrompt(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "The chemical symbol for gold is"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 128803, 671, 7658, 7380, 362, 8495, 344, 128804, 128822}
	if !reflect.DeepEqual(chat.TokenIDs, want) || chat.RendererID != "deepseek41-chat-nothink-v2" {
		t.Fatalf("V4.1 chat prompt = %s %v, want deepseek41-chat-nothink-v2 %v", chat.RendererID, chat.TokenIDs, want)
	}
}
