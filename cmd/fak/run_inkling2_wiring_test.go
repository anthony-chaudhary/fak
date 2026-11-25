package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// The synthetic set embeds no tokenizer, so bind a toy vocab sized to it.
func withInk2RunToyTokenizer(t *testing.T) {
	t.Helper()
	tokens := make([]string, 0, ink2RunVocab)
	for c := 'a'; c <= 'z'; c++ {
		tokens = append(tokens, string(c))
	}
	tokens = append(tokens, "ab", "Ċ") // Ċ is the byte-level newline the system prefix adds
	for c := 'A'; len(tokens) < ink2RunVocab; c++ {
		tokens = append(tokens, string(c))
	}
	types := make([]int32, len(tokens))
	for i := range types {
		types[i] = 1
	}
	tok, err := tokenizer.FromGGML(tokens, []string{"a b"}, types, "qwen2")
	if err != nil {
		t.Fatal(err)
	}
	prev := runInkling2Tokenizer
	runInkling2Tokenizer = func(*ggufload.File) (*tokenizer.Tokenizer, int, error) { return tok, -1, nil }
	t.Cleanup(func() { runInkling2Tokenizer = prev })
}

func writeInk2RunLlamaGGUF(t *testing.T) string {
	t.Helper()
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
	return path
}

// completeInk2Run runs one raw-prompt turn so the toy vocab needs no template markers.
func completeInk2Run(t *testing.T, p agent.Planner, maxTokens int) string {
	t.Helper()
	c, err := p.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "abc"}}, nil,
		agent.WithRawPrompt(), agent.WithMaxTokens(maxTokens))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if c.Usage.CompletionTokens != maxTokens {
		t.Fatalf("completion tokens = %d, want %d", c.Usage.CompletionTokens, maxTokens)
	}
	return c.Message.Content
}

func TestRunInkling2ArmDecodesSplitSetOnHost(t *testing.T) {
	withInk2RunToyTokenizer(t)
	t.Setenv("FAK_BACKEND", "")
	t.Setenv("FAK_METAL", "")
	_, shard2 := writeInk2RunSplit(t)
	// Any shard of the set selects the arm; only shard 1 carries the config.
	p, release, err := openInkling2Run(shard2, "inkling2-small", "cpu", false, nativeControlConfig{})
	if err != nil || p == nil {
		t.Fatalf("openInkling2Run = %v, %v; want the inkling2 arm", p, err)
	}
	defer release()
	if strings.TrimSpace(completeInk2Run(t, p, 2)) == "" {
		t.Fatal("inkling2 turn produced no token text")
	}
}

func TestRunInkling2BuildRunPlannerRoutesToInklingArm(t *testing.T) {
	withInk2RunToyTokenizer(t)
	t.Setenv("FAK_BACKEND", "")
	t.Setenv("FAK_METAL", "")
	shard1, _ := writeInk2RunSplit(t)
	planner, release := buildRunPlanner(context.Background(), shard1, "cpu", false, nativeControlConfig{})
	if release != nil {
		defer release()
	}
	if _, generic := planner.(*agent.InKernelPlanner); planner == nil || generic {
		t.Fatalf("buildRunPlanner = %T; want the inkling2 run planner", planner)
	}
	completeInk2Run(t, planner, 1)
}

func TestRunInkling2NonInklingGGUFTakesGenericPath(t *testing.T) {
	path := writeInk2RunLlamaGGUF(t)
	notGGUF := filepath.Join(t.TempDir(), "notgguf.gguf")
	if err := os.WriteFile(notGGUF, []byte("not a gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{path, notGGUF} {
		if planner, release, err := openInkling2Run(p, "m", "cpu", false, nativeControlConfig{}); planner != nil || release != nil || err != nil {
			t.Fatalf("openInkling2Run(%s) = %v, %v; want the generic loader", filepath.Base(p), planner, err)
		}
	}
	// The serve sniff must not claim a non-inkling GGUF as inkling2.
	if serveIsInkling2GGUF(path) || serveIsInkling2GGUF(notGGUF) {
		t.Fatal("generic GGUF recognised as inkling2")
	}
}

// Every shard of an arch-inkling2 split set routes to the dedicated serve arm,
// and the generic loaders refuse it with the typed CUDA-route refusal.
func TestRunInkling2ServeSniffsSplitInkling2(t *testing.T) {
	shard1, shard2 := writeInk2RunSplit(t)
	for _, p := range []string{shard1, shard2} {
		if !serveIsInkling2GGUF(p) {
			t.Fatalf("%s not recognised as inkling2", filepath.Base(p))
		}
	}
	_, _, _, _, _, err := loadServeInKernelModelPlacedErr(shard1, nil, false, false, 0, nil, 1, nil)
	var refusal *fakmodel.UnsupportedInkling2LoopExecutionError
	if !errors.As(err, &refusal) || !strings.Contains(err.Error(), "--backend cuda") {
		t.Fatalf("generic serve load of inkling2 = %v; want the typed refusal naming the cuda route", err)
	}
}
