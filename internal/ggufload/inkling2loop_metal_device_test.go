//go:build darwin && arm64 && cgo

package ggufload

import (
	"errors"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime slow est=120s
func TestInkling2MetalDeviceGreedyMatchesHost(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	table, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatal("checkpoint has no embedded BPE tokenizer")
	}
	tok, err := tokenizer.FromGGML(table.Tokens, table.Merges, table.TokenTypes, table.Pre)
	if err != nil {
		t.Fatal(err)
	}
	control := func(s string) tokenizer.Segment { return tokenizer.Segment{Text: s, Trusted: true} }
	prompt, err := tok.EncodeSegments([]tokenizer.Segment{
		control("<|message_system|><|content_text|>"), {Text: "Thinking effort level: 0.9"}, control("<|end_message|>"),
		control("<|message_user|><|content_text|>"), {Text: "Name the capital of France."}, control("<|end_message|>"),
	})
	if err != nil {
		t.Fatal(err)
	}
	const n = 16

	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 6 * time.Minute, Logf: t.Logf})
	if errors.Is(err, gpulease.ErrTimeout) {
		t.Skipf("NOT RUN: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()

	host, err := OpenInkling2Loop(path)
	if err != nil {
		t.Fatal(err)
	}
	want, hostStats, err := Inkling2LoopGreedyTimed(host, nil, prompt, n)
	host.Close()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("host: %d prompt tokens in %s, %d decode steps in %s (%.2f tok/s), ids %v",
		len(prompt), hostStats.Prefill, hostStats.DecodeSteps, hostStats.Decode, hostStats.DecodeTokensPerSecond(), want)

	m, be, err := OpenInkling2LoopMetalDevice(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// The first run compiles every kernel and faults the weights in.
	for run := range 2 {
		got, st, err := Inkling2LoopGreedyTimed(m, be, prompt, n)
		if err != nil {
			t.Fatal(err)
		}
		text, _ := tok.Decode(got)
		t.Logf("resident Metal run %d: %d prompt tokens in %s, %d decode steps in %s (%.2f tok/s), ids %v, text %q",
			run, len(prompt), st.Prefill, st.DecodeSteps, st.Decode, st.DecodeTokensPerSecond(), got, text)
		if !slices.Equal(got, want) {
			t.Fatalf("resident Metal run %d decoded %v, host %v", run, got, want)
		}
	}
}
