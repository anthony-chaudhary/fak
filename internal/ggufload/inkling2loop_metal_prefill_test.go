//go:build darwin && arm64 && cgo

package ggufload

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// inkling2MetalPrefillPrompt is a chat prompt of exactly n tokens: a system
// turn and one long user turn cut to length before its end marker.
func inkling2MetalPrefillPrompt(t *testing.T, tok *tokenizer.Tokenizer, n int) []int {
	t.Helper()
	control := func(s string) tokenizer.Segment { return tokenizer.Segment{Text: s, Trusted: true} }
	encode := func(segs ...tokenizer.Segment) []int {
		ids, err := tok.EncodeSegments(segs)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	head := encode(control("<|message_system|><|content_text|>"), tokenizer.Segment{Text: "Thinking effort level: 0.9"}, control("<|end_message|>"),
		control("<|message_user|><|content_text|>"))
	tail := encode(control("<|end_message|>"))
	var text strings.Builder
	text.WriteString("Summarize the survey log below in one sentence.\n")
	for i := 1; text.Len() < 16*n; i++ {
		fmt.Fprintf(&text, "Entry %d: the river crosses the old stone bridge near marker %d, where the path turns north.\n", i, 7*i+3)
	}
	body := encode(tokenizer.Segment{Text: text.String()})
	room := n - len(head) - len(tail)
	if room < 1 || room > len(body) {
		t.Fatalf("a %d-token prompt leaves %d body tokens of %d", n, room, len(body))
	}
	return slices.Concat(head, body[:room], tail)
}

// The batched prefill leaves the caches and conv histories the per-token
// resident path does: the greedy continuation of a prompt that spans a chunk
// boundary is the same on both. FAK_INKLING2_PREFILL_TOKENS overrides the
// prompt length.
// fak-test:runtime slow est=180s
func TestInkling2MetalBatchedPrefillMatchesPerToken(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	// One chunk and 25 tokens of a second.
	tokens := inkling2PrefillChunk + 25
	if v := os.Getenv("FAK_INKLING2_PREFILL_TOKENS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 32 {
			t.Fatalf("bad FAK_INKLING2_PREFILL_TOKENS %q", v)
		}
		tokens = n
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
	prompt := inkling2MetalPrefillPrompt(t, tok, tokens)
	const n = 16

	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 8 * time.Minute, Logf: t.Logf})
	if errors.Is(err, gpulease.ErrTimeout) {
		t.Skipf("NOT RUN: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()

	m, be, err := OpenInkling2LoopMetalDevice(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !inkling2ResidentEligible(m, be) {
		t.Fatal("the mapped checkpoint does not run on the resident device forward")
	}
	hp := m.HP
	t.Logf("%d layers (%d cache slots), hidden %d, %d experts of width %d (%d routed), window %d, chunk %d",
		hp.Layers, hp.Layers+(hp.Passes-1)*(hp.CoreEnd-hp.CoreStart), hp.Hidden, hp.Experts, hp.ExpertWidth, hp.ExpertsUsed,
		hp.SlidingWindow, inkling2PrefillChunk)
	run := func(arm, batched string, prompt []int) []int {
		t.Helper()
		t.Setenv("FAK_INKLING2_BATCHED_PREFILL", batched)
		s, err := NewInkling2LoopSession(m, be)
		if err != nil {
			t.Fatalf("%s: %v", arm, err)
		}
		defer s.Close()
		before, err := compute.MetalInkling2DeviceStats()
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		next, err := s.Prefill(context.Background(), prompt)
		if err != nil {
			t.Fatalf("%s: %v", arm, err)
		}
		wall := time.Since(start)
		after, _ := compute.MetalInkling2DeviceStats()
		got := []int{next}
		start = time.Now()
		for len(got) < n {
			if next, err = s.Step(next); err != nil {
				t.Fatalf("%s: %v", arm, err)
			}
			got = append(got, next)
		}
		decode := time.Since(start)
		// Wall time carries the host's load; the GPU span of the submissions does not.
		gpu := after.GPUSeconds - before.GPUSeconds
		t.Logf("resident Metal %s: %d prompt tokens in %s = %.1f prefill tok/s (%.3fs on the GPU = %.1f tok/s, %d submissions, %d launches); %d decode steps in %s (%.2f tok/s); ids %v",
			arm, len(prompt), wall, float64(len(prompt))/wall.Seconds(), gpu, float64(len(prompt))/gpu,
			after.Submissions-before.Submissions, after.Dispatches-before.Dispatches, n-1, decode, float64(n-1)/decode.Seconds(), got)
		return got
	}
	// A short run of each arm compiles its kernels and faults the weights in.
	short := inkling2MetalPrefillPrompt(t, tok, 48)
	if a, b := run("per-token warm-up", "0", short), run("batched warm-up", "1", short); !slices.Equal(a, b) {
		t.Fatalf("48-token prompt: batched prefill decoded %v, per-token %v", b, a)
	}
	want := run("per-token", "0", prompt)
	for pass := range 2 {
		if got := run(fmt.Sprintf("batched pass %d", pass), "1", prompt); !slices.Equal(got, want) {
			text, _ := tok.Decode(got)
			t.Fatalf("batched prefill decoded %v (%q), per-token %v", got, text, want)
		}
	}
	text, _ := tok.Decode(want)
	t.Logf("continuation: %q", text)
}
