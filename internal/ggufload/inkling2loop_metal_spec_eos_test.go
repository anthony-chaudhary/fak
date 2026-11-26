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

// FAK_INKLING2_TEST_GGUF runs the resident Metal greedy decode the way fak run
// does with FAK_INKLING2_SPEC unset: the default lookup speculation must decode
// the ids of FAK_INKLING2_SPEC=0, and both must stop at the first EOS instead of
// running to the token limit.
//
// fak-test:runtime slow est=120s
func TestInkling2MetalSpecDefaultStopsAtEOS(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 20 * time.Minute, Logf: t.Logf})
	if errors.Is(err, gpulease.ErrTimeout) {
		t.Skipf("NOT RUN: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()

	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	table, ok := f.GGMLTokenizer()
	if !ok || table.EOS < 0 {
		t.Fatal("checkpoint has no embedded BPE tokenizer and EOS")
	}
	tok, err := tokenizer.FromGGML(table.Tokens, table.Merges, table.TokenTypes, table.Pre)
	if err != nil {
		t.Fatal(err)
	}
	m, be, err := OpenInkling2LoopMetalDevice(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.SpecDefault = Inkling2MetalSpecDefault(be)
	if m.SpecDefault.Depth == 0 {
		t.Fatal("the Metal backend has no default speculation")
	}
	control := func(s string) tokenizer.Segment { return tokenizer.Segment{Text: s, Trusted: true} }
	// the fak run chat template at its default effort
	chat := func(user string) []int {
		ids, err := tok.EncodeSegments([]tokenizer.Segment{
			control("<|message_system|><|content_text|>"), {Text: "Thinking effort level: 0.9"}, control("<|end_message|>"),
			control("<|message_user|><|content_text|>"), {Text: user}, control("<|end_message|>"),
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	const n = 256
	workloads := []struct {
		name   string
		prompt []int
	}{
		{"math", chat("What is 2 + 2? Reply with the number only.")},
		{"code edit", chat("Rename the variable acc to total in this Go function and reply with the whole function only, no prose:\n\nfunc sumPositive(xs []int) int {\n\tacc := 0\n\tfor _, x := range xs {\n\t\tif x > 0 {\n\t\t\tacc += x\n\t\t}\n\t}\n\treturn acc\n}")},
	}
	t.Setenv("FAK_INKLING2_SPEC_DRAFT", "")
	os.Unsetenv("FAK_INKLING2_SPEC_DRAFT")
	// the first run compiles every kernel and faults the weights in
	t.Setenv("FAK_INKLING2_SPEC", "0")
	if _, _, err := Inkling2LoopGreedyUntil(m, be, workloads[0].prompt, 4, table.EOS); err != nil {
		t.Fatal(err)
	}
	for _, w := range workloads {
		t.Setenv("FAK_INKLING2_SPEC", "0")
		plain, pst, err := Inkling2LoopGreedyUntil(m, be, w.prompt, n, table.EOS)
		if err != nil {
			t.Fatalf("%s plain: %v", w.name, err)
		}
		os.Unsetenv("FAK_INKLING2_SPEC")
		spec, sst, err := Inkling2LoopGreedyUntil(m, be, w.prompt, n, table.EOS)
		if err != nil {
			t.Fatalf("%s default spec: %v", w.name, err)
		}
		ss := inkling2LastSpec
		t.Logf("%s (%d prompt tokens): plain %d ids, prefill %s, %.2f tok/s; default spec %d ids, prefill %s, %.2f tok/s, %d verifies accepted %d/%d; %s",
			w.name, len(w.prompt), len(plain), pst.Prefill, pst.DecodeTokensPerSecond(), len(spec), sst.Prefill, sst.DecodeTokensPerSecond(),
			ss.Verifies, ss.Accepted, ss.Drafted, inkling2MetalSpecLoad())
		if !slices.Equal(spec, plain) {
			t.Errorf("%s: default spec decoded %v, FAK_INKLING2_SPEC=0 %v", w.name, spec, plain)
		}
		if sst.DecodeSteps != pst.DecodeSteps {
			t.Errorf("%s: default spec ran %d decode steps, plain %d", w.name, sst.DecodeSteps, pst.DecodeSteps)
		}
		if slices.Contains(plain, table.EOS) {
			t.Errorf("%s: reply carries EOS %d", w.name, table.EOS)
		}
		if w.name == "math" && len(plain) >= n {
			t.Errorf("math: decoded all %d tokens without stopping at EOS", n)
		}
		if len(plain) < n {
			// the stop is the first EOS of the unstopped run
			t.Setenv("FAK_INKLING2_SPEC", "0")
			full, _, err := Inkling2LoopGreedyUntil(m, be, w.prompt, len(plain)+1, -1)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(full[:len(plain)], plain) || full[len(plain)] != table.EOS {
				t.Errorf("%s: stopped run %v is not the unstopped run %v cut at EOS %d", w.name, plain, full, table.EOS)
			}
		}
	}
}
