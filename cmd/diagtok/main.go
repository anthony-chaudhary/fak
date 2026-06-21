// Command diagtok is a throwaway diagnostic: it loads a GGUF via the same path
// simpledemo uses, then (1) round-trips a known string through the embedded
// tokenizer and (2) greedily decodes a ChatML prompt while printing each raw
// token id next to its decode. This disambiguates a broken forward pass
// (degenerate/looping token ids) from a broken tokenizer (sensible ids, garbled
// text). DELETE after diagnosis.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func argmax(v []float32) int {
	best, idx := float32(-math.MaxFloat32), 0
	for i, x := range v {
		if x > best {
			best, idx = x, i
		}
	}
	return idx
}

func main() {
	gguf := flag.String("gguf", "", "gguf path")
	n := flag.Int("n", 16, "tokens to greedily decode")
	quant := flag.Bool("quant", true, "use Q8 session")
	cacheless := flag.Bool("cacheless", false, "greedy-decode via repeated cacheless Forward (for arches without a cached Session path yet, e.g. gemma4)")
	promptFlag := flag.String("prompt", "", "override the decode prompt (raw text; no chat template applied)")
	loadMode := flag.String("load", "q8", "weight residency: q8 | q4k (f32 activations) | f32")
	flag.Parse()
	if *gguf == "" {
		fmt.Fprintln(os.Stderr, "need -gguf")
		os.Exit(2)
	}

	f, err := ggufload.Open(*gguf)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		fmt.Fprintln(os.Stderr, "no embedded tokenizer")
		os.Exit(1)
	}
	fmt.Printf("tokenizer: pre=%q ntokens=%d nmerges=%d\n", gt.Pre, len(gt.Tokens), len(gt.Merges))
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		fmt.Fprintln(os.Stderr, "FromGGML:", err)
		os.Exit(1)
	}

	// (1) round-trip
	probe := "The capital of France is Paris."
	ids, err := tok.Encode(probe)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode:", err)
		os.Exit(1)
	}
	back, _ := tok.Decode(ids)
	fmt.Printf("\n[roundtrip] in=%q\n  ids=%v\n  out=%q  match=%v\n", probe, ids, back, back == probe)
	// per-id decode
	var parts []string
	for _, id := range ids {
		s, _ := tok.Decode([]int{id})
		parts = append(parts, fmt.Sprintf("%d:%q", id, s))
	}
	fmt.Printf("  per-id: %s\n", strings.Join(parts, " "))

	// (2) greedy decode. Q4_K-resident keeps activations in f32 (the q8 path quantizes
	// activations per-vector, which is lossy for Gemma 4's high-dynamic-range norms).
	var m *model.Model
	switch *loadMode {
	case "f32":
		m, err = ggufload.LoadModel(*gguf)
	case "q4k":
		m, err = ggufload.LoadModelQ4K(*gguf)
	default:
		m, err = ggufload.LoadModelQuant(*gguf)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	prompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
	if *promptFlag != "" {
		prompt = *promptFlag
	}
	pids, _ := tok.Encode(prompt)
	fmt.Printf("\n[prompt] %d ids; first8=%v last8=%v\n", len(pids), pids[:min(8, len(pids))], pids[max(0, len(pids)-8):])

	if *cacheless {
		// Cacheless greedy: re-run the full Forward over the growing sequence each step.
		// O(n^2) and slow, but it exercises arches that only have a cacheless Forward
		// path so far (e.g. gemma4), independent of the cached Session/KV machinery.
		ids := append([]int(nil), pids...)
		// Gemma sets add_bos_token=true; prepend <bos> unless the prompt already has it.
		if gt.BOS >= 0 && (len(ids) == 0 || ids[0] != gt.BOS) {
			ids = append([]int{gt.BOS}, ids...)
			fmt.Printf("[prepended bos=%d]\n", gt.BOS)
		}
		var gen []int
		for i := 0; i < *n; i++ {
			act := m.Forward(ids)
			logits := act.Logits[len(act.Logits)-1]
			next := argmax(logits)
			gen = append(gen, next)
			one, _ := tok.Decode([]int{next})
			fmt.Printf("  step %2d: id=%-7d decode=%q\n", i, next, one)
			ids = append(ids, next)
		}
		full, _ := tok.Decode(gen)
		fmt.Printf("\n[greedy full] %q\n", full)
		return
	}

	s := m.NewSession()
	s.Quant = *quant
	logits := s.Prefill(pids)
	fmt.Printf("[forward] vocab=%d  argmax0=%d  (top via argmax each step):\n", len(logits), argmax(logits))
	var gen []int
	for i := 0; i < *n; i++ {
		next := argmax(logits)
		gen = append(gen, next)
		one, _ := tok.Decode([]int{next})
		fmt.Printf("  step %2d: id=%-7d decode=%q\n", i, next, one)
		logits = s.Step(next)
	}
	full, _ := tok.Decode(gen)
	fmt.Printf("\n[greedy full] %q\n", full)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
