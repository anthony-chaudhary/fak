// Command gemma4diag loads a gemma4 GGUF ONCE and sweeps the uncertain forward axes
// (softmax scale 1.0 vs 1/sqrt(d), rope_freqs on/off, decoder-norm plain vs (1+w)),
// printing the top-k next-token predictions for a fixed prompt under each. It is a
// throwaway debugging aid for bringing up the gemma4 forward against a known answer
// ("The capital of France is" -> " Paris"). DELETE after the forward is witnessed.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func main() {
	gguf := flag.String("gguf", "", "gguf path")
	prompt := flag.String("prompt", "The capital of France is", "completion prompt")
	idsFlag := flag.String("ids", "", "comma-separated raw token ids (bypass the tokenizer; for parity vs a reference tokenization)")
	gen := flag.Int("gen", 0, "if >0, greedily generate this many tokens (cacheless) and decode")
	loadMode := flag.String("load", "q8", "weight residency: q8 (quantize activations) | q4k (f32 activations) | f32")
	sweep := flag.Bool("sweep", false, "sweep scale/rope_freqs/norm-gain hypotheses (one load) instead of per-layer stats")
	topk := flag.Int("topk", 10, "top-k tokens to print per config")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*gguf = pathutil.ExpandTilde(*gguf)
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
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		fmt.Fprintln(os.Stderr, "FromGGML:", err)
		os.Exit(1)
	}
	var ids []int
	if *idsFlag != "" {
		for _, s := range strings.Split(*idsFlag, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil {
				fmt.Fprintln(os.Stderr, "bad id:", s)
				os.Exit(2)
			}
			ids = append(ids, n)
		}
		fmt.Printf("raw ids=%v\n", ids)
	} else {
		pids, _ := tok.Encode(*prompt)
		ids = pids
		if gt.BOS >= 0 {
			ids = append([]int{gt.BOS}, pids...)
		}
		fmt.Printf("prompt=%q ids=%v\n", *prompt, ids)
	}

	fmt.Printf("loading (%s, ~minutes for 12B)...\n", *loadMode)
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
	fmt.Println("loaded.")

	if *sweep {
		type cfgCase struct {
			name       string
			normGain1p bool
			scaleSqrt  bool
			noRopeFreq bool
		}
		cases := []cfgCase{
			{"baseline (scale=1, ropefreqs, plain-norm)", false, false, false},
			{"scale=1/sqrt(d)", false, true, false},
			{"no rope_freqs", false, false, true},
			{"(1+w) norms", true, false, false},
			{"scale=1/sqrt(d) + no rope_freqs", false, true, true},
		}
		// also a single-BOS forward to probe the basic (attention-trivial) path.
		bosAct := m.Forward([]int{ids[0]})
		fmt.Printf("\n=== single-token [%d] top-%d (basic path) ===\n", ids[0], *topk)
		printTopK(tok, bosAct.Logits[0], *topk)
		for _, c := range cases {
			setEnv("FAK_GEMMA4_SCALE_SQRT", c.scaleSqrt)
			setEnv("FAK_GEMMA4_NO_ROPEFREQS", c.noRopeFreq)
			m.Cfg.NormGain1p = c.normGain1p
			a := m.Forward(ids)
			fmt.Printf("\n=== %s ===\n", c.name)
			printTopK(tok, a.Logits[len(a.Logits)-1], *topk)
		}
		return
	}

	// One baseline forward, then per-layer hidden-state magnitude (last position) to
	// localize where the residual stream explodes.
	H := m.Cfg.HiddenSize
	act := m.Forward(ids)
	last := len(ids) - 1
	fmt.Printf("\n=== per-layer hidden stats (last token, layer_type) ===\n")
	for l := 0; l < len(act.Hidden); l++ {
		h := act.Hidden[l]
		row := h[last*H : (last+1)*H]
		var ss, mx float64
		for _, v := range row {
			ss += float64(v) * float64(v)
			if a := math.Abs(float64(v)); a > mx {
				mx = a
			}
		}
		lt := ""
		if l >= 1 && l-1 < len(m.Cfg.LayerTypes) {
			lt = m.Cfg.LayerTypes[l-1]
		}
		label := "embed"
		if l >= 1 {
			label = fmt.Sprintf("after L%-2d %s", l-1, lt)
		}
		fmt.Printf("  %-28s L2=%.3e maxabs=%.3e\n", label, math.Sqrt(ss), mx)
	}
	fmt.Printf("\n=== baseline top-%d ===\n", *topk)
	printTopK(tok, act.Logits[len(act.Logits)-1], *topk)

	if *gen > 0 {
		seq := append([]int(nil), ids...)
		var out []int
		for i := 0; i < *gen; i++ {
			a := m.Forward(seq)
			next := argmax(a.Logits[len(a.Logits)-1])
			out = append(out, next)
			one, _ := tok.Decode([]int{next})
			fmt.Printf("  gen %2d: id=%-7d decode=%q\n", i, next, one)
			seq = append(seq, next)
		}
		full, _ := tok.Decode(out)
		fmt.Printf("\n=== greedy continuation ===\n%q\n", full)
	}
}

func setEnv(k string, v bool) {
	if v {
		os.Setenv(k, "1")
	} else {
		os.Unsetenv(k)
	}
}

func argmax(v []float32) int {
	best, idx := float32(-math.MaxFloat32), 0
	for i, x := range v {
		if x > best {
			best, idx = x, i
		}
	}
	return idx
}

func printTopK(tok *tokenizer.Tokenizer, logits []float32, k int) {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	for i := 0; i < k && i < len(idx); i++ {
		id := idx[i]
		s, _ := tok.Decode([]int{id})
		fmt.Printf("  %2d. id=%-7d logit=%+.4f decode=%q\n", i, id, logits[id], s)
	}
}
