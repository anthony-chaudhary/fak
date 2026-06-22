package main

// q4kdiag — P1 correctness diagnostic. Loads Qwen3.6-27B-Q4_K_M via LoadModelQ4K, reports
// resident-q4k population/shapes, prefills the exact 22-token "Say OK." oracle prompt, and
// prints the top-5 first-token logits. The llama.cpp q4_k_m oracle (and fak's Q8 path) put
// 248068 (<think>) first at logit ~28.3; a sane resident-Q4_K path must agree closely.
//
// Run: FAK_Q4K=1 go run ./cmd/q4kdiag -gguf <Qwen3.6-27B.q4_k_m.gguf>
import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// prompt_ids from llamacpp-qwen36-multitoken-oracle-20260619.json (ChatML "Say OK.").
var oraclePrompt = []int{
	248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198,
	248045, 846, 198, 44240, 10092, 13, 248046, 198, 248045, 74455, 198,
}

func main() {
	gguf := flag.String("gguf", "", "GGUF path")
	membw := flag.Float64("membw", 0, "machine memory bandwidth GB/s; if >0, print the bandwidth-bound decode tok/s ceiling")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*gguf = pathutil.ExpandTilde(*gguf)
	if *gguf == "" {
		fmt.Fprintln(os.Stderr, "usage: q4kdiag -gguf <model.gguf> [-membw 100]")
		os.Exit(2)
	}
	t0 := time.Now()
	m, err := ggufload.LoadModelQ4K(*gguf)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	loadMS := time.Since(t0).Milliseconds()
	// Resident inventory + decode-bandwidth stream: SEE the loader's memory shape + the
	// predicted decode ceiling WITHOUT running generation. This is the small observability
	// surface for the Q4_K path (27B-free at the logic level; the 27B run only confirms it).
	rep := m.ResidentReport()
	fmt.Fprintf(os.Stderr, "load=%dms  %s\n", loadMS, model.FormatResidentReport(rep))
	if *membw > 0 {
		fmt.Fprintf(os.Stderr, "decode ceiling @ %.0f GB/s = %.2f tok/s (bandwidth-bound; q4_k_m bar = 7.29)\n",
			*membw, rep.DecodeTokSCeiling(*membw))
	}
	// Dump a few resident Q4_K shapes (linear_attn names + an FFN) to sanity-check routing.
	for _, n := range []string{
		"model.layers.0.linear_attn.in_proj_z.weight",
		"model.layers.0.linear_attn.out_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
	} {
		o, i := m.Q4KShape(n)
		fmt.Fprintf(os.Stderr, "  q4k[%s] out=%d in=%d\n", n, o, i)
	}

	s := m.NewSession()
	s.Quant = true
	s.Q4K = true
	logits := s.Prefill(oraclePrompt)
	top := topK(logits, 8)
	fmt.Fprintf(os.Stderr, "first-token top-8 (oracle wants 248068=<think> ~28.3):\n")
	for _, t := range top {
		fmt.Fprintf(os.Stderr, "  id=%-7d logit=%.4f\n", t.id, t.logit)
	}

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Fprintf(os.Stderr, "Go heap=%.2fGB sys=%.2fGB\n", float64(ms.Alloc)/1e9, float64(ms.Sys)/1e9)
}

type kv struct {
	id    int
	logit float32
}

func topK(logits []float32, k int) []kv {
	out := make([]kv, len(logits))
	for i, v := range logits {
		out[i] = kv{i, v}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].logit > out[j].logit })
	if k > len(out) {
		k = len(out)
	}
	return out[:k]
}
