// Command q4kdiag runs the P1 Q4_K correctness diagnostic for a GGUF model.
package main

// It loads Qwen3.6-27B-Q4_K_M via LoadModelQ4K, reports
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

	"github.com/anthony-chaudhary/fak/internal/compute"
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
	planOnly := flag.Bool("plan-only", false, "print header-only GGUF memory plans and exit before loading tensors")
	expertParallel := flag.Int("expert-parallel", 1, "expert-parallel ranks for -plan-only per-rank memory plan")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*gguf = pathutil.ExpandTilde(*gguf)
	if *gguf == "" {
		fmt.Fprintln(os.Stderr, "usage: q4kdiag -gguf <model.gguf> [-membw 100] [-plan-only -expert-parallel N]")
		os.Exit(2)
	}
	if *planOnly {
		if err := printHeaderMemoryPlans(*gguf, *expertParallel); err != nil {
			fmt.Fprintln(os.Stderr, "plan:", err)
			os.Exit(1)
		}
		return
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

func printHeaderMemoryPlans(path string, expertParallel int) error {
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		return err
	}
	defer ws.Close()

	full, err := ws.EstimateLoadMemoryPlan()
	if err != nil {
		return fmt.Errorf("monolith load: %w", err)
	}
	ep, err := ws.EstimateExpertParallelLoadMemoryPlan(expertParallel)
	if err != nil {
		return fmt.Errorf("expert-parallel load: %w", err)
	}

	fmt.Fprintf(os.Stdout, "gguf tensors=%d\n", len(ws.File.Tensors))
	if cfg, err := ws.File.Config(); err == nil {
		fmt.Fprintln(os.Stdout, formatConfigLine(cfg))
	}
	// Lane F (#3074): fold the header's expert axis into the DERIVED active routed-expert
	// bytes/token — the single-stream decode ceiling divisor. q4kdiag already holds K and the
	// routed-expert band header-only, so this is the box-agnostic half of the ground-truth read;
	// the box-side per-op byte trace remains the separate witness.
	if as, ok, err := ws.RoutedExpertActiveSet(); err == nil && ok {
		fmt.Fprintln(os.Stdout, formatActiveSetLine(as))
	}
	printMemoryPlan("monolith", full)
	printMemoryPlan(fmt.Sprintf("expert_parallel_per_rank ranks=%d", expertParallel), ep)
	return nil
}

// formatConfigLine renders the one-line header-config summary q4kdiag -plan-only emits. It
// carries the full MoE active-set axis, not just the total expert count: experts_used (K,
// cfg.NumExpertsPerTok) and expert_ffn_len (cfg.MoEIntermediateSize) are the two header scalars
// Lane F (#3074) needs to derive active-bytes/token off the roofline estimate. The loader already
// reads all three into cfg (ggufload.applyMoEExpertCounts); surfacing them here means no future
// ceiling re-derivation is blocked on a fresh operator header read.
func formatConfigLine(cfg model.Config) string {
	return fmt.Sprintf("config model_type=%s layers=%d hidden=%d experts=%d experts_used=%d expert_ffn_len=%d",
		cfg.ModelType, cfg.NumLayers, cfg.HiddenSize, cfg.NumExperts, cfg.NumExpertsPerTok, cfg.MoEIntermediateSize)
}

// formatActiveSetLine renders the DERIVED Lane F (#3074) active-set: the routed-expert resident
// band, the per-expert resident bytes, the non-expert (attention/dense/shared/embedding) remainder,
// and — once experts_used (K) is read — the two per-token roofline divisors the GPU-server roofline
// needs MEASURED rather than estimated: active-bytes/token (K×per-expert + non-expert stream, the
// single-stream decode divisor, an upper bound) and active-params/token (K×per-expert params +
// non-expert params, the FLOP divisor). It is header arithmetic (no serve, no per-op trace); when K
// is unread (experts_used=0) it prints the band + per-expert + non-expert and flags the per-token
// divisors PENDING(K) rather than guessing.
func formatActiveSetLine(as ggufload.RoutedExpertActiveSet) string {
	base := fmt.Sprintf("active_set experts=%d routed_expert_resident=%.2fGiB per_expert=%.4fGiB non_expert_resident=%.2fGiB",
		as.NumExperts, bytesGiB(as.RoutedResident), bytesGiB(as.PerExpert), bytesGiB(as.NonExpertResident))
	if as.ExpertsUsed <= 0 {
		return base + " experts_used=0 active_bytes_per_tok=PENDING(K) active_params_per_tok=PENDING(K)"
	}
	return fmt.Sprintf("%s experts_used=%d active_bytes_per_tok=%.4fGiB active_params_per_tok=%.2fB (DERIVED header arithmetic; box-side per-op trace is the witness)",
		base, as.ExpertsUsed, bytesGiB(as.ActiveBytesPerToken), paramsB(as.ActiveParamsPerToken))
}

// paramsB renders an element count as billions of parameters (the roofline's active-params unit).
func paramsB(n int64) float64 {
	if n <= 0 {
		return 0
	}
	return float64(n) / 1e9
}

func printMemoryPlan(name string, plan compute.MemoryPlan) {
	fmt.Fprintf(os.Stdout, "%s device_total=%d bytes (%.2f GiB) total=%d bytes (%.2f GiB)\n",
		name, plan.DeviceTotal(), bytesGiB(plan.DeviceTotal()), plan.Total(), bytesGiB(plan.Total()))
	for _, d := range plan {
		scope := d.ScopeOrDefault()
		dtype := d.DType
		if dtype == "" {
			dtype = "unknown"
		}
		fmt.Fprintf(os.Stdout, "  detail=%s class=%s scope=%s dtype=%s bytes=%d (%.2f GiB)\n",
			d.Detail, d.Class, scope, dtype, d.Bytes, bytesGiB(d.Bytes))
	}
}

func bytesGiB(n int64) float64 {
	if n <= 0 {
		return 0
	}
	return float64(n) / (1024 * 1024 * 1024)
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
