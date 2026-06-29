// Command batchbench measures the AGGREGATE decode throughput of the multi-user batched
// decode (internal/model.BatchSession) as a function of batch size B — the "continuous
// batching" / multi-user serving regime MODEL-BASELINE-RESULTS.md scoped out as "vLLM's
// claim, not fak's". This is the throughput lane: batch-1 decode is memory-bandwidth-bound
// (the weights are re-streamed per token), so stacking B users' decode steps into one GEMM
// per layer amortises that weight stream B-fold and aggregate tokens/sec scales with B until
// the GEMM goes compute-bound.
//
// The headline number is the THROUGHPUT MULTIPLIER: aggregate tok/s at the best batch size
// divided by the batch-1 (serial-equivalent) tok/s — and, for the cumulative story, divided
// by the naive f32-serial baseline (52.1 ms/tok = 19.2 tok/s) the whole optimisation effort
// started from in Act 1.
//
// Apples-to-apples: every B runs the SAME per-user work (one short prompt prefill + D decode
// steps). Token VALUES never affect matmul/attention cost, so deterministic LCG ids measure
// the identical work a real fleet of B concurrent users would drive. Pin FAK_WORKERS to fix
// the core budget; default uses GOMAXPROCS.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/intlist"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// lcgIDs builds n deterministic token ids in [0,vocab) — same recurrence as modelbench so the
// inputs are comparable across the two benchmarks.
func lcgIDs(n, vocab, seed int) []int {
	ids := make([]int, n)
	state := uint64(2463534242 + seed)
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

func capPositive(n, cap int) int {
	if cap > 0 && n > cap {
		return cap
	}
	if n < 1 {
		return 1
	}
	return n
}

// bestMS returns the MINIMUM per-step time across reps. On a shared box, contention from
// other tenants only ever SLOWS a step down, so the fastest observed step is the closest
// estimate of the true (isolated) hardware capability — the "least-contended sampling" the
// MODEL-BASELINE methodology already uses for the bandwidth-sensitive decode numbers.
func bestMS(ds []time.Duration) float64 {
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return float64(cp[0].Nanoseconds()) / 1e6
}

// naiveSerialTokPerSec is the Act-1 naive f32 serial decode baseline (52.1 ms/tok), the
// starting point of MODEL-BASELINE-RESULTS.md — quoted so the cumulative multiplier (Q8 +
// parallel + multi-user batching) can be reported against the honest origin.
const naiveSerialMsPerTok = 52.1

func hfName(cfg model.Config, dir string) string {
	base := filepath.Base(strings.TrimRight(dir, `/\`))
	if cfg.ModelType == "" {
		return base
	}
	return fmt.Sprintf("%s (%s)", base, cfg.ModelType)
}

func hfSource(dir string) string {
	index := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(index); err == nil {
		return index
	}
	return filepath.Join(dir, "model.safetensors")
}

type batchPoint struct {
	Batch           int     `json:"batch"`
	DecodeSteps     int     `json:"decode_steps"`
	Reps            int     `json:"reps"`
	StepMS          float64 `json:"step_ms"`                 // best (least-contended) wall-clock per batched decode step
	PerUserMsPerTok float64 `json:"per_user_ms_per_tok"`     // step_ms / B — latency a single user sees
	AggTokPerSec    float64 `json:"agg_tok_per_sec"`         // B / step_ms — the throughput headline
	SpeedupVsB1     float64 `json:"speedup_vs_b1"`           // agg tok/s ÷ real serial batch-1 decode tok/s
	SpeedupVsNaive  float64 `json:"speedup_vs_naive_serial"` // agg tok/s ÷ 19.2 (naive f32 serial, Act 1)
}

// loadBenchModel loads the benchmark model from either a HuggingFace snapshot
// dir (hf, which overrides) or a fak export dir (dir), returning the model plus
// the display name and source path the report records. It exits the process on a
// load failure (the only sensible action for a benchmark CLI with no model).
func loadBenchModel(hf, dir string) (*model.Model, string, string) {
	if hf != "" {
		cfg, err := benchcli.ReadHFConfig(hf)
		if err != nil {
			fmt.Fprintln(os.Stderr, "load:", err)
			os.Exit(1)
		}
		m, err := model.LoadSafetensorsDir(hf, cfg)
		if err != nil {
			fmt.Fprintln(os.Stderr, "load: safetensors:", err)
			os.Exit(1)
		}
		return m, hfName(cfg, hf), hfSource(hf)
	}
	m, err := model.Load(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	return m, filepath.Base(strings.TrimRight(dir, `/\`)), filepath.Join(dir, "weights.f32")
}

func main() {
	dir := flag.String("dir", "internal/model/.cache/smollm2-135m", "model export dir")
	hf := flag.String("hf", "", "HuggingFace snapshot dir (config.json + model.safetensors); overrides -dir")
	out := flag.String("out", "", "write JSON result here (default stdout)")
	reps := flag.Int("reps", 5, "reps per batch size (median over per-step wall-clock)")
	decodeSteps := flag.Int("decode-steps", 16, "decode steps to time per user")
	promptLen := flag.Int("prompt", 16, "prompt length prefilled per user before decode")
	quant := flag.Bool("quant", false, "use the Q8_0 quantized lane (else f32)")
	batchesArg := flag.String("batches", "1,2,4,8,16,32,64,128,256,512,768,896,960,1024", "comma-separated batch sizes to sweep")
	workloadPath := flag.String("workload", "", "optional recorded agent workload JSON; uses recorded prompt lengths per user")
	workloadPromptCap := flag.Int("workload-prompt-cap", 0, "cap recorded workload prompt lengths for smoke runs (0 = full recorded length)")
	budget := flag.Float64("budget", 0, "fractional core budget: 0.75 = use up to 75% of the machine's logical cores (portable; 75 or 0.75 accepted). 0 = unset. FAK_WORKERS still overrides.")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*dir = pathutil.ExpandTilde(*dir)
	*hf = pathutil.ExpandTilde(*hf)

	if *budget > 0 {
		if os.Getenv("FAK_WORKERS") != "" {
			fmt.Fprintf(os.Stderr, "[fak] FAK_WORKERS is set; ignoring -budget %g (absolute override wins)\n", *budget)
		} else if err := model.SetWorkerBudget(*budget); err != nil {
			fmt.Fprintln(os.Stderr, "budget:", err)
			os.Exit(2)
		}
	}

	batches := intlist.Parse(*batchesArg)
	var workload *model.BenchWorkload
	if *workloadPath != "" {
		var err error
		workload, err = model.LoadBenchWorkload(*workloadPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "workload:", err)
			os.Exit(1)
		}
	}

	m, modelName, modelSource := loadBenchModel(*hf, *dir)
	vocab := m.Cfg.VocabSize
	if *quant {
		m.Quantize()
	}
	promptIDs := func(caseIdx, seed int) []int {
		n := *promptLen
		if workload != nil {
			c := workload.Cases[caseIdx%len(workload.Cases)]
			n = capPositive(c.PromptTokens, *workloadPromptCap)
		}
		return lcgIDs(n, vocab, seed)
	}

	// Warm up: page in the weights + JIT the allocation paths so we time steady state.
	{
		bs := m.NewBatchSession(2)
		bs.SetQuant(*quant)
		bs.PrefillEach([][]int{promptIDs(0, 1), promptIDs(1, 2)})
		bs.StepBatch([]int{0, 1})
	}

	// Honest batch-1 baseline: the REAL single-user decode path (Session.Step → qdot8 SIMD for
	// Q8 / parMatRows for f32), NOT stepBatchB(1). At B<4 the batched Q8 GEMM tile (NR=4) falls
	// to its scalar remainder path, so stepBatch(1) is pathologically slow and would FLATTER
	// the multiplier; the fair denominator is the production single-stream decode.
	b1MsPerTok := func() float64 {
		ds := make([]time.Duration, *reps)
		for r := 0; r < *reps; r++ {
			s := m.NewSession()
			s.Quant = *quant
			s.Prefill(promptIDs(0, r))
			id := r % vocab
			t := time.Now()
			for i := 0; i < *decodeSteps; i++ {
				logits := s.Step(id)
				id = (id*48271 + 1) % vocab
				_ = logits
			}
			ds[r] = time.Since(t) / time.Duration(*decodeSteps)
		}
		return bestMS(ds)
	}()
	b1Agg := 1000.0 / b1MsPerTok
	fmt.Fprintf(os.Stderr, "[batchbench] batch-1 serial decode (Session.Step): %.2f ms/tok = %.1f tok/s\n", b1MsPerTok, b1Agg)

	precision := "f32"
	engine := "fak multi-user batched decode (pure-Go, matMulBatch GEMM, kernel-owned per-user KV)"
	if *quant {
		precision = "Q8_0"
		engine = "fak multi-user batched decode Q8_0 (pure-Go, register-blocked int8 tile GEMM)"
	}

	var points []batchPoint
	for _, B := range batches {
		ds := make([]time.Duration, *reps)
		for r := 0; r < *reps; r++ {
			bs := m.NewBatchSession(B)
			bs.SetQuant(*quant)
			prompts := make([][]int, B)
			for b := 0; b < B; b++ {
				prompts[b] = promptIDs(b, r*1000+b)
			}
			bs.PrefillEach(prompts)
			bs.Reserve(*decodeSteps)
			ids := make([]int, B)
			for b := range ids {
				ids[b] = b % vocab
			}
			t := time.Now()
			for i := 0; i < *decodeSteps; i++ {
				logits := bs.StepBatch(ids)
				for b := 0; b < B; b++ {
					ids[b] = (ids[b]*48271 + 1) % vocab // deterministic, cost-irrelevant
				}
				_ = logits
			}
			ds[r] = time.Since(t) / time.Duration(*decodeSteps) // per-step wall-clock
		}
		stepMS := bestMS(ds)
		agg := float64(B) / (stepMS / 1e3)
		points = append(points, batchPoint{
			Batch: B, DecodeSteps: *decodeSteps, Reps: *reps,
			StepMS:          stepMS,
			PerUserMsPerTok: stepMS / float64(B),
			AggTokPerSec:    agg,
			SpeedupVsB1:     agg / b1Agg,
			SpeedupVsNaive:  agg / (1000.0 / naiveSerialMsPerTok),
		})
		fmt.Fprintf(os.Stderr, "[batchbench %s] B=%-4d step=%7.2f ms  per-user=%6.2f ms/tok  agg=%8.1f tok/s  (%.1fx vs b1, %.1fx vs naive)\n",
			precision, B, stepMS, stepMS/float64(B), agg, agg/b1Agg, agg/(1000.0/naiveSerialMsPerTok))
	}

	// headline: peak aggregate throughput and its multipliers.
	peak := points[0]
	for _, p := range points {
		if p.AggTokPerSec > peak.AggTokPerSec {
			peak = p
		}
	}
	report := map[string]any{
		"app_version":             appversion.Current(),
		"engine":                  engine,
		"model":                   modelName,
		"source":                  modelSource,
		"precision":               precision,
		"workers":                 model.NumWorkers(),
		"budget":                  model.WorkerBudget(),
		"go_threads":              fmt.Sprintf("GOMAXPROCS=%d, matmul workers=%d", runtime.GOMAXPROCS(0), model.NumWorkers()),
		"baseline_b1_ms_per_tok":  b1MsPerTok,
		"baseline_b1_tok_per_sec": b1Agg,
		"naive_serial_ms_per_tok": naiveSerialMsPerTok,
		"points":                  points,
		"peak": map[string]any{
			"batch":                   peak.Batch,
			"agg_tok_per_sec":         peak.AggTokPerSec,
			"speedup_vs_b1":           peak.SpeedupVsB1,
			"speedup_vs_naive_serial": peak.SpeedupVsNaive,
		},
	}
	if workload != nil {
		report["workload"] = map[string]any{
			"path":               *workloadPath,
			"schema":             workload.Schema,
			"name":               workload.Name,
			"source":             workload.Source,
			"cases":              len(workload.Cases),
			"prompt_cap":         *workloadPromptCap,
			"decode_steps_timed": *decodeSteps,
			"token_ids":          "deterministic LCG IDs at recorded prompt lengths; token values are cost-irrelevant for this compute benchmark",
		}
	}
	fmt.Fprintf(os.Stderr, "[batchbench %s] PEAK B=%d: %.0f tok/s = %.1fx vs batch-1, %.1fx vs naive f32 serial\n",
		precision, peak.Batch, peak.AggTokPerSec, peak.SpeedupVsB1, peak.SpeedupVsNaive)

	b, _ := benchcli.MarshalReport(report)
	if *out != "" {
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write:", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "wrote", *out)
	} else {
		fmt.Println(string(b))
	}
}
