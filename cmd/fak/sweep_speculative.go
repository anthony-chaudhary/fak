package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// sweep_speculative.go — speculative combinatorial sweep and Pareto frontier generator
// for issue #10844 and Stage 5 simulation methodology (docs/benchmarks/speculative-hardware-simulation-methodology.md).
//
// Synthesizes analytical hardware roofline models and empirical/statistical acceptance profiles
// across (K, batch, context_len) combinatorial grid points. Identifies the 2D non-dominated
// Pareto frontier (TPOT latency vs request throughput) and highlights operational break-even
// boundaries to guide physical accelerator benchmarking.

// SweepSpeculativeConfig captures input parameters for the speculative grid sweep.
type SweepSpeculativeConfig struct {
	Hardware    string
	TargetModel string
	DraftModel  string
	KMin        int
	KMax        int
	BatchSizes  []int
	ContextLens []int
	Alpha       float64
	IsTree      bool
	JSON        bool
	Output      string
}

// SweepPoint captures analytical performance projections for one evaluated configuration.
type SweepPoint struct {
	Hardware               string  `json:"hardware"`
	TargetModel            string  `json:"target_model"`
	DraftModel             string  `json:"draft_model"`
	K                      int     `json:"k"`
	BatchSize              int     `json:"batch_size"`
	ContextLen             int     `json:"context_len"`
	DraftLatencyMs         float64 `json:"draft_latency_ms"`
	VerifyLatencyMs        float64 `json:"verify_latency_ms"`
	StepLatencyMs          float64 `json:"step_latency_ms"`
	TPOTMs                 float64 `json:"tpot_ms"`
	BaselineTPOTMs         float64 `json:"baseline_tpot_ms"`
	ExpectedAcceptedTokens float64 `json:"expected_accepted_tokens"`
	Speedup                float64 `json:"speedup"`
	Throughput             float64 `json:"throughput"`         // tokens/sec across concurrent sequences
	RequestThroughput      float64 `json:"request_throughput"` // requests/sec normalized
	Viable                 bool    `json:"viable"`
	Regime                 string  `json:"regime"`
}

// BreakEvenBoundaries captures key operational thresholds where performance characteristics shift.
type BreakEvenBoundaries struct {
	MaxViableBatchSize int     `json:"max_viable_batch_size"`
	CrossoverBatchSize int     `json:"crossover_batch_size"`
	MinTPOTMs          float64 `json:"min_tpot_ms"`
	MaxThroughput      float64 `json:"max_throughput"`
}

// RecommendedConfig marks a chosen operating point from the Pareto frontier.
type RecommendedConfig struct {
	Role        string     `json:"role"`
	Description string     `json:"description"`
	Point       SweepPoint `json:"point"`
}

// GridSpec records the sweep parameters evaluated.
type GridSpec struct {
	KMin        int   `json:"k_min"`
	KMax        int   `json:"k_max"`
	BatchSizes  []int `json:"batch_sizes"`
	ContextLens []int `json:"context_lens"`
}

// SweepSpeculativeResult aggregates the complete outcome of the combinatorial sweep and Pareto analysis.
type SweepSpeculativeResult struct {
	Schema          string                        `json:"schema"`
	Hardware        compute.HardwareProfile       `json:"hardware"`
	TargetModel     compute.ModelCostProfile      `json:"target_model"`
	DraftModel      compute.ModelCostProfile      `json:"draft_model"`
	Acceptance      compute.SpecAcceptanceProfile `json:"acceptance"`
	Grid            GridSpec                      `json:"grid"`
	TotalEvaluated  int                           `json:"total_evaluated"`
	ParetoCount     int                           `json:"pareto_count"`
	ParetoFrontier  []SweepPoint                  `json:"pareto_frontier"`
	BreakEven       BreakEvenBoundaries           `json:"break_even_boundaries"`
	RecommendedTop3 []RecommendedConfig           `json:"recommended_top3"`
}

func cmdSweepSpeculative(argv []string) {
	os.Exit(runSweepSpeculative(os.Stdout, os.Stderr, argv))
}

func parseSweepSpeculativeFlags(argv []string, stderr io.Writer) (*SweepSpeculativeConfig, int) {
	fs := flag.NewFlagSet("fak sweep-speculative", flag.ContinueOnError)
	fs.SetOutput(stderr)

	hardware := fs.String("hardware", "l4", "hardware profile: l4, a100, h100, m3max")
	profile := fs.String("profile", "", "alias for --hardware")
	target := fs.String("target", "qwen_7b", "target model profile (e.g. qwen_7b, qwen_72b)")
	draft := fs.String("draft", "qwen_0_5b", "draft model profile (e.g. qwen_0_5b)")
	kMin := fs.Int("k-min", 1, "minimum draft length")
	kMax := fs.Int("k-max", 10, "maximum draft length")
	batchSizes := fs.String("batch-sizes", "1,4,8,16,32,64", "comma-separated batch sizes")
	contextLens := fs.String("context-lens", "512,2048,8192,16384", "comma-separated context lengths")
	alpha := fs.Float64("alpha", 0.75, "mean speculative acceptance rate")
	tree := fs.Bool("tree", false, "model tree-structured speculative verification")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON Pareto frontier")
	output := fs.String("output", "", "optional output file path")
	outAlias := fs.String("out", "", "alias for --output")

	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return nil, code
	}

	hwName := *hardware
	if *profile != "" {
		hwName = *profile
	}

	outPath := *output
	if *outAlias != "" && outPath == "" {
		outPath = *outAlias
	}

	if *kMin < 1 {
		fmt.Fprintf(stderr, "fak sweep-speculative: --k-min must be >= 1, got %d\n", *kMin)
		return nil, 2
	}
	if *kMax < *kMin {
		fmt.Fprintf(stderr, "fak sweep-speculative: --k-max (%d) cannot be less than --k-min (%d)\n", *kMax, *kMin)
		return nil, 2
	}
	if *alpha < 0.0 || *alpha > 1.0 {
		fmt.Fprintf(stderr, "fak sweep-speculative: --alpha must be between 0.0 and 1.0, got %f\n", *alpha)
		return nil, 2
	}

	batches, err := parseSweepIntList(*batchSizes)
	if err != nil {
		fmt.Fprintf(stderr, "fak sweep-speculative: invalid --batch-sizes: %v\n", err)
		return nil, 2
	}
	contexts, err := parseSweepIntList(*contextLens)
	if err != nil {
		fmt.Fprintf(stderr, "fak sweep-speculative: invalid --context-lens: %v\n", err)
		return nil, 2
	}

	if len(batches) == 0 {
		fmt.Fprintln(stderr, "fak sweep-speculative: at least one batch size required")
		return nil, 2
	}
	if len(contexts) == 0 {
		fmt.Fprintln(stderr, "fak sweep-speculative: at least one context length required")
		return nil, 2
	}

	cfg := &SweepSpeculativeConfig{
		Hardware:    hwName,
		TargetModel: *target,
		DraftModel:  *draft,
		KMin:        *kMin,
		KMax:        *kMax,
		BatchSizes:  batches,
		ContextLens: contexts,
		Alpha:       *alpha,
		IsTree:      *tree,
		JSON:        *asJSON,
		Output:      outPath,
	}
	return cfg, 0
}

func parseSweepIntList(raw string) ([]int, error) {
	tokens := splitCommaList(raw)
	if len(tokens) == 0 {
		return nil, errors.New("empty integer list")
	}
	var out []int
	seen := make(map[int]bool)
	for _, tok := range tokens {
		v, err := strconv.Atoi(tok)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q: %w", tok, err)
		}
		if v <= 0 {
			return nil, fmt.Errorf("integer must be positive, got %d", v)
		}
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Ints(out)
	return out, nil
}

func runSweepSpeculative(stdout, stderr io.Writer, argv []string) int {
	cfg, code := parseSweepSpeculativeFlags(argv, stderr)
	if code != 0 {
		return code
	}

	hw, ok := compute.LookupHardwareProfile(cfg.Hardware)
	if !ok {
		fmt.Fprintf(stderr, "fak sweep-speculative: unknown hardware profile %q; supported: l4, a100, h100, m3max\n", cfg.Hardware)
		return 2
	}

	targetProfile, ok := compute.LookupModelCostProfile(cfg.TargetModel)
	if !ok {
		fmt.Fprintf(stderr, "fak sweep-speculative: unknown target model profile %q; supported: qwen_7b, qwen_0_5b, qwen_72b\n", cfg.TargetModel)
		return 2
	}

	draftProfile, ok := compute.LookupModelCostProfile(cfg.DraftModel)
	if !ok {
		fmt.Fprintf(stderr, "fak sweep-speculative: unknown draft model profile %q; supported: qwen_7b, qwen_0_5b, qwen_72b\n", cfg.DraftModel)
		return 2
	}

	method := "linear"
	if cfg.IsTree {
		method = "tree"
	}
	acceptance := compute.SpecAcceptanceProfile{
		Method:    method,
		MeanAlpha: cfg.Alpha,
	}

	allPoints := EvaluateSpeculativeGrid(cfg, hw, targetProfile, draftProfile, acceptance)
	frontier := FilterParetoFrontier(allPoints)
	breakEven := ComputeBreakEvenBoundaries(allPoints, frontier)
	recommended := IdentifyTop3Configurations(frontier)

	result := &SweepSpeculativeResult{
		Schema:          "fak-sweep-speculative/1",
		Hardware:        hw,
		TargetModel:     targetProfile,
		DraftModel:      draftProfile,
		Acceptance:      acceptance,
		Grid:            GridSpec{KMin: cfg.KMin, KMax: cfg.KMax, BatchSizes: cfg.BatchSizes, ContextLens: cfg.ContextLens},
		TotalEvaluated:  len(allPoints),
		ParetoCount:     len(frontier),
		ParetoFrontier:  frontier,
		BreakEven:       breakEven,
		RecommendedTop3: recommended,
	}

	var outWriter io.Writer = stdout
	if cfg.Output != "" {
		dir := filepath.Dir(cfg.Output)
		if dir != "." && dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				fmt.Fprintf(stderr, "fak sweep-speculative: mkdir: %v\n", err)
				return 1
			}
		}
		f, err := os.Create(cfg.Output)
		if err != nil {
			fmt.Fprintf(stderr, "fak sweep-speculative: create output: %v\n", err)
			return 1
		}
		defer f.Close()
		outWriter = io.MultiWriter(stdout, f)
	}

	if cfg.JSON {
		enc := json.NewEncoder(outWriter)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak sweep-speculative: json encode: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprint(outWriter, FormatHumanReadableTable(result))
	return 0
}

// EvaluateSpeculativeGrid evaluates every combination of (K, batch, context_len) against the analytical roofline.
func EvaluateSpeculativeGrid(cfg *SweepSpeculativeConfig, hw compute.HardwareProfile, target, draft compute.ModelCostProfile, acceptance compute.SpecAcceptanceProfile) []SweepPoint {
	var points []SweepPoint

	for k := cfg.KMin; k <= cfg.KMax; k++ {
		for _, b := range cfg.BatchSizes {
			for _, ctxLen := range cfg.ContextLens {
				res := compute.ProjectSpeculativeSpeedup(hw, target, draft, acceptance, b, ctxLen, k, cfg.IsTree)

				draftTotalSec := float64(k) * res.DraftStep.TTotalSec
				if cfg.IsTree {
					draftTotalSec = res.DraftStep.TTotalSec
				}
				stepLatencySec := draftTotalSec + res.VerifyStep.TTotalSec
				if k == 0 {
					stepLatencySec = res.AutoregressiveTPOTSec
				}

				tpotSec := res.SpeculativeTPOTSec
				var tput float64
				if tpotSec > 0 {
					tput = float64(b) / tpotSec
				}

				pt := SweepPoint{
					Hardware:               hw.Name,
					TargetModel:            target.Name,
					DraftModel:             draft.Name,
					K:                      k,
					BatchSize:              b,
					ContextLen:             ctxLen,
					DraftLatencyMs:         draftTotalSec * 1000.0,
					VerifyLatencyMs:        res.VerifyStep.TTotalSec * 1000.0,
					StepLatencyMs:          stepLatencySec * 1000.0,
					TPOTMs:                 tpotSec * 1000.0,
					BaselineTPOTMs:         res.AutoregressiveTPOTSec * 1000.0,
					ExpectedAcceptedTokens: res.ExpectedAcceptedTokens,
					Speedup:                res.Speedup,
					Throughput:             tput,
					RequestThroughput:      tput,
					Viable:                 res.Viable,
					Regime:                 res.VerifyStep.Regime,
				}
				points = append(points, pt)
			}
		}
	}

	return points
}

// FilterParetoFrontier filters candidates down to the 2D non-dominated Pareto frontier:
// minimizing TPOT latency and maximizing request throughput.
func FilterParetoFrontier(points []SweepPoint) []SweepPoint {
	if len(points) == 0 {
		return nil
	}

	var frontier []SweepPoint

	for i := range points {
		p1 := &points[i]
		dominated := false

		for j := range points {
			if i == j {
				continue
			}
			p2 := &points[j]

			// p2 dominates p1 iff:
			//   p2.TPOTMs <= p1.TPOTMs AND p2.Throughput >= p1.Throughput
			// with at least one strictly better.
			betterOrEqual := p2.TPOTMs <= p1.TPOTMs && p2.Throughput >= p1.Throughput
			strictlyBetter := p2.TPOTMs < p1.TPOTMs || p2.Throughput > p1.Throughput

			if betterOrEqual && strictlyBetter {
				dominated = true
				break
			}

			// Deterministic tie-breaker: keep the lower index to eliminate duplicates
			if p2.TPOTMs == p1.TPOTMs && p2.Throughput == p1.Throughput && j < i {
				dominated = true
				break
			}
		}

		if !dominated {
			frontier = append(frontier, *p1)
		}
	}

	// Sort frontier by TPOTMs ascending. Along a 2D Pareto front,
	// Throughput is also monotonically increasing.
	sort.Slice(frontier, func(i, j int) bool {
		if frontier[i].TPOTMs != frontier[j].TPOTMs {
			return frontier[i].TPOTMs < frontier[j].TPOTMs
		}
		return frontier[i].Throughput < frontier[j].Throughput
	})

	return frontier
}

// ComputeBreakEvenBoundaries identifies critical operational transition boundaries.
func ComputeBreakEvenBoundaries(allPoints []SweepPoint, frontier []SweepPoint) BreakEvenBoundaries {
	b := BreakEvenBoundaries{}

	for _, p := range allPoints {
		if p.Viable && p.BatchSize > b.MaxViableBatchSize {
			b.MaxViableBatchSize = p.BatchSize
		}
		if p.Regime == "compute_bound" {
			if b.CrossoverBatchSize == 0 || p.BatchSize < b.CrossoverBatchSize {
				b.CrossoverBatchSize = p.BatchSize
			}
		}
	}

	if len(frontier) > 0 {
		b.MinTPOTMs = frontier[0].TPOTMs
		b.MaxThroughput = frontier[len(frontier)-1].Throughput
	}

	return b
}

// IdentifyTop3Configurations identifies candidate configurations for Stage 5 targeted physical witnessing:
// 1. Single-Stream Latency (minimizing TPOT at low concurrency)
// 2. Balanced Medium Concurrency (optimal throughput/latency balance)
// 3. Maximum Throughput (peak tok/s on Pareto frontier)
func IdentifyTop3Configurations(frontier []SweepPoint) []RecommendedConfig {
	if len(frontier) == 0 {
		return nil
	}

	var recs []RecommendedConfig

	// 1. Single-Stream Latency: find lowest TPOT point, preferring Batch=1
	bestLatencyIdx := 0
	for i, pt := range frontier {
		if pt.BatchSize == 1 {
			bestLatencyIdx = i
			break
		}
	}
	recs = append(recs, RecommendedConfig{
		Role:        "single_stream_latency",
		Description: "Optimal single-stream latency configuration (lowest TPOT)",
		Point:       frontier[bestLatencyIdx],
	})

	if len(frontier) == 1 {
		return recs
	}

	// 2. Maximum Throughput: highest tok/s point (last element in sorted frontier)
	maxTputIdx := len(frontier) - 1
	if maxTputIdx == bestLatencyIdx {
		for i := len(frontier) - 1; i >= 0; i-- {
			if i != bestLatencyIdx {
				maxTputIdx = i
				break
			}
		}
	}

	// 3. Balanced Medium Concurrency: search for batch size between 4 and 16 with highest speedup
	balancedIdx := -1
	bestSpeedup := -1.0
	for i, pt := range frontier {
		if i == bestLatencyIdx || i == maxTputIdx {
			continue
		}
		if pt.BatchSize >= 4 && pt.BatchSize <= 16 {
			if pt.Speedup > bestSpeedup {
				bestSpeedup = pt.Speedup
				balancedIdx = i
			}
		}
	}

	// Fallback for balanced: highest speedup among remaining points
	if balancedIdx == -1 {
		for i, pt := range frontier {
			if i == bestLatencyIdx || i == maxTputIdx {
				continue
			}
			if pt.Speedup > bestSpeedup {
				bestSpeedup = pt.Speedup
				balancedIdx = i
			}
		}
	}

	// Fallback to middle element
	if balancedIdx == -1 && len(frontier) >= 3 {
		mid := len(frontier) / 2
		if mid != bestLatencyIdx && mid != maxTputIdx {
			balancedIdx = mid
		}
	}

	if balancedIdx != -1 {
		recs = append(recs, RecommendedConfig{
			Role:        "balanced_concurrency",
			Description: "Balanced medium-concurrency configuration (optimal trade-off between latency and throughput)",
			Point:       frontier[balancedIdx],
		})
	}

	recs = append(recs, RecommendedConfig{
		Role:        "max_throughput",
		Description: "Maximum throughput configuration (peak tokens/sec under saturation)",
		Point:       frontier[maxTputIdx],
	})

	return recs
}

// FormatHumanReadableTable formats the sweep results and Pareto frontier into an aligned console report.
func FormatHumanReadableTable(r *SweepSpeculativeResult) string {
	var sb strings.Builder
	sb.WriteString("====================================================================================================\n")
	sb.WriteString("               SPECULATIVE COMBINATORIAL SWEEP & PARETO FRONTIER\n")
	sb.WriteString("====================================================================================================\n")
	sb.WriteString(fmt.Sprintf("Hardware:     %s (%s, %.1f GB/s peak, %.1f GB/s eff, %.1f TFLOPs)\n",
		r.Hardware.Name, r.Hardware.Arch, r.Hardware.PeakMemoryBandwidthGBs,
		r.Hardware.EffectiveBandwidthGBs, r.Hardware.PeakComputeTFLOPs))
	sb.WriteString(fmt.Sprintf("Target Model: %s (%.2fB params, %d layers)\n",
		r.TargetModel.Name, float64(r.TargetModel.ActiveParams)/1e9, r.TargetModel.NLayers))
	sb.WriteString(fmt.Sprintf("Draft Model:  %s (%.2fB params, %d layers)\n",
		r.DraftModel.Name, float64(r.DraftModel.ActiveParams)/1e9, r.DraftModel.NLayers))
	method := "Linear"
	if r.Acceptance.Method == "tree" || r.Acceptance.TreeYield > 0 {
		method = "Tree"
	}
	sb.WriteString(fmt.Sprintf("Acceptance:   Mean Alpha = %.2f (%s speculation)\n", r.Acceptance.MeanAlpha, method))
	sb.WriteString(fmt.Sprintf("Search Grid:  K=%d..%d | Batch=%v | Context=%v\n",
		r.Grid.KMin, r.Grid.KMax, r.Grid.BatchSizes, r.Grid.ContextLens))
	sb.WriteString(fmt.Sprintf("Total Evaluated: %d configurations | Pareto Optimal: %d configurations\n",
		r.TotalEvaluated, r.ParetoCount))
	sb.WriteString("----------------------------------------------------------------------------------------------------\n")
	sb.WriteString("PARETO OPTIMAL FRONTIER:\n")
	sb.WriteString(fmt.Sprintf("%-4s  %-6s  %-8s  %-14s  %-10s  %-19s  %-8s  %-14s  %-6s\n",
		"K", "Batch", "Context", "Step Lat (ms)", "TPOT (ms)", "Throughput (tok/s)", "Speedup", "Regime", "Viable"))
	sb.WriteString("----------------------------------------------------------------------------------------------------\n")

	for _, pt := range r.ParetoFrontier {
		viableStr := "YES"
		if !pt.Viable {
			viableStr = "NO"
		}
		speedupStr := fmt.Sprintf("%.2fx", pt.Speedup)
		sb.WriteString(fmt.Sprintf("%-4d  %-6d  %-8d  %-14.2f  %-10.2f  %-19.2f  %-8s  %-14s  %-6s\n",
			pt.K, pt.BatchSize, pt.ContextLen, pt.StepLatencyMs, pt.TPOTMs, pt.Throughput, speedupStr, pt.Regime, viableStr))
	}

	sb.WriteString("----------------------------------------------------------------------------------------------------\n")
	sb.WriteString("OPERATIONAL BREAK-EVEN BOUNDARIES:\n")
	if r.BreakEven.MaxViableBatchSize > 0 {
		sb.WriteString(fmt.Sprintf("  - Max Viable Speculative Batch Size: B <= %d (Speedup drops to <1.0x at larger batches)\n",
			r.BreakEven.MaxViableBatchSize))
	} else {
		sb.WriteString("  - Max Viable Speculative Batch Size: None (no evaluated configuration achieved Speedup > 1.0x)\n")
	}
	if r.BreakEven.CrossoverBatchSize > 0 {
		sb.WriteString(fmt.Sprintf("  - Memory-to-Compute Transition:       Batch B >= %d transitions to compute_bound\n",
			r.BreakEven.CrossoverBatchSize))
	} else {
		sb.WriteString("  - Memory-to-Compute Transition:       All evaluated points remain memory_bound\n")
	}
	sb.WriteString(fmt.Sprintf("  - Single-Stream Latency Floor:       %.2f ms/tok\n", r.BreakEven.MinTPOTMs))
	sb.WriteString(fmt.Sprintf("  - Peak Viable Speculative Throughput: %.2f tok/s\n", r.BreakEven.MaxThroughput))

	if len(r.RecommendedTop3) > 0 {
		sb.WriteString("\nRECOMMENDED STAGE 5 PHYSICAL WITNESS CONFIGURATIONS (Top-3):\n")
		labels := map[string]string{
			"single_stream_latency": "Single-Stream Latency",
			"balanced_concurrency":  "Balanced Concurrency",
			"max_throughput":        "Maximum Throughput",
		}
		for i, rec := range r.RecommendedTop3 {
			label := labels[rec.Role]
			if label == "" {
				label = rec.Role
			}
			sb.WriteString(fmt.Sprintf("  [%d] %-22s: K=%d, Batch=%d, Ctx=%-5d -> TPOT: %6.2f ms, Throughput: %8.2f tok/s, Speedup: %.2fx\n",
				i+1, label, rec.Point.K, rec.Point.BatchSize, rec.Point.ContextLen,
				rec.Point.TPOTMs, rec.Point.Throughput, rec.Point.Speedup))
		}
	}
	sb.WriteString("====================================================================================================\n")
	return sb.String()
}
