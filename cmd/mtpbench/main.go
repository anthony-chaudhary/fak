// Command mtpbench runs the physical fak-native Qwen3.8 27B MTP comparison.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mtpbench"
)

func main() {
	var cfg mtpbench.Config
	var prompt, output string
	var preflight bool
	flag.StringVar(&cfg.ArtifactPath, "artifact", "", "pinned Qwen3.8-27B Q4_K_M GGUF")
	flag.StringVar(&cfg.ExpectedArtifactShardSetSHA256, "artifact-shard-set-sha256", "", "required SHA-256 of the ordered shard manifest (path, size, raw-file SHA-256); not a raw GGUF file SHA-256")
	flag.StringVar(&cfg.TokenizerPath, "tokenizer", "", "tokenizer artifact used to produce prompt IDs")
	flag.StringVar(&cfg.ExpectedTokenizerSHA256, "tokenizer-sha256", "", "required tokenizer SHA-256")
	flag.StringVar(&cfg.BaselineIndexPath, "baseline-index", "", "latest-compatible fak/llama/MLX baseline index JSON")
	flag.StringVar(&cfg.ExpectedBaselineIndexSHA256, "baseline-index-sha256", "", "required baseline-index SHA-256")
	flag.StringVar(&prompt, "prompt-ids", "", "fixed comma-separated token IDs")
	flag.IntVar(&cfg.GeneratedTokens, "tokens", 64, "generated tokens per sample")
	flag.IntVar(&cfg.SamplesPerArm, "samples", 20, "even samples per arm (minimum 20)")
	flag.Float64Var(&cfg.RequiredSpeedup, "required-speedup", 2, "minimum candidate/baseline p50 ratio")
	flag.Float64Var(&cfg.MaximumCoefficientVariation, "max-cv", .05, "maximum per-arm throughput CV")
	flag.BoolVar(&preflight, "preflight", false, "read GGUF header and estimate bytes without loading payload")
	flag.StringVar(&output, "out", "", "optional JSON output path")
	flag.Parse()

	runner := mtpbench.Runner{Executor: mtpbench.ProductionExecutor{}}
	var value any
	var err error
	if preflight {
		value, err = runner.Preflight(context.Background(), cfg)
	} else {
		cfg.PromptIDs, err = parseIDs(prompt)
		if err == nil {
			value, err = runner.Run(context.Background(), cfg)
		}
	}
	raw, marshalErr := json.MarshalIndent(value, "", "  ")
	if marshalErr != nil {
		fmt.Fprintln(os.Stderr, marshalErr)
		os.Exit(1)
	}
	raw = append(raw, '\n')
	if output == "" {
		_, _ = os.Stdout.Write(raw)
	} else if writeErr := os.WriteFile(output, raw, 0o644); writeErr != nil {
		fmt.Fprintln(os.Stderr, writeErr)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func parseIDs(raw string) ([]int, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("mtpbench: --prompt-ids is required")
	}
	parts := strings.Split(raw, ",")
	ids := make([]int, len(parts))
	for i, part := range parts {
		id, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || id < 0 {
			return nil, fmt.Errorf("mtpbench: invalid prompt token %q", part)
		}
		ids[i] = id
	}
	return ids, nil
}
