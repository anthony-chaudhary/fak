package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

func cmdAgentX(argv []string) {
	os.Exit(runAgentX(os.Stdout, os.Stderr, argv))
}

func runAgentX(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak agentx", flag.ContinueOnError)
	fs.SetOutput(stderr)

	endpoint := fs.String("endpoint", envOrDefault("FAK_AGENTX_ENDPOINT", "http://127.0.0.1:8155/v1"), "OpenAI-compatible chat/completions endpoint")
	model := fs.String("model", "Qwen3.8-27B-Q4_K_M", "model identifier")
	engine := fs.String("engine", "", "engine the endpoint serves (required for a live run)")
	hardware := fs.String("hardware", "", "hardware the endpoint runs on (required for a live run)")
	agents := fs.Int("agents", 5, "number of concurrent agent sessions")
	turns := fs.Int("turns", 3, "number of sequential turns per agent")
	maxTokens := fs.Int("max-tokens", 32, "maximum tokens per turn")
	temp := fs.Float64("temperature", 0.0, "sampling temperature")
	prefix := fs.String("prefix", "", "shared master goal prompt prefix")
	outPath := fs.String("out", "", "output receipt JSON path")
	selfcheck := fs.Bool("selfcheck", false, "execute hermetic self-check without external endpoint (SIMULATED, never evidence)")
	validateFile := fs.String("validate", "", "admit an existing receipt JSON file as evidence (refuses mock and legacy receipts)")
	jsonOutput := fs.Bool("json", false, "output JSON receipt to stdout")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *validateFile != "" {
		data, err := os.ReadFile(*validateFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak agentx: read file %s: %v\n", *validateFile, err)
			return 1
		}
		var receipt agentxbench.AgentXReceipt
		if err := json.Unmarshal(data, &receipt); err != nil {
			fmt.Fprintf(stderr, "fak agentx: unmarshal receipt: %v\n", err)
			return 1
		}
		errs := agentxbench.EvidenceErrors(&receipt)
		if len(errs) > 0 {
			fmt.Fprintf(stderr, "FAIL: %d validation error(s):\n", len(errs))
			for _, e := range errs {
				fmt.Fprintf(stderr, "  - %s\n", e)
			}
			return 1
		}
		fmt.Fprintf(stdout, "PASS: receipt %s verified (%s %s on %s, %d requests, token counts %s, prefix speedup: %s; synthetic workload, not InferenceX-comparable)\n",
			receipt.BenchmarkID, receipt.ExecutionMode, receipt.Model, receipt.Hardware, receipt.Aggregated.TotalRequests,
			receipt.TokenCountSource, speedupText(receipt.Aggregated.PrefixSpeedupRatio))
		return 0
	}

	cfg := agentxbench.Config{
		EndpointURL:       *endpoint,
		Model:             *model,
		Engine:            *engine,
		Hardware:          *hardware,
		AgentCount:        *agents,
		TurnsPerAgent:     *turns,
		MaxTokens:         *maxTokens,
		Temperature:       *temp,
		SharedPrefix:      *prefix,
		DeterministicSeed: time.Now().UnixNano(),
		MockExecution:     *selfcheck,
	}

	runner := agentxbench.NewRunner(nil)
	receipt, err := runner.Run(context.Background(), cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak agentx: execution failed: %v\n", err)
		return 1
	}

	if *outPath != "" {
		data, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak agentx: marshal receipt: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*outPath, data, 0644); err != nil {
			fmt.Fprintf(stderr, "fak agentx: write output %s: %v\n", *outPath, err)
			return 1
		}
	}

	if *jsonOutput {
		data, _ := json.MarshalIndent(receipt, "", "  ")
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintf(stdout, "AgentX Benchmark Result: %s\n", receipt.BenchmarkID)
		fmt.Fprintf(stdout, "Mode / Workload: %s / %s (not InferenceX-comparable)\n", receipt.ExecutionMode, receipt.Workload)
		fmt.Fprintf(stdout, "Model / Engine : %s / %s\n", receipt.Model, receipt.Engine)
		fmt.Fprintf(stdout, "Hardware       : %s\n", receipt.Hardware)
		fmt.Fprintf(stdout, "Concurrency    : %d agents x %d turns (%d total requests)\n",
			receipt.AgentCount, receipt.TurnsPerAgent, receipt.Aggregated.TotalRequests)
		fmt.Fprintf(stdout, "Success Rate   : %.1f%% (%d/%d)\n",
			receipt.Aggregated.SuccessRate*100.0, receipt.Aggregated.SuccessfulRequests, receipt.Aggregated.TotalRequests)
		fmt.Fprintf(stdout, "Cold TTFT Mean : %.2f ms\n", receipt.Aggregated.ColdTTFTMeanMS)
		fmt.Fprintf(stdout, "Warm TTFT Mean : %.2f ms\n", receipt.Aggregated.WarmTTFTMeanMS)
		fmt.Fprintf(stdout, "Prefix Speedup : %s\n", speedupText(receipt.Aggregated.PrefixSpeedupRatio))
		fmt.Fprintf(stdout, "ITL p50 / p95  : %.2f ms / %.2f ms\n",
			receipt.Aggregated.ITLP50MS, receipt.Aggregated.ITLP95MS)
		fmt.Fprintf(stdout, "Interactivity  : %.2f tok/s\n", receipt.Aggregated.NormalizedInteractivity)
		fmt.Fprintf(stdout, "Throughput     : %.2f req/s (%.2f out tok/s, %.2f cluster tok/s)\n",
			receipt.Aggregated.RequestThroughputPerSec,
			receipt.Aggregated.OutputTokenThroughputPerSec,
			receipt.Aggregated.ClusterTokenThroughputPerSec)
		fmt.Fprintf(stdout, "Status         : %s\n", receipt.ValidationStatus)
	}

	want := agentxbench.StatusVerifiedPass
	if *selfcheck {
		want = agentxbench.StatusSimulated
	}
	if receipt.ValidationStatus != want {
		return 1
	}
	return 0
}

// speedupText prints n/a when no warm turn was observed.
func speedupText(ratio float64) string {
	if ratio <= 0 {
		return "n/a (no warm turn)"
	}
	return fmt.Sprintf("%.2fx", ratio)
}
