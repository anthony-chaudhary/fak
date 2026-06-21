// Command loadgen is a pure-Go (stdlib-only) load generator for OpenAI-compatible
// /v1/chat/completions endpoints. It runs a concurrency sweep and writes a MATRIX.json
// with aggregate decode tok/s and latency percentiles per concurrency point. It is the
// DGX benchmark's load matrix and needs no Python.
//
//	go run ./cmd/loadgen -url http://127.0.0.1:8080/v1/chat/completions \
//	  -model smollm2-135m -concurrency 1,4,16 -requests-per 32 -out MATRIX.json
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	var (
		url          = flag.String("url", "", "full /v1/chat/completions endpoint URL")
		model        = flag.String("model", "mock", "model id to send")
		stack        = flag.String("stack", "", "stack label for the report (e.g. fak-engine, raw-sglang)")
		prompt       = flag.String("prompt", "Write one sentence about why benchmarks matter.", "prompt text")
		maxTokens    = flag.Int("max-tokens", 64, "max_tokens per request")
		concurrency  = flag.String("concurrency", "1,4,16", "comma-separated concurrency levels")
		requestsPer  = flag.Int("requests-per", 32, "requests per concurrency level")
		apiKeyEnv    = flag.String("api-key-env", "", "env var holding the API key (optional)")
		maxErrorRate = flag.Float64("max-error-rate", 0.0, "fail if any point exceeds this error rate")
		timeout      = flag.Duration("timeout", 20*time.Minute, "overall timeout")
		out          = flag.String("out", "", "write MATRIX.json here (default: stdout)")
	)
	flag.Parse()
	if *url == "" {
		fmt.Fprintln(os.Stderr, "loadgen: -url is required")
		os.Exit(2)
	}
	concs, err := parseInts(*concurrency)
	if err != nil {
		fmt.Fprintln(os.Stderr, "loadgen: bad -concurrency:", err)
		os.Exit(2)
	}
	apiKey := ""
	if *apiKeyEnv != "" && *apiKeyEnv != "NONE_LOCAL" {
		apiKey = os.Getenv(*apiKeyEnv)
	}

	cfg := &Config{
		URL: *url, Model: *model, Stack: *stack, Prompt: *prompt,
		MaxTokens: *maxTokens, Concurrencies: concs, RequestsPer: *requestsPer, APIKey: apiKey,
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	res := Run(ctx, cfg)
	enc, _ := json.MarshalIndent(res, "", "  ")
	if *out == "" {
		fmt.Println(string(enc))
	} else {
		if err := os.WriteFile(*out, enc, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "loadgen: write:", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "loadgen: wrote %s (%d points)\n", *out, len(res.Points))
	}

	// Summary + error-rate gate to stderr so JSON stdout stays clean.
	worst := 0.0
	for _, p := range res.Points {
		fmt.Fprintf(os.Stderr, "  c=%-4d tok/s=%-8.1f p50=%-7.1fms p95=%-7.1fms err=%.0f%%\n",
			p.Concurrency, p.TokensPerSecond, p.P50LatencyMS, p.P95LatencyMS, p.ErrorRate*100)
		if p.ErrorRate > worst {
			worst = p.ErrorRate
		}
	}
	if worst > *maxErrorRate {
		fmt.Fprintf(os.Stderr, "loadgen: FAIL worst error rate %.1f%% > max %.1f%%\n", worst*100, *maxErrorRate*100)
		os.Exit(1)
	}
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no concurrency values")
	}
	return out, nil
}
