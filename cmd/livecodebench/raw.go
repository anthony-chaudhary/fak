package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

// runRaw implements `livecodebench raw` (#2104, epic #2085): the "raw" A/B arm.
// It sends each suite problem's rendered prompt to the fak OpenAI-compatible
// gateway /v1/chat/completions UNADJUDICATED and collects n samples per problem
// with bounded concurrency that mirrors the closed-API lcb_runner --multiprocess
// fan-out. The fan-out and report shape are pure in internal/livecodebench
// (RunRawArm); this is the only place that touches the wire. It grades nothing —
// the emitted report carries generations for the official evaluator, never a score.
func runRaw(argv []string) int {
	fs := flag.NewFlagSet("livecodebench raw", flag.ContinueOnError)
	suitePath := fs.String("suite", "", "normalized suite JSON whose problems are sent to the gateway (required)")
	model := fs.String("model", "", "model id sent on each request and recorded in the report (required)")
	endpoint := fs.String("endpoint", "http://127.0.0.1:8080/v1", "gateway base URL (…/v1) the completions are POSTed to")
	n := fs.Int("n", 1, "samples to generate per problem (mirrors lcb_runner -n)")
	temperature := fs.Float64("temperature", 0.2, "sampling temperature sent and recorded")
	concurrency := fs.Int("concurrency", 8, "max in-flight gateway requests (mirrors closed-API --multiprocess)")
	maxTokens := fs.Int("max-tokens", 2048, "max_tokens sent on each completion request")
	timeout := fs.Duration("timeout", 120*time.Second, "per-request HTTP timeout")
	out := fs.String("out", "", "write the raw-arm report JSON to this path (default: stdout)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench raw: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*suitePath) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench raw: --suite is required")
		return 2
	}
	if strings.TrimSpace(*model) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench raw: --model is required")
		return 2
	}

	suite, err := livecodebench.LoadSuiteFile(*suitePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench raw: %v\n", err)
		return 1
	}

	cfg := livecodebench.RawArmConfig{
		Model:       *model,
		Endpoint:    *endpoint,
		N:           *n,
		Temperature: *temperature,
		Concurrency: *concurrency,
	}
	client := &http.Client{Timeout: *timeout}
	sampler := gatewaySampler(client, *endpoint, *model, *temperature, *maxTokens)

	report, err := livecodebench.RunRawArm(context.Background(), cfg, suite.Problems, sampler)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench raw: %v\n", err)
		return 1
	}

	w := io.Writer(os.Stdout)
	if strings.TrimSpace(*out) != "" {
		file, cerr := os.Create(*out)
		if cerr != nil {
			fmt.Fprintf(os.Stderr, "livecodebench raw: %v\n", cerr)
			return 1
		}
		defer file.Close()
		w = file
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench raw: %v\n", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "livecodebench raw: %d problem(s) × n=%d via %s (model %s), %d cached prompt tokens\n",
		len(report.Problems), report.N, report.Endpoint, report.Model, report.Usage.CachedPromptTokens)
	return 0
}

// gatewaySampler returns a RawArmSampler that POSTs one OpenAI-compatible
// chat-completions request per call and returns the completion text plus the
// provider-relayed usage. RunRawArm folds usage through Usage.CachedPromptTokens().
func gatewaySampler(client *http.Client, endpoint, model string, temperature float64, maxTokens int) livecodebench.RawArmSampler {
	url := strings.TrimRight(endpoint, "/") + "/chat/completions"
	return func(ctx context.Context, p livecodebench.Problem, _ int) (string, agent.Usage, error) {
		reqBody := chatCompletionsRequest{
			Model:       model,
			Temperature: temperature,
			MaxTokens:   maxTokens,
			Messages:    []agent.Message{{Role: agent.RoleUser, Content: p.Prompt}},
		}
		buf, err := json.Marshal(reqBody)
		if err != nil {
			return "", agent.Usage{}, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
		if err != nil {
			return "", agent.Usage{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return "", agent.Usage{}, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", agent.Usage{}, err
		}
		if resp.StatusCode != http.StatusOK {
			return "", agent.Usage{}, fmt.Errorf("gateway HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}
		var out chatCompletionsResponse
		if err := json.Unmarshal(body, &out); err != nil {
			return "", agent.Usage{}, err
		}
		if len(out.Choices) == 0 {
			return "", agent.Usage{}, fmt.Errorf("gateway returned no choices")
		}
		return out.Choices[0].Message.Content, out.Usage, nil
	}
}

type chatCompletionsRequest struct {
	Model       string          `json:"model"`
	Temperature float64         `json:"temperature"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Messages    []agent.Message `json:"messages"`
}

type chatCompletionsResponse struct {
	Choices []struct {
		Message agent.Message `json:"message"`
	} `json:"choices"`
	Usage agent.Usage `json:"usage"`
}
