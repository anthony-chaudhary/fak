package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/logprobparity"
	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// logprobsDefaultPrompts is the live-reference battery when --ref is an
// endpoint and no --prompt is given; it mirrors the recorded oracles' prompts.
var logprobsDefaultPrompts = []string{
	"The chemical symbol for gold is",
	"The capital of France is",
	"Say hello.",
	"What is 17 times 3?",
}

type logprobsPromptList []string

func (s *logprobsPromptList) String() string     { return strings.Join(*s, ",") }
func (s *logprobsPromptList) Set(v string) error { *s = append(*s, v); return nil }

// runComputeTraceLogprobs scores an endpoint's greedy continuations against a
// reference oracle file or a live llama-server (check_logprobs_close).
func runComputeTraceLogprobs(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("compute-trace logprobs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	endpoint := fs.String("endpoint", "", "candidate endpoint base URL (required)")
	api := fs.String("api", logprobparity.APIOpenAI, "candidate API: openai (/v1/completions, text) or llama (/completion, ids + top-k)")
	apiKey := fs.String("api-key", "", "candidate bearer key (default $FAK_PARITY_API_KEY)")
	model := fs.String("model", "", "candidate model id (openai api)")
	ref := fs.String("ref", "", "reference: a reference-oracle JSON file or a llama-server base URL (required)")
	refKey := fs.String("ref-api-key", "", "reference bearer key for a live llama-server")
	var prompts logprobsPromptList
	fs.Var(&prompts, "prompt", "prompt for a live reference (repeatable; default: the 4-prompt oracle battery)")
	maxTokens := fs.Int("max-tokens", 16, "tokens per prompt for a live reference")
	topK := fs.Int("topk", 5, "top-k for the divergence membership check")
	timeout := fs.Duration("timeout", 10*time.Minute, "whole-run budget")
	jsonOut := fs.Bool("json", false, "print the fak.logprob_parity.v1 report as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if strings.TrimSpace(*endpoint) == "" || strings.TrimSpace(*ref) == "" || *topK < 1 || *maxTokens < 1 {
		fmt.Fprintln(stderr, "usage: fak compute-trace logprobs --endpoint <url> --ref <oracle.json|llama-server url> [--api openai|llama] [--topk 5] [--json]")
		return 2
	}
	key := *apiKey
	if key == "" {
		key = os.Getenv("FAK_PARITY_API_KEY")
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	refs, err := logprobsReference(ctx, *ref, *refKey, prompts, *maxTokens, *topK)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	actual := logprobparity.Client{BaseURL: *endpoint, APIKey: key, Model: *model, API: *api}
	prs, scoreErr := logprobparity.Score(ctx, actual, refs, *topK)
	rep := logprobparity.FoldPrompts(prs)
	rep.Reference, rep.Actual, rep.TopK = *ref, *endpoint, *topK
	rep.Results = nil
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(struct {
			logprobparity.Report
			Prompts []logprobparity.PromptResult `json:"prompts"`
		}{rep, prs})
	} else {
		for _, p := range prs {
			at := "-"
			if p.DivergedAt != nil {
				at = fmt.Sprint(*p.DivergedAt)
			}
			fmt.Fprintf(stdout, "%-9s match=%d/%d diverged_at=%s method=%s prompt=%q\n", p.Verdict, p.GreedyMatchTokens, p.Compared, at, p.Method, p.Prompt)
		}
		fmt.Fprintf(stdout, "verdict=%s greedy_match_tokens=%d compared_tokens=%d top_k=%d\n", rep.Verdict, rep.GreedyMatchTokens, rep.Compared, rep.TopK)
	}
	if scoreErr != nil {
		fmt.Fprintln(stderr, scoreErr)
		return 1
	}
	if rep.Verdict == logprobparity.NotClose || rep.Verdict == logprobparity.Empty {
		return 1
	}
	return 0
}

func logprobsReference(ctx context.Context, ref, key string, prompts []string, maxTokens, topK int) ([]logprobparity.RefPrompt, error) {
	if strings.HasPrefix(ref, "http://") || strings.HasPrefix(ref, "https://") {
		if len(prompts) == 0 {
			prompts = logprobsDefaultPrompts
		}
		c := logprobparity.Client{BaseURL: ref, APIKey: key, API: logprobparity.APILlama}
		var out []logprobparity.RefPrompt
		for _, p := range prompts {
			g, err := c.Generate(ctx, p, maxTokens, topK)
			if err != nil {
				return nil, fmt.Errorf("reference: %w", err)
			}
			out = append(out, logprobparity.RefPrompt{Text: p, InputTokens: g.PromptTokens, Positions: g.Positions})
		}
		return out, nil
	}
	o, err := referenceoracle.ReadFile(ref)
	if err != nil {
		return nil, err
	}
	if len(prompts) > 0 {
		return nil, fmt.Errorf("--prompt applies only to a live reference; %s fixes its prompts", ref)
	}
	return logprobparity.OraclePrompts(o), nil
}
