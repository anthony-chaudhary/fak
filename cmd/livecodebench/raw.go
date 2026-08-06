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
	"path/filepath"
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
	// Shared sampling surface (#2106): flags default to upstream lcb_runner
	// (n=10, temperature=0.2) and register identically on both arms.
	sampling := livecodebench.DefaultSampling()
	sampling.RegisterFlags(fs)
	maxTokens := fs.Int("max-tokens", 2048, "max_tokens sent on each completion request")
	timeout := fs.Duration("timeout", 120*time.Second, "per-request HTTP timeout")
	out := fs.String("out", "", "write the raw-arm report JSON to this path (default: stdout)")
	// lcb_runner cache/resume parity (#2108).
	useCache := fs.Bool("use-cache", false, "reuse cached completions from --cache-dir instead of regenerating (mirrors lcb_runner --use_cache; cache key = model+prompt+n+temperature+release)")
	cacheDir := fs.String("cache-dir", filepath.Join(".fak", "lcb-gencache"), "generation-cache directory consulted and populated with --use-cache")
	continueExisting := fs.Bool("continue-existing", false, "resume from the report already at --out: problems it completed are not regenerated (mirrors lcb_runner --continue_existing; requires --out)")
	continueWithEval := fs.Bool("continue-existing-with-eval", false, "like --continue-existing (mirrors lcb_runner --continue_existing_with_eval); grading still defers to the official evaluator, so no local eval is re-run")
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
	if err := sampling.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench raw: %v\n", err)
		return 2
	}

	if *continueWithEval {
		*continueExisting = true
		fmt.Fprintln(os.Stderr, "livecodebench raw: --continue-existing-with-eval resumes generation; grading defers to the official lcb_runner evaluator, so no local eval is re-run")
	}
	if *continueExisting && strings.TrimSpace(*out) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench raw: --continue-existing requires --out (the existing report is resumed in place)")
		return 2
	}

	suite, err := livecodebench.LoadSuiteFile(*suitePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench raw: %v\n", err)
		return 1
	}

	// #2108 resume: an existing --out report is the prior run to continue from.
	// A missing file is a fresh run; an unreadable one is an error, so a corrupt
	// artifact is never silently regenerated over.
	var prior *livecodebench.RawArmReport
	if *continueExisting {
		raw, rerr := os.ReadFile(*out)
		switch {
		case rerr == nil:
			var p livecodebench.RawArmReport
			if uerr := json.Unmarshal(raw, &p); uerr != nil {
				fmt.Fprintf(os.Stderr, "livecodebench raw: --continue-existing: %s is not a raw-arm report: %v\n", *out, uerr)
				return 1
			}
			prior = &p
		case os.IsNotExist(rerr):
			fmt.Fprintf(os.Stderr, "livecodebench raw: --continue-existing: no report at %s yet, running fresh\n", *out)
		default:
			fmt.Fprintf(os.Stderr, "livecodebench raw: --continue-existing: %v\n", rerr)
			return 1
		}
	}
	var cache livecodebench.GenCache
	if *useCache {
		cache = livecodebench.DirGenCache{Dir: *cacheDir}
	}

	cfg := sampling.ArmConfig(*model, *endpoint)
	client := &http.Client{Timeout: *timeout}
	sampler := gatewaySampler(client, cfg, *maxTokens)

	res, err := livecodebench.RunRawArmCached(context.Background(), cfg, suite.ReleaseVersion, suite.Problems, sampler, cache, prior)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench raw: %v\n", err)
		return 1
	}
	report := res.Report

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
	fmt.Fprintln(os.Stderr, rawArmSummary(report))
	// #2108 honest accounting: the rate is stated against genuine lookups only;
	// problems resumed from a prior report are reported separately, never as hits.
	if *useCache {
		fmt.Fprintf(os.Stderr, "livecodebench raw: %s\n", res.Stats.Summary())
	}
	if *continueExisting {
		fmt.Fprintf(os.Stderr, "livecodebench raw: resumed %d problem(s) from the existing report\n", res.Resumed)
	}
	return 0
}

// bearerFromEnv returns the bearer credential the raw arm presents to the
// gateway, or "" when the endpoint is unauthenticated. A `fak serve` started
// with --require-key-env rejects an unauthenticated POST with HTTP 401, so
// benchmarking a key-guarded in-kernel serve was previously impossible without
// taking the key requirement off the serve. The credential is read from the
// environment (never a flag) so it stays out of the process table and out of
// the emitted report.
func rawArmSummary(report livecodebench.RawArmReport) string {
	return fmt.Sprintf("livecodebench raw: %d problem(s)  n=%d via %s (model %s), %d cached prompt tokens, %d truncated, %d reasoning-only",
		len(report.Problems), report.N, report.Endpoint, report.Model, report.Usage.CachedPromptTokens, report.Usage.Truncated, report.Usage.ReasoningOnly)
}

func bearerFromEnv() string {
	for _, env := range []string{"LCB_API_KEY", "FAK_GATEWAY_KEY", "OPENAI_API_KEY"} {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			return v
		}
	}
	return ""
}

// gatewaySampler returns a RawArmSampler that POSTs one OpenAI-compatible
// chat-completions request per call and returns the completion text plus the
// provider-relayed usage, normalized here through agent.Usage.CachedPromptTokens()
// into livecodebench.RawSampleUsage (the agent import stays on this cmd side so
// internal/livecodebench keeps its foundation tier).
func gatewaySampler(client *http.Client, cfg livecodebench.RawArmConfig, maxTokens int) livecodebench.RawArmSampler {
	url := strings.TrimRight(cfg.Endpoint, "/") + "/chat/completions"
	return func(ctx context.Context, p livecodebench.Problem, _ int) (string, livecodebench.RawSampleUsage, error) {
		reqBody := chatCompletionsRequest{
			Model:       cfg.Model,
			Temperature: cfg.Temperature,
			Seed:        seedParam(cfg.Seed),
			MaxTokens:   maxTokens,
			Messages:    []agent.Message{{Role: agent.RoleUser, Content: p.Prompt}},
		}
		buf, err := json.Marshal(reqBody)
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		if key := bearerFromEnv(); key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := client.Do(req)
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, err
		}
		if resp.StatusCode != http.StatusOK {
			return "", livecodebench.RawSampleUsage{}, fmt.Errorf("gateway HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}
		var out chatCompletionsResponse
		if err := json.Unmarshal(body, &out); err != nil {
			return "", livecodebench.RawSampleUsage{}, err
		}
		if len(out.Choices) == 0 {
			return "", livecodebench.RawSampleUsage{}, fmt.Errorf("gateway returned no choices")
		}
		choice := out.Choices[0]
		content, reasoningOnly := choice.Message.Content, false
		if strings.TrimSpace(content) == "" && strings.TrimSpace(choice.Message.ReasoningContent) != "" {
			content, reasoningOnly = choice.Message.ReasoningContent, true
		}
		return content, livecodebench.RawSampleUsage{PromptTokens: out.Usage.PromptTokens, CompletionTokens: out.Usage.CompletionTokens, CachedPromptTokens: out.Usage.CachedPromptTokens(), FinishReason: choice.FinishReason, ReasoningOnly: reasoningOnly}, nil
	}
}

type chatCompletionsRequest struct {
	Model       string          `json:"model"`
	Temperature float64         `json:"temperature"`
	Seed        *int64          `json:"seed,omitempty"` // sent only when pinned (#2106)
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Messages    []agent.Message `json:"messages"`
}

// seedParam maps the shared sampling seed onto the wire: 0 means "provider
// default", so the request field is omitted rather than sent as a literal 0.
func seedParam(seed int64) *int64 {
	if seed == 0 {
		return nil
	}
	return &seed
}

type chatCompletionsResponse struct {
	Choices []struct {
		Message      agent.Message `json:"message"`
		FinishReason string        `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage agent.Usage `json:"usage"`
}
