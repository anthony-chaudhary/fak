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

// runFak implements `livecodebench fak` (#2105, epic #2085): the "fak" A/B arm.
// It sends the SAME suite problems through the fak gateway's ADJUDICATED
// /v1/chat/completions path with the same fan-out semantics as the raw arm
// (same problems / model / n / temperature / release), and folds the gateway's
// `fak` response extension (adjudications, policy denials, safe-resolve
// repairs, result admissions) into the report as evidence. It grades nothing —
// pair the emitted report with the raw arm's via `livecodebench ab`.
func runFak(argv []string) int {
	fs := flag.NewFlagSet("livecodebench fak", flag.ContinueOnError)
	suitePath := fs.String("suite", "", "normalized suite JSON whose problems are sent to the gateway (required; the SAME suite as the raw arm)")
	model := fs.String("model", "", "model id sent on each request and recorded in the report (required; must match the raw arm)")
	endpoint := fs.String("endpoint", "http://127.0.0.1:18080/v1", "fak gateway base URL (…/v1) the adjudicated completions are POSTed to")
	n := fs.Int("n", 1, "samples to generate per problem (must match the raw arm)")
	temperature := fs.Float64("temperature", 0.2, "sampling temperature sent and recorded (must match the raw arm)")
	concurrency := fs.Int("concurrency", 8, "max in-flight gateway requests")
	maxTokens := fs.Int("max-tokens", 2048, "max_tokens sent on each completion request")
	timeout := fs.Duration("timeout", 120*time.Second, "per-request HTTP timeout")
	out := fs.String("out", "", "write the fak-arm report JSON to this path (default: stdout)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench fak: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*suitePath) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench fak: --suite is required")
		return 2
	}
	if strings.TrimSpace(*model) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench fak: --model is required")
		return 2
	}

	suite, err := livecodebench.LoadSuiteFile(*suitePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench fak: %v\n", err)
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
	sampler := fakGatewaySampler(client, *endpoint, *model, *temperature, *maxTokens)

	report, err := livecodebench.RunFakArm(context.Background(), cfg, suite.ReleaseVersion, suite.Problems, sampler)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench fak: %v\n", err)
		return 1
	}

	if err := writeJSON(*out, report); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench fak: %v\n", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "livecodebench fak: %d problem(s) × n=%d via %s (model %s, release %s); adjudicated samples %d/%d, denied %d, safe-resolves %d\n",
		len(report.Problems), report.N, report.Endpoint, report.Model, report.Release,
		report.Adjudication.AdjudicatedSamples, report.Usage.Samples, report.Adjudication.Denied, report.Adjudication.SafeResolves)
	return 0
}

// runAB implements `livecodebench ab` (#2105): it loads one raw-arm and one
// fak-arm report and emits the two-arm comparison — per-arm summaries, the
// SameProblemIDs / SamePromptHash identity assertions, token deltas, and the
// fak arm's adjudication evidence. The comparison never carries a pass-rate
// delta: grading stays with the official lcb_runner evaluator.
func runAB(argv []string) int {
	fs := flag.NewFlagSet("livecodebench ab", flag.ContinueOnError)
	rawPath := fs.String("raw", "", "raw-arm report JSON (from `livecodebench raw --out`) (required)")
	fakPath := fs.String("fak", "", "fak-arm report JSON (from `livecodebench fak --out`) (required)")
	out := fs.String("out", "", "write the comparison JSON to this path (default: stdout)")
	check := fs.Bool("check", false, "exit nonzero unless SameProblemIDs and SamePromptHash both hold")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench ab: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*rawPath) == "" || strings.TrimSpace(*fakPath) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench ab: --raw and --fak are both required")
		return 2
	}

	var raw livecodebench.RawArmReport
	if err := readJSON(*rawPath, &raw); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab: --raw: %v\n", err)
		return 1
	}
	var fak livecodebench.FakArmReport
	if err := readJSON(*fakPath, &fak); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab: --fak: %v\n", err)
		return 1
	}

	c := livecodebench.CompareArms(raw, fak)
	if err := writeJSON(*out, c); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab: %v\n", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "livecodebench ab: same_problem_ids=%t same_prompt_hash=%t same_model=%t same_release=%t; %s\n",
		c.SameProblemIDs, c.SamePromptHash, c.SameModel, c.SameRelease, c.PassRateDelta)
	for _, m := range c.Mismatches {
		fmt.Fprintf(os.Stderr, "livecodebench ab: mismatch: %s\n", m)
	}
	if *check && (!c.SameProblemIDs || !c.SamePromptHash) {
		fmt.Fprintln(os.Stderr, "livecodebench ab: --check failed: the arms did not run the identical problems/prompts")
		return 1
	}
	return 0
}

// fakGatewaySampler mirrors gatewaySampler (raw.go) but also extracts the
// gateway's non-standard `fak` response extension into FakSampleEvidence, so
// the fak arm's report carries adjudication evidence per sample. A response
// without the extension yields zero evidence — the arm never invents it.
func fakGatewaySampler(client *http.Client, endpoint, model string, temperature float64, maxTokens int) livecodebench.FakArmSampler {
	url := strings.TrimRight(endpoint, "/") + "/chat/completions"
	return func(ctx context.Context, p livecodebench.Problem, _ int) (string, livecodebench.RawSampleUsage, livecodebench.FakSampleEvidence, error) {
		var ev livecodebench.FakSampleEvidence
		reqBody := chatCompletionsRequest{
			Model:       model,
			Temperature: temperature,
			MaxTokens:   maxTokens,
			Messages:    []agent.Message{{Role: agent.RoleUser, Content: p.Prompt}},
		}
		buf, err := json.Marshal(reqBody)
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, ev, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, ev, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, ev, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", livecodebench.RawSampleUsage{}, ev, err
		}
		if resp.StatusCode != http.StatusOK {
			return "", livecodebench.RawSampleUsage{}, ev, fmt.Errorf("gateway HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}
		var out fakChatCompletionsResponse
		if err := json.Unmarshal(body, &out); err != nil {
			return "", livecodebench.RawSampleUsage{}, ev, err
		}
		if len(out.Choices) == 0 {
			return "", livecodebench.RawSampleUsage{}, ev, fmt.Errorf("gateway returned no choices")
		}
		if out.Fak != nil {
			ev.Adjudicated = true
			ev.Adjudications = len(out.Fak.Adjudications)
			for _, a := range out.Fak.Adjudications {
				if !a.Admitted {
					ev.Denied++
				}
				if len(a.RepairedArguments) > 0 {
					ev.SafeResolves++
				}
			}
			ev.ResultAdmissions = len(out.Fak.ResultAdmissions)
		}
		return out.Choices[0].Message.Content, livecodebench.RawSampleUsage{
			PromptTokens:       out.Usage.PromptTokens,
			CompletionTokens:   out.Usage.CompletionTokens,
			CachedPromptTokens: out.Usage.CachedPromptTokens(),
		}, ev, nil
	}
}

// fakChatCompletionsResponse is chatCompletionsResponse plus the gateway's
// `fak` extension, decoded loosely: only the adjudication fields the arm folds
// are named, so wire additions never break the sampler.
type fakChatCompletionsResponse struct {
	Choices []struct {
		Message agent.Message `json:"message"`
	} `json:"choices"`
	Usage agent.Usage `json:"usage"`
	Fak   *struct {
		Adjudications []struct {
			Admitted          bool            `json:"admitted"`
			RepairedArguments json.RawMessage `json:"repaired_arguments"`
		} `json:"adjudications"`
		ResultAdmissions []json.RawMessage `json:"result_admissions"`
	} `json:"fak"`
}

func readJSON(path string, v any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, v)
}

func writeJSON(path string, v any) error {
	w := io.Writer(os.Stdout)
	if strings.TrimSpace(path) != "" {
		file, err := os.Create(path)
		if err != nil {
			return err
		}
		defer file.Close()
		w = file
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
