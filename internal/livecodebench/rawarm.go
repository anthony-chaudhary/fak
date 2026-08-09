package livecodebench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

// rawarm.go is the "raw" A/B arm (#2104, epic #2085): LCB prompts sent to the
// fak OpenAI-compatible gateway /v1/chat/completions UNADJUDICATED, collecting
// n samples per problem with bounded concurrency that mirrors the closed-API
// lcb_runner --multiprocess fan-out. The fan-out and the report are pure here so
// they are unit-tested without a network; the CLI (cmd/livecodebench raw) injects
// the real gateway sampler. The report records the run identity the acceptance
// criteria demand — model id, endpoint, n, temperature — and folds provider-cache
// accounting from the per-sample RawSampleUsage the sampler returns. The sampler
// normalizes its provider's usage shape (agent.Usage.CachedPromptTokens(), not any
// single provider's field) BEFORE returning, so this package stays a foundation
// leaf with no upward import of the integrator-tier agent client.

// RawArmConfig pins the raw arm's run identity and its fan-out width. The
// sampling fields mirror SamplingConfig (#2106) — build it via
// SamplingConfig.ArmConfig so both arms share one sampling identity.
type RawArmConfig struct {
	Model       string  // model id recorded in the report and sent on each request
	Endpoint    string  // gateway base URL (…/v1) the completions are POSTed to
	N           int     // samples per problem (mirrors lcb_runner -n)
	Temperature float64 // sampling temperature recorded and sent
	Seed        int64   // sampling seed sent when nonzero and recorded; 0 = provider default
	Concurrency int     // max in-flight requests (mirrors closed-API --multiprocess)
	MaxRetries  int     // per-sample retry budget (#2106); a failed sample is retried and counted, never silently dropped
}

// RawSampleUsage is the token accounting for ONE sample, already normalized by
// the sampler: CachedPromptTokens must be folded from the provider's own shape
// (the gateway sampler uses agent.Usage.CachedPromptTokens()) so a provider-cache
// hit counts the same on OpenAI-compatible, DeepSeek, and Anthropic-shaped usage.
type RawSampleUsage struct {
	PromptTokens       int
	CompletionTokens   int
	CachedPromptTokens int
	FinishReason       string
	ReasoningOnly      bool
}

// RawArmSampler produces ONE completion for problem p at sample index i. The real
// implementation POSTs to the gateway's /v1/chat/completions and normalizes the
// provider-relayed usage into RawSampleUsage; tests inject a deterministic stub.
type RawArmSampler func(ctx context.Context, p Problem, i int) (content string, u RawSampleUsage, err error)

// RawArmReport is the machine-readable result of the raw arm: the run identity
// (model / endpoint / n / temperature) plus per-problem completions and the folded
// provider-cache-aware token usage.
type RawArmReport struct {
	Arm                string          `json:"arm"` // always "raw"
	Model              string          `json:"model"`
	Endpoint           string          `json:"endpoint"`
	N                  int             `json:"n"`
	Temperature        float64         `json:"temperature"`
	Seed               int64           `json:"seed,omitempty"` // 0 = provider default, omitted
	Concurrency        int             `json:"concurrency"`
	MaxRetries         int             `json:"max_retries"`       // per-sample retry budget the run honored (#2106)
	Release            string          `json:"release,omitempty"` // dataset release the suite pinned (stamped by RunRawArmCached)
	Problems           []RawArmProblem `json:"problems"`
	Usage              RawArmUsage     `json:"usage"`
	ResultClaimAllowed bool            `json:"result_claim_allowed"`
}

// RawArmProblem holds the n completions collected for one problem, in sample order.
// PromptSHA256 is the hash of the exact rendered prompt this arm sent, so a
// cross-arm comparison can assert SamePromptHash from the artifacts alone (#2105).
type RawArmProblem struct {
	QuestionID    string   `json:"question_id"`
	PromptSHA256  string   `json:"prompt_sha256,omitempty"`
	Completions   []string `json:"completions"`
	FinishReasons []string `json:"finish_reasons,omitempty"`
	ReasoningOnly []bool   `json:"reasoning_only,omitempty"`
}

// promptSHA256 is the per-problem prompt identity both arms stamp on their
// reports; equality per question_id is the SamePromptHash assertion (#2105).
func promptSHA256(prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(h[:])
}

// RawArmUsage is the run's token accounting folded across every sample. CachedPromptTokens
// is summed from the sampler-normalized RawSampleUsage.CachedPromptTokens so a provider-cache
// hit is counted the same way on OpenAI-compatible, DeepSeek, and Anthropic-shaped usage.
type RawArmUsage struct {
	Samples            int `json:"samples"`
	PromptTokens       int `json:"prompt_tokens"`
	CompletionTokens   int `json:"completion_tokens"`
	CachedPromptTokens int `json:"cached_prompt_tokens"`
	Retries            int `json:"retries"` // failed sample attempts that were retried (#2106) — counted, never silently dropped
	Truncated          int `json:"truncated"`
	ReasoningOnly      int `json:"reasoning_only"`
}

// RunRawArm fans the sampler out over every (problem, sample) pair with at most
// cfg.Concurrency requests in flight, then assembles a deterministic report. A
// failed sample is retried up to cfg.MaxRetries times, each retry counted in
// Usage.Retries (#2106); a sample that exhausts its budget aborts the run (the
// context is cancelled so in-flight siblings stop) and is returned as an error
// naming the problem and attempt count — a failure is never silently dropped.
// Completions are ordered by problem then sample index regardless of completion
// order, so the report is reproducible.
func RunRawArm(ctx context.Context, cfg RawArmConfig, problems []Problem, sample RawArmSampler) (RawArmReport, error) {
	if sample == nil {
		return RawArmReport{}, fmt.Errorf("livecodebench raw arm: sampler is required")
	}
	if cfg.N < 1 {
		return RawArmReport{}, fmt.Errorf("livecodebench raw arm: n must be >= 1, got %d", cfg.N)
	}
	conc := cfg.Concurrency
	if conc < 1 {
		conc = 1
	}
	maxRetries := cfg.MaxRetries
	if maxRetries < 0 {
		maxRetries = 0
	}

	report := RawArmReport{
		Arm:                "raw",
		Model:              cfg.Model,
		Endpoint:           cfg.Endpoint,
		N:                  cfg.N,
		Temperature:        cfg.Temperature,
		Seed:               cfg.Seed,
		Concurrency:        conc,
		MaxRetries:         maxRetries,
		Problems:           make([]RawArmProblem, len(problems)),
		ResultClaimAllowed: true,
	}

	type slot struct {
		content string
		usage   RawSampleUsage
	}
	slots := make([][]slot, len(problems))
	for pi := range problems {
		slots[pi] = make([]slot, cfg.N)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, conc)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	var retried int

	for pi := range problems {
		for si := 0; si < cfg.N; si++ {
			mu.Lock()
			stop := firstErr != nil
			mu.Unlock()
			if stop || ctx.Err() != nil {
				break
			}
			wg.Add(1)
			sem <- struct{}{}
			go func(pi, si int) {
				defer wg.Done()
				defer func() { <-sem }()
				var content string
				var u RawSampleUsage
				var err error
				attempts := 0
				for {
					attempts++
					content, u, err = sample(ctx, problems[pi], si)
					// A cancelled run doesn't burn retries: the abort cause is
					// the sibling's error, not this sample's.
					if err == nil || attempts > maxRetries || ctx.Err() != nil {
						break
					}
					mu.Lock()
					retried++
					mu.Unlock()
				}
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("livecodebench raw arm: problem %q sample %d failed after %d attempt(s): %w", problems[pi].QuestionID, si, attempts, err)
					}
					mu.Unlock()
					cancel()
					return
				}
				slots[pi][si] = slot{content: content, usage: u}
			}(pi, si)
		}
	}
	wg.Wait()

	if firstErr != nil {
		return RawArmReport{}, firstErr
	}
	report.Usage.Retries = retried

	for pi := range problems {
		comps := make([]string, cfg.N)
		finishReasons := make([]string, cfg.N)
		reasoningOnly := make([]bool, cfg.N)
		for si := 0; si < cfg.N; si++ {
			comps[si] = slots[pi][si].content
			finishReasons[si] = slots[pi][si].usage.FinishReason
			reasoningOnly[si] = slots[pi][si].usage.ReasoningOnly
			u := slots[pi][si].usage
			report.Usage.Samples++
			report.Usage.PromptTokens += u.PromptTokens
			report.Usage.CompletionTokens += u.CompletionTokens
			report.Usage.CachedPromptTokens += u.CachedPromptTokens
			if u.FinishReason == "length" {
				report.Usage.Truncated++
				report.ResultClaimAllowed = false
			}
			if u.ReasoningOnly {
				report.Usage.ReasoningOnly++
			}
		}
		report.Problems[pi] = RawArmProblem{QuestionID: problems[pi].QuestionID, PromptSHA256: promptSHA256(problems[pi].Prompt), Completions: comps, FinishReasons: finishReasons, ReasoningOnly: reasoningOnly}
	}

	return report, nil
}
