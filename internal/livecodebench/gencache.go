package livecodebench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// gencache.go wires the #2108 primitives (CacheKeyInput, PendingWork,
// CacheStats in cache.go) into the raw generation arm so a re-run does not
// re-spend tokens. GenCache is the reuse seam behind --use-cache; DirGenCache
// is its on-disk implementation; RunRawArmCached composes cache reuse and
// --continue-existing resume around RunRawArm without changing it.

// GenCache stores the n completions generated for one cache key (the
// CacheKeyInput identity: model + prompt + n + temperature + release). A Get
// that returns ok=false means the caller must regenerate; Put is best-effort —
// a failed persist costs a future miss, never this run's correctness.
type GenCache interface {
	Get(key string) ([]string, bool)
	Put(key string, completions []string)
}

// DirGenCache is a GenCache holding one JSON file per key under Dir. Keys are
// CacheKeyInput.CacheKey() values ("lcbgen_" + hex), so they are always
// filesystem-safe. An unreadable or corrupt entry reads as a miss, never an
// error: the cache can only save tokens, not fail a run.
type DirGenCache struct {
	Dir string
}

func (c DirGenCache) path(key string) string { return filepath.Join(c.Dir, key+".json") }

// Get returns the completions stored for key, or ok=false on any failure.
func (c DirGenCache) Get(key string) ([]string, bool) {
	raw, err := os.ReadFile(c.path(key))
	if err != nil {
		return nil, false
	}
	var comps []string
	if err := json.Unmarshal(raw, &comps); err != nil {
		return nil, false
	}
	return comps, true
}

// Put persists the completions for key. Failures are swallowed by design:
// generation already succeeded, so the worst outcome is a miss next run.
func (c DirGenCache) Put(key string, completions []string) {
	if err := os.MkdirAll(c.Dir, 0o755); err != nil {
		return
	}
	raw, err := json.Marshal(completions)
	if err != nil {
		return
	}
	_ = os.WriteFile(c.path(key), raw, 0o644)
}

// RawCachedResult is what a cached/resumable raw-arm run reports beyond the
// artifact itself. Stats counts ONLY genuine cache lookups (the honest
// denominator from cache.go); problems carried from a prior report via
// --continue-existing are counted in Resumed, never as cache hits.
type RawCachedResult struct {
	Report  RawArmReport
	Stats   CacheStats
	Resumed int // suite problems carried complete from the prior report
}

// cacheKeyFor binds one problem to the #2108 cache identity: model + prompt +
// n + temperature + release, nothing else.
func cacheKeyFor(cfg RawArmConfig, release string, p Problem) string {
	return CacheKeyInput{
		Model:       cfg.Model,
		Prompt:      p.Prompt,
		N:           cfg.N,
		Temperature: cfg.Temperature,
		Release:     release,
	}.CacheKey()
}

// RunRawArmCached runs the raw arm with #2108's reuse semantics layered around
// RunRawArm:
//
//   - prior (from --continue-existing) carries every problem the existing
//     report already completed with exactly n completions; those problems are
//     never regenerated (PendingWork guarantees no duplicates). The prior run's
//     identity must match cfg — resuming under a different model, n,
//     temperature, or seed is refused rather than silently mixed.
//   - cache (from --use-cache) is consulted once per still-pending problem; a
//     hit with exactly n completions is reused, anything else is a miss and the
//     problem is regenerated. Freshly generated completions are Put back so the
//     NEXT run can reuse them — including runs interrupted before writing a
//     report, since Put happens per problem, not per run.
//   - the merged report lists every suite problem in suite order. Its usage sums
//     the prior report's usage with this run's generation usage; completions
//     served from the cache add zero, because this run spent zero tokens on them.
//
// With cache == nil and prior == nil it is exactly RunRawArm.
func RunRawArmCached(ctx context.Context, cfg RawArmConfig, release string, problems []Problem, sample RawArmSampler, cache GenCache, prior *RawArmReport) (RawCachedResult, error) {
	var res RawCachedResult
	if cfg.N < 1 {
		return res, fmt.Errorf("livecodebench raw arm: n must be >= 1, got %d", cfg.N)
	}

	done := make(map[string][]string)
	var priorUsage RawArmUsage
	if prior != nil {
		if prior.Model != cfg.Model || prior.N != cfg.N || prior.Temperature != cfg.Temperature || prior.Seed != cfg.Seed {
			return res, fmt.Errorf("livecodebench raw arm: --continue-existing identity mismatch: prior run was model=%q n=%d temperature=%v seed=%d, this run is model=%q n=%d temperature=%v seed=%d",
				prior.Model, prior.N, prior.Temperature, prior.Seed, cfg.Model, cfg.N, cfg.Temperature, cfg.Seed)
		}
		for _, pp := range prior.Problems {
			if len(pp.Completions) == cfg.N {
				done[pp.QuestionID] = pp.Completions
			}
		}
		priorUsage = prior.Usage
	}

	ids := make([]string, len(problems))
	byID := make(map[string]Problem, len(problems))
	for i, p := range problems {
		ids[i] = p.QuestionID
		byID[p.QuestionID] = p
	}
	doneSet := make(map[string]bool, len(done))
	for id := range done {
		doneSet[id] = true
	}
	pendingIDs := PendingWork(ids, doneSet)

	cached := make(map[string][]string)
	var toGen []Problem
	for _, id := range pendingIDs {
		p := byID[id]
		if cache != nil {
			if comps, ok := cache.Get(cacheKeyFor(cfg, release, p)); ok && len(comps) == cfg.N {
				res.Stats.Hit()
				cached[id] = comps
				continue
			}
			res.Stats.Miss()
		}
		toGen = append(toGen, p)
	}

	conc := cfg.Concurrency
	if conc < 1 {
		conc = 1
	}
	maxRetries := cfg.MaxRetries
	if maxRetries < 0 {
		maxRetries = 0
	}
	var genUsage RawArmUsage
	genByID := make(map[string]RawArmProblem)
	if len(toGen) > 0 {
		genReport, err := RunRawArm(ctx, cfg, toGen, sample)
		if err != nil {
			return res, err
		}
		genUsage = genReport.Usage
		for i, pp := range genReport.Problems {
			genByID[pp.QuestionID] = pp
			if cache != nil {
				cache.Put(cacheKeyFor(cfg, release, toGen[i]), pp.Completions)
			}
		}
	}

	merged := make([]RawArmProblem, 0, len(pendingIDs)+len(done))
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		var comps []string
		var finishReasons []string
		var reasoningOnly []bool
		switch {
		case done[id] != nil:
			comps = done[id]
			res.Resumed++
		case cached[id] != nil:
			comps = cached[id]
		default:
			generated := genByID[id]
			comps = generated.Completions
			finishReasons = generated.FinishReasons
			reasoningOnly = generated.ReasoningOnly
		}
		merged = append(merged, RawArmProblem{QuestionID: id, PromptSHA256: promptSHA256(byID[id].Prompt), Completions: comps, FinishReasons: finishReasons, ReasoningOnly: reasoningOnly})
	}

	res.Report = RawArmReport{
		Arm:                "raw",
		Model:              cfg.Model,
		Endpoint:           cfg.Endpoint,
		N:                  cfg.N,
		Temperature:        cfg.Temperature,
		Seed:               cfg.Seed,
		Concurrency:        conc,
		MaxRetries:         maxRetries,
		Release:            release,
		Problems:           merged,
		ResultClaimAllowed: priorUsage.Truncated+genUsage.Truncated == 0,
		Usage: RawArmUsage{
			Samples:            priorUsage.Samples + genUsage.Samples,
			PromptTokens:       priorUsage.PromptTokens + genUsage.PromptTokens,
			CompletionTokens:   priorUsage.CompletionTokens + genUsage.CompletionTokens,
			CachedPromptTokens: priorUsage.CachedPromptTokens + genUsage.CachedPromptTokens,
			Retries:            priorUsage.Retries + genUsage.Retries,
			Truncated:          priorUsage.Truncated + genUsage.Truncated,
			ReasoningOnly:      priorUsage.ReasoningOnly + genUsage.ReasoningOnly,
		},
	}
	return res, nil
}
