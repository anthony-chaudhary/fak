package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

type dogfoodValueAxis struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}
type dogfoodKernelValue struct {
	Schema                  string                  `json:"schema"`
	RepoPulseLaunches       int                     `json:"repo_pulse_launches"`
	ContextTokensSaved      int                     `json:"context_tokens_saved"`
	ToolTurnsSkipped        int                     `json:"tool_turns_skipped"`
	Cohort                  dogfoodValueAxis        `json:"cohort"`
	Cache                   dogfoodValueAxis        `json:"cache"`
	Velocity                dogfoodVelocityReadback `json:"velocity"`
	CacheCachedPromptTokens int                     `json:"cache_cached_prompt_tokens,omitempty"`
}

const dogfoodRepoPulseDefaultCutoff = "20260813-122202"

var dogfoodDispatchWitnessName = regexp.MustCompile(`^resolve-[0-9]+-([0-9]{8}-[0-9]{6})\.witness$`)

type dogfoodOutcomeCohort struct {
	Launches int     `json:"launches"`
	Shipped  int     `json:"shipped"`
	ShipRate float64 `json:"ship_rate"`
}

type dogfoodVelocityReadback struct {
	Status         string               `json:"status"`
	Reason         string               `json:"reason"`
	Cutoff         string               `json:"cutoff"`
	MatchedSamples int                  `json:"matched_samples"`
	Pre            dogfoodOutcomeCohort `json:"pre"`
	Post           dogfoodOutcomeCohort `json:"post"`
	ShipRateDelta  float64              `json:"ship_rate_delta"`
}

type dogfoodWitnessOutcome struct {
	Timestamp string
	Shipped   bool
}

func collectDogfoodVelocity(runsDir string, minimum int) dogfoodVelocityReadback {
	if minimum < 1 {
		minimum = 1
	}
	matches, _ := filepath.Glob(filepath.Join(runsDir, "resolve-*.witness"))
	var pre, post []dogfoodWitnessOutcome
	for _, path := range matches {
		name := filepath.Base(path)
		parts := dogfoodDispatchWitnessName.FindStringSubmatch(name)
		if len(parts) != 2 {
			continue
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var row struct {
			Claim string `json:"claim"`
		}
		if json.Unmarshal(raw, &row) != nil || !strings.HasPrefix(row.Claim, "CLAIM_") {
			continue
		}
		outcome := dogfoodWitnessOutcome{Timestamp: parts[1], Shipped: row.Claim == "CLAIM_WITNESSED"}
		if outcome.Timestamp < dogfoodRepoPulseDefaultCutoff {
			pre = append(pre, outcome)
		} else {
			post = append(post, outcome)
		}
	}
	sort.Slice(pre, func(i, j int) bool { return pre[i].Timestamp > pre[j].Timestamp })
	sort.Slice(post, func(i, j int) bool { return post[i].Timestamp < post[j].Timestamp })
	n := min(len(pre), len(post))
	pre, post = pre[:n], post[:n]
	out := dogfoodVelocityReadback{Status: "not-yet", Cutoff: dogfoodRepoPulseDefaultCutoff, MatchedSamples: n}
	for _, row := range pre {
		out.Pre.Launches++
		if row.Shipped {
			out.Pre.Shipped++
		}
	}
	for _, row := range post {
		out.Post.Launches++
		if row.Shipped {
			out.Post.Shipped++
		}
	}
	if n > 0 {
		out.Pre.ShipRate = float64(out.Pre.Shipped) / float64(n)
		out.Post.ShipRate = float64(out.Post.Shipped) / float64(n)
		out.ShipRateDelta = out.Post.ShipRate - out.Pre.ShipRate
	}
	if n < minimum {
		out.Reason = fmt.Sprintf("need %d more matched dispatch outcome(s) before velocity comparison", minimum-n)
		return out
	}
	if out.ShipRateDelta > 0 {
		out.Status = "improved"
		out.Reason = "post-default matched dispatch ship rate exceeds the immediately preceding cohort"
	} else if out.ShipRateDelta < 0 {
		out.Status = "regressed"
		out.Reason = "post-default matched dispatch ship rate is below the immediately preceding cohort"
	} else {
		out.Status = "flat"
		out.Reason = "post-default matched dispatch ship rate is unchanged"
	}
	return out
}

type cacheWitnessReadback struct {
	Schema     string        `json:"schema"`
	CapturedAt time.Time     `json:"captured_at"`
	Verdict    string        `json:"verdict"`
	Reason     string        `json:"reason"`
	On         microCacheArm `json:"affinity_on"`
	Off        microCacheArm `json:"affinity_off"`
}

const dogfoodCacheWitnessMaxAge = 24 * time.Hour

var dogfoodNow = time.Now

func collectDogfoodKernelValue(runsDir, cacheReceipt string, minimum int) dogfoodKernelValue {
	t := foldDispatchRepoPulseReceipts(runsDir)
	readiness := assessRepoPulseCohort(runsDir, minimum)
	out := dogfoodKernelValue{Schema: "fak-dogfood-kernel-value/1", RepoPulseLaunches: t.Launches, ContextTokensSaved: int(t.SavedTokens), ToolTurnsSkipped: int(t.ToolTurnsSkipped), Cohort: dogfoodValueAxis{Status: readiness.Verdict, Reason: readiness.Reason}, Cache: dogfoodValueAxis{Status: "not-yet", Reason: "no typed cache-affinity witness supplied"}, Velocity: collectDogfoodVelocity(runsDir, minimum)}
	if cacheReceipt == "" {
		return out
	}
	raw, err := os.ReadFile(cacheReceipt)
	if os.IsNotExist(err) {
		return out
	}
	if err != nil {
		out.Cache.Reason = "cache witness unreadable: " + err.Error()
		return out
	}
	var receipt cacheWitnessReadback
	if err := json.Unmarshal(raw, &receipt); err != nil {
		out.Cache.Reason = "cache witness is invalid JSON: " + err.Error()
		return out
	}
	if receipt.Schema != microCacheWitnessSchema {
		out.Cache.Reason = fmt.Sprintf("cache witness schema %q is unsupported; want %q", receipt.Schema, microCacheWitnessSchema)
		return out
	}
	if receipt.CapturedAt.IsZero() {
		out.Cache.Reason = "cache witness capture time is missing"
		return out
	}
	if receipt.Verdict != "ready" && receipt.Verdict != "not-yet" {
		out.Cache.Reason = fmt.Sprintf("cache witness verdict %q is unsupported", receipt.Verdict)
		return out
	}
	now := dogfoodNow().UTC()
	age := now.Sub(receipt.CapturedAt)
	if age < 0 {
		out.Cache.Reason = fmt.Sprintf("cache witness capture time %s is in the future", receipt.CapturedAt.UTC().Format(time.RFC3339))
		return out
	}
	if age > dogfoodCacheWitnessMaxAge {
		out.Cache.Reason = fmt.Sprintf("cache witness is stale (%s old; maximum %s); capture a fresh live affinity-on/off receipt", age.Round(time.Second), dogfoodCacheWitnessMaxAge)
		return out
	}
	out.Cache = dogfoodValueAxis{Status: receipt.Verdict, Reason: receipt.Reason}
	out.CacheCachedPromptTokens = receipt.On.CachedPromptTokens + receipt.Off.CachedPromptTokens
	return out
}
