package main

import (
	"encoding/json"
	"os"
)

type dogfoodValueAxis struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}
type dogfoodKernelValue struct {
	Schema                  string           `json:"schema"`
	RepoPulseLaunches       int              `json:"repo_pulse_launches"`
	ContextTokensSaved      int              `json:"context_tokens_saved"`
	ToolTurnsSkipped        int              `json:"tool_turns_skipped"`
	Cohort                  dogfoodValueAxis `json:"cohort"`
	Cache                   dogfoodValueAxis `json:"cache"`
	CacheCachedPromptTokens int              `json:"cache_cached_prompt_tokens,omitempty"`
}

type cacheWitnessReadback struct {
	Verdict string        `json:"verdict"`
	Reason  string        `json:"reason"`
	On      microCacheArm `json:"affinity_on"`
	Off     microCacheArm `json:"affinity_off"`
}

func collectDogfoodKernelValue(runsDir, cacheReceipt string, minimum int) dogfoodKernelValue {
	t := foldDispatchRepoPulseReceipts(runsDir)
	readiness := assessRepoPulseCohort(runsDir, minimum)
	out := dogfoodKernelValue{Schema: "fak-dogfood-kernel-value/1", RepoPulseLaunches: t.Launches, ContextTokensSaved: int(t.SavedTokens), ToolTurnsSkipped: int(t.ToolTurnsSkipped), Cohort: dogfoodValueAxis{Status: readiness.Verdict, Reason: readiness.Reason}, Cache: dogfoodValueAxis{Status: "not-yet", Reason: "no typed cache-affinity witness supplied"}}
	if cacheReceipt == "" {
		return out
	}
	raw, err := os.ReadFile(cacheReceipt)
	if err != nil {
		out.Cache.Reason = "cache witness unreadable: " + err.Error()
		return out
	}
	var receipt cacheWitnessReadback
	if json.Unmarshal(raw, &receipt) != nil || receipt.Verdict == "" {
		out.Cache.Reason = "cache witness is not fak-micro-cache-affinity-witness/1 evidence"
		return out
	}
	out.Cache = dogfoodValueAxis{Status: receipt.Verdict, Reason: receipt.Reason}
	out.CacheCachedPromptTokens = receipt.On.CachedPromptTokens + receipt.Off.CachedPromptTokens
	return out
}
