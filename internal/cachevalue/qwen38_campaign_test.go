package cachevalue

import "testing"

func TestFoldQwen38CampaignPassesFourModesAndInvalidation(t *testing.T) {
	c := fixtureQwen38Campaign()
	r, err := FoldQwen38Campaign(c)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" || !r.EquivalenceVerified || !r.InvalidationVerified {
		t.Fatalf("report = %+v", r)
	}
	if r.CacheKey == "" || len(r.Modes) != 4 {
		t.Fatalf("cache key/modes missing: %+v", r)
	}
	if r.Modes[2].NetWallSavedP50MS == nil || *r.Modes[2].NetWallSavedP50MS <= 0 {
		t.Fatalf("fak net savings = %+v", r.Modes[2])
	}
}

func TestFoldQwen38CampaignFailsClosedOnWrongIdentity(t *testing.T) {
	c := fixtureQwen38Campaign()
	c.Identity.Ref = "hf://some/other-model"
	if _, err := FoldQwen38Campaign(c); err == nil {
		t.Fatal("expected exact-identity error")
	}
}

func TestFoldQwen38CampaignHoldsStaleReuse(t *testing.T) {
	c := fixtureQwen38Campaign()
	c.Observations[2].OutputHash = "stale"
	r, err := FoldQwen38Campaign(c)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "HOLD" || r.EquivalenceVerified {
		t.Fatalf("report = %+v", r)
	}
}

func fixtureQwen38Campaign() Qwen38Campaign {
	id := Qwen38Identity{Alias: Qwen38DefaultAlias, Ref: Qwen38DefaultRef, Revision: "rev", SHA256: "weights", TokenizerSHA256: "tokenizer", ChatTemplateHash: "template", Quant: "Q4_K_M", Backend: "metal", ToolSchemaHash: "tools", PolicyHash: "policy"}
	obs := func(mode string, trial int, wall float64, reused int64, hit bool) Qwen38Observation {
		return Qwen38Observation{Mode: mode, Trial: trial, WallMS: wall, TTFTMS: wall / 2, PrefillTokensPerSec: 100, DecodeTokensPerSec: 20, PromptTokens: 1000, ReusedPromptTokens: reused, CacheLookupMS: 1, SerializationMS: 1, OutputHash: "text", ToolCallHash: "tool", StructuredJSONHash: "json", CacheHit: hit}
	}
	return Qwen38Campaign{Schema: Qwen38CampaignSchema, Workload: Qwen38Workload{Turns: 5, RepeatedSystemPrompt: true, RepeatedToolSchema: true, GrowingConversation: true, CorrelatedToolCalls: true, PrefixMutation: true, RestartBoundary: true}, Corpus: "qwen38-workflow-cache-spine-v1", Hardware: "apple-m4-max", Identity: id, Observations: []Qwen38Observation{obs("cold", 1, 100, 0, false), obs("native", 1, 80, 400, true), obs("fak", 1, 60, 700, true), obs("combined", 1, 50, 800, true), {Mode: "fak", Trial: 2, WallMS: 95, TTFTMS: 45, PromptTokens: 1000, OutputHash: "mutated", ToolCallHash: "tool2", StructuredJSONHash: "json2", ExpectedInvalidation: true, CacheHit: false}}}
}
