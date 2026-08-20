package cachevalue

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const (
	Qwen38CampaignSchema = "fak.qwen38_cache_campaign.v1"
	Qwen38ReportSchema   = "fak.qwen38_cache_report.v1"
	Qwen38DefaultAlias   = "qwen38:27b"
	Qwen38DefaultRef     = "hf://unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
)

var qwen38Modes = []string{"cold", "native", "fak", "combined"}

type Qwen38Identity struct {
	Alias            string `json:"alias"`
	Ref              string `json:"ref"`
	Revision         string `json:"revision"`
	SHA256           string `json:"sha256"`
	TokenizerSHA256  string `json:"tokenizer_sha256"`
	ChatTemplateHash string `json:"chat_template_hash"`
	Quant            string `json:"quant"`
	Backend          string `json:"backend"`
	ToolSchemaHash   string `json:"tool_schema_hash"`
	PolicyHash       string `json:"policy_hash"`
}

type Qwen38Observation struct {
	Mode                 string  `json:"mode"`
	Trial                int     `json:"trial"`
	WallMS               float64 `json:"wall_ms"`
	TTFTMS               float64 `json:"ttft_ms"`
	PrefillTokensPerSec  float64 `json:"prefill_tokens_per_sec"`
	DecodeTokensPerSec   float64 `json:"decode_tokens_per_sec"`
	PromptTokens         int64   `json:"prompt_tokens"`
	ReusedPromptTokens   int64   `json:"reused_prompt_tokens"`
	CacheLookupMS        float64 `json:"cache_lookup_ms"`
	SerializationMS      float64 `json:"serialization_ms"`
	OutputHash           string  `json:"output_hash"`
	ToolCallHash         string  `json:"tool_call_hash"`
	StructuredJSONHash   string  `json:"structured_json_hash"`
	ExpectedInvalidation bool    `json:"expected_invalidation,omitempty"`
	CacheHit             bool    `json:"cache_hit"`
}

type Qwen38Workload struct {
	Turns                int  `json:"turns"`
	RepeatedSystemPrompt bool `json:"repeated_system_prompt"`
	RepeatedToolSchema   bool `json:"repeated_tool_schema"`
	GrowingConversation  bool `json:"growing_conversation"`
	CorrelatedToolCalls  bool `json:"correlated_tool_calls"`
	PrefixMutation       bool `json:"prefix_mutation"`
	RestartBoundary      bool `json:"restart_boundary"`
}

type Qwen38Campaign struct {
	Schema       string              `json:"schema"`
	Corpus       string              `json:"corpus"`
	Hardware     string              `json:"hardware"`
	Identity     Qwen38Identity      `json:"identity"`
	Workload     Qwen38Workload      `json:"workload"`
	Observations []Qwen38Observation `json:"observations"`
}

type Qwen38ModeReport struct {
	Mode                string   `json:"mode"`
	Status              string   `json:"status"`
	Reason              string   `json:"reason,omitempty"`
	Trials              int      `json:"trials"`
	WallP50MS           *float64 `json:"wall_p50_ms,omitempty"`
	WallP95MS           *float64 `json:"wall_p95_ms,omitempty"`
	TTFTP50MS           *float64 `json:"ttft_p50_ms,omitempty"`
	TTFTP95MS           *float64 `json:"ttft_p95_ms,omitempty"`
	PromptTokens        int64    `json:"prompt_tokens"`
	ReusedPromptTokens  int64    `json:"reused_prompt_tokens"`
	HitRate             *float64 `json:"hit_rate,omitempty"`
	GrossWallSavedP50MS *float64 `json:"gross_wall_saved_p50_ms,omitempty"`
	NetWallSavedP50MS   *float64 `json:"net_wall_saved_p50_ms,omitempty"`
}

type Qwen38Report struct {
	Schema               string             `json:"schema"`
	CampaignSchema       string             `json:"campaign_schema"`
	Corpus               string             `json:"corpus"`
	Hardware             string             `json:"hardware"`
	Alias                string             `json:"alias"`
	Ref                  string             `json:"ref"`
	CacheKey             string             `json:"cache_key"`
	Verdict              string             `json:"verdict"`
	Reasons              []string           `json:"reasons,omitempty"`
	EquivalenceVerified  bool               `json:"equivalence_verified"`
	InvalidationVerified bool               `json:"invalidation_verified"`
	Modes                []Qwen38ModeReport `json:"modes"`
}

func FoldQwen38Campaign(c Qwen38Campaign) (Qwen38Report, error) {
	r := Qwen38Report{Schema: Qwen38ReportSchema, CampaignSchema: c.Schema, Corpus: c.Corpus, Hardware: c.Hardware, Alias: c.Identity.Alias, Ref: c.Identity.Ref, Verdict: "HOLD"}
	if c.Schema != Qwen38CampaignSchema {
		return r, fmt.Errorf("schema: got %q, want %q", c.Schema, Qwen38CampaignSchema)
	}
	if strings.TrimSpace(c.Corpus) == "" || strings.TrimSpace(c.Hardware) == "" {
		return r, errors.New("corpus and hardware are required")
	}
	if err := validateQwen38Identity(c.Identity); err != nil {
		return r, err
	}
	if c.Workload.Turns < 4 || !c.Workload.RepeatedSystemPrompt || !c.Workload.RepeatedToolSchema || !c.Workload.GrowingConversation || !c.Workload.CorrelatedToolCalls || !c.Workload.PrefixMutation || !c.Workload.RestartBoundary {
		return r, errors.New("workload must prove repeated setup, growing turns, correlated tools, mutation, and restart")
	}
	r.CacheKey = qwen38CacheKey(c.Identity)
	byMode := map[string][]Qwen38Observation{}
	var expectedInvalidations, correctInvalidations int
	baselineHashes := map[string]bool{}
	for i, o := range c.Observations {
		if !containsQwen38Mode(qwen38Modes, o.Mode) {
			return r, fmt.Errorf("observation %d: unknown mode %q", i, o.Mode)
		}
		if o.Trial < 1 || o.WallMS <= 0 || o.TTFTMS < 0 || o.PromptTokens < 0 || o.ReusedPromptTokens < 0 || o.ReusedPromptTokens > o.PromptTokens {
			return r, fmt.Errorf("observation %d: invalid measurements", i)
		}
		if o.OutputHash == "" || o.ToolCallHash == "" || o.StructuredJSONHash == "" {
			return r, fmt.Errorf("observation %d: all three equivalence hashes are required", i)
		}
		if o.ExpectedInvalidation {
			expectedInvalidations++
			if !o.CacheHit && o.ReusedPromptTokens == 0 {
				correctInvalidations++
			}
		}
		if o.Mode == "cold" {
			baselineHashes[hashTriple(o)] = true
		}
		byMode[o.Mode] = append(byMode[o.Mode], o)
	}
	if len(byMode["cold"]) == 0 {
		return r, errors.New("cold mode requires at least one observation")
	}
	coldWall := percentile(observationValues(byMode["cold"], func(o Qwen38Observation) float64 { return o.WallMS }), .50)
	equivalent := true
	for _, mode := range qwen38Modes {
		obs := byMode[mode]
		mr := Qwen38ModeReport{Mode: mode, Status: "N/A", Reason: "no observations for this backend/hardware"}
		if len(obs) == 0 {
			r.Modes = append(r.Modes, mr)
			continue
		}
		mr.Status, mr.Reason, mr.Trials = "MEASURED", "", len(obs)
		walls := observationValues(obs, func(o Qwen38Observation) float64 { return o.WallMS })
		ttfts := observationValues(obs, func(o Qwen38Observation) float64 { return o.TTFTMS })
		wall50, wall95, ttft50, ttft95 := percentile(walls, .50), percentile(walls, .95), percentile(ttfts, .50), percentile(ttfts, .95)
		mr.WallP50MS, mr.WallP95MS, mr.TTFTP50MS, mr.TTFTP95MS = &wall50, &wall95, &ttft50, &ttft95
		var hits int
		var overhead float64
		for _, o := range obs {
			mr.PromptTokens += o.PromptTokens
			mr.ReusedPromptTokens += o.ReusedPromptTokens
			overhead += o.CacheLookupMS + o.SerializationMS
			if o.CacheHit {
				hits++
			}
			if !o.ExpectedInvalidation && !baselineHashes[hashTriple(o)] {
				equivalent = false
			}
		}
		hitRate := float64(hits) / float64(len(obs))
		mr.HitRate = &hitRate
		if mode != "cold" {
			gross := coldWall - wall50
			net := gross - overhead/float64(len(obs))
			mr.GrossWallSavedP50MS, mr.NetWallSavedP50MS = &gross, &net
		}
		r.Modes = append(r.Modes, mr)
	}
	r.EquivalenceVerified = equivalent
	r.InvalidationVerified = expectedInvalidations > 0 && expectedInvalidations == correctInvalidations
	if !r.EquivalenceVerified {
		r.Reasons = append(r.Reasons, "OUTPUT_EQUIVALENCE_FAILED")
	}
	if !r.InvalidationVerified {
		r.Reasons = append(r.Reasons, "INVALIDATION_NOT_PROVED")
	}
	if len(byMode["fak"]) == 0 {
		r.Reasons = append(r.Reasons, "FAK_MODE_MISSING")
	}
	if len(byMode["combined"]) == 0 {
		r.Reasons = append(r.Reasons, "COMBINED_MODE_MISSING")
	}
	if len(r.Reasons) == 0 {
		r.Verdict = "PASS"
	}
	return r, nil
}

func validateQwen38Identity(i Qwen38Identity) error {
	if i.Alias != Qwen38DefaultAlias || i.Ref != Qwen38DefaultRef {
		return fmt.Errorf("exact Qwen3.8 default identity required: %s -> %s", Qwen38DefaultAlias, Qwen38DefaultRef)
	}
	for name, value := range map[string]string{"revision": i.Revision, "sha256": i.SHA256, "tokenizer_sha256": i.TokenizerSHA256, "chat_template_hash": i.ChatTemplateHash, "quant": i.Quant, "backend": i.Backend, "tool_schema_hash": i.ToolSchemaHash, "policy_hash": i.PolicyHash} {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("identity %s is required", name)
		}
	}
	return nil
}

func qwen38CacheKey(i Qwen38Identity) string {
	parts := []string{i.Ref, i.Revision, i.SHA256, i.TokenizerSHA256, i.ChatTemplateHash, i.Quant, i.Backend, i.ToolSchemaHash, i.PolicyHash}
	h := sha256.Sum256([]byte(strings.Join(parts, "\x00")))
	return "sha256:" + hex.EncodeToString(h[:])
}
func hashTriple(o Qwen38Observation) string {
	return o.OutputHash + "\x00" + o.ToolCallHash + "\x00" + o.StructuredJSONHash
}
func containsQwen38Mode(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
func observationValues(xs []Qwen38Observation, f func(Qwen38Observation) float64) []float64 {
	out := make([]float64, len(xs))
	for i, x := range xs {
		out[i] = f(x)
	}
	return out
}
func percentile(xs []float64, q float64) float64 {
	ys := append([]float64(nil), xs...)
	sort.Float64s(ys)
	if len(ys) == 0 {
		return 0
	}
	idx := int(float64(len(ys)-1)*q + .5)
	return ys[idx]
}
