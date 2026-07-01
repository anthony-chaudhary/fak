package ratelimit

import (
	"encoding/json"
	"math"
	"strconv"
)

// UnlimitedHeadroom is returned for a headroom dimension with no configured cap.
const UnlimitedHeadroom int64 = -1

// UnlimitedConcurrentHeadroom is returned for concurrency when MaxConcurrent is
// unset.
const UnlimitedConcurrentHeadroom = -1

const (
	TokenCapConcurrency         = "concurrency"
	TokenCapInputTokens         = "input_tokens"
	TokenCapCachedInputTokens   = "cached_input_tokens"
	TokenCapUncachedInputTokens = "uncached_input_tokens"
	TokenCapOutputTokens        = "output_tokens"
	TokenCapTotalTokens         = "total_tokens"
)

// TokenUsage is provider-normalized token accounting. InputTokens is the
// provider's total prompt/input token count; CachedInputTokens and
// UncachedInputTokens partition that input count; OutputTokens is the generated
// completion/output count.
type TokenUsage struct {
	InputTokens         int64
	CachedInputTokens   int64
	UncachedInputTokens int64
	OutputTokens        int64
}

// NewTokenUsage returns a normalized usage record from the common provider shape:
// total input, cached input, and output. Uncached input is derived as
// input-cached.
func NewTokenUsage(inputTokens, cachedInputTokens, outputTokens int64) TokenUsage {
	return normalizeTokenUsage(inputTokens, cachedInputTokens, 0, outputTokens, true, true, false)
}

// Normalize clamps negative values and restores the invariant
// InputTokens == CachedInputTokens + UncachedInputTokens.
func (u TokenUsage) Normalize() TokenUsage {
	return normalizeTokenUsage(
		u.InputTokens,
		u.CachedInputTokens,
		u.UncachedInputTokens,
		u.OutputTokens,
		u.InputTokens > 0,
		u.CachedInputTokens > 0,
		u.UncachedInputTokens > 0,
	)
}

// Add returns the normalized sum of two usage records.
func (u TokenUsage) Add(v TokenUsage) TokenUsage {
	u = u.Normalize()
	v = v.Normalize()
	return NewTokenUsage(
		u.InputTokens+v.InputTokens,
		u.CachedInputTokens+v.CachedInputTokens,
		u.OutputTokens+v.OutputTokens,
	)
}

// TotalTokens is the provider-visible input+output total.
func (u TokenUsage) TotalTokens() int64 {
	u = u.Normalize()
	return u.InputTokens + u.OutputTokens
}

// UncachedTotalTokens is the token volume that was not satisfied by input-cache
// reuse, plus output tokens. It is a capacity primitive, not a billing claim.
func (u TokenUsage) UncachedTotalTokens() int64 {
	u = u.Normalize()
	return u.UncachedInputTokens + u.OutputTokens
}

// NormalizeProviderTokens accepts the common usage payloads returned by OpenAI
// compatible providers and the Codex-style cached/uncached split. It accepts
// either the usage object itself or an object with a top-level "usage" field.
//
// Recognized aliases:
//   - input_tokens or prompt_tokens
//   - output_tokens or completion_tokens
//   - nested input_tokens_details.cached_tokens or
//     prompt_tokens_details.cached_tokens
//   - cached_input_tokens and uncached_input_tokens
func NormalizeProviderTokens(raw map[string]any) TokenUsage {
	usage := providerUsageMap(raw)
	if len(usage) == 0 {
		return TokenUsage{}
	}

	input, inputSet := tokenIntAt(usage, "input_tokens", "prompt_tokens")
	output, outputSet := tokenIntAt(usage, "output_tokens", "completion_tokens")
	cached, cachedSet := tokenIntAt(usage, "cached_input_tokens", "input_cached_tokens", "prompt_cached_tokens")
	uncached, uncachedSet := tokenIntAt(usage, "uncached_input_tokens", "input_uncached_tokens", "prompt_uncached_tokens")

	if details := tokenMapAt(usage, "input_tokens_details"); len(details) > 0 {
		if n, ok := tokenIntAt(details, "cached_tokens"); ok {
			cached, cachedSet = n, true
		}
	}
	if details := tokenMapAt(usage, "prompt_tokens_details"); len(details) > 0 {
		if n, ok := tokenIntAt(details, "cached_tokens"); ok {
			cached, cachedSet = n, true
		}
	}

	if !inputSet && outputSet {
		if total, totalSet := tokenIntAt(usage, "total_tokens"); totalSet {
			input, inputSet = total-output, true
		}
	}

	return normalizeTokenUsage(input, cached, uncached, output, inputSet, cachedSet, uncachedSet)
}

// TokenCaps are the admission ceilings a future slot scheduler or seat pool can
// consult before placing a microagent. Zero means unlimited for that dimension.
type TokenCaps struct {
	MaxConcurrent          int
	MaxInputTokens         int64
	MaxCachedInputTokens   int64
	MaxUncachedInputTokens int64
	MaxOutputTokens        int64
	MaxTotalTokens         int64
	// TargetUtilization optionally lowers every configured cap to preserve
	// provider headroom. Zero means use the full cap; 0.9 keeps roughly 10% free.
	TargetUtilization float64
}

// TokenLoad is the currently reserved or in-flight token load.
type TokenLoad struct {
	Concurrent int
	Tokens     TokenUsage
}

// Normalize clamps impossible negative load and token values.
func (l TokenLoad) Normalize() TokenLoad {
	if l.Concurrent < 0 {
		l.Concurrent = 0
	}
	l.Tokens = l.Tokens.Normalize()
	return l
}

// AfterAdmit returns the load that would result from admitting request. It does
// not mutate state; callers still own the scheduler/seat-pool mechanics.
func (l TokenLoad) AfterAdmit(request TokenUsage) TokenLoad {
	l = l.Normalize()
	l.Concurrent++
	l.Tokens = l.Tokens.Add(request)
	return l
}

// TokenHeadroom reports remaining capacity by dimension. Unlimited dimensions
// report UnlimitedHeadroom (or UnlimitedConcurrentHeadroom for concurrency).
type TokenHeadroom struct {
	Concurrent          int
	InputTokens         int64
	CachedInputTokens   int64
	UncachedInputTokens int64
	OutputTokens        int64
	TotalTokens         int64
}

// Headroom returns the current unused capacity for each configured cap.
func (c TokenCaps) Headroom(load TokenLoad) TokenHeadroom {
	load = load.Normalize()
	tokens := load.Tokens
	return TokenHeadroom{
		Concurrent:          concurrentHeadroom(c.effectiveConcurrentLimit(), load.Concurrent),
		InputTokens:         tokenHeadroom(c.effectiveTokenLimit(c.MaxInputTokens), tokens.InputTokens),
		CachedInputTokens:   tokenHeadroom(c.effectiveTokenLimit(c.MaxCachedInputTokens), tokens.CachedInputTokens),
		UncachedInputTokens: tokenHeadroom(c.effectiveTokenLimit(c.MaxUncachedInputTokens), tokens.UncachedInputTokens),
		OutputTokens:        tokenHeadroom(c.effectiveTokenLimit(c.MaxOutputTokens), tokens.OutputTokens),
		TotalTokens:         tokenHeadroom(c.effectiveTokenLimit(c.MaxTotalTokens), tokens.TotalTokens()),
	}
}

// TokenAdmission is the pure decision value for a token/concurrency admission
// check. Admit=false names the cap that fired and the requested-vs-headroom
// arithmetic needed by a caller to decide whether to wait, split, or reject.
type TokenAdmission struct {
	Admit     bool
	Cap       string
	Limit     int64
	Used      int64
	Requested int64
	Headroom  int64
}

// Decide checks whether request fits under the configured concurrency and token
// caps for the current load. It is intentionally side-effect free: reservation,
// release, fairness, and queueing remain the later scheduler's job.
func (c TokenCaps) Decide(load TokenLoad, request TokenUsage) TokenAdmission {
	load = load.Normalize()
	request = request.Normalize()

	maxConcurrent := c.effectiveConcurrentLimit()
	if maxConcurrent > 0 && load.Concurrent+1 > maxConcurrent {
		return denyTokenAdmission(TokenCapConcurrency, int64(maxConcurrent), int64(load.Concurrent), 1)
	}
	if decision := checkTokenCap(TokenCapInputTokens, c.effectiveTokenLimit(c.MaxInputTokens), load.Tokens.InputTokens, request.InputTokens); !decision.Admit {
		return decision
	}
	if decision := checkTokenCap(TokenCapCachedInputTokens, c.effectiveTokenLimit(c.MaxCachedInputTokens), load.Tokens.CachedInputTokens, request.CachedInputTokens); !decision.Admit {
		return decision
	}
	if decision := checkTokenCap(TokenCapUncachedInputTokens, c.effectiveTokenLimit(c.MaxUncachedInputTokens), load.Tokens.UncachedInputTokens, request.UncachedInputTokens); !decision.Admit {
		return decision
	}
	if decision := checkTokenCap(TokenCapOutputTokens, c.effectiveTokenLimit(c.MaxOutputTokens), load.Tokens.OutputTokens, request.OutputTokens); !decision.Admit {
		return decision
	}
	if decision := checkTokenCap(TokenCapTotalTokens, c.effectiveTokenLimit(c.MaxTotalTokens), load.Tokens.TotalTokens(), request.TotalTokens()); !decision.Admit {
		return decision
	}
	return TokenAdmission{Admit: true}
}

func (c TokenCaps) effectiveTokenLimit(limit int64) int64 {
	if limit <= 0 {
		return 0
	}
	utilization := c.targetUtilization()
	if utilization >= 1 {
		return limit
	}
	effective := int64(math.Floor(float64(limit) * utilization))
	if effective < 1 {
		return 1
	}
	return effective
}

func (c TokenCaps) effectiveConcurrentLimit() int {
	if c.MaxConcurrent <= 0 {
		return 0
	}
	utilization := c.targetUtilization()
	if utilization >= 1 {
		return c.MaxConcurrent
	}
	effective := int(math.Floor(float64(c.MaxConcurrent) * utilization))
	if effective < 1 {
		return 1
	}
	return effective
}

func (c TokenCaps) targetUtilization() float64 {
	if math.IsNaN(c.TargetUtilization) || c.TargetUtilization <= 0 || c.TargetUtilization > 1 {
		return 1
	}
	return c.TargetUtilization
}

func checkTokenCap(name string, limit, used, requested int64) TokenAdmission {
	if limit <= 0 || used+requested <= limit {
		return TokenAdmission{Admit: true}
	}
	return denyTokenAdmission(name, limit, used, requested)
}

func denyTokenAdmission(name string, limit, used, requested int64) TokenAdmission {
	return TokenAdmission{
		Admit:     false,
		Cap:       name,
		Limit:     limit,
		Used:      used,
		Requested: requested,
		Headroom:  limit - used,
	}
}

func normalizeTokenUsage(input, cached, uncached, output int64, inputSet, cachedSet, uncachedSet bool) TokenUsage {
	input = nonNegative(input)
	cached = nonNegative(cached)
	uncached = nonNegative(uncached)
	output = nonNegative(output)

	if !inputSet && (cachedSet || uncachedSet) {
		input = cached + uncached
		inputSet = true
	}
	if inputSet && !cachedSet && uncachedSet {
		if uncached > input {
			uncached = input
		}
		cached = input - uncached
		cachedSet = true
	}
	if !cachedSet {
		cached = 0
	}
	if cached > input {
		cached = input
	}
	uncached = input - cached

	return TokenUsage{
		InputTokens:         input,
		CachedInputTokens:   cached,
		UncachedInputTokens: uncached,
		OutputTokens:        output,
	}
}

func providerUsageMap(raw map[string]any) map[string]any {
	if raw == nil {
		return nil
	}
	if usage, ok := raw["usage"].(map[string]any); ok {
		return usage
	}
	return raw
}

func tokenMapAt(m map[string]any, key string) map[string]any {
	if m == nil {
		return nil
	}
	nested, _ := m[key].(map[string]any)
	return nested
}

func tokenIntAt(m map[string]any, keys ...string) (int64, bool) {
	for _, key := range keys {
		if v, ok := m[key]; ok {
			if n, ok := tokenInt(v); ok {
				return n, true
			}
		}
	}
	return 0, false
}

func tokenInt(v any) (int64, bool) {
	switch n := v.(type) {
	case int:
		return int64(n), true
	case int8:
		return int64(n), true
	case int16:
		return int64(n), true
	case int32:
		return int64(n), true
	case int64:
		return n, true
	case uint:
		return int64(n), true
	case uint8:
		return int64(n), true
	case uint16:
		return int64(n), true
	case uint32:
		return int64(n), true
	case uint64:
		if n > uint64(^uint64(0)>>1) {
			return 0, false
		}
		return int64(n), true
	case float32:
		return int64(n), true
	case float64:
		return int64(n), true
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return i, true
		}
		if f, err := strconv.ParseFloat(n.String(), 64); err == nil {
			return int64(f), true
		}
	case string:
		i, err := strconv.ParseInt(n, 10, 64)
		if err == nil {
			return i, true
		}
	}
	return 0, false
}

func tokenHeadroom(limit, used int64) int64 {
	if limit <= 0 {
		return UnlimitedHeadroom
	}
	return limit - used
}

func concurrentHeadroom(limit, used int) int {
	if limit <= 0 {
		return UnlimitedConcurrentHeadroom
	}
	return limit - used
}

func nonNegative(n int64) int64 {
	if n < 0 {
		return 0
	}
	return n
}
