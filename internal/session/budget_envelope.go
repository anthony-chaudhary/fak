package session

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// budget_envelope.go defines the user-facing managed-context budget envelope
// syntax (#1573). The compact CLI form is a comma-separated key=value list:
//
//	turns=20,tokens=200000,context=64000,wall=2h,spend=$25,throughput=40/s,max-tokens=1024,gap=250ms
//
// The parsed value is deterministic data only. Runtime callers project it onto the
// existing session Budget, Pace, TimeBudget, spend, and throughput axes they support.

// BudgetEnvelope is the canonical parsed form of a managed-context budget envelope.
type BudgetEnvelope struct {
	Budget              Budget             `json:"budget"`
	WallClockLimitNanos int64              `json:"wall_clock_limit_nanos,omitempty"`
	Pace                Pace               `json:"pace,omitempty,omitzero"`
	Spend               SpendEnvelope      `json:"spend,omitempty,omitzero"`
	Throughput          ThroughputEnvelope `json:"throughput,omitempty,omitzero"`
}

// SpendEnvelope records the user's spend ceiling in minor units so parsing is exact.
type SpendEnvelope struct {
	MaxCents int64  `json:"max_cents,omitempty"`
	Currency string `json:"currency,omitempty"`
}

// IsZero supports json omitzero.
func (s SpendEnvelope) IsZero() bool { return s.MaxCents == 0 && s.Currency == "" }

// ThroughputEnvelope records the expected/minimum throughput rates named by the user.
type ThroughputEnvelope struct {
	ExpectedTokensPerSec float64 `json:"expected_tokens_per_sec,omitempty"`
	MinTokensPerSec      float64 `json:"min_tokens_per_sec,omitempty"`
}

// IsZero supports json omitzero.
func (t ThroughputEnvelope) IsZero() bool {
	return t.ExpectedTokensPerSec == 0 && t.MinTokensPerSec == 0
}

// NewBudgetEnvelope returns the permissive default: unbounded turn/output-token
// budgets, no context/time/spend/throughput envelope, and no pace opinion.
func NewBudgetEnvelope() BudgetEnvelope {
	return BudgetEnvelope{
		Budget: Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded},
	}
}

// ParseBudgetEnvelope parses the compact CLI syntax into the deterministic envelope.
func ParseBudgetEnvelope(spec string) (BudgetEnvelope, error) {
	env := NewBudgetEnvelope()
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return env, fmt.Errorf("empty budget envelope")
	}
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, val, ok := strings.Cut(part, "=")
		if !ok {
			return env, fmt.Errorf("budget envelope item %q must be key=value", part)
		}
		key = budgetEnvelopeKey(key)
		val = strings.TrimSpace(val)
		if val == "" {
			return env, fmt.Errorf("budget envelope key %q has an empty value", key)
		}
		var err error
		switch key {
		case "turns":
			env.Budget.TurnsLeft, err = parseEnvelopeInt(val, true)
		case "tokens", "output_tokens":
			env.Budget.TokensLeft, err = parseEnvelopeInt(val, true)
		case "context", "context_tokens":
			env.Budget.ContextTokensLeft, err = parseEnvelopeInt(val, true)
		case "queries", "clarifications", "clarification_queries":
			env.Budget.ClarificationQueriesLeft, err = parseEnvelopeInt(val, false)
			if env.Budget.ClarificationQueriesLeft > 0 {
				env.Budget.ClarificationQueriesCap = env.Budget.ClarificationQueriesLeft
			}
		case "wall", "wall_clock", "duration", "time":
			var d time.Duration
			d, err = parseEnvelopeDuration(val)
			env.WallClockLimitNanos = int64(d)
		case "spend", "usd":
			env.Spend, err = parseEnvelopeSpend(val)
		case "throughput", "expected_throughput", "expected_tps", "tps":
			env.Throughput.ExpectedTokensPerSec, err = parseEnvelopeRate(val)
		case "min_throughput", "min_tps":
			env.Throughput.MinTokensPerSec, err = parseEnvelopeRate(val)
		case "max_tokens", "max_tokens_per_turn":
			env.Pace.MaxTokensPerTurn, err = parseEnvelopeInt(val, false)
		case "gap", "gap_ms", "min_gap", "min_turn_gap":
			env.Pace.MinTurnGapMs, err = parseEnvelopeGapMillis(key, val)
		default:
			return env, fmt.Errorf("unknown budget envelope key %q", key)
		}
		if err != nil {
			return env, fmt.Errorf("%s: %w", key, err)
		}
	}
	env.Budget = env.Budget.withContextCap()
	return env, nil
}

// SessionBudget projects the envelope onto the session budget axes.
func (e BudgetEnvelope) SessionBudget() Budget { return e.Budget.withContextCap() }

// SessionPace projects the envelope onto the per-turn pace axes.
func (e BudgetEnvelope) SessionPace() Pace { return e.Pace }

// WallClockLimit returns the configured wall-clock limit.
func (e BudgetEnvelope) WallClockLimit() time.Duration {
	return time.Duration(e.WallClockLimitNanos)
}

// TimeBudget returns the unstarted wall-clock budget for this envelope.
func (e BudgetEnvelope) TimeBudget() TimeBudget {
	return NewTimeBudget().WithLimit(e.WallClockLimit())
}

// ExpectedThroughput returns the runtime throughput expectation carried by the envelope.
func (e BudgetEnvelope) ExpectedThroughput() Throughput {
	return Throughput{ExpectedTokensPerSec: e.Throughput.ExpectedTokensPerSec}
}

func budgetEnvelopeKey(key string) string {
	key = strings.ToLower(strings.TrimSpace(key))
	key = strings.NewReplacer("-", "_", " ", "_").Replace(key)
	for strings.Contains(key, "__") {
		key = strings.ReplaceAll(key, "__", "_")
	}
	return key
}

func parseEnvelopeInt(v string, allowUnbounded bool) (int, error) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "inf", "infinite", "unbounded":
		if allowUnbounded {
			return Unbounded, nil
		}
		return 0, fmt.Errorf("unbounded is not valid for this axis")
	case "off", "none":
		return 0, nil
	}
	n64, err := strconv.ParseInt(strings.ReplaceAll(v, "_", ""), 10, 0)
	if err != nil {
		return 0, fmt.Errorf("want integer: %w", err)
	}
	if !allowUnbounded && n64 < 0 {
		return 0, fmt.Errorf("must be non-negative")
	}
	return int(n64), nil
}

func parseEnvelopeDuration(v string) (time.Duration, error) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "off", "none", "unbounded", "inf", "infinite":
		return 0, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("want duration with unit, e.g. 30m or 2h: %w", err)
	}
	if d < 0 {
		return 0, fmt.Errorf("must be non-negative")
	}
	return d, nil
}

func parseEnvelopeGapMillis(key, v string) (int, error) {
	if key == "gap_ms" {
		return parseEnvelopeInt(v, false)
	}
	d, err := parseEnvelopeDuration(v)
	if err != nil {
		return 0, err
	}
	return int(d / time.Millisecond), nil
}

func parseEnvelopeSpend(v string) (SpendEnvelope, error) {
	raw := strings.TrimSpace(v)
	raw = strings.TrimPrefix(raw, "$")
	currency := "USD"
	if strings.HasSuffix(strings.ToUpper(raw), "USD") {
		raw = strings.TrimSpace(raw[:len(raw)-3])
		currency = "USD"
	}
	raw = strings.ReplaceAll(raw, "_", "")
	if raw == "" {
		return SpendEnvelope{}, fmt.Errorf("want amount")
	}
	whole, frac, _ := strings.Cut(raw, ".")
	if strings.HasPrefix(whole, "-") {
		return SpendEnvelope{}, fmt.Errorf("must be non-negative")
	}
	dollars, err := strconv.ParseInt(whole, 10, 64)
	if err != nil {
		return SpendEnvelope{}, fmt.Errorf("want decimal amount: %w", err)
	}
	if len(frac) > 2 {
		return SpendEnvelope{}, fmt.Errorf("use at most two decimal places")
	}
	for len(frac) < 2 {
		frac += "0"
	}
	cents := int64(0)
	if frac != "" {
		cents, err = strconv.ParseInt(frac, 10, 64)
		if err != nil {
			return SpendEnvelope{}, fmt.Errorf("want decimal cents: %w", err)
		}
	}
	return SpendEnvelope{MaxCents: dollars*100 + cents, Currency: currency}, nil
}

func parseEnvelopeRate(v string) (float64, error) {
	raw := strings.ToLower(strings.TrimSpace(v))
	for _, suffix := range []string{"tokens/sec", "token/sec", "tokens/s", "token/s", "tok/s", "t/s", "/sec", "/s", "tps"} {
		raw = strings.TrimSpace(strings.TrimSuffix(raw, suffix))
	}
	r, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("want tokens/sec rate: %w", err)
	}
	if r < 0 || math.IsNaN(r) || math.IsInf(r, 0) {
		return 0, fmt.Errorf("must be a finite non-negative rate")
	}
	return r, nil
}
