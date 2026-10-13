package deadlineadmit

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"time"
)

// MaxRefusalBytes bounds the refusal envelope accepted by ReadEvidence.
const MaxRefusalBytes = 2048

// RateBasis names the timing inputs used by the existing estimator. Stale
// means the oldest contributing rate is older than its configured half-life;
// it describes evidence currency without changing admission policy.
type RateBasis string

const (
	RateUnknown            RateBasis = "unknown"
	RateDecodeOnly         RateBasis = "decode_only"
	RatePrefillDecode      RateBasis = "prefill_decode"
	RateStaleDecodeOnly    RateBasis = "stale_decode_only"
	RateStalePrefillDecode RateBasis = "stale_prefill_decode"
)

// Evidence is a bounded projection of the exact admission snapshot. Durations
// saturate at 24 hours and counts at MaxInt32; negative inputs become zero.
// CachedTokens is clamped to PromptTokens before projection. OutputCeiling is
// the requested max_tokens (zero means unset), not the predicted decode length.
// Nil budgets or ages explicitly mean unknown, including legacy verdicts.
type Evidence struct {
	DeclaredBudgetMS  *int64    `json:"declared_budget_ms"`
	RemainingBudgetMS int64     `json:"remaining_budget_ms"`
	EstimateMS        int64     `json:"estimate_ms"`
	PromptTokens      int64     `json:"prompt_tokens"`
	CachedTokens      int64     `json:"cached_tokens"`
	OutputCeiling     int64     `json:"output_ceiling"`
	InFlight          int64     `json:"in_flight"`
	RateSampleAgeMS   *int64    `json:"rate_sample_age_ms"`
	RateBasis         RateBasis `json:"rate_basis"`
}

func boundedMS(d time.Duration) int64 {
	return min(max(d, 0), maxBudget).Milliseconds()
}

func boundedCount(n int) int64 { return min(max(int64(n), 0), math.MaxInt32) }

func (e *Estimator) evidenceLocked(prompt, cached, output int, remaining time.Duration) *Evidence {
	prompt = max(prompt, 0)
	cached = min(max(cached, 0), prompt)
	v := &Evidence{
		RemainingBudgetMS: boundedMS(remaining), PromptTokens: boundedCount(prompt),
		CachedTokens: boundedCount(cached), OutputCeiling: boundedCount(output),
		InFlight: boundedCount(e.inFlight), RateBasis: RateUnknown,
	}
	if !e.hasDec || e.decode <= 0 || e.samples < minRateSamples {
		return v
	}
	oldest := e.decodeLast
	v.RateBasis = RateDecodeOnly
	if prompt > cached && e.hasPre && e.prefill > 0 {
		v.RateBasis = RatePrefillDecode
		if e.prefillLast.Before(oldest) {
			oldest = e.prefillLast
		}
	}
	age := max(e.now().Sub(oldest), 0)
	ms := boundedMS(age)
	v.RateSampleAgeMS = &ms
	if age > e.cfg.HalfLife {
		if v.RateBasis == RateDecodeOnly {
			v.RateBasis = RateStaleDecodeOnly
		} else {
			v.RateBasis = RateStalePrefillDecode
		}
	}
	return v
}

func (e Evidence) normalized() Evidence {
	capMS := maxBudget.Milliseconds()
	if e.DeclaredBudgetMS != nil {
		n := min(max(*e.DeclaredBudgetMS, 0), capMS)
		e.DeclaredBudgetMS = &n
	}
	e.RemainingBudgetMS = min(max(e.RemainingBudgetMS, 0), capMS)
	e.EstimateMS = min(max(e.EstimateMS, 0), capMS)
	e.PromptTokens = min(max(e.PromptTokens, 0), math.MaxInt32)
	e.CachedTokens = min(max(e.CachedTokens, 0), e.PromptTokens)
	e.OutputCeiling = min(max(e.OutputCeiling, 0), math.MaxInt32)
	e.InFlight = min(max(e.InFlight, 0), math.MaxInt32)
	switch e.RateBasis {
	case RateDecodeOnly, RatePrefillDecode, RateStaleDecodeOnly, RateStalePrefillDecode:
		if e.RateSampleAgeMS == nil {
			e.RateBasis = RateUnknown
		}
	default:
		e.RateBasis = RateUnknown
	}
	if e.RateBasis == RateUnknown {
		e.RateSampleAgeMS = nil
	} else {
		n := min(max(*e.RateSampleAgeMS, 0), capMS)
		e.RateSampleAgeMS = &n
	}
	return e
}

// ReadEvidence reads optional admission evidence from a refusal envelope.
// Legacy envelopes return nil without inventing measured inputs. Other error
// metadata is ignored; evidence itself is strictly allowlisted and bounded.
// Invalid evidence, oversized bodies, and trailing JSON return an error.
func ReadEvidence(r io.Reader) (*Evidence, error) {
	b, err := io.ReadAll(io.LimitReader(r, MaxRefusalBytes+1))
	if err != nil {
		return nil, err
	}
	if len(b) > MaxRefusalBytes {
		return nil, errors.New("deadline refusal exceeds size bound")
	}
	var envelope struct {
		Error struct {
			Code     string          `json:"code"`
			Evidence json.RawMessage `json:"evidence"`
		} `json:"error"`
	}
	if err := json.Unmarshal(b, &envelope); err != nil {
		return nil, err
	}
	if envelope.Error.Code != CodeDeadlineInfeasible {
		return nil, errors.New("not a deadline refusal")
	}
	raw := envelope.Error.Evidence
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var e Evidence
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&e); err != nil {
		return nil, err
	}
	// Require every field, so a partial projection cannot masquerade as an
	// observed zero. Nullable fields must still explicitly appear on the wire.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || len(fields) != 9 {
		return nil, errors.New("incomplete deadline evidence")
	}
	for name, value := range fields {
		if name != "declared_budget_ms" && name != "rate_sample_age_ms" && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return nil, errors.New("null deadline evidence input")
		}
	}
	n := e.normalized()
	if e.RemainingBudgetMS != n.RemainingBudgetMS || e.EstimateMS != n.EstimateMS ||
		e.PromptTokens != n.PromptTokens || e.CachedTokens != n.CachedTokens ||
		e.OutputCeiling != n.OutputCeiling || e.InFlight != n.InFlight || e.RateBasis != n.RateBasis ||
		!equalOptional(e.DeclaredBudgetMS, n.DeclaredBudgetMS) || !equalOptional(e.RateSampleAgeMS, n.RateSampleAgeMS) {
		return nil, errors.New("invalid deadline evidence input")
	}
	return &e, nil
}

func equalOptional(a, b *int64) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}
