package deadlineadmit

import (
	"encoding/json"
	"math"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestDeadlineRefusalCarriesEstimatorEvidence(t *testing.T) {
	now := time.Unix(1800000000, 0)
	e := NewEstimator(Config{}, func() time.Time { return now })
	e.Observe(Observation{PromptTokens: 100, Prefill: time.Second, CompletionTokens: 100, Decode: time.Second})
	release := e.Begin()
	defer release()
	v := e.AdmitCached(100, 40, 10, time.Millisecond, true)
	w := httptest.NewRecorder()
	WriteRefusal(w, v)
	var env map[string]map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if len(env["error"]["evidence"]) == 0 {
		t.Fatalf("deadline refusal omits the estimator snapshot: %s", w.Body.String())
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestDeadlineEvidenceSnapshotAndRateAge(t *testing.T) {
	now := time.Unix(1800000000, 0)
	e := NewEstimator(Config{HalfLife: time.Minute}, func() time.Time { return now })
	v := e.AdmitCachedBudget(100, 40, 10, time.Second, time.Millisecond, true)
	if !v.Admit || v.Measured || v.Evidence == nil || v.Evidence.RateBasis != RateBasis("unknown") || v.Evidence.RateSampleAgeMS != nil {
		t.Fatalf("unmeasured evidence fabricated rates: %+v", v)
	}
	e.Observe(Observation{PromptTokens: 100, Prefill: time.Second, CompletionTokens: 100, Decode: time.Second})
	now = now.Add(2 * time.Minute)
	// A fresh decode observation cannot make an old, still-used prefill rate fresh.
	e.Observe(Observation{CompletionTokens: 100, Decode: time.Second})
	release := e.Begin()
	v = e.AdmitCachedBudget(100, 40, 10, time.Second, time.Millisecond, true)
	release()
	x := v.Evidence
	if v.Admit || !v.Measured || x == nil {
		t.Fatalf("expected measured refusal: %+v", v)
	}
	if x.DeclaredBudgetMS == nil || *x.DeclaredBudgetMS != 1000 || x.RemainingBudgetMS != 1 || x.EstimateMS != v.Estimate.Milliseconds() || x.PromptTokens != 100 || x.CachedTokens != 40 || x.OutputCeiling != 10 || x.InFlight != 1 {
		t.Fatalf("snapshot does not describe verdict inputs: %+v", x)
	}
	if x.RateBasis != RateBasis("stale_prefill_decode") || x.RateSampleAgeMS == nil || *x.RateSampleAgeMS != 120000 {
		t.Fatalf("fresh decode hid the old prefill rate: %+v", x)
	}
	fullCached := e.AdmitCachedBudget(100, 100, 10, time.Second, time.Millisecond, true).Evidence
	if fullCached == nil || fullCached.RateBasis != RateBasis("decode_only") || fullCached.RateSampleAgeMS == nil || *fullCached.RateSampleAgeMS != 0 {
		t.Fatalf("unused old prefill tainted fully cached estimate provenance: %+v", fullCached)
	}
	if x.InFlight != 1 || e.InFlight() != 0 {
		t.Fatal("evidence was changed by later reservation release")
	}
	legacy := e.AdmitCached(100, 40, 10, time.Millisecond, true)
	if legacy.Evidence == nil || legacy.Evidence.DeclaredBudgetMS != nil {
		t.Fatalf("legacy caller fabricated declared budget: %+v", legacy.Evidence)
	}
	decode := NewEstimator(Config{HalfLife: time.Minute}, func() time.Time { return now })
	decode.Observe(Observation{CompletionTokens: 100, Decode: time.Second})
	for _, step := range []struct {
		advance time.Duration
		basis   string
		age     int64
	}{{0, "decode_only", 0}, {time.Minute + time.Millisecond, "stale_decode_only", 60001}} {
		now = now.Add(step.advance)
		x := decode.AdmitCachedBudget(100, 0, 10, time.Second, time.Millisecond, true).Evidence
		if x == nil || x.RateBasis != RateBasis(step.basis) || x.RateSampleAgeMS == nil || *x.RateSampleAgeMS != step.age {
			t.Fatalf("decode rate provenance: %+v", x)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeadlineEvidenceBoundsAndExpired(t *testing.T) {
	e := NewEstimator(Config{}, nil)
	x := e.AdmitCachedBudget(-1, math.MaxInt, -1, -time.Second, -time.Second, true).Evidence
	if x == nil || x.PromptTokens != 0 || x.CachedTokens != 0 || x.RemainingBudgetMS != 0 || x.DeclaredBudgetMS == nil || *x.DeclaredBudgetMS != 0 || x.OutputCeiling < 0 {
		t.Fatalf("negative inputs escaped normalization: %+v", x)
	}
	x = e.AdmitCachedBudget(math.MaxInt, math.MaxInt, math.MaxInt, time.Duration(math.MaxInt64), time.Duration(math.MaxInt64), true).Evidence
	if x == nil || x.PromptTokens != math.MaxInt32 || x.CachedTokens != math.MaxInt32 || x.OutputCeiling > math.MaxInt32 || x.DeclaredBudgetMS == nil || *x.DeclaredBudgetMS != 86400000 || x.RemainingBudgetMS != 86400000 {
		t.Fatalf("overflow inputs escaped bounds: %+v", x)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestDeadlineReadEvidenceBoundedCompatibility(t *testing.T) {
	e := NewEstimator(Config{}, nil)
	e.Observe(Observation{PromptTokens: 100, Prefill: time.Second, CompletionTokens: 100, Decode: time.Second})
	v := e.AdmitCachedBudget(100, 40, 10, time.Second, time.Millisecond, true)
	w := httptest.NewRecorder()
	WriteRefusal(w, v)
	x, err := ReadEvidence(strings.NewReader(w.Body.String()))
	if err != nil || x == nil || x.PromptTokens != 100 || x.CachedTokens != 40 || x.RemainingBudgetMS != 1 {
		t.Fatalf("writer/reader round trip: %+v, %v", x, err)
	}
	legacy := httptest.NewRecorder()
	WriteRefusal(legacy, Verdict{Reason: ReasonDeadlineExpired, RetryAfter: time.Second})
	if strings.Contains(legacy.Body.String(), "evidence") {
		t.Fatal("manual legacy verdict changed shape")
	}
	for _, body := range []string{legacy.Body.String(), `{"error":{"code":"deadline_infeasible","future_metadata":true}}`} {
		if x, err := ReadEvidence(strings.NewReader(body)); err != nil || x != nil {
			t.Fatalf("legacy/forward envelope rejected: %+v %v", x, err)
		}
	}
	valid := w.Body.String()
	for name, body := range map[string]string{
		"unknown_field":     strings.Replace(valid, `"prompt_tokens":`, `"host_path":"secret","prompt_tokens":`, 1),
		"unknown_basis":     strings.Replace(valid, `"prefill_decode"`, `"invented"`, 1),
		"negative":          strings.Replace(valid, `"prompt_tokens":100`, `"prompt_tokens":-1`, 1),
		"overflow":          strings.Replace(valid, `"prompt_tokens":100`, `"prompt_tokens":2147483648`, 1),
		"duration_overflow": strings.Replace(valid, `"remaining_budget_ms":1`, `"remaining_budget_ms":86400001`, 1),
		"trailing":          valid + `{}`,
		"oversize":          strings.Repeat(" ", MaxRefusalBytes+1) + valid,
	} {
		t.Run(name, func(t *testing.T) {
			if x, err := ReadEvidence(strings.NewReader(body)); err == nil {
				t.Fatalf("invalid evidence accepted: %+v body=%s", x, body)
			}
		})
	}
}
