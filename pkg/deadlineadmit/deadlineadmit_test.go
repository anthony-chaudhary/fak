package deadlineadmit

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newClock() *fakeClock { return &fakeClock{t: time.Unix(1_800_000_000, 0)} }

func TestBudgetPicksSmallestPositiveDeclaration(t *testing.T) {
	cases := []struct {
		name   string
		header map[string]string
		want   time.Duration
		ok     bool
	}{
		{"none", nil, 0, false},
		{"stainless seconds", map[string]string{HeaderStainlessTimeout: "300"}, 300 * time.Second, true},
		{"request timeout fractional", map[string]string{HeaderRequestTimeout: "1.5"}, 1500 * time.Millisecond, true},
		{"fak ms wins when smaller", map[string]string{HeaderStainlessTimeout: "300", HeaderFakDeadlineMs: "120000"}, 120 * time.Second, true},
		{"garbage ignored", map[string]string{HeaderStainlessTimeout: "soon", HeaderRequestTimeout: "10"}, 10 * time.Second, true},
		{"non-positive ignored", map[string]string{HeaderRequestTimeout: "0", HeaderFakDeadlineMs: "-5"}, 0, false},
		{"NaN ignored", map[string]string{HeaderRequestTimeout: "NaN"}, 0, false},
		{"huge capped", map[string]string{HeaderRequestTimeout: "1e30"}, maxBudget, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := http.Header{}
			for k, v := range tc.header {
				h.Set(k, v)
			}
			got, ok := Budget(h)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("Budget = (%v, %v), want (%v, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestSetRemainingRoundTripsAndNeverReadsAsUnset(t *testing.T) {
	h := http.Header{}
	SetRemaining(h, 42*time.Second)
	if got, ok := Budget(h); !ok || got != 42*time.Second {
		t.Fatalf("round trip = (%v, %v), want 42s", got, ok)
	}
	SetRemaining(h, -time.Second)
	got, ok := Budget(h)
	if !ok || got != time.Millisecond {
		t.Fatalf("spent budget = (%v, %v), want (1ms, true)", got, ok)
	}
}

// incidentEstimator reproduces the strix1 saturation: six requests in flight,
// a 1.6k-token prefill taking 87 s and a 315-token decode taking 163 s.
func incidentEstimator(clk *fakeClock) (*Estimator, []func()) {
	e := NewEstimator(Config{}, clk.now)
	var ends []func()
	for i := 0; i < 6; i++ {
		ends = append(ends, e.Begin())
	}
	e.Observe(Observation{PromptTokens: 1600, Prefill: 87 * time.Second, CompletionTokens: 315, Decode: 163 * time.Second})
	return e, ends
}

func TestAdmitRefusesWorkThatCannotFinishBeforeTheDeadline(t *testing.T) {
	clk := newClock()
	e, _ := incidentEstimator(clk)

	v := e.Admit(1600, 0, 300*time.Second, true)
	if v.Admit || v.Reason != ReasonDeadlineInfeasible || !v.Measured {
		t.Fatalf("verdict = %+v, want refused deadline_infeasible", v)
	}
	if !errors.Is(v.Err(), ErrDeadlineInfeasible) {
		t.Fatalf("Err() = %v, want wrapping ErrDeadlineInfeasible", v.Err())
	}
	// 250 s of work at 6-way contention, run 7-way: 250*7/6.
	want := 250 * time.Second * 7 / 6
	if d := v.Estimate - want; d < -time.Second || d > time.Second {
		t.Fatalf("estimate = %v, want ~%v", v.Estimate, want)
	}
	if v.RetryAfter < time.Second || v.RetryAfter > DefaultRetryAfterMax {
		t.Fatalf("RetryAfter = %v out of [1s, %v]", v.RetryAfter, DefaultRetryAfterMax)
	}
}

func TestAdmitAdmitsFeasibleAndUnmeasuredAndUndeclared(t *testing.T) {
	clk := newClock()
	e, ends := incidentEstimator(clk)
	for _, end := range ends {
		end()
	}
	// Idle node: never predicted faster than measured (scale floors at 1).
	if v := e.Admit(1600, 0, 300*time.Second, true); !v.Admit {
		t.Fatalf("idle feasible verdict = %+v, want admit", v)
	}
	// A short max_tokens shrinks the decode term.
	if v := e.Admit(100, 10, 30*time.Second, true); !v.Admit {
		t.Fatalf("short request verdict = %+v, want admit", v)
	}
	if v := e.Admit(1_000_000, 0, time.Second, false); !v.Admit {
		t.Fatalf("no declared budget verdict = %+v, want admit", v)
	}
	fresh := NewEstimator(Config{}, clk.now)
	if v := fresh.Admit(1_000_000, 0, time.Second, true); !v.Admit || v.Measured {
		t.Fatalf("unmeasured verdict = %+v, want admit unmeasured", v)
	}
}

func TestAdmitRefusesSpentBudget(t *testing.T) {
	e := NewEstimator(Config{}, newClock().now)
	v := e.Admit(10, 10, 0, true)
	if v.Admit || v.Reason != ReasonDeadlineExpired || !errors.Is(v.Err(), ErrDeadlineInfeasible) {
		t.Fatalf("verdict = %+v, want refused deadline_expired", v)
	}
}

func TestObserveDecaysTowardRecentThroughput(t *testing.T) {
	clk := newClock()
	e := NewEstimator(Config{HalfLife: time.Minute}, clk.now)
	end := e.Begin()
	e.Observe(Observation{PromptTokens: 1000, Prefill: time.Second, CompletionTokens: 100, Decode: 10 * time.Second})
	end()
	fast, _ := e.Estimate(1000, 0)
	// Ten half-lives later the node got 10x slower; the average follows.
	clk.advance(10 * time.Minute)
	end = e.Begin()
	e.Observe(Observation{PromptTokens: 1000, Prefill: 10 * time.Second, CompletionTokens: 100, Decode: 100 * time.Second})
	end()
	slow, _ := e.Estimate(1000, 0)
	if slow < 9*fast {
		t.Fatalf("estimate after slowdown = %v, fast = %v; want ~10x", slow, fast)
	}
}

func TestBeginEndIsIdempotent(t *testing.T) {
	e := NewEstimator(Config{}, nil)
	end := e.Begin()
	end()
	end()
	if got := e.InFlight(); got != 0 {
		t.Fatalf("InFlight = %d, want 0", got)
	}
}

func TestWriteRefusalIsTyped503WithRetryAfter(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteRefusal(rec, Verdict{Reason: ReasonDeadlineInfeasible, Estimate: 292 * time.Second, RetryAfter: 1500 * time.Millisecond})
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "2" {
		t.Fatalf("Retry-After = %q, want 2 (ceil)", got)
	}
	if got := rec.Header().Get(HeaderReject); got != string(ReasonDeadlineInfeasible) {
		t.Fatalf("%s = %q", HeaderReject, got)
	}
	if got := rec.Header().Get(HeaderEstimateMs); got != strconv.Itoa(292000) {
		t.Fatalf("%s = %q", HeaderEstimateMs, got)
	}
}

func TestBufferedObservationMeasuresWithoutPrefillSplit(t *testing.T) {
	e := NewEstimator(Config{}, newClock().now)
	end := e.Begin()
	// A buffered reply: no time-to-first-token, 100 tokens over 50 s total.
	e.Observe(Observation{PromptTokens: 1000, CompletionTokens: 100, Decode: 50 * time.Second})
	end()
	got, ok := e.Estimate(1000, 0)
	if !ok || got != 50*time.Second {
		t.Fatalf("Estimate = (%v, %v), want (50s, true): decode rate carries prefill", got, ok)
	}
}

// haloPiEstimator primes the strix3 incident shape: one in-flight-free node
// measured at 86 prompt tokens/s prefill and 21 tokens/s decode.
func haloPiEstimator(t *testing.T) *Estimator {
	t.Helper()
	e := NewEstimator(Config{}, newClock().now)
	e.Observe(Observation{PromptTokens: 8600, Prefill: 100 * time.Second, CompletionTokens: 210, Decode: 10 * time.Second})
	return e
}

func TestAdmitCachedAdmitsWarmMultiTurnPromptWithinBudget(t *testing.T) {
	e := haloPiEstimator(t)
	v := e.AdmitCached(50_000, 48_000, 0, 600*time.Second, true)
	if err := v.Err(); err != nil || !v.Measured {
		t.Fatalf("50k prompt with 48k cached prefix: err=%v measured=%v estimate=%v; want measured admission", err, v.Measured, v.Estimate)
	}
}

func TestAdmitCachedRefusesSameColdPrompt(t *testing.T) {
	e := haloPiEstimator(t)
	for name, v := range map[string]Verdict{
		"cold cached=0": e.AdmitCached(50_000, 0, 0, 600*time.Second, true),
		"legacy Admit":  e.Admit(50_000, 0, 600*time.Second, true),
	} {
		if err := v.Err(); !errors.Is(err, ErrDeadlineInfeasible) || v.Reason != ReasonDeadlineInfeasible {
			t.Fatalf("%s: err=%v reason=%q; want ErrDeadlineInfeasible/%q", name, err, v.Reason, ReasonDeadlineInfeasible)
		}
	}
}

func TestEstimateCachedClampsCachedToPrompt(t *testing.T) {
	e := haloPiEstimator(t)
	over, _ := e.EstimateCached(1000, 5000, 0)
	neg, _ := e.EstimateCached(1000, -5, 0)
	full, _ := e.EstimateCached(0, 0, 0)
	cold, _ := e.Estimate(1000, 0)
	if over != full || neg != cold {
		t.Fatalf("over-cached=%v want %v (no prefill); negative-cached=%v want cold %v", over, full, neg, cold)
	}
}

func TestObserveWithCacheMeasuresUncachedPrefillRate(t *testing.T) {
	warm := NewEstimator(Config{}, newClock().now)
	// 50k prompt, 47.85k served from the KV prefix: 2150 prefilled in 25 s = 86 t/s.
	warm.Observe(Observation{PromptTokens: 50_000, CachedTokens: 47_850, Prefill: 25 * time.Second, CompletionTokens: 210, Decode: 10 * time.Second})
	cold := haloPiEstimator(t)
	got, ok1 := warm.Estimate(8600, 0)
	want, ok2 := cold.Estimate(8600, 0)
	if !ok1 || !ok2 || got != want {
		t.Fatalf("estimate after cached observation = (%v,%v), want the 86 t/s cold-measured (%v,%v)", got, ok1, want, ok2)
	}
}

func TestObserveSkipsPrefillSampleForNearFullyCachedTurn(t *testing.T) {
	e := haloPiEstimator(t)
	before, _ := e.Estimate(8600, 0)
	// 4 uncached tokens in 2 s is per-request overhead, not a 2 t/s prefill rate.
	e.Observe(Observation{PromptTokens: 50_000, CachedTokens: 49_996, Prefill: 2 * time.Second, CompletionTokens: 210, Decode: 10 * time.Second})
	after, _ := e.Estimate(8600, 0)
	if after != before {
		t.Fatalf("estimate moved %v -> %v on a near-fully-cached observation; want the prefill rate untouched", before, after)
	}
}
