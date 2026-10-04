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
