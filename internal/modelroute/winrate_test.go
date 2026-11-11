package modelroute

import (
	"errors"
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Win-rate-threshold routing contract (borrowed from lm-sys/RouteLLM @
// 0b64fdafe049e596a3f5657c219329f24af24198, routellm/routers/routers.py:32-48
// and routellm/controller.py:93-103; ADAPT, Apache-2.0): one
// CalculateStrongWinRate per scorer, strong iff win rate >= threshold, and the
// "router-<name>-<threshold>" model-id grammar. These tests pin the CLOSED
// contract — values and sentinel errors via errors.Is, never error prose.
// ---------------------------------------------------------------------------

// winRateStub is a deterministic WinRateScorer stub for dispatch tests.
type winRateStub struct {
	rate float64
	err  error
}

func (f winRateStub) CalculateStrongWinRate(Subject) (float64, error) { return f.rate, f.err }

func TestWinRateThresholdRouting(t *testing.T) {
	t.Run("ModelIDGrammar", func(t *testing.T) {
		cases := []struct {
			id      string
			name    string
			thr     float64
			wantErr error
		}{
			{"router-random-0.5", "random", 0.5, nil},
			{"router-mf-0", "mf", 0, nil},
			{"router-causal_llm-1", "causal_llm", 1, nil},
			{"router-bert-.5", "bert", 0.5, nil},
			// The study note's documented calibration example (50% strong calls
			// for the mf router at the pinned revision).
			{"router-mf-0.11593", "mf", 0.11593, nil},
			{"router-sw_ranking-0.30", "sw_ranking", 0.30, nil},
			{"router-mf", "", 0, ErrInvalidRouterModel},       // no threshold token
			{"router--0.5", "", 0, ErrInvalidRouterModel},     // empty name
			{"router-mf-0.5-x", "", 0, ErrInvalidRouterModel}, // extra hyphen segment lands in the token
			{"plain-model", "", 0, ErrInvalidRouterModel},     // missing prefix
			{"Router-mf-0.5", "", 0, ErrInvalidRouterModel},   // prefix is a literal, not a pattern
			{"router-mf-x", "", 0, ErrInvalidRouterModel},     // non-numeric threshold
			{"router-mf-nan", "", 0, ErrInvalidRouterModel},   // special forms are not grammar tokens
			{"router-mf-inf", "", 0, ErrInvalidRouterModel},
			{"router-mf-+0.5", "", 0, ErrInvalidRouterModel}, // signs are not grammar tokens
			{"router-mf-1e-3", "", 0, ErrInvalidRouterModel}, // exponent form is not a plain decimal
			{"router-mf-1.", "", 0, ErrInvalidRouterModel},   // trailing dot
			{"router-mf-1.5", "", 0, ErrInvalidThreshold},    // grammatical but outside [0,1]
			{"router-mf-2", "", 0, ErrInvalidThreshold},
		}
		for _, c := range cases {
			name, thr, err := ParseRouterModel(c.id)
			if c.wantErr != nil {
				if !errors.Is(err, c.wantErr) {
					t.Fatalf("ParseRouterModel(%q) = %v, want errors.Is %v", c.id, err, c.wantErr)
				}
				continue
			}
			if err != nil {
				t.Fatalf("ParseRouterModel(%q) unexpected error: %v", c.id, err)
			}
			if name != c.name || thr != c.thr {
				t.Fatalf("ParseRouterModel(%q) = (%q, %v), want (%q, %v)", c.id, name, thr, c.name, c.thr)
			}
		}
	})

	t.Run("DispatchBoundary", func(t *testing.T) {
		pair := ModelPair{Strong: "gpt-strong", Weak: "mixtral-weak"}
		subj := Subject{Aspect: AspectRequest, PromptTokens: 128}

		// Inclusive boundary: win rate exactly at the cutoff takes the strong tier.
		r := ThresholdRouter{Name: "fixed", Scorer: winRateStub{rate: 0.5}, Pair: pair, Threshold: 0.5}
		got, err := r.Route(subj)
		if err != nil || got != pair.Strong {
			t.Fatalf("at-cutoff: got (%q, %v), want strong %q", got, err, pair.Strong)
		}

		// One step below the cutoff takes the weak tier.
		r.Scorer = winRateStub{rate: 0.499999}
		got, err = r.Route(subj)
		if err != nil || got != pair.Weak {
			t.Fatalf("below-cutoff: got (%q, %v), want weak %q", got, err, pair.Weak)
		}

		// One step above takes the strong tier.
		r.Scorer = winRateStub{rate: 0.500001}
		got, err = r.Route(subj)
		if err != nil || got != pair.Strong {
			t.Fatalf("above-cutoff: got (%q, %v), want strong %q", got, err, pair.Strong)
		}

		// Cutoff 0: every valid probability meets it, even exactly 0.
		r.Threshold = 0
		r.Scorer = winRateStub{rate: 0}
		got, err = r.Route(subj)
		if err != nil || got != pair.Strong {
			t.Fatalf("zero-cutoff: got (%q, %v), want strong %q", got, err, pair.Strong)
		}

		// Cutoff 1: only a perfect win rate meets it.
		r.Threshold = 1
		r.Scorer = winRateStub{rate: 0.999999}
		got, err = r.Route(subj)
		if err != nil || got != pair.Weak {
			t.Fatalf("unit-cutoff near miss: got (%q, %v), want weak %q", got, err, pair.Weak)
		}
		r.Scorer = winRateStub{rate: 1}
		got, err = r.Route(subj)
		if err != nil || got != pair.Strong {
			t.Fatalf("unit-cutoff exact: got (%q, %v), want strong %q", got, err, pair.Strong)
		}
	})

	t.Run("ClosedContractFailures", func(t *testing.T) {
		pair := ModelPair{Strong: "s", Weak: "w"}
		subj := Subject{}

		badScores := map[string]float64{
			"above-one": 1.5,
			"negative":  -0.1,
			"nan":       math.NaN(),
			"+inf":      math.Inf(1),
			"-inf":      math.Inf(-1),
		}
		for label, rate := range badScores {
			r := ThresholdRouter{Name: "bad", Scorer: winRateStub{rate: rate}, Pair: pair, Threshold: 0.5}
			if _, err := r.Route(subj); !errors.Is(err, ErrInvalidWinRate) {
				t.Fatalf("%s score: got %v, want errors.Is ErrInvalidWinRate", label, err)
			}
		}

		sameTier := ModelPair{Strong: "same", Weak: "same"}
		r := ThresholdRouter{Name: "bad", Scorer: winRateStub{rate: 0.5}, Pair: sameTier, Threshold: 0.5}
		if _, err := r.Route(subj); !errors.Is(err, ErrInvalidModelPair) {
			t.Fatalf("identical tiers: got %v, want errors.Is ErrInvalidModelPair", err)
		}
		empty := ModelPair{Strong: "s", Weak: ""}
		r = ThresholdRouter{Name: "bad", Scorer: winRateStub{rate: 0.5}, Pair: empty, Threshold: 0.5}
		if _, err := r.Route(subj); !errors.Is(err, ErrInvalidModelPair) {
			t.Fatalf("empty tier: got %v, want errors.Is ErrInvalidModelPair", err)
		}

		for label, thr := range map[string]float64{"above": 1.2, "below": -0.2, "nan": math.NaN()} {
			r = ThresholdRouter{Name: "bad", Scorer: winRateStub{rate: 0.5}, Pair: pair, Threshold: thr}
			if _, err := r.Route(subj); !errors.Is(err, ErrInvalidThreshold) {
				t.Fatalf("%s threshold: got %v, want errors.Is ErrInvalidThreshold", label, err)
			}
		}

		unwired := ThresholdRouter{Name: "bad", Pair: pair, Threshold: 0.5}
		if _, err := unwired.Route(subj); !errors.Is(err, ErrInvalidModelPair) {
			t.Fatalf("unwired scorer: got %v, want errors.Is ErrInvalidModelPair", err)
		}

		// A scorer transport error propagates untouched — dispatch does not mask it.
		boom := errors.New("scorer backend down")
		r = ThresholdRouter{Name: "bad", Scorer: winRateStub{err: boom}, Pair: pair, Threshold: 0.5}
		if _, err := r.Route(subj); !errors.Is(err, boom) {
			t.Fatalf("scorer error: got %v, want the scorer's own error", err)
		}
	})

	t.Run("CatalogResolve", func(t *testing.T) {
		pair := ModelPair{Strong: "gpt-strong", Weak: "mixtral-weak"}
		cat := NewRouterCatalog(map[string]WinRateScorer{
			"mf":     winRateStub{rate: 0.9},
			"random": winRateStub{rate: 0.1},
			"":       nil, // nil entries are dropped, never registered
		})

		r, err := cat.Resolve("router-mf-0.5", pair)
		if err != nil {
			t.Fatalf("resolve mf: %v", err)
		}
		if r.Name != "mf" || r.Threshold != 0.5 || r.Pair != pair {
			t.Fatalf("resolved router %+v carries wrong identity", r)
		}
		got, err := r.Route(Subject{Aspect: AspectRequest})
		if err != nil || got != pair.Strong { // 0.9 >= 0.5
			t.Fatalf("resolved dispatch: got (%q, %v), want strong", got, err)
		}

		r, err = cat.Resolve("router-random-0.5", pair)
		if err != nil {
			t.Fatalf("resolve random: %v", err)
		}
		got, err = r.Route(Subject{Aspect: AspectRequest})
		if err != nil || got != pair.Weak { // 0.1 < 0.5
			t.Fatalf("resolved weak dispatch: got (%q, %v), want weak", got, err)
		}

		if _, err = cat.Resolve("router-nope-0.5", pair); !errors.Is(err, ErrUnknownRouter) {
			t.Fatalf("unknown router: got %v, want errors.Is ErrUnknownRouter", err)
		}
		if _, err = cat.Resolve("router-mf-0.5-x", pair); !errors.Is(err, ErrInvalidRouterModel) {
			t.Fatalf("malformed id: got %v, want errors.Is ErrInvalidRouterModel", err)
		}
		if _, err = cat.Resolve("router-mf-0.5", ModelPair{Strong: "a", Weak: "a"}); !errors.Is(err, ErrInvalidModelPair) {
			t.Fatalf("bad pair through resolve: got %v, want errors.Is ErrInvalidModelPair", err)
		}
	})
}
