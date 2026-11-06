package gateway

// decisions_test.go — the POST /v1/decisions closed contract (ticket
// oss-port-tensorfold-decision-probabilities-20261103, adapted from
// ashhart/TensorFold zig/src/server/decisions.zig @9e8c368b0b4c2c7892a0c9f0f302d7c164016209,
// Apache-2.0): a valid choice answers object="decisions" with A-Z labels whose
// probabilities sum to 1 plus the per-label mass; a missing hook, a repeated or
// empty id, a non-positive temperature, an unsupported prompt format, zero or
// >26 options, and a non-finite logit each refuse with their typed code. The
// assertions bind on statuses, error-envelope codes, response values, and the
// sentinel errors (errors.Is) — never on message prose.

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

// decisionsErrorEnv is the OpenAI-style error envelope the refusals carry.
// Code decodes to "" when the envelope has code:null.
type decisionsErrorEnv struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func postDecisions(body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/decisions", strings.NewReader(body))
	rr := httptest.NewRecorder()
	(&Server{}).handleDecisions(rr, req)
	return rr
}

// Two package-level drift gates classify every routeTable() pattern from
// sibling test files: openapi_spec_test.go requires a specPathFor entry, and
// ep_fanout_coverage_test.go requires an EP-fanout classification (probe or
// named exemption). This ticket's scope fence pins the diff to decisions.go /
// decisions_test.go / http.go / docs/fak/openapi.yaml, so both registrations
// live here, in the feature's own test file, instead of extending those
// tables: each gate still enforces its invariant on every run.
func init() {
	specPathFor["/v1/decisions"] = "/v1/decisions"
	// Route-specific reason (none of the shared exemption constants applies):
	// the handler validates the closed option-set contract and maps the
	// host-installed engine hook's logits through a stable softmax; it never
	// calls a planner or enters a model decode, so there is no collective for
	// a follower rank to join.
	epFanoutExemptRoutes["/v1/decisions"] = "thin scoring surface: validates the option set and maps host-hook logits through a stable softmax without calling a planner or entering a model decode, so there is no collective for a follower rank to join"
}

// installScorer installs fn for the duration of the test and detaches it
// afterward, so no decisions hook leaks into sibling tests.
func installScorer(t *testing.T, fn DecisionsScorerFunc) {
	t.Helper()
	InstallDecisionsScorer(fn)
	t.Cleanup(func() { InstallDecisionsScorer(nil) })
}

// assertDecisionsRefusal checks the closed refusal triple: status, typed code,
// and error type derived from the status class.
func assertDecisionsRefusal(t *testing.T, rr *httptest.ResponseRecorder, wantStatus int, wantCode string) {
	t.Helper()
	if rr.Code != wantStatus {
		t.Fatalf("answered %d, want %d; body=%s", rr.Code, wantStatus, rr.Body.String())
	}
	var env decisionsErrorEnv
	if err := json.Unmarshal(rr.Body.Bytes(), &env); err != nil {
		t.Fatalf("refusal body is not the error envelope: %v; body=%s", err, rr.Body.String())
	}
	if env.Error.Code != wantCode {
		t.Fatalf("error code = %q, want %q", env.Error.Code, wantCode)
	}
}

func TestDecisionsEndpointScoresAndRefusals(t *testing.T) {
	t.Run("RouteIsRegistered", func(t *testing.T) {
		var found bool
		for _, rt := range (&Server{}).routeTable() {
			if rt.pattern == "/v1/decisions" {
				found = true
				// The registered handler must be the decisions surface: a GET
				// through it answers 405 (requireMethod), and with no hook
				// installed a POST answers the inert-route refusal.
				InstallDecisionsScorer(nil)
				rr := httptest.NewRecorder()
				rt.handler(rr, httptest.NewRequest(http.MethodGet, "/v1/decisions", nil))
				if rr.Code != http.StatusMethodNotAllowed {
					t.Fatalf("GET /v1/decisions answered %d, want 405", rr.Code)
				}
				rr = httptest.NewRecorder()
				rt.handler(rr, httptest.NewRequest(http.MethodPost, "/v1/decisions",
					strings.NewReader(`{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[{"id":"a"}]}`)))
				assertDecisionsRefusal(t, rr, http.StatusNotFound, "decisions_hook_unwired")
				return
			}
		}
		if !found {
			t.Fatal("routeTable() does not register /v1/decisions")
		}
	})

	t.Run("ValidChoiceScoresToDecisionsObject", func(t *testing.T) {
		installScorer(t, func(ctx context.Context, in DecisionsScoreInput) ([]float64, error) {
			if in.Prompt != "ship?" || len(in.OptionIDs) != 3 || in.OptionIDs[0] != "ship" {
				t.Errorf("hook input = %+v, want the validated request context in order", in)
			}
			return []float64{0, 0, 0}, nil // uniform logits: every option gets 1/3
		})
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"ship?",` +
			`"options":[{"id":"ship","text":"Ship now"},{"id":"hold"},{"id":"abort"}]}`)
		if rr.Code != http.StatusOK {
			t.Fatalf("valid choice answered %d, want 200; body=%s", rr.Code, rr.Body.String())
		}
		var res DecisionsResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &res); err != nil {
			t.Fatalf("200 body is not JSON: %v; body=%s", err, rr.Body.String())
		}
		if res.Object != "decisions" {
			t.Fatalf("object = %q, want \"decisions\"", res.Object)
		}
		if res.PromptFormatVersion != DecisionsPromptFormatVersion || res.Temperature != 1 {
			t.Fatalf("echoed identity = (%d, %v), want (1, 1)", res.PromptFormatVersion, res.Temperature)
		}
		if len(res.Results) != 3 {
			t.Fatalf("results = %d rows, want 3 in request order", len(res.Results))
		}
		sum := 0.0
		for i, want := range []struct {
			label, id string
		}{{"A", "ship"}, {"B", "hold"}, {"C", "abort"}} {
			got := res.Results[i]
			if got.Label != want.label || got.ID != want.id {
				t.Fatalf("result[%d] = (%s, %s), want (%s, %s)", i, got.Label, got.ID, want.label, want.id)
			}
			if math.Abs(got.Probability-1.0/3.0) > 1e-12 {
				t.Fatalf("probability[%d] = %v, want 1/3 for uniform logits", i, got.Probability)
			}
			if math.Abs(res.LabelMass[want.label]-1.0/3.0) > 1e-12 {
				t.Fatalf("label_mass[%q] = %v, want 1/3", want.label, res.LabelMass[want.label])
			}
			sum += got.Probability
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Fatalf("probabilities sum to %v, want ~1", sum)
		}
	})

	t.Run("TemperatureSharpensTheDistribution", func(t *testing.T) {
		logits := []float64{0, 1}
		score := func(temp float64) []float64 {
			probs, err := softmaxDecisionProbs(logits, temp)
			if err != nil {
				t.Fatalf("softmax at temperature %v: %v", temp, err)
			}
			return probs
		}
		atOne := score(1)
		atHalf := score(0.5)
		if atHalf[1] <= atOne[1] {
			t.Fatalf("lower temperature did not sharpen: p(option B) at T=0.5 is %v, at T=1 is %v", atHalf[1], atOne[1])
		}
		for name, p := range map[string][]float64{"T=1": atOne, "T=0.5": atHalf} {
			sum := p[0] + p[1]
			if math.Abs(sum-1.0) > 1e-9 {
				t.Fatalf("%s probabilities sum to %v, want ~1", name, sum)
			}
		}
	})

	t.Run("AbsentHookRefuses404", func(t *testing.T) {
		InstallDecisionsScorer(nil)
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"ship?","options":[{"id":"a"}]}`)
		assertDecisionsRefusal(t, rr, http.StatusNotFound, "decisions_hook_unwired")
	})

	t.Run("RepeatedOptionIDRefuses400", func(t *testing.T) {
		called := false
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) {
			called = true
			return nil, nil
		})
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"p",` +
			`"options":[{"id":"x"},{"id":"y"},{"id":"x"}]}`)
		assertDecisionsRefusal(t, rr, http.StatusBadRequest, "decisions_duplicate_option_id")
		if called {
			t.Fatal("engine hook ran on a refused request; validation must precede scoring")
		}
	})

	t.Run("NonPositiveTemperatureRefuses400", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) { return []float64{0}, nil })
		// Finite-but-invalid temperatures travel the wire as ordinary JSON
		// numbers; each must refuse with the typed code.
		for _, temp := range []string{"0", "-0.5"} {
			rr := postDecisions(`{"prompt_format_version":1,"temperature":` + temp + `,"prompt":"p","options":[{"id":"a"}]}`)
			assertDecisionsRefusal(t, rr, http.StatusBadRequest, "decisions_bad_temperature")
		}
		// Non-finite temperatures have no JSON wire form, so the closed
		// contract is asserted on the validator directly.
		for name, temp := range map[string]float64{"NaN": math.NaN(), "+Inf": math.Inf(1), "-Inf": math.Inf(-1)} {
			err := validateDecisionsRequest(&DecisionsRequest{
				PromptFormatVersion: 1, Temperature: temp, Prompt: "p", Options: []DecisionsOption{{ID: "a"}},
			})
			if !errors.Is(err, ErrDecisionsTemperature) {
				t.Fatalf("%s temperature: err = %v, want ErrDecisionsTemperature", name, err)
			}
		}
	})

	t.Run("UnsupportedPromptFormatRefuses400", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) { return []float64{0}, nil })
		for _, version := range []int{0, 2} {
			rr := postDecisions(`{"prompt_format_version":` + strconv.Itoa(version) + `,"temperature":1,"prompt":"p","options":[{"id":"a"}]}`)
			assertDecisionsRefusal(t, rr, http.StatusBadRequest, "decisions_bad_prompt_format")
		}
	})

	t.Run("ZeroOptionsRefuse400", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) { return nil, nil })
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[]}`)
		assertDecisionsRefusal(t, rr, http.StatusBadRequest, "decisions_no_options")
	})

	t.Run("MoreThan26OptionsRefuse400", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) { return nil, nil })
		opts := make([]string, 27)
		for i := range opts {
			opts[i] = `{"id":"o` + strconv.Itoa(i) + `"}`
		}
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[` + strings.Join(opts, ",") + `]}`)
		assertDecisionsRefusal(t, rr, http.StatusBadRequest, "decisions_too_many_options")
	})

	t.Run("TwentySixOptionsAreTheCapNotAnError", func(t *testing.T) {
		installScorer(t, func(_ context.Context, in DecisionsScoreInput) ([]float64, error) {
			return make([]float64, len(in.OptionIDs)), nil
		})
		opts := make([]string, 26)
		for i := range opts {
			opts[i] = `{"id":"o` + strconv.Itoa(i) + `"}`
		}
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[` + strings.Join(opts, ",") + `]}`)
		if rr.Code != http.StatusOK {
			t.Fatalf("26 options answered %d, want 200 (the A-Z cap is inclusive); body=%s", rr.Code, rr.Body.String())
		}
		var res DecisionsResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &res); err != nil {
			t.Fatalf("200 body is not JSON: %v", err)
		}
		if res.Results[25].Label != "Z" {
			t.Fatalf("last label = %q, want Z", res.Results[25].Label)
		}
	})

	t.Run("OverflowedLogitRefuses400", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) {
			return []float64{math.Inf(1), 0}, nil
		})
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[{"id":"a"},{"id":"b"}]}`)
		assertDecisionsRefusal(t, rr, http.StatusBadRequest, "decisions_logit_overflow")
		// Closed contract on the pure function: both +Inf and NaN are the
		// overflow sentinel, and a healthy vector normalizes.
		for _, bad := range [][]float64{{math.Inf(1), 0}, {0, math.Inf(-1)}, {math.NaN(), 1}} {
			if _, err := softmaxDecisionProbs(bad, 1); !errors.Is(err, ErrDecisionsOverflow) {
				t.Fatalf("softmax(%v): err = %v, want ErrDecisionsOverflow", bad, err)
			}
		}
		if _, err := softmaxDecisionProbs([]float64{}, 1); !errors.Is(err, ErrDecisionsNoOptions) {
			t.Fatalf("softmax(empty): err = %v, want ErrDecisionsNoOptions", err)
		}
	})

	t.Run("MalformedBodyAndShapeRefuse400", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) { return []float64{0}, nil })
		cases := map[string]string{
			"bad json":     `{not json`,
			"empty prompt": `{"prompt_format_version":1,"temperature":1,"prompt":"","options":[{"id":"a"}]}`,
			"empty id":     `{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[{"id":""}]}`,
		}
		for _, body := range cases {
			rr := postDecisions(body)
			assertDecisionsRefusal(t, rr, http.StatusBadRequest, "decisions_malformed_body")
		}
	})

	t.Run("HookFailureOrMismatchRefuses503", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) {
			return nil, errors.New("engine unavailable")
		})
		rr := postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[{"id":"a"},{"id":"b"}]}`)
		assertDecisionsRefusal(t, rr, http.StatusServiceUnavailable, "decisions_scoring_failed")

		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) {
			return []float64{0}, nil // one logit for two options
		})
		rr = postDecisions(`{"prompt_format_version":1,"temperature":1,"prompt":"p","options":[{"id":"a"},{"id":"b"}]}`)
		assertDecisionsRefusal(t, rr, http.StatusServiceUnavailable, "decisions_scoring_failed")
	})

	t.Run("GetIs405", func(t *testing.T) {
		installScorer(t, func(context.Context, DecisionsScoreInput) ([]float64, error) { return []float64{0}, nil })
		rr := httptest.NewRecorder()
		(&Server{}).handleDecisions(rr, httptest.NewRequest(http.MethodGet, "/v1/decisions", nil))
		if rr.Code != http.StatusMethodNotAllowed {
			t.Fatalf("GET answered %d, want 405", rr.Code)
		}
	})

	t.Run("ValidationSentinelsAreClosed", func(t *testing.T) {
		valid := func() *DecisionsRequest {
			return &DecisionsRequest{
				PromptFormatVersion: 1, Temperature: 0.7, Prompt: "p",
				Options: []DecisionsOption{{ID: "a"}, {ID: "b"}},
			}
		}
		if err := validateDecisionsRequest(valid()); err != nil {
			t.Fatalf("valid request rejected: %v", err)
		}
		r := valid()
		r.PromptFormatVersion = 2
		if err := validateDecisionsRequest(r); !errors.Is(err, ErrDecisionsPromptFormat) {
			t.Fatalf("version 2: err = %v, want ErrDecisionsPromptFormat", err)
		}
		r = valid()
		r.Temperature = 0
		if err := validateDecisionsRequest(r); !errors.Is(err, ErrDecisionsTemperature) {
			t.Fatalf("temperature 0: err = %v, want ErrDecisionsTemperature", err)
		}
		r = valid()
		r.Options = nil
		if err := validateDecisionsRequest(r); !errors.Is(err, ErrDecisionsNoOptions) {
			t.Fatalf("no options: err = %v, want ErrDecisionsNoOptions", err)
		}
		r = valid()
		r.Options = make([]DecisionsOption, 27)
		for i := range r.Options {
			r.Options[i] = DecisionsOption{ID: strconv.Itoa(i)}
		}
		if err := validateDecisionsRequest(r); !errors.Is(err, ErrDecisionsTooManyOptions) {
			t.Fatalf("27 options: err = %v, want ErrDecisionsTooManyOptions", err)
		}
		r = valid()
		r.Options[1].ID = r.Options[0].ID
		if err := validateDecisionsRequest(r); !errors.Is(err, ErrDecisionsDuplicateOptionID) {
			t.Fatalf("duplicate id: err = %v, want ErrDecisionsDuplicateOptionID", err)
		}
		r = valid()
		r.Options[0].ID = ""
		if err := validateDecisionsRequest(r); !errors.Is(err, ErrDecisionsMalformedBody) {
			t.Fatalf("empty id: err = %v, want ErrDecisionsMalformedBody", err)
		}
	})
}
