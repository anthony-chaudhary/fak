// winrate.go — the WIN-RATE-THRESHOLD ROUTING CONTRACT: a strong/weak model
// pair, a single-method win-rate scorer, and the dispatch rule that sends a
// subject to the strong model exactly when the scored win rate meets the
// threshold. Plus the "router-<name>-<threshold>" model-id grammar that names
// a registered router and its cutoff in one wire string.
//
// BORROWED CONTRACT (ADAPT, not a port): lm-sys/RouteLLM @
// 0b64fdafe049e596a3f5657c219329f24af24198 (Apache-2.0; upstream dormant since
// 2024-08-10, so this is a design reference, never a live dependency). The
// contract adapted here is the two-tier dispatch in routellm/routers/routers.py:32-48
// (one abstract calculate_strong_win_rate returning P(strong wins) in [0,1],
// and a concrete route() that picks the strong model iff the value is >= the
// user-defined cutoff) together with the "router-[router name]-[threshold]"
// model-name grammar and its validation in routellm/controller.py:93-103.
// Re-expressed in idiomatic Go against this package's Subject vocabulary; no
// upstream code block is copied. Study note: fak-private
// docs/research/STUDY-PRODUCTION-SCALING-ECON-2026-11-07.md, candidate
// routellm-1. Ticket: oss-port-routellm-routellm-1.
//
// WHY THIS BELONGS ON THE SPINE. modelroute.go decides an abstract model id
// per Aspect; account.go binds ids to provider accounts. This file adds the
// missing ECONOMIC tier between them: given a strong/weak PAIR of already-routable
// ids, decide which half of the pair serves a subject from a probability, with
// the decision rule owned by ONE place (ThresholdRouter.Route) so every scorer
// plugs into the same closed contract instead of re-implementing the compare.
// The package stays pure (stdlib only): a scorer is injected, Route does no I/O.
package modelroute

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Closed refusal vocabulary for the win-rate-threshold contract. Callers match
// these with errors.Is; the wrapped detail after ": " is diagnostic only.
var (
	// ErrInvalidModelPair is a strong/weak pair that is empty or has both tiers
	// bound to the same model id (routing would be a no-op coin flip on cost).
	ErrInvalidModelPair = errors.New("modelroute: invalid strong/weak model pair")
	// ErrInvalidThreshold is a cutoff outside the closed [0,1] interval (or
	// non-finite), where no probability can meet it honestly.
	ErrInvalidThreshold = errors.New("modelroute: win-rate threshold out of range")
	// ErrInvalidWinRate is a scorer output that is not a finite probability in
	// [0,1]; dispatch refuses rather than guess a tier.
	ErrInvalidWinRate = errors.New("modelroute: strong win rate out of range")
	// ErrInvalidRouterModel is a model id that does not match the
	// "router-<name>-<threshold>" grammar at all.
	ErrInvalidRouterModel = errors.New("modelroute: malformed router-<name>-<threshold> model id")
	// ErrUnknownRouter is a grammatical id whose router name is not registered
	// in the resolving catalog.
	ErrUnknownRouter = errors.New("modelroute: router not registered in catalog")
)

// ModelPair is the two-tier pair a win-rate router dispatches over: Strong is
// the higher-capability (usually costlier) model id, Weak the cheaper fallback.
// Both are ABSTRACT model ids the rest of the spine already routes and binds.
type ModelPair struct {
	Strong string `json:"strong"`
	Weak   string `json:"weak"`
}

// Validate enforces the closed pair contract: both ids present and distinct.
func (p ModelPair) Validate() error {
	if p.Strong == "" || p.Weak == "" {
		return fmt.Errorf("%w: both strong and weak model ids must be set (got strong=%q weak=%q)", ErrInvalidModelPair, p.Strong, p.Weak)
	}
	if p.Strong == p.Weak {
		return fmt.Errorf("%w: strong and weak must be distinct model ids (both %q)", ErrInvalidModelPair, p.Strong)
	}
	return nil
}

// WinRateScorer is the single-method routing contract borrowed from
// RouteLLM's abstract Router.calculate_strong_win_rate: score ONE subject and
// return P(strong wins) in [0,1]. A scorer NEVER dispatches — the threshold
// compare belongs to ThresholdRouter.Route, so every scorer (calibrated
// quantile, matrix factorization, similarity-weighted ranking, …) plugs into
// the same closed rule without re-implementing it.
type WinRateScorer interface {
	// CalculateStrongWinRate returns the probability that the pair's strong
	// model beats its weak model on s, as a finite value in [0,1].
	CalculateStrongWinRate(s Subject) (float64, error)
}

// ThresholdRouter binds one scorer to one pair and one cutoff — the whole
// economic routing unit. Name is the registry token used in the
// "router-<name>-<threshold>" model id; Scorer is the injected win-rate
// function; Pair is the strong/weak target; Threshold is the user-defined
// cutoff the win rate must MEET (>=, inclusive) to take the strong tier.
type ThresholdRouter struct {
	Name      string        `json:"name,omitempty"`
	Scorer    WinRateScorer `json:"-"`
	Pair      ModelPair     `json:"pair"`
	Threshold float64       `json:"threshold"`
}

// Validate checks the router is ready to dispatch: a wired scorer, a valid
// pair, and a cutoff inside [0,1].
func (r ThresholdRouter) Validate() error {
	if r.Scorer == nil {
		return fmt.Errorf("%w: router %q has no win-rate scorer", ErrInvalidModelPair, r.Name)
	}
	if err := r.Pair.Validate(); err != nil {
		return err
	}
	if !inUnitInterval(r.Threshold) {
		return fmt.Errorf("%w: threshold %v must be a finite probability in [0,1]", ErrInvalidThreshold, r.Threshold)
	}
	return nil
}

// Route dispatches s under the borrowed contract: score once, then send to
// the strong model iff the win rate is >= Threshold (inclusive boundary, the
// upstream rule verbatim in semantics), else to the weak model. It fails loud
// on an unwired router or a scorer output that is not a probability — never a
// silent tier guess.
func (r ThresholdRouter) Route(s Subject) (string, error) {
	if err := r.Validate(); err != nil {
		return "", err
	}
	w, err := r.Scorer.CalculateStrongWinRate(s)
	if err != nil {
		return "", err
	}
	if !inUnitInterval(w) {
		return "", fmt.Errorf("%w: scorer %q returned %v for the subject", ErrInvalidWinRate, r.Name, w)
	}
	if w >= r.Threshold {
		return r.Pair.Strong, nil
	}
	return r.Pair.Weak, nil
}

// inUnitInterval reports whether x is a finite value inside [0,1]. NaN fails
// both comparisons and ±Inf falls outside, so one predicate covers the whole
// non-probability surface.
func inUnitInterval(x float64) bool { return x >= 0 && x <= 1 }

// ---------------------------------------------------------------------------
// THE MODEL-ID GRAMMAR — "router-<name>-<threshold>".
// ---------------------------------------------------------------------------

// routerModelPrefix is the literal wire prefix of a win-rate-routed model id.
const routerModelPrefix = "router-"

// ParseRouterModel parses the "router-<name>-<threshold>" model-id grammar:
// the literal "router-" prefix, a router NAME of one or more
// [A-Za-z0-9_] characters carrying no further hyphen, a separating "-", and a
// decimal threshold token ([0-9]* "."? [0-9]+ with at least one digit, so
// "0.5", ".5", "0" and "1" parse while "nan", "inf", "+0.5" and "1e-3" do
// not). The parsed threshold must additionally sit in [0,1] or
// ErrInvalidThreshold is returned. This is the clean-room tightening of
// RouteLLM controller._parse_model_name + _validate_router_threshold at the
// pinned revision: the upstream split-and-recheck order is replaced by one
// deterministic pass with a closed refusal vocabulary.
func ParseRouterModel(id string) (name string, threshold float64, err error) {
	rest := strings.TrimPrefix(id, routerModelPrefix)
	if rest == id { // prefix absent (TrimPrefix leaves the input unchanged)
		return "", 0, fmt.Errorf("%w: %q does not start with %q", ErrInvalidRouterModel, id, routerModelPrefix)
	}
	i := strings.LastIndex(rest, "-")
	if i <= 0 {
		return "", 0, fmt.Errorf("%w: %q needs a router name and a threshold after the prefix", ErrInvalidRouterModel, id)
	}
	name, token := rest[:i], rest[i+len("-"):]
	if !validRouterName(name) {
		return "", 0, fmt.Errorf("%w: %q has an invalid router name %q", ErrInvalidRouterModel, id, name)
	}
	if !validThresholdToken(token) {
		return "", 0, fmt.Errorf("%w: %q has a non-numeric threshold token %q", ErrInvalidRouterModel, id, token)
	}
	threshold, err = strconv.ParseFloat(token, 64)
	if err != nil {
		return "", 0, fmt.Errorf("%w: %q has an unparseable threshold %q", ErrInvalidRouterModel, id, token)
	}
	if !inUnitInterval(threshold) {
		return "", 0, fmt.Errorf("%w: %q threshold %v must be in [0,1]", ErrInvalidThreshold, id, threshold)
	}
	return name, threshold, nil
}

// validRouterName accepts the closed charset of a router registry token.
func validRouterName(name string) bool {
	if name == "" {
		return false
	}
	for _, c := range name {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_') {
			return false
		}
	}
	return true
}

// validThresholdToken accepts a plain decimal in [0,1] shape: optional leading
// digits, at most one dot, and one or more digits AFTER the dot when one is
// present (so "0.5", ".5", "0" and "1" parse while "1.", "nan", "inf", "+0.5"
// and "1e-3" do not).
func validThresholdToken(token string) bool {
	if token == "" {
		return false
	}
	dots, trailingDigits, anyDigit := 0, 0, false
	for _, c := range token {
		switch {
		case c >= '0' && c <= '9':
			anyDigit = true
			trailingDigits++
		case c == '.':
			dots++
			trailingDigits = 0
		default:
			return false
		}
	}
	if dots > 1 || dots == 1 && trailingDigits == 0 {
		return false
	}
	return anyDigit
}

// RouterCatalog is the set of router names a deployment has wired scorers for
// — the analog of RouteLLM Controller.routers / ROUTER_CLS lookup. Resolution
// is exact-name, case-sensitive.
type RouterCatalog struct {
	scorers map[string]WinRateScorer
}

// NewRouterCatalog builds a catalog from name → scorer entries. Nil scorers
// are dropped (registering nothing is better than registering a dead router);
// a nil map yields an empty catalog.
func NewRouterCatalog(scorers map[string]WinRateScorer) RouterCatalog {
	c := RouterCatalog{scorers: make(map[string]WinRateScorer, len(scorers))}
	for name, s := range scorers {
		if s != nil {
			c.scorers[name] = s
		}
	}
	return c
}

// Has reports whether name is a registered router.
func (c RouterCatalog) Has(name string) bool {
	_, ok := c.scorers[name]
	return ok
}

// Resolve turns a "router-<name>-<threshold>" model id into a ready-to-dispatch
// ThresholdRouter over pair: parse the grammar, require the name registered,
// and validate the assembled router before handing it back. The returned
// router is guaranteed to pass Validate.
func (c RouterCatalog) Resolve(id string, pair ModelPair) (ThresholdRouter, error) {
	name, threshold, err := ParseRouterModel(id)
	if err != nil {
		return ThresholdRouter{}, err
	}
	scorer, ok := c.scorers[name]
	if !ok {
		return ThresholdRouter{}, fmt.Errorf("%w: %q (id %q)", ErrUnknownRouter, name, id)
	}
	r := ThresholdRouter{Name: name, Scorer: scorer, Pair: pair, Threshold: threshold}
	if err := r.Validate(); err != nil {
		return ThresholdRouter{}, err
	}
	return r, nil
}
