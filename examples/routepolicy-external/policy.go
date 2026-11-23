// Package external shows both directions of the routepolicy seam from a module
// that is not fak: routing with fak's reference Manifest, and supplying an
// independent Policy implementation that passes fak's conformance kit.
package external

import (
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy"
)

// RouteWithFak parses a fak-route/v1 manifest and routes one subject with fak's
// reference implementation.
func RouteWithFak(manifestJSON []byte, s routepolicy.Subject) (routepolicy.Decision, error) {
	m, err := routepolicy.ParseManifest(manifestJSON)
	if err != nil {
		return routepolicy.Decision{}, err
	}
	return m.Route(s), nil
}

// IndexedPolicy is an independent Policy: it pre-buckets rules by aspect at
// build time so a subject only scans the rules that can apply to its aspect,
// re-implementing the match semantics rather than calling Match.Matches.
type IndexedPolicy struct {
	byAspect  map[routepolicy.Aspect][]indexedRule
	anyAspect []indexedRule
	fallback  routepolicy.Plan
}

type indexedRule struct {
	order int
	rule  routepolicy.Rule
}

// NewIndexedPolicy builds an IndexedPolicy from a manifest.
func NewIndexedPolicy(m routepolicy.Manifest) *IndexedPolicy {
	p := &IndexedPolicy{byAspect: map[routepolicy.Aspect][]indexedRule{}, fallback: m.Default}
	for i, r := range m.Rules {
		ir := indexedRule{order: i, rule: r}
		if r.Match.Aspect == "" {
			p.anyAspect = append(p.anyAspect, ir)
			continue
		}
		p.byAspect[r.Match.Aspect] = append(p.byAspect[r.Match.Aspect], ir)
	}
	return p
}

// Route merges the aspect bucket with the aspect-wildcard rules in declared
// order, so the first declared matching rule still wins.
func (p *IndexedPolicy) Route(s routepolicy.Subject) routepolicy.Decision {
	a, b := p.byAspect[s.Aspect], p.anyAspect
	for len(a) > 0 || len(b) > 0 {
		var next indexedRule
		if len(b) == 0 || (len(a) > 0 && a[0].order < b[0].order) {
			next, a = a[0], a[1:]
		} else {
			next, b = b[0], b[1:]
		}
		if holds(next.rule.Match, s) {
			return routepolicy.Decision{Subject: s, RuleName: next.rule.Name, Matched: true, Plan: next.rule.Plan}
		}
	}
	return routepolicy.Decision{Subject: s, Plan: p.fallback}
}

var complexityOrder = map[routepolicy.Complexity]int{
	routepolicy.ComplexityLow:    1,
	routepolicy.ComplexityMedium: 2,
	routepolicy.ComplexityHigh:   3,
}

func holds(m routepolicy.Match, s routepolicy.Subject) bool {
	switch {
	case m.Tool != "" && !toolHolds(m.Tool, s.Tool):
		return false
	case s.PromptTokens < m.MinPromptTokens:
		return false
	case m.MaxPromptTokens > 0 && s.PromptTokens > m.MaxPromptTokens:
		return false
	case m.Latency != "" && m.Latency != s.Latency:
		return false
	case m.MinComplexity != "" && complexityOrder[s.Complexity] < complexityOrder[m.MinComplexity]:
		return false
	case m.InputTrigger != "" && m.InputTrigger != s.InputTrigger:
		return false
	}
	for k, v := range m.Labels {
		if s.Labels[k] != v {
			return false
		}
	}
	return true
}

func toolHolds(pattern, tool string) bool {
	if prefix, wildcard := strings.CutSuffix(pattern, "*"); wildcard {
		return strings.HasPrefix(tool, prefix)
	}
	return pattern == tool
}
