// Package modelroute is fak's model-routing spine: choose WHICH model — or which
// ENSEMBLE of models — serves any ASPECT of a request, under one declarative,
// deterministic, verifiable policy.
//
// The pure policy core (Subject, Match, Rule, Manifest, Plan, Route, Validate,
// Combine and the manifest round-trip) lives in the public package
// pkg/routepolicy, where an out-of-tree program can import it or replace it with
// its own Policy. This package re-exports that core through type aliases and
// forwarding functions, and adds the in-tree layers built on it: account rosters,
// tier policy, serving, cross-audit, escalation and the polymodel bridge.
//
// The package contract (schema tag, first-match-wins, fail-closed default,
// determinism, closed versus open vocabularies) and the wiring contract (route
// before adjudicate, one adjudicated call per ensemble member, member order
// preserved into the fold) are documented at pkg/routepolicy.
package modelroute

import "github.com/anthony-chaudhary/fak/pkg/routepolicy"

const Version = routepolicy.Version

type (
	Aspect     = routepolicy.Aspect
	Latency    = routepolicy.Latency
	Complexity = routepolicy.Complexity
	Subject    = routepolicy.Subject
	Member     = routepolicy.Member
	Reduction  = routepolicy.Reduction
	Plan       = routepolicy.Plan
	Match      = routepolicy.Match
	Rule       = routepolicy.Rule
	Manifest   = routepolicy.Manifest
	Decision   = routepolicy.Decision
	Vote       = routepolicy.Vote
	Result     = routepolicy.Result
)

const (
	AspectRequest  = routepolicy.AspectRequest
	AspectToolCall = routepolicy.AspectToolCall
	AspectQuery    = routepolicy.AspectQuery
	AspectState    = routepolicy.AspectState
	AspectStep     = routepolicy.AspectStep
	AspectScout    = routepolicy.AspectScout
)

const (
	LatencyAny         = routepolicy.LatencyAny
	LatencyInteractive = routepolicy.LatencyInteractive
	LatencyBatch       = routepolicy.LatencyBatch
)

const (
	ComplexityAny    = routepolicy.ComplexityAny
	ComplexityLow    = routepolicy.ComplexityLow
	ComplexityMedium = routepolicy.ComplexityMedium
	ComplexityHigh   = routepolicy.ComplexityHigh
)

const (
	ReduceFirst     = routepolicy.ReduceFirst
	ReduceVote      = routepolicy.ReduceVote
	ReduceBestOf    = routepolicy.ReduceBestOf
	ReduceAllReduce = routepolicy.ReduceAllReduce
	ReduceConcat    = routepolicy.ReduceConcat
)

func ParseManifest(b []byte) (Manifest, error) { return routepolicy.ParseManifest(b) }

func LoadManifest(path string) (Manifest, error) { return routepolicy.LoadManifest(path) }

func DefaultManifest() Manifest { return routepolicy.DefaultManifest() }

func Combine(reduce Reduction, votes []Vote) (Result, error) {
	return routepolicy.Combine(reduce, votes)
}

func weightOf(m Member) float64 { return routepolicy.EffectiveWeight(m) }
