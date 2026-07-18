package main

import (
	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// The additive lever seam (#5108) — the lever half of internal/dojo's
// RegisterClaim. A KPI-cell leaf ships its lever + catalog row in its OWN file
// — a plain package-level `var _ = RegisterLever(...)` — with no edit to the
// central dojoLeverCatalogBase / allDojoLevers literals, so parallel KPI-cell
// workers stop serializing on cmd/fak/dojo.go. Composition only: concrete
// corpus-scanning levers still live in cmd/fak (internal/architest pins
// `dojo: 1`, stdlib-only), and no claim, metric, or acceptance gate moves.

// dojoLeverEnv carries the per-run parameters allDojoLevers threads into every
// lever: the workspace root (the loop-ledger home for dispatch-yield-style
// levers), the resume cache TTL, and the corpus file cap. A registered builder
// takes the whole env so the seam's signature never churns when a future lever
// needs a parameter its predecessors ignored.
type dojoLeverEnv struct {
	Root     string
	TTL      resume.CacheTTL
	MaxFiles int
}

// dojoLeverEntry pairs a registered lever's catalog row with the builder that
// constructs it for one run. The row is the same dojoLeverInfo shape the
// central catalog literal holds, so `fak dojo list` and the catalog-vs-emitted
// witnesses (TestDojoCatalogMatches*) see registered levers exactly like
// literal ones.
type dojoLeverEntry struct {
	info  dojoLeverInfo
	build func(dojoLeverEnv) dojo.Lever
}

// registeredLevers is the additive lever seam: the composed home for levers a
// KPI leaf declares in its own file via RegisterLever, kept separate from the
// dojoLeverCatalogBase / allDojoLevers literals so adding a lever never edits
// — and never conflicts on — those central literals. allDojoLevers and
// dojoLeverCatalog fold it in after the literal set, so a registered lever is
// selectable, listed, and part of the default `dojo run` fold.
var registeredLevers []dojoLeverEntry

// RegisterLever adds one lever + its catalog row to the additive seam so a KPI
// cell lands in its own file — a plain package-level `var _ = RegisterLever(...)`
// — with no edit to allDojoLevers() or dojoLeverCatalog(), letting parallel
// KPI-cell workers avoid colliding on one file and two literals. It mirrors
// dojo.RegisterClaim: it panics on a duplicate lever name (already in the
// central catalog literal or already registered), because a doubly-registered
// lever is a programming error surfaced loudly at init rather than a silent
// shadow. It returns the info so it composes as a var initializer.
func RegisterLever(info dojoLeverInfo, build func(dojoLeverEnv) dojo.Lever) dojoLeverInfo {
	if info.Name == "" {
		panic("dojo: RegisterLever with an empty lever name")
	}
	if build == nil {
		panic("dojo: RegisterLever with a nil builder: " + info.Name)
	}
	for _, lv := range dojoLeverCatalogBase() {
		if lv.Name == info.Name {
			panic("dojo: RegisterLever on a lever already in the central catalog: " + info.Name)
		}
	}
	for _, e := range registeredLevers {
		if e.info.Name == info.Name {
			panic("dojo: RegisterLever on an already-registered lever: " + info.Name)
		}
	}
	registeredLevers = append(registeredLevers, dojoLeverEntry{info: info, build: build})
	return info
}

// registeredLeverInfos returns the catalog rows of the additively registered
// levers, in registration order, for dojoLeverCatalog to fold after the
// central literal.
func registeredLeverInfos() []dojoLeverInfo {
	out := make([]dojoLeverInfo, 0, len(registeredLevers))
	for _, e := range registeredLevers {
		out = append(out, e.info)
	}
	return out
}

// registeredLeverSet builds the additively registered levers for one run, in
// registration order, for allDojoLevers to fold after the central literal.
func registeredLeverSet(env dojoLeverEnv) []dojo.Lever {
	out := make([]dojo.Lever, 0, len(registeredLevers))
	for _, e := range registeredLevers {
		out = append(out, e.build(env))
	}
	return out
}
