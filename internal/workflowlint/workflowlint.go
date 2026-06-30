// Package workflowlint refutes "fak-blind" ultracode Workflow scripts — the ones
// an ultracode session emits that never touch fak's own self-index, memory algebra,
// or shared-path leasing. It is the checkable form of epic #1494 / C4 #1502: a
// fak-guarded session should not be able to generate a workflow that orchestrates
// generic agents instead of fak-native ones.
//
// A workflow is FAK-NATIVE iff it references all three concept classes:
//
//   - SelfIndex  — query fak's own facts instead of re-surveying prose
//     (`fak index` / fak_index_*)
//   - Memory     — the memory algebra: recall before work, compact/clean after
//     (`fak memory` / fak_memory_* / `--driver recall|compact|clean` / memq)
//   - SharedPath — arbitrated leasing so a spawn wave shares the path without
//     colliding (dos_arbitrate / lease / COLLISION_RISK)
//
// Anything missing a class is FAK-BLIND and Lint refuses it (Report.Native == false).
// The package is tier-1 (stdlib only) so it can be embedded anywhere the ultracode
// generation path runs, the same way internal/devindex backs `fak index`.
package workflowlint

import (
	_ "embed"
	"regexp"
	"sort"
	"strings"
)

// SeedTemplate is the canonical fak-native Workflow seed. An ultracode session
// running inside the fak kernel should emit a workflow shaped like this — and by
// construction Lint(SeedTemplate).Native is true (asserted in the tests).
//
//go:embed seed.js
var SeedTemplate string

// Verdicts. These are the two closed outcomes Lint returns.
const (
	VerdictNative = "FAK-NATIVE" // all three concept classes present
	VerdictBlind  = "FAK-BLIND"  // at least one concept class absent — refused
)

// Concept-class keys (stable identifiers used in JSON and by callers).
const (
	ClassSelfIndex  = "self-index"
	ClassMemory     = "memory"
	ClassSharedPath = "shared-path"
)

// A class is one fak concept the lint requires, plus the patterns that witness it.
// Patterns are matched case-insensitively against the script text.
type class struct {
	key      string
	why      string
	patterns []*regexp.Regexp
}

// classes is the ordered requirement set. Order is stable so Report.Classes and the
// rendered output never reshuffle between runs.
var classes = []class{
	{
		key: ClassSelfIndex,
		why: "query fak's own facts (fak index / fak_index_*) instead of re-surveying prose",
		patterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)fak[ _]index`),
		},
	},
	{
		key: ClassMemory,
		why: "use the fak memory algebra (recall before work, compact/clean after) — fak_memory_* / --driver recall|compact|clean / memq",
		patterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)fak[ _]memory`),
			regexp.MustCompile(`(?i)--driver\s+(?:recall|compact|clean|render|dream)`),
			regexp.MustCompile(`(?i)\bmemq\b`),
		},
	},
	{
		key: ClassSharedPath,
		why: "lease a disjoint shared path per agent (dos_arbitrate / lease / COLLISION_RISK) so a spawn wave never collides",
		patterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)dos_arbitrate`),
			regexp.MustCompile(`(?i)\blease\b`),
			regexp.MustCompile(`(?i)collision_risk`),
		},
	},
}

// ClassHit is the per-class result: whether the concept was found, and a sample of
// the concrete tokens that witnessed it (empty when absent).
type ClassHit struct {
	Key     string   `json:"key"`
	Present bool     `json:"present"`
	Why     string   `json:"why"`
	Matched []string `json:"matched,omitempty"`
}

// Report is the closed lint verdict for one workflow script.
type Report struct {
	Native  bool       `json:"native"`            // all three concept classes present
	Verdict string     `json:"verdict"`           // VerdictNative | VerdictBlind
	Classes []ClassHit `json:"classes"`           // per-class detail, in requirement order
	Missing []string   `json:"missing,omitempty"` // keys of the absent classes (FAK-BLIND reason)
}

// maxMatched caps how many witness tokens a class reports, so a script that mentions
// `fak index` fifty times doesn't bloat the report.
const maxMatched = 4

// Lint adjudicates one Workflow script. It never errors: an empty or non-workflow
// script is simply FAK-BLIND (all classes absent), which is the conservative refusal.
func Lint(script string) Report {
	rep := Report{Verdict: VerdictNative, Native: true}
	for _, c := range classes {
		hit := ClassHit{Key: c.key, Why: c.why}
		seen := map[string]bool{}
		for _, p := range c.patterns {
			for _, m := range p.FindAllString(script, -1) {
				tok := strings.ToLower(strings.Join(strings.Fields(m), " "))
				if tok == "" || seen[tok] {
					continue
				}
				seen[tok] = true
				hit.Matched = append(hit.Matched, tok)
			}
		}
		sort.Strings(hit.Matched)
		if len(hit.Matched) > maxMatched {
			hit.Matched = hit.Matched[:maxMatched]
		}
		hit.Present = len(hit.Matched) > 0
		if !hit.Present {
			rep.Native = false
			rep.Missing = append(rep.Missing, c.key)
		}
		rep.Classes = append(rep.Classes, hit)
	}
	if !rep.Native {
		rep.Verdict = VerdictBlind
	}
	return rep
}
