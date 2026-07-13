package commitintent

// changed_surface_selection.go is the changed-surface-selection child of the
// quality spine (#4575, sibling of the loader-parity child #4545 in
// internal/modelengine): it maps a commit's changed paths — the exact Paths an
// Intent already carries — to the quality cases that must run, assigns each to a
// PR / nightly / release tier, and EXPANDS coverage when a changed path matches
// no known surface (an unknown change is never silently unselected — it routes
// to the broad sentinel canary). The full rule table plus expand-on-unknown is
// the reference selector; a dropped rule, a disabled expansion, or a mis-tiered
// case surfaces as the FIRST actionable divergence — the exact case that should
// (not) have run, and how — with a scrubbed replay artifact, before the suite is
// trusted.
//
// The oracle is deterministic and self-contained (no git, no diff bytes, no
// network): it routes repo-relative path prefixes to named surfaces by
// longest-match and folds the rule table into a baseline fingerprint, so any
// selection defect that changes which cases run perturbs the very case where it
// first bites. The package stays pure (it never runs git — see doc.go): the
// caller hands it the already-known changed paths. Every case is scrubbed to its
// surface routing and directory prefix, never a raw diff hunk or file body.
// Runtime/resource cost: pure in-process, microseconds per diff, no external
// fixtures. Tier: PR (runs in the package unit gate).

import (
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
)

// cselTier is the CI tier a selected quality case runs in — the cheap-PR vs
// nightly-accuracy vs release-canary split the parent contract (#4509) requires.
type cselTier string

const (
	cselTierPR      cselTier = "PR"
	cselTierNightly cselTier = "nightly"
	cselTierRelease cselTier = "release"
)

func cselValidTier(t cselTier) bool {
	switch t {
	case cselTierPR, cselTierNightly, cselTierRelease:
		return true
	default:
		return false
	}
}

// cselSurface is a named quality surface a diff can touch. The scope maps diffs
// to models, backends, samplers, caches, and report rubrics, plus sentinels.
type cselSurface string

const (
	cselSurfaceModel    cselSurface = "model"
	cselSurfaceBackend  cselSurface = "backend"
	cselSurfaceSampler  cselSurface = "sampler"
	cselSurfaceCache    cselSurface = "cache"
	cselSurfaceReport   cselSurface = "report-rubric"
	cselSurfaceSentinel cselSurface = "sentinel"
)

// cselSeed / cselRevision / cselTolerance are the deterministic-oracle
// provenance every selected case records.
const (
	cselSeed      uint64 = 0x5e1ec7ed
	cselRevision         = "commitintent@changed-surface-selection-1"
	cselTolerance        = "exact (deterministic selection oracle)"
)

// cselSurfaceProfile is a surface's representative model/tokenizer/backend — the
// per-case provenance the acceptance criteria require. Every surface has a
// non-empty profile so an unprovenanced case is impossible for a real surface.
type cselSurfaceProfile struct {
	model     string
	tokenizer string
	backend   string
}

var cselProfiles = map[cselSurface]cselSurfaceProfile{
	cselSurfaceModel:    {"fak-fixture-7b", "fak-bpe-v1", "inkernel-forward"},
	cselSurfaceBackend:  {"fak-fixture-7b", "fak-bpe-v1", "inkernel-dispatch"},
	cselSurfaceSampler:  {"fak-fixture-7b", "fak-bpe-v1", "greedy+topk-sampler"},
	cselSurfaceCache:    {"fak-fixture-7b", "fak-bpe-v1", "kv-mmu-cache"},
	cselSurfaceReport:   {"report-judge-v1", "rubric-tokenizer-v1", "milestonereport-pairwise"},
	cselSurfaceSentinel: {"fak-fixture-7b", "fak-bpe-v1", "canary-broad-serve"},
}

// cselRule maps a repo-relative path substring to a surface, the tier the surface
// runs in, and a one-line runtime/resource cost. Longest match wins, so a more
// specific prefix (a sampler dir under the model tree) routes past a broader one.
type cselRule struct {
	match   string
	surface cselSurface
	tier    cselTier
	cost    string
}

// cselRuleTable is the reference surface map. It is deliberately ordered
// specific-first for readability; selection is by longest match, not table
// order, so a reordering never changes the result (the oracle's stability).
func cselRuleTable() []cselRule {
	return []cselRule{
		{"internal/model/sample", cselSurfaceSampler, cselTierNightly, "nightly: sampler distribution sweep (~2m CPU)"},
		{"internal/model/", cselSurfaceModel, cselTierPR, "PR: model-forward parity (in-process, ms)"},
		{"internal/modelengine/", cselSurfaceBackend, cselTierPR, "PR: engine dispatch parity (in-process, ms)"},
		{"internal/modelroute/", cselSurfaceBackend, cselTierPR, "PR: engine route parity (in-process, ms)"},
		{"internal/ctxmmu/", cselSurfaceCache, cselTierNightly, "nightly: KV-cache reuse soak (~5m CPU)"},
		{"internal/kvmmu/", cselSurfaceCache, cselTierNightly, "nightly: KV quarantine soak (~5m CPU)"},
		{"internal/milestonereport/", cselSurfaceReport, cselTierPR, "PR: report-rubric pairwise eval (in-process, ms)"},
		{"internal/preflight/", cselSurfaceSentinel, cselTierRelease, "release: sentinel canary suite (full serve, GPU)"},
	}
}

// cselSentinelCost is the tier/cost a coverage-expansion sentinel case runs at:
// an unknown surface is treated as a release-tier canary until it is mapped.
const cselSentinelCost = "release: sentinel canary suite (unknown-surface expansion, full serve)"

// cselStrategy is the selector configuration. The faithful strategy is the full
// rule table with expand-on-unknown enabled; each planted defect disables or
// perturbs one dimension so the oracle can localize it.
type cselStrategy struct {
	rules         []cselRule
	expandUnknown bool
}

func cselFaithfulStrategy() cselStrategy {
	return cselStrategy{rules: cselRuleTable(), expandUnknown: true}
}

// cselCase is one selected quality case: which surface, which tier, its cost, and
// the human-readable reason it was selected (the "selector explains choices"
// acceptance). Reason carries directory prefixes only, never full file paths.
type cselCase struct {
	ID      string
	Surface cselSurface
	Tier    cselTier
	Cost    string
	Reason  string
}

func cselCaseID(s cselSurface) string { return "case/" + string(s) }

// cselDir scrubs a changed path down to its directory prefix, dropping the file
// basename so no selection artifact ever carries a full file path (or its body).
func cselDir(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[:i]
	}
	return p
}

// cselSelect maps changed paths to selected quality cases under a strategy. Each
// path routes to its longest-matching rule; a path that matches no rule is an
// unknown surface that, when expansion is enabled, adds (or joins) the sentinel
// canary case — so unknown changes EXPAND coverage rather than silently drop.
func cselSelect(paths []string, st cselStrategy) []cselCase {
	type acc struct {
		c       cselCase
		reasons []string
	}
	byID := map[string]*acc{}
	var unknown []string

	sorted := append([]string(nil), paths...)
	sort.Strings(sorted)
	for _, p := range sorted {
		if strings.TrimSpace(p) == "" {
			continue
		}
		best := cselRule{}
		bestLen := -1
		for _, r := range st.rules {
			if strings.Contains(p, r.match) && len(r.match) > bestLen {
				best, bestLen = r, len(r.match)
			}
		}
		if bestLen < 0 {
			unknown = append(unknown, cselDir(p))
			continue
		}
		id := cselCaseID(best.surface)
		a := byID[id]
		if a == nil {
			a = &acc{c: cselCase{ID: id, Surface: best.surface, Tier: best.tier, Cost: best.cost}}
			byID[id] = a
		}
		a.reasons = append(a.reasons, "changed:"+cselDir(p))
	}

	if len(unknown) > 0 && st.expandUnknown {
		id := cselCaseID(cselSurfaceSentinel)
		a := byID[id]
		if a == nil {
			a = &acc{c: cselCase{ID: id, Surface: cselSurfaceSentinel, Tier: cselTierRelease, Cost: cselSentinelCost}}
			byID[id] = a
		}
		// An expanded sentinel is always a release canary regardless of any
		// explicit sentinel rule that also matched.
		a.c.Tier = cselTierRelease
		a.c.Cost = cselSentinelCost
		for _, d := range unknown {
			a.reasons = append(a.reasons, "expanded:"+d)
		}
	}

	ids := make([]string, 0, len(byID))
	for id := range byID {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]cselCase, 0, len(ids))
	for _, id := range ids {
		a := byID[id]
		a.c.Reason = strings.Join(cselDedupSorted(a.reasons), " ")
		out = append(out, a.c)
	}
	return out
}

func cselDedupSorted(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}

// cselReferenceDiff is the pinned fixture diff: one path per known surface plus
// one unmapped path that must expand to the sentinel canary.
func cselReferenceDiff() []string {
	return []string{
		"internal/model/forward.go",        // -> model (PR)
		"internal/model/sample/topk.go",    // -> sampler (nightly), longest match
		"internal/modelengine/dispatch.go", // -> backend (PR)
		"internal/ctxmmu/kvcache.go",       // -> cache (nightly)
		"internal/milestonereport/roll.go", // -> report-rubric (PR)
		"docs/notes/NEW-SURFACE.md",        // -> unknown -> sentinel expansion (release)
	}
}

// cselBaseline is the rule-table fingerprint: a changed rule (a re-tiered surface
// or a dropped mapping) changes these bytes and thus every case's recorded
// baseline provenance.
func cselBaseline() [32]byte {
	h := sha256.New()
	for _, r := range cselRuleTable() {
		fmt.Fprintf(h, "%s\x00%s\x00%s\x00%s\n", r.match, r.surface, r.tier, r.cost)
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

// cselProvenance records everything the acceptance criteria require per case:
// surface, model, tokenizer, engine/backend, seed/oracle, code revision, and
// tolerance/baseline. It carries no raw diff — only the routed surface.
type cselProvenance struct {
	CaseID    string
	Surface   string
	Model     string
	Tokenizer string
	Backend   string
	Seed      uint64
	Revision  string
	Baseline  string
	Tolerance string
	Tier      string
}

func cselProvenanceOf(c cselCase, baseline [32]byte) cselProvenance {
	prof := cselProfiles[c.Surface]
	return cselProvenance{
		CaseID: c.ID, Surface: string(c.Surface),
		Model: prof.model, Tokenizer: prof.tokenizer, Backend: prof.backend,
		Seed: cselSeed, Revision: cselRevision,
		Baseline:  fmt.Sprintf("%x", baseline[:6]),
		Tolerance: cselTolerance, Tier: string(c.Tier),
	}
}

// complete reports whether every required provenance field is populated — an
// unprovenanced case is inconclusive and must never be reported as pass.
func (p cselProvenance) complete() bool {
	return p.CaseID != "" && p.Surface != "" && p.Model != "" && p.Tokenizer != "" &&
		p.Backend != "" && p.Revision != "" && p.Baseline != "" && p.Tolerance != "" &&
		p.Tier != "" && p.Seed != 0
}

// cselDivergence is the first actionable divergence between the reference
// selection and a candidate: which case, which field diverged, and the two
// values.
type cselDivergence struct {
	CaseID    string
	Field     string // "presence" | "tier" | "surface"
	Reference string
	Candidate string
}

// cselReplayArtifact is the scrubbed, independently-replayable failure bundle:
// full provenance plus the first divergence and the candidate's surface routing,
// carrying surface names but never a raw diff or file body.
type cselReplayArtifact struct {
	Provenance cselProvenance
	Reason     string
	Divergence *cselDivergence
	Surfaces   []string
}

func (a cselReplayArtifact) String() string {
	cid, field, ref, cand := "<none>", "<none>", "<none>", "<none>"
	if a.Divergence != nil {
		cid, field, ref, cand = a.Divergence.CaseID, a.Divergence.Field, a.Divergence.Reference, a.Divergence.Candidate
	}
	p := a.Provenance
	return fmt.Sprintf("replay{case=%s surface=%s model=%s tok=%s backend=%s seed=%#x rev=%s baseline=%s tol=%q tier=%s reason=%s divergence=%s@%s ref=%q cand=%q surfaces=%s}",
		p.CaseID, p.Surface, p.Model, p.Tokenizer, p.Backend, p.Seed, p.Revision, p.Baseline, p.Tolerance, p.Tier,
		a.Reason, field, cid, ref, cand, strings.Join(a.Surfaces, ","))
}

type cselVerdict struct {
	Pass     bool
	Detail   string
	Artifact *cselReplayArtifact
}

func cselIndex(cs []cselCase) map[string]cselCase {
	out := make(map[string]cselCase, len(cs))
	for _, c := range cs {
		out[c.ID] = c
	}
	return out
}

func cselSurfacesOf(cs []cselCase) []string {
	out := make([]string, 0, len(cs))
	for _, c := range cs {
		out = append(out, string(c.Surface))
	}
	return cselDedupSorted(out)
}

// cselFirstDiff returns the first case (by sorted case id) where the reference
// and candidate selections disagree on presence, tier, or surface, or nil if
// they are identical. It lets the defect tests assert localization without
// hard-coding an index.
func cselFirstDiff(ref, got []cselCase) *cselDivergence {
	r, g := cselIndex(ref), cselIndex(got)
	ids := map[string]bool{}
	for id := range r {
		ids[id] = true
	}
	for id := range g {
		ids[id] = true
	}
	sortedIDs := make([]string, 0, len(ids))
	for id := range ids {
		sortedIDs = append(sortedIDs, id)
	}
	sort.Strings(sortedIDs)
	for _, id := range sortedIDs {
		rc, rok := r[id]
		gc, gok := g[id]
		switch {
		case rok && !gok:
			return &cselDivergence{CaseID: id, Field: "presence", Reference: "selected(" + string(rc.Tier) + ")", Candidate: "absent"}
		case !rok && gok:
			return &cselDivergence{CaseID: id, Field: "presence", Reference: "absent", Candidate: "selected(" + string(gc.Tier) + ")"}
		case rc.Tier != gc.Tier:
			return &cselDivergence{CaseID: id, Field: "tier", Reference: string(rc.Tier), Candidate: string(gc.Tier)}
		case rc.Surface != gc.Surface:
			return &cselDivergence{CaseID: id, Field: "surface", Reference: string(rc.Surface), Candidate: string(gc.Surface)}
		}
	}
	return nil
}

// cselJudge is the differential oracle: a candidate selection must equal the
// reference selection case-for-case (same cases, same tiers). Empty evidence is
// never a pass; any divergence is reported as the first case with a scrubbed
// replay artifact.
func cselJudge(ref, got []cselCase, prov cselProvenance) cselVerdict {
	surfaces := cselSurfacesOf(got)
	mk := func(reason string, d *cselDivergence) *cselReplayArtifact {
		return &cselReplayArtifact{Provenance: prov, Reason: reason, Divergence: d, Surfaces: surfaces}
	}
	if len(got) == 0 {
		return cselVerdict{Pass: false, Detail: "selector produced no cases — inconclusive evidence is never pass",
			Artifact: mk("no-evidence", &cselDivergence{CaseID: "<none>", Field: "presence", Reference: "selected", Candidate: "absent"})}
	}
	if d := cselFirstDiff(ref, got); d != nil {
		return cselVerdict{Pass: false,
			Detail:   fmt.Sprintf("selection diverged at %s: %s reference=%q candidate=%q — the changed-surface selector routed the diff wrong", d.CaseID, d.Field, d.Reference, d.Candidate),
			Artifact: mk("divergence", d)}
	}
	return cselVerdict{Pass: true, Detail: fmt.Sprintf("selection reproduced the reference: %d cases identical", len(ref))}
}

// --- planted representative defects -----------------------------------------

// cselDroppedCacheRuleStrategy models a selector that forgot to map the cache
// surface — a changed KV-cache file selects nothing, so a cache regression ships
// untested. It under-selects: the cache case goes missing.
func cselDroppedCacheRuleStrategy() cselStrategy {
	var rules []cselRule
	for _, r := range cselRuleTable() {
		if r.surface == cselSurfaceCache {
			continue
		}
		rules = append(rules, r)
	}
	return cselStrategy{rules: rules, expandUnknown: true}
}

// cselNoExpandStrategy models a selector that does NOT expand coverage on an
// unknown change — the "unknown changes expand coverage" acceptance, inverted.
// The sentinel canary case goes missing.
func cselNoExpandStrategy() cselStrategy {
	return cselStrategy{rules: cselRuleTable(), expandUnknown: false}
}

// cselMisTierStrategy models a selector that mis-tiers the cache surface, running
// an expensive nightly soak as if it were a cheap PR check — the case is selected
// but at the wrong tier, so it either over-runs a PR or silently skips nightly.
func cselMisTierStrategy() cselStrategy {
	rules := append([]cselRule(nil), cselRuleTable()...)
	for i := range rules {
		if rules[i].surface == cselSurfaceCache {
			rules[i].tier = cselTierPR
		}
	}
	return cselStrategy{rules: rules, expandUnknown: true}
}
