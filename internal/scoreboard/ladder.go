package scoreboard

import (
	"fmt"
	"sort"
	"strings"
)

// The quality-ladder dashboard (issue #4585, epic #4509) is the PUBLISH surface
// for the "missing middle" validation ladder: the layer of evidence between
// primitive correctness tests (too local to catch a fluent-but-wrong decode) and
// end benchmarks (too coarse to localize an engine regression). The quality spine
// (internal/quality) PRODUCES per-case results; this renderer PUBLISHES a folded
// status board over a set of them to the scoreboard channel.
//
// It is deliberately decoupled from the quality engine, exactly like Update is
// decoupled from where its numbers came from (FromPayload): the dashboard takes a
// plain []LadderCase and folds it, so a CLI can map real quality.Result values in
// without this render layer importing the compute layer. That keeps the leaf
// hermetic and the render logic unit-testable with hand-built fixtures.

// LadderStatus is the closed set of outcomes the dashboard renders (acceptance
// criterion 1). It is a closed vocabulary on purpose: a case whose evidence does
// not clearly say "pass" lands in one of the non-green states below rather than
// being quietly counted green.
type LadderStatus string

const (
	StatusPass         LadderStatus = "pass"         // evidence present, fresh, and agreed with the reference
	StatusFail         LadderStatus = "fail"         // a real divergence, localized and replayable
	StatusStale        LadderStatus = "stale"        // evidence exists but aged out of its freshness window
	StatusSkipped      LadderStatus = "skipped"      // deliberately not run for this revision/tier
	StatusInconclusive LadderStatus = "inconclusive" // ran but cannot be trusted (missing provenance / unlocalized fail)
	StatusNoData       LadderStatus = "no-data"      // no result recorded at all
)

// LadderTier is the explicit cost tier a case is assigned to (acceptance
// criterion 4): a cheap PR check, a heavier nightly accuracy/hardware suite, or a
// release-gating run. Assigning every case to a tier is what lets the board split
// "cheap and always-on" from "expensive and periodic", the exact separation the
// top stacks (vLLM/SGLang) make.
type LadderTier string

const (
	TierPR      LadderTier = "pr"
	TierNightly LadderTier = "nightly"
	TierRelease LadderTier = "release"
)

// LadderDivergence pins the FIRST actionable divergence of a failing case
// (acceptance criterion 3). It mirrors quality.Divergence — a token index and the
// reference-vs-engine token there — so a defect localizes to a step rather than
// being observed only in prose. Detail carries a length/whole-case divergence when
// there is no single token index.
type LadderDivergence struct {
	Index     int    `json:"index"`
	Reference string `json:"reference"`
	Engine    string `json:"engine"`
	Detail    string `json:"detail"`
}

// String renders the divergence as one actionable line.
func (d LadderDivergence) String() string {
	if d.Detail != "" {
		return d.Detail
	}
	return fmt.Sprintf("token %d: reference %q, engine %q", d.Index, d.Reference, d.Engine)
}

// LadderCase is one row of the dashboard: the provenance that makes a result
// attributable and reproducible (acceptance criterion 2), the tier + cost that
// place it on the ladder (criterion 4), and the evidence that decides its status.
//
// The producer records a Status, but the dashboard NEVER trusts that field
// blindly — Effective re-derives the trustworthy status so a green claim that is
// not backed by provenance (or by localization, on a failure) cannot surface as
// pass.
type LadderCase struct {
	ID string `json:"id"`

	// Provenance (criterion 2): every case names what produced its evidence.
	Model     string `json:"model"`               // e.g. "qwen3-0.6b"
	Tokenizer string `json:"tokenizer"`           // e.g. "qwen3-bpe"
	Engine    string `json:"engine"`              // engine/backend under test, e.g. "fak-cpu"
	Seed      int64  `json:"seed,omitempty"`      // stochastic seed; 0 => determinism comes from Oracle instead
	Oracle    string `json:"oracle,omitempty"`    // deterministic oracle name (seed-free determinism), e.g. "greedy-token-diff"
	Revision  string `json:"revision"`            // code/module revision the run was built from
	Tolerance string `json:"tolerance,omitempty"` // tolerance provenance, e.g. "exact" or "atol=1e-3"
	Baseline  string `json:"baseline,omitempty"`  // baseline provenance, e.g. "golden@abc123"

	// Tier + cost (criterion 4).
	Tier LadderTier `json:"tier"`
	Cost string     `json:"cost"` // documented runtime/resource cost, e.g. "0.4s / 1 CPU"

	// Evidence.
	Status          LadderStatus      `json:"status"`                     // producer-declared outcome; re-derived by Effective
	AgeSeconds      int64             `json:"age_seconds,omitempty"`      // age of the evidence, for freshness
	Prev            LadderStatus      `json:"prev,omitempty"`             // prior run's status, for trend/regression detection
	FirstDivergence *LadderDivergence `json:"first_divergence,omitempty"` // set on a fail: the localized first departure
	Replay          string            `json:"replay,omitempty"`           // reference to the scrubbed, replay-complete failure artifact
}

// hasProvenance reports whether the case carries the full provenance criterion 2
// requires. A case missing any of it cannot be trusted as evidence, so Effective
// downgrades it to inconclusive rather than honoring a declared pass.
func (c LadderCase) hasProvenance() bool {
	if strings.TrimSpace(c.Model) == "" || strings.TrimSpace(c.Tokenizer) == "" ||
		strings.TrimSpace(c.Engine) == "" || strings.TrimSpace(c.Revision) == "" {
		return false
	}
	// tolerance/baseline provenance: at least one must be named.
	if strings.TrimSpace(c.Tolerance) == "" && strings.TrimSpace(c.Baseline) == "" {
		return false
	}
	// "seed OR deterministic oracle": exactly the two allowed determinism sources.
	if c.Seed == 0 && strings.TrimSpace(c.Oracle) == "" {
		return false
	}
	// A case must be placed on the ladder — an un-tiered case has no cost story.
	if strings.TrimSpace(string(c.Tier)) == "" {
		return false
	}
	return true
}

// Effective re-derives the trustworthy status of a case, independent of what the
// producer DECLARED. It is the honesty gate behind acceptance criterion 3
// ("missing or inconclusive evidence is never pass"): the producer's Status is
// only honored when the case actually backs it up.
//
//   - No status at all           -> no-data (never pass).
//   - A non-green declared status -> passes through unchanged (skipped/inconclusive/…).
//   - A declared pass or fail     -> must carry full provenance, else inconclusive.
//   - A declared fail             -> must localize (FirstDivergence) AND ship a
//     scrubbed Replay artifact, else it is an unactionable claim -> inconclusive.
//   - A declared pass past its freshness window -> stale (an aged pass is not live).
func (c LadderCase) Effective(staleAfterSeconds int64) LadderStatus {
	switch c.Status {
	case "":
		return StatusNoData
	case StatusNoData, StatusSkipped, StatusInconclusive, StatusStale:
		return c.Status
	}
	// Only pass and fail remain — both are evidence CLAIMS that must be backed.
	if !c.hasProvenance() {
		return StatusInconclusive
	}
	if c.Status == StatusFail {
		if c.FirstDivergence == nil || strings.TrimSpace(c.Replay) == "" {
			// A failure that does not localize or does not emit its replay
			// artifact is not actionable evidence — refuse it rather than let a
			// bare "fail" stand without the proof criterion 3 demands.
			return StatusInconclusive
		}
		return StatusFail
	}
	// c.Status == StatusPass
	if staleAfterSeconds > 0 && c.AgeSeconds > staleAfterSeconds {
		return StatusStale
	}
	return StatusPass
}

// LadderCaseView is a case as the dashboard resolved it: the honest Effective
// status alongside what the producer Declared, so a downgraded green (the planted
// defect the honesty gate catches) is visible rather than silently absorbed.
type LadderCaseView struct {
	ID              string            `json:"id"`
	Status          LadderStatus      `json:"status"`   // effective (honest) status
	Declared        LadderStatus      `json:"declared"` // what the producer claimed
	Tier            LadderTier        `json:"tier"`
	Cost            string            `json:"cost"`
	Revision        string            `json:"revision"`
	FirstDivergence *LadderDivergence `json:"first_divergence,omitempty"`
	Replay          string            `json:"replay,omitempty"`
}

// LadderInput is the dashboard's single input shape (the ladder equivalent of the
// scorecard Payload that FromPayload folds): the cases to render, a title, and the
// freshness window past which a pass renders stale.
type LadderInput struct {
	Title      string       `json:"title"`
	Cases      []LadderCase `json:"cases"`
	StaleAfter int64        `json:"stale_after_seconds,omitempty"` // 0 => no freshness demotion
}

// Dashboard is the folded quality-ladder view (the scope: status, trends, slices,
// freshness, regressions, covered revisions). It is a pure fold of LadderInput —
// same input, same Dashboard — so it renders identically in CI and on a host.
type Dashboard struct {
	Total        int                             `json:"total"`
	Counts       map[LadderStatus]int            `json:"counts"`       // status -> count (all six states, always present)
	Verdict      string                          `json:"verdict"`      // OK | ACTION
	Regressions  []string                        `json:"regressions"`  // case IDs that went pass -> non-pass
	Improvements []string                        `json:"improvements"` // case IDs that went non-pass -> pass
	Stale        []string                        `json:"stale"`        // case IDs whose evidence is stale
	Revisions    []string                        `json:"revisions"`    // covered code/module revisions (deduped, sorted)
	Slices       map[string]map[LadderStatus]int `json:"slices"`       // dimension value -> status counts, e.g. "tier:pr"
	Cases        []LadderCaseView                `json:"cases"`
}

// allStatuses is the closed render vocabulary in display order. The dashboard
// always reports a count for EACH — a state with zero cases renders as 0, so a
// reader can tell "no failures" (fail: 0) from "we forgot to check for failures".
var allStatuses = []LadderStatus{
	StatusPass, StatusFail, StatusStale, StatusSkipped, StatusInconclusive, StatusNoData,
}

// Render folds the input into a Dashboard: it re-derives each case's honest
// status, tallies the six states, buckets slices by tier and engine, tracks
// covered revisions, and diffs each case against its prior status for
// trends/regressions.
func (in LadderInput) Render() Dashboard {
	d := Dashboard{
		Counts: map[LadderStatus]int{},
		Slices: map[string]map[LadderStatus]int{},
	}
	// Seed every state so the board reports a full, honest zero-line.
	for _, s := range allStatuses {
		d.Counts[s] = 0
	}
	revSet := map[string]bool{}
	for _, c := range in.Cases {
		eff := c.Effective(in.StaleAfter)
		d.Total++
		d.Counts[eff]++
		d.sliceInc("tier:"+string(c.Tier), eff)
		d.sliceInc("engine:"+c.Engine, eff)
		if r := strings.TrimSpace(c.Revision); r != "" {
			revSet[r] = true
		}
		if c.Prev != "" {
			wasPass := c.Prev == StatusPass
			nowPass := eff == StatusPass
			switch {
			case wasPass && !nowPass:
				d.Regressions = append(d.Regressions, c.ID)
			case !wasPass && nowPass:
				d.Improvements = append(d.Improvements, c.ID)
			}
		}
		if eff == StatusStale {
			d.Stale = append(d.Stale, c.ID)
		}
		d.Cases = append(d.Cases, LadderCaseView{
			ID:              c.ID,
			Status:          eff,
			Declared:        c.Status,
			Tier:            c.Tier,
			Cost:            c.Cost,
			Revision:        c.Revision,
			FirstDivergence: c.FirstDivergence,
			Replay:          c.Replay,
		})
	}
	d.Revisions = sortedKeys(revSet)
	d.Verdict = d.verdict()
	return d
}

func (d *Dashboard) sliceInc(key string, s LadderStatus) {
	m := d.Slices[key]
	if m == nil {
		m = map[LadderStatus]int{}
		d.Slices[key] = m
	}
	m[s]++
}

// verdict is OK only when there is evidence AND nothing needs attention. Any
// fail, inconclusive, no-data, or stale case — or any regression — forces ACTION.
// A board with zero cases is ACTION, not OK: no evidence is never a green board
// (the ladder-level restatement of "missing evidence is never pass").
func (d Dashboard) verdict() string {
	if d.Total == 0 {
		return "ACTION"
	}
	needsAttention := d.Counts[StatusFail] + d.Counts[StatusInconclusive] +
		d.Counts[StatusNoData] + d.Counts[StatusStale]
	if needsAttention > 0 || len(d.Regressions) > 0 {
		return "ACTION"
	}
	return "OK"
}

// FirstActionable returns the first failing case's localized divergence and its
// replay artifact — the single "here is where it broke, here is how to reproduce
// it" line the board leads its next-step with. ok is false when nothing failed.
func (d Dashboard) FirstActionable() (view LadderCaseView, ok bool) {
	for _, v := range d.Cases {
		if v.Status == StatusFail {
			return v, true
		}
	}
	return LadderCaseView{}, false
}

// Summary renders the dashboard as a compact human-readable board — the text a
// test captures and a reader scans. It always prints every status count, the
// covered-revision set, any regressions, and, on a failure, the first actionable
// divergence plus its replay artifact.
func (d Dashboard) Summary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "quality ladder — %s (%d case(s))\n", d.Verdict, d.Total)
	parts := make([]string, 0, len(allStatuses))
	for _, s := range allStatuses {
		parts = append(parts, fmt.Sprintf("%s=%d", s, d.Counts[s]))
	}
	fmt.Fprintf(&b, "status: %s\n", strings.Join(parts, "  "))
	if len(d.Revisions) > 0 {
		fmt.Fprintf(&b, "revisions: %s\n", strings.Join(d.Revisions, ", "))
	}
	if len(d.Regressions) > 0 {
		fmt.Fprintf(&b, "regressions: %s\n", strings.Join(d.Regressions, ", "))
	}
	if len(d.Improvements) > 0 {
		fmt.Fprintf(&b, "improvements: %s\n", strings.Join(d.Improvements, ", "))
	}
	if len(d.Stale) > 0 {
		fmt.Fprintf(&b, "stale: %s\n", strings.Join(d.Stale, ", "))
	}
	if v, ok := d.FirstActionable(); ok {
		fmt.Fprintf(&b, "first actionable: case %s", v.ID)
		if v.FirstDivergence != nil {
			fmt.Fprintf(&b, " — %s", v.FirstDivergence.String())
		}
		if v.Replay != "" {
			fmt.Fprintf(&b, " (replay: %s)", v.Replay)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// ToUpdate folds the dashboard into an Update so it publishes on the SAME
// scoreboard surface as every other card (chat.postMessage via Text/Blocks). The
// verdict drives the channel glyph; the first actionable divergence + replay
// becomes the prominent next-step; the per-status counts and covered revisions
// ride the stat line. source names who posted (e.g. "ci" | "nightly").
func (in LadderInput) ToUpdate(source string) Update {
	d := in.Render()
	u := Update{
		Title:   firstNonBlank(in.Title, "quality ladder"),
		Verdict: d.Verdict,
		Source:  source,
		Detail:  d.detailLine(),
	}
	for _, s := range allStatuses {
		u.Lines = append(u.Lines, fmt.Sprintf("%s: %d", s, d.Counts[s]))
	}
	if len(d.Revisions) > 0 {
		u.Lines = append(u.Lines, "revs: "+strings.Join(d.Revisions, ","))
	}
	if v, ok := d.FirstActionable(); ok {
		next := fmt.Sprintf("first actionable: case %s", v.ID)
		if v.FirstDivergence != nil {
			next += " — " + v.FirstDivergence.String()
		}
		if v.Replay != "" {
			next += fmt.Sprintf(" · replay %s", v.Replay)
		}
		u.NextStep = next
	}
	return u
}

// detailLine is the one-line finding under the headline: the verdict framed by the
// counts that most need attention.
func (d Dashboard) detailLine() string {
	if d.Total == 0 {
		return "no cases recorded — an empty ladder is never green"
	}
	fail := d.Counts[StatusFail]
	inc := d.Counts[StatusInconclusive] + d.Counts[StatusNoData]
	stale := d.Counts[StatusStale]
	if fail == 0 && inc == 0 && stale == 0 && len(d.Regressions) == 0 {
		return fmt.Sprintf("all %d case(s) passed", d.Counts[StatusPass])
	}
	return fmt.Sprintf("%d fail · %d inconclusive/no-data · %d stale · %d regression(s)",
		fail, inc, stale, len(d.Regressions))
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func firstNonBlank(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}
