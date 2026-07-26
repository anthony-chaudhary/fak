package metrics

import (
	"fmt"
	"sort"
)

// anchor_refusal.go — the live breakpoint-placement outcome monitor and the
// volatile-head refusal alarm (#3622, a leaf of cache-verify epic #3569).
//
// `fak_gateway_cache_breakpoint_placement_total{outcome}` already records whether
// fak spliced a stable-head breakpoint (the star-anchor) or bailed
// (internal/gateway/metrics_observe.go observePlacement, over the closed
// agent.BreakpointReason* vocabulary). A counter alone is a value nobody reads:
// when the head turns volatile across turns the anchor quietly stops EARNING
// caching, the placed bucket simply stops rising, and no signal fires. This file
// is the loop that watches the mix and says so.
//
//   - AnchorPlacementClass is the CLOSED reduction of the placement vocabulary to
//     the only question the alarm asks: did this turn EARN a fak-authored cache
//     breakpoint, was it REFUSED one, or was the turn not evidence either way.
//   - ClassifyAnchorOutcome performs that reduction. It is the false-positive
//     guard: `already_set` is DEFERRED, not refused — a Claude-Code-shaped session
//     is ~100% already_set by design, and pricing that as a refusal would alarm
//     every healthy client on earth.
//   - AnchorRefusalMonitor.Observe folds one placement outcome into a rolling
//     window of DECISIVE turns and returns the verdict, raising the
//     ANCHOR_REFUSED_RISING finding when the refused fraction crosses the
//     threshold.
//   - AnchorRefusalReport / BannerRow are the per-session fold and the guard
//     banner row: the operator-facing witness that the anchor is or is not still
//     earning its keep.
//
// THE RATIO IS OVER DECISIVE TURNS ONLY. earned + refused is the denominator;
// deferred (the client owns the breakpoint), inapplicable (not a Messages body at
// all), and unknown (an outcome outside the vocabulary this monitor was written
// against) are counted for visibility but never move the fraction. A monitor that
// let a non-decisive turn move the ratio would report the client's shape, or a
// vocabulary drift, as fak's anchor failing.
//
// THE FINDING IS EDGE-TRIGGERED. It fires on the crossing from below the
// threshold to at-or-above it, and re-arms only after the fraction falls back
// below — the same "advance the baseline so a persistent condition does not count
// forever" discipline CacheBreakDetector.Observe uses. RISING is a transition;
// a monitor that re-raised it every turn of a long volatile stretch would be a
// stuck horn, not an alarm.
//
// UNKNOWN NEVER ARMS THE ALARM. An out-of-vocabulary outcome is recorded in the
// report (so drift between this monitor and the agent's vocabulary is visible)
// but is not counted as a refusal: alarming on a string this file has never seen
// would be a fabricated verdict, not a detection.
//
// This package stays pure — no engine, no kernel import — so the seam is
// unit-testable and any serve or guard path can fold its own outcomes through it.
// The gateway lowers a verdict onto its live surface at the seam that already
// exists there (internal/gateway/metrics_observe.go), alongside the counter it
// already increments:
//
//	if v := mon.Observe(outcome); v.Finding != "" {
//		// banner row / debug var: v.Banner
//	}
//
// Generation intent: gen/next foundation (cache-verify epic #3569) — the same
// classification its siblings cache_break.go / cache_break_detector.go carry,
// since the cache/context program map
// (docs/notes/GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md) holds live
// provider/cache metrics at gen/next until a live caller corroborates them.
//   - Promotion evidence (toward "now"): a real serve session feeds every
//     observePlacement outcome through Observe, an operator-visible banner row
//     carries the report, and a session whose head genuinely turned volatile is
//     seen raising ANCHOR_REFUSED_RISING on live traffic — at which point the
//     rolling threshold can gate a regression instead of merely narrating one.
//   - Demotion / retirement evidence: if live placement mixes turn out to be
//     bimodal — a session is either ~all placed or ~all already_set, and
//     volatile_head is vanishingly rare or permanently pinned for a given client
//     shape — then a rolling FRACTION earns nothing over a plain nonzero
//     volatile_head counter, and this monitor should be retired down to that bit.
//   - Invalidating assumption: that a rising refused fraction means the anchor
//     STOPPED earning caching. It does not distinguish fak's anchor degrading
//     from the client legitimately changing shape mid-session (a harness that
//     starts sending its own cache_control moves turns from earned to DEFERRED,
//     not refused — but a harness that starts injecting a per-request token into
//     every head span moves them to refused for a reason that is the caller's,
//     not fak's). The finding names WHERE caching stopped being earned, never
//     WHOSE fault it is; attributing that is the anchor transform's job, which
//     this issue holds out of scope.

// AnchorPlacementClass is the closed reduction of one
// `fak_gateway_cache_breakpoint_placement_total{outcome}` bucket to the alarm's
// only question: was a fak-authored cache breakpoint earned, refused, or is this
// turn not evidence either way.
type AnchorPlacementClass string

const (
	// AnchorEarned is a turn where fak spliced the breakpoint: the star-anchor
	// worked and the cache_read it unlocks is fak-authored.
	AnchorEarned AnchorPlacementClass = "earned"
	// AnchorRefused is a turn where fak wanted to anchor and could not — the
	// volatile head this alarm is named for, plus the sibling bails that leave the
	// turn equally uncached.
	AnchorRefused AnchorPlacementClass = "refused"
	// AnchorDeferred is a turn fak deliberately left to the client's own cache
	// (`already_set`). It is NOT a refusal: the breakpoint exists, fak just did not
	// author it, and the turn is cached.
	AnchorDeferred AnchorPlacementClass = "deferred"
	// AnchorInapplicable is a turn the anchor could not apply to at all — the body
	// is not a JSON Messages request, so there was no placement question to answer.
	AnchorInapplicable AnchorPlacementClass = "inapplicable"
	// AnchorUnknown is an outcome outside the vocabulary this monitor was written
	// against. Recorded so drift is visible, never counted as a refusal.
	AnchorUnknown AnchorPlacementClass = "unknown"
)

// AnchorFindingRefusedRising is the finding this monitor raises when the refused
// fraction of decisive turns crosses the rolling threshold: the anchor has
// stopped earning caching often enough to be worth an operator's attention.
const AnchorFindingRefusedRising = "ANCHOR_REFUSED_RISING"

// AnchorOutcomePlaced is the placement bucket the gateway records for an actual
// splice. The agent-side vocabulary spells that outcome as the empty reason; the
// metric spells it "placed", and ClassifyAnchorOutcome accepts both.
const AnchorOutcomePlaced = "placed"

// anchorOutcomeClasses is the closed outcome -> class map, mirroring
// internal/agent's BreakpointReason* vocabulary. It is deliberately explicit
// rather than pattern-matched: a new bail reason should land here as a considered
// classification, not be silently absorbed as a refusal by a default branch.
var anchorOutcomeClasses = map[string]AnchorPlacementClass{
	"":                AnchorEarned,       // agent.BreakpointReasonNone — a breakpoint was spliced
	"placed":          AnchorEarned,       // the metric's spelling of the same outcome
	"volatile_head":   AnchorRefused,      // every cacheable head span carries a per-request token
	"no_stable_head":  AnchorRefused,      // no system[] or tools[] block to anchor on
	"splice_failed":   AnchorRefused,      // the target block is not a spliceable object
	"redecode_failed": AnchorRefused,      // the spliced body failed to re-decode
	"already_set":     AnchorDeferred,     // the client's own cache_control layout — respected, not refused
	"non_json":        AnchorInapplicable, // not a JSON object body
}

// ClassifyAnchorOutcome reduces one placement outcome to its class. An outcome
// outside the closed vocabulary folds to AnchorUnknown, which is recorded but
// never counted as a refusal.
func ClassifyAnchorOutcome(outcome string) AnchorPlacementClass {
	if c, ok := anchorOutcomeClasses[outcome]; ok {
		return c
	}
	return AnchorUnknown
}

// Decisive reports whether a class is evidence about whether fak's anchor earned
// caching. Only earned and refused are: they are the numerator and denominator of
// the rolling fraction.
func (c AnchorPlacementClass) Decisive() bool {
	return c == AnchorEarned || c == AnchorRefused
}

// Default rolling-window tuning. The window is short enough that a session which
// turns volatile mid-conversation alarms within a few turns, and the minimum
// sample is high enough that one unlucky bail cannot fire a 100% fraction.
const (
	// DefaultAnchorWindow is how many decisive turns the rolling fraction spans.
	DefaultAnchorWindow = 8
	// DefaultAnchorMinSamples is the fewest decisive turns that may raise the
	// finding, so a cold session cannot alarm off a single refusal.
	DefaultAnchorMinSamples = 4
	// DefaultAnchorThreshold is the refused fraction at or above which the anchor
	// is judged to have stopped earning caching.
	DefaultAnchorThreshold = 0.5
)

// AnchorRefusalThresholds is the rolling tuning the alarm is armed with. The zero
// value is valid and normalizes to the documented defaults, so a caller that just
// wants the monitor gets a sane loop without restating the tuning.
//
// It is deliberately NOT named a "policy": in this repo that word is reserved for
// the capability/authorization surface (the security floor, the on-disk manifest,
// the compiled table). These are three numbers that tune one alarm.
type AnchorRefusalThresholds struct {
	// Window is the number of most-recent DECISIVE turns the fraction spans.
	Window int `json:"window"`
	// MinSamples is the fewest decisive turns in the window that may raise the
	// finding.
	MinSamples int `json:"min_samples"`
	// Threshold is the refused fraction (0..1) at or above which the finding is
	// raised.
	Threshold float64 `json:"threshold"`
}

// normalize folds an under-specified or out-of-range tuning onto the defaults. A
// threshold outside (0,1] cannot arm a meaningful alarm — 0 would fire on a clean
// session — so it folds rather than silently arming a horn that is always on.
func (p AnchorRefusalThresholds) normalize() AnchorRefusalThresholds {
	if p.Window <= 0 {
		p.Window = DefaultAnchorWindow
	}
	if p.MinSamples <= 0 {
		p.MinSamples = DefaultAnchorMinSamples
	}
	if p.MinSamples > p.Window {
		p.MinSamples = p.Window
	}
	if p.Threshold <= 0 || p.Threshold > 1 {
		p.Threshold = DefaultAnchorThreshold
	}
	return p
}

// AnchorRefusalVerdict is the answer for one observed placement outcome: how the
// turn classified, where the rolling fraction now stands, and whether this turn is
// the crossing that raises the finding.
type AnchorRefusalVerdict struct {
	// Outcome is the raw placement bucket that was observed.
	Outcome string `json:"outcome"`
	// Class is the closed reduction of that outcome.
	Class AnchorPlacementClass `json:"class"`
	// Decisive reports whether this turn moved the rolling window.
	Decisive bool `json:"decisive"`
	// Finding is AnchorFindingRefusedRising on the turn that crosses the
	// threshold, and empty on every other turn — including the turns of a stretch
	// that is already alarming.
	Finding string `json:"finding,omitempty"`
	// RefusedFraction is the refused share of the decisive turns currently in the
	// window, 0 when the window is empty.
	RefusedFraction float64 `json:"refused_fraction"`
	// WindowRefused / WindowDecisive are the fraction's raw terms, so a reader
	// never has to trust a float alone.
	WindowRefused  int `json:"window_refused"`
	WindowDecisive int `json:"window_decisive"`
	// TopRefusal is the dominant refusal outcome in the current window, empty when
	// nothing in the window was refused. It is the first thing an operator wants
	// after being told the anchor stopped earning.
	TopRefusal string `json:"top_refusal,omitempty"`
	// Alarmed reports whether the monitor is currently in the raised state, which
	// stays true across the turns after the crossing.
	Alarmed bool `json:"alarmed"`
	// Banner is the operator row for the turn that raised the finding, empty
	// otherwise.
	Banner string `json:"banner,omitempty"`
	// Reason is the short human-readable explanation, matching the Reason-field
	// convention the sibling cache primitives use.
	Reason string `json:"reason"`
	// ObservedTurn is the 1-indexed count of Observe calls on this monitor.
	ObservedTurn int `json:"observed_turn"`
}

// AnchorRefusalMonitor tracks one session's placement-outcome mix and raises the
// volatile-head alarm. Its retained state is a bounded ring of outcome labels and
// a handful of counters — no request bytes — so arming it across a fleet costs no
// transcript retention.
//
// It is a single-session, turn-at-a-time state machine and is not safe for
// concurrent use without external synchronization — the same contract as
// CacheBreakDetector.
type AnchorRefusalMonitor struct {
	limits AnchorRefusalThresholds

	window  []string // the most recent decisive outcomes, oldest first
	turns   int
	alarmed bool

	byOutcome map[string]int
	byClass   map[AnchorPlacementClass]int
	findings  int
}

// NewAnchorRefusalMonitor builds a monitor under the given rolling tuning. The
// zero value normalizes to the documented defaults.
func NewAnchorRefusalMonitor(p AnchorRefusalThresholds) *AnchorRefusalMonitor {
	return &AnchorRefusalMonitor{
		limits:    p.normalize(),
		byOutcome: map[string]int{},
		byClass:   map[AnchorPlacementClass]int{},
	}
}

// Thresholds reports the normalized rolling tuning this monitor is armed with.
func (m *AnchorRefusalMonitor) Thresholds() AnchorRefusalThresholds { return m.limits }

// Observe folds one placement outcome into the session and returns the verdict.
//
// Non-decisive turns are recorded in the report but leave the rolling window and
// the alarm state exactly as they were: a run of `already_set` turns can neither
// raise the finding nor clear an alarm that a run of volatile heads raised.
func (m *AnchorRefusalMonitor) Observe(outcome string) AnchorRefusalVerdict {
	if m == nil {
		return AnchorRefusalVerdict{}
	}
	m.turns++
	class := ClassifyAnchorOutcome(outcome)
	m.byOutcome[outcome]++
	m.byClass[class]++

	v := AnchorRefusalVerdict{
		Outcome:      outcome,
		Class:        class,
		Decisive:     class.Decisive(),
		ObservedTurn: m.turns,
	}

	if !v.Decisive {
		v.RefusedFraction, v.WindowRefused, v.WindowDecisive = m.fraction()
		v.TopRefusal = m.topRefusal()
		v.Alarmed = m.alarmed
		v.Reason = fmt.Sprintf("outcome %q is %s; not evidence about whether the anchor earned caching", outcome, class)
		return v
	}

	m.window = append(m.window, outcome)
	if len(m.window) > m.limits.Window {
		m.window = m.window[len(m.window)-m.limits.Window:]
	}

	v.RefusedFraction, v.WindowRefused, v.WindowDecisive = m.fraction()
	v.TopRefusal = m.topRefusal()

	switch {
	case v.WindowDecisive < m.limits.MinSamples:
		v.Reason = fmt.Sprintf("%d/%d decisive turns — below the %d-sample floor to judge the anchor",
			v.WindowDecisive, m.limits.Window, m.limits.MinSamples)
	case v.RefusedFraction >= m.limits.Threshold:
		if !m.alarmed {
			m.alarmed = true
			m.findings++
			v.Finding = AnchorFindingRefusedRising
			v.Banner = m.banner(v)
			v.Reason = fmt.Sprintf("refused fraction %.2f crossed the %.2f threshold — the anchor stopped earning caching",
				v.RefusedFraction, m.limits.Threshold)
		} else {
			v.Reason = fmt.Sprintf("refused fraction %.2f still at or above the %.2f threshold (already raised)",
				v.RefusedFraction, m.limits.Threshold)
		}
	default:
		if m.alarmed {
			m.alarmed = false
			v.Reason = fmt.Sprintf("refused fraction %.2f fell back below the %.2f threshold — alarm re-armed",
				v.RefusedFraction, m.limits.Threshold)
		} else {
			v.Reason = fmt.Sprintf("refused fraction %.2f below the %.2f threshold — the anchor is earning caching",
				v.RefusedFraction, m.limits.Threshold)
		}
	}
	v.Alarmed = m.alarmed
	return v
}

// fraction reports the refused share of the current window alongside its raw
// terms. An empty window is 0/0, not a divide.
func (m *AnchorRefusalMonitor) fraction() (float64, int, int) {
	decisive := len(m.window)
	if decisive == 0 {
		return 0, 0, 0
	}
	refused := 0
	for _, o := range m.window {
		if ClassifyAnchorOutcome(o) == AnchorRefused {
			refused++
		}
	}
	return float64(refused) / float64(decisive), refused, decisive
}

// topRefusal names the dominant refusal outcome in the current window. Ties break
// by outcome name so the witness is deterministic across runs.
func (m *AnchorRefusalMonitor) topRefusal() string {
	counts := map[string]int{}
	for _, o := range m.window {
		if ClassifyAnchorOutcome(o) == AnchorRefused {
			counts[o]++
		}
	}
	top, best := "", 0
	for o, n := range counts {
		if n > best || (n == best && o < top) {
			top, best = o, n
		}
	}
	return top
}

// banner builds the operator row for a raised finding.
func (m *AnchorRefusalMonitor) banner(v AnchorRefusalVerdict) string {
	cause := v.TopRefusal
	if cause == "" {
		cause = "unattributed"
	}
	return fmt.Sprintf("%s: %d/%d recent anchor placements refused (%.0f%% >= %.0f%%, top=%s) — breakpoint no longer earning caching",
		AnchorFindingRefusedRising, v.WindowRefused, v.WindowDecisive,
		v.RefusedFraction*100, m.limits.Threshold*100, cause)
}

// AnchorOutcomeTally is one placement outcome's fold: its class and how many
// turns carried it.
type AnchorOutcomeTally struct {
	Outcome string               `json:"outcome"`
	Class   AnchorPlacementClass `json:"class"`
	Turns   int                  `json:"turns"`
}

// AnchorRefusalReport is the per-session operator readout: the whole
// placement-outcome mix, where the rolling fraction ended, and how many times the
// alarm was raised.
type AnchorRefusalReport struct {
	// Turns is every observed placement attempt, decisive or not.
	Turns int `json:"turns"`
	// Earned / Refused / Deferred / Inapplicable / Unknown are the session totals
	// per class.
	Earned       int `json:"earned"`
	Refused      int `json:"refused"`
	Deferred     int `json:"deferred"`
	Inapplicable int `json:"inapplicable"`
	Unknown      int `json:"unknown"`
	// RefusedFraction / WindowRefused / WindowDecisive are the rolling window as
	// it stands at the end of the session.
	RefusedFraction float64 `json:"refused_fraction"`
	WindowRefused   int     `json:"window_refused"`
	WindowDecisive  int     `json:"window_decisive"`
	// TopRefusal is the dominant refusal outcome in that window.
	TopRefusal string `json:"top_refusal,omitempty"`
	// Alarmed is whether the session ended in the raised state.
	Alarmed bool `json:"alarmed"`
	// Findings is how many times ANCHOR_REFUSED_RISING was raised: crossings, not
	// alarming turns.
	Findings int `json:"findings"`
	// ByOutcome is the full mix in deterministic order.
	ByOutcome []AnchorOutcomeTally `json:"by_outcome"`
	// Thresholds is the rolling tuning the session was judged against, so a banner
	// row is readable without knowing how the monitor was armed.
	Thresholds AnchorRefusalThresholds `json:"thresholds"`
}

// Report folds the session's observed outcomes into the operator readout.
func (m *AnchorRefusalMonitor) Report() AnchorRefusalReport {
	if m == nil {
		return AnchorRefusalReport{Thresholds: AnchorRefusalThresholds{}.normalize()}
	}
	r := AnchorRefusalReport{
		Turns:        m.turns,
		Earned:       m.byClass[AnchorEarned],
		Refused:      m.byClass[AnchorRefused],
		Deferred:     m.byClass[AnchorDeferred],
		Inapplicable: m.byClass[AnchorInapplicable],
		Unknown:      m.byClass[AnchorUnknown],
		TopRefusal:   m.topRefusal(),
		Alarmed:      m.alarmed,
		Findings:     m.findings,
		Thresholds:   m.limits,
	}
	r.RefusedFraction, r.WindowRefused, r.WindowDecisive = m.fraction()
	r.ByOutcome = make([]AnchorOutcomeTally, 0, len(m.byOutcome))
	for o, n := range m.byOutcome {
		r.ByOutcome = append(r.ByOutcome, AnchorOutcomeTally{Outcome: o, Class: ClassifyAnchorOutcome(o), Turns: n})
	}
	sort.SliceStable(r.ByOutcome, func(i, j int) bool {
		if r.ByOutcome[i].Turns != r.ByOutcome[j].Turns {
			return r.ByOutcome[i].Turns > r.ByOutcome[j].Turns
		}
		return r.ByOutcome[i].Outcome < r.ByOutcome[j].Outcome
	})
	return r
}

// BannerRow is the guard banner line for the session: one row naming whether the
// anchor is still earning caching. A session that never made a decisive placement
// says so rather than reporting a 0% refusal it did not measure.
func (r AnchorRefusalReport) BannerRow() string {
	limits := r.Thresholds.normalize()
	switch {
	case r.Turns == 0:
		return "cache anchor: no placement attempts observed"
	case r.WindowDecisive == 0:
		return fmt.Sprintf("cache anchor: %d turn(s), none decisive (deferred=%d inapplicable=%d unknown=%d) — nothing to judge",
			r.Turns, r.Deferred, r.Inapplicable, r.Unknown)
	case r.Alarmed:
		cause := r.TopRefusal
		if cause == "" {
			cause = "unattributed"
		}
		return fmt.Sprintf("cache anchor: %s — %d/%d recent placements refused (%.0f%% >= %.0f%%, top=%s), %d finding(s)",
			AnchorFindingRefusedRising, r.WindowRefused, r.WindowDecisive,
			r.RefusedFraction*100, limits.Threshold*100, cause, r.Findings)
	default:
		return fmt.Sprintf("cache anchor: earning — %d/%d recent placements refused (%.0f%% < %.0f%%), %d finding(s) this session",
			r.WindowRefused, r.WindowDecisive, r.RefusedFraction*100, limits.Threshold*100, r.Findings)
	}
}
