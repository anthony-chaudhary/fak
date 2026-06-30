package nodeusagepost

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// ParseSnapshot decodes the JSON `fak lab status --json` emits into a fleet.Snapshot.
// It is the inverse of the render path: the CLI reads a snapshot file/stdin and folds
// it into a node-usage card, so the same pure fleet.Snapshot shape is the contract
// between `fak lab status` and `fak nodeusage post`.
func ParseSnapshot(raw []byte) (fleet.Snapshot, error) {
	var s fleet.Snapshot
	if err := json.Unmarshal(raw, &s); err != nil {
		return fleet.Snapshot{}, fmt.Errorf("parse fleet snapshot: %w", err)
	}
	return s, nil
}

// bucket is the honest classification of a node-usage snapshot. The whole point of the
// card is to NOT lie about what the fleet is doing, and the load-bearing distinction is
// SILENT (no box reported — a visibility gap) vs DOWN (a box reported itself not
// serving — a real outage). The fleet model already draws this line: a `down` box is
// "reachable" (knowing it's down is a real observation), an `unknown` box is not. So we
// classify off the per-state COUNTS (ByState), never off snap.Reachable (which a
// down-with-error box drives to 0, hiding a real outage behind "no visibility") and
// never off the fold's conflated "N box(es) down or unreachable" attention TITLE (which
// lumps unknown in with down — the original bug).
type bucket int

const (
	bucketEmpty        bucket = iota // no roster — nothing to report on
	bucketProblem                    // a real, observed problem: a reported `down`, or skew/stale among reporters
	bucketNoVisibility               // every box silent — a visibility gap, NOT an outage
	bucketClean                      // at least one healthy report, no down, no warn (some boxes may still be silent)
)

// signals are the three honest quantities every decision keys off, read once from the
// snapshot integers so grade/verdict/detail/lines can never disagree.
type signals struct {
	down    int  // boxes that reported "down" (ByState[StateDown]) — a real observation
	silent  int  // boxes that gave no real word (ByState[StateUnknown]) — silent or errored
	hasCrit bool // actionable crit attention beyond the down/unreachable visibility bucket
	hasWarn bool // any warn-level attention item: version skew / staleness AMONG reporters
}

func readSignals(snap fleet.Snapshot) signals {
	s := signals{
		down:   snap.ByState[fleet.StateDown],
		silent: snap.ByState[fleet.StateUnknown],
	}
	for _, it := range snap.Attention {
		switch it.Level {
		case "crit":
			if !isVisibilityCrit(it) {
				s.hasCrit = true
			}
		case "warn":
			s.hasWarn = true
		}
	}
	return s
}

func isVisibilityCrit(it fleet.Item) bool {
	return strings.Contains(it.Title, "down or unreachable")
}

// classify buckets the snapshot. Order is load-bearing: a REAL problem (down/warn) is
// checked BEFORE the visibility gap, so an all-down-with-errors fleet (Reachable==0 but
// down>0) is reported as the outage it is, never swallowed by the no-visibility branch.
func classify(snap fleet.Snapshot, sig signals) bucket {
	switch {
	case snap.Total == 0:
		return bucketEmpty
	case sig.down > 0 || sig.hasCrit || sig.hasWarn:
		return bucketProblem
	case sig.silent == snap.Total: // every box silent (and down==0, by the case above)
		return bucketNoVisibility
	default:
		return bucketClean
	}
}

// FromSnapshot folds a fleet.Snapshot (the node-usage signal) into a scoreboard.Update
// — the one card shape every Slack feeder posts. It carries the readiness score, the
// per-state and per-class node counts, and a reporting/visibility line, so #node-usage
// scans the current compute-node usage at a glance.
//
// HONESTY CONTRACT (why this is not the original conflated card):
//   - A fleet where every box is SILENT (no report) is a VISIBILITY GAP, not an outage:
//     it reads a neutral card (no grade, no verdict → the :bar_chart: glyph) with the
//     "populate liveness" guidance the human `fak lab status` footer prints — never a
//     red F/ACTION.
//   - A REAL, OBSERVED problem (a box that reported `down`, or version skew / staleness
//     among reporting boxes) reads ACTION with a clamped grade so it renders red. The
//     grade is clamped because the card renderer picks its glyph from the grade prefix
//     BEFORE the verdict, so an A/B grade would otherwise paint a real down green.
//   - `unknown` (silent) is never counted as `down` and never escalates on its own.
func FromSnapshot(snap fleet.Snapshot, source string) scoreboard.Update {
	sig := readSignals(snap)
	b := classify(snap, sig)
	return scoreboard.Update{
		Title:   "node usage",
		Grade:   gradeFor(b, snap, sig),
		Score:   fmt.Sprintf("%d/%d reachable", snap.Reachable, snap.Total),
		Verdict: verdictFor(b),
		Detail:  detailFor(b, snap, sig),
		Lines:   linesFor(b, snap, sig),
		Source:  source,
	}
}

// gradeFor picks the grade per bucket. It does NOT pass snap.Score through blindly: the
// card renderer's gradeEmoji checks HasPrefix(grade,"A"/"B") before it consults the
// verdict, so an A/B grade with an ACTION verdict would still render green/yellow and
// mask a real problem. So a PROBLEM bucket clamps the grade below B, and the
// NO-VISIBILITY bucket withholds the grade entirely (the neutral glyph).
func gradeFor(b bucket, snap fleet.Snapshot, sig signals) string {
	switch b {
	case bucketEmpty:
		return "N/A" // starts with 'N', so it never prefix-matches the A/B emoji arms
	case bucketNoVisibility:
		return "" // empty grade + empty verdict → the neutral :bar_chart: glyph
	case bucketProblem:
		if sig.down > 0 {
			return "F" // a reported-down box is a real outage — force red
		}
		return clampBelowB(gradeOf(snap.Score)) // warn only: never green/yellow
	default: // bucketClean
		return gradeOf(snap.Score)
	}
}

// clampBelowB caps an A/B grade at C so a real (warn-level) problem can never short-
// circuit the renderer's grade-first emoji to green/yellow. A grade already C or worse
// is left as-is.
func clampBelowB(g string) string {
	if g == "A" || g == "B" {
		return "C"
	}
	return g
}

// verdictFor is ACTION only on an OBSERVED problem; silence is never ACTION. The
// NO-VISIBILITY verdict MUST be empty: a non-empty non-OK verdict would fall through
// gradeEmoji to the red default and re-introduce the false-outage lie.
func verdictFor(b bucket) string {
	switch b {
	case bucketProblem:
		return "ACTION"
	case bucketNoVisibility:
		return "" // neutral, with the empty grade → :bar_chart:
	default: // bucketEmpty, bucketClean
		return "OK"
	}
}

// detailFor is the one-line headline per bucket, built from the snapshot COUNTS — never
// copied from the fold's conflated "N box(es) down or unreachable" title.
func detailFor(b bucket, snap fleet.Snapshot, sig signals) string {
	switch b {
	case bucketEmpty:
		return "no nodes in the roster"
	case bucketProblem:
		if sig.down > 0 {
			return fmt.Sprintf("%d box(es) reported down", sig.down)
		}
		if sig.hasCrit {
			return firstCritTitle(snap)
		}
		return firstWarnTitle(snap) // skew/stale, already honestly worded by the fold
	case bucketNoVisibility:
		return "no live reports — every box reads unknown/errored (not down)"
	default: // bucketClean
		if sig.silent > 0 {
			return fmt.Sprintf("%d of %d reporting; %d silent (unknown, not down)",
				snap.Total-sig.silent, snap.Total, sig.silent)
		}
		return "" // fully clean — Verdict OK already renders the check
	}
}

// firstCritTitle returns the first actionable crit attention item's title, skipping the
// visibility bucket ("down or unreachable") because down is named from ByState and
// unknown/silent is not an outage.
func firstCritTitle(snap fleet.Snapshot) string {
	for _, it := range snap.Attention {
		if it.Level == "crit" && !isVisibilityCrit(it) {
			return it.Title
		}
	}
	return "compute capacity needs attention"
}

// firstWarnTitle returns the first warn-level attention item's title, or a generic
// fallback if none is present (defensive: the PROBLEM bucket only reaches here when
// hasWarn is true, but a snapshot authored without Attention rows could still set it).
func firstWarnTitle(snap fleet.Snapshot) string {
	for _, it := range snap.Attention {
		if it.Level == "warn" {
			return it.Title
		}
	}
	return "reporting boxes need attention (version skew or staleness)"
}

// linesFor renders the at-a-glance breakdown: a reporting/visibility line (so silent is
// never read as down), the per-state counts, the per-class counts, the readiness score,
// and — for a visibility gap — the populate-liveness guidance the human footer prints.
func linesFor(b bucket, snap fleet.Snapshot, sig signals) []string {
	var lines []string

	// Reporting line first (skip for an empty roster, where it is vacuous). Derived from
	// Total-silent, NOT snap.Reachable, so both numbers stay consistent with the silent
	// count even on a down-with-error snapshot.
	if b != bucketEmpty {
		if sig.silent > 0 {
			lines = append(lines, fmt.Sprintf("reporting: %d/%d (%d silent=unknown, not down)",
				snap.Total-sig.silent, snap.Total, sig.silent))
		} else {
			lines = append(lines, fmt.Sprintf("reporting: %d/%d", snap.Total, snap.Total))
		}
	}

	lines = append(lines, capacityLine(snap, sig))
	if line := gpuLine(snap); line != "" {
		lines = append(lines, line)
	}
	lines = append(lines, stateLines(snap)...)
	lines = append(lines, classLines(snap)...)
	if line := versionLine(snap); line != "" {
		lines = append(lines, line)
	}
	lines = append(lines, fmt.Sprintf("readiness: %d", snap.Score))
	lines = append(lines, attentionLines(snap)...)

	if b == bucketNoVisibility {
		lines = append(lines, "next: populate liveness with the private Slack bridge, or `fak lab report --id <box> --state live` from a box that can self-report")
	} else if next := nextLine(b, snap, sig); next != "" {
		lines = append(lines, next)
	}
	return lines
}

// capacityLine names usable and unavailable boxes in one line, so the card answers
// "how much can I schedule onto right now?" instead of only listing raw states.
func capacityLine(snap fleet.Snapshot, sig signals) string {
	usable := snap.ByState[fleet.StateLive] + snap.ByState[fleet.StateIdle] + snap.ByState[fleet.StateDraining]
	parts := []string{fmt.Sprintf("usable capacity: %d/%d boxes", usable, snap.Total)}
	var state []string
	for _, kv := range []struct {
		name string
		n    int
	}{
		{"live", snap.ByState[fleet.StateLive]},
		{"idle", snap.ByState[fleet.StateIdle]},
		{"draining", snap.ByState[fleet.StateDraining]},
	} {
		if kv.n > 0 {
			state = append(state, fmt.Sprintf("%s %d", kv.name, kv.n))
		}
	}
	if len(state) > 0 {
		parts = append(parts, strings.Join(state, ", "))
	}
	var unavailable []string
	if sig.down > 0 {
		unavailable = append(unavailable, fmt.Sprintf("down %d", sig.down))
	}
	if sig.silent > 0 {
		unavailable = append(unavailable, fmt.Sprintf("unknown %d", sig.silent))
	}
	if len(unavailable) > 0 {
		parts = append(parts, "unavailable "+strings.Join(unavailable, ", "))
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return parts[0] + " (" + strings.Join(parts[1:], "; ") + ")"
}

func gpuLine(snap fleet.Snapshot) string {
	g := snap.GPUUtil
	if g == nil {
		return ""
	}
	return fmt.Sprintf("gpu capacity: busy %d/%d, idle %d (%d%% util)",
		g.Busy, g.Total, idleGPUs(g), g.UtilPct)
}

func idleGPUs(g *fleet.GPUStats) int {
	if g == nil || g.Total <= g.Busy {
		return 0
	}
	return g.Total - g.Busy
}

func versionLine(snap fleet.Snapshot) string {
	if snap.ModalVersion == "" {
		if snap.Reachable > 0 {
			return "version: none reported"
		}
		return ""
	}
	off := snap.Reachable - modalVersionCount(snap)
	if off > 0 {
		return fmt.Sprintf("version: %s (%d reachable box(es) other/none)", snap.ModalVersion, off)
	}
	return "version: " + snap.ModalVersion
}

func modalVersionCount(snap fleet.Snapshot) int {
	for _, v := range snap.Versions {
		if v.Key == snap.ModalVersion {
			return v.Count
		}
	}
	return 0
}

func attentionLines(snap fleet.Snapshot) []string {
	var lines []string
	total := 0
	for _, it := range snap.Attention {
		if it.Level != "ok" {
			total++
		}
	}
	for _, it := range snap.Attention {
		if it.Level == "ok" {
			continue
		}
		if len(lines) == 3 {
			lines = append(lines, fmt.Sprintf("attention: +%d more", total-len(lines)))
			break
		}
		line := fmt.Sprintf("attention[%s]: %s", it.Level, it.Title)
		if it.Detail != "" {
			line += " - " + it.Detail
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 && snap.Total > 0 {
		lines = append(lines, "attention: none")
	}
	return lines
}

func nextLine(b bucket, snap fleet.Snapshot, sig signals) string {
	switch b {
	case bucketEmpty:
		return "next: add roster boxes or disable the capacity feed for this fleet"
	case bucketProblem:
		if sig.down > 0 {
			return "next: remove reported-down boxes from scheduling, then restart or re-report them"
		}
		if sig.hasCrit {
			return "next: repack work onto busy GPUs or stop idle-GPU leases"
		}
		return "next: refresh stale/skewed reporters before trusting capacity"
	case bucketClean:
		if sig.silent > 0 {
			return "next: restore reports for silent boxes before counting them as spare capacity"
		}
		if g := snap.GPUUtil; idleGPUs(g) > 0 {
			return "next: schedule new GPU work onto idle capacity"
		}
		return "next: no operator action"
	default:
		return ""
	}
}

// stateLines renders the per-state node counts (live / idle / draining / down /
// unknown) in a fixed operational order so the line is stable across runs.
func stateLines(snap fleet.Snapshot) []string {
	var lines []string
	for _, st := range []fleet.State{
		fleet.StateLive, fleet.StateIdle, fleet.StateDraining, fleet.StateDown, fleet.StateUnknown,
	} {
		if n := snap.ByState[st]; n > 0 {
			lines = append(lines, fmt.Sprintf("%s: %d", st, n))
		}
	}
	return lines
}

// classLines renders the per-class node counts as "class=N" entries, sorted for a
// stable, scannable line independent of count ties.
func classLines(snap fleet.Snapshot) []string {
	out := make([]string, 0, len(snap.ByClass))
	for _, c := range snap.ByClass {
		out = append(out, fmt.Sprintf("%s=%d", c.Key, c.Count))
	}
	sort.Strings(out)
	return out
}

// gradeOf maps the 0-100 fleet readiness score onto an A-F grade — the same coarse
// banding the capacity feed uses, so a node-usage card and a capacity card read the
// same way. Used only for the CLEAN bucket and as the pre-clamp input for a warn-level
// PROBLEM; the NO-VISIBILITY and EMPTY buckets override it (a score of 0 from an
// all-silent fleet is a visibility artifact, not a graded readiness failure).
func gradeOf(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 75:
		return "B"
	case score >= 50:
		return "C"
	case score >= 25:
		return "D"
	default:
		return "F"
	}
}
