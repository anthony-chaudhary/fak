package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/claimcheck"
	"github.com/anthony-chaudhary/fak/internal/fleetbottleneck"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// info_fleet.go — rendering the live fleet-CONTROL-PANE aggregate (the cross-MACHINE
// health fold, twin of `fak fleetpane fleet`) in the `fak info` pane. All pure
// (SessionFleet in, strings out); the data arrives via /debug/vars (the guard's
// TTL-cached, snapshot-only fleet provider), so this stays payload-free. Distinct from
// info_endpoints.go, which shows THIS session's own accounts+nodes.

// fleet-verdict glyphs: healthy fleet, a fleet needing attention, and an unknown/empty
// verdict. Mirrors the account-chip glyph vocabulary so the pane reads consistently.
const (
	guardFleetGlyphOK      = "●"
	guardFleetGlyphAction  = "⊘"
	guardFleetGlyphUnknown = "○"
)

// --- metric provenance (issue #3605) ----------------------------------------
//
// Every fleet-status FIGURE the pane renders carries a provenance label naming WHO authored
// the number, reusing the closed WITNESSED/OBSERVED vocabulary from internal/claimcheck (the
// same axis net-true-value and the conflation scorecard draw). A value POLLED from the peer
// /debug/vars snapshot fold is relayed from another machine, so it is OBSERVED; a value fak
// reads straight from local git (e.g. a lands-shipped count) is one fak authored, so it is
// WITNESSED. Arithmetic never upgrades provenance. The pane suffixes each figure with a
// per-provenance glyph so a reader can tell fak's own witnessed numbers from relayed ones.

// fleetMetricSource names WHERE a fleet figure is sourced from; the source FIXES its
// provenance (see fleetSourceProvenance). The set is closed.
type fleetMetricSource int

const (
	// fleetSourcePoll — relayed from a peer /debug/vars snapshot fold ⇒ OBSERVED.
	fleetSourcePoll fleetMetricSource = iota
	// fleetSourceGit — read from local git by fak itself (e.g. lands shipped) ⇒ WITNESSED.
	fleetSourceGit
)

// fleetSourceProvenance maps a figure's source to its closed provenance label: poll-sourced
// is OBSERVED (relayed), git-sourced is WITNESSED (fak-authored). Total over the closed set.
func fleetSourceProvenance(src fleetMetricSource) claimcheck.Provenance {
	if src == fleetSourceGit {
		return claimcheck.Witnessed
	}
	return claimcheck.Observed
}

// fleetMetricProvenance declares the source (hence provenance) of every SessionFleet FIGURE
// field the pane renders. The gate test (info_fleet_test.go) reflects over
// gateway.SessionFleet and FAILS if an integer figure field is added without a declaration
// here, so a new fleet metric can never ship unlabeled. Every field is poll-sourced today
// (the /debug/vars fleet fold relays peer snapshots); a future git-witnessed figure would
// declare fleetSourceGit and render with the WITNESSED glyph.
var fleetMetricProvenance = map[string]fleetMetricSource{
	"Machines":          fleetSourcePoll,
	"Stale":             fleetSourcePoll,
	"Action":            fleetSourcePoll,
	"Sessions":          fleetSourcePoll,
	"AuthBlocked":       fleetSourcePoll,
	"VersionMismatches": fleetSourcePoll,
}

// fleetFieldProvenance returns the declared provenance of a SessionFleet figure field and
// ok=false when the field is undeclared (which the gate test forbids for any figure field).
func fleetFieldProvenance(field string) (claimcheck.Provenance, bool) {
	src, ok := fleetMetricProvenance[field]
	if !ok {
		return claimcheck.ProvNone, false
	}
	return fleetSourceProvenance(src), true
}

// fleet provenance glyphs: an OBSERVED (polled, relayed) figure carries "~", a WITNESSED
// (git-authored) figure carries "✓". They are distinct so the two trust classes read
// differently in the pane; an unlabeled provenance carries nothing.
const (
	fleetGlyphObserved  = "~"
	fleetGlyphWitnessed = "✓"
)

// fleetProvenanceGlyph is the per-figure suffix for a provenance label.
func fleetProvenanceGlyph(p claimcheck.Provenance) string {
	switch p {
	case claimcheck.Witnessed:
		return fleetGlyphWitnessed
	case claimcheck.Observed:
		return fleetGlyphObserved
	default:
		return ""
	}
}

// fleetFigure renders one labeled fleet figure as "<n> <unit><glyph>", tagging the number
// with its declared-provenance glyph (field names the SessionFleet field it came from) so
// an observed (polled) count and a witnessed (git) count read distinctly in the pane.
func fleetFigure(n int, unit, field string) string {
	s := fmt.Sprintf("%d %s", n, unit)
	if p, ok := fleetFieldProvenance(field); ok {
		s += fleetProvenanceGlyph(p)
	}
	return s
}

// guardInfoFleetPanelRows is the fleet-aggregate sub-pane. Full form is the verdict head,
// a states/totals line, and a few per-machine rows; mini folds to one row. Silent (nil)
// when the gateway reported no fleet block (a standalone box with no peers, or a fak serve
// gateway) — a silent panel costs zero rows.
func guardInfoFleetPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	f := ctx.v.Fleet
	if f == nil || f.Machines == 0 {
		return nil
	}
	if level == guardPanelMini {
		return []string{" fleet " + guardInfoFleetMiniText(f)}
	}
	rows := []string{" fleet    " + guardInfoFleetHeadText(f)}
	if totals := guardInfoFleetTotalsText(f); totals != "" {
		rows = append(rows, "          "+totals)
	}
	if ranked := fleetbottleneck.Rank(fleetBottleneckSnapshot(f)); len(ranked) > 0 {
		rows = append(rows, fmt.Sprintf("          bottleneck: %s (%s)", ranked[0].Class, ranked[0].Evidence))
	}
	rows = append(rows, guardInfoFleetMachineRows(f.Rows)...)
	return rows
}

// guardInfoFleetMiniText is the one-row fold: glyph verdict · N machines (with an
// attention count when any machine is stale/needs-action).
func guardInfoFleetMiniText(f *gateway.SessionFleet) string {
	s := guardFleetVerdictGlyph(f.Verdict) + guardFleetVerdictWord(f.Verdict)
	s += " · " + fleetFigure(f.Machines, "machines", "Machines")
	if att := f.Action + f.Stale; att > 0 {
		// The attention fold sums two poll-sourced (OBSERVED) counts; arithmetic never
		// upgrades provenance, so it carries the OBSERVED glyph.
		s += fmt.Sprintf(" · %d need attention%s", att, fleetProvenanceGlyph(claimcheck.Observed))
	}
	return s
}

// guardInfoFleetHeadText is the fleet header value: glyph verdict + machine count, with a
// breakdown of how many machines are needing-action / stale when any are.
func guardInfoFleetHeadText(f *gateway.SessionFleet) string {
	head := guardFleetVerdictGlyph(f.Verdict) + guardFleetVerdictWord(f.Verdict)
	head += " · " + fleetFigure(f.Machines, "machines", "Machines")
	var marks []string
	if f.Action > 0 {
		marks = append(marks, fleetFigure(f.Action, "action", "Action"))
	}
	if f.Stale > 0 {
		marks = append(marks, fleetFigure(f.Stale, "stale", "Stale"))
	}
	if len(marks) > 0 {
		head += " · " + strings.Join(marks, " · ")
	}
	return head
}

// guardInfoFleetTotalsText is the rolled-up totals row: sessions across the fleet, plus
// auth-blocked and version-mismatch counts when non-zero. Empty when there is nothing to
// roll up (so the row is dropped rather than shown as all-zero).
func guardInfoFleetTotalsText(f *gateway.SessionFleet) string {
	var parts []string
	if f.Sessions > 0 {
		parts = append(parts, fleetFigure(f.Sessions, "sessions", "Sessions"))
	}
	if f.AuthBlocked > 0 {
		parts = append(parts, fleetFigure(f.AuthBlocked, "auth-blocked", "AuthBlocked"))
	}
	if f.VersionMismatches > 0 {
		parts = append(parts, fleetFigure(f.VersionMismatches, "version-skew", "VersionMismatches"))
	}
	return strings.Join(parts, " · ")
}

// guardInfoFleetMachineRows renders the per-machine rows: id, state, snapshot age, and
// session count. Each is one continuation row under the header/totals.
func guardInfoFleetMachineRows(rows []gateway.SessionFleetMachine) []string {
	out := make([]string, 0, len(rows))
	for _, m := range rows {
		out = append(out, "          "+guardFleetStateGlyph(m.State)+guardInfoFleetMachineText(m))
	}
	return out
}

// guardInfoFleetMachineText is one machine's inline text: id (state) · age · sessions,
// dropping the pieces that are absent so a lean snapshot reads clean.
func guardInfoFleetMachineText(m gateway.SessionFleetMachine) string {
	id := strings.TrimSpace(m.ID)
	if id == "" {
		id = "?"
	}
	s := id
	if st := strings.TrimSpace(m.State); st != "" {
		s += " (" + st + ")"
	}
	var parts []string
	if m.AgeMin > 0 {
		parts = append(parts, guardFleetAgeText(m.AgeMin))
	}
	if m.Sessions > 0 {
		parts = append(parts, fmt.Sprintf("%d sess", m.Sessions))
	}
	if v := strings.TrimSpace(m.Version); v != "" {
		parts = append(parts, v)
	}
	if len(parts) > 0 {
		s += " · " + strings.Join(parts, " · ")
	}
	return s
}

// guardFleetAgeText renders a snapshot age in minutes compactly: "<1m" under a minute,
// whole minutes under an hour, else "Nh Mm".
func guardFleetAgeText(ageMin float64) string {
	if ageMin < 1 {
		return "<1m"
	}
	if ageMin < 60 {
		return fmt.Sprintf("%dm", int(ageMin))
	}
	h := int(ageMin) / 60
	m := int(ageMin) % 60
	if m == 0 {
		return fmt.Sprintf("%dh", h)
	}
	return fmt.Sprintf("%dh %dm", h, m)
}

// guardFleetVerdictGlyph maps a fleet verdict word to its status glyph.
func guardFleetVerdictGlyph(verdict string) string {
	switch strings.ToUpper(strings.TrimSpace(verdict)) {
	case "OK", "HEALTHY":
		return guardFleetGlyphOK
	case "ACTION", "STALE":
		return guardFleetGlyphAction
	default:
		return guardFleetGlyphUnknown
	}
}

// guardFleetVerdictWord is the verdict shown beside the glyph, defaulting to "unknown"
// when the fold reported no verdict.
func guardFleetVerdictWord(verdict string) string {
	if v := strings.TrimSpace(verdict); v != "" {
		return strings.ToLower(v)
	}
	return "unknown"
}

// guardFleetStateGlyph maps a per-machine state to a leading glyph: healthy, needs-
// attention (action/stale/unknown/invalid), else idle.
func guardFleetStateGlyph(state string) string {
	switch strings.ToUpper(strings.TrimSpace(state)) {
	case "OK", "HEALTHY", "RUNNING", "ALIVE":
		return guardFleetGlyphOK
	case "ACTION", "STALE", "UNKNOWN", "INVALID", "DEAD", "STALLED", "UNHEALTHY":
		return guardFleetGlyphAction
	default:
		return guardFleetGlyphUnknown
	}
}

func fleetBottleneckSnapshot(f *gateway.SessionFleet) fleetbottleneck.Snapshot {
	if f == nil {
		return fleetbottleneck.Snapshot{}
	}
	return fleetbottleneck.Snapshot{Machines: f.Machines, Sessions: f.Sessions, SeatCapacity: f.SeatCapacity, ThrottledSeats: f.ThrottledSeats, HealthySeats: f.HealthySeats, ResumeBacklog: f.ResumeBacklog, HostLoad: f.HostLoad, AuthBlocked: f.AuthBlocked}
}
