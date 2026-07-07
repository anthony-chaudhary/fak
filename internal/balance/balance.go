// Package balance renders the NIGHT-BALANCE surface: the two forces a self-driving run
// must hold in equilibrium, side by side, in one glanceable readout. A fleet that runs
// all night has two ways to fall out of balance, and until now each was legible only in
// its own subsystem:
//
//   - RECOVERY vs STRANDING — the resume budget (#3124) is progress-earned: a session
//     that keeps producing turns earns headroom, one that re-strands with no progress
//     exhausts fast. Folded across the fleet, the one question that matters is whether
//     recovery is KEEPING UP: are more sessions taking than re-stranding, or is the
//     budget underwater?
//   - GARDENING vs THROUGHPUT — the work mix (#3126) splits each walk's worklist into
//     quality-tending versus backlog-draining, against a declared soft target. A night
//     that drains issues while the gardens rot is as out of balance as one that tends
//     quality while the backlog piles up.
//
// This package is the READOUT that folds both halves into one panel so an operator — or
// the night loop itself — sees both balances in one place. It is PURE and fixture-driven
// in the same spirit as the fak-info panels (cmd/fak/info_panels.go): a renderer that
// projects TYPED evidence into gutter-labeled rows, reads no files and no clock, and is
// pinned by fixtures. The impure shells collect the evidence — `fak resume status` folds
// the per-session resume states, `fak superloop walk` folds the work mix — and hand it in.
//
// # Graceful degradation
//
// The two halves are INDEPENDENTLY optional. Each carries a measured bit (the resume
// rollup's Measured flag; a nil Mix pointer), and an unmeasured half renders a single
// honest "not measured" row instead of fabricated zeros — so the surface is useful the
// moment EITHER of #3124/#3126 has data, and says "no data" plainly when neither does.
// This is the load-bearing property: the balance surface never invents a balance it did
// not witness.
package balance

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// ResumeBudget is the resume-recovery rollup — the #3124 half, folded across a fleet of
// stranded sessions. It tallies where each session landed in its resume journey
// (resume.FoldResumeState), reduced to the one question the progress-earned budget
// exists to answer: is recovery KEEPING UP with stranding?
type ResumeBudget struct {
	// Took counts resumes that provably progressed (new turns + a clean terminal): the
	// budget's completions. ReStranded counts resumes that fired and re-hit a wall —
	// eligible for another attempt, but not yet recovered; the slippage the RED test
	// weighs against Took. GaveUp counts sessions past their earned cap or on an
	// unrecoverable wall — recovery abandoned. Launched fired but is not yet proven
	// either way; Pending crashed with no resume yet; Settled was closed by an operator
	// by hand (outside the automatic budget — carried for the total, never for the alarm).
	Took       int `json:"took"`
	ReStranded int `json:"re_stranded"`
	GaveUp     int `json:"gave_up"`
	Launched   int `json:"launched"`
	Pending    int `json:"pending"`
	Settled    int `json:"settled"`
	// Measured is false when no resume ledger/store was read this render. An unmeasured
	// rollup degrades to a "not measured" row; a MEASURED rollup that happens to be all
	// zeros is a real fact (a fleet with nothing stranded), not the same thing.
	Measured bool `json:"measured"`
}

// ReStrandingOutpacesCompletion is the RED condition: strictly more sessions re-stranded
// than completed. It is the fleet-level echo of #3124's per-session floor — when the
// budget spends more resumes re-stranding than taking, recovery is underwater and a human
// should look. A never-measured or empty rollup is never red (0 > 0 is false): the alarm
// only fires on witnessed slippage, never on absence of data.
func (b ResumeBudget) ReStrandingOutpacesCompletion() bool {
	return b.ReStranded > b.Took
}

// FoldResumeBudget tallies a fleet's per-session resume states — one resume.ResumeState
// per stranded session, exactly as `fak resume status` folds them — into the rollup, and
// marks it Measured. An empty slice folds to a measured all-zero rollup (a fleet with
// nothing stranded is a real, reportable balance), NOT an unmeasured one: callers that
// did not read a store at all should leave ResumeBudget{} zero-valued (Measured false)
// rather than call this with nil. Pure and total over any input.
func FoldResumeBudget(states []resume.ResumeState) ResumeBudget {
	b := ResumeBudget{Measured: true}
	for _, s := range states {
		switch s {
		case resume.ResumeTook:
			b.Took++
		case resume.ResumeReStranded:
			b.ReStranded++
		case resume.ResumeGaveUp:
			b.GaveUp++
		case resume.ResumeLaunched:
			b.Launched++
		case resume.ResumePending:
			b.Pending++
		case resume.ResumeSettled:
			b.Settled++
		}
	}
	return b
}

// Evidence is the full input to Render: the two balances, each independently optional so
// the surface degrades to whichever half landed. Resume carries its own Measured bit; Mix
// is a pointer whose nil means "no superloop walk this render" — the mix half then reads
// as not measured. A zero Evidence{} (Resume unmeasured, Mix nil) renders an honest
// no-data surface, which is the correct output before either #3124 or #3126 has run.
type Evidence struct {
	Resume ResumeBudget
	// Mix is the #3126 work-mix split, taken straight from a WalkReport. Nil = not
	// measured this render.
	Mix *superloop.WorkMix
}

// Status folds the whole surface into one closed headline token, most-alarming first:
//
//   - "red"     — the resume budget is measured and re-stranding outpaces completion.
//     This is the only hard alarm; it outranks any mix imbalance because a recovery
//     budget underwater strands work outright, while a mix lean is a soft nudge.
//   - "leaning" — not red, but the work mix is measured and off its declared target
//     (the walk chose to favor a class). Advisory: the run is working, just lopsided.
//   - "no data" — neither half was measured this render; there is no balance to report.
//   - "ok"      — everything measured is in balance (or only the healthy half landed).
//
// Total over any Evidence; the vocabulary is closed so a monitor can switch on it.
func (ev Evidence) Status() string {
	mixMeasured := ev.Mix != nil
	if !ev.Resume.Measured && !mixMeasured {
		return "no data"
	}
	if ev.Resume.Measured && ev.Resume.ReStrandingOutpacesCompletion() {
		return "red"
	}
	if mixMeasured && ev.Mix.Favor != "" {
		return "leaning"
	}
	return "ok"
}

// Render projects the evidence into the balance panel's rows: a headline row carrying the
// folded Status, then one row per half. A measured half renders its numbers with a
// keeping-up / lean marker; an unmeasured half renders a single "not measured" row naming
// which shell would fill it — never a fabricated zero. The row shape (2-space gutter,
// label column, "·" section divider, ⚠/✓ markers) matches the fak-info panels so the
// surface reads the same whether it is printed standalone or dropped into a pane. Pure
// and deterministic: the same Evidence always renders the same rows.
func Render(ev Evidence) []string {
	rows := []string{fmt.Sprintf("night balance — %s", ev.Status())}
	rows = append(rows, "  resume   "+resumeLine(ev.Resume))
	rows = append(rows, "  work     "+mixLine(ev.Mix))
	return rows
}

// resumeLine renders the resume-budget half. Measured: the tally plus a keeping-up marker
// that shows the deciding comparison (re-stranded vs took) in the clear, so an operator
// sees WHY it is red, not just that it is. Unmeasured: one honest "not measured" row.
func resumeLine(b ResumeBudget) string {
	if !b.Measured {
		return "not measured — no resume ledger read this render"
	}
	head := fmt.Sprintf("took %d  re-stranded %d  gave-up %d  ·  launched %d  pending %d",
		b.Took, b.ReStranded, b.GaveUp, b.Launched, b.Pending)
	if b.Settled > 0 {
		head += fmt.Sprintf("  settled %d", b.Settled)
	}
	if b.ReStrandingOutpacesCompletion() {
		return head + fmt.Sprintf("   ⚠ re-stranding outpaces completion (%d>%d)", b.ReStranded, b.Took)
	}
	return head + fmt.Sprintf("   ✓ recovery keeping up (%d≤%d)", b.ReStranded, b.Took)
}

// mixLine renders the work-mix half. Measured: the gardening/throughput/neutral counts,
// the measured throughput share against the declared target, and the soft favor the walk
// leaned (or "on target" when it did not). Unmeasured (nil): one honest "not measured"
// row. The share is "n/a" when no gardening/throughput members are on the worklist —
// there is no ratio to take — mirroring superloop's own no-lean guard.
func mixLine(m *superloop.WorkMix) string {
	if m == nil {
		return "not measured — no superloop walk this render"
	}
	head := fmt.Sprintf("gardening %d  throughput %d  neutral %d  ·  target %d%%  mix %s",
		m.Gardening, m.Throughput, m.Neutral, m.TargetThroughputPct, sharePct(m.Gardening, m.Throughput))
	if m.Favor == "" {
		return head + "   → on target"
	}
	return head + fmt.Sprintf("   → lean %s", m.Favor)
}

// sharePct is the measured throughput share of the gardening+throughput work, as a
// whole-percent string, or "n/a" when neither class is present (no ratio to take). It is
// the same numerator/denominator superloop.favoredClass uses, surfaced for the reader.
func sharePct(gardening, throughput int) string {
	denom := gardening + throughput
	if denom == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%d%%", throughput*100/denom)
}
