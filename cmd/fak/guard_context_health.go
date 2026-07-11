package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/trajhook"
)

// guard_context_health.go — issue #3099: surface the deterministic context-health
// VERDICT live on the guard status channel, the sibling of guardTrajectoryWarningLine.
//
// #3098 landed the verdict fold (internal/trajhook.ContextHealth: the closed
// HEALTHY/STALL/DRIFT/DETOUR_OVERRUN vocabulary over the deterministic behavioral
// fields) and #3096 the shed-span refcount detector (context use-after-free), but
// both spoke only OFFLINE over an exported corpus file (`fak traj score`). This rung
// folds the SAME two detectors over the LIVE process-global trajectory recorder — the
// corpus the kernel populates as it adjudicates THIS session — and emits one line on
// the same exit-summary channel the guard-decision counter rides, so the verdict moves
// as the live counters change: a repeat-failure loop reads STALL/DRIFT, a
// shed-then-reference bumps the use-after-free count.
//
// It is a pure read of the live recorder (no I/O, never blocks the session) and adds
// NO new signal string — the four verdicts are trajctl's closed vocabulary verbatim,
// the exact reuse #3098 mandates. When trajectory recording is off (FAK_TRAJECTORY
// unset — the default) there is no live corpus and the line stays empty, exactly like
// the other exit formatters stay quiet when their signal is absent.

// guardContextHealthLine renders the live context-health verdict for the guard exit
// summary, reading the process-global trajectory recorder (trajectory.Default) — the
// same live front door `fak traj` and the trajectory-garden skill read.
func guardContextHealthLine() string {
	return contextHealthLineFrom(trajectory.Default())
}

// contextHealthLineFrom folds one recorder's live corpus into the context-health line.
// Split from guardContextHealthLine so a test can drive a fresh recorder fed with real
// ABI turns instead of the process-global instance. Empty for a nil recorder
// (recording off) or a corpus that carries no verdict.
func contextHealthLineFrom(rec *trajectory.Recorder) string {
	if rec == nil {
		return ""
	}
	return renderContextHealthLine(rec.Turns())
}

// renderContextHealthLine is the pure fold: the #3098 verdict scorer plus the #3096
// shed-span refcount detector over a corpus, rendered as one worst-first line. Empty
// when the corpus carries no turns (nothing to report). The verdict is shown even when
// HEALTHY so the line reads as a live meter the operator watches move, not a
// fire-only-on-trouble warning.
func renderContextHealthLine(corpus []trajectory.Turn) string {
	if len(corpus) == 0 {
		return ""
	}
	worst, ok := worstContextHealthVerdict(trajhook.ContextHealth(trajhook.ContextHealthConfig{})(corpus))
	if !ok {
		return ""
	}
	line := "context health: " + worst.Label
	if worst.Reason != "" {
		line += " — " + worst.Reason
	}
	// The shed-span use-after-free count is an INDEPENDENT axis (#3096): a trace can
	// read HEALTHY on behavior yet still have shed a span a later turn referenced.
	// Append it only when it fired so a clean session stays terse.
	if uaf := len(trajhook.ShedRefcount()(corpus)); uaf > 0 {
		line += fmt.Sprintf("; %d context use-after-free(s)", uaf)
	}
	return line + "\n"
}

// worstContextHealthVerdict picks the most actionable per-trace verdict by the trajctl
// classify precedence (DETOUR_OVERRUN > DRIFT > STALL > HEALTHY), ties broken by
// descending score then trace id for a deterministic pick. ok is false only for an
// empty verdict set.
func worstContextHealthVerdict(verdicts []trajhook.Finding) (trajhook.Finding, bool) {
	var worst trajhook.Finding
	found := false
	for _, f := range verdicts {
		if !found || moreSevereContextHealth(f, worst) {
			worst, found = f, true
		}
	}
	return worst, found
}

// moreSevereContextHealth is the total order worstContextHealthVerdict ranks by:
// higher trajctl precedence first, then higher score, then lower trace id.
func moreSevereContextHealth(a, b trajhook.Finding) bool {
	sa, sb := contextHealthSeverity(a.Label), contextHealthSeverity(b.Label)
	if sa != sb {
		return sa > sb
	}
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	return a.TraceID < b.TraceID
}

// contextHealthSeverity ranks a verdict label by the trajctl classify precedence a
// controller acts on. An unknown or HEALTHY label is 0 (nothing to act on).
func contextHealthSeverity(label string) int {
	switch label {
	case string(trajctl.SignalDetourOverrun):
		return 3
	case string(trajctl.SignalDrift):
		return 2
	case string(trajctl.SignalStall):
		return 1
	default:
		return 0
	}
}
