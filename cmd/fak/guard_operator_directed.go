package main

// guard_operator_directed.go — the enforce rung for the "stopped to ask a human" pathology.
//
// The Stop hook already SENSES this: readGuardStopTranscript -> applyHeadlessLintSignal
// (guard_stops.go) folds the FINAL assistant turn through internal/headlesslint and stamps the
// row when that turn ended by addressing a person — "Do you want me to push?", "waiting for your
// approval", "let me know how to proceed". In an ATTENDED interactive `fak guard -- claude` that
// question is answered by the human at the TUI. In an UNATTENDED headless run (`fak guard --
// claude -p …`) there is no one to answer: the question hangs and the work silently stalls.
//
// This file is the ACT half. On a clean stop whose final turn is operator-directed, an autonomous
// worker is told to do what a headless process should do INSTEAD of asking — the choicetriage
// remediation the sensor already computed (take the obvious action, restate the assumption, file a
// ticket) — and the stop is blocked so the agent gets that guidance and keeps going. It mirrors the
// deny-all ladder's off|shadow|enforce and adds a warn rung between shadow and enforce so the
// pathology can SOAK (remediation printed, stop still allowed) before we promote to blocking.
//
// Two guardrails, both structural:
//   - Only enforce when the OPERATOR IS ABSENT. guardOperatorDirectedEffectiveMode caps an attended
//     interactive session at warn (or off), exactly like guardTaskHandoffEffectiveMode caps the
//     handoff gate — so an `enforce` ever reaching the hook means the operator was absent: the child
//     was headless, OR an orchestrator marked an interactive child unattended (a fleet/remote session
//     driven by an agent, not a responsive human — guardOperatorUnattendedEnv). That operator-driven
//     interactive corner (#4951) is the reason "interactive" alone is not "a human is present": the
//     unattended flag is the first-class signal that lifts the attended cap so a prose "stopped to ask
//     a human" question is escalated, not silently stalled, in exactly the sessions no human watches.
//   - HUMAN_RESIDUAL routes to a typed ESCALATION, not a re-prompt. An authority/approval wall
//     (a release gate, a sign-off) is a legitimate stop; blocking-and-reprompting it would just spin.
//     The gate allows that stop and emits a typed escalation line for the operator to route.

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/headlesslint"
)

const (
	// guardStopHookOperatorDirectedEnvMode is the resolved operator-directed gate mode the guard
	// installer injects into the Stop-hook child. The install-time resolution
	// (guardOperatorDirectedEffectiveMode) has already applied the operator-absent cap, so the value
	// here is authoritative: an `enforce` means the operator was absent — the child was headless, or an
	// interactive child an orchestrator marked unattended (guardOperatorUnattendedEnv).
	guardStopHookOperatorDirectedEnvMode = "FAK_GUARD_OPERATOR_DIRECTED_MODE"

	// guardOperatorUnattendedEnv marks an INTERACTIVE child (no `-p`, so guardChildInteractive is
	// true) as operator-driven: a fleet/remote orchestrator drives the session and no responsive human
	// is at the TUI. Set truthy (1|true|yes|on), it lifts the attended cap in
	// guardOperatorDirectedEffectiveMode so the operator-directed gate is FIRST-CLASS for the session —
	// its configured off|shadow|warn|enforce is honored (a prose "stopped to ask a human" turn escalates
	// instead of hanging), exactly as for a headless `-p` worker. Default false: an ordinary attended
	// `fak guard -- claude` is byte-for-byte unchanged. This is the #4951 gap's operator-presence axis:
	// "interactive" is a weak proxy for "a human can answer", and this env is the explicit override.
	guardOperatorUnattendedEnv = "FAK_GUARD_OPERATOR_UNATTENDED"

	// guardOperatorDirectedModeWarn is the SOAK rung between shadow and enforce and the shipped
	// default for a headless child: the choicetriage remediation is printed for the operator to see,
	// but the stop is still allowed. off/shadow/enforce reuse the guardPreCompactMode* vocabulary.
	guardOperatorDirectedModeWarn = "warn"
)

// normalizeGuardOperatorDirectedMode canonicalizes the gate mode. Empty/whitespace defaults to warn
// (the soak default, matching the shipped `fak guard --operator-directed` flag default); an unknown
// value is an error the caller treats as fail-open (never block a stop on a bad mode string).
func normalizeGuardOperatorDirectedMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardOperatorDirectedModeWarn:
		return guardOperatorDirectedModeWarn, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeEnforce:
		return guardPreCompactModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid --operator-directed mode %q (want off, shadow, warn, or enforce)", mode)
	}
}

// guardOperatorDirectedNormalizedOrWarn is the total form of normalizeGuardOperatorDirectedMode for
// the env-injection boundary: it never returns an error, falling back to the warn soak default on a
// bad string so a misconfigured value is pinned as warn (advisory) rather than accidentally enforce.
func guardOperatorDirectedNormalizedOrWarn(mode string) string {
	normalized, err := normalizeGuardOperatorDirectedMode(mode)
	if err != nil {
		return guardOperatorDirectedModeWarn
	}
	return normalized
}

// guardOperatorUnattended reports whether the current interactive session was marked operator-driven
// (no responsive human at the TUI) via guardOperatorUnattendedEnv. It is the operator-presence signal
// that guardOperatorDirectedEffectiveMode consults to decide whether the attended cap applies; on a
// headless child the value is irrelevant (the cap never applies there anyway).
func guardOperatorUnattended() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(guardOperatorUnattendedEnv))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// guardOperatorDirectedEffectiveMode resolves the operator-directed gate mode for a session,
// applying the operator-absent cap. It mirrors guardTaskHandoffEffectiveMode: an ATTENDED interactive
// TUI child (a human is present to answer a genuine question) must never have a stop BLOCKED by this
// gate. The load-bearing distinction (#4951) is that "interactive" alone is a weak proxy for "a human
// is present": a fleet/remote orchestrator can drive an interactive `claude` (no `-p`) with no human
// watching, and there a prose "stopped to ask a human" turn hangs exactly like a headless one. The
// operatorUnattended axis (guardOperatorUnattendedEnv) is the explicit override for that corner, so:
//   - headless child, OR interactive+unattended -> the configured mode verbatim (warn by default): the
//     operator is absent, so the full off|shadow|warn|enforce ladder is first-class for the session.
//   - attended interactive, not explicitly set  -> off (no friction in an attended session at all)
//   - attended interactive, explicit enforce     -> capped to warn (surface the remediation, never block)
//   - attended interactive, explicit off/shadow/warn -> honored as given
//
// So an `enforce` value ever seen by the Stop hook implies the operator was absent — headless, or an
// interactive child an orchestrator marked unattended — and blocking that false stop cannot silence a
// real human question, because there is no human present to have asked one.
func guardOperatorDirectedEffectiveMode(configured string, explicitlySet, childInteractive, operatorUnattended bool) string {
	mode, err := normalizeGuardOperatorDirectedMode(configured)
	if err != nil {
		mode = guardOperatorDirectedModeWarn // fail safe to the soak default, never to enforce
	}
	if !childInteractive || operatorUnattended {
		return mode
	}
	if !explicitlySet {
		return guardPreCompactModeOff
	}
	if mode == guardPreCompactModeEnforce {
		return guardOperatorDirectedModeWarn
	}
	return mode
}

// runGuardOperatorDirectedGate applies the operator-directed rung to a clean stop whose final turn
// the headlesslint sensor flagged as addressing a human (the OperatorDirected* fields on tr). rawMode
// is the RESOLVED effective mode (the operator-absent cap has already been applied at install), so an
// enforce here is safe to act on. It returns:
//
//	exit  — 2 to BLOCK the stop (feed the remediation to the model), 0 to allow it.
//	disp  — the typed disposition to stamp on the stop row (only meaningful when fired).
//	fired — false when the gate is inert: off, no transcript, or a clean (non-directed) final turn.
//	        A false fired tells the caller to proceed as an ordinary clean stop.
//
// enforce blocks a RESOLVABLE finding (TAKE_OBVIOUS / FRESH_CONTEXT / FILE_TICKET) so the agent acts
// instead of asking, but ALLOWS a HUMAN_RESIDUAL finding — a genuine authority wall — and emits a
// typed escalation line rather than re-prompting it. warn/shadow always allow (soak/observe). It is
// fail-open: a bad mode string is treated as "do not fire".
func runGuardOperatorDirectedGate(stderr io.Writer, rawMode string, tr *guardStopTranscript) (exit int, disp guardStopDisposition, fired bool) {
	mode, err := normalizeGuardOperatorDirectedMode(rawMode)
	if err != nil {
		return 0, "", false // fail-open: never block a stop on a bad mode
	}
	if mode == guardPreCompactModeOff || tr == nil || !tr.OperatorDirected {
		return 0, "", false
	}
	switch mode {
	case guardPreCompactModeShadow:
		fmt.Fprintln(stderr, guardOperatorDirectedShadowLine(tr))
		return 0, stopDispOperatorDirectedShadow, true
	case guardOperatorDirectedModeWarn:
		fmt.Fprintln(stderr, guardOperatorDirectedWarnLine(tr))
		return 0, stopDispOperatorDirectedWarn, true
	default: // enforce
		if tr.OperatorDirectedDisposition == string(choicetriage.HumanResidual) {
			fmt.Fprintln(stderr, guardOperatorDirectedEscalationLine(tr))
			return 0, stopDispOperatorDirectedEscalate, true
		}
		fmt.Fprintln(stderr, guardOperatorDirectedContinueMessage(tr))
		return 2, stopDispOperatorDirectedContinue, true
	}
}

// guardOperatorDirectedContinueMessage is the exact stderr text fed back to the MODEL when enforce
// blocks the stop (exit 2). It names the linguistic shape of the question and hands over the
// choicetriage remediation the sensor already computed — what an autonomous worker does INSTEAD of
// asking — then points at the sanctioned wrap-up for the case where a protected boundary genuinely
// blocks the last step.
func guardOperatorDirectedContinueMessage(tr *guardStopTranscript) string {
	if tr != nil && tr.OperatorDirectedClass == string(headlesslint.PrematureSurrender) {
		return fmt.Sprintf("fak guard Stop: this turn prematurely surrendered (%s) without completing the task or verifying deliverables. Do not give up. Instead: %s Then finish the turn. If a protected boundary genuinely blocks the last step, note it on one line (`no allowed path: <reason>`) and stop cleanly — that is a complete outcome.",
			guardOperatorDirectedClassLabel(tr), guardOperatorDirectedResolveText(tr))
	}
	return fmt.Sprintf("fak guard Stop: this turn ended by asking a human (%s), but this is a headless run with no operator to answer — the question will hang and the work stalls. Do not stop to ask. Instead: %s Then finish the turn. If a protected boundary genuinely blocks the last step, note it on one line (`no allowed path: <reason>`) and stop cleanly — that is a complete outcome.",
		guardOperatorDirectedClassLabel(tr), guardOperatorDirectedResolveText(tr))
}

// guardOperatorDirectedWarnLine is the OPERATOR-facing soak line (exit 0, NOT fed to the model): it
// shows exactly what enforce WOULD tell the model, so an operator can watch the rate and the
// remediation quality before promoting the gate to enforce.
func guardOperatorDirectedWarnLine(tr *guardStopTranscript) string {
	return fmt.Sprintf("fak guard Stop: [warn] operator-directed stop (%s -> %s) — enforce would auto-continue so the agent acts instead of asking. Remediation: %s (allowing the stop; soak mode).",
		guardOperatorDirectedClassLabel(tr), tr.OperatorDirectedDisposition, guardOperatorDirectedResolveText(tr))
}

// guardOperatorDirectedShadowLine is the terse OPERATOR-facing observe line (exit 0): the decision
// the gate WOULD reach, for metrics-only soaks.
func guardOperatorDirectedShadowLine(tr *guardStopTranscript) string {
	return fmt.Sprintf("fak guard Stop: shadow — operator-directed stop (%s -> %s); enforce would auto-continue. Allowing the stop.",
		guardOperatorDirectedClassLabel(tr), tr.OperatorDirectedDisposition)
}

// guardOperatorDirectedEscalationLine is the OPERATOR-facing line (exit 0) for a HUMAN_RESIDUAL
// finding: a genuine authority/approval wall the agent was RIGHT to stop on. The gate allows the
// stop and records a typed escalation to route, rather than re-prompting a question no autonomous
// action can resolve.
func guardOperatorDirectedEscalationLine(tr *guardStopTranscript) string {
	return fmt.Sprintf("fak guard Stop: operator-directed stop folds to HUMAN_RESIDUAL (%s) — a genuine escalation, not a re-promptable question. Routing it rather than auto-continuing. Next: %s Allowing the stop.",
		guardOperatorDirectedClassLabel(tr), guardOperatorDirectedResolveText(tr))
}

// guardOperatorDirectedClassLabel returns the finding's linguistic Class for a message, defaulting
// to a generic label if the sensor left it blank (defensive; the sensor always sets it when fired).
func guardOperatorDirectedClassLabel(tr *guardStopTranscript) string {
	if c := strings.TrimSpace(tr.OperatorDirectedClass); c != "" {
		return c
	}
	return "operator-directed"
}

// guardOperatorDirectedResolveText returns the choicetriage remediation text, defaulting to a
// generic instruction if the sensor left it blank.
func guardOperatorDirectedResolveText(tr *guardStopTranscript) string {
	if r := strings.TrimSpace(tr.OperatorDirectedResolve); r != "" {
		return r
	}
	return "take the obvious next action yourself and state the assumption you made."
}
