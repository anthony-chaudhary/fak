package main

// guard_output_style.go — the enforce rung for the "closed on a prose wall" pathology.
//
// The Stop hook already SENSES this: readGuardStopTranscript -> applyClosingSignal
// (guard_stops.go) folds the FINAL assistant turn through internal/headlesslint.ScanClosing
// and stamps the row when that turn's last block is a trailing prose wall — a long paragraph
// with no scannable bullets, burying the verdict and the next step in text the operator has to
// re-read. AGENTS.md binds the shape: "Close operator-facing turns with scannable bullets,
// verdict first; make the last line a bullet carrying the next checkable step."
//
// This file is the ACT half. On a clean stop whose final turn closed on a prose wall, enforce
// blocks the stop and feeds back the re-cast remediation the sensor already computed (bullets,
// verdict first, next step as the last bullet), so the agent closes scannable and keeps going.
// It mirrors the operator-directed ladder's off|shadow|warn|enforce.
//
// Two deliberate departures from the operator-directed gate:
//   - It SHIPS OFF. Closing shape is a cosmetic-adjacent nicety, not a correctness stall like an
//     unanswered question — so the default is inert (guardPreCompactModeOff), and the gate only
//     ever acts once an operator opts in with --output-style. It still soaks through shadow/warn
//     before an operator promotes it to enforce.
//   - There is NO HUMAN_RESIDUAL exemption. A prose wall has no authority-wall escape hatch the
//     way an operator-directed question does; every finding is resolvable by re-casting, so
//     enforce always blocks-and-remediates rather than routing an escalation.
//
// The operator-absent cap still applies (guardOutputStyleEffectiveMode): an attended interactive
// child is capped at warn, so an enforce ever reaching the hook means the child was headless —
// symmetry with the operator-directed and hardware-gate rungs, even though OFF-by-default already
// makes an accidental attended block unlikely.

import (
	"fmt"
	"io"
	"strings"
)

const (
	// guardStopHookOutputStyleEnvMode is the resolved output-style gate mode the guard installer
	// injects into the Stop-hook child (alongside stopHookEnv). The install-time resolution
	// (guardOutputStyleEffectiveMode) has already applied the operator-absent cap, so the value
	// here is authoritative: an `enforce` means the child was headless.
	guardStopHookOutputStyleEnvMode = "FAK_GUARD_OUTPUT_STYLE_MODE"

	// guardOutputStyleModeWarn is the SOAK rung between shadow and enforce: the re-cast remediation
	// is printed for the operator to see, but the stop is still allowed. off/shadow/enforce reuse
	// the guardPreCompactMode* vocabulary. Unlike the operator-directed gate, warn is NOT the
	// shipped default — this gate ships OFF (see guardOutputStyleEffectiveMode).
	guardOutputStyleModeWarn = "warn"
)

// normalizeGuardOutputStyleMode canonicalizes the gate mode. Empty/whitespace defaults to OFF
// (the shipped default — this rung is inert until an operator opts in); an unknown value is an
// error the caller treats as fail-open (never block a stop on a bad mode string).
func normalizeGuardOutputStyleMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardOutputStyleModeWarn:
		return guardOutputStyleModeWarn, nil
	case guardPreCompactModeEnforce:
		return guardPreCompactModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid --output-style mode %q (want off, shadow, warn, or enforce)", mode)
	}
}

// guardOutputStyleNormalizedOrOff is the total form of normalizeGuardOutputStyleMode for the
// env-injection boundary: it never returns an error, falling back to OFF on a bad string so a
// misconfigured value is pinned inert rather than accidentally enforce.
func guardOutputStyleNormalizedOrOff(mode string) string {
	normalized, err := normalizeGuardOutputStyleMode(mode)
	if err != nil {
		return guardPreCompactModeOff
	}
	return normalized
}

// guardOutputStyleEffectiveMode resolves the output-style gate mode for a session, applying the
// operator-absent cap. It mirrors guardOperatorDirectedEffectiveMode but the un-set default is
// OFF, not warn:
//   - not explicitly set                -> off (the shipped default; the rung is inert)
//   - explicit enforce + interactive     -> capped to warn (surface the remediation, never block)
//   - explicit off/shadow/warn           -> honored as given
//   - explicit enforce + headless child  -> enforce verbatim
//
// So an `enforce` value ever seen by the Stop hook implies the child was headless, and an
// unconfigured session never fires this gate at all.
func guardOutputStyleEffectiveMode(configured string, explicitlySet, childInteractive bool) string {
	mode, err := normalizeGuardOutputStyleMode(configured)
	if err != nil {
		mode = guardPreCompactModeOff // fail safe to inert, never to enforce
	}
	if !explicitlySet {
		return guardPreCompactModeOff
	}
	if childInteractive && mode == guardPreCompactModeEnforce {
		return guardOutputStyleModeWarn
	}
	return mode
}

// runGuardOutputStyleGate applies the output-style rung to a clean stop whose final turn the
// ScanClosing sensor flagged as a prose wall (the ClosingProseWall/ClosingResolve fields on tr).
// rawMode is the RESOLVED effective mode (the operator-absent cap has already been applied at
// install), so an enforce here is safe to act on. It returns:
//
//	exit  — 2 to BLOCK the stop (feed the re-cast remediation to the model), 0 to allow it.
//	disp  — the typed disposition to stamp on the stop row (only meaningful when fired).
//	fired — false when the gate is inert: off, no transcript, or a scannable (non-wall) close.
//	        A false fired tells the caller to proceed as an ordinary clean stop.
//
// enforce blocks so the agent re-casts the closing; warn/shadow always allow (soak/observe). It
// is fail-open: a bad mode string is treated as "do not fire".
func runGuardOutputStyleGate(stderr io.Writer, rawMode string, tr *guardStopTranscript) (exit int, disp guardStopDisposition, fired bool) {
	mode, err := normalizeGuardOutputStyleMode(rawMode)
	if err != nil {
		return 0, "", false // fail-open: never block a stop on a bad mode
	}
	if mode == guardPreCompactModeOff || tr == nil || !tr.ClosingProseWall {
		return 0, "", false
	}
	switch mode {
	case guardPreCompactModeShadow:
		fmt.Fprintln(stderr, guardOutputStyleShadowLine())
		return 0, stopDispOutputStyleShadow, true
	case guardOutputStyleModeWarn:
		fmt.Fprintln(stderr, guardOutputStyleWarnLine(tr))
		return 0, stopDispOutputStyleWarn, true
	default: // enforce
		fmt.Fprintln(stderr, guardOutputStyleContinueMessage(tr))
		return 2, stopDispOutputStyleContinue, true
	}
}

// guardOutputStyleContinueMessage is the exact stderr text fed back to the MODEL when enforce
// blocks the stop (exit 2). It names the shape of the close and hands over the re-cast
// remediation the sensor already computed — verdict first, scannable bullets, next step last.
func guardOutputStyleContinueMessage(tr *guardStopTranscript) string {
	return fmt.Sprintf("fak guard Stop: this turn closed on a prose wall — a trailing paragraph that buries the verdict and the next step in text the operator has to re-read. Do not stop on it. Instead: %s Then finish the turn.",
		guardOutputStyleResolveText(tr))
}

// guardOutputStyleWarnLine is the OPERATOR-facing soak line (exit 0, NOT fed to the model): it
// shows exactly what enforce WOULD tell the model, so an operator can watch the rate and the
// remediation quality before promoting the gate to enforce.
func guardOutputStyleWarnLine(tr *guardStopTranscript) string {
	return fmt.Sprintf("fak guard Stop: [warn] closing prose wall — enforce would ask the agent to re-cast the closing as bullets. Remediation: %s (allowing the stop; soak mode).",
		guardOutputStyleResolveText(tr))
}

// guardOutputStyleShadowLine is the terse OPERATOR-facing observe line (exit 0): the decision the
// gate WOULD reach, for metrics-only soaks.
func guardOutputStyleShadowLine() string {
	return "fak guard Stop: shadow — closing prose wall; enforce would ask for a bulleted, verdict-first close. Allowing the stop."
}

// guardOutputStyleResolveText returns the ScanClosing remediation text, defaulting to a generic
// instruction if the sensor left it blank (defensive; the sensor always sets it when fired).
func guardOutputStyleResolveText(tr *guardStopTranscript) string {
	if r := strings.TrimSpace(tr.ClosingResolve); r != "" {
		return r
	}
	return "re-cast the closing as bullets, verdict first; put the next checkable step as the final bullet."
}
