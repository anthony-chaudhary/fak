package main

// guard_hardware_gate.go — the Stop-hook rung for the "local machine is the compute
// boundary" pathology (the hardware-gate regression).
//
// The pathology: a headless turn ends by declaring a LOCAL-hardware blocker as terminal
// — "no GPU on this host", "can't run the CUDA witness without a device", "not
// reproducible on this laptop" — and stops. But in the fleet this host is only the
// CONTROL POINT; the device/GPU/heavy work belongs on a sanctioned compute node (GCP
// fak-realmodel, the DGX via dgxbridge, da33, the nightrun pipeline). Stopping at the
// local boundary strands work the fleet could witness.
//
// This file is BOTH the sensor and the act: it reads the final assistant turn from the
// transcript itself, folds it through internal/hwgatelint, and — on a hardware-gated stop
// — tells an autonomous worker to do what the fleet exists for INSTEAD of stopping:
// dispatch the work to the right node, or, if the credential/bridge session is missing,
// produce the exact ready-to-run command sequence and hand it to the operator (that still
// counts as using the fleet). It is a fully dedicated hook: it does not read any state that
// another rung's sensor left behind, so it fires whether or not the reporting sensor ran.
// It mirrors the operator-directed ladder's off|shadow|warn|enforce.
//
// Structural guardrail: like the operator-directed gate, this one only reaches `enforce`
// when the OPERATOR IS ABSENT — its install-time mode inherits the operator-directed
// effective mode (guardOperatorDirectedEffectiveMode has already applied the
// operator-absent cap), so an attended interactive session can never have a stop BLOCKED
// here. An operator can still tune it independently via FAK_GUARD_HARDWARE_GATE_MODE /
// --hardware-gate.

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hwgatelint"
	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
)

const (
	// guardStopHookHardwareGateEnvMode is the resolved hardware-gate mode the guard installer
	// injects into the Stop-hook child. Its install-time value inherits the operator-directed
	// resolved posture (operator-absent cap already applied), so an `enforce` here means the
	// child was headless.
	guardStopHookHardwareGateEnvMode = "FAK_GUARD_HARDWARE_GATE_MODE"

	// guardHardwareGateModeWarn is the SOAK rung between shadow and enforce and the shipped
	// default for a headless child: the sanctioned-route redirect is printed for the operator to
	// see, but the stop is still allowed. off/shadow/enforce reuse the guardPreCompactMode* vocab.
	guardHardwareGateModeWarn = "warn"

	// guardHardwareGateTailBytes bounds the transcript tail this gate re-reads. The tail carries
	// the terminal turns, which is all the final-turn scan needs; keeping the read self-contained
	// (rather than depending on another rung's sensor) is what makes this a dedicated hook.
	guardHardwareGateTailBytes = 512 * 1024
)

// hardware-gate stop dispositions: the final turn declared a LOCAL-hardware blocker as
// terminal instead of dispatching to a sanctioned compute node. Enforce BLOCKS it and feeds
// the sanctioned-route redirect back; warn/shadow allow the stop while recording the outcome.
const (
	stopDispHardwareGateContinue guardStopDisposition = "hardware_gate_continue" // block: "no local GPU" stop; feed the sanctioned-compute-node redirect back
	stopDispHardwareGateWarn     guardStopDisposition = "hardware_gate_warn"     // allow: warn soak — redirect printed, stop allowed
	stopDispHardwareGateShadow   guardStopDisposition = "hardware_gate_shadow"   // allow: shadow — would-enforce decision logged, stop allowed
)

// normalizeGuardHardwareGateMode canonicalizes the gate mode. Empty/whitespace defaults to
// warn (the soak default, matching the shipped `fak guard --hardware-gate` flag default); an
// unknown value is an error the caller treats as fail-open (never block a stop on a bad mode).
func normalizeGuardHardwareGateMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardHardwareGateModeWarn:
		return guardHardwareGateModeWarn, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeEnforce:
		return guardPreCompactModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid --hardware-gate mode %q (want off, shadow, warn, or enforce)", mode)
	}
}

// guardHardwareGateNormalizedOrWarn is the total form of normalizeGuardHardwareGateMode for the
// env-injection boundary: it never returns an error, falling back to the warn soak default on a
// bad string so a misconfigured value is pinned as warn (advisory) rather than enforce.
func guardHardwareGateNormalizedOrWarn(mode string) string {
	normalized, err := normalizeGuardHardwareGateMode(mode)
	if err != nil {
		return guardHardwareGateModeWarn
	}
	return normalized
}

// hardwareGateFinalProseTurn reads a bounded tail of the session transcript and returns the
// text of the FINAL real (non-synthetic) assistant turn IF that turn ended WITHOUT a tool call
// — the prose-only end_turn that actually stops the session. It returns "" otherwise (empty
// path, unreadable/empty file, or a final turn that still tried a tool: the harness feeds that
// tool result back and the session keeps going, so it is not stopping-to-declare-a-blocker).
// Self-contained and FAIL-OPEN: any miss yields "", leaving the gate inert. This mirrors the
// final-turn gate the reporting sensor uses, so the ACT half senses exactly what it would.
func hardwareGateFinalProseTurn(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	recs := transcript.LoadFileTail(path, guardHardwareGateTailBytes)
	var lastText string
	var lastHadTool bool
	for _, r := range recs {
		if r.Role() != "assistant" || r.IsSynthetic() {
			continue
		}
		lastHadTool = r.LastToolUseName() != ""
		lastText = r.Text()
	}
	if lastHadTool {
		return ""
	}
	return lastText
}

// runGuardHardwareGateGate applies the hardware-gate rung to a clean stop. It reads the final
// prose turn from transcriptPath itself and folds it through internal/hwgatelint; if that turn
// declared a local-hardware blocker as terminal, it applies the ladder. rawMode is the RESOLVED
// effective mode (the operator-absent cap has already been applied at install), so an enforce
// here is safe to act on. It returns:
//
//	exit  — 2 to BLOCK the stop (feed the sanctioned-route redirect to the model), 0 to allow it.
//	disp  — the typed disposition to stamp on the stop row (only meaningful when fired).
//	fired — false when the gate is inert: off/bad mode, no transcript, or a clean (non-gated) turn.
//
// enforce always BLOCKS a hardware-gated stop and redirects (there is no legitimate "give up for
// lack of local hardware" outcome — the fleet always offers a node or an operator handoff).
// warn/shadow always allow (soak/observe). Fail-open: a bad mode string is treated as "do not fire".
func runGuardHardwareGateGate(stderr io.Writer, rawMode, transcriptPath string) (exit int, disp guardStopDisposition, fired bool) {
	mode, err := normalizeGuardHardwareGateMode(rawMode)
	if err != nil {
		return 0, "", false // fail-open: never block a stop on a bad mode
	}
	if mode == guardPreCompactModeOff {
		return 0, "", false
	}
	finalTurn := hardwareGateFinalProseTurn(transcriptPath)
	if finalTurn == "" {
		return 0, "", false
	}
	rep := hwgatelint.Scan(finalTurn)
	if rep.Count == 0 {
		return 0, "", false // clean turn (or one that already named the sanctioned route)
	}
	return guardHardwareGateDecide(stderr, mode, rep.Findings[0])
}

// guardHardwareGateDecide is the pure ladder decision on a fired finding. The mode is already
// normalized and non-off, and top is the report's first (most-specific) finding. Split from the
// transcript read so a test can drive every rung from a finding without a transcript fixture.
func guardHardwareGateDecide(stderr io.Writer, mode string, top hwgatelint.Finding) (exit int, disp guardStopDisposition, fired bool) {
	switch mode {
	case guardPreCompactModeShadow:
		fmt.Fprintln(stderr, guardHardwareGateShadowLine(top))
		return 0, stopDispHardwareGateShadow, true
	case guardHardwareGateModeWarn:
		fmt.Fprintln(stderr, guardHardwareGateWarnLine(top))
		return 0, stopDispHardwareGateWarn, true
	default: // enforce
		fmt.Fprintln(stderr, guardHardwareGateContinueMessage(top))
		return 2, stopDispHardwareGateContinue, true
	}
}

// guardHardwareGateContinueMessage is the exact stderr text fed back to the MODEL when enforce
// blocks the stop (exit 2). It names the local resource the turn treated as a wall and hands over
// the fixed sanctioned-route redirect — dispatch to a compute node, or hand the operator the
// ready-to-run command — then points at the sanctioned wrap-up for the genuinely-blocked case.
func guardHardwareGateContinueMessage(top hwgatelint.Finding) string {
	return fmt.Sprintf("fak guard Stop: this turn stopped by declaring a LOCAL-hardware blocker (%s), but this host is the control point, not the compute boundary — stopping here strands work a sanctioned node could witness. Do not stop for lack of local hardware. Instead: %s Route to %s, then finish the turn. If the credential/bridge session is genuinely missing and you cannot even hand off a command, note it on one line (`no allowed path: <reason>`) and stop cleanly.",
		guardHardwareGateClassLabel(top), hwgatelint.Redirect, guardHardwareGateNodeText(top))
}

// guardHardwareGateWarnLine is the OPERATOR-facing soak line (exit 0, NOT fed to the model): it
// shows what enforce WOULD tell the model, so an operator can watch the rate before promoting.
func guardHardwareGateWarnLine(top hwgatelint.Finding) string {
	return fmt.Sprintf("fak guard Stop: [warn] hardware-gated stop (%s) — enforce would auto-redirect to a sanctioned compute node (%s) so the agent dispatches instead of stopping. (allowing the stop; soak mode).",
		guardHardwareGateClassLabel(top), guardHardwareGateNodeText(top))
}

// guardHardwareGateShadowLine is the terse OPERATOR-facing observe line (exit 0): the decision the
// gate WOULD reach, for metrics-only soaks.
func guardHardwareGateShadowLine(top hwgatelint.Finding) string {
	return fmt.Sprintf("fak guard Stop: shadow — hardware-gated stop (%s); enforce would auto-redirect to a sanctioned compute node. Allowing the stop.",
		guardHardwareGateClassLabel(top))
}

// guardHardwareGateClassLabel returns the finding's Class for a message, defaulting to a generic
// label if the scan left it blank (defensive; Scan always sets it when it yields a finding).
func guardHardwareGateClassLabel(top hwgatelint.Finding) string {
	if c := strings.TrimSpace(string(top.Class)); c != "" {
		return c
	}
	return "local-hardware"
}

// guardHardwareGateNodeText returns the sanctioned node the finding's Class most naturally routes
// to, defaulting to the generic fleet menu if the finding left it blank.
func guardHardwareGateNodeText(top hwgatelint.Finding) string {
	if n := strings.TrimSpace(top.Node); n != "" {
		return n
	}
	return "the sanctioned compute node for the task (GCP fak-realmodel, the DGX, da33, or nightrun)"
}
