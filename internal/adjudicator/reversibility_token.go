package adjudicator

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// ReversibilityConfirmToken derives the stable preview token. Confirmation keys
// AND incidental keys (a Bash call's free-text "description", and the client
// supervision knobs "timeout"/"timeout_ms"/"run_in_background" — see
// isIncidentalTokenKey) are excluded from the hashed payload, and the
// action-bearing command text is whitespace-normalized (see normalizeCommandForToken),
// so the token binds only to the call's effect-bearing subject and is stable across a
// re-proposal even when the client's confirm key, prose annotation, supervision knobs,
// OR cosmetic command spacing drift between proposals.
//
// The annotation exclusion is the fix for the confirm-loop that wedged a fleet
// session (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md): Claude Code's Bash
// tool carries a mandatory "description" the model regenerates every turn, so a
// re-proposal of the identical destructive command still hashed to a FRESH token
// and the advertised "re-propose byte-identical + add _fak_confirm" recovery
// could never converge. Excluding annotations cannot weaken any floor — they
// never influence the call's effect, the classifier never reads them, and hard
// denies still win before this rung.
func ReversibilityConfirmToken(class ReversibilityClass, tool string, args map[string]any) string {
	canon, _ := json.Marshal(argsForToken(args))
	sum := sha256.Sum256([]byte(string(class) + "\x00" + strings.ToLower(tool) + "\x00" + string(canon)))
	return "fak-" + hex.EncodeToString(sum[:8])
}

func argsWithoutConfirmation(args map[string]any) map[string]any {
	if len(args) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		if isConfirmationKey(k) {
			continue
		}
		out[k] = v
	}
	return out
}

// argsForToken canonicalizes a call for the confirm-token hash: it drops both
// the confirmation keys (so echoing _fak_confirm does not change the token) and
// the incidental annotation keys (so a reworded description does not either).
// The result binds the token to exactly the effect-bearing arguments. This is
// strictly narrower than the dispatch-time strip (argsWithoutConfirmation, which
// keeps the description so the tool still sees it): only the token hash ignores
// annotations, never dispatch.
func argsForToken(args map[string]any) map[string]any {
	if len(args) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		if isConfirmationKey(k) || isIncidentalTokenKey(k) {
			continue
		}
		// Cosmetic whitespace in the action-bearing command text (the model
		// re-indenting or padding runs of spaces between the SAME words) rotated
		// the nonce just like a reworded description did, so normalize it here —
		// the second half of the #2777 convergence fix. Only command-bearing
		// string keys are touched; every other argument is hashed verbatim.
		if s, ok := v.(string); ok && isCommandBearingKey(k) {
			out[k] = normalizeCommandForToken(s)
			continue
		}
		out[k] = v
	}
	return out
}

// commandBearingTokenKeys are the argument keys whose value IS the action-bearing
// command text (mirrors commandText's string-key list). Only these are whitespace-
// normalized in the confirm token; a benign non-command field that happens to
// carry incidental spacing is left byte-exact.
var commandBearingTokenKeys = map[string]bool{
	"command": true, "cmd": true, "shell": true, "script": true,
}

func isCommandBearingKey(k string) bool {
	return commandBearingTokenKeys[strings.ToLower(k)]
}

// normalizeCommandForToken collapses cosmetic whitespace in a command for the
// confirm-token hash: line-continuation reflows are unwrapped, leading/trailing
// whitespace is trimmed, and every run of UNQUOTED spaces/tabs between tokens
// becomes a single space, while whitespace INSIDE a quoted span is preserved
// byte-for-byte. Collapsing only unquoted runs is semantically faithful — the
// shell already word-splits on unquoted spacing, so `rm  foo` and `rm foo` are
// the same action — and cannot let two different commands collide, because quoted
// content (the only whitespace the shell keeps) is left untouched. This is why
// the fix does not weaken the token's binding while still letting a reflowed
// re-proposal converge (#2777).
func normalizeCommandForToken(cmd string) string {
	return strings.Join(quoteAwareFields(foldLineContinuations(cmd)), " ")
}

// foldLineContinuations deletes shell line-continuation sequences — a backslash
// immediately followed by a newline (LF or CRLF) OUTSIDE a quoted span — exactly
// as the shell does when it joins the wrapped lines, so a long command re-proposed
// wrapped across lines normalizes to its unwrapped form for the confirm token. A
// backslash-newline INSIDE quotes is literal content and is left intact, and a
// BARE unquoted newline — a command SEPARATOR, not whitespace — is never touched
// (folding it would merge two commands into one and could transplant a confirm
// across distinct actions). Other backslash escapes pass through unchanged, matching
// quoteAwareFields, which reads this view next.
func foldLineContinuations(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			end := quotedSpanEnd(s, i, dollar)
			b.WriteString(s[i:end])
			i = end
		case c == '\\' && i+1 < len(s) && s[i+1] == '\n':
			i += 2
		case c == '\\' && i+2 < len(s) && s[i+1] == '\r' && s[i+2] == '\n':
			i += 3
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// isIncidentalTokenKey reports whether an argument is incidental to the call's
// EFFECT — carried by the client for its own bookkeeping, never changing what the
// call does to the outside world. Clients re-emit these fields per turn, so folding
// them into the confirm token rotated it on every re-proposal and made the
// preview-confirm handshake unsatisfiable. Two categories qualify:
//
//   - Human-readable annotation: the free-text "description" Claude Code attaches to
//     every Bash call, or the "explanation" other harnesses use (#2777).
//   - Client supervision knobs: how long the CALLER waits ("timeout"/"timeout_ms")
//     and whether it detaches ("run_in_background"). These say how the caller
//     SUPERVISES the call, not what it does — `git push` pushes the same commits to
//     the same remote at any timeout, foreground or background — and none is read by
//     the classifier or rendered into the preview the operator acknowledges. They are
//     the residual axis of the confirm-loop (fak-private#21): the gate advertises
//     "re-propose byte-identical … the free-text description need not match", but a
//     model that re-proposed the identical outward-facing publish while nudging its timeout
//     (the ordinary reaction to a slow or timed-out call over a flaky remote bridge)
//     drew a FRESH token, so the advertised recovery could never converge and the
//     operator's repeated approval never became executable.
//
// Excluding a key here cannot weaken the floor: the token must bind to exactly the
// effect-bearing subject the preview shows, and every argument that steers the effect
// — a command's text, an MCP git_push `force`/`branch`, a `dangerouslyDisableSandbox`,
// a `workdir` — is still hashed verbatim, so a confirm can never be transplanted onto
// a materially different call. Hashing everything by default and excluding a NAMED
// incidental is deliberate: a novel effect-bearing arg binds automatically.
func isIncidentalTokenKey(k string) bool {
	switch strings.ToLower(k) {
	case "description", "explanation":
		return true
	case "timeout", "timeout_ms", "run_in_background":
		return true
	default:
		return false
	}
}

func hasConfirmationArg(args map[string]any) bool {
	for k := range args {
		if isConfirmationKey(k) {
			return true
		}
	}
	return false
}

func confirmationToken(args map[string]any) string {
	for k, v := range args {
		if !isConfirmationKey(k) {
			continue
		}
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func isConfirmationKey(k string) bool {
	switch strings.ToLower(k) {
	case ReversibilityConfirmArg, "_fak_confirm_token", "confirm_token", "confirmation_token":
		return true
	default:
		return false
	}
}
