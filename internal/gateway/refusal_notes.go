package gateway

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// refusalNote is one "actionable half of a refusal" renderer: given a refused
// ToolAdjudication it emits the in-band remedy string for that refusal facet
// (preview-confirm recipe, sanctioned alternative, livelock nudge, ...), or ""
// when the facet does not apply to this adjudication.
//
// The registry below is the single seam every content-channel wire folds over
// (#2750). A NEW refusal kind that carries actionable Meta is surfaced by
// appending its renderer here - NOT by hand-stitching another call at each
// denySummary / adjudicationNote site, which is exactly how reversibilityGateNote,
// remedyNote, and livelockInBandNote each grew a bespoke retrofit.
type refusalNote struct {
	render func(ToolAdjudication) string
	// confirmRecipe marks a non-empty render as a preview-confirm PAUSE rather
	// than a denial, so the trailer that sanctions re-proposing the same call
	// (only reversibilityGateNote today) can be scoped correctly.
	confirmRecipe bool
}

// refusalNotes is the ordered registry of actionable-half renderers. Order is
// load-bearing: the allowed/sanctioned alternative leads, while the confirm
// recipe and livelock constraint trail it. The wire tests pin this order.
var refusalNotes = []refusalNote{
	{render: remedyNote},
	{render: defaultDenyOperatorRemedy},
	{render: reversibilityGateNote, confirmRecipe: true},
	{render: livelockInBandNote},
}

// defaultDenyOperatorRemedy turns an otherwise bare DEFAULT_DENY token into a
// live recovery choice. The agent cannot widen its own floor, so the command is
// explicitly addressed to an operator outside the wrapped session. A short TTL
// keeps the workaround bounded while the normal fix (updating a stale harness
// profile or deliberately authoring the policy) is applied.
func defaultDenyOperatorRemedy(a ToolAdjudication) string {
	if reasonOrKind(a.Verdict) != "DEFAULT_DENY" {
		return ""
	}
	tool := refusalCommandTool(a.Tool)
	return "for autonomous agents: decompose into S0/S1 leaves, pivot to permitted tools, land safe verified partial progress, or emit a structured ABSTAIN record. Operator choice (outside this wrapped agent): run `fak guard allow --ttl 15m " + tool +
		"`; the live guard reloads the temporary repo overlay automatically, then retry the open task." +
		" Surface this choice to the operator instead of invoking it through the refused tool." +
		" If " + tool + " is a standard harness tool, update fak too; the built-in floor should cover it."
}

func refusalCommandTool(tool string) string {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return "<tool>"
	}
	for _, r := range tool {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') ||
			r == '_' || r == '-' || r == '.' || r == ':' {
			continue
		}
		return "<tool>"
	}
	return tool
}

// renderRefusalNotes folds every registered renderer over one refused
// adjudication and returns the space-joined actionable notes plus whether any of
// them was a preview-confirm recipe. Both content-channel call sites (denySummary
// and adjudicationNote) consume this instead of stitching each note by hand.
func renderRefusalNotes(a ToolAdjudication) (notes string, confirmRecipe bool) {
	parts := make([]string, 0, len(refusalNotes))
	for _, n := range refusalNotes {
		if s := n.render(a); s != "" {
			parts = append(parts, s)
			if n.confirmRecipe {
				confirmRecipe = true
			}
		}
	}
	return strings.Join(parts, " "), confirmRecipe
}

// compiledSidestepRemedy reports whether a refusal's sanctioned alternative names
// a compiled `fak` verb that sidesteps the gate entirely (e.g. `fak sync push` for
// a gated `git push`), as opposed to a mere preview affordance (`--dry-run`). The
// reversibility family hints that carry a compiled verb are the only remedies that
// contain a `fak ` token; the preview-only hints (npm publish --dry-run, a webhook
// preview) never do. When one is present the trailer leads with it — running the
// verb needs no confirm token at all — instead of the _fak_confirm recipe, whose
// (correctly command-bound) token an agent misread as rotating and looped on (#3306).
func compiledSidestepRemedy(a ToolAdjudication) bool {
	return strings.Contains(a.Verdict.Detail["remedy"], "fak ")
}

// denySummary renders a short human-readable note when every proposed tool_call
// was refused, so a client that ignores the `fak` extension still adapts.
func denySummary(adjs []ToolAdjudication) string {
	parts := make([]string, 0, len(adjs))
	for _, a := range adjs {
		constraint := fmt.Sprintf("%s: %s (%s/%s)", a.Tool, a.Verdict.Kind, reasonOrKind(a.Verdict), a.Verdict.Disposition)
		notes, _ := renderRefusalNotes(a)
		if notes == "" {
			notes = errorAffordance(reasonOrKind(a.Verdict))
		}
		parts = append(parts, notes+" Constraint: "+constraint)
	}
	// Emit-time reframe (#3566): flip any unambiguous negative idiom to lead with the
	// affordance before this note is pushed to the model. The pass is token-superset
	// safe, so every structured reason token (POLICY_BLOCK, OFF_TRUNK, ...) and every
	// load-bearing judgement prohibition ("do not re-propose") survives byte-for-byte.
	return negframe.ReframeFakOnly(
		negframe.Fak("Allowed next step for each refused tool call: "),
		negframe.Opaque(strings.Join(parts, "; ")),
	)
}

// adjudicationNote renders a short, agent-readable summary of the kernel's
// non-trivial decisions (drops + repairs) on a turn, for clients that read only
// the in-band content channel and never the `fak` extension — Claude Code on the
// Anthropic wire is exactly that client. It is the difference between a denied
// call SILENTLY VANISHING (the agent re-proposes it, or proceeds on a false
// premise) and the agent being told "fak refused rm -rf for POLICY_BLOCK" so it
// can adapt. Returns "" when every call was a clean ALLOW (nothing worth saying).
func adjudicationNote(adjs []ToolAdjudication) string {
	denied := make([]string, 0, len(adjs))
	deniedAdjs := make([]ToolAdjudication, 0, len(adjs))
	repaired := make([]string, 0, len(adjs))
	allowedLoops := make([]string, 0, len(adjs))
	hasConfirmRecipe := false
	hasCompiledSidestep := false
	for _, a := range adjs {
		switch {
		case !a.Admitted:
			constraint := fmt.Sprintf("%s (%s/%s)", a.Tool, reasonOrKind(a.Verdict), a.Verdict.Disposition)
			notes, recipe := renderRefusalNotes(a)
			if notes == "" {
				notes = errorAffordance(reasonOrKind(a.Verdict))
			}
			entry := notes + " Constraint: " + constraint
			if recipe {
				hasConfirmRecipe = true
				// A compiled sidestep (a `fak` verb that avoids the gate
				// entirely) is the resolution to LEAD with; the confirm-token
				// dance is the fallback for when only the raw call will do.
				if compiledSidestepRemedy(a) {
					hasCompiledSidestep = true
				}
			}
			denied = append(denied, entry)
			deniedAdjs = append(deniedAdjs, a)
		case a.Admitted && a.Livelock != nil:
			allowedLoops = append(allowedLoops, livelockInBandNote(a))
		case a.Verdict.Kind == "TRANSFORM":
			repaired = append(repaired, a.Tool)
		}
	}
	if len(denied) == 0 && len(repaired) == 0 && len(allowedLoops) == 0 {
		return ""
	}
	fragments := make([]negframe.Fragment, 0, 12)
	writeFak := func(text string) { fragments = append(fragments, negframe.Fak(text)) }
	writeOpaque := func(text string) { fragments = append(fragments, negframe.Opaque(text)) }
	writeFak("[fak] ")
	if len(denied) > 0 {
		writeFak("Allowed next step for ")
		writeFak(strconv.Itoa(len(denied)))
		writeFak(" refused tool call(s): ")
		joined := strings.Join(denied, "; ")
		writeOpaque(joined)
		if !strings.HasSuffix(joined, ".") {
			writeFak(".")
		}
		// The blanket "do not re-propose" trailer contradicts the preview-confirm
		// recipe, whose sanctioned recovery IS re-proposing the same call (plus the
		// confirm key). Witnessed wedging a fleet session for 1h+: the agent obeyed
		// the trailer, never echoed the token, and the push never happened. When any
		// denied call carries a confirm recipe, the trailer must except it.
		//
		// Emphasis (#3306): when a compiled sidestep exists (a `fak` verb that
		// avoids the gate outright, e.g. `fak sync push` for `git push`), the
		// trailer LEADS with it and DEMOTES the _fak_confirm recipe to a fallback.
		// Leading with the token dance instead re-wedged an obedient agent that
		// chased a (correctly) command-bound token it wrongly believed was rotating
		// while the sidestep that worked first try sat one demoted sentence away.
		switch {
		case hasConfirmRecipe && hasCompiledSidestep:
			writeFak(" A preview-confirm refusal is a pause, not a denial — and the simplest resolution is the sanctioned compiled alternative named above, which sidesteps the gate entirely and needs no token. Prefer it. Only if you specifically need the raw gated call, re-propose that same call byte-identical with only the _fak_confirm key added. This is per-tool feedback, not a session stop. Do not re-propose any other refused call unchanged; choose an allowed alternative. A session stop only comes from a declared stop policy.")
		case hasConfirmRecipe:
			writeFak(" A preview-confirm refusal is a pause, not a denial: the sanctioned next step is to re-propose that same call with only the _fak_confirm key added. This is per-tool feedback, not a session stop. Do not re-propose any other refused call unchanged; choose an allowed alternative. A session stop only comes from a declared stop policy.")
		default:
			writeFak(" This is per-tool feedback, not a session stop. Do not re-propose a refused call unchanged; fix the arguments/tool choice or choose an allowed alternative. A session stop only comes from a declared stop policy.")
		}
		writeOpaque(complaintHint(deniedAdjs))
	}
	if len(repaired) > 0 {
		if len(denied) > 0 {
			writeFak(" ")
		}
		writeFak("repaired arguments for: ")
		writeOpaque(strings.Join(repaired, ", "))
		writeFak(".")
	}
	if len(allowedLoops) > 0 {
		if len(denied) > 0 || len(repaired) > 0 {
			writeFak(" ")
		}
		writeFak("observed repeated admitted tool call(s): ")
		writeOpaque(strings.Join(allowedLoops, "; "))
		writeFak(". This is advisory: do not repeat a successful identical call unchanged unless you can name the new evidence it will produce.")
	}
	// Emit-time reframe (#3566/#4430): only fak-authored framing enters the
	// positive-voice pass. Tool names, remedies, notes, and other external spans remain
	// byte-identical, even when they contain text that resembles a reframe idiom.
	return journalReframeFragments(reframeJournalPath(), "", "gateway.refusal_note", reframeJournalArm(), fragments, time.Now())
}

func prependAdjudicationContentNote(content string, adjs []ToolAdjudication) string {
	note := adjudicationNote(adjs)
	if note == "" {
		return content
	}
	if strings.TrimSpace(content) == "" {
		return note
	}
	return note + "\n" + content
}

func livelockInBandNote(a ToolAdjudication) string {
	if a.Livelock == nil {
		return ""
	}
	note := fmt.Sprintf("LIVELOCK_DETECTED repeat=%d repeated_call=%s approach=%s",
		a.Livelock.RepeatCount,
		livelockCallLabel(*a.Livelock),
		a.Livelock.SuggestedChange)
	if a.Livelock.Escalate {
		// The retryable fuse itself was ignored turn after turn. This refusal is now
		// TERMINAL: the turn escalates to a deny-all and the session's bounded give-up
		// policy will end it. Tell the model the loop is over so it stops and reports
		// the blocker instead of burning more tokens re-proposing the same call.
		note += " ABORT=terminal (this identical call has been refused too many times; it will NOT be admitted — stop retrying, end the turn, and report the blocker with a witness)"
	} else if a.Livelock.Fuse {
		// The advisory note repeated for several turns and the loop kept going, so the
		// fuse converted this call into a refusal. Say so plainly: the model must change
		// approach, not re-propose — re-proposing the identical call fuses again.
		note += " fuse=armed (this repeated call was refused; changing approach is required, not optional)"
	}
	return note
}

// complaintHint surfaces the agent's APPEAL channel (`fak complain`) on every
// refusal, so a governed agent that judges a guard decision wrong has a
// sanctioned, in-band way to say so — instead of silently looping, giving up, or
// proceeding on a false premise. A false-positive DENY is byte-identical to a
// correct one in the decision journal, so the kernel's own RSI fold
// (guardroute / `fak guard-verdict-rsi`) cannot detect it: only the agent that
// made the call knows it was legitimate (internal/guardcomplaint).
//
// The hint LEADS the reader back to adapting first, so it never reads as an
// invitation to appeal in lieu of choosing an allowed alternative; it is the
// escape hatch for when the agent is confident the guard, not its call, is
// wrong. When the turn has exactly one denial the concrete reason/tool are
// substituted into the command so the appeal is copy-pasteable, AND the call's
// args_digest is emitted as an exact `--args-digest` selector so `--from-journal`
// binds the witness by construction rather than filing witness-less on a busy
// journal (guardcomplaint.SelectDenial refuses an ambiguous reason/tool match).
// A mixed turn keeps the <REASON>/<TOOL> placeholders and emits no selector,
// rather than misattributing one call's scope — or digest — to another.
func complaintHint(denied []ToolAdjudication) string {
	if len(denied) == 0 {
		return ""
	}
	reason, tool := "<REASON>", "<TOOL>"
	// selector binds the appeal to the EXACT refused call so `--from-journal` attaches
	// its witness by construction, instead of hitting SelectDenial's ambiguous→no-witness
	// path (guardcomplaint.SelectDenial) — which is what a busy journal serves for a bare
	// reason/tool match, silently filing the appeal witness-less. args_digest is the
	// per-call identity the DENY journal row carries, and it is only unambiguous for a
	// single denial, so a mixed turn keeps the placeholders and emits no selector.
	selector := ""
	if len(denied) == 1 {
		if r := strings.TrimSpace(reasonOrKind(denied[0].Verdict)); r != "" {
			reason = r
		}
		if t := strings.TrimSpace(denied[0].Tool); t != "" {
			tool = t
		}
		if d := strings.TrimSpace(denied[0].ArgsDigest); d != "" {
			selector = " --args-digest " + d
		}
	}
	return fmt.Sprintf(" Judge a refusal wrong — a false positive, or a gate that"+
		" over-refuses? Appeal it: `fak complain --summary \"…\" --reason %s --tool %s"+
		" --from-journal%s` files one deduplicating issue with the witnessed verdict"+
		" attached. Adapt first; appeal only when you are confident the guard, not"+
		" your call, is wrong.", reason, tool, selector)
}

var (
	reFakHeaderTag        = regexp.MustCompile(`\[FAK GATE:\s*([^\]]+)\]`)
	reParenVerdict        = regexp.MustCompile(`\(([A-Z0-9_]+)/[A-Z0-9_]+\)`)
	reExplicitReasonKw    = regexp.MustCompile(`(?i)(?:^|[\r\n\s,;])reason:\s*([A-Za-z0-9_]+)`)
	reNextActionPrefix    = regexp.MustCompile(`(?i)^(?:next\s*action|next\s*step|actionable\s*affordance|action):\s*`)
	reBoilerplateTrailers = []*regexp.Regexp{
		regexp.MustCompile(`(?i)This is per-tool feedback, not a session stop\..*`),
		regexp.MustCompile(`(?i)Judge a refusal wrong.*`),
		regexp.MustCompile(`(?i)A session stop only comes from a declared stop policy\..*`),
		regexp.MustCompile(`(?i)Do not re-propose.*`),
		regexp.MustCompile(`(?i)Contact [^\s]+ for policy adjustments\..*`),
		regexp.MustCompile(`(?i)For support, visit:.*`),
	}
	reBoilerplateHeaders = []*regexp.Regexp{
		regexp.MustCompile(`(?i)^refusal\s+details:\s*`),
		regexp.MustCompile(`(?i)^detailed\s+policy\s+refusal[^:]*:\s*`),
		regexp.MustCompile(`(?i)^error:\s*`),
		regexp.MustCompile(`(?i)^allowed\s+next\s+step[^\:]*:\s*`),
	}
	reMultiSpace        = regexp.MustCompile(`[ \t]+`)
	reNextActionMarkers = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:next\s+action|next\s+step|actionable\s+affordance):\s*([^\r\n]+)`),
		regexp.MustCompile(`(?i)\baction:\s*([^\r\n]+)`),
		regexp.MustCompile(`(?i)\bsanctioned\s+alternative:\s*([^\r\n;]+)`),
		regexp.MustCompile(`(?i)\boperator\s+choice(?:\s*\([^)]*\))?:\s*([^\r\n;]+)`),
		regexp.MustCompile(`(?i)\bremedy:\s*([^\r\n;]+)`),
		regexp.MustCompile(`(?i)\ballowed\s+next\s+step(?:\s+for[^\:]*)?:\s*([^\r\n;]+)`),
	}
)

func cleanReasonToken(r string) string {
	r = strings.TrimSpace(r)
	if strings.HasPrefix(strings.ToUpper(r), "[FAK GATE:") {
		r = strings.TrimPrefix(r, "[FAK GATE:")
		r = strings.TrimPrefix(r, "[fak gate:")
		r = strings.TrimSuffix(r, "]")
	}
	r = strings.TrimSpace(r)
	if idx := strings.Index(r, "/"); idx > 0 {
		r = r[:idx]
	}
	var buf strings.Builder
	for _, ch := range r {
		if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' {
			buf.WriteRune(ch)
		} else {
			break
		}
	}
	return strings.ToUpper(buf.String())
}

func extractReasonFromText(text string) string {
	if m := reFakHeaderTag.FindStringSubmatch(text); len(m) > 1 {
		if c := cleanReasonToken(m[1]); c != "" {
			return c
		}
	}
	if m := reParenVerdict.FindStringSubmatch(text); len(m) > 1 {
		if c := cleanReasonToken(m[1]); c != "" {
			return c
		}
	}
	if m := reExplicitReasonKw.FindStringSubmatch(text); len(m) > 1 {
		if c := cleanReasonToken(m[1]); c != "" {
			return c
		}
	}
	knownReasons := []string{
		"POLICY_BLOCK",
		"DEFAULT_DENY",
		"REVERSIBILITY_CONFIRM",
		"FILE_ADMISSION",
		"OFF_TRUNK",
		"OUT_OF_TREE_WRITE",
		"SELF_MODIFY",
		"OVERHEAD_BUDGET_EXCEEDED",
		"INVALID_TOOL_ARGUMENTS",
		"LIVELOCK_DETECTED",
		"NEEDS_WITNESS",
		"PERMISSION_DENIED",
		"RECOVERY_REQUIRED",
		"CORE_LOCK",
	}
	for _, kr := range knownReasons {
		if strings.Contains(text, kr) {
			return kr
		}
	}
	return ""
}

func extractNextActionAndCleanMessage(msg string) (string, string) {
	msgClean := msg
	var action string

	for _, pat := range reNextActionMarkers {
		loc := pat.FindStringSubmatchIndex(msgClean)
		if len(loc) >= 4 {
			captured := strings.TrimSpace(msgClean[loc[2]:loc[3]])
			before := msgClean[:loc[0]]
			after := msgClean[loc[1]:]
			msgClean = before + " " + after

			delimiters := []string{
				"Constraint:",
				"constraint:",
				"This is per-tool feedback",
				"Judge a refusal",
				"Session ID:",
				"Audit ID:",
				"Timestamp:",
			}
			for _, d := range delimiters {
				if idx := strings.Index(captured, d); idx > 0 {
					trailer := captured[idx:]
					captured = strings.TrimSpace(captured[:idx])
					msgClean += " " + trailer
				}
			}
			action = captured
			break
		}
	}

	return action, msgClean
}

func defaultAffordanceForReason(reason string) string {
	if aff := errorAffordance(reason); aff != "" && aff != reason {
		return aff
	}
	switch reason {
	case "REVERSIBILITY_CONFIRM":
		return "re-propose with _fak_confirm key or choose sanctioned alternative"
	case "DEFAULT_DENY":
		return "decompose into S0/S1 leaves, pivot to permitted tools, or emit a structured ABSTAIN record"
	case "LIVELOCK_DETECTED":
		return "change approach; identical repeated calls are refused"
	case "NEEDS_WITNESS":
		return "attach required witness artifact and retry"
	case "FILE_ADMISSION":
		return "stage only admitted workspace files"
	case "OFF_TRUNK":
		return "commit on main with fak commit --path <owned-path> -m <message>"
	case "OUT_OF_TREE_WRITE":
		return "write inside the workspace or place scratch data in the OS temp directory"
	case "SELF_MODIFY":
		return "route an authorized core-lock edit through maintenance witness or compiled verb"
	case "OVERHEAD_BUDGET_EXCEEDED":
		return "reduce overhead or update declared envelope"
	case "INVALID_TOOL_ARGUMENTS":
		return "correct tool arguments to match schema and retry"
	default:
		return "choose an admitted tool, land safe partial deliverables, or emit a structured ABSTAIN record"
	}
}

func defaultBriefReasonForReason(reason string) string {
	switch reason {
	case "POLICY_BLOCK":
		return "tool execution blocked by policy"
	case "DEFAULT_DENY":
		return "unregistered tool blocked by default-deny floor"
	case "REVERSIBILITY_CONFIRM":
		return "state-modifying operation requires confirmation"
	case "OFF_TRUNK":
		return "commits must be made on main trunk"
	case "OUT_OF_TREE_WRITE":
		return "write outside workspace blocked"
	case "SELF_MODIFY":
		return "core-lock modification blocked"
	case "INVALID_TOOL_ARGUMENTS":
		return "tool arguments invalid"
	case "LIVELOCK_DETECTED":
		return "repeated identical call blocked"
	case "NEEDS_WITNESS":
		return "operation requires witness artifact"
	case "FILE_ADMISSION":
		return "file path admission refused"
	default:
		if reason != "" {
			return fmt.Sprintf("operation blocked by %s gate", reason)
		}
		return "operation blocked by gate"
	}
}

func selectInformativeLine(text string) string {
	text = strings.ReplaceAll(text, "\r", "")
	lines := strings.Split(text, "\n")
	var informative []string
	var violationLine string
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		lower := strings.ToLower(line)
		if strings.HasPrefix(lower, "refusal details") ||
			strings.HasPrefix(lower, "detailed policy refusal") ||
			strings.HasPrefix(lower, "error:") ||
			strings.HasPrefix(lower, "reason:") ||
			strings.HasPrefix(lower, "status:") ||
			strings.HasPrefix(lower, "audit id:") ||
			strings.HasPrefix(lower, "session id:") ||
			strings.HasPrefix(lower, "diagnostic id:") ||
			strings.HasPrefix(lower, "timestamp:") ||
			strings.HasPrefix(lower, "turn:") ||
			strings.HasPrefix(lower, "rule id:") ||
			strings.HasPrefix(lower, "policy file:") {
			continue
		}
		if strings.HasPrefix(lower, "violation:") {
			violationLine = strings.TrimSpace(line[len("violation:"):])
			continue
		}
		informative = append(informative, line)
	}
	if violationLine != "" {
		return violationLine
	}
	if len(informative) == 0 {
		return text
	}
	for _, l := range informative {
		low := strings.ToLower(l)
		if strings.Contains(low, "violation") ||
			strings.Contains(low, "blocked") ||
			strings.Contains(low, "refused") ||
			strings.Contains(low, "denied") ||
			strings.Contains(low, "prohibited") ||
			strings.Contains(low, "forbidden") {
			return l
		}
	}
	return informative[0]
}

func cleanBriefReason(reason, msg string) string {
	msg = selectInformativeLine(msg)

	msg = reFakHeaderTag.ReplaceAllString(msg, "")
	msg = strings.TrimPrefix(msg, "[fak]")
	msg = strings.TrimSpace(msg)

	for _, re := range reBoilerplateTrailers {
		msg = re.ReplaceAllString(msg, "")
	}

	for _, re := range reBoilerplateHeaders {
		msg = re.ReplaceAllString(msg, "")
	}

	msg = strings.ReplaceAll(msg, "\r", " ")
	msg = strings.ReplaceAll(msg, "\n", " ")
	msg = reMultiSpace.ReplaceAllString(msg, " ")
	msg = strings.TrimSpace(msg)

	if strings.HasPrefix(strings.ToLower(msg), "constraint:") {
		msg = strings.TrimSpace(msg[len("constraint:"):])
	}
	if idx := strings.Index(msg, "Constraint:"); idx > 15 {
		msg = strings.TrimSpace(msg[:idx])
	}

	if len(msg) > 160 {
		if idx := strings.Index(msg[20:], ". "); idx >= 0 && (20+idx+1) <= 160 {
			msg = strings.TrimSpace(msg[:20+idx+1])
		} else if idx := strings.Index(msg[20:], "; "); idx >= 0 && (20+idx) <= 160 {
			msg = strings.TrimSpace(msg[:20+idx])
		} else if lastSpace := strings.LastIndex(msg[:140], " "); lastSpace > 30 {
			msg = strings.TrimSpace(msg[:lastSpace]) + "..."
		}
	}

	msg = negframe.Reframe(msg)

	if msg == "" {
		msg = defaultBriefReasonForReason(reason)
	}

	msg = strings.ReplaceAll(msg, "\r", " ")
	msg = strings.ReplaceAll(msg, "\n", " ")
	msg = reMultiSpace.ReplaceAllString(msg, " ")
	return strings.TrimSpace(msg)
}

// FormatCompactRefusalNote formats a refusal into a compact 2-line envelope:
// Line 1: [FAK GATE: <REASON>] <Brief affordance-first reason>
// Line 2: Next Action: <Actionable affordance>
func FormatCompactRefusalNote(reason string, message string, nextAction string) string {
	cleanedReason := cleanReasonToken(reason)
	if cleanedReason == "" {
		cleanedReason = extractReasonFromText(message)
	}
	if cleanedReason == "" {
		cleanedReason = "POLICY_BLOCK"
	}

	extractedAction, cleanedMsg := extractNextActionAndCleanMessage(message)
	action := strings.TrimSpace(nextAction)
	if action != "" {
		action = reNextActionPrefix.ReplaceAllString(action, "")
		action = strings.TrimSpace(action)
	} else if extractedAction != "" {
		action = extractedAction
	} else {
		action = defaultAffordanceForReason(cleanedReason)
	}

	action = strings.ReplaceAll(action, "\r", " ")
	action = strings.ReplaceAll(action, "\n", " ")
	action = reMultiSpace.ReplaceAllString(action, " ")
	action = strings.TrimSpace(action)

	briefReason := cleanBriefReason(cleanedReason, cleanedMsg)

	line1 := fmt.Sprintf("[FAK GATE: %s] %s", cleanedReason, briefReason)
	line2 := fmt.Sprintf("Next Action: %s", action)

	line1 = strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(line1, "\r", " "), "\n", " "))
	line2 = strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(line2, "\r", " "), "\n", " "))

	return line1 + "\n" + line2
}

// CompressRefusalNote compresses an existing multi-line or verbose refusal note
// into the compact 2-line envelope by extracting reason, message, and next action.
func CompressRefusalNote(rawNote string) string {
	reason := extractReasonFromText(rawNote)
	return FormatCompactRefusalNote(reason, rawNote, "")
}

// CompactDenySummary renders a compressed 2-line refusal note envelope for refused tool calls.
func CompactDenySummary(adjs []ToolAdjudication) string {
	raw := denySummary(adjs)
	return CompressRefusalNote(raw)
}
