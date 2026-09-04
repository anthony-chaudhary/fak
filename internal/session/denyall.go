package session

// denyall.go — the bounded deny-all loop breaker (issue #2593).
//
// THE BUG IT CLOSES. A guarded Codex/fakc `/goal` session can spin when every
// continuation proposes the same refused host tool. In the audited `fakc` loop
// Codex repeatedly tried `update_plan`; `fak guard` correctly returned
// `DENY (DEFAULT_DENY/TERMINAL)`, but the active goal kept auto-continuing and
// re-entering the same failure — 211k tokens / 257s of zero-progress retries
// (docs/notes/HARNESS-TOOL-DIALECT-GUARD-FLOOR-2026-07-03.md). The Claude Code
// path already has a bounded auto-continue Stop hook (cmd/fak/guard_stophook.go)
// that reads the gateway's `fak_guard_deny_all_consecutive` gauge. That hook is
// Claude-only (Codex/OpenCode have no Stop hook), and the gateway gauge it polls
// is GLOBAL and carries NEITHER the refused tool name NOR the reason — so nothing
// could detect the #2593 shape: the SAME tool, refused for the SAME reason, turn
// after turn, with no progress.
//
// THIS LEAF. DenyAllBreaker is the harness-AGNOSTIC, per-session kernel primitive
// both harness paths can share. It is a pure state machine: each served turn is
// fed to Observe as a DenyAllObservation; Observe folds it and returns a
// DenyAllVerdict — continue, or a bounded STOP that ends auto-continuation and
// surfaces a concrete diagnostic. The "stuck" test is exactly the issue's four
// criteria: the turn's only/dominant tool call was denied, the denied tool name
// is UNCHANGED from the prior stuck turn, the reason is UNCHANGED, and no useful
// tool progress occurred between attempts. After a small threshold it stops.
//
// FAIL-CLOSED INVARIANT. The breaker only ever stops AUTO-CONTINUATION. It NEVER
// auto-allows a refused tool and it NEVER weakens the capability floor — a
// genuinely dangerous repeated denial hits the threshold and STOPS (the floor
// keeps denying; the loop just stops feeding it the same refused call). That is
// the issue's "still fails closed and does not auto-allow" criterion, and it is
// structural: Observe has no path that lifts a refusal. The diagnostic's recovery
// line is the ONLY thing that varies with the disposition — for host PLUMBING it
// names a floor-COVERAGE fix; for an EFFECTFUL tool it explicitly says do NOT
// allow-list without mirroring the dangerous-command arg rules.
//
// FENCE / SCOPE. This is the detection+decision+diagnostic primitive plus its
// replay fixture (the issue's named witness). The live wiring that feeds
// observations from the served-turn adjudication path and reads the stop in the
// Codex `/goal` runner is the integration step this leaf enables; it lives where
// the gateway already records deny-all outcomes (internal/gateway) and where the
// goal runner decides to re-prompt, both outside this leaf's tree. The breaker is
// pure data + logic (stdlib-only), so the regression fixture exercises the REAL
// decision, not a stub.

// ReasonDenyAllLoop is the closed stop token the breaker stamps when it bounds a
// deny-all auto-continue loop — so "why did this session stop re-prompting" is a
// checkable field, never free text, mirroring the Reason* discipline Decide uses.
const ReasonDenyAllLoop = "DENY_ALL_LOOP_BOUNDED"

// DenyAllDefaultThreshold is the default consecutive-stuck turn count at which the
// breaker stops auto-continuing. It is deliberately SMALL: a model that re-proposes
// the same refused call three turns running is not going to route around it on the
// fourth, and every extra continuation is pure waste (the audited loop burned 211k
// tokens this way). It mirrors the first escalation rung of the Claude Stop-hook
// ladder (guardStopHookDefaultWarn) so the two paths bound at the same depth.
const DenyAllDefaultThreshold = 3

// DenyAllDisposition classifies the refused tool so the diagnostic's recovery line
// matches the failure class. The SAME bounded-stop fires either way (fail-closed is
// structural); only the advice differs — a coverage fix for plumbing, a "do not
// allow-list" warning for an effectful tool.
type DenyAllDisposition int

const (
	// DenyAllHostPlumbing marks an orchestration / read-only host tool
	// (update_plan, todowrite, tool_search, MCP list/read, planning/state helpers). A
	// DEFAULT_DENY here is a harness-dialect COVERAGE problem — the tool's schema
	// is plan-state / read-only, so the recovery is to admit it on the floor, not
	// to weaken the guard.
	DenyAllHostPlumbing DenyAllDisposition = iota
	// DenyAllEffectful marks a write / shell / mutation tool. The floor is CORRECT
	// to deny it; the recovery must NOT auto-allow it without mirroring the
	// dangerous-command argument rules the named shell aliases carry.
	DenyAllEffectful
)

// String renders the disposition as the lowercase noun the diagnostic embeds.
func (d DenyAllDisposition) String() string {
	switch d {
	case DenyAllHostPlumbing:
		return "host-plumbing"
	case DenyAllEffectful:
		return "effectful"
	default:
		return "unknown"
	}
}

// DenyAllObservation is one served turn's deny-all shape, fed to the breaker. It is
// the minimal information needed to run the issue's four-criterion stuck test.
type DenyAllObservation struct {
	// Tool is the name of the only/dominant tool call the floor refused this turn
	// (e.g. "update_plan"). Empty means the turn had NO refused call — a clean or
	// pure-text turn that resets the run.
	Tool string
	// Reason is the refusal reason token (DEFAULT_DENY in the observed case). It is
	// part of the "unchanged" test: a run only counts while the reason is identical
	// across turns, so a flap between DEFAULT_DENY and a policy deny re-seeds.
	Reason string
	// Progress is true if the turn made useful tool progress — at least one
	// surviving call, or meaningful non-tool work. A turn with Progress resets the
	// run (the loop is NOT stuck if it is getting somewhere).
	Progress bool
	// Disposition classifies Tool for the diagnostic's recovery line. It does not
	// change the decision — the bounded stop fires either way.
	Disposition DenyAllDisposition
}

// DenyAllVerdict is the result of folding one observation. Continue is the common
// case; Stopped ends auto-continuation with a closed reason and a diagnostic.
type DenyAllVerdict struct {
	// Continue is true when the loop may keep auto-continuing (under threshold, or
	// the turn reset the run). False once the bounded stop fires.
	Continue bool
	// Stopped is true exactly when the consecutive-stuck threshold was reached this
	// turn — the loop must stop re-prompting. Mutually exclusive with Continue.
	Stopped bool
	// Consecutive is the current run depth AFTER this observation (0 on a reset, 1
	// for the first stuck turn, up to Threshold on the stop turn).
	Consecutive int
	// Threshold is the effective bound the verdict was measured against (the
	// breaker's configured value or the default), surfaced so a caller rendering
	// the diagnostic does not need to re-derive it.
	Threshold int
	// Reason is the closed stop token (ReasonDenyAllLoop) when Stopped, else "".
	Reason string
	// Diagnostic is the surfaced explanation when Stopped, else "". It names the
	// refused tool, the reason, the disposition, the floor source, and the
	// disposition-specific recovery — never a recommendation to auto-allow.
	Diagnostic string
}

// DenyAllBreaker is the per-session bounded deny-all loop breaker. The zero value
// is usable: a zero Threshold falls back to DenyAllDefaultThreshold (so an
// unconfigured breaker bounds at the default rather than never firing). It is not
// safe for concurrent use without external serialization — like the rest of a
// session's per-turn fold, it is driven from one turn boundary at a time.
type DenyAllBreaker struct {
	// Threshold is the consecutive-stuck turn count at which Observe returns a
	// bounded stop. <=0 falls back to DenyAllDefaultThreshold.
	Threshold int
	// FloorSource names the capability-floor origin the diagnostic points at for
	// recovery (the embedded guard policy path). Empty falls back to the canonical
	// embedded floor path so the diagnostic always names a concrete source.
	FloorSource string

	// run state — the consecutive count and the tool/reason/disposition that seed it.
	count       int
	tool        string
	reason      string
	disposition DenyAllDisposition
}

// effectiveThreshold returns the configured bound or the default, so a zero-value
// breaker still bounds.
func (b *DenyAllBreaker) effectiveThreshold() int {
	if b.Threshold > 0 {
		return b.Threshold
	}
	return DenyAllDefaultThreshold
}

// effectiveFloorSource returns the configured floor source or the canonical embedded
// guard-policy path, so the diagnostic always names a concrete recovery target.
func (b *DenyAllBreaker) effectiveFloorSource() string {
	if b.FloorSource != "" {
		return b.FloorSource
	}
	return "cmd/fak/guard-default-policy.json"
}

// reset clears the run to zero (a clean or progressing turn ends any stuck streak).
// Caller does not need the lock — Observe is single-threaded per session.
func (b *DenyAllBreaker) reset() {
	b.count = 0
	b.tool = ""
	b.reason = ""
	b.disposition = 0
}

// Reset clears any in-progress stuck run. A loop driver calls this at an objective
// boundary (a new /goal, a session resume, a manual retry) to drop a stale streak
// without synthesizing a fake clean observation — so a breaker carried across
// objectives cannot false-stop on a fresh goal because of a run the previous goal
// accrued. It is idempotent and a no-op on an already-clear breaker.
func (b *DenyAllBreaker) Reset() {
	b.reset()
}

// Observe folds one served turn's deny-all shape and returns the verdict. It is the
// ONE call a loop driver makes per turn (the deny-all twin of Table.Decide). The
// decision is pure over (breaker state, observation); the diagnostic is built only
// on a stop so the hot path pays nothing for string formatting.
//
// Stuck test (all four, mirroring the issue): Tool is non-empty (a call was
// refused), Progress is false (no useful work this turn), Tool equals the prior
// stuck turn's Tool, and Reason equals the prior Reason. A stuck turn that changes
// EITHER Tool or Reason re-seeds the run at 1 (a flap is not the same spin). A
// clean or progressing turn resets the run to 0.
func (b *DenyAllBreaker) Observe(o DenyAllObservation) DenyAllVerdict {
	threshold := b.effectiveThreshold()

	// A clean turn (no refused call) or a turn that made progress ends the streak.
	if o.Tool == "" || o.Progress {
		b.reset()
		return DenyAllVerdict{Continue: true, Consecutive: 0, Threshold: threshold}
	}

	// Deny-all turn with no progress. Same tool AND same reason as the prior stuck
	// turn extends the run; otherwise this is the seed of a new (different) spin.
	if b.count > 0 && o.Tool == b.tool && o.Reason == b.reason {
		b.count++
	} else {
		b.count = 1
		b.tool = o.Tool
		b.reason = o.Reason
		b.disposition = o.Disposition
	}

	if b.count < threshold {
		return DenyAllVerdict{Continue: true, Consecutive: b.count, Threshold: threshold}
	}

	// Bounded stop: the threshold was reached. The disposition recorded on the run
	// (set when the run seeded) drives the recovery line — a tool cannot sneak a
	// softer diagnostic by flipping its disposition on the final turn, because the
	// run only extended while Tool+Reason stayed identical.
	return DenyAllVerdict{
		Continue:    false,
		Stopped:     true,
		Consecutive: b.count,
		Threshold:   threshold,
		Reason:      ReasonDenyAllLoop,
		Diagnostic:  denyAllDiagnostic(b.tool, b.reason, b.disposition, b.count, threshold, b.effectiveFloorSource()),
	}
}

// denyAllDiagnostic builds the surfaced explanation for a bounded stop. It carries
// every field the issue names — refused tool, reason + disposition, floor source,
// and a disposition-specific recovery — and NEVER recommends auto-allowing a
// refused tool. For host plumbing it names the coverage fix; for an effectful tool
// it warns against allow-listing without the dangerous-command arg rules.
func denyAllDiagnostic(tool, reason string, dispo DenyAllDisposition, consecutive, threshold int, floorSource string) string {
	var recovery string
	switch dispo {
	case DenyAllHostPlumbing:
		recovery = "this is a harness-dialect COVERAGE problem, not a dangerous action: the tool's schema is plan-state / read-only, so admit \"" +
			tool + "\" (and its namespace-qualified spellings) on the harness profile / embedded floor. Do NOT disable fak guard."
	case DenyAllEffectful:
		recovery = denyAllEffectfulRecovery(tool, reason)
	default:
		recovery = "inspect why the floor refuses \"" + tool + "\" before changing anything (fak guard --dump-policy)."
	}
	return "fak deny-all loop breaker: stopped after " + itoa(consecutive) + " consecutive turns (bound " + itoa(threshold) +
		") refusing the same tool the same way — auto-continue ends here so the loop cannot spin.\n" +
		"  refused tool : " + tool + "\n" +
		"  reason       : " + reason + denyAllReasonNote(reason) + "\n" +
		"  disposition  : " + dispo.String() + "\n" +
		"  floor source : " + floorSource + " (inspect: fak guard --dump-policy)\n" +
		"  recovery     : " + recovery + "\n" +
		"A genuinely dangerous repeated denial stays fail-closed: the breaker only stops AUTO-CONTINUATION; it never auto-allows a refused tool."
}

// denyAllEffectfulRecovery is the last line an agent reads before auto-continuation
// ends, so a route it fails to name is, from the agent's side, a route that does not
// exist. This copy of the delete guidance used to close with a flat "recursive/forced
// deletes stay operator-only", which the shipped floor contradicts: a recursive delete
// whose targets all resolve strictly under a declared scratchpad root is ADMITTED
// (adjudicator's psRemoveItemAllTargetsInScratch exists precisely to make that
// carve-out reachable on Windows). An agent cleaning its own per-session scratch was
// therefore told to escalate to an operator for work the floor already allows — and
// escalating is the one move that cannot resolve it, because there is nothing for the
// operator to approve.
//
// Naming the route is not a loosening: the breaker still never auto-allows anything,
// and the operator-only claim is preserved for a target OUTSIDE such a root, which is
// the genuinely fatal case. The two conditions that keep the deny — the scratch root
// itself, and a glob or unexpanded variable — are named too, since a remedy that
// silently fails the second time is the same defect one turn later.
//
// That last sentence is why this also names the SPELLING. The carve-out resolves on a
// forward-slash path and not on a Windows-spelled one, because the POSIX tokenizer
// reads `\` as an escape: `rm -rf C:\scratch\sess\build` arrives as the single token
// `C:scratchsessbuild`, which is under no root, so it is refused. That refusal is
// CORRECT — Git Bash really would aim the delete at a different path, and admitting
// the intended path while the shell runs another one is the one thing this decider
// must never do — but it is indistinguishable, from the agent's side, from the route
// not existing: it re-targeted into the scratchpad exactly as instructed and was
// refused again. Naming the spelling is what closes that loop, and it is text only.
// The PowerShell surface has no such trap (its operands keep their backslashes), so
// only the POSIX-capable surfaces carry the note.
func denyAllEffectfulRecovery(tool, reason string) string {
	const (
		lead = ": follow the sanctioned alternative from the refusal text/policy and do NOT re-propose the refused call unchanged."

		// The scratch carve-out, stated with the two conditions that keep the deny.
		scratchRoute = " A recursive/forced delete is ALREADY admitted when every target is a literal path strictly INSIDE a declared scratchpad root (not the root itself, and no glob or unexpanded variable) — re-target it there and retry rather than asking the operator; outside such a root recursive/forced deletes stay operator-only."

		// Spelling is load-bearing on the POSIX surfaces and silently is not on the
		// PowerShell one, so the route above is only reachable if the retry is spelled
		// the way the receiving shell reads it.
		posixSpelling = " Spell that path with FORWARD slashes (C:/scratch/sess/build): at a POSIX shell a backslash is an ESCAPE, so C:\\scratch\\sess\\build reaches the floor as C:scratchsessbuild, resolves under no root at all, and is refused a second time — which reads as the route not existing rather than as a spelling error."
	)
	switch reason {
	case "POLICY_BLOCK":
		switch tool {
		case "PowerShell":
			return "the floor is CORRECT to deny \"" + tool + "\"" + lead +
				" For a PowerShell Remove-Item -Recurse/-Force block, remove exact files with Remove-Item <file> or move them aside." + scratchRoute
		case "Bash", "bash":
			// Bash used to fall through to the generic text, which names no delete
			// route at all — so the surface that carries `rm -rf`, and the only one
			// where the spelling trap bites, was the one told the least.
			return "the floor is CORRECT to deny \"" + tool + "\"" + lead +
				" For an rm -r/-f block, remove exact files with rm <file> or move them aside." + scratchRoute + posixSpelling
		case "shell_command", "functions.shell_command":
			// This surface is POSIX on one host and PowerShell on another, so it
			// carries both spellings for the same reason isRmRfArgRule recognises
			// both rules on it.
			return "the floor is CORRECT to deny \"" + tool + "\"" + lead +
				" For a recursive/forced delete block, remove exact files with rm <file> / Remove-Item <file>, or move them aside." + scratchRoute + posixSpelling
		}
		return "the floor is CORRECT to deny \"" + tool + "\"" + lead + " Do NOT loosen the policy unless the same dangerous-command argument rules remain in force."
	default:
		return "the floor is CORRECT to deny \"" + tool + "\": do NOT allow-list it without mirroring the dangerous-command argument rules " +
			"the named shell aliases carry (rm -rf / sudo / curl|sh). If the tool is genuinely unavailable, answer in text only."
	}
}

func denyAllReasonNote(reason string) string {
	switch reason {
	case "DEFAULT_DENY":
		return " (not on the capability floor)"
	case "POLICY_BLOCK":
		return " (explicit policy rule blocked this call)"
	case "SELF_MODIFY":
		return " (guarded write target)"
	case "MALFORMED":
		return " (tool arguments did not match the declared schema)"
	case "MISROUTE":
		return " (tool or argument shape did not match the intended route)"
	case "RATE_LIMITED":
		return " (rate or quota limit)"
	case "LEASE_HELD":
		return " (conflicting lease)"
	case "SECRET_EXFIL", "RESULT_SECRET_DISCOVERED":
		return " (secret-shaped content blocked)"
	case "UNWITNESSED":
		return " (required independent witness missing)"
	case "OVERSIZE":
		return " (payload over the configured size bound)"
	case "UNKNOWN_TOOL":
		return " (tool not exposed by this harness)"
	default:
		return ""
	}
}

// itoa is a stdlib-free int->string for the diagnostic so this leaf stays stdlib-only
// (matching the package's foundation-leaf import discipline). It handles the small
// non-negative counts the breaker emits.
func itoa(n int) string {
	if n < 0 {
		return "-" + itoa(-n)
	}
	if n < 10 {
		return string(rune('0' + n))
	}
	return itoa(n/10) + string(rune('0'+n%10))
}
