package adjudicator

import (
	"context"
	"encoding/json"
	"net/url"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// selfModifyPathVerdict is the SELF_MODIFY rung for write-shaped calls carrying a
// path ARG: a target matching a protected glob is a PROVABLE refusal. Bounded
// disclosure: the witness carries ONLY the offending glob, never the whole policy
// (deny channel is not a policy oracle). The shell-command and synth-tool halves
// of the rung stay inline in Adjudicate, where the architest wiring gates pin
// their calls. Returns the resolved verdict and whether it decided this call.
func (a *Adjudicator) selfModifyPathVerdict(ctx context.Context, p Policy, c *abi.ToolCall, lowerTool string, args map[string]any, cl class, pr *RungProfile) (abi.Verdict, bool) {
	if !pr.runs(cl, rungSelfModify) || !writeShapedLower(lowerTool) {
		return abi.Verdict{}, false
	}
	target := targetPath(args)
	if g := matchGlob(target, p.BlockedPathGlobs); g != "" {
		return credentialBlockVerdict(g), true
	}
	if g := matchGlob(target, p.SelfModifyGlobs); g != "" {
		if directDevEditTool(c.Tool) && a.devEditAttested(ctx, c, target) {
			return abi.Verdict{Kind: abi.VerdictAllow, By: "monitor/dev-lease", Meta: map[string]string{"dev_attested": "true"}}, true
		}
		return p.soften(abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonSelfModify,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: g},
			// Three different rungs cite SELF_MODIFY and disclose only the
			// glob, so on the wire they are one undifferentiated bucket. The rule
			// id separates them (#5863) — and it is what decides whether a
			// SELF_MODIFY on a `cd fak && …` compound came in through a path ARG or
			// through the shell command line, the hypothesis the journal could not
			// confirm.
			Meta: denyRule(abi.DenyRuleSelfModifyPath),
		}, nil), true
	}
	return abi.Verdict{}, false
}

// egressRungVerdict is the EGRESS rung: a tool call that reaches a blocked
// NETWORK DESTINATION is refused with the egressfloor class witness. The blocked
// set is HARDWIRED in egressfloor (these addresses are never a legitimate
// destination), so the rung needs no policy field and fires under the zero
// Policy too; it is mandatory for every class (mustRun's default), so a
// RungProfile can never elide it. Bounded disclosure: the witness names only
// the offending host + class, never the policy. Returns the verdict and whether
// it decided this call.
func egressRungVerdict(p Policy, egressList *egresslist.List, tool string, args map[string]any, cl class, pr *RungProfile) (abi.Verdict, bool) {
	if !pr.runs(cl, rungEgress) {
		return abi.Verdict{}, false
	}
	if host, label := egressfloor.Classify(tool, args, p.EgressExtraDenyHosts...); host != "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  egressfloor.ReasonEgressBlock,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: label + ": " + host},
		}, true
	}
	// The operator-configurable band of the SAME rung, deliberately AFTER the
	// hardwired check above: block/allow lists and the restrict posture (egresslist.go).
	// Running second is what makes the floor un-openable — an allow rule naming the
	// metadata host never gets asked, because Classify has already returned.
	if v, ok := egressListVerdict(egressList, p, tool, args); ok {
		return v, true
	}
	return abi.Verdict{}, false
}

// lintWriteRungVerdict is the LINT-WRITES rung (opt-in, #536): a whole-file write
// of unparseable code is refused MALFORMED with a bounded file:line:col witness.
// The Go/JSON grammars parse in-process (stdlib, no exec); any other language has
// no in-process checker here and DEFERs (fail open). Bounded disclosure: the
// witness names only the first finding, never the file content. Returns the
// verdict and whether it decided this call.
func lintWriteRungVerdict(p Policy, c *abi.ToolCall, args map[string]any, advisoryNotes []string, cl class, pr *RungProfile) (abi.Verdict, bool) {
	if !pr.runs(cl, rungLintWrite) || !p.LintWrites || !wholeFileWrite(c.Tool) {
		return abi.Verdict{}, false
	}
	if w := lintWriteMalformed(targetPath(args), args); w != "" {
		return p.soften(abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonMalformed,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: w},
		}, advisoryNotes), true
	}
	return abi.Verdict{}, false
}

// applyRedactionTransforms prepares the dispatch-time argument transform: the CLI
// grammar attenuation (an argument transform, not a textual advisory — a valid
// read-only command with forbidden gh search scope qualifiers is rewritten before
// dispatch; a non-read-only form was denied above) and the rungTransform
// secret-field redaction. Returns the transformed args (nil when nothing was
// rewritten) and whether any transform is pending.
func applyRedactionTransforms(p Policy, argPreds []ArgPredicate, args map[string]any, cl class, pr *RungProfile) (map[string]any, bool) {
	redactedArgs, redacted := map[string]any(nil), false
	for _, pred := range argPreds {
		if pred.Kind != ArgCLIReadOnly {
			continue
		}
		command, ok := argString(args, pred.Arg)
		if !ok {
			continue
		}
		rewritten, changed, _ := attenuateCLIGrammar(command)
		if changed {
			redactedArgs = cloneArgs(args)
			redactedArgs[pred.Arg] = rewritten
			redacted = true
		}
	}
	if pr.runs(cl, rungTransform) && len(p.RedactFields) > 0 && args != nil {
		if more, did := redact(selectArgs(redactedArgs, args), p.RedactFields); did {
			redactedArgs, redacted = more, true
		}
	}
	return redactedArgs, redacted
}

// reversibilityRungResult carries the reversibility rung's state mutations back
// to Adjudicate: the rung may replace the dispatch args (stripping the
// confirmation arg) and re-run the redaction over them, so Adjudicate must adopt
// the post-rung values on its transform/allow paths. handled/verdict carry an
// early verdict (the untracked-removal allow, the autorepair transform, or the
// hold); when handled is false the caller continues with the returned state.
type reversibilityRungResult struct {
	args               map[string]any
	redactedArgs       map[string]any
	redacted           bool
	confirmedWithToken bool
	verdict            abi.Verdict
	handled            bool
}

// runReversibilityRung is the REVERSIBILITY rung (#2156): before any path that
// would otherwise dispatch the call, hold irreversible/outward-facing calls
// behind a deterministic preview token. Hard refusals before it still win first,
// and a normally denied call is not handed a preview token it cannot use.
func (a *Adjudicator) runReversibilityRung(ctx context.Context, p Policy, c *abi.ToolCall, args map[string]any, redactedArgs map[string]any, redacted bool, lowerTool string, cl class, pr *RungProfile) reversibilityRungResult {
	confirmedWithToken := false
	if pr.runs(cl, rungReversibility) && (redacted || wouldAdmit(p, c.Tool, lowerTool)) {
		if a.selfAuthoredUntrackedRemoval(c, args) {
			return reversibilityRungResult{handled: true, verdict: abi.Verdict{Kind: abi.VerdictAllow, By: "monitor/reversibility", Meta: map[string]string{
				"witness": "trace-authored-git-untracked",
			}}}
		}
		env, ok := ReversibilityConfirmed(c.Tool, args)
		if !ok {
			// In-flight repair of a sanctioned compiled sidestep: when the operator opts
			// in (policy.AutoRepairEnv=sidestep) and the matched family offered a machine-
			// applicable substitution for THIS call, substitute the sanctioned verb instead
			// of holding. env carries a RewriteCommand only when the producer's safe-subset
			// gate passed (a bare `git push`, never a --force/--delete/refspec push), so the
			// dangerous variants never reach here with one and still take the hold below.
			if p.AutoRepairSidestep && env.RewriteCommand != "" {
				// Preserve every non-confirmation arg (workdir, timeout, description, ...) and
				// swap only the effect-bearing command, so the sanctioned verb still runs in
				// the same working directory the operator targeted -- dropping workdir here
				// would silently push a different repo.
				na := argsWithoutConfirmation(args)
				na["command"] = env.RewriteCommand
				if ref, ok := putJSON(ctx, na); ok {
					newTool := ""
					if !strings.EqualFold(c.Tool, env.RewriteTool) {
						newTool = env.RewriteTool // cross-tool sidestep (e.g. MCP git_push -> Bash)
					}
					return reversibilityRungResult{handled: true, verdict: abi.Verdict{
						Kind:    abi.VerdictTransform,
						By:      "monitor/reversibility",
						Payload: abi.TransformPayload{NewArgs: ref, NewTool: newTool},
						Meta: map[string]string{
							"reversibility_autorepair":   "sidestep",
							"reversibility_class":        string(env.Class),
							"reversibility_substitution": env.RewriteCommand,
						},
					}}
				}
			}
			return reversibilityRungResult{handled: true, verdict: reversibilityGateVerdict(env)}
		}
		confirmedWithToken = env.Class != ReversibilityReversible && hasConfirmationArg(args)
		if confirmedWithToken {
			args = argsWithoutConfirmation(args)
			if redacted {
				if more, did := redact(selectArgs(redactedArgs, args), p.RedactFields); did {
					redactedArgs, redacted = more, true
				}
			}
		}
	}
	return reversibilityRungResult{args: args, redactedArgs: redactedArgs, redacted: redacted, confirmedWithToken: confirmedWithToken}
}

// affirmativeAllowVerdict resolves the affirmative-allow rung: an explicit Allow
// entry or an AllowPrefix match admits the call (stripping a spent confirmation
// arg first when the reversibility rung confirmed with a token). Returns the
// verdict and whether an affirmative allow matched; a false result falls through
// to the read-only-hint and default-deny rungs.
func affirmativeAllowVerdict(ctx context.Context, p Policy, tool string, args map[string]any, confirmedWithToken bool, advisoryNotes []string) (abi.Verdict, bool) {
	if p.Allow[tool] {
		if confirmedWithToken {
			if v, ok := stripConfirmationTransform(ctx, args, advisoryNotes); ok {
				return v, true
			}
		}
		return allowWithNotes("monitor", advisoryNotes), true
	}
	for _, pre := range p.AllowPrefix {
		if strings.HasPrefix(tool, pre) {
			if confirmedWithToken {
				if v, ok := stripConfirmationTransform(ctx, args, advisoryNotes); ok {
					return v, true
				}
			}
			return allowWithNotes("monitor", advisoryNotes), true
		}
	}
	return abi.Verdict{}, false
}

func reversibilityGateVerdict(env ReversibilityEnvelope) abi.Verdict {
	claim, err := json.Marshal(env)
	if err != nil {
		claim = []byte(`{"class":"` + string(env.Class) + `"}`)
	}
	meta := map[string]string{
		"reversibility_class": string(env.Class),
		"preview":             env.Preview,
		"confirm_token":       env.ConfirmToken,
	}
	if env.DryRunHint != "" {
		meta["dry_run_hint"] = env.DryRunHint
	}
	return abi.Verdict{
		Kind:    abi.VerdictRequireWitness,
		By:      "monitor/reversibility",
		Payload: abi.WitnessPayload{Claim: string(claim)},
		Meta:    meta,
	}
}

func stripConfirmationTransform(ctx context.Context, args map[string]any, advisoryNotes []string) (abi.Verdict, bool) {
	ref, ok := putJSON(ctx, args)
	if !ok {
		return abi.Verdict{}, false
	}
	v := abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      "monitor",
		Payload: abi.TransformPayload{NewArgs: ref},
		Meta:    map[string]string{"reversibility_confirmed": "true"},
	}
	if len(advisoryNotes) > 0 {
		v.Meta["advisory_violations"] = strings.Join(advisoryNotes, "; ")
	}
	return v, true
}

func researchEgressVerdict(tool string, args map[string]any, allowHosts []string) (abi.Verdict, bool) {
	if !strings.EqualFold(tool, "WebFetch") || len(allowHosts) == 0 {
		return abi.Verdict{}, false
	}
	raw, _ := args["url"].(string)
	scheme, host := webURLParts(raw)
	if host == "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonMalformed,
			By:      "monitor/research-egress",
			Payload: abi.WitnessPayload{Claim: "research egress missing WebFetch.url"},
		}, true
	}
	if scheme != "http" && scheme != "https" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonPolicyBlock,
			By:      "monitor/research-egress",
			Payload: abi.WitnessPayload{Claim: "research egress unsupported WebFetch scheme: " + scheme},
		}, true
	}
	if researchHostAllowed(host, allowHosts) {
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor/research-egress",
			Meta: map[string]string{
				"research_egress": "allowlisted",
				"host":            host,
			},
		}, true
	}
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonPolicyBlock,
		By:      "monitor/research-egress",
		Payload: abi.WitnessPayload{Claim: "research egress host not allowlisted: " + host},
	}, true
}

func webURLParts(raw string) (scheme, host string) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Scheme == "" || u.Host == "" {
		return "", ""
	}
	return strings.ToLower(u.Scheme), strings.ToLower(strings.Trim(u.Hostname(), "[]"))
}

func researchHostAllowed(host string, allowHosts []string) bool {
	host = strings.ToLower(strings.Trim(strings.TrimSpace(host), "[]"))
	for _, allowed := range allowHosts {
		a := strings.ToLower(strings.Trim(strings.TrimSpace(allowed), "[]"))
		if a == "" {
			continue
		}
		if host == a || strings.HasSuffix(host, "."+a) {
			return true
		}
	}
	return false
}

// admitOrDefaultDeny resolves a call that passed every refusal rung, in order:
// the pending redaction transform, the affirmative allow, the exec_command
// read-only grammar, and finally the posture-aware default deny. A spent
// reversibility confirmation is stripped from the args before any admit.
func (a *Adjudicator) admitOrDefaultDeny(ctx context.Context, p Policy, c *abi.ToolCall, lowerTool string, rev reversibilityRungResult, advisoryNotes []string) abi.Verdict {
	args, confirmedWithToken := rev.args, rev.confirmedWithToken
	if rev.redacted {
		if ref, ok := putJSON(ctx, rev.redactedArgs); ok {
			v := abi.Verdict{Kind: abi.VerdictTransform, By: "monitor",
				Payload: abi.TransformPayload{NewArgs: ref}}
			if len(advisoryNotes) > 0 {
				v.Meta = map[string]string{"advisory_violations": strings.Join(advisoryNotes, "; ")}
			}
			return v
		}
	}
	if v, ok := affirmativeAllowVerdict(ctx, p, c.Tool, args, confirmedWithToken, advisoryNotes); ok {
		return v
	}
	// exec_command's readOnlyHint is a classification request, not authority: only
	// the narrow positive grammar may replace the default deny.
	if v, ok := a.execCommandReadOnlyVerdict(c, args); ok {
		return v
	}
	if confirmedWithToken && (p.Posture == PostureDefaultOpen || p.admitAndLogLower(c.Tool, lowerTool) || p.AdvisoryReasons[abi.ReasonDefaultDeny]) {
		if v, ok := stripConfirmationTransform(ctx, args, advisoryNotes); ok {
			return v
		}
	}
	return defaultDeny(p, c.Tool, lowerTool, advisoryNotes)
}

// credentialBlockVerdict refuses a write that reaches a credential or host-config
// path. It is never softened: POLICY_BLOCK is not advisory-eligible.
func credentialBlockVerdict(glob string) abi.Verdict {
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonPolicyBlock,
		By:      "monitor/credential-block",
		Payload: abi.WitnessPayload{Claim: glob},
		Meta:    denyRule(abi.DenyRuleCredentialPathBlock),
	}
}

// selfModifyDeny is the SELF_MODIFY refusal the shell and synth-tool rungs cite.
// The rule id separates them on the wire, since all disclose only the glob.
func selfModifyDeny(glob, rule string) abi.Verdict {
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonSelfModify,
		By:      "monitor",
		Payload: abi.WitnessPayload{Claim: glob},
		Meta:    denyRule(rule),
	}
}

// preAdmitVerdict runs the refusal rungs between the arg predicates and the
// redaction transform: the hardwired-then-listed egress floor, the research
// WebFetch allowlist, the opt-in lint-write grammar, and, under a default-open
// posture, the dangerous-gotcha floor (#11258), which must precede any allow.
func preAdmitVerdict(p Policy, egressList *egresslist.List, c *abi.ToolCall, args map[string]any, advisoryNotes []string, cl class, pr *RungProfile) (abi.Verdict, bool) {
	if v, ok := egressRungVerdict(p, egressList, c.Tool, args, cl, pr); ok {
		return v, true
	}
	if v, ok := researchEgressVerdict(c.Tool, args, p.ResearchEgressAllowHosts); ok {
		return v, true
	}
	if v, ok := lintWriteRungVerdict(p, c, args, advisoryNotes, cl, pr); ok {
		return v, true
	}
	if p.Posture == PostureDefaultOpen {
		if v, ok := EvalDangerousGotchas(c.Tool, args); ok {
			return v, true
		}
	}
	return abi.Verdict{}, false
}
