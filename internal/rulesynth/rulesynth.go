// Package rulesynth closes AUTOHARNESS's loop on OUR hand-authored harness: it
// mines the kernel's refusal/near-miss log to PROPOSE the next STRUCTURAL adjudicator
// rule, then PROVES it model-free before it can ship — never self-certifying (#537).
//
// THE GAP IT FILLS. internal/adjudicator's write-shaped denylist (shellWriteVerbs /
// interpreterEvalFlags / the commandWrites idioms) is patched hole-by-hole by hand
// (#172 walks `sed -i` -> `perl/ruby -i` -> `awk -i inplace` -> `python/node -c` ->
// `tar/unzip/rsync`, one round-trip each). The kernel already REFUSES thousands of
// calls and records why, but nothing turns that log into the next rule. This package
// is the missing consumer: refusal log in, a reviewable policy diff out.
//
// THE THREE RUNGS (each maps to a function below):
//
//  1. CORPUS EXTENSION (Detect). abi.LabelRow carries only {CallHash, rungs, Verdict,
//     Reason} — no command, so the refusal log is not mineable for a NEW write verb.
//     Detect adds the args/command-bearing NEAR-MISS capture LabelRow lacks: a shell/
//     exec call whose COMMAND reached a guarded tree (named a SelfModifyGlob) but which
//     the CURRENT floor ADMITTED — a write that slipped because its verb/interpreter is
//     not yet recognized. It is computed against the REAL adjudicator, so the capture
//     can never drift from the floor it is mining.
//
//  2. SYNTHESIS PROPOSER (Propose). internal/rsiloop's only shipped Proposer rewrites
//     an integer literal; #503's policy search hill-climbs a HAND-SUPPLIED deny pool.
//     Propose clusters the near-miss corpus by its unrecognized write VERB and emits,
//     per cluster, a CANDIDATE STRUCTURAL RULE as a policy-manifest ArgRule (a new
//     ArgPredicate — the manifest-native form of a new shellWriteVerbs token). The new
//     alleles come from the refusal log, not a fixed genome.
//
//  3. THE HONESTY GATE (Validate + SelfModify). A synthesized rule is NEVER self-
//     certified. Validate replays it through the REAL adjudicator (model-free,
//     deterministic, ZERO model calls) over the frozen near-miss corpus plus a benign
//     corpus, and folds the result into shipgate's non-forgeable keep-bit: KEEP only if
//     the rule newly CATCHES near-misses without REGRESSING any benign call. And because
//     a rule that edits the policy floor edits the harness itself, every candidate is a
//     SELF_MODIFY: it is emitted as a reviewable ManifestDiff for the require-witness
//     rung (#386/#387/#388), NEVER applied as a live in-process mutation — the loop
//     cannot widen its own authority by grading its own homework.
//
// SCOPE / FENCES. Additive and model-free: it rides on internal/policy (the manifest
// diff form), internal/adjudicator (the real floor as the replay oracle), and
// internal/shipgate (the keep-bit). It NEVER mutates the live policy; its only output
// is a proposal an operator reviews and lands. Out of scope (the issue's anti-thesis):
// replacing the model's decisions with an LLM-free decision policy — fak adjudicates
// the model, it does not replace it.
package rulesynth

import (
	"context"
	"encoding/json"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// DefaultHarnessGlobs are the harness/witness trees a synthesized rule must never edit
// without a witness — the adjudicator, the kernel, the ship-stamp config. A candidate
// whose near-misses reached any of these is a SELF_MODIFY (see Candidate.SelfModify).
// They mirror adjudicator.DefaultPolicy's SelfModifyGlobs (the protected floor).
var DefaultHarnessGlobs = []string{
	"internal/abi/", "internal/kernel/", "internal/adjudicator/",
	"internal/architest/", "internal/shipgate/", "internal/policy/",
	"dos.toml", ".dos/",
}

// Call is one shell/exec tool call as submitted: the tool name plus the arg key and
// value that carry the command string (e.g. {"Bash", "command", "sed -i ..."}).
type Call struct {
	Tool    string
	Arg     string // the arg key carrying the command, e.g. "command" or "cmd"
	Command string
}

// NearMiss is the args/command-bearing capture rung-1 adds: a Call whose Command
// reached a guarded tree (named GuardedGlob, a SelfModifyGlob fragment) yet was
// ADMITTED by the current floor — a write that slipped because its verb is not yet
// recognized. This is the mineable signal abi.LabelRow lacks.
type NearMiss struct {
	Call
	GuardedGlob string
}

// Candidate is one synthesized STRUCTURAL rule proposed from a near-miss cluster,
// carried as a policy-manifest ArgRule (an ArgPredicate). It is a PROPOSAL — the
// package never applies it; an operator reviews ManifestDiff and lands it.
type Candidate struct {
	// Verb is the unrecognized write token the cluster shares (e.g. "ruby -e",
	// "sponge") — the new allele mined from the refusal log.
	Verb string
	// Rule is the manifest-native synthesized rule: a deny_regex ArgPredicate on the
	// command arg that fires when the command names BOTH the new verb and a guarded
	// tree (in either order). RE2-safe (no lookaround), so it compiles on the floor.
	Rule policy.ArgRule
	// Globs are the guarded-tree fragments this rule guards (sorted, de-duplicated).
	Globs []string
	// Support is how many near-misses the rule was clustered from.
	Support int
	// SelfModify reports that the rule's guarded trees intersect the harness/witness
	// trees — so landing it edits the floor itself and MUST route through the
	// require-witness rung (#386/#387/#388), never self-certify.
	SelfModify bool
}

// Verdict is the honesty-gate result for one candidate: the model-free replay counts
// plus shipgate's non-forgeable keep-bit. Every count is a real adjudicator verdict on
// the frozen corpus — none is self-reported, and ModelCalls is ZERO by construction.
type Verdict struct {
	// Caught is the number of near-misses the rule now DENIES that the baseline floor
	// ADMITTED — the gain axis (more caught is better).
	Caught int
	// Regressed is the number of benign calls the rule now WRONGLY denies — the cost
	// axis (must be zero to keep).
	Regressed int
	// CatchesCluster reports the rule catches EVERY near-miss in its own cluster (not a
	// partial/over-narrow regex) — the correctness bit folded into TruthClean.
	CatchesCluster bool
	Decision       shipgate.Decision
	Kept           bool
	Witness        shipgate.Witness
	Score          Scorecard
}

// Scorecard is the structured audit payload carried by a rulesynth Verdict. It is
// additive evidence: the keep-bit still comes from shipgate.Evaluate, while the
// score names the gain, regression, cluster, and self-modify axes for direct
// callers that do not route through internal/rsiloop's journal.
type Scorecard struct {
	Name       string           `json:"name,omitempty"`
	Value      float64          `json:"value"`
	Grade      string           `json:"grade,omitempty"`
	Components []ScoreComponent `json:"components,omitempty"`
}

// ScoreComponent is one named numeric axis of a Scorecard.
type ScoreComponent struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
	Unit  string  `json:"unit,omitempty"`
}

// Detect computes the args/command-bearing near-miss for one Call against the REAL
// floor (rung 1). It returns a NearMiss iff the command textually NAMES one of
// guardedGlobs AND the current floor (a permissive policy that allows the tool but
// guards those globs) does NOT deny it as a SELF_MODIFY — i.e. a write that reached a
// guarded tree by a verb the floor does not yet recognize. A command that names no
// guarded tree, or that the floor already catches, is not a near-miss.
func Detect(c Call, guardedGlobs []string) (NearMiss, bool) {
	glob := firstNamedGlob(c.Command, guardedGlobs)
	if glob == "" {
		return NearMiss{}, false // touches no guarded tree — nothing to mine
	}
	// The current floor: allow the tool, guard the trees. If it already DENIES the
	// command as SELF_MODIFY, the verb is recognized and there is no near-miss.
	pol := adjudicator.Policy{
		Allow:           map[string]bool{c.Tool: true},
		SelfModifyGlobs: append([]string(nil), guardedGlobs...),
	}
	if denied := adjudicateSelfModify(pol, c); denied {
		return NearMiss{}, false // already caught — not a near-miss
	}
	return NearMiss{Call: c, GuardedGlob: glob}, true
}

// Propose clusters the near-miss corpus by its unrecognized write VERB and emits one
// candidate structural rule per cluster (rung 2). Each candidate is a deny_regex
// ArgPredicate that fires when a command names BOTH the verb and a guarded tree. The
// output is sorted by Verb for a deterministic, reviewable diff.
func Propose(corpus []NearMiss) []Candidate {
	// Cluster: verb -> set of {arg, tool, globs} observed for it.
	type cluster struct {
		tool, arg string
		globs     map[string]bool
		support   int
	}
	byVerb := map[string]*cluster{}
	for _, nm := range corpus {
		verb := writeVerb(nm.Command)
		if verb == "" {
			continue
		}
		cl := byVerb[verb]
		if cl == nil {
			cl = &cluster{tool: nm.Tool, arg: nm.Arg, globs: map[string]bool{}}
			byVerb[verb] = cl
		}
		if nm.GuardedGlob != "" {
			cl.globs[nm.GuardedGlob] = true
		}
		cl.support++
	}

	verbs := make([]string, 0, len(byVerb))
	for v := range byVerb {
		verbs = append(verbs, v)
	}
	sort.Strings(verbs)

	out := make([]Candidate, 0, len(verbs))
	for _, verb := range verbs {
		cl := byVerb[verb]
		globs := maputil.SortedKeys(cl.globs)
		rule := policy.ArgRule{
			Tool:      cl.tool,
			Arg:       cl.arg,
			DenyRegex: synthRegex(verb, globs),
			Reason:    abi.ReasonName(abi.ReasonSelfModify),
		}
		out = append(out, Candidate{
			Verb:       verb,
			Rule:       rule,
			Globs:      globs,
			Support:    cl.support,
			SelfModify: intersectsHarness(globs, DefaultHarnessGlobs),
		})
	}
	return out
}

// Validate is the honesty gate (rung 3a): it replays the candidate's rule through the
// REAL adjudicator over the frozen near-miss corpus plus the benign corpus — model-
// free, deterministic, ZERO model calls — and folds the result into shipgate's non-
// forgeable keep-bit. It KEEPs only when the rule (1) newly catches at least one near-
// miss (a strict gain), (2) regresses ZERO benign calls (suite-green), and (3) catches
// EVERY near-miss in its own cluster (truth-clean: not an over-narrow regex). The
// baseline allows every tool in the corpora so a benign call is admitted UNLESS the
// candidate's rule denies it — isolating the rule's effect.
func Validate(c Candidate, nearMisses []NearMiss, benign []Call) (Verdict, error) {
	tools := toolSet(nearMisses, benign)

	basePol := adjudicator.Policy{Allow: tools}
	candRule, err := compileCandidate(c, tools)
	if err != nil {
		return Verdict{}, err
	}

	var v Verdict
	clusterTotal, clusterCaught := 0, 0
	for _, nm := range nearMisses {
		baseDeny := adjudicateDeny(basePol, nm.Call)
		candDeny := adjudicateDeny(candRule, nm.Call)
		// A catch is only credited where the baseline ADMITTED the call (a true near-
		// miss) and the candidate now DENIES it.
		if !baseDeny && candDeny {
			v.Caught++
		}
		if writeVerb(nm.Command) == c.Verb {
			clusterTotal++
			if candDeny {
				clusterCaught++
			}
		}
	}
	for _, b := range benign {
		// A benign call admitted by the baseline that the candidate now denies is a
		// regression — the cost the keep-bit refuses to pay.
		if !adjudicateDeny(basePol, b) && adjudicateDeny(candRule, b) {
			v.Regressed++
		}
	}
	v.CatchesCluster = clusterTotal > 0 && clusterCaught == clusterTotal

	// Fold into the non-forgeable keep-bit. Metric=caught (more is better); suite-green
	// = no regression; truth-clean = the rule catches its whole cluster. None of the
	// three is the candidate's say-so — each is a measured adjudicator verdict.
	w := shipgate.Witness{
		Metric:      "near_misses_caught",
		Before:      0,
		After:       float64(v.Caught),
		LowerBetter: false,
		SuiteGreen:  v.Regressed == 0,
		TruthClean:  v.CatchesCluster,
	}
	decision, ev := shipgate.Evaluate(w)
	v.Decision = decision
	v.Kept = ev.Kept()
	v.Witness = ev
	v.Score = scoreVerdict(c, v)
	return v, nil
}

func scoreVerdict(c Candidate, v Verdict) Scorecard {
	catchesCluster := 0.0
	if v.CatchesCluster {
		catchesCluster = 1
	}
	selfModify := 0.0
	if c.SelfModify {
		selfModify = 1
	}
	kept := 0.0
	if v.Kept {
		kept = 1
	}
	return Scorecard{
		Name:  "near_misses_caught",
		Value: float64(v.Caught),
		Grade: scoreVerdictGrade(v),
		Components: []ScoreComponent{
			{Name: "caught", Value: float64(v.Caught), Unit: "calls"},
			{Name: "regressed", Value: float64(v.Regressed), Unit: "calls"},
			{Name: "support", Value: float64(c.Support), Unit: "calls"},
			{Name: "guarded_globs", Value: float64(len(c.Globs)), Unit: "globs"},
			{Name: "catches_cluster", Value: catchesCluster, Unit: "bool"},
			{Name: "self_modify", Value: selfModify, Unit: "bool"},
			{Name: "kept", Value: kept, Unit: "bool"},
		},
	}
}

func scoreVerdictGrade(v Verdict) string {
	switch {
	case v.Regressed > 0:
		return "regressing"
	case !v.CatchesCluster:
		return "partial"
	case v.Caught == 0:
		return "no-catch"
	case v.Kept:
		return "clean"
	default:
		return "reverted"
	}
}

// ManifestDiff renders the candidate as the reviewable policy-manifest fragment to be
// landed (rung 3b): a one-rule manifest an operator diffs against the live policy. The
// package emits this diff and STOPS — it never applies the rule in-process, so a floor-
// widening rule always lands as a reviewed change, never a self-grading mutation.
func (c Candidate) ManifestDiff() policy.Manifest {
	return policy.Manifest{
		Version:  policy.Version,
		ArgRules: []policy.ArgRule{c.Rule},
	}
}

// --- helpers -------------------------------------------------------------------

// writeVerb extracts the leading write token of a command: the first field, plus the
// second field when it is a flag (so `ruby -e`, `python3 -c` cluster as the interpreter
// + eval-flag pair, not the bare interpreter that also runs benign scripts). It is the
// allele the proposer clusters on.
func writeVerb(cmd string) string {
	fields := strings.Fields(strings.TrimSpace(cmd))
	if len(fields) == 0 {
		return ""
	}
	verb := fields[0]
	if len(fields) > 1 && strings.HasPrefix(fields[1], "-") {
		verb += " " + fields[1]
	}
	return verb
}

// synthRegex builds the deny_regex for a verb + its guarded globs: a case-insensitive
// RE2 pattern that matches when the command contains the verb AND a guarded glob in
// EITHER order. Requiring the guarded glob is what keeps a benign use of the verb on an
// unguarded path admitted (no regression). Both operands are escaped (QuoteMeta), and
// `[\s\S]*` (RE2-safe, no lookaround) bridges them so the rule compiles on the floor.
func synthRegex(verb string, globs []string) string {
	v := regexp.QuoteMeta(verb)
	alts := make([]string, 0, len(globs))
	for _, g := range globs {
		alts = append(alts, regexp.QuoteMeta(g))
	}
	if len(alts) == 0 {
		return "(?i)" + v // no guarded glob observed — verb alone (rare; keeps the rule valid)
	}
	g := "(?:" + strings.Join(alts, "|") + ")"
	return "(?i)(?:" + v + `[\s\S]*` + g + "|" + g + `[\s\S]*` + v + ")"
}

// compileCandidate builds the adjudicator policy that allows every corpus tool and adds
// the candidate's one ArgRule, via the real manifest compiler (so the synthesized rule
// is validated exactly as a hand-authored manifest entry would be).
func compileCandidate(c Candidate, tools map[string]bool) (adjudicator.Policy, error) {
	m := policy.Manifest{
		Version:  policy.Version,
		Allow:    maputil.SortedKeys(tools),
		ArgRules: []policy.ArgRule{c.Rule},
	}
	return m.ToPolicy()
}

// adjudicateDeny reports whether the policy DENIES the call (any deny reason).
func adjudicateDeny(p adjudicator.Policy, c Call) bool {
	v := adjudicator.New(p).Adjudicate(context.Background(), inlineCall(c))
	return v.Kind == abi.VerdictDeny
}

// adjudicateSelfModify reports whether the policy denies the call specifically as a
// SELF_MODIFY (the floor's shell-write catch) — the signal Detect uses to tell an
// already-recognized write from a near-miss.
func adjudicateSelfModify(p adjudicator.Policy, c Call) bool {
	v := adjudicator.New(p).Adjudicate(context.Background(), inlineCall(c))
	return v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonSelfModify
}

// inlineCall builds an abi.ToolCall carrying the command as an inline JSON arg, so the
// adjudicator's commandSelfModify / ArgDenyRegex rungs see it exactly as a live call.
func inlineCall(c Call) *abi.ToolCall {
	arg := c.Arg
	if arg == "" {
		arg = "command"
	}
	obj := map[string]string{arg: c.Command}
	b, _ := json.Marshal(obj)
	return &abi.ToolCall{Tool: c.Tool, Args: abi.Ref{Kind: abi.RefInline, Inline: b}}
}

// firstNamedGlob returns the first guarded glob the command names, or "".
func firstNamedGlob(cmd string, globs []string) string {
	lc := strings.ToLower(cmd)
	for _, g := range globs {
		if g != "" && strings.Contains(lc, strings.ToLower(g)) {
			return g
		}
	}
	return ""
}

// intersectsHarness reports whether any guarded glob is (or is under) a harness tree.
func intersectsHarness(globs, harness []string) bool {
	for _, g := range globs {
		lc := strings.ToLower(g)
		for _, h := range harness {
			hl := strings.ToLower(h)
			if strings.HasPrefix(lc, hl) || strings.HasPrefix(hl, lc) {
				return true
			}
		}
	}
	return false
}

func toolSet(nm []NearMiss, benign []Call) map[string]bool {
	tools := map[string]bool{}
	for _, m := range nm {
		tools[m.Tool] = true
	}
	for _, b := range benign {
		tools[b.Tool] = true
	}
	return tools
}
