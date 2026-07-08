package policy

// No-unauditable-full-bypass conformance (issue #2845, child of the "mine Hermes
// for what a kernel does better" epic #2834 track B). Sibling of the flag-bypass
// witness in flag_bypass_capfloor_conformance_test.go (#2921, track C): that test
// proves no FLAG or headless signal can move the fail-closed floor. This one proves
// the complementary property for the ONE seam fak does expose — the operator
// OVERRIDE — so that even the widest override fak has is (a) unable to cross the
// hard floor and (b) never un-witnessed.
//
// Hermes mechanism (contrast). `force=True` on the terminal tool skips ALL guards
// (tools/terminal_tool.py:2272) and a headless session auto-approves with only a
// warning (approval.py:2029): a structural FULL bypass — anything that can set the
// flag runs unadjudicated, and nothing records that it did.
//
// fak's answer. There is no `force` and no headless auto-allow (that is #2921's
// witness). The only capability-widening the floor performs is the ADMIT-AND-LOG
// family — its structural analogue of an operator override:
//
//   - Posture=PostureAdmitAndLog: a low-risk read-shaped DEFAULT_DENY is downgraded
//     to an Allow that CARRIES a forensic record (Verdict.Meta posture + would_deny);
//   - Complain (a per-tool logged trial, #670): the same downgrade for one named
//     tool even when it is not read-shaped;
//   - AdvisoryReasons (#advisory): a HEURISTIC-rung refusal (self-modify / malformed
//     / default-deny) downgraded to a logged Allow that records the would-deny.
//
// All three downgrade ONLY the default-deny / heuristic rungs — the genuine-danger
// rungs (explicit Deny/POLICY_BLOCK, arg-value POLICY_BLOCK, SECRET_EXFIL, the
// hardwired EGRESS_BLOCK) return BEFORE the downgrade, and AdvisoryReasons is CLAMPED
// by New/SetPolicy to the eligible heuristic reasons, so the danger reasons can never
// be softened. Net: an override cannot cross the hard floor, every override it does
// perform leaves a journaled record, and an unconfigured (headless) session with no
// affirmative grant defaults to the floor, not to allow.
//
// This lives in internal/policy (not internal/adjudicator) for the same reason its
// sibling does: the adjudicator tree is a hard-self core lock, so the floor is driven
// through the adjudicator's EXPORTED API.

import (
	"context"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// overrideFloor turns on EVERY override surface fak has, at once — the maximal
// override a caller could ever request — over a small floor that exercises each
// deny rung plus one affirmative grant. If any override could cross the hard floor
// or widen a capability without a record, this floor is where it would show.
func overrideFloor() adjudicator.Policy {
	return adjudicator.Policy{
		// Global read-shaped admit-and-log — the "headless operator waved it through"
		// analogue. Only ever widens the default-deny rung.
		Posture: adjudicator.PostureAdmitAndLog,
		// One affirmative, in-git grant. The reviewed Policy allow is fak's genuinely
		// witnessed override: a capability reachable only because a human committed the
		// grant (the git hash chain IS the record). A clean Allow is legitimate ONLY
		// when it traces here.
		Allow: map[string]bool{"Read": true},
		// Per-tool logged trial (#670): puts even non-read tools — including the
		// hard-denied ones — on admit-and-log. It still cannot reach past the hard
		// rungs, which return before default-deny.
		Complain: map[string]bool{
			"weird_tool": true, "git_push": true, "exfiltrate": true,
			"Bash": true, "WebFetch": true, "Edit": true,
		},
		// Ask for EVERY reason to be softened. New() clamps this to the eligible
		// heuristic set {SELF_MODIFY, MALFORMED, DEFAULT_DENY}; the genuine-danger
		// reasons below are dropped, which is the whole point.
		AdvisoryReasons: map[abi.ReasonCode]bool{
			abi.ReasonSelfModify:          true,
			abi.ReasonMalformed:           true,
			abi.ReasonDefaultDeny:         true,
			abi.ReasonPolicyBlock:         true,
			abi.ReasonSecretExfil:         true,
			egressfloor.ReasonEgressBlock: true,
		},
		Deny: map[string]abi.ReasonCode{
			"git_push":   abi.ReasonPolicyBlock,
			"exfiltrate": abi.ReasonSecretExfil,
		},
		SelfModifyGlobs: []string{"internal/kernel/", ".git/"},
		ArgPredicates: []adjudicator.ArgPredicate{{
			Tool:   "Bash",
			Arg:    "command",
			Kind:   adjudicator.ArgDenyRegex,
			Re:     regexp.MustCompile(`\brm\s+-[A-Za-z]*[rRfF]`), // mirrors the shipped dogfood rm rule
			Reason: abi.ReasonPolicyBlock,
		}},
	}
}

func bypassCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

// witnessedOverride reports whether an Allow carries the journaled override record
// (posture + would_deny) that makes it auditable after the fact.
func witnessedOverride(v abi.Verdict) bool {
	return v.Kind == abi.VerdictAllow && v.Meta["posture"] != "" && v.Meta["would_deny"] != ""
}

// affirmativeGrant reports whether an Allow traces to an explicit in-Policy grant —
// the reviewed, git-witnessed override (Allow or AllowPrefix). A clean Allow (no
// forensic Meta) is legitimate ONLY when this is true.
func affirmativeGrant(p adjudicator.Policy, tool string) bool {
	if p.Allow[tool] {
		return true
	}
	for _, pre := range p.AllowPrefix {
		if len(tool) >= len(pre) && tool[:len(pre)] == pre {
			return true
		}
	}
	return false
}

func TestNoUnauditableBypass(t *testing.T) {
	// (a) The hard floor cannot be crossed by ANY override. With every override
	// surface maxed out, the genuine-danger rungs still Deny with their own reason —
	// the "mkfs/dd/fork-bomb never bypassable, even by force" property. Each of these
	// tools is ALSO on the complain list and its reason is ALSO requested advisory;
	// neither moves the verdict.
	t.Run("hard floor uncrossable by any override", func(t *testing.T) {
		a := adjudicator.New(overrideFloor())
		hard := []struct {
			name   string
			tool   string
			args   string
			reason abi.ReasonCode
		}{
			{"explicit name deny", "git_push", `{}`, abi.ReasonPolicyBlock},
			{"secret exfil", "exfiltrate", `{}`, abi.ReasonSecretExfil},
			{"destructive arg value", "Bash", `{"command":"rm -rf /tmp/x"}`, abi.ReasonPolicyBlock},
			{"hardwired egress floor", "WebFetch", `{"url":"http://169.254.169.254/latest/meta-data/"}`, egressfloor.ReasonEgressBlock},
		}
		for _, c := range hard {
			v := a.Adjudicate(context.Background(), bypassCall(c.tool, c.args))
			if v.Kind != abi.VerdictDeny || v.Reason != c.reason {
				t.Errorf("%s: got Kind=%v Reason=%s, want Deny/%s — a maxed-out override must never cross the hard floor",
					c.name, v.Kind, abi.ReasonName(v.Reason), abi.ReasonName(c.reason))
			}
		}
	})

	// (b) Every capability the override DOES widen is witnessed. The three downgrade
	// surfaces each produce an Allow that carries the journaled would-deny record — so
	// no bypass is silent.
	t.Run("every widened capability carries a journaled override record", func(t *testing.T) {
		a := adjudicator.New(overrideFloor())
		widened := []struct {
			name          string
			tool          string
			args          string
			wantWouldDeny abi.ReasonCode
		}{
			{"admit-and-log read-shaped default deny", "search_web", `{"query":"x"}`, abi.ReasonDefaultDeny},
			{"complain-set non-read default deny", "weird_tool", `{}`, abi.ReasonDefaultDeny},
			{"advisory-softened self-modify", "Edit", `{"file_path":"internal/kernel/x.go"}`, abi.ReasonSelfModify},
		}
		for _, c := range widened {
			v := a.Adjudicate(context.Background(), bypassCall(c.tool, c.args))
			if v.Kind != abi.VerdictAllow {
				t.Errorf("%s: got Kind=%v, want Allow (the override widens this capability)", c.name, v.Kind)
				continue
			}
			if !witnessedOverride(v) {
				t.Errorf("%s: Allow carries no journaled override record (Meta=%v) — an un-witnessed bypass", c.name, v.Meta)
				continue
			}
			if got := v.Meta["would_deny"]; got != abi.ReasonName(c.wantWouldDeny) {
				t.Errorf("%s: journaled would_deny=%q, want %q", c.name, got, abi.ReasonName(c.wantWouldDeny))
			}
		}
	})

	// (c) A headless session with no explicit approval defaults to the floor, not to
	// allow. The unconfigured / fail-closed Policy denies every capability-shaped call
	// even with every headless/non-interactive/CI ambient signal set — "silence != consent,
	// warning != approval." (The flag/env MATRIX is #2921's witness; this pins the
	// default-to-floor half for the empty policy.)
	t.Run("headless session with no grant defaults to the floor", func(t *testing.T) {
		for k, val := range map[string]string{"CI": "true", "FAK_HEADLESS": "1", "FAK_NONINTERACTIVE": "1", "TERM": ""} {
			t.Setenv(k, val)
		}
		a := adjudicator.New(adjudicator.Policy{}) // zero value: fail-closed, nothing granted
		for _, tc := range []struct{ tool, args string }{
			{"weird_tool", `{}`},
			{"Bash", `{"command":"echo hi"}`},
			{"WebFetch", `{"url":"https://api.anthropic.com/v1/messages"}`},
			{"Edit", `{"file_path":"cmd/fak/main.go"}`},
		} {
			v := a.Adjudicate(context.Background(), bypassCall(tc.tool, tc.args))
			if v.Kind != abi.VerdictDeny {
				t.Errorf("headless default: %s got Kind=%v, want Deny — an unconfigured session must default to the floor, not allow", tc.tool, v.Kind)
			}
		}
	})

	// The core invariant (#2845 witness): no capability-reaching code path reaches an
	// Allow without EITHER a deny OR a journaled override record OR an affirmative in-git
	// grant. Sweep a representative call set under the maxed-out override floor; a bare
	// Allow (no forensic record) that does not trace to an affirmative Policy grant is an
	// unauditable bypass and fails the test.
	t.Run("no capability reached without a deny or a journaled override", func(t *testing.T) {
		p := overrideFloor()
		a := adjudicator.New(p)
		calls := []struct{ tool, args string }{
			{"Read", `{"file_path":"README.md"}`},                              // affirmative in-git grant
			{"git_push", `{}`},                                                 // hard deny
			{"exfiltrate", `{}`},                                               // hard deny
			{"Bash", `{"command":"rm -rf /tmp/x"}`},                            // hard deny (arg value)
			{"WebFetch", `{"url":"http://169.254.169.254/latest/meta-data/"}`}, // hard deny (egress)
			{"Edit", `{"file_path":"internal/kernel/x.go"}`},                   // witnessed override (advisory self-modify)
			{"weird_tool", `{}`},                                               // witnessed override (complain)
			{"search_web", `{"query":"x"}`},                                    // witnessed override (admit-and-log)
			{"delete_everything", `{}`},                                        // write-shaped, not granted → hard default deny
		}
		for _, c := range calls {
			v := a.Adjudicate(context.Background(), bypassCall(c.tool, c.args))
			if v.Kind != abi.VerdictAllow {
				continue // Deny / RequireWitness / Transform never reach the capability unadjudicated
			}
			if witnessedOverride(v) || affirmativeGrant(p, c.tool) {
				continue // legitimate: journaled override, or the reviewed in-git grant
			}
			t.Errorf("%s reached an Allow with no journaled override record and no affirmative grant (Meta=%v) — an unauditable full bypass",
				c.tool, v.Meta)
		}
	})

	// Structural pin behind (a): the genuine-danger reasons are NOT advisory-eligible,
	// so no override posture can ever soften them. If a future edit adds one to the
	// eligible set, the hard floor becomes crossable and this fails loud.
	t.Run("genuine-danger reasons stay fail-closed under every posture", func(t *testing.T) {
		for _, r := range []abi.ReasonCode{abi.ReasonPolicyBlock, abi.ReasonSecretExfil, egressfloor.ReasonEgressBlock} {
			if adjudicator.AdvisoryEligible(r) {
				t.Errorf("reason %s is advisory-eligible; a genuine-danger reason must never be softenable by an override", abi.ReasonName(r))
			}
		}
		// The heuristic reasons ARE eligible — but softening them produces a WITNESSED
		// Allow (proven in (b)), never a silent one.
		for _, r := range []abi.ReasonCode{abi.ReasonSelfModify, abi.ReasonMalformed, abi.ReasonDefaultDeny} {
			if !adjudicator.AdvisoryEligible(r) {
				t.Errorf("heuristic reason %s should be advisory-eligible (softened to a logged Allow)", abi.ReasonName(r))
			}
		}
	})
}
