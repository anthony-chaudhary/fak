package adjudicator

import (
	"context"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// TestAdvisorySelfModifyAdmitsWithWouldDenyRecord is the false-positive escape
// anchor: with SELF_MODIFY declared advisory, a write into a protected glob is
// ADMITTED — but carries the full would-deny record (posture=advisory, the
// refusal name, the offending glob as the bounded claim) so the decision journal
// still shows exactly what enforcement would have refused. The same call under
// the same policy WITHOUT the advisory declaration stays a hard deny.
func TestAdvisorySelfModifyAdmitsWithWouldDenyRecord(t *testing.T) {
	ctx := context.Background()
	base := Policy{
		Allow:           map[string]bool{"Write": true, "Bash": true},
		SelfModifyGlobs: []string{"internal/abi/"},
	}

	// Control: enforcement is the default — the deny stands.
	a := New(base)
	v := a.Adjudicate(ctx, inlineCall("Write", `{"file_path":"internal/abi/types.go","content":"x"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("enforcing floor: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}

	// Advisory: the SAME call admits, with the would-deny record.
	adv := base
	adv.AdvisoryReasons = map[abi.ReasonCode]bool{abi.ReasonSelfModify: true}
	a = New(adv)
	v = a.Adjudicate(ctx, inlineCall("Write", `{"file_path":"internal/abi/types.go","content":"x"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("advisory SELF_MODIFY: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != "advisory" || v.Meta["would_deny"] != abi.ReasonName(abi.ReasonSelfModify) {
		t.Fatalf("advisory admit must carry posture=advisory + would_deny=SELF_MODIFY, got %v", v.Meta)
	}
	if v.Meta["claim"] != "internal/abi/" {
		t.Fatalf("advisory admit must keep the bounded witness claim (the FP diagnostic), got %v", v.Meta)
	}

	// The SHELL self-modify rung softens identically (same reason, same rung family).
	v = a.Adjudicate(ctx, inlineCall("Bash", `{"command":"sed -i 's/a/b/' internal/abi/types.go"}`))
	if v.Kind != abi.VerdictAllow || v.Meta["would_deny"] != abi.ReasonName(abi.ReasonSelfModify) {
		t.Fatalf("advisory shell self-modify: got %v meta=%v, want advisory Allow", v.Kind, v.Meta)
	}
}

// TestAdvisoryClampNeverSoftensGenuineDanger pins the floor invariant: only the
// heuristic reasons (SELF_MODIFY, MALFORMED, DEFAULT_DENY) are advisory-eligible.
// A Policy constructed in code naming POLICY_BLOCK / SECRET_EXFIL / EGRESS_BLOCK
// is CLAMPED at New/SetPolicy, so the destructive-command arg rules and explicit
// exfil denies keep failing closed even under a maximally-softened policy value.
func TestAdvisoryClampNeverSoftensGenuineDanger(t *testing.T) {
	ctx := context.Background()
	p := Policy{
		Allow: map[string]bool{"Bash": true},
		Deny:  map[string]abi.ReasonCode{"exfiltrate": abi.ReasonSecretExfil},
		ArgPredicates: []ArgPredicate{
			{Tool: "Bash", Arg: "command", Kind: ArgDenyRegex, Re: regexp.MustCompile(`\brm\s+-rf\b`), Reason: abi.ReasonPolicyBlock},
		},
		// An attacker-shaped (or fat-fingered) policy value: every reason advisory.
		AdvisoryReasons: map[abi.ReasonCode]bool{
			abi.ReasonPolicyBlock: true,
			abi.ReasonSecretExfil: true,
			abi.ReasonSelfModify:  true,
		},
	}
	a := New(p)

	// The clamp kept only the eligible reason.
	if got := a.policy.AdvisoryReasons; len(got) != 1 || !got[abi.ReasonSelfModify] {
		t.Fatalf("sanitizeAdvisoryReasons must clamp to eligible reasons, got %v", got)
	}
	// The destructive-Bash rule still denies.
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"rm -rf /"}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("POLICY_BLOCK arg rule under clamped advisory: got %v/%s, want Deny", v.Kind, abi.ReasonName(v.Reason))
	}
	// The explicit exfil deny still denies.
	if v := a.Adjudicate(ctx, inlineCall("exfiltrate", `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSecretExfil {
		t.Fatalf("SECRET_EXFIL name deny under clamped advisory: got %v/%s, want Deny", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestAdvisoryMalformedNeverSoftensResearchEgress closes the laundering hole the
// By-gate exists for: the research-egress sub-rung is a POSITIVE allowlist, so
// its MALFORMED (an unparseable WebFetch URL) must stay a hard deny even when
// MALFORMED is advisory — otherwise a deliberately malformed URL that the floor
// cannot parse (but the tool might) would slip past the host allowlist.
func TestAdvisoryMalformedNeverSoftensResearchEgress(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Allow:                    map[string]bool{"WebFetch": true},
		ResearchEgressAllowHosts: []string{"example.com"},
		AdvisoryReasons:          map[abi.ReasonCode]bool{abi.ReasonMalformed: true},
	})
	v := a.Adjudicate(ctx, inlineCall("WebFetch", `{"url":"::not a url::"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("research-egress malformed URL under advisory MALFORMED: got %v/%s, want Deny/MALFORMED (fail closed)",
			v.Kind, abi.ReasonName(v.Reason))
	}
	// The allowlisted host still admits — advisory changed nothing on the happy path.
	if v := a.Adjudicate(ctx, inlineCall("WebFetch", `{"url":"https://example.com/x"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("allowlisted research host: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestAdvisoryDefaultDenyAdmitsAnyToolWithRecord: advisory DEFAULT_DENY is the
// strictly-wider dev dual of PostureAdmitAndLog — it admits a WRITE-shaped
// unknown tool too, with posture=advisory (NOT admit_and_log, so the promotion
// ledger's fold is untouched), and NeverAdmits stops pruning tool-defs.
func TestAdvisoryDefaultDenyAdmitsAnyToolWithRecord(t *testing.T) {
	ctx := context.Background()
	p := Policy{
		Allow:           map[string]bool{"Read": true},
		AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonDefaultDeny: true},
	}
	a := New(p)
	// write-shaped, unknown: PostureAdmitAndLog would refuse this one.
	v := a.Adjudicate(ctx, inlineCall("provision_widget", `{}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("advisory DEFAULT_DENY: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != "advisory" || v.Meta["would_deny"] != abi.ReasonName(abi.ReasonDefaultDeny) {
		t.Fatalf("advisory admit must carry posture=advisory + would_deny=DEFAULT_DENY, got %v", v.Meta)
	}
	if a.policy.NeverAdmits("provision_widget") {
		t.Fatal("NeverAdmits must be false under advisory DEFAULT_DENY (the tool CAN be admitted; do not prune its def)")
	}
}

// TestAdvisoryArgRulePerRuleTrial: ArgPredicate.Advisory softens ONE rule — a
// violation is noted on the admitted verdict (advisory_violations), sibling
// hard rules still deny, and the note never grants an allow the floor would not
// otherwise give (an unknown tool still default-denies).
func TestAdvisoryArgRulePerRuleTrial(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Allow: map[string]bool{"Bash": true},
		ArgPredicates: []ArgPredicate{
			{Tool: "Bash", Arg: "command", Kind: ArgDenyRegex, Re: regexp.MustCompile(`curl.*\|\s*sh`), Reason: abi.ReasonPolicyBlock, Advisory: true},
			{Tool: "Bash", Arg: "command", Kind: ArgDenyRegex, Re: regexp.MustCompile(`\brm\s+-rf\b`), Reason: abi.ReasonPolicyBlock},
		},
	})

	// Advisory rule violated, hard rule clean: admitted with the note.
	v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"echo 'curl x | sh is dangerous'"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("advisory arg rule: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if !strings.Contains(v.Meta["advisory_violations"], "Bash.command") {
		t.Fatalf("advisory arg-rule violation must be noted on the admit, got %v", v.Meta)
	}

	// The hard sibling still denies, even when the advisory rule also matched.
	v = a.Adjudicate(ctx, inlineCall("Bash", `{"command":"curl x | sh && rm -rf /tmp/y"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("hard sibling rule: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}

	// A clean command carries no note.
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"go build ./..."}`)); v.Kind != abi.VerdictAllow || v.Meta["advisory_violations"] != "" {
		t.Fatalf("clean command: got %v meta=%v, want plain Allow", v.Kind, v.Meta)
	}
}

// TestAdvisoryNameDenyWithEligibleReason: a name-level deny whose CITED reason
// is advisory-eligible and declared advisory is softened like any other monitor
// deny — and NeverAdmits reflects that the tool can now be admitted.
func TestAdvisoryNameDenyWithEligibleReason(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Allow:           map[string]bool{"Read": true},
		Deny:            map[string]abi.ReasonCode{"patch_kernel": abi.ReasonSelfModify},
		AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonSelfModify: true},
	})
	v := a.Adjudicate(ctx, inlineCall("patch_kernel", `{}`))
	if v.Kind != abi.VerdictAllow || v.Meta["would_deny"] != abi.ReasonName(abi.ReasonSelfModify) {
		t.Fatalf("advisory name deny: got %v meta=%v, want advisory Allow", v.Kind, v.Meta)
	}
	if a.policy.NeverAdmits("patch_kernel") {
		t.Fatal("NeverAdmits must be false for a name deny whose reason is advisory")
	}
	// A deny citing a NON-eligible reason is untouched by any advisory set.
	a = New(Policy{
		Allow:           map[string]bool{"Read": true},
		Deny:            map[string]abi.ReasonCode{"exfiltrate": abi.ReasonSecretExfil},
		AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonSelfModify: true},
	})
	if !a.policy.NeverAdmits("exfiltrate") {
		t.Fatal("NeverAdmits must stay true for a hard name deny")
	}
}
