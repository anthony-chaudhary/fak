package adjudicator

import (
	"context"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestArgDenyStampsClosedVocabularyRuleID drives the REAL adjudicator over the
// REAL shipped recursive/forced-delete rule and asserts the refusal now names its
// rung on Verdict.Meta. Before #5863 this rung's identity existed only inside the
// witness Claim's prose ("Bash.command rm_rf recursive/forced delete"), which is
// why a consumer folding .dispatch-runs/guard-audit/*.jsonl could not separate it
// from any other ("monitor", "POLICY_BLOCK") refusal without a bespoke parser.
func TestArgDenyStampsClosedVocabularyRuleID(t *testing.T) {
	a := rcePipeAdj(t, defaultRmRfDenyRegex)
	v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(`cd fak && rm -rf build`)))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("got %v/%s, want Deny", v.Kind, abi.ReasonName(v.Reason))
	}
	if got := v.Meta[abi.MetaDenyRule]; got != abi.DenyRuleRmRf {
		t.Fatalf("Meta[%s] = %q, want %q", abi.MetaDenyRule, got, abi.DenyRuleRmRf)
	}
	// The rule id is a promotion of what the claim already said — no new disclosure.
	claim, _ := v.Payload.(abi.WitnessPayload)
	if !strings.Contains(claim.Claim, abi.DenyRuleRmRf) {
		t.Fatalf("rule id %q is not already present in the witness claim %q — "+
			"the stamp must disclose nothing the claim did not", abi.DenyRuleRmRf, claim.Claim)
	}
}

// TestArgDenyRuleIDsAreDeclared walks the arg-predicate kinds whose detail the
// rung authors and asserts each maps onto a DECLARED id. It is the sync guard
// between decide_argpredicates.go's detail literals and abi's vocabulary: a rung
// that invents a new detail prefix without declaring it stamps nothing (safe),
// and this test says so out loud instead of letting the field go quietly blank.
func TestArgDenyRuleIDsAreDeclared(t *testing.T) {
	details := []struct {
		detail string
		want   string
	}{
		{"rm_rf recursive/forced delete", abi.DenyRuleRmRf},
		{"rce_pipe download|interpreter", abi.DenyRuleRCEPipe},
		{"out_of_tree_write", abi.DenyRuleOutOfTreeWrite},
		{"shell_dialect Get-Content", abi.DenyRuleShellDialect},
		{"sudo_local sudo", abi.DenyRuleSudoLocal},
		{"runas_elevation", abi.DenyRuleRunAsElevation},
		{"terraform_destroy", abi.DenyRuleTerraformDestroy},
		{"device_op", abi.DenyRuleDeviceOp},
		{"cli_read_only", abi.DenyRuleCLIReadOnly},
		{"max_bytes 4096", abi.DenyRuleMaxBytes},
		{"allow_glob ./out/**", abi.DenyRuleAllowGlob},
		{`deny_regex /\bsudo\b/`, abi.DenyRuleDenyRegex},
	}
	pr := &ArgPredicate{Tool: "Bash", Arg: "command", Reason: abi.ReasonPolicyBlock}
	for _, d := range details {
		v := argDeny(pr, d.detail)
		if got := v.Meta[abi.MetaDenyRule]; got != d.want {
			t.Errorf("argDeny(%q) stamped %q, want %q", d.detail, got, d.want)
		}
	}
}

// TestArgDenyNeverStampsTheArgValue is the disclosure guard on this seam. The
// trailing half of a detail can carry POLICY text (a glob, a regex) and the
// witness claim carries the tool.arg — none of it may reach the rule id, whose
// value space is the declared vocabulary and nothing else.
func TestArgDenyNeverStampsTheArgValue(t *testing.T) {
	pr := &ArgPredicate{
		Tool: "Bash", Arg: "command", Reason: abi.ReasonPolicyBlock,
		Glob: "/home/agent/.ssh/**",
		Re:   regexp.MustCompile(`hunter2`),
	}
	hostile := []string{
		"allow_glob /home/agent/.ssh/**",
		`deny_regex /hunter2/`,
		"MYVAR=hunter2 go test",                // a detail an unlucky refactor could pass through
		"sk-ant-api03-SHAPED recursive/forced", // a secret-shaped leading atom
		"rm_rf=hunter2",                        // a real id fused to a payload
	}
	declared := map[string]bool{}
	for _, id := range abi.DenyRuleIDs() {
		declared[id] = true
	}
	for _, detail := range hostile {
		v := argDeny(pr, detail)
		got := v.Meta[abi.MetaDenyRule]
		if got == "" {
			continue // rejected whole: the correct outcome for an undeclared atom
		}
		if !declared[got] {
			t.Errorf("argDeny(%q) stamped %q, which is not a declared id", detail, got)
		}
		if strings.Contains(got, "hunter2") || strings.Contains(got, "ssh") || strings.Contains(got, "sk-") {
			t.Errorf("argDeny(%q) leaked into the rule id: %q", detail, got)
		}
	}
}

// TestSelfModifyRungsAreDistinguishable is the second half of the win: three
// rungs that all cite SELF_MODIFY and all disclose only the offending glob were
// one undifferentiated bucket on the wire (34 rows in the measured corpus). The
// rule id separates the path-arg rung from the shell-command rung, which is
// exactly the distinction #5863 hypothesised about but could not confirm — a
// `cd fak && …` compound refused as SELF_MODIFY now says which door it came in.
func TestSelfModifyRungsAreDistinguishable(t *testing.T) {
	a := New(DefaultPolicy())
	pathDeny := a.Adjudicate(context.Background(), inlineCall("write_file", `{"path":"internal/abi/types.go"}`))
	if pathDeny.Kind != abi.VerdictDeny || abi.ReasonName(pathDeny.Reason) != "SELF_MODIFY" {
		t.Fatalf("path write: got %v/%s, want Deny/SELF_MODIFY", pathDeny.Kind, abi.ReasonName(pathDeny.Reason))
	}
	cmdDeny := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(`cd fak && sed -i s/a/b/ internal/abi/types.go`)))
	if cmdDeny.Kind != abi.VerdictDeny || abi.ReasonName(cmdDeny.Reason) != "SELF_MODIFY" {
		t.Fatalf("shell write: got %v/%s, want Deny/SELF_MODIFY", cmdDeny.Kind, abi.ReasonName(cmdDeny.Reason))
	}
	// Same tool-agnostic (By, Reason) pair, same witness SHAPE (a bare glob) —
	// separable only by the rule id.
	if pathDeny.By != cmdDeny.By || pathDeny.Reason != cmdDeny.Reason {
		t.Fatalf("precondition: the two rungs should be indistinguishable on (By,Reason); got %q/%v vs %q/%v",
			pathDeny.By, pathDeny.Reason, cmdDeny.By, cmdDeny.Reason)
	}
	if got := pathDeny.Meta[abi.MetaDenyRule]; got != abi.DenyRuleSelfModifyPath {
		t.Errorf("path-arg rung id = %q, want %q", got, abi.DenyRuleSelfModifyPath)
	}
	if got := cmdDeny.Meta[abi.MetaDenyRule]; got != abi.DenyRuleSelfModifyCommand {
		t.Errorf("shell-command rung id = %q, want %q", got, abi.DenyRuleSelfModifyCommand)
	}
}
