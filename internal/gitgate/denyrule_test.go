package gitgate

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// gitgateCall builds a shell tool call the way the guard wires one.
func gitgateCall(command string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: "Bash",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":` + quoteJSON(command) + `}`)},
	}
}

func quoteJSON(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// TestGitGateDenyNamesItsLaw is the #5863 win for the cluster the issue measured:
// five terminal worker deaths whose args_label read only "command=cd fak" and
// four more reading "command=git config", ALL citing the same
// ("gitgate", "POLICY_BLOCK") pair. The law that refused was on the wire only as
// the opening words of a claim up to 447 characters long; now each refusal names
// its law as a closed-vocabulary id a fold can group on.
func TestGitGateDenyNamesItsLaw(t *testing.T) {
	g := New()
	cases := []struct {
		command string
		want    string
	}{
		// The exact two shapes behind the undecodable cluster.
		{`cd fak && git config core.hooksPath /tmp/nohooks`, abi.DenyRuleSkipHooks},
		{`cd fak && git checkout -b feature/x`, abi.DenyRuleOffTrunk},
		{`git config core.hooksPath .nohooks`, abi.DenyRuleSkipHooks},
		{`git config commit.gpgsign false`, abi.DenyRuleSkipSigning},
		// Siblings that collapse onto the same (By, Reason) today.
		{`git commit --amend -m x`, abi.DenyRuleNeverAmendShared},
		{`git push --force origin main`, abi.DenyRuleNeverAmendShared},
		{`git add -A`, abi.DenyRuleCommitByPath},
		{`git reset --hard origin/main`, abi.DenyRuleResetHard},
		{`git clean -f`, abi.DenyRuleCleanForce},
		{`git push --mirror origin`, abi.DenyRulePushMirror},
	}
	for _, c := range cases {
		v := g.Adjudicate(context.Background(), gitgateCall(c.command))
		if v.Kind != abi.VerdictDeny {
			t.Errorf("%q: got %v, want Deny", c.command, v.Kind)
			continue
		}
		if got := v.Meta[abi.MetaDenyRule]; got != c.want {
			t.Errorf("%q: Meta[%s] = %q, want %q", c.command, abi.MetaDenyRule, got, c.want)
		}
		// The remedy channel is untouched — the id is additive, not a replacement.
		if v.Meta["fix"] == "" {
			t.Errorf("%q: lost Meta[fix]", c.command)
		}
	}
}

// TestEveryGitGateLawHasADeclaredRuleID is the sync guard between gitgate's laws
// and abi's closed vocabulary. Every law is AUTHORED as "<law-id>[ refused]:
// <prose>", so its leading atom must resolve to a declared id. A new law with an
// undeclared prefix stamps nothing — safe, but silently unroutable — so this
// test makes that omission a build failure instead of a blank field discovered
// months later in a corpus fold.
func TestEveryGitGateLawHasADeclaredRuleID(t *testing.T) {
	laws := make([]string, 0, len(defaultHazards)+8)
	for _, h := range defaultHazards {
		laws = append(laws, h.law)
	}
	laws = append(laws,
		dotAddLaw, offTrunkBranchLaw, historyRewriteLaw, configHooksLaw, configSignLaw,
		neverAmendSharedLaw,
	)
	for _, law := range laws {
		id, ok := abi.DenyRuleID(law)
		if !ok {
			head := law
			if i := strings.IndexAny(head, " \t"); i >= 0 {
				head = head[:i]
			}
			t.Errorf("law prefix %q is not in the abi deny-rule vocabulary; "+
				"declare it in internal/abi/denyrules.go (law: %.60s…)", head, law)
			continue
		}
		if strings.Contains(id, " ") || len(id) > 64 {
			t.Errorf("law %.40s… resolved to a malformed id %q", law, id)
		}
	}
}

// TestGitGateNeverStampsLawProse is the disclosure guard: the id must be the
// law's authored key, never any of the prose that follows it — the prose names
// concrete paths and command lines, and this field is copied verbatim into the
// exported guard corpus.
func TestGitGateNeverStampsLawProse(t *testing.T) {
	declared := map[string]bool{}
	for _, id := range abi.DenyRuleIDs() {
		declared[id] = true
	}
	g := New()
	v := g.Adjudicate(context.Background(), gitgateCall(`git config core.hooksPath /home/agent/.secrets/hooks`))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("got %v, want Deny", v.Kind)
	}
	id := v.Meta[abi.MetaDenyRule]
	if !declared[id] {
		t.Fatalf("stamped %q, which is not a declared id", id)
	}
	for _, leak := range []string{"/home/agent", ".secrets", "hooksPath", "refused", "`git"} {
		if strings.Contains(id, leak) {
			t.Fatalf("rule id %q leaked %q", id, leak)
		}
	}
}
