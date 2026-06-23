package adjudicator

import (
	"context"
	"fmt"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// manyPreds builds a policy with ONE real arg rule on write_file plus n unrelated
// per-tool predicates, to exercise the per-tool index (Fix E).
func manyPreds(n int) Policy {
	preds := []ArgPredicate{{
		Tool: "write_file", Arg: "path", Kind: ArgAllowGlob, Glob: "./out/**", Reason: abi.ReasonPolicyBlock,
	}}
	for i := 0; i < n; i++ {
		preds = append(preds, ArgPredicate{
			Tool: fmt.Sprintf("other-%d", i), Arg: "x", Kind: ArgDenyRegex,
			Re: regexp.MustCompile("zzz"), Reason: abi.ReasonPolicyBlock,
		})
	}
	return Policy{Allow: map[string]bool{"write_file": true, "read_x": true}, ArgPredicates: preds}
}

// TestArgPredicatesIndexedByTool is the Fix E behavior-identity guard: evaluating
// only the predicates that target the call's tool yields the SAME verdict it would
// with a flat scan, no matter how many unrelated-tool predicates the policy carries.
func TestArgPredicatesIndexedByTool(t *testing.T) {
	a := New(manyPreds(2000)) // 2001 predicates total, 1 for write_file
	ctx := context.Background()

	if v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"./out/r.txt"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("in-bound write: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"./out/../secrets"}`)); v.Kind != abi.VerdictDeny {
		t.Fatalf("path escape: got %v, want Deny (its own predicate, unaffected by 2000 others)", v.Kind)
	}
	// A tool with no predicate is unaffected by the 2001 predicates.
	if v := a.Adjudicate(ctx, inlineCall("read_x", `{}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("unconstrained allowed tool: got %v, want Allow", v.Kind)
	}
}

// TestArgPredicatesCaseInsensitiveTool locks the multi-agent guarantee: a rule
// authored against one tool casing ("Bash") still gates a differently-cased call
// ("bash" / "BASH"), so a capability floor cannot fail OPEN just because an agent
// names its shell tool in lowercase (Claude Code "Bash" vs OpenCode "bash").
func TestArgPredicatesCaseInsensitiveTool(t *testing.T) {
	p := Policy{
		Allow: map[string]bool{"Bash": true, "bash": true, "BASH": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(`\brm\s+-rf\b`), Reason: abi.ReasonPolicyBlock,
		}},
	}
	a := New(p)
	ctx := context.Background()
	for _, tool := range []string{"Bash", "bash", "BASH"} {
		if v := a.Adjudicate(ctx, inlineCall(tool, `{"command":"rm -rf /"}`)); v.Kind != abi.VerdictDeny {
			t.Errorf("%s rm -rf: got %v, want Deny (the Bash rule must gate every casing)", tool, v.Kind)
		}
		if v := a.Adjudicate(ctx, inlineCall(tool, `{"command":"ls -la"}`)); v.Kind != abi.VerdictAllow {
			t.Errorf("%s benign: got %v, want Allow", tool, v.Kind)
		}
	}
}

// TestSelfModifyCamelCasePath locks that the self-modify glob check reads a camelCase
// filePath argument (OpenCode / AI-SDK tools), not only snake_case file_path — else an
// agent that names the arg differently could write into a guarded tree unchecked.
func TestSelfModifyCamelCasePath(t *testing.T) {
	a := New(Policy{
		Allow:           map[string]bool{"write": true},
		SelfModifyGlobs: []string{".ssh/", ".git/"},
	})
	ctx := context.Background()
	if v := a.Adjudicate(ctx, inlineCall("write", `{"filePath":".ssh/authorized_keys","content":"x"}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Errorf("write into .ssh via filePath: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := a.Adjudicate(ctx, inlineCall("write", `{"filePath":"notes.txt","content":"x"}`)); v.Kind != abi.VerdictAllow {
		t.Errorf("in-tree write via filePath: got %v, want Allow", v.Kind)
	}
}

// BenchmarkAdjudicateArgScaling shows the per-call cost of adjudicating a
// constrained tool stays FLAT as the number of unrelated-tool predicates grows —
// the index makes it O(predicates-for-this-tool), not O(all predicates).
func BenchmarkAdjudicateArgScaling(b *testing.B) {
	ctx := context.Background()
	c := inlineCall("write_file", `{"path":"./out/r.txt"}`)
	for _, n := range []int{0, 100, 2000} {
		b.Run(fmt.Sprintf("unrelated=%d", n), func(b *testing.B) {
			a := New(manyPreds(n))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = a.Adjudicate(ctx, c)
			}
		})
	}
}
