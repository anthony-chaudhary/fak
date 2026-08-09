package adjudicator

import (
	"strings"
	"testing"
)

// TestDeclaredToolActionRulesSeeOnlySinkArgs retires the recorded "commit
// message mentions rm" over-block class (#5898, the over-broad complaint in
// internal/guardaccuracy/complaints_test.go): a commit MESSAGE that mentions a
// destructive command is payload, not an action. Joined into command text, its
// newline manufactures a segment whose head lands on `rm` — quote-stripping
// (#2752) cannot help because a slice element carries no quote bytes. With
// git_commit's sink declaration, action rules never read the message, so the
// call classifies by what the tool DOES, not what the prose SAYS.
func TestDeclaredToolActionRulesSeeOnlySinkArgs(t *testing.T) {
	prose := map[string]any{
		"args": []string{"-m", "cleanup: drop the build helper\nrm -rf build is no longer needed && git push"},
	}
	for _, tool := range []string{"git_commit", "mcp__git__git_commit"} {
		env := ClassifyReversibility(tool, prose)
		if env.Class != ReversibilityReversible {
			t.Fatalf("%s: prose in a declared tool's payload classified as %q; the message must be invisible to action rules", tool, env.Class)
		}
		// The sidestep reads the same scoped view: prose mentioning `git push`
		// must not offer a rewrite either.
		if env.RewriteCommand != "" {
			t.Fatalf("%s: payload prose populated a sidestep rewrite %q", tool, env.RewriteCommand)
		}
	}
}

// TestUndeclaredToolKeepsFullArgumentScan is the fail-closed witness: a tool
// with NO declaration keeps the old full-argument scan, in both directions —
// a real argv destroy still classifies irreversible, and the exact prose shape
// the declaration retires for git_commit STILL escalates on an undeclared
// tool. Together with the test above this proves the declaration, not some
// global loosening, is what retires the over-block class.
func TestUndeclaredToolKeepsFullArgumentScan(t *testing.T) {
	env := ClassifyReversibility("run_process", map[string]any{
		"argv": []string{"rm", "-rf", "build"},
	})
	if env.Class != ReversibilityIrreversible {
		t.Fatalf("undeclared tool's argv destroy classified %q, want irreversible", env.Class)
	}

	env = ClassifyReversibility("run_process", map[string]any{
		"args": []string{"-m", "cleanup: drop the build helper\nrm -rf build is no longer needed && git push"},
	})
	if env.Class == ReversibilityReversible {
		t.Fatal("undeclared tool loosened: the prose shape stopped scanning without a declaration")
	}
}

// TestDataRuleUnaffectedBySinkDeclaration pins the other half of the split: a
// DATA rule keeps its full-argument view regardless of any declaration. The
// preview secret redaction must still see (and redact) a credential riding in
// git_commit's NON-sink message argument — narrowing a data rule to a sink
// would be a leak, not a false-positive fix.
func TestDataRuleUnaffectedBySinkDeclaration(t *testing.T) {
	env := ClassifyReversibility("git_commit", map[string]any{
		"args": []string{"-m", "note the rotation api_key=secret123"},
	})
	if strings.Contains(env.Preview, "secret123") {
		t.Fatalf("preview leaked a secret from a sink-scoped argument: %q", env.Preview)
	}
	if !strings.Contains(env.Preview, "api_key=[REDACTED]") {
		t.Fatalf("data redaction stopped reading non-sink arguments: %q", env.Preview)
	}
}

// TestDeclaredIssueToolKeepsNameFamilyEscalation proves an empty sink set does
// not neuter tool-NAME matching: create_issue stays outward-facing via the
// issue-create-tool family (its capability is real) with its own redirect,
// while body prose naming another family's trigger cannot drag that family's
// hint onto the call.
func TestDeclaredIssueToolKeepsNameFamilyEscalation(t *testing.T) {
	env := ClassifyReversibility("create_issue", map[string]any{
		"args": []string{"--title", "bug", "--body", "repro: rm -rf build\nthen git push --force"},
	})
	if env.Class != ReversibilityOutwardFacing {
		t.Fatalf("create_issue class = %q, want outward-facing (the tool-name family must survive an empty sink set)", env.Class)
	}
	if !strings.Contains(env.DryRunHint, "fak issue create") {
		t.Fatalf("issue-create redirect lost: %q", env.DryRunHint)
	}
	if strings.Contains(env.DryRunHint, "fak sync push") {
		t.Fatalf("body prose dragged the git-push redirect onto an issue create: %q", env.DryRunHint)
	}
}

// TestSinkScopedArgsKeepSinkKeys is the unit-level positive witness: a
// declared sink key stays visible to action rules; everything else is dropped.
func TestSinkScopedArgsKeepSinkKeys(t *testing.T) {
	decl := toolSinkDecl{sinkKeys: map[string]bool{"command": true}}
	got := sinkScopedArgs(decl, map[string]any{"command": "rm -rf build", "message": "prose"})
	if got["command"] != "rm -rf build" {
		t.Fatalf("sink key dropped: %v", got)
	}
	if _, ok := got["message"]; ok {
		t.Fatalf("non-sink key survived scoping: %v", got)
	}
}

// TestToolSinkDeclsAreReviewable pins table hygiene: lookups lower the tool
// and argument names, so a declaration keyed with uppercase would silently
// never match — fail loud here instead. Every entry must also name its
// capability, the provenance a reviewer needs to judge the sink set.
func TestToolSinkDeclsAreReviewable(t *testing.T) {
	for name, d := range toolSinkDecls {
		if name != strings.ToLower(name) {
			t.Errorf("declaration key %q is not lowercase; sinkDeclFor would never match it", name)
		}
		if strings.TrimSpace(d.capability) == "" {
			t.Errorf("declaration %q names no capability", name)
		}
		for k := range d.sinkKeys {
			if k != strings.ToLower(k) {
				t.Errorf("declaration %q sink key %q is not lowercase; sinkScopedArgs would never match it", name, k)
			}
		}
	}
}
