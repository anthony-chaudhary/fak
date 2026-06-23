package gitgate

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestClassify is the table that pins the rung's structural decisions: which git
// command shapes are PROVABLE hazards (deny) and which are allowed-through
// (defer). The defer cases carry the load: they prove the rung does NOT
// false-positive on a safe push, an explicit-path add, a commit whose MESSAGE
// mentions a hazardous flag, or a non-git command that merely contains the word
// "git" (echo/grep).
func TestClassify(t *testing.T) {
	g := New()
	cases := []struct {
		name   string
		cmd    string
		deny   bool
		lawHas string // substring the cited law must contain (deny cases only)
	}{
		// ---- force-push ------------------------------------------------------
		{"force long", "git push --force origin main", true, "force-push"},
		{"force short", "git push -f", true, "force-push"},
		{"force bundled", "git push -fq origin main", true, "force-push"},
		{"force-with-lease", "git push --force-with-lease", true, "force-push"},
		{"force-with-lease value", "git push --force-with-lease=origin/main", true, "force-push"},
		{"push upstream OK", "git push -u origin main", false, ""},
		{"plain push OK", "git push origin main", false, ""},
		{"push set-upstream OK", "git push --set-upstream origin main", false, ""},

		// ---- skip hooks / signing -------------------------------------------
		{"push no-verify", "git push --no-verify", true, "skip-hooks"},
		{"commit no-verify long", "git commit --no-verify -m x", true, "skip-hooks"},
		{"commit -n", `git commit -n -m "x"`, true, "skip-hooks"},
		{"commit no-gpg-sign", "git commit --no-gpg-sign -m x", true, "skip-signing"},
		{"global hooksPath", "git -c core.hooksPath=/dev/null commit -m x", true, "skip-hooks"},
		{"global hooksPath case", "git -c CORE.HooksPath= commit -m x", true, "skip-hooks"},

		// ---- amend ----------------------------------------------------------
		{"amend", "git commit --amend", true, "amend"},
		{"amend after -m", `git commit -m "msg" --amend`, true, "amend"},
		{"amend no-edit", "git commit --amend --no-edit", true, "amend"},

		// ---- remote-ref delete ----------------------------------------------
		{"push delete long", "git push --delete origin feature", true, "remote-ref delete"},
		{"push delete short", "git push -d origin feature", true, "remote-ref delete"},

		// ---- commit-by-explicit-path ----------------------------------------
		{"commit -a", "git commit -a", true, "explicit-path"},
		{"commit -am", `git commit -am "msg"`, true, "explicit-path"},
		{"commit --all", "git commit --all", true, "explicit-path"},
		{"add -A", "git add -A", true, "explicit-path"},
		{"add --all", "git add --all", true, "explicit-path"},
		{"add -u", "git add -u", true, "explicit-path"},
		{"add dot", "git add .", true, "explicit-path"},
		{"add dot after dashdash", "git add -- .", true, "explicit-path"},

		// ---- tag / rebase ---------------------------------------------------
		{"tag force", "git tag -f v1", true, "tag-force"},
		{"tag delete", "git tag -d v1", true, "tag-delete"},
		{"rebase -i", "git rebase -i HEAD~3", true, "history-rewrite"},
		{"rebase interactive long", "git rebase --interactive origin/main", true, "history-rewrite"},

		// ---- allowed git ops (DEFER) ----------------------------------------
		{"commit explicit OK", `git commit -s -m "fix(x): do y (fak x)"`, false, ""},
		{"commit -sm OK", `git commit -sm "msg"`, false, ""},
		{"add explicit OK", "git add internal/gitgate/gitgate.go", false, ""},
		{"add -p OK", "git add -p internal/x.go", false, ""},
		{"add file named -A after dashdash", "git add -- -A", false, ""},
		{"tag create OK", "git tag v1.2.3", false, ""},
		{"rebase plain OK", "git rebase origin/main", false, ""},
		{"status OK", "git status", false, ""},
		{"log OK", "git log --oneline -n 20", false, ""},
		{"diff OK", "git diff --cached", false, ""},
		{"fetch OK", "git fetch origin", false, ""},
		{"pull OK", "git pull --rebase origin main", false, ""},

		// ---- KEY false-positive guards --------------------------------------
		// A hazardous flag mentioned INSIDE a quoted commit message is an operand,
		// not a flag, so it must NOT trigger.
		{"flag inside message", `git commit -m "always use git push --force everywhere"`, false, ""},
		{"add -A inside message", `git commit -m "stop using git add -A please"`, false, ""},
		// A non-git command that merely contains the word git/force must DEFER.
		{"echo mentions force", `echo "git push --force"`, false, ""},
		{"grep mentions add -A", `grep -r "git add -A" .`, false, ""},
		{"non-git command", "ls -la", false, ""},
		{"git in path name", "ls /home/git/repo", false, ""},

		// ---- prefixes, chains, paths, casing --------------------------------
		{"env prefix", "env FOO=bar git push -f", true, "force-push"},
		{"assignment prefix", "GIT_TRACE=1 git push --force", true, "force-push"},
		{"chained cd then force", "cd repo && git push --force", true, "force-push"},
		{"semicolon chain", "git status; git push -f", true, "force-push"},
		{"global -C then force", "git -C sub push --force", true, "force-push"},
		{"abs path git", "/usr/bin/git push --force", true, "force-push"},
		{"git.exe", `git.exe push --force`, true, "force-push"},
		{"uppercase GIT", "GIT push --force", true, "force-push"},
		{"subshell launder caught", "echo $(git push --force)", true, "force-push"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			law, denied := g.Classify(tc.cmd)
			if denied != tc.deny {
				t.Fatalf("Classify(%q) denied=%v, want %v (law=%q)", tc.cmd, denied, tc.deny, law)
			}
			if tc.deny && !strings.Contains(law, tc.lawHas) {
				t.Fatalf("Classify(%q) law=%q, want substring %q", tc.cmd, law, tc.lawHas)
			}
		})
	}
}

// cmdCall builds a shell tool call with the given command as inline JSON args
// (no resolver needed — refBytes reads RefInline directly), using json.Marshal so
// quotes in cmd survive into the args bytes.
func cmdCall(tool, key, cmd string) *abi.ToolCall {
	b, _ := json.Marshal(map[string]string{key: cmd})
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: b}}
}

// TestAdjudicate proves the end-to-end verdict path: a hazardous shell call Denies
// with ReasonPolicyBlock + a witness Claim; a safe one and a non-shell one Defer.
func TestAdjudicate(t *testing.T) {
	g := New()
	ctx := context.Background()

	v := g.Adjudicate(ctx, cmdCall("Bash", "command", "git push --force origin main"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("force push: Kind=%v, want Deny", v.Kind)
	}
	if v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("force push: Reason=%s, want POLICY_BLOCK", abi.ReasonName(v.Reason))
	}
	if v.By != "gitgate" {
		t.Fatalf("force push: By=%q, want gitgate", v.By)
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok || !strings.Contains(wp.Claim, "force-push") {
		t.Fatalf("force push: witness Claim=%v, want a force-push law", v.Payload)
	}

	// The `cmd` key convention is read too.
	if v := g.Adjudicate(ctx, cmdCall("exec", "cmd", "git commit --amend")); v.Kind != abi.VerdictDeny {
		t.Fatalf("amend via cmd key: Kind=%v, want Deny", v.Kind)
	}

	// Safe shell call defers.
	if v := g.Adjudicate(ctx, cmdCall("Bash", "command", "git status")); v.Kind != abi.VerdictDefer {
		t.Fatalf("git status: Kind=%v, want Defer", v.Kind)
	}
	// A non-shell call (no command/cmd arg) defers.
	notShell := &abi.ToolCall{Tool: "read_file", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"x"}`)}}
	if v := g.Adjudicate(ctx, notShell); v.Kind != abi.VerdictDefer {
		t.Fatalf("read_file: Kind=%v, want Defer", v.Kind)
	}
	// A nil call defers (no panic).
	if v := g.Adjudicate(ctx, nil); v.Kind != abi.VerdictDefer {
		t.Fatalf("nil call: Kind=%v, want Defer", v.Kind)
	}
}

// TestEmptyRulesDefers proves a gate with no rules is inert (the FAK_GITGATE=off
// shape, modeled as an empty rule set).
func TestEmptyRulesDefers(t *testing.T) {
	g := &GitGate{}
	if v := g.Adjudicate(context.Background(), cmdCall("Bash", "command", "git push --force")); v.Kind != abi.VerdictDefer {
		t.Fatalf("empty rules: Kind=%v, want Defer", v.Kind)
	}
}

// TestCapsNil mirrors the other rungs: the baseline gate advertises no caps.
func TestCapsNil(t *testing.T) {
	if c := New().Caps(); c != nil {
		t.Fatalf("Caps()=%v, want nil", c)
	}
}

// TestRegistered proves init() put the rung + capability on the kernel (so the
// architest registration-completeness gate has something real to bind to). Skipped
// only when the operator opt-out is active in the test environment.
func TestRegistered(t *testing.T) {
	if strings.EqualFold(os.Getenv("FAK_GITGATE"), "off") {
		t.Skip("FAK_GITGATE=off: rung intentionally unregistered")
	}
	if !abi.Supported("gitgate.v1") {
		t.Fatal("gitgate.v1 capability not registered — init() did not run or did not register")
	}
}
