package gitgate

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/witness"
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
		{"rebase autostash", "git rebase --autostash origin/main", true, "autostash refused"},
		{"pull rebase autostash", "git pull --rebase --autostash origin main", true, "autostash refused"},

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

		// ---- shell-laundering recovered by the unwrap pass (#823) -------------
		// A git hazard the shell grammar wraps around the call: a pipe, an operator,
		// a `$(...)` / backtick command substitution, or a `bash -c`/`sh -c` string.
		// The unwrap pass makes each VISIBLE to the SAME defaultHazards rules.
		{"pipe to force", "echo x | git push --force origin main", true, "force-push"},
		{"and-and force", "true && git push --force", true, "force-push"},
		{"or-or force", "git status || git push -f", true, "force-push"},
		{"cmdsubst force inner", "git push $(printf -- --force)", false, ""}, // the FLAG is in a $() printf builds at runtime — undecidable, defer
		{"cmdsubst whole git call", "echo $(git push -f)", true, "force-push"},
		{"backtick git call", "echo `git commit --amend`", true, "amend"},
		{"bash -c single quote", `bash -c 'git push --force origin main'`, true, "force-push"},
		{"bash -c double quote", `bash -c "git commit --amend"`, true, "amend"},
		{"sh -c amend", `sh -c 'git commit --amend --no-edit'`, true, "amend"},
		{"sh -c add -A", `sh -c "git add -A"`, true, "explicit-path"},
		{"absolute bash -c", `/bin/bash -c 'git push -f'`, true, "force-push"},
		{"bash -lc cluster", `bash -lc 'git push --force'`, true, "force-push"},
		{"nested cmdsubst in bash -c", `bash -c 'echo $(git tag -f v1)'`, true, "tag-force"},
		{"bash -c nested bash -c", `bash -c "bash -c 'git push -f'"`, true, "force-push"},
		{"pipe inside bash -c", `bash -c 'echo x | git push --force'`, true, "force-push"},

		// ---- expansion stays out of scope (degrades to defer, NEVER allow) ----
		// $VAR / eval / alias need runtime state a static pass cannot have; the git
		// call they reconstruct is unrecoverable here and must DEFER (the git hooks
		// floor is the backstop). The key property: a defer, not a false-allow.
		{"var-expanded subcommand", "git $CMD --force", false, ""},
		{"var-expanded program", "$GIT push --force", false, ""},
		{"eval launder", `eval "git push --force"`, false, ""},
		{"alias then push", "alias g=git; g push -f", false, ""},
		// A subcommand laundered through a command-substitution RESULT (#823's
		// `git $(echo push) --force`) is the SAME undecidable class as $VAR: the verb is
		// whatever `echo push` prints at runtime, which no STATIC parse — this lexer OR a
		// real shell AST (mvdan/sh) — can resolve, so it must DEFER (opaque), never allow.
		{"cmdsubst-result subcommand", "git $(echo push) --force", false, ""},
		// ...but a REAL hazard paired with an unrecoverable one is still caught.
		{"var plus real hazard", "git $CMD; git push --force", true, "force-push"},

		// ---- unwrap is fail-safe on malformed input (defer, never crash/allow) -
		{"unbalanced cmdsubst still caught", "echo $(git push --force", true, "force-push"}, // outer string still tokenizes; the bare $( does not hide it
		{"empty bash -c", "bash -c ''", false, ""},
		{"bash -c no operand", "bash -c", false, ""},
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

// TestUnwrapShellSources pins the pure recovery layer directly: every yielded source
// always includes the original command, the recovered `$(...)`/backtick bodies and
// `bash -c` strings are present, and the pass is bounded + crash-free on adversarial input.
func TestUnwrapShellSources(t *testing.T) {
	contains := func(srcs []string, want string) bool {
		for _, s := range srcs {
			if s == want {
				return true
			}
		}
		return false
	}

	t.Run("always yields the original", func(t *testing.T) {
		got := unwrapShellSources("git status")
		if !contains(got, "git status") {
			t.Fatalf("unwrapShellSources must always include cmd itself; got %q", got)
		}
	})
	t.Run("recovers cmdsubst body", func(t *testing.T) {
		got := unwrapShellSources("echo $(git push -f)")
		if !contains(got, "git push -f") {
			t.Fatalf("expected the $() body recovered; got %q", got)
		}
	})
	t.Run("recovers backtick body", func(t *testing.T) {
		got := unwrapShellSources("x=`git commit --amend`")
		if !contains(got, "git commit --amend") {
			t.Fatalf("expected the backtick body recovered; got %q", got)
		}
	})
	t.Run("recovers bash -c string", func(t *testing.T) {
		got := unwrapShellSources(`bash -c 'git push --force'`)
		if !contains(got, "git push --force") {
			t.Fatalf("expected the bash -c string recovered; got %q", got)
		}
	})
	t.Run("does not extract from single quotes", func(t *testing.T) {
		// A $() literally inside single quotes is inert in the shell, so it must NOT
		// be lifted out as a live source (it stays only as part of the whole string).
		got := unwrapShellSources(`echo 'literal $(git push -f) text'`)
		if contains(got, "git push -f") {
			t.Fatalf("a $() inside single quotes is inert and must not be lifted; got %q", got)
		}
	})
	t.Run("bounded + crash-free on adversarial nesting", func(t *testing.T) {
		deep := strings.Repeat("$(", 50) + "git push -f" + strings.Repeat(")", 50)
		got := unwrapShellSources(deep) // must not panic or run away
		if len(got) > maxUnwrapSources {
			t.Fatalf("unwrap exceeded the source bound: %d > %d", len(got), maxUnwrapSources)
		}
	})
}

// cmdCall builds a shell tool call with the given command as inline JSON args
// (no resolver needed — refBytes reads RefInline directly), using json.Marshal so
// quotes in cmd survive into the args bytes.
func cmdCall(tool, key, cmd string) *abi.ToolCall {
	b, _ := json.Marshal(map[string]string{key: cmd})
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: b}}
}

func decisionRecorder(t *testing.T) (*witness.Recorder, *[]witness.Decision) {
	t.Helper()
	var captured []witness.Decision
	runner := func(_ context.Context, _ string, args ...string) (string, int, error) {
		for i, a := range args {
			if a != "-F" || i+1 >= len(args) {
				continue
			}
			body, err := os.ReadFile(args[i+1])
			if err != nil {
				return "", 1, err
			}
			var d witness.Decision
			if err := json.Unmarshal([]byte(strings.TrimSpace(string(body))), &d); err != nil {
				return "", 1, err
			}
			captured = append(captured, d)
		}
		return "", 0, nil
	}
	return witness.NewRecorderWithRunner(runner, ""), &captured
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

func TestAdjudicateRecordsRefusalWhenRecorderWired(t *testing.T) {
	g := New()
	rec, captured := decisionRecorder(t)
	g.SetRecorder(rec)

	v := g.Adjudicate(context.Background(), cmdCall("Bash", "command", "git push --force origin main"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("force push: Kind=%v, want Deny", v.Kind)
	}
	if len(*captured) != 1 {
		t.Fatalf("expected one recorded decision, got %d: %+v", len(*captured), *captured)
	}
	got := (*captured)[0]
	if got.Op != "gitgate" || got.Verdict != witness.VerdictRefuse || got.ReasonClass != "POLICY_BLOCK" {
		t.Fatalf("recorded decision = %+v, want gitgate/refuse/POLICY_BLOCK", got)
	}
	if strings.Join(got.RefusedArgv, " ") != "shell -c git push --force origin main" {
		t.Fatalf("RefusedArgv = %+v", got.RefusedArgv)
	}
}

func TestCollectiveCommitAllowsPairwiseDisjointLeases(t *testing.T) {
	plan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{ID: "kernel", Leases: []string{"internal/kernel/**"}, Paths: []string{"internal/kernel/kernel.go"}},
			{ID: "docs", Leases: []string{"docs/**"}, Paths: []string{"docs/cli-reference.md"}},
		},
		CommitPaths: []string{"internal/kernel/kernel.go", "docs/cli-reference.md"},
	}
	finding := CheckCollectiveCommit(plan)
	if !finding.OK {
		t.Fatalf("CheckCollectiveCommit valid plan=%+v, want OK", finding)
	}

	v := New().Adjudicate(context.Background(), collectiveCall(plan))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("collective Adjudicate Kind=%v, want Allow (verdict=%+v)", v.Kind, v)
	}
	if v.By != ToolCollectiveCommit {
		t.Fatalf("collective Adjudicate By=%q, want %q", v.By, ToolCollectiveCommit)
	}
}

func TestCollectiveCommitRefusesOverlappingLeases(t *testing.T) {
	plan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{ID: "wide", Leases: []string{"internal/**"}, Paths: []string{"internal/gitgate/gitgate.go"}},
			{ID: "narrow", Leases: []string{"internal/gitgate/**"}, Paths: []string{"internal/gitgate/gitgate_test.go"}},
		},
		CommitPaths: []string{"internal/gitgate/gitgate.go", "internal/gitgate/gitgate_test.go"},
	}
	finding := CheckCollectiveCommit(plan)
	assertCollectiveRefusal(t, finding, "lease conflict")
}

func TestCollectiveCommitRefusesPathOutsideLeasedTree(t *testing.T) {
	plan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{ID: "gitgate", Leases: []string{"internal/gitgate/**"}, Paths: []string{"internal/kernel/kernel.go"}},
		},
		CommitPaths: []string{"internal/kernel/kernel.go"},
	}
	finding := CheckCollectiveCommit(plan)
	assertCollectiveRefusal(t, finding, "outside leased tree")

	v := New().Adjudicate(context.Background(), collectiveCall(plan))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonLeaseHeld {
		t.Fatalf("collective outside-lease verdict=%+v, want Deny LEASE_HELD", v)
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok || !strings.Contains(wp.Claim, "outside leased tree") {
		t.Fatalf("collective outside-lease witness=%v, want outside leased tree claim", v.Payload)
	}
}

func TestCollectiveCommitRecordsRefusalWhenRecorderWired(t *testing.T) {
	g := New()
	rec, captured := decisionRecorder(t)
	g.SetRecorder(rec)
	plan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{ID: "gitgate", Leases: []string{"internal/gitgate/**"}, Paths: []string{"internal/kernel/kernel.go"}},
		},
		CommitPaths: []string{"internal/kernel/kernel.go"},
	}

	v := g.Adjudicate(context.Background(), collectiveCall(plan))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonLeaseHeld {
		t.Fatalf("collective outside-lease verdict=%+v, want Deny LEASE_HELD", v)
	}
	if len(*captured) != 1 {
		t.Fatalf("expected one recorded decision, got %d: %+v", len(*captured), *captured)
	}
	got := (*captured)[0]
	if got.Op != ToolCollectiveCommit || got.Verdict != witness.VerdictRefuse || got.ReasonClass != "LEASE_HELD" {
		t.Fatalf("recorded decision = %+v, want collective/refuse/LEASE_HELD", got)
	}
	if len(got.Tree) != 1 || got.Tree[0] != "internal/kernel/kernel.go" {
		t.Fatalf("Tree = %+v", got.Tree)
	}
}

func TestCollectiveCommitRefusesUnionViolation(t *testing.T) {
	plan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{ID: "gitgate", Leases: []string{"internal/gitgate/**"}, Paths: []string{"internal/gitgate/gitgate.go"}},
		},
		// The second path sits inside the held lease, so this is not a lease-tree
		// containment failure. It is outside the writer-declared path union.
		CommitPaths: []string{"internal/gitgate/gitgate.go", "internal/gitgate/gitgate_test.go"},
	}
	finding := CheckCollectiveCommit(plan)
	assertCollectiveRefusal(t, finding, "union violation")
}

func collectiveCall(plan CollectiveCommitPlan) *abi.ToolCall {
	b, _ := json.Marshal(plan)
	return &abi.ToolCall{Tool: ToolCollectiveCommit, Args: abi.Ref{Kind: abi.RefInline, Inline: b}}
}

func assertCollectiveRefusal(t *testing.T, finding CollectiveFinding, claimHas string) {
	t.Helper()
	if finding.OK {
		t.Fatalf("finding OK=true, want refusal containing %q", claimHas)
	}
	if finding.Reason != abi.ReasonLeaseHeld {
		t.Fatalf("finding reason=%s, want LEASE_HELD (finding=%+v)", abi.ReasonName(finding.Reason), finding)
	}
	if !strings.Contains(finding.Claim, claimHas) {
		t.Fatalf("finding claim=%q, want substring %q", finding.Claim, claimHas)
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
