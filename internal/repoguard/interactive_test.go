// interactive_test.go — the case table for the INTERACTIVE_HANG rung (#2080):
// every curated would-hang invocation is refused with a runnable fix, and the
// benign non-interactive forms pass unchanged. Hermetic: no filesystem, no spawn.
package repoguard

import (
	"strings"
	"testing"
)

func TestInteractiveHangDenied(t *testing.T) {
	for _, c := range []string{
		// interactive git
		"git rebase -i HEAD~3",
		"git rebase --interactive main",
		"git -C /repo rebase -i HEAD~2",
		"git add -i",
		"git add --interactive",
		"git add -p internal/",
		"git add -up",
		"git checkout -p main -- file.go",
		"git reset --patch",
		"git restore -p file.go",
		"git stash -p",
		// editor-opening commit forms
		"git commit",
		"git commit --amend",
		"git commit -a",
		"git commit -s --amend",
		"git commit -m fix -e", // -e forces the editor even with -m
		"git commit -c HEAD",   // reedit-message opens the editor
		// editors
		"vim internal/repoguard/guard.go",
		"vi notes.txt",
		"nano config.toml",
		"emacs main.go",
		"nvim -d a.go b.go",
		// credential prompts / sudoers / crontab editors
		"gh auth login",
		"gh auth login --web",
		"visudo",
		"crontab -e",
		// compound: the interactive segment is caught inside a chain
		"make ci && git commit",
	} {
		v := classifyInteractive(c)
		if len(v) == 0 {
			t.Errorf("%q: expected INTERACTIVE_HANG, got allow", c)
			continue
		}
		if v[0].Reason != ReasonInteractiveHang {
			t.Errorf("%q: reason = %q, want %s", c, v[0].Reason, ReasonInteractiveHang)
		}
		if strings.TrimSpace(v[0].Fix) == "" {
			t.Errorf("%q: finding has no runnable fix", c)
		}
	}
}

// TestInteractiveFixesAreNotSelfRefuting pins that the runnable command an
// INTERACTIVE_HANG fix recommends is itself allowed — following the guard's own
// advice must clear the rung, not re-trip it (the self-refuting-remedy class of
// docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md). Two shipped instances this
// locks: `visudo` recommended `visudo -cf <file>` while the recognizer only
// accepted the un-clustered `-c`; `git commit -p` recommended `git commit --
// <paths>`, which itself re-opens the commit editor and re-refuses.
func TestInteractiveFixesAreNotSelfRefuting(t *testing.T) {
	cases := []struct {
		interactive     string // must be denied
		fixContains     string // the recommended command shape
		instantiatedFix string // that recommendation, placeholders filled — must be allowed
	}{
		{"visudo", "visudo -cf", "visudo -cf /etc/sudoers.d/mycfg"},
		{"git commit -p", "git commit -s -m", `git commit -s -m "fix(x): y" -- cmd/fak/main.go`},
	}
	for _, tc := range cases {
		v := classifyInteractive(tc.interactive)
		if len(v) == 0 {
			t.Errorf("%q: expected INTERACTIVE_HANG denial", tc.interactive)
			continue
		}
		if !strings.Contains(v[0].Fix, tc.fixContains) {
			t.Errorf("%q: fix %q does not recommend %q", tc.interactive, v[0].Fix, tc.fixContains)
		}
		if rv := classifyInteractive(tc.instantiatedFix); len(rv) != 0 {
			t.Errorf("self-refuting remedy: %q recommends %q, but that instantiated command is itself denied: %v", tc.interactive, tc.instantiatedFix, rv)
		}
	}
}

func TestNonInteractiveFormsAllowed(t *testing.T) {
	for _, c := range []string{
		// message-carrying commits
		`git commit -s -m "fix(x): y" -- cmd/fak/main.go`,
		`git commit -sm "fix(x): y"`,
		`git commit --message="fix(x): y"`,
		"git commit -F msg.txt",
		"git commit --amend --no-edit",
		"git commit -C HEAD",
		"git commit --fixup=abc123",
		"git commit --dry-run",
		"git commit --help",
		// plain staging / rebase / restore
		"git add -- internal/repoguard/interactive.go",
		"git add -A",
		"git rebase --continue",
		"git rebase main",
		"git restore file.go",
		"git reset -- file.go",
		"git checkout -b feature",
		"git stash pop",
		// a pathspec after -- is never a flag
		"git add -- -p-weird-name",
		// explicit editor override signals scripted intent
		"GIT_SEQUENCE_EDITOR=: git rebase -i HEAD~3",
		"env GIT_EDITOR=true git commit",
		// scripted editor modes
		"emacs --batch -l script.el",
		"vim -es +wq file.txt",
		// non-login gh, token login
		"gh pr list",
		"gh auth login --with-token < token.txt",
		"gh auth status",
		// validate-only visudo (incl. the CLUSTERED -cf the fix recommends), non-edit crontab
		"visudo -c",
		"visudo -cf /etc/sudoers.d/mycfg",
		"crontab -l",
		"crontab schedule.txt",
		// pagers self-disable without a TTY — never refused
		"git log --oneline | less",
		"less README.md",
		// everyday commands
		"go test ./...",
		"git status",
		"git diff --stat",
	} {
		if v := classifyInteractive(c); len(v) != 0 {
			t.Errorf("%q: expected allow, got %v", c, v)
		}
	}
}

func TestInteractiveHangViaEvaluate(t *testing.T) {
	// The hook path: Evaluate must surface the rung for Bash tool calls.
	v := eval("Bash", cmd("git rebase -i HEAD~3"))
	if len(v) != 1 || v[0].Reason != ReasonInteractiveHang {
		t.Fatalf("Evaluate(Bash, rebase -i) = %v, want one INTERACTIVE_HANG", v)
	}
	// The rebase fix is fully runnable: the original invocation behind a no-op
	// sequence editor.
	if want := "GIT_SEQUENCE_EDITOR=: git rebase -i HEAD~3"; v[0].Fix != want {
		t.Errorf("fix = %q, want %q", v[0].Fix, want)
	}
}

func TestRenderReasonInteractiveBlock(t *testing.T) {
	v := classifyInteractive("git commit")
	if len(v) != 1 {
		t.Fatalf("classifyInteractive(git commit) = %v, want one finding", v)
	}
	reason := renderReason(v)
	if !strings.HasPrefix(reason, ReasonInteractiveHang+":") {
		t.Errorf("reason %q must lead with the structured token", reason)
	}
	if !strings.Contains(reason, v[0].Fix) {
		t.Errorf("reason %q must carry the runnable fix %q", reason, v[0].Fix)
	}
}

func TestRenderReasonMixedBlocks(t *testing.T) {
	// An out-of-tree write AND an interactive segment in one compound command:
	// both tokens must appear, each in its own block.
	v := eval("Bash", cmd("echo x > ../tools/y && git commit"))
	reason := renderReason(v)
	for _, tok := range []string{guardReason, ReasonInteractiveHang} {
		if !strings.Contains(reason, tok) {
			t.Errorf("mixed reason %q missing token %s", reason, tok)
		}
	}
}
