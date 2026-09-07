package adjudicator

import (
	"context"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func gitPushAdj(tool, arg, re string) *Adjudicator {
	return New(Policy{
		Allow: map[string]bool{tool: true},
		ArgPredicates: []ArgPredicate{{
			Tool: tool, Arg: arg, Kind: ArgDenyRegex,
			Re:     regexp.MustCompile(re),
			Reason: abi.ReasonPolicyBlock,
			Fix:    "git push is restricted by capability floor; commits must be reviewed and pushed via safe sync",
		}},
	})
}

func TestGitPushDirectClassification(t *testing.T) {
	blocked := []string{
		"git push",
		"git push origin main",
		"git push origin HEAD:refs/heads/main",
		"git --no-pager push origin",
		"git -C /path push",
		"git -c foo=bar push",
		"git -cfoo=bar push",
		"git --git-dir=/repo/.git push",
		"git --git-dir /repo/.git push",
		"git --work-tree=/repo push",
		"git --work-tree /repo push",
		"git --namespace=foo push",
		"git --namespace foo push",
		"git --bare push",
		"git -p push",
		"git -P push",
		"git --no-replace-objects push",
		"git --no-lazy-fetch push",
		"git --literal-pathspecs push",
		"git -C /path -c foo=bar --no-pager push origin main",
		"git push --force",
		"git push -u origin feature",
		"git -- push",
		"FOO=1 git push",
		"sudo git push",
		"sh -c 'git push'",
		"git status && git push",
		"git push; echo done",
		"git push | cat",
		"echo done | git push",
	}

	for _, cmd := range blocked {
		if !commandExecutesGitPush(cmd) {
			t.Errorf("commandExecutesGitPush(%q) = false, want true (blocked)", cmd)
		}
	}

	permitted := []string{
		`git commit -m "fix git push"`,
		`git commit -m "git push is restricted"`,
		`git log --grep="git push"`,
		`echo "git push is restricted"`,
		`grep -rn 'git push' docs/`,
		`git stash push`,
		`git stash push -m "wip"`,
		`git status`,
		`git pull`,
		`git fetch`,
		`git help push`,
		"cat <<EOF\ngit push\nEOF",
		"cat <<'EOF'\ngit push\nEOF",
		"cat <<-EOF\n\tgit push\nEOF",
		"cat <<EOF > notes.txt\ngit push\nEOF",
	}

	for _, cmd := range permitted {
		if commandExecutesGitPush(cmd) {
			t.Errorf("commandExecutesGitPush(%q) = true, want false (permitted)", cmd)
		}
	}
}

func TestGitPushPowerShell(t *testing.T) {
	blocked := []string{
		"git push",
		"git.exe push",
		"git push origin main",
		"& git push",
		`& "git" push`,
		"git --no-pager push origin",
		`git -C C:\work push`,
		"git -c core.autocrlf=true push",
		`powershell -Command "git push"`,
		`pwsh -Command "git push"`,
		"git push --force-with-lease",
	}

	for _, cmd := range blocked {
		if !commandExecutesGitPush(cmd) {
			t.Errorf("PowerShell commandExecutesGitPush(%q) = false, want true (blocked)", cmd)
		}
	}

	permitted := []string{
		`git commit -m "fix git push"`,
		`git log --grep="git push"`,
		`echo "git push is restricted"`,
		`Write-Output "git push"`,
		`git stash push`,
		`git stash push -m "wip"`,
		`Select-String -Pattern "git push" -Path docs\*.md`,
	}

	for _, cmd := range permitted {
		if commandExecutesGitPush(cmd) {
			t.Errorf("PowerShell commandExecutesGitPush(%q) = true, want false (permitted)", cmd)
		}
	}
}

func TestGitPushAdjudicatorPolicy(t *testing.T) {
	tools := []struct {
		name string
		arg  string
		re   string
	}{
		{"Bash", "command", defaultGitPushDenyRegex},
		{"PowerShell", "command", defaultPSGitPushDenyRegex},
		{"shell_command", "command", defaultGitPushDenyRegex},
		{"functions.shell_command", "command", defaultGitPushDenyRegex},
		{"exec_command", "cmd", defaultGitPushDenyRegex},
	}

	for _, tc := range tools {
		t.Run(tc.name, func(t *testing.T) {
			a := gitPushAdj(tc.name, tc.arg, tc.re)

			// Blocked commands
			for _, cmd := range []string{
				"git push",
				"git push origin main",
				"git --no-pager push origin",
				"git -C /path push",
				"git -c foo=bar push",
			} {
				v := a.Adjudicate(context.Background(), inlineCall(tc.name, `{"`+tc.arg+`":"`+cmd+`"}`))
				if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
					t.Errorf("%s %q = %v/%s, want Deny/POLICY_BLOCK", tc.name, cmd, v.Kind, abi.ReasonName(v.Reason))
				}
				if v.Meta["fix"] == "" {
					t.Errorf("%s %q: expected non-empty fix meta", tc.name, cmd)
				}
			}

			// Permitted commands
			for _, cmd := range []string{
				`git commit -m "fix git push"`,
				`git log --grep="git push"`,
				`echo "git push is restricted"`,
				`grep -rn "git push" docs/`,
				`git stash push`,
			} {
				v := a.Adjudicate(context.Background(), inlineCall(tc.name, `{"`+tc.arg+`":"`+cmd+`"}`))
				if v.Kind != abi.VerdictAllow {
					t.Errorf("%s %q = %v/%s, want Allow", tc.name, cmd, v.Kind, abi.ReasonName(v.Reason))
				}
			}
		})
	}
}

func TestGitPushSelfRefutingRemedyIsAdmitted(t *testing.T) {
	remedy := "git push is restricted by capability floor; commits must be reviewed and pushed via safe sync"
	testCmds := []string{
		`echo "` + remedy + `"`,
		`git commit -m "` + remedy + `"`,
		`grep -rn "git push is restricted" docs/`,
	}

	for _, tool := range []struct {
		name string
		arg  string
		re   string
	}{
		{"Bash", "command", defaultGitPushDenyRegex},
		{"PowerShell", "command", defaultPSGitPushDenyRegex},
	} {
		a := gitPushAdj(tool.name, tool.arg, tool.re)
		for _, cmd := range testCmds {
			v := a.Adjudicate(context.Background(), inlineCall(tool.name, `{"`+tool.arg+`":"`+cmd+`"}`))
			if v.Kind != abi.VerdictAllow {
				t.Errorf("%s quoting remedy %q: got %v/%s, want Allow", tool.name, cmd, v.Kind, abi.ReasonName(v.Reason))
			}
		}
	}
}
