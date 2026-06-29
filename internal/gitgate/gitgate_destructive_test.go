package gitgate

import (
	"strings"
	"testing"
)

// TestClassifyDestructiveAndOffTrunk pins the destructive-working-tree and
// branch/worktree-open hazards added to the table: `reset --hard`, `clean -f`,
// whole-tree `checkout .` / `restore .`, and `checkout -b` / `switch -c` /
// `worktree add`. The defer rows carry the weight — they prove the rung does NOT
// false-positive on the SAFE neighbors (a soft reset, a clean dry-run, a
// specific-path revert, a switch/checkout to an existing branch, a worktree list).
func TestClassifyDestructiveAndOffTrunk(t *testing.T) {
	g := New()
	cases := []struct {
		name   string
		cmd    string
		deny   bool
		lawHas string
	}{
		// ---- reset --hard ---------------------------------------------------
		{"reset hard", "git reset --hard", true, "reset-hard"},
		{"reset hard to ref", "git reset --hard origin/main", true, "reset-hard"},
		{"reset soft OK", "git reset --soft HEAD~1", false, ""},
		{"reset mixed OK", "git reset HEAD~1", false, ""},
		{"reset unstage OK", "git reset -- internal/foo.go", false, ""},

		// ---- clean -f -------------------------------------------------------
		{"clean force short", "git clean -f", true, "clean-force"},
		{"clean force dirs", "git clean -fd", true, "clean-force"},
		{"clean force long", "git clean --force", true, "clean-force"},
		{"clean force x", "git clean -fdx", true, "clean-force"},
		{"clean dry-run OK", "git clean -n", false, ""},
		{"clean interactive OK", "git clean -i", false, ""},

		// ---- whole-tree discard --------------------------------------------
		{"checkout dot", "git checkout .", true, "whole-tree-discard"},
		{"checkout dashdash dot", "git checkout -- .", true, "whole-tree-discard"},
		{"restore dot", "git restore .", true, "whole-tree-discard"},
		{"restore staged dot", "git restore --staged .", true, "whole-tree-discard"},
		{"checkout file OK", "git checkout -- internal/foo.go", false, ""},
		{"restore file OK", "git restore -- internal/foo.go", false, ""},
		{"checkout branch OK", "git checkout main", false, ""},

		// ---- off-trunk: branch / worktree open ------------------------------
		{"checkout -b", "git checkout -b feature", true, "off-trunk"},
		{"checkout -B", "git checkout -B feature", true, "off-trunk"},
		{"switch -c", "git switch -c feature", true, "off-trunk"},
		{"switch -C", "git switch -C feature", true, "off-trunk"},
		{"switch --create", "git switch --create feature", true, "off-trunk"},
		{"switch --create=", "git switch --create=feature", true, "off-trunk"},
		{"switch --force-create", "git switch --force-create feature", true, "off-trunk"},
		{"worktree add", "git worktree add ../wt", true, "off-trunk"},
		{"switch existing OK", "git switch main", false, ""},
		{"worktree list OK", "git worktree list", false, ""},
		{"worktree remove OK", "git worktree remove ../wt", false, ""},

		// ---- catastrophic remote / history rewrite -------------------------
		{"push mirror", "git push --mirror origin", true, "push-mirror"},
		{"filter-branch", "git filter-branch --tree-filter rm HEAD", true, "history-rewrite"},
		{"filter-repo", "git filter-repo --path secret --invert-paths", true, "history-rewrite"},
		{"clone mirror OK", "git clone --mirror https://example.com/x.git", false, ""},

		// ---- persistent hook-disable via config -----------------------------
		{"config hooksPath set", "git config core.hooksPath /dev/null", true, "skip-hooks"},
		{"config hooksPath global", "git config --global core.hooksPath /tmp/h", true, "skip-hooks"},
		{"config hooksPath case", "git config core.hookspath .nohooks", true, "skip-hooks"},
		{"config hooksPath get OK", "git config --get core.hooksPath", false, ""},
		{"config hooksPath unset OK", "git config --unset core.hooksPath", false, ""},
		{"config unrelated OK", "git config user.name Alice", false, ""},

		// ---- laundering: the unwrap pass must still see these ---------------
		{"reset hard via bash -c", `bash -c "git reset --hard"`, true, "reset-hard"},
		{"clean via pipe", "echo go | git clean -fd", true, "clean-force"},
		{"checkout -b via subst", "x=$(git checkout -b feature)", true, "off-trunk"},

		// ---- the MESSAGE-mentions-a-flag false-positive guard ---------------
		{"commit msg mentions reset --hard", `git commit -m "do not run git reset --hard here"`, false, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			law, denied := g.Classify(c.cmd)
			if denied != c.deny {
				t.Fatalf("Classify(%q) deny=%v, want %v (law=%q)", c.cmd, denied, c.deny, law)
			}
			if c.deny && !strings.Contains(law, c.lawHas) {
				t.Fatalf("Classify(%q) law=%q, want it to contain %q", c.cmd, law, c.lawHas)
			}
		})
	}
}
