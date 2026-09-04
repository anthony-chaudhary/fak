package adjudicator

import "testing"

// The reversibility classifier names a sanctioned sidestep for the SAFE push subset
// (ReversibilityEnvelope.RewriteCommand / RewriteTool) so a later opt-in rung can
// substitute it in flight. The rung itself is not wired yet; these tests pin the
// invariant that makes it safe to write when it is — the safe-subset test lives in the
// classifier, so a non-empty RewriteCommand is by construction never a dangerous push.

func TestSidestepRewriteOnSafeBarePush(t *testing.T) {
	// The safe subset is a CURRENT-BRANCH push: bare `git push`, or `git push <remote>`
	// with no branch. Also covered: the leading env-assignment / wrapper heads
	// commandSegments strips, which must not hide the `git push` head.
	for _, cmd := range []string{
		"git push",
		"git push origin",
		"  git   push   origin  ",
		"GIT_SSH_COMMAND=ssh git push",
		"env git push origin",
	} {
		env := ClassifyReversibility("Bash", map[string]any{"command": cmd})
		if env.RewriteCommand != gitPushSidestepCommand {
			t.Errorf("safe push %q: RewriteCommand = %q, want %q", cmd, env.RewriteCommand, gitPushSidestepCommand)
		}
		if env.RewriteTool != "Bash" {
			t.Errorf("safe push %q: RewriteTool = %q, want Bash", cmd, env.RewriteTool)
		}
	}
}

func TestSidestepRewriteAbsentOnDangerousPush(t *testing.T) {
	// Every one of these differs from `fak sync push` (non-force, current-branch,
	// fast-forward-only), so laundering it into that verb would silently change intent —
	// worse than holding. None may carry a rewrite target.
	for _, cmd := range []string{
		"git push --force origin main",
		"git push -f",
		"git push --force-with-lease origin main",
		"git push --delete origin feature",
		"git push -d origin feature",
		"git push --tags",
		"git push --all",
		"git push --mirror",
		"git push --prune origin",
		"git push origin local:remote", // an explicit refspec
		"git push -u origin main",      // sets upstream
		"git push origin main",         // names a specific branch, not the current one
		"git push --dry-run",           // already a preview; nothing to substitute
		"git add -A && git push",       // a sibling command rides along
		"git push; rm -rf /",           // ditto, via `;`
		`git push "origin&&evil"`,      // a sequencing byte inside a quoted payload
	} {
		env := ClassifyReversibility("Bash", map[string]any{"command": cmd})
		if env.RewriteCommand != "" {
			t.Errorf("dangerous push %q carried rewrite %q — the safe-subset gate leaked", cmd, env.RewriteCommand)
		}
		if env.RewriteTool != "" {
			t.Errorf("dangerous push %q carried rewrite tool %q", cmd, env.RewriteTool)
		}
	}
}

func TestSidestepRewriteOnSafeBarePull(t *testing.T) {
	for _, cmd := range []string{
		"git pull",
		"git pull origin",
		"git pull origin main",
		"  git   pull   origin   main  ",
		"GIT_SSH_COMMAND=ssh git pull",
		"env git pull origin",
	} {
		env := ClassifyReversibility("Bash", map[string]any{"command": cmd})
		if env.RewriteCommand != gitPullSidestepCommand {
			t.Errorf("safe pull %q: RewriteCommand = %q, want %q", cmd, env.RewriteCommand, gitPullSidestepCommand)
		}
		if env.RewriteTool != "Bash" {
			t.Errorf("safe pull %q: RewriteTool = %q, want Bash", cmd, env.RewriteTool)
		}
	}
}

func TestSidestepRewriteAbsentOnUnrelatedCalls(t *testing.T) {
	// A rewrite target is a git-push/git-pull affordance: no other command, and no
	// argument-less tool call, may pick one up.
	for _, cmd := range []string{
		"git status",
		"gh pr create --title x",
		"fak sync push",
		"pushd /tmp",
		"",
	} {
		env := ClassifyReversibility("Bash", map[string]any{"command": cmd})
		if env.RewriteCommand != "" {
			t.Errorf("command %q carried rewrite %q, want none", cmd, env.RewriteCommand)
		}
	}
	if env := ClassifyReversibility("Write", map[string]any{"file_path": "a.go"}); env.RewriteCommand != "" {
		t.Errorf("Write carried rewrite %q, want none", env.RewriteCommand)
	}
}

// TestSidestepRewriteMatchesRemedyTable pins the machine-readable rewrite target to the
// same string the human-facing remedy table advertises, so the substituted verb and the
// advisory hint can never drift apart. familyRemedyCommands entries are themselves proved
// reversible by TestFamilyRedirectCommandsAreReversible, which is what makes this the
// non-self-refuting remedy.
func TestSidestepRewriteMatchesRemedyTable(t *testing.T) {
	want := familyRemedyCommands["git-push"][0]
	if gitPushSidestepCommand != want {
		t.Fatalf("sidestep verb = %q, remedy table git-push[0] = %q — the rewrite drifted from the advertised remedy", gitPushSidestepCommand, want)
	}
	env := ClassifyReversibility("Bash", map[string]any{"command": "git push origin"})
	if env.RewriteCommand != want {
		t.Fatalf("classifier rewrite = %q, remedy table = %q", env.RewriteCommand, want)
	}
	// The hold is still in force: naming a sidestep does not itself release the call.
	if env.Class == ReversibilityReversible {
		t.Fatalf("bare push classified %q — a rewrite target must not relax the class", env.Class)
	}
	if env.ConfirmToken == "" {
		t.Fatal("bare push carries no confirm token — the preview-confirm hold was dropped")
	}

	wantPull := familyRemedyCommands["git-pull"][0]
	if gitPullSidestepCommand != wantPull {
		t.Fatalf("sidestep verb = %q, remedy table git-pull[0] = %q — the rewrite drifted from the advertised remedy", gitPullSidestepCommand, wantPull)
	}
	envPull := ClassifyReversibility("Bash", map[string]any{"command": "git pull origin main"})
	if envPull.RewriteCommand != wantPull {
		t.Fatalf("classifier rewrite = %q, remedy table = %q", envPull.RewriteCommand, wantPull)
	}
	if envPull.Class == ReversibilityReversible {
		t.Fatalf("bare pull classified %q — a rewrite target must not relax the class", envPull.Class)
	}
	if envPull.ConfirmToken == "" {
		t.Fatal("bare pull carries no confirm token — the preview-confirm hold was dropped")
	}
}
