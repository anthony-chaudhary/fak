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
func TestOffTrunkRefusalNamesSanctionedDetachedWorkerRecovery(t *testing.T) {
	got, ok := New().Classify(`git worktree add ..\worker`)
	if !ok {
		t.Fatal("raw git worktree add must remain refused")
	}
	for _, want := range []string{
		"fak worktree worker prepare --id <worker-id> --scope <path>",
		"fak worktree worker land --id <worker-id>",
		"fak worktree worker reap --id <worker-id>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("refusal must preserve the goal with sanctioned recovery %q; got %q", want, got)
		}
	}
	if strings.Contains(got, "never branch or spin a worktree") {
		t.Fatalf("stale terminal recovery contradicts the sanctioned detached-worker route: %q", got)
	}
}
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
		{"push prune", "git push --prune origin refs/fak/locks/*", true, "push-prune"},
		{"push prune flag last", "git push origin refs/heads/* --prune", true, "push-prune"},
		{"filter-branch", "git filter-branch --tree-filter rm HEAD", true, "history-rewrite"},
		{"filter-repo", "git filter-repo --path secret --invert-paths", true, "history-rewrite"},
		{"clone mirror OK", "git clone --mirror https://example.com/x.git", false, ""},
		// The SAFE neighbors carry the weight: --prune is refused ONLY on push. A
		// `git fetch --prune` prunes local remote-tracking refs (standard hygiene) and
		// `git remote prune` likewise touches nothing on the remote — both must defer.
		{"fetch prune OK", "git fetch --prune origin", false, ""},
		{"fetch prune all OK", "git fetch --all --prune", false, ""},
		{"remote prune OK", "git remote prune origin", false, ""},

		// ---- the CONFIG spelling of --mirror (#5360) ------------------------
		// `remote.<name>.mirror=true` makes a PLAIN `git push` behave as --mirror,
		// so refusing only the flag left the same mass delete reachable with no
		// hazardous flag on the argv.
		{"mirror config per-invocation", "git -c remote.origin.mirror=true push origin", true, "push-mirror"},
		{"mirror config no value is true", "git -c remote.origin.mirror push origin", true, "push-mirror"},
		{"mirror config persistent", "git config remote.origin.mirror true", true, "push-mirror"},
		{"mirror config joined", "git config remote.origin.mirror=yes", true, "push-mirror"},
		{"mirror config dotted remote name", "git config remote.my.fleet.mirror on", true, "push-mirror"},
		// The SAFE neighbors: turning it OFF, reading it, unsetting it, and any other
		// remote.* key must all stay deferred — the rung refuses an ENABLE, not the key.
		{"mirror config false OK", "git config remote.origin.mirror false", false, ""},
		{"mirror config -c false OK", "git -c remote.origin.mirror=false push origin", false, ""},
		{"mirror config read OK", "git config --get remote.origin.mirror", false, ""},
		{"mirror config unset OK", "git config --unset remote.origin.mirror", false, ""},
		{"other remote key OK", "git config remote.origin.pushurl https://example.com/x.git", false, ""},
		{"remote fetch refspec OK", "git config remote.origin.fetch +refs/heads/*:refs/remotes/origin/*", false, ""},

		// ---- unattributable push: refspecs arriving on a pipe (#5360) -------
		{"xargs push", "git for-each-ref --format=':%(refname)' refs/fak/wip | xargs git push origin", true, "unattributable-push"},
		{"xargs push with flags", "git for-each-ref | xargs -n 200 -P 4 git push origin", true, "unattributable-push"},
		{"xargs push delete", "git for-each-ref | xargs git push origin --delete", true, "unattributable-push"},
		{"xargs push absolute git", "git for-each-ref | xargs /usr/bin/git push origin", true, "unattributable-push"},
		// Seeing THROUGH xargs also makes the existing rules apply to it...
		{"xargs add -A", "git status | xargs git add -A", true, "explicit-path"},
		// ...but it is not "refuse anything with xargs in it": a non-push git call
		// behind xargs is judged by the same table and keeps deferring.
		{"xargs status OK", "git for-each-ref | xargs git status", false, ""},
		{"xargs add explicit paths OK", "git diff --name-only | xargs git add", false, ""},
		{"xargs non-git program OK", "git for-each-ref | xargs echo", false, ""},

		// ---- mass vs ordinary remote-ref delete (#5360) ---------------------
		// A converge spelled one refspec at a time is the same act as --prune.
		{"mass delete refspecs", "git push origin :a :b :c :d :e :f :g :h", true, "mass-remote-ref-delete"},
		{"mass delete via --delete", "git push origin --delete a b c d e f g h", true, "mass-remote-ref-delete"},
		// Retiring ONE ref is a different act and keeps the singular law.
		{"single delete refspec", "git push origin :refs/fak/locks/session-abc", true, "remote-ref delete refused"},
		{"few deletes stay singular", "git push origin --delete a b c", true, "remote-ref delete refused"},
		// The counter counts DELETIONS, not operands: a many-branch ordinary push
		// is not a mass delete and must still defer.
		{"many-branch push OK", "git push origin a b c d e f g h i j", false, ""},

		// ---- persistent hook-disable via config -----------------------------
		{"config hooksPath set", "git config core.hooksPath /dev/null", true, "skip-hooks"},
		{"config hooksPath global", "git config --global core.hooksPath /tmp/h", true, "skip-hooks"},
		{"config hooksPath case", "git config core.hookspath .nohooks", true, "skip-hooks"},
		{"config hooksPath get OK", "git config --get core.hooksPath", false, ""},
		{"config hooksPath unset OK", "git config --unset core.hooksPath", false, ""},
		{"config unrelated OK", "git config user.name Alice", false, ""},
		// A different key whose VALUE merely mentions the string is not a hooks
		// disable — the substring-match false positives the key-scoping closes.
		{"config alias value mentions hooksPath OK", `git config alias.st "status; note core.hooksPath"`, false, ""},
		{"config editor value mentions hooksPath OK", `git config core.editor "vim ; echo core.hooksPath"`, false, ""},
		{"config safe.dir path contains hooksPath OK", "git config --add safe.directory /repo/core.hooksPathBackup", false, ""},

		// ---- persistent signing-disable via config --------------------------
		{"config gpgsign false", "git config commit.gpgsign false", true, "skip-signing"},
		{"config gpgsign global off", "git config --global commit.gpgsign off", true, "skip-signing"},
		{"config gpgsign joined", "git config commit.gpgsign=false", true, "skip-signing"},
		{"config gpgsign zero", "git config commit.gpgsign 0", true, "skip-signing"},
		{"config gpgsign true OK", "git config commit.gpgsign true", false, ""},
		{"config gpgsign get OK", "git config --get commit.gpgsign", false, ""},
		{"config gpgsign unset OK", "git config --unset commit.gpgsign", false, ""},

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

// TestUnattributablePushFailsClosed pins the fail-closed half of #5360: the
// 2026-07-22 converge deleted ~8400 refs, and 8400 refspecs do not fit on a command
// line — a deletion set that size reaches git only through a generator pipe, where the
// argv this rung reads contains NONE of the refs being destroyed. The rung therefore
// cannot show the act is bounded, and on a mass-delete-capable path "cannot show it is
// bounded" must resolve to REFUSE. The refusal must also name the legitimate route, or
// it is indistinguishable from having no route.
//
// Mutant that reds this: delete the `if viaXargs { return xargsPushLaw, true }` branch
// in inspectGitArgs (the push refspecs are then invisible and the call defers), or drop
// the gitArgvViaXargs arm from classify (git behind xargs stops being seen at all).
func TestUnattributablePushFailsClosed(t *testing.T) {
	g := New()
	const converge = "git for-each-ref --format=':%(refname)' refs/fak/wip | xargs -n 200 git push origin"
	law, denied := g.Classify(converge)
	if !denied {
		t.Fatalf("a push fed by a refspec pipe must fail CLOSED; Classify(%q) deferred", converge)
	}
	for _, want := range []string{
		"unattributable-push refused", // the structured reason
		"fails CLOSED",                // the reasoning, cited back to the agent
		"one explicit refspec per",    // remedy for the narrow case
		"fak sync push",               // remedy for the bulk case
		"PROVEN expired",              // ...and its precondition
	} {
		if !strings.Contains(law, want) {
			t.Fatalf("refusal must name a real route; law %q lacks %q", law, want)
		}
	}
	// Not "refuse everything with a pipe in it": a NON-push git call behind xargs is
	// judged by the ordinary table, so a harmless one still defers.
	if law, denied := g.Classify("git for-each-ref | xargs git status"); denied {
		t.Fatalf("xargs see-through must not refuse a harmless non-push git call; got deny %q", law)
	}
	// And the see-through is load-bearing for the EXISTING rules, not just the new
	// one: a spelling the table already refuses cannot be laundered through xargs.
	if _, denied := g.Classify("git status | xargs git add -A"); !denied {
		t.Fatal("`xargs git add -A` must be refused exactly like the direct spelling")
	}
}

// TestMassRemoteRefDeleteIsDistinctFromOneRefRetirement pins the scale distinction
// #5360 asks for: wiping a namespace and retiring one stale lease are not the same act
// and must not cite the same law. Deleting 8400 refs one refspec at a time is a
// --prune converge typed longhand, and its remedy is the sanctioned compiled-verb
// drainer, not "do not delete a remote branch". Both sides still DENY — the threshold
// only selects which law and which remedy is cited.
//
// Mutants that red this: (a) set massRemoteRefDeleteThreshold = 1, and the one-ref
// retirement wrongly gets the mass law (the "singular" assertion fires); (b) delete the
// countRemoteRefDeletes branch in inspectGitArgs, and the 8-ref converge falls back to
// the singular law (the "mass" assertion fires); (c) make countRemoteRefDeletes count
// every operand instead of deletions, and the ordinary many-branch push in the table
// above flips from defer to deny.
func TestMassRemoteRefDeleteIsDistinctFromOneRefRetirement(t *testing.T) {
	g := New()
	mass, denied := g.Classify("git push origin :a :b :c :d :e :f :g :h")
	if !denied {
		t.Fatal("a push naming 8 remote refs for deletion must be refused")
	}
	for _, want := range []string{
		"mass-remote-ref-delete refused", // the structured reason
		"names 8 remote refs",            // the measured scale, not a vague adjective
		"threshold 8",                    // the chosen bound, disclosed
		"fak sync push",                  // the sanctioned bulk route
		"FLEET_ALLOW_REF_PRUNE=1",        // ...and what it must set to get through
	} {
		if !strings.Contains(mass, want) {
			t.Fatalf("mass-delete refusal must name scale and route; law %q lacks %q", mass, want)
		}
	}
	// Retiring ONE ref is the ordinary act: still refused, but by the singular law, so
	// the scale distinction is real rather than cosmetic.
	single, denied := g.Classify("git push origin :refs/fak/locks/session-abc")
	if !denied {
		t.Fatal("a single remote-ref delete must stay refused (no existing refusal weakened)")
	}
	if strings.Contains(single, "mass-remote-ref-delete") {
		t.Fatalf("one-ref retirement must not cite the mass law: %q", single)
	}
	// The threshold is a boundary, not a mood: one below it is singular, one at it is mass.
	below := "git push origin" + strings.Repeat(" :r", massRemoteRefDeleteThreshold-1)
	if law, _ := g.Classify(below); strings.Contains(law, "mass-remote-ref-delete") {
		t.Fatalf("%d deletions is below the threshold %d and must cite the singular law; got %q",
			massRemoteRefDeleteThreshold-1, massRemoteRefDeleteThreshold, law)
	}
	at := "git push origin" + strings.Repeat(" :r", massRemoteRefDeleteThreshold)
	if law, _ := g.Classify(at); !strings.Contains(law, "mass-remote-ref-delete") {
		t.Fatalf("%d deletions is at the threshold and must cite the mass law; got %q",
			massRemoteRefDeleteThreshold, law)
	}
}

// TestMirrorConfigIsTheUnflaggedSpellingOfPushMirror pins the config half of #5360.
// `git push --mirror` was already refused, but git-config(1) says a
// `remote.<name>.mirror=true` clone pushes AS IF --mirror were given — so the flag
// refusal alone was a half-closed door: the identical mass delete rides in on a plain
// `git push origin` with no hazardous flag anywhere on the argv.
//
// Mutants that red this: delete the isMirrorEnable check in the `-c` global-option arm
// (the per-invocation rows defer), or the mirrorOn check in the `config` arm (the
// persistent row defers). Flipping isMirrorEnable to require an explicit `=true` reds
// the no-value row, which git itself reads as true.
func TestMirrorConfigIsTheUnflaggedSpellingOfPushMirror(t *testing.T) {
	g := New()
	for _, cmd := range []string{
		"git -c remote.origin.mirror=true push origin",
		"git -c remote.origin.mirror push origin", // git reads a valueless -c key as true
		"git config remote.origin.mirror true",
		"git config --global remote.origin.mirror 1",
	} {
		law, denied := g.Classify(cmd)
		if !denied {
			t.Fatalf("Classify(%q): the config spelling of --mirror must be refused", cmd)
		}
		if !strings.Contains(law, "push-mirror refused") {
			t.Fatalf("Classify(%q): want the push-mirror law, got %q", cmd, law)
		}
		if !strings.Contains(law, "git fetch --prune") {
			t.Fatalf("Classify(%q): refusal must name the safe local-tracking cleanup; got %q", cmd, law)
		}
	}
	// Disarming it, reading it, and unsetting it are the REPAIRS — refusing those would
	// trap a clone that already has the bit set (the self-refuting-remedy loop).
	for _, cmd := range []string{
		"git config remote.origin.mirror false",
		"git config --unset remote.origin.mirror",
		"git config --get remote.origin.mirror",
	} {
		if law, denied := g.Classify(cmd); denied {
			t.Fatalf("Classify(%q) must defer (it is the repair, not the hazard); got %q", cmd, law)
		}
	}
}
