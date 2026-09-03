package adjudicator

import "testing"

// familyRemedyCommands maps each escalated reversibility family to the concrete,
// self-runnable shell command(s) its redirect hint recommends. The invariant a
// test below pins: a command the guard RECOMMENDS as the safe alternative must
// itself classify reversible — otherwise following the guard's own advice
// re-trips the gate and the agent loops (the self-refuting-remedy class of the
// confirm-gate deadlock, docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md). Two
// live instances of exactly this shipped: fs-destroy recommended "Remove-Item
// -WhatIf" and fs-destroy/git-destroy recommended "git clean -nd", both of which
// the classifier itself gated until hasDryRunPreview learned -WhatIf / git clean
// -n. Families whose hint names no runnable command (a prose "review the
// recipient", "use a transaction") map to an empty slice.
var familyRemedyCommands = map[string][]string{
	"issue-create-tool": {"fak issue create --title x --body-file /tmp/b"},
	"slack":             {"fak slack send -c ops hi"},
	"git-push":          {"fak sync push", "git push --dry-run"},
	"gh-write":          {}, // "read instead (gh api GET), or review the repo/endpoint/method" — prose hint, no single runnable command
	"npm-publish":       {"npm publish --dry-run"},
	"mail":              {}, // "review recipient/body in a draft" — no runnable command
	"webhook":           {}, // "send to a test endpoint" — no single command
	"registry-publish":  {"cargo publish --dry-run", "twine check dist/*"},
	"http-write":        {}, // "use a read-only request" — no single command
	"messaging-tool":    {"fak slack send -c ops hi"},
	"pr-create-tool":    {}, // "the host's draft or dry-run path" — host-specific
	"fs-destroy":        {"Remove-Item -WhatIf ./build", "git clean -nd"},
	"git-destroy":       {"git status", "git clean -nd"},
	"infra-destroy":     {"terraform plan -destroy", "kubectl diff -f x.yaml", "kubectl delete --dry-run=server -f x.yaml"},
	"sql-drop":          {}, // "a transaction that rolls back" — no single command
	"sql-truncate":      {}, // "a transaction with rollback" — no single command
	"sql-alter":         {}, // "verify schema compatibility or capture a snapshot" — no single command
	"db-migration-down": {}, // "capture a database snapshot" — no single command
	"dd-device-write":   {}, // "write to an image file first" — no single command
	"destructive-tool":  {}, // "the tool's preview or dry-run mode" — tool-specific
}

// TestFamilyRedirectCommandsAreReversible is the systematic guard against the
// self-refuting-remedy class: every concrete command a reversibility redirect
// recommends must itself classify reversible, so an agent that follows the
// guard's own advice clears the gate instead of re-tripping it.
func TestFamilyRedirectCommandsAreReversible(t *testing.T) {
	for family, cmds := range familyRemedyCommands {
		for _, c := range cmds {
			env := ClassifyReversibility("Bash", map[string]any{"command": c})
			if env.Class != ReversibilityReversible {
				t.Errorf("family %q recommends %q, but it classifies %q — the guard gates its own advice (self-refuting remedy)", family, c, env.Class)
			}
		}
	}
}

// TestFamilyRemedyTableIsComplete forces a NEW escalated family to declare its
// recommended command(s) here (even an empty slice for a prose-only hint), so
// the redirect-is-reversible invariant can never silently skip a family.
func TestFamilyRemedyTableIsComplete(t *testing.T) {
	for _, f := range reversibilityFamilies {
		if _, ok := familyRemedyCommands[f.name]; !ok {
			t.Errorf("reversibility family %q has no familyRemedyCommands entry — add its recommended command(s), or an empty slice for a prose-only hint, so the redirect-is-reversible invariant covers it", f.name)
		}
	}
}

// TestDryRunFlagsDoNotWeakenDestructiveFloor pins the negative half of the
// -WhatIf / git-clean-`-n` recognition: teaching the classifier those preview
// spellings must NOT let a genuinely destructive command through. In particular
// `-n` is a git-clean dry-run ONLY inside a `git clean`; elsewhere it means
// unrelated things (`kubectl delete -n <ns>` is a REAL delete) and must stay
// gated.
func TestDryRunFlagsDoNotWeakenDestructiveFloor(t *testing.T) {
	gated := []string{
		"git clean -fd",                   // force + dirs: real delete, no 'n'
		"git clean -f",                    // force: real delete
		"Remove-Item -Recurse -Force ./x", // no -WhatIf: real delete
		"rm -rf build",                    // no preview at all
		"kubectl delete -n prod pod/x",    // -n is a NAMESPACE here, a real delete
		"git push -f origin main",         // force push: real mutation, no 'n'
		"git push --force origin main",    // force push, long form: no 'n'
		"git push -d origin feature",      // remote-branch DELETE: -d has no 'n'
		"git push --delete origin feat",   // remote-branch delete, long form
	}
	for _, c := range gated {
		env := ClassifyReversibility("Bash", map[string]any{"command": c})
		if env.Class == ReversibilityReversible {
			t.Errorf("destructive floor weakened: %q classified reversible", c)
		}
	}
}

// TestGitPushShortDryRunIsPreview pins the positive half of extending the
// git-clean-`-n` recognition to `git push`: `git push -n` is the documented short
// spelling of `git push --dry-run` (already reversible via the --dry-run
// substring), so the short form — including 'n'-bearing clusters like -nf, and
// behind an env/wrapper head — must be reversible too. Without this the guard
// gates the short form of the very preview its own git-push redirect recommends.
func TestGitPushShortDryRunIsPreview(t *testing.T) {
	previews := []string{
		"git push -n origin main",                       // the bare short dry-run
		"git push -n",                                   // no explicit remote/branch
		"git push -nf origin main",                      // dry-run + force cluster: still sends nothing
		"git push -fn origin main",                      // cluster order does not matter
		"env GIT_SSH=x git push -n",                     // env-prefixed head is stripped like commandSegments
		"git commit -m done && git push -n origin main", // dry-run in the second segment
	}
	for _, c := range previews {
		env := ClassifyReversibility("Bash", map[string]any{"command": c})
		if env.Class != ReversibilityReversible {
			t.Errorf("git push short dry-run gated: %q classified %q, want reversible", c, env.Class)
		}
	}
}
