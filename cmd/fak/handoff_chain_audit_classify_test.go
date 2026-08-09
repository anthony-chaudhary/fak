// handoff_chain_audit_classify_test.go — the honesty proof for the #5519 retry.
//
// TestHandoffToWitnessedCloseChain and TestRelayHandoffRotateClose retry
// `dos commit-audit` a bounded number of times, because dos folds a timed-out git
// read into "the commit is EMPTY (touched no files)" and exits non-zero
// (dos/vcs.py `_GIT_TIMEOUT_S` = 10s -> dos/commit_audit.py `read_commit` treats a
// None diffstat as a zero-file diff). A retry that swallowed ANY failure would turn
// those smokes into tests that cannot fail, which is how this pair stayed invisible
// for so long. The retry decision therefore lives in one PURE function,
// classifyDosAudit, and this file pins its contract: only a could-not-read result is
// retryable, and a real disagreement about the diff reds on the first attempt.
package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestClassifyDosAuditRetriesOnlyUnreadableReads(t *testing.T) {
	const witnessedJSON = `[{"sha":"abc1234","verdict":"OK","claim_kind":"code",` +
		`"witness":"diff-witnessed","reason":"code claim; diff touches source"}]`
	const emptyJSON = `[{"sha":"abc1234","verdict":"CLAIM_UNWITNESSED","claim_kind":"code",` +
		`"witness":"subject-only","reason":"code-effect claim but the commit is EMPTY (touched no files)"}]`
	const noSourceJSON = `[{"sha":"abc1234","verdict":"CLAIM_UNWITNESSED","claim_kind":"code",` +
		`"witness":"subject-only","reason":"code-effect claim but the diff touches no SOURCE file ` +
		`(only: README.md) — the claim rests on the subject text"}]`
	const abstainJSON = `[{"sha":"abc1234","verdict":"ABSTAIN","claim_kind":"none",` +
		`"witness":"none","reason":"no checkable claim"}]`
	unwitnessedExit := errors.New("exit status 1")

	cases := []struct {
		name          string
		runErr        error
		stdout        string
		localNonEmpty bool
		want          dosAuditOutcome
	}{{
		name:          "OK/diff-witnessed on a clean exit is the pass",
		stdout:        witnessedJSON,
		localNonEmpty: true,
		want:          dosAuditWitnessed,
	}, {
		// The #5519 shape: git says the commit touched a file, dos says it is empty.
		// dos cannot have read the diff, so the verdict is about dos, not the commit.
		name:          "EMPTY over a commit git says is non-empty is a failed read",
		runErr:        unwitnessedExit,
		stdout:        emptyJSON,
		localNonEmpty: true,
		want:          dosAuditUnreadable,
	}, {
		// Without a local witness the classifier does NOT invent a reason to retry.
		name:   "EMPTY with no local witness is taken at face value",
		runErr: unwitnessedExit,
		stdout: emptyJSON,
		want:   dosAuditStableFail,
	}, {
		// The regression this smoke exists to catch: dos read the diff and disagreed.
		// One attempt, no retry -- otherwise a real break costs three runs to surface.
		name:          "a real claim-vs-diff disagreement is never retried",
		runErr:        unwitnessedExit,
		stdout:        noSourceJSON,
		localNonEmpty: true,
		want:          dosAuditStableFail,
	}, {
		name:          "an abstain is a stable disagreement, not a failed read",
		stdout:        abstainJSON,
		localNonEmpty: true,
		want:          dosAuditStableFail,
	}, {
		// dos's contract_error arm prints to stderr and emits no JSON at all.
		name:          "no JSON on stdout is a failed read",
		runErr:        errors.New("exit status 2"),
		stdout:        "",
		localNonEmpty: true,
		want:          dosAuditUnreadable,
	}, {
		name:          "a stderr line merged into stdout is a failed read, not a pass",
		runErr:        errors.New("exit status 2"),
		stdout:        "commit-audit: cannot read 'abc1234' in /tmp/x (not a git repo, or bad ref)\n",
		localNonEmpty: true,
		want:          dosAuditUnreadable,
	}, {
		name:          "a sweep-shaped object instead of a row array is a failed read",
		stdout:        `{"commits":1,"witnessed":1}`,
		localNonEmpty: true,
		want:          dosAuditUnreadable,
	}, {
		name:          "more than one row is a failed read (the smokes audit exactly one commit)",
		stdout:        `[` + witnessedJSON[1:len(witnessedJSON)-1] + `,` + witnessedJSON[1:len(witnessedJSON)-1] + `]`,
		localNonEmpty: true,
		want:          dosAuditUnreadable,
	}, {
		// A verdict body that says OK while the process exited non-zero is incoherent;
		// it must not count as the pass.
		name:          "OK text with a non-zero exit is not the pass",
		runErr:        unwitnessedExit,
		stdout:        witnessedJSON,
		localNonEmpty: true,
		want:          dosAuditStableFail,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, row, parsed := classifyDosAudit(tc.runErr, []byte(tc.stdout), tc.localNonEmpty)
			if got != tc.want {
				t.Fatalf("classifyDosAudit(%v, %q, local=%v) = %v (row=%+v parsed=%v), want %v",
					tc.runErr, tc.stdout, tc.localNonEmpty, got, row, parsed, tc.want)
			}
		})
	}
}

// TestModuleRootFromLocatesAnExportWithoutGit pins the other half of the smokes'
// robustness: they resolve tools/issue_resolve_witnessed.py through repoRootFromCwd,
// which used to be a bare `git rev-parse --show-toplevel`. In a `git archive HEAD |
// tar -x` clean-room -- an export with no .git in it -- that exits 128 and the smokes
// died on a naked "exit status 128" with no hint why. The go.mod walk-up answers the
// same question without git, so an export resolves to the EXPORT (not to whatever
// repo happens to contain it), and a directory with no module above it reports "not
// found" so the caller can say so in words.
func TestModuleRootFromLocatesAnExportWithoutGit(t *testing.T) {
	// An export of the tree: go.mod at the top, the package dir below, NO .git.
	export := t.TempDir()
	if err := os.WriteFile(filepath.Join(export, "go.mod"),
		[]byte("module example.test\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	pkgDir := filepath.Join(export, "cmd", "fak")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	got, ok := moduleRootFrom(pkgDir)
	if !ok {
		t.Fatalf("moduleRootFrom(%q) found no module root; an export with a go.mod at its top must resolve", pkgDir)
	}
	if got != export {
		t.Fatalf("moduleRootFrom(%q) = %q, want the export root %q", pkgDir, got, export)
	}

	// The nearest module wins, so a nested export inside another checkout resolves to
	// itself rather than to the outer tree.
	inner := filepath.Join(export, "vendorish")
	if err := os.MkdirAll(filepath.Join(inner, "cmd"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(inner, "go.mod"),
		[]byte("module example.inner\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, ok := moduleRootFrom(filepath.Join(inner, "cmd")); !ok || got != inner {
		t.Fatalf("moduleRootFrom(nested) = (%q, %v), want the nearest module root %q", got, ok, inner)
	}

	// No module anywhere above: report not-found rather than guessing, so
	// repoRootFromCwd falls through to git and then to a message that names both.
	bare := t.TempDir()
	deep := filepath.Join(bare, "a", "b")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	// t.TempDir() lives under the OS temp root; only assert not-found when no stray
	// go.mod exists above it, which is the normal case and the one worth pinning.
	if root, ok := moduleRootFrom(deep); ok && root == bare {
		t.Fatalf("moduleRootFrom(%q) resolved %q with no go.mod written", deep, root)
	}
}
