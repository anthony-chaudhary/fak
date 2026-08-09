package safecommit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Canned object ids for the untracked-path check. The tip is a full-length sha so the
// refusal's abbreviation is exercised rather than passed through.
const (
	staleUntrackedTip       = "0f1e2d3c4b5a69788796a5b4c3d2e1f009182736"
	staleUntrackedTrunkBlob = "1111111111111111111111111111111111111111"
	staleUntrackedLocalBlob = "2222222222222222222222222222222222222222"
)

// trunkBlobLine is one `git ls-tree -l` row: mode, type, object, right-padded size, TAB, name.
func trunkBlobLine(blob, name string) string {
	return fmt.Sprintf("100644 blob %s     412\t%s\n", blob, name)
}

// staleUntrackedReply overlays onTrunkBase with everything checkStaleUntrackedPath reads: a
// resolvable origin/main whose tip DIFFERS from the merge base (this HEAD is behind the
// trunk), the requested path reported `??`, and the two blob probes. The (4b) line-run diffs
// are deliberately LEFT EMPTY here so a test that expects no refusal is not measuring the
// other check; withStaleBaseTrap adds them when the hand-off itself is what is under test.
func staleUntrackedReply(lsTree, hashObject string) map[string]reply {
	rep := onTrunkBase()
	rep["status"] = reply{out: "?? internal/foo/bar.go\n", code: 0}
	rep["rev-parse --verify --quiet origin/main"] = reply{out: staleUntrackedTip + "\n", code: 0}
	rep["merge-base HEAD origin/main"] = reply{out: "ffff1111\n", code: 0}
	rep["ls-tree"] = reply{out: lsTree, code: 0}
	rep["hash-object"] = reply{out: hashObject, code: 0}
	return rep
}

// withStaleBaseTrap arms the (4b) content check with the diff shape git actually produces for
// an UNTRACKED path: the path is absent from the index, so trunk's whole file reads as
// deleted, and the line-run check therefore reports "would drop 3 line(s)" — the misleading
// diagnosis #5408 was written about. Any path (4a2) hands on to (4b) refuses visibly here, so
// "did (4a2) claim this path?" is observable in Result.Reason.
func withStaleBaseTrap(rep map[string]reply) map[string]reply {
	rep["diff-peer"] = reply{out: "@@ -0,0 +1,3 @@\n+peerLineOne()\n+peerLineTwo()\n+peerLineThree()\n", code: 0}
	rep["diff-wt"] = reply{out: "@@ -1,3 +0,0 @@\n-peerLineOne()\n-peerLineTwo()\n-peerLineThree()\n", code: 0}
	return rep
}

// TestStaleUntrackedPath_refusesDifferingCopyOfATrunkFile is the bug-reproducing case of
// #5408: internal/foo/bar.go is `??` in this behind-HEAD checkout yet already on origin/main
// with DIFFERENT content. Committing it by pathspec would supersede the trunk copy with an
// older one, so the commit is refused before any add — and the refusal names the TRUNK TIP,
// both blob ids, and the content-to-content comparison, never a line count.
func TestStaleUntrackedPath_refusesDifferingCopyOfATrunkFile(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "") // default = block
	g := &fakeGit{reply: staleUntrackedReply(
		trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go"),
		staleUntrackedLocalBlob+"\n",
	)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonStaleUntrackedPath {
		t.Fatalf("want %q, got reason=%q detail=%q", ReasonStaleUntrackedPath, res.Reason, res.Detail)
	}
	if res.Committed {
		t.Fatalf("a stale-untracked refusal must not commit, got %+v", res)
	}
	for _, forbidden := range []string{"add", "commit"} {
		if g.sawSubcommand(forbidden) {
			t.Fatalf("the refusal must not %q; calls=%v", forbidden, g.calls)
		}
	}
	// The acceptance is specific about WHICH sha: the trunk tip, not this clone's fork point.
	for _, want := range []string{
		"internal/foo/bar.go",
		staleUntrackedTip[:12],
		staleUntrackedTrunkBlob[:12],
		staleUntrackedLocalBlob[:12],
		"git show origin/main:internal/foo/bar.go",
		"git fetch origin main",
	} {
		if !strings.Contains(res.Detail, want) {
			t.Errorf("detail should contain %q, got %q", want, res.Detail)
		}
	}
	if strings.Contains(res.Detail, "ffff1111") {
		t.Errorf("detail must name the trunk tip, not the fork point, got %q", res.Detail)
	}
	// It must not reach for the diagnosis #5408 documents as actively misleading.
	if strings.Contains(res.Detail, "would drop") {
		t.Errorf("an untracked path has no line-run delta; detail must not claim one, got %q", res.Detail)
	}
	if code, ok := RefusalExitCode(ReasonStaleUntrackedPath); !ok || code != ExitRefused {
		t.Errorf("RefusalExitCode(%s) = (%d, %v), want (%d, true) — nothing landed, but the path is "+
			"already on the trunk, so re-running the same commit can never change the answer",
			ReasonStaleUntrackedPath, code, ok, ExitRefused)
	}
}

// TestStaleUntrackedPath_byteIdenticalIsANoOpNotARefusal pins the over-refusal half of #5408,
// and it is the invariant most worth protecting: 40 of the 69 untracked paths measured in the
// field were byte-identical to trunk. Committing such a path supersedes NOTHING, so it must
// not refuse — it is reported and allowed. The (4b) trap is armed, so this also proves the
// claim is real: without the hand-off, the very same fixture refuses with the wrong reason and
// a fabricated "would drop 3 line(s)".
func TestStaleUntrackedPath_byteIdenticalIsANoOpNotARefusal(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "") // default = block
	g := &fakeGit{reply: withStaleBaseTrap(staleUntrackedReply(
		trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go"),
		staleUntrackedTrunkBlob+"\n", // identical to what the trunk holds
	))}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonStaleUntrackedPath {
		t.Fatalf("a byte-identical copy supersedes nothing; it must not refuse, got detail=%q", res.Detail)
	}
	if res.Reason == ReasonStaleBaseDeletion {
		t.Fatalf("the claimed path leaked to the line-run check, which reports a fabricated line count for an untracked path: %q", res.Detail)
	}
	if res.Reason != "" || !res.Verified {
		t.Fatalf("the no-op case must commit and verify, got %+v", res)
	}
	if !strings.Contains(res.Detail, "STALE_UNTRACKED (no-op)") || !strings.Contains(res.Detail, "internal/foo/bar.go") {
		t.Fatalf("the no-op must still be NAMED so the operator learns HEAD is behind, got %q", res.Detail)
	}
	if !strings.Contains(res.Detail, "BYTE-IDENTICAL") {
		t.Fatalf("the note should say why it is not a refusal, got %q", res.Detail)
	}
}

// TestStaleUntrackedPath_trackedPathStaysWithTheLineRunCheck proves the hand-off to (4b) is
// conditional, not a blanket suppression: the same armed fixture, with the path TRACKED in
// this index, is not this class at all and still refuses at (4b). Without this, "identical
// paths stop refusing" could be a silent disarming of the older check.
func TestStaleUntrackedPath_trackedPathStaysWithTheLineRunCheck(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	rep := withStaleBaseTrap(staleUntrackedReply(
		trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go"),
		staleUntrackedLocalBlob+"\n",
	))
	rep["status"] = reply{out: " M internal/foo/bar.go\n", code: 0}
	rep["ls-files"] = reply{out: "internal/foo/bar.go\n", code: 0} // tracked here
	g := &fakeGit{reply: rep}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonStaleBaseDeletion {
		t.Fatalf("a TRACKED path is the line-run check's class; want %q, got reason=%q detail=%q",
			ReasonStaleBaseDeletion, res.Reason, res.Detail)
	}
}

// TestStaleUntrackedPath_genuinelyNewPathCommitsNormally is the "no new friction on the 25
// real ones" acceptance: a path absent from origin/main is new work and commits untouched.
// The check also stops before hashing the working copy — an absent upstream entry already
// settles it.
func TestStaleUntrackedPath_genuinelyNewPathCommitsNormally(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	g := &fakeGit{reply: staleUntrackedReply("", staleUntrackedLocalBlob+"\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Verified {
		t.Fatalf("a genuinely new path must commit, got %+v", res)
	}
	if strings.Contains(res.Detail, "STALE_UNTRACKED") {
		t.Fatalf("a genuinely new path must not be annotated, got %q", res.Detail)
	}
	if g.sawSubcommand("hash-object") {
		t.Fatalf("absent upstream settles it; the working copy need not be hashed. calls=%v", g.calls)
	}
}

// TestStaleUntrackedPath_unknownsFallBackToPriorBehavior is the direction-of-safety case. A
// refusal is emitted only from a positive reading; every state the check cannot establish an
// answer from must leave the commit exactly as it was. This matters more than the bug it
// closes: a check that refused on an unreadable ref would wedge every lane in a shared tree.
func TestStaleUntrackedPath_unknownsFallBackToPriorBehavior(t *testing.T) {
	lsTree := trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go")
	local := staleUntrackedLocalBlob + "\n" // differs from trunk: everything else is armed

	for _, tc := range []struct {
		name   string
		mutate func(map[string]reply)
	}{
		{"no remote-tracking ref (fresh clone)", func(r map[string]reply) {
			r["rev-parse --verify --quiet origin/main"] = reply{out: "", code: 1}
		}},
		{"unreadable merge base", func(r map[string]reply) {
			r["merge-base HEAD origin/main"] = reply{out: "", code: 1}
		}},
		{"HEAD already contains the trunk tip", func(r map[string]reply) {
			r["merge-base HEAD origin/main"] = reply{out: staleUntrackedTip + "\n", code: 0}
		}},
		{"index unreadable for the path", func(r map[string]reply) {
			r["ls-files"] = reply{out: "", code: 128}
		}},
		{"upstream listing unreadable", func(r map[string]reply) {
			r["ls-tree"] = reply{out: "", code: 128}
		}},
		{"upstream entry is a tree, not a file", func(r map[string]reply) {
			r["ls-tree"] = reply{out: "040000 tree 3333333333333333333333333333333333333333       -\tinternal/foo\n", code: 0}
		}},
		{"pathspec matches several upstream entries", func(r map[string]reply) {
			r["ls-tree"] = reply{out: trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go") +
				trunkBlobLine(staleUntrackedLocalBlob, "internal/foo/baz.go"), code: 0}
		}},
		{"working copy unhashable", func(r map[string]reply) {
			r["hash-object"] = reply{out: "", code: 128}
		}},
		{"working copy hash empty", func(r map[string]reply) {
			r["hash-object"] = reply{out: "\n", code: 0}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(staleBaseEnvVar, "")
			rep := staleUntrackedReply(lsTree, local)
			tc.mutate(rep)
			g := &fakeGit{reply: rep}

			res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
			if err != nil {
				t.Fatalf("unexpected infra error: %v", err)
			}
			if res.Reason == ReasonStaleUntrackedPath {
				t.Fatalf("an unknown must never manufacture a refusal, got detail=%q", res.Detail)
			}
			if res.Reason != "" || !res.Verified {
				t.Fatalf("the commit must proceed exactly as before, got %+v", res)
			}
		})
	}
}

// TestStaleUntrackedPath_noRefLeavesTheCheckUnrun: the cheapest read comes first, so a clone
// with no origin/main never even resolves a merge base.
func TestStaleUntrackedPath_noRefLeavesTheCheckUnrun(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	rep := staleUntrackedReply(trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go"), staleUntrackedLocalBlob+"\n")
	rep["rev-parse --verify --quiet origin/main"] = reply{out: "", code: 1}
	g := &fakeGit{reply: rep}

	if _, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts()); err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	for _, sub := range []string{"merge-base", "ls-tree", "hash-object"} {
		if g.sawSubcommand(sub) {
			t.Fatalf("no origin ref must short-circuit before %q; calls=%v", sub, g.calls)
		}
	}
}

// TestStaleUntrackedPath_warnModeCommits: the documented one-shot escape. An operator who has
// seen the content-to-content comparison and genuinely means to supersede trunk's copy gets
// the finding recorded in Detail instead of a refusal.
func TestStaleUntrackedPath_warnModeCommits(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "warn")
	g := &fakeGit{reply: staleUntrackedReply(
		trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go"),
		staleUntrackedLocalBlob+"\n",
	)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Verified {
		t.Fatalf("warn mode should commit and verify, got %+v", res)
	}
	if !strings.Contains(res.Detail, "STALE_UNTRACKED (warn)") {
		t.Fatalf("warn mode should record the would-be refusal in Detail, got %q", res.Detail)
	}
}

// TestStaleUntrackedPath_offModeSkips: off falls through entirely — the check does not even
// read the upstream listing.
func TestStaleUntrackedPath_offModeSkips(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "off")
	g := &fakeGit{reply: staleUntrackedReply(
		trunkBlobLine(staleUntrackedTrunkBlob, "internal/foo/bar.go"),
		staleUntrackedLocalBlob+"\n",
	)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Verified {
		t.Fatalf("off mode should skip the check and commit, got %+v", res)
	}
	for _, sub := range []string{"ls-tree", "hash-object"} {
		if g.sawSubcommand(sub) {
			t.Fatalf("off mode must not run the blob probes; calls=%v", g.calls)
		}
	}
}

// TestStaleUntrackedNoOpNote_boundsTheList keeps the no-op advisory readable when the field
// case (40 paths) shows up: it names a few and counts the rest.
func TestStaleUntrackedNoOpNote_boundsTheList(t *testing.T) {
	many := []string{"a.go", "b.go", "c.go", "d.go", "e.go"}
	note := staleUntrackedNoOpNote("main", "origin/main", staleUntrackedTip, many)
	if !strings.Contains(note, "5 requested path(s)") {
		t.Errorf("note should count every path, got %q", note)
	}
	if !strings.Contains(note, "a.go, b.go, c.go") || strings.Contains(note, "d.go") {
		t.Errorf("note should name the first %d and count the rest, got %q", staleUntrackedNoOpListed, note)
	}
	if !strings.Contains(note, "(+2 more)") {
		t.Errorf("note should say how many it left unnamed, got %q", note)
	}
	if one := staleUntrackedNoOpNote("main", "origin/main", staleUntrackedTip, []string{"a.go"}); strings.Contains(one, "more)") {
		t.Errorf("a single no-op needs no overflow count, got %q", one)
	}
}

// --- real git, hermetic -----------------------------------------------------------------

// tempRepoGit runs one git command in dir through the package's own runner (so the invocation
// is configured exactly as production does) and fails the test on any non-zero exit. Identity
// and signing are supplied per-invocation with -c, so the test never reads or writes the
// developer's global git configuration.
func tempRepoGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	full := append([]string{
		"-c", "user.name=fak test",
		"-c", "user.email=fak-test@example.invalid",
		"-c", "commit.gpgsign=false",
		"-c", "core.autocrlf=false",
		"-c", "init.defaultBranch=main",
	}, args...)
	out, code, err := realRunner(context.Background(), dir, full...)
	if err != nil || code != 0 {
		t.Fatalf("git %v in %s: code=%d err=%v out=%s", args, dir, code, err, out)
	}
	return out
}

func writeTempRepoFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestStaleUntrackedPath_realGitBehindTrunk drives the check against REAL git in a throwaway
// repo, because the canned tests cannot catch a wrong flag: `ls-tree -l` column order and
// `hash-object`'s filtered object id are the two things this check's answer rests on.
//
// The repo is built with `git init` under t.TempDir() and a local bare repo standing in for
// origin — no network, no remote, and nothing read from the developer's checkout. The shape
// reproduced is exactly #5408's: HEAD rewound one commit behind origin/main, with the peer's
// file left in the working tree where `git status` calls it `??`.
func TestStaleUntrackedPath_realGitBehindTrunk(t *testing.T) {
	root := t.TempDir()
	bare := filepath.Join(root, "up.git")
	work := filepath.Join(root, "wk")
	tempRepoGit(t, root, "init", "-q", "--bare", "-b", "main", bare)
	tempRepoGit(t, root, "clone", "-q", bare, work)
	tempRepoGit(t, work, "symbolic-ref", "HEAD", "refs/heads/main")

	writeTempRepoFile(t, filepath.Join(work, "seed.txt"), "base\n")
	tempRepoGit(t, work, "add", "--", "seed.txt")
	tempRepoGit(t, work, "commit", "-qm", "base")
	tempRepoGit(t, work, "push", "-q", "origin", "main")

	// A peer adds a brand-new file upstream...
	peer := "peer line one\npeer line two\npeer line three\npeer line four\n"
	writeTempRepoFile(t, filepath.Join(work, "sub", "p.go"), peer)
	tempRepoGit(t, work, "add", "--", "sub/p.go")
	tempRepoGit(t, work, "commit", "-qm", "peer")
	tempRepoGit(t, work, "push", "-q", "origin", "main")
	tip := strings.TrimSpace(tempRepoGit(t, work, "rev-parse", "origin/main"))

	// ...and this checkout falls behind, leaving sub/p.go untracked in the working tree.
	tempRepoGit(t, work, "reset", "-q", "--hard", "HEAD~1")

	for _, tc := range []struct {
		name        string
		body        string
		wantRefusal bool
	}{
		{"differs from the trunk copy", "peer line one\nLOCAL STALE LINE\n", true},
		{"byte-identical to the trunk copy", peer, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			writeTempRepoFile(t, filepath.Join(work, "sub", "p.go"), tc.body)
			status := strings.TrimSpace(tempRepoGit(t, work, "status", "--porcelain", "--", "sub/p.go"))
			if !strings.HasPrefix(status, "??") {
				t.Fatalf("fixture is not the #5408 shape: status = %q, want an untracked path", status)
			}

			refusal, advisory, unclaimed := checkStaleUntrackedPath(context.Background(), realRunner, work, "main", []string{"sub/p.go"})
			if len(unclaimed) != 0 {
				t.Fatalf("a path present upstream must be claimed, so the line-run check never sees it; unclaimed=%v", unclaimed)
			}
			if tc.wantRefusal {
				if refusal == "" {
					t.Fatalf("a differing copy of a trunk file must be refused; advisory=%q", advisory)
				}
				if !strings.Contains(refusal, tip[:12]) {
					t.Errorf("the refusal must name the trunk tip %s, got %q", tip[:12], refusal)
				}
				if advisory != "" {
					t.Errorf("a differing copy is not a no-op; advisory=%q", advisory)
				}
				return
			}
			if refusal != "" {
				t.Fatalf("a byte-identical copy supersedes nothing and must not be refused, got %q", refusal)
			}
			if !strings.Contains(advisory, "sub/p.go") || !strings.Contains(advisory, "BYTE-IDENTICAL") {
				t.Errorf("the no-op case must still be named, got %q", advisory)
			}
		})
	}

	// Brought up to date with the trunk: the path is tracked again, so the check has nothing
	// to say — the remedy it prints actually clears the condition it refuses on.
	tempRepoGit(t, work, "reset", "-q", "--hard", "origin/main")
	refusal, advisory, unclaimed := checkStaleUntrackedPath(context.Background(), realRunner, work, "main", []string{"sub/p.go"})
	if refusal != "" || advisory != "" || len(unclaimed) != 1 {
		t.Fatalf("after the merge the path is tracked and up to date; want silence, got refusal=%q advisory=%q unclaimed=%v",
			refusal, advisory, unclaimed)
	}
}
