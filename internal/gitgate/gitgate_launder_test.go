package gitgate

import "testing"

// flatClassify reproduces the pre-unwrap classify() path EXACTLY: it tokenizes the
// raw command once with tokenizeSegments and runs the hazard table over each
// segment, WITHOUT the unwrapShellSources recovery pass in front. It is the
// "structurally blind" floor the issue (#823) describes — the argv prefilter as it
// behaved before the unwrap pass. Holding it next to the real Classify() is what
// proves the unwrap pass is load-bearing, not decorative.
func flatClassify(g *GitGate, cmd string) (string, bool) {
	for _, seg := range tokenizeSegments(cmd) {
		argv := gitArgv(seg)
		if argv == nil {
			continue
		}
		if law, ok := g.inspectGit(argv); ok {
			return law, true
		}
	}
	return "", false
}

// TestUnwrapPassIsLoadBearing proves the security claim of #823 directly: a git
// hazard the shell grammar hides inside a quoted `bash -c`/`sh -c` string or a
// backtick substitution is INVISIBLE to the flat tokenizer (flatClassify defers),
// yet the full Classify() — with the unwrapShellSources recovery in front — REFUSES
// it. Each case is one the flat floor provably misses, so the assertion fails the
// moment the unwrap pass is removed or stops feeding the recovered string to the
// same hazard rules. (Pipes / && / || / ; and a paren-exposed $(...) are already
// segmented by tokenizeSegments, so they are NOT load-bearing for unwrap and are
// deliberately excluded here — only the quoted-string laundering the flat path
// cannot see belongs in this table.)
func TestUnwrapPassIsLoadBearing(t *testing.T) {
	g := New()
	cases := []struct {
		name   string
		cmd    string
		lawHas string
	}{
		{"bash -c single quote force", `bash -c 'git push --force origin main'`, "force-push"},
		{"bash -c double quote amend", `bash -c "git commit --amend"`, "amend"},
		{"sh -c add -A", `sh -c 'git add -A'`, "explicit-path"},
		{"absolute bash -c force", `/bin/bash -c 'git push -f'`, "force-push"},
		{"bash -lc cluster force", `bash -lc 'git push --force'`, "force-push"},
		{"backtick force-push", "echo `git push -f`", "force-push"},
		{"backtick amend", "x=`git commit --amend`", "amend"},
		{"bash -c nesting a bash -c", `bash -c "bash -c 'git push -f'"`, "force-push"},
		{"bash -c hiding a no-verify", `bash -c 'git commit --no-verify -m x'`, "skip-hooks"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// The flat floor is blind to it — this is the gap #823 names.
			if law, denied := flatClassify(g, tc.cmd); denied {
				t.Fatalf("flat floor unexpectedly caught %q (law %q); pick a case the flat path actually misses so the load-bearing claim is honest", tc.cmd, law)
			}
			// The unwrap pass makes it visible to the SAME rules → refuse.
			law, denied := g.Classify(tc.cmd)
			if !denied {
				t.Fatalf("Classify(%q): unwrap pass failed to recover the laundered hazard (got defer, want deny %q)", tc.cmd, tc.lawHas)
			}
			if !containsSub(law, tc.lawHas) {
				t.Fatalf("Classify(%q): cited law %q does not mention %q", tc.cmd, law, tc.lawHas)
			}
		})
	}
}

// TestUnwrapPreservesFlatVerdicts pins the other half of the #823 contract: the
// unwrap pass is a STRICT SUPERSET of the flat floor — it only widens what the
// existing rules can SEE, it never flips a verdict the flat path already produces.
// For any command with no quoted-string / substitution laundering, Classify() and
// flatClassify() must agree exactly (both the deny bit and, when denied, the cited
// law). This guards against the unwrap pass introducing a false-positive on a safe
// command or re-citing a different law than the flat tokenizer would.
func TestUnwrapPreservesFlatVerdicts(t *testing.T) {
	g := New()
	cmds := []string{
		"git push origin main",    // safe push → defer
		"git push -u origin main", // upstream → defer
		"git status",              // non-hazard → defer
		"git commit -m \"always run git push --force\"", // hazard word in a quoted message → defer
		"echo git push --force",                         // not a git call in command position → defer
		"git push --force origin main",                  // direct hazard → deny force-push
		"git commit --amend",                            // direct hazard → deny amend
		"git add -A",                                    // direct hazard → deny explicit-path
		"true && git push --force",                      // operator-segmented hazard the flat floor already catches
	}
	for _, cmd := range cmds {
		t.Run(cmd, func(t *testing.T) {
			fl, fd := flatClassify(g, cmd)
			cl, cd := g.Classify(cmd)
			if fd != cd || fl != cl {
				t.Fatalf("unwrap pass changed a non-laundered verdict for %q: flat=(%q,%v) full=(%q,%v); the pass must be additive-only", cmd, fl, fd, cl, cd)
			}
		})
	}
}

// containsSub is a tiny local substring check kept dependency-free so this file
// stands alone without importing strings just for one call.
func containsSub(s, sub string) bool {
	if sub == "" {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
