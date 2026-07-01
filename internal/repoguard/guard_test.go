// guard_test.go — the parity witness for the Go repo-guard port. Every case here
// is ported 1:1 from tools/repo_guard_test.py so the compiled binary and the
// retired Python script classify identically. Hermetic: no filesystem, no spawn.
package repoguard

import (
	"strings"
	"testing"
)

const wsTest = "C:/Users/u/work/fak"

var safeTest = []string{"/tmp", "/var/tmp", "C:/Users/u/.cache", "C:/Users/u/Downloads"}

func eval(tool string, input map[string]any) []Violation {
	return evaluate(tool, input, wsTest, safeTest)
}

func TestIncidentRelativeEscapeDenied(t *testing.T) {
	// The build-output escape that seeded the incident.
	v := eval("Bash", cmd("go build -o ../tools/.bin/fak.exe ./cmd/fak"))
	if len(v) == 0 {
		t.Fatal("expected a violation for the out-of-tree build output")
	}
	if v[0].Reason != "OUT_OF_TREE_WRITE" {
		t.Errorf("reason = %q, want OUT_OF_TREE_WRITE", v[0].Reason)
	}
}

func TestIncidentAbsoluteSiblingRmDenied(t *testing.T) {
	// The exact rm that destroyed the sibling repo (absolute path a regex can't judge).
	v := eval("Bash", cmd("rm -rf /c/Users/u/work/tools"))
	if len(v) == 0 {
		t.Fatal("expected a violation for the absolute sibling rm")
	}
	if v[0].Why != "sibling of workspace" {
		t.Errorf("why = %q, want \"sibling of workspace\"", v[0].Why)
	}
}

func TestWriteToolEscapeDenied(t *testing.T) {
	if len(eval("Write", fp("../tools/poison.txt"))) == 0 {
		t.Error("expected a violation for a Write escaping the tree")
	}
}

func TestInRepoOpsAllowed(t *testing.T) {
	for _, c := range []string{
		"go build -o fak.exe ./cmd/fak",
		"go build -o tools/.bin/fak.exe ./cmd/fak",
		"rm -rf ./build",
		"rm -rf internal/model/.cache",
		"mv internal/a internal/b",
	} {
		if v := eval("Bash", cmd(c)); len(v) != 0 {
			t.Errorf("%q: expected allow, got %v", c, v)
		}
	}
}

func TestScratchRootsAllowed(t *testing.T) {
	for _, c := range []string{
		"echo x > /tmp/log.txt",
		"cp a.txt /var/tmp/b.txt",
		"cp a.txt ~/.cache/b.txt", // ~ is unresolvable but not a textual escape -> allow
	} {
		if v := eval("Bash", cmd(c)); len(v) != 0 {
			t.Errorf("%q: expected allow, got %v", c, v)
		}
	}
}

func TestSafeRootAncestorDoesNotSwallowWorkspaceSiblings(t *testing.T) {
	// WSL fast tests mirror the repo under ~/.cache. If the ~/.cache safe root is allowed
	// to cover a workspace below it, an out-of-tree sibling write becomes invisible.
	ws := "/home/user/.cache/fak-src/cmd/fak/.loop-containment-123/repo"
	v := ClassifyCommand("echo bad > ../victim.txt", ws, []string{"/home/user/.cache"})
	if len(v) == 0 {
		t.Fatal("expected a violation for a sibling write under an ancestor safe root")
	}
	if v[0].Reason != "OUT_OF_TREE_WRITE" {
		t.Errorf("reason = %q, want OUT_OF_TREE_WRITE", v[0].Reason)
	}
}

func TestClaudeStateDirIsASafeRoot(t *testing.T) {
	if !strings.Contains(strings.Join(defaultSafeRoots(), ""), "/.claude") {
		t.Error("default safe roots must include the agent's ~/.claude state tree")
	}
}

func TestAgentMemoryWriteAllowed(t *testing.T) {
	roots := append(append([]string{}, safeTest...), "C:/Users/u/.claude")
	fpath := "C:/Users/u/.claude/projects/C--Users-u-work-fak/memory/note.md"
	if v := evaluate("Write", fp(fpath), wsTest, roots); len(v) != 0 {
		t.Errorf("%q: expected allow, got %v", fpath, v)
	}
}

func TestAgentStateRootsAdmitsAccountVariantNotLookalike(t *testing.T) {
	roots := agentStateRoots("C:/Users/u", []string{".claude", ".claude-gem8-netra", ".claude.json", ".claudex", "Documents"})
	want := []string{"C:/Users/u/.claude", "C:/Users/u/.claude-gem8-netra", "C:/Users/u/.claude.json"}
	for _, w := range want {
		if !contains(roots, w) {
			t.Errorf("agentStateRoots missing %q (got %v)", w, roots)
		}
	}
	for _, bad := range []string{"C:/Users/u/.claudex", "C:/Users/u/Documents"} {
		if contains(roots, bad) {
			t.Errorf("agentStateRoots must not admit %q (got %v)", bad, roots)
		}
	}
}

func TestAccountVariantMemoryWriteAllowed(t *testing.T) {
	// The exact path the live hook was denying before the original fix.
	roots := append(append([]string{}, safeTest...), agentStateRoots("C:/Users/u", []string{".claude-gem8-netra"})...)
	fpath := "C:/Users/u/.claude-gem8-netra/projects/C--Users-u-work-fak/memory/note.md"
	if v := evaluate("Write", fp(fpath), wsTest, roots); len(v) != 0 {
		t.Errorf("%q: expected allow, got %v", fpath, v)
	}
}

func TestPrivateCompanionIsTheSameNamedSiblingOnly(t *testing.T) {
	if got := privateCompanionRoots(wsTest); len(got) != 1 || got[0] != "C:/Users/u/work/fak-private" {
		t.Errorf("privateCompanionRoots(%q) = %v, want [C:/Users/u/work/fak-private]", wsTest, got)
	}
	if got := privateCompanionRoots("/c/work/fak"); len(got) != 1 || got[0] != "C:/work/fak-private" {
		t.Errorf("privateCompanionRoots(/c/work/fak) = %v, want [C:/work/fak-private]", got)
	}
}

func TestPrivateCompanionEmptyWhenWorkspaceIsPrivate(t *testing.T) {
	// A workspace that is ITSELF the private repo has no further companion: a
	// <ws>-private-private does not exist, so we must NOT synthesize it as a safe
	// root (that would silently admit an out-of-tree write to a nonexistent sibling).
	if got := privateCompanionRoots("C:/work/fak-private"); len(got) != 0 {
		t.Errorf("privateCompanionRoots of a -private workspace = %v, want empty", got)
	}
	// And the write to that nonexistent -private-private sibling must DENY.
	ws := "C:/work/fak-private"
	roots := append(append([]string{}, safeTest...), privateCompanionRoots(ws)...)
	if v := evaluate("Write", fp("C:/work/fak-private-private/evil.md"), ws, roots); len(v) == 0 {
		t.Error("write to fak-private-private must be denied (no auto-safe companion)")
	}
}

func TestPrivateCompanionWriteAllowed(t *testing.T) {
	roots := append(append([]string{}, safeTest...), privateCompanionRoots(wsTest)...)
	fpath := "C:/Users/u/work/fak-private/MEMORY-glm52-2026-06-21.md"
	if v := evaluate("Write", fp(fpath), wsTest, roots); len(v) != 0 {
		t.Errorf("%q: expected allow, got %v", fpath, v)
	}
}

func TestPrivateCompanionDoesNotLeakToLookalikeOrSiblings(t *testing.T) {
	roots := append(append([]string{}, safeTest...), privateCompanionRoots(wsTest)...)
	roots = append(roots, "C:/Users/u/.claude-gem8-netra")
	for _, fpath := range []string{
		"C:/Users/u/work/fak-private-evil/x.md", // look-alike component, not the companion
		"C:/Users/u/work/fak-ci/x.md",           // a real but unrelated sibling
		"C:/Users/u/work/tools/poison.txt",      // the incident sibling
		"C:/Users/u/.claudex/leak.md",           // .claude look-alike, not the state tree
	} {
		if v := evaluate("Write", fp(fpath), wsTest, roots); len(v) == 0 {
			t.Errorf("%q: expected DENY, got allow", fpath)
		}
	}
}

func TestGrepDashOIsNotAnOutputPath(t *testing.T) {
	// -o is overloaded: grep -o is only-matching, not a build output file.
	if v := eval("Bash", cmd("grep -o ../foo internal/policy/x.go")); len(v) != 0 {
		t.Errorf("grep -o must not be read as an output path, got %v", v)
	}
}

func TestReadsAreNeverFlagged(t *testing.T) {
	if v := eval("Bash", cmd("cat ../README.md")); len(v) != 0 {
		t.Errorf("a read must never be flagged, got %v", v)
	}
}

func TestNullDeviceSinksAllowed(t *testing.T) {
	// `> /dev/null` and friends resolve outside the workspace but can never harm a
	// sibling repo, so they are exempt (otherwise the idiom would push operators to
	// disable the whole guard).
	for _, c := range []string{
		"make ci > /dev/null 2>&1",
		"go test ./... > /dev/null",
		"echo done >> /dev/stderr",
	} {
		if v := eval("Bash", cmd(c)); len(v) != 0 {
			t.Errorf("%q: device sink must be allowed, got %v", c, v)
		}
	}
}

func TestHeredocProgramComparisonsAreNotRedirects(t *testing.T) {
	command := "python - file <<'EOF'\nif depth > 3:\n    return depth\nEOF\n"
	if v := eval("Bash", cmd(command)); len(v) != 0 {
		t.Errorf("heredoc program text must not be scanned as shell redirects, got %v", v)
	}
}

func TestInterpreterProgramComparisonIsNotBareDriveRedirect(t *testing.T) {
	command := "python -c 'if depth > 3: print(depth)'"
	if v := eval("Bash", cmd(command)); len(v) != 0 {
		t.Errorf("bare comparison operand 3: must not be treated as a Windows drive redirect, got %v", v)
	}
}

func TestOutOfTreeRedirectStillDenied(t *testing.T) {
	if v := eval("Bash", cmd("echo bad > /c/Users/u/work/tools/log.txt")); len(v) == 0 {
		t.Fatal("expected a violation for a real absolute sibling redirect")
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func cmd(c string) map[string]any { return map[string]any{"command": c} }
func fp(p string) map[string]any  { return map[string]any{"file_path": p} }
