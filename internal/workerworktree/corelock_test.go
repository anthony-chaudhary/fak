package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	// The gate this file exercises (internal/corelockgate) resolves witness claims
	// through a factory that internal/witness registers from its init. Production
	// binaries link witness anyway (cmd/fak reaches it through safecommit), but this
	// test binary links only workerworktree's own dependency closure — and
	// workerworktree is a tier-1 foundation leaf that MUST NOT import witness(2) in
	// non-test code. So the driver is linked here, in the test, exactly the way a
	// registration-seam driver always is. Without it the gate would fail CLOSED and
	// the witnessed-land cases below would refuse: that is the correct posture, not
	// the behaviour under test.
	_ "github.com/anthony-chaudhary/fak/internal/witness"
)

// The core-lock land gate (#5392). Every test here drives the REAL gate through
// Land: none of them stubs the classifier or the witness resolver, so a test that
// passes while the gate is absent is not possible — the refusal cases assert that
// NOTHING was applied or committed, which is exactly what a missing gate would do
// the opposite of.

const kernelPatch = "diff --git a/internal/adjudicator/decide.go b/internal/adjudicator/decide.go\n" +
	"--- a/internal/adjudicator/decide.go\n+++ b/internal/adjudicator/decide.go\n@@\n-old\n+new\n"

const leafPatch = "diff --git a/internal/tools/a.go b/internal/tools/a.go\n" +
	"--- a/internal/tools/a.go\n+++ b/internal/tools/a.go\n@@\n-old\n+new\n"

// coreLockFake stubs one land: declared-path admission, the captured patch, then
// the whole-diff pathset, followed by the worktree-tip message and green
// apply+commit. A
// land that reaches apply/commit therefore SUCCEEDS — so a refusal in these tests
// can only come from the gate under test.
func coreLockFake(patch, nameOnly, tipMsg string) *fakeGit {
	return replyLandDiff(newFakeGit(), nameOnly, patch, nameOnly).
		reply("log", 0, tipMsg).
		reply("apply", 0, "").
		reply("commit", 0, "[main abc1234] landed")
}

// installLiveDisambiguationFixture gives the tiny live repositories a complete,
// self-consistent concept corpus. The scorecard emits the same generated README
// and reverse-index bytes that are tracked in the repository, while an empty
// family catalog truthfully reports no ambiguity or coverage debt. Live land
// tests must satisfy the production post-apply gate, never stub or disable it.
func installLiveDisambiguationFixture(t *testing.T, repo string) {
	t.Helper()
	files := map[string]string{
		"docs/concept-disambiguation-scorecard/README.md":        "# Synthetic concept scorecard\n\nNo concepts are declared in this fixture.\n",
		"docs/concept-disambiguation-scorecard/INDEX.md":         "# Synthetic concept index\n\nNo concept names are declared in this fixture.\n",
		"docs/fak/concept-glossary.md":                           "# Synthetic concept glossary\n\nThis live-land fixture declares no concepts.\n",
		"tools/concept_disambiguation_scorecard.data/_meta.json": "{\"families\":[]}\n",
		"tools/concept_disambiguation_scorecard.py": `import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--workspace")
parser.add_argument("--json", action="store_true")
parser.add_argument("--markdown-dir", required=True)
args = parser.parse_args()
os.makedirs(args.markdown_dir, exist_ok=True)
with open(os.path.join(args.markdown_dir, "README.md"), "w", encoding="utf-8") as handle:
    handle.write("# Synthetic concept scorecard\n\nNo concepts are declared in this fixture.\n")
with open(os.path.join(args.markdown_dir, "INDEX.md"), "w", encoding="utf-8") as handle:
    handle.write("# Synthetic concept index\n\nNo concept names are declared in this fixture.\n")
print(json.dumps({"ok": True, "reason": "", "corpus": {"coverage_debt": 0, "clarity_defects": 0, "coverage": {"coverage_pct": 100, "per_family": []}}}))
`,
	}
	for rel, body := range files {
		path := filepath.Join(repo, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func landTouchedTrunk(g *fakeGit) bool {
	for _, verb := range []string{"apply", "commit", "read-tree", "write-tree", "commit-tree", "update-ref", "checkout"} {
		if len(g.callsWithPrefix(verb)) > 0 || len(g.envCallsWithPrefix(verb)) > 0 {
			return true
		}
	}
	return false
}

// TestLandRefusesUnwitnessedHardSelfCoreLockPath is the hole in #5392: a kernel
// path landing through the sanctioned worktree with no witness. Defaults are left
// ON (isolated index + readback) so the assertion covers the path production
// actually takes — the one that commits through commit-tree and runs no git hook.
func TestLandRefusesUnwitnessedHardSelfCoreLockPath(t *testing.T) {
	g := coreLockFake(kernelPatch, "internal/adjudicator/decide.go\n",
		"fix(adjudicator): rewire the decision kernel (fak adjudicator)")

	res := Land("/trunk", "/wt/fak-worker-wt-adj-abc", "feedface", "", []string{"internal/adjudicator"}, nil, g.run)

	if res.OK || res.Committed || res.Applied {
		t.Fatalf("unwitnessed hard-self land must be refused, got %+v", res)
	}
	for _, want := range []string{ReasonCoreSelfModify, "internal/adjudicator/decide.go", "missing maintenance witness", "--core-lock-maintenance-witness", CoreLockWitnessTrailer} {
		if !strings.Contains(res.Reason, want) {
			t.Fatalf("refusal reason missing %q:\n%s", want, res.Reason)
		}
	}
	if landTouchedTrunk(g) {
		t.Fatalf("a refused core-lock land must leave the trunk untouched; calls=%v envCalls=%v", g.calls, g.envCalls)
	}
	// The refusal an operator actually reads, logged so `-v` shows the whole
	// reason string (token, cause, both remedies, offending paths) verbatim.
	t.Logf("refusal: %s", res.Reason)
}

// TestLandAllowsHardSelfWithConfirmedWitnessFlag is the escape hatch: the SAME
// claim vocabulary `fak commit --core-lock-maintenance-witness` accepts, resolved
// through the injected git seam. It also pins that the claim is genuinely
// RESOLVED (a cat-file evidence call is issued), not merely present.
func TestLandAllowsHardSelfWithConfirmedWitnessFlag(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "0")
	t.Setenv(LandReadbackEnv, "0")
	g := coreLockFake(kernelPatch, "internal/adjudicator/decide.go\n",
		"fix(adjudicator): rewire the decision kernel (fak adjudicator)").
		reply("cat-file", 0, "")

	res := Land("/trunk", "/wt/fak-worker-wt-adj-abc", "feedface", "", []string{"internal/adjudicator"}, nil, g.run,
		WithCoreLockWitness("commit:0f1e2d3c4b5a"))

	if !res.OK || !res.Committed {
		t.Fatalf("a CONFIRMED maintenance witness must let the land through, got %+v", res)
	}
	evidence := g.callsWithPrefix("cat-file")
	if len(evidence) == 0 {
		t.Fatalf("the witness claim must be resolved against evidence, not just accepted; calls=%v", g.calls)
	}
	if !contains(evidence[0], "0f1e2d3c4b5a^{commit}") {
		t.Fatalf("witness evidence call did not carry the claimed object: %v", evidence[0])
	}
}

// TestLandAllowsHardSelfWithConfirmedWitnessTrailer covers the DISPATCHED land,
// which has no CLI to pass a flag through: the worker expresses the same claim as
// a trailer in the commit message Land derives from its worktree tip.
func TestLandAllowsHardSelfWithConfirmedWitnessTrailer(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "0")
	t.Setenv(LandReadbackEnv, "0")
	tip := "fix(adjudicator): rewire the decision kernel (fak adjudicator)\n\n" +
		CoreLockWitnessTrailer + ": commit:0f1e2d3c4b5a\n"
	g := coreLockFake(kernelPatch, "internal/adjudicator/decide.go\n", tip).
		reply("cat-file", 0, "")

	res := Land("/trunk", "/wt/fak-worker-wt-adj-abc", "feedface", "", []string{"internal/adjudicator"}, nil, g.run)

	if !res.OK || !res.Committed {
		t.Fatalf("a CONFIRMED trailer witness must let the land through, got %+v", res)
	}
	if len(g.callsWithPrefix("cat-file")) == 0 {
		t.Fatalf("the trailer claim must be resolved against evidence; calls=%v", g.calls)
	}
}

// TestLandRefusesHardSelfWhenWitnessRefuted is the non-weakening half: a claim
// that is PRESENT but does not check out keeps the lock closed. Without this a
// "witness accepted" test would pass against a gate that only looks for a
// non-empty string.
func TestLandRefusesHardSelfWhenWitnessRefuted(t *testing.T) {
	g := coreLockFake(kernelPatch, "internal/adjudicator/decide.go\n",
		"fix(adjudicator): rewire the decision kernel (fak adjudicator)").
		reply("cat-file", 1, "") // the claimed object does not exist -> REFUTED

	res := Land("/trunk", "/wt/fak-worker-wt-adj-abc", "feedface", "", []string{"internal/adjudicator"}, nil, g.run,
		WithCoreLockWitness("commit:0f1e2d3c4b5a"))

	if res.OK || res.Committed {
		t.Fatalf("a refuted witness must keep the land refused, got %+v", res)
	}
	if !strings.Contains(res.Reason, "refuted") {
		t.Fatalf("refusal should name the refuted witness:\n%s", res.Reason)
	}
	if landTouchedTrunk(g) {
		t.Fatalf("a refused core-lock land must leave the trunk untouched; calls=%v envCalls=%v", g.calls, g.envCalls)
	}
}

// TestLandOrdinaryLeafPathIsUnaffected pins the blast radius: the gate fires ONLY
// on the declared core-locked surface. An ordinary leaf lands with no witness and
// no evidence call at all.
func TestLandOrdinaryLeafPathIsUnaffected(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "0")
	t.Setenv(LandReadbackEnv, "0")
	g := coreLockFake(leafPatch, "internal/tools/a.go\n", "feat(tools): widen the lane (fak tools)")

	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "", []string{"internal/tools"}, nil, g.run)

	if !res.OK || !res.Committed {
		t.Fatalf("an ordinary leaf land must be unaffected by the core lock, got %+v", res)
	}
	if len(g.callsWithPrefix("cat-file")) != 0 {
		t.Fatalf("no witness may be demanded for an open-leaf path; calls=%v", g.calls)
	}
}

// TestLandUnreadableNameListStillClassifiesFromTheDiff closes the obvious bypass:
// if `git diff --name-only` fails, the gate must fall back to the captured diff's
// own headers rather than see an empty pathset and wave the land through.
func TestLandUnreadableNameListStillClassifiesFromTheDiff(t *testing.T) {
	g := newFakeGit().
		replyOnce("diff", 0, kernelPatch).
		replyOnce("diff", 128, ""). // --name-only fails
		reply("log", 0, "fix(adjudicator): rewire the decision kernel (fak adjudicator)").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc1234] landed")

	res := Land("/trunk", "/wt/fak-worker-wt-adj-abc", "feedface", "", nil, nil, g.run)

	if res.OK || res.Committed {
		t.Fatalf("an unreadable name list must not disable the core lock, got %+v", res)
	}
	if !strings.Contains(res.Reason, "internal/adjudicator/decide.go") {
		t.Fatalf("fallback classification should name the locked path:\n%s", res.Reason)
	}
}

func TestCoreLockWitnessFromMessageReadsTrailerNotProse(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"absent", "fix(x): thing\n", ""},
		{"trailer", "fix(x): thing\n\n" + CoreLockWitnessTrailer + ": commit:abc123\n", "commit:abc123"},
		{"case-insensitive key", "fix(x): thing\n\ncore-lock-maintenance-witness: ancestor:abc\n", "ancestor:abc"},
		{"last one wins", "h\n\n" + CoreLockWitnessTrailer + ": commit:aaa\n" + CoreLockWitnessTrailer + ": commit:bbb\n", "commit:bbb"},
		{"empty value is no claim", "h\n\n" + CoreLockWitnessTrailer + ":   \n", ""},
		{"mid-line prose is not a trailer", "h\n\nI would add " + CoreLockWitnessTrailer + ": commit:aaa here\n", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := coreLockWitnessFromMessage(tc.body); got != tc.want {
				t.Fatalf("claim = %q, want %q", got, tc.want)
			}
		})
	}
}

// ---- live git ------------------------------------------------------------- //

// TestLiveCoreLockRefusesUnwitnessedKernelLand drives the whole sanctioned path
// against a REAL repo and a REAL detached worktree: prepare -> edit a core-locked
// kernel file -> land. The land is refused, the trunk is byte-identical, and the
// SAME land with a witness that really resolves (the base commit exists) goes
// through. No fakes anywhere in this test.
func TestLiveCoreLockRefusesUnwitnessedKernelLand(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	git := func(args ...string) string {
		c := exec.Command("git", args...)
		c.Dir = repo
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	git("init", "-q", "-b", "main")
	git("config", "user.email", "corelock@test")
	git("config", "user.name", "corelock")
	git("config", "commit.gpgsign", "false")
	kernel := filepath.Join(repo, "internal", "adjudicator")
	if err := os.MkdirAll(kernel, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(kernel, "decide.go"), []byte("package adjudicator\n\nfunc Decide() bool { return false }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	installLiveDisambiguationFixture(t, repo)
	git("add", ".")
	git("commit", "-q", "-m", "base")

	base := TrunkHeadSHA(repo, nil)
	if base == "" {
		t.Fatal("no trunk head")
	}

	wtRoot := t.TempDir()
	prep := Prepare(repo, "adjudicator", "5392", base, wtRoot, nil)
	if !prep.OK {
		t.Fatalf("prepare: %+v", prep)
	}
	wc := func(args ...string) {
		c := exec.Command("git", args...)
		c.Dir = prep.Path
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("worktree git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	wc("config", "user.email", "worker@test")
	wc("config", "user.name", "worker")
	wc("config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(prep.Path, "internal", "adjudicator", "decide.go"),
		[]byte("package adjudicator\n\nfunc Decide() bool { return true }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	wc("add", "internal/adjudicator/decide.go")
	wc("commit", "-q", "-m", "fix(adjudicator): always allow (#5392) (fak adjudicator)")

	// (1) No witness -> refused, and the trunk is untouched.
	refused := Land(repo, prep.Path, base, "", []string{"internal/adjudicator"}, nil, nil)
	if refused.OK || refused.Committed || refused.Applied {
		t.Fatalf("live unwitnessed kernel land must be refused, got %+v", refused)
	}
	if !strings.Contains(refused.Reason, ReasonCoreSelfModify) {
		t.Fatalf("refusal reason = %q, want %s", refused.Reason, ReasonCoreSelfModify)
	}
	if head := TrunkHeadSHA(repo, nil); head != base {
		t.Fatalf("trunk HEAD moved on a refused land: %s -> %s", base, head)
	}
	if status := strings.TrimSpace(git("status", "--porcelain")); status != "" {
		t.Fatalf("refused land dirtied the trunk worktree:\n%s", status)
	}
	if body, _ := os.ReadFile(filepath.Join(kernel, "decide.go")); strings.Contains(string(body), "return true") {
		t.Fatalf("refused land still wrote the kernel edit to the trunk:\n%s", body)
	}

	// (2) The SAME land with a witness that really resolves (the base commit
	// exists in this repo, so `commit:<base>` is CONFIRMED) goes through.
	landed := Land(repo, prep.Path, base, "", []string{"internal/adjudicator"}, nil, nil,
		WithCoreLockWitness("commit:"+base))
	if !landed.OK || !landed.Committed {
		t.Fatalf("witnessed kernel land must go through, got %+v", landed)
	}
	if head := TrunkHeadSHA(repo, nil); head == base {
		t.Fatalf("witnessed land did not move the trunk (still %s)", base)
	}
	body, _ := os.ReadFile(filepath.Join(kernel, "decide.go"))
	if !strings.Contains(string(body), "return true") {
		t.Fatalf("witnessed land did not carry the kernel edit:\n%s", body)
	}
	_ = Reap(repo, prep.Path, nil)
}

// TestLiveCoreLockOrdinaryLeafLandsWithoutWitness is the live control: the same
// machinery, an ordinary leaf path, no witness — it lands, unchanged by #5392.
func TestLiveCoreLockOrdinaryLeafLandsWithoutWitness(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	git := func(args ...string) string {
		c := exec.Command("git", args...)
		c.Dir = repo
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	git("init", "-q", "-b", "main")
	git("config", "user.email", "corelock@test")
	git("config", "user.name", "corelock")
	git("config", "commit.gpgsign", "false")
	leaf := filepath.Join(repo, "internal", "tools")
	if err := os.MkdirAll(leaf, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(leaf, "a.go"), []byte("package tools\n\nfunc A() int { return 1 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	installLiveDisambiguationFixture(t, repo)
	git("add", ".")
	git("commit", "-q", "-m", "base")

	base := TrunkHeadSHA(repo, nil)
	wtRoot := t.TempDir()
	prep := Prepare(repo, "tools", "5392-leaf", base, wtRoot, nil)
	if !prep.OK {
		t.Fatalf("prepare: %+v", prep)
	}
	wc := func(args ...string) {
		c := exec.Command("git", args...)
		c.Dir = prep.Path
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("worktree git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	wc("config", "user.email", "worker@test")
	wc("config", "user.name", "worker")
	wc("config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(prep.Path, "internal", "tools", "a.go"),
		[]byte("package tools\n\nfunc A() int { return 42 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	wc("add", "internal/tools/a.go")
	wc("commit", "-q", "-m", "feat(tools): return 42 (#5392) (fak tools)")

	res := Land(repo, prep.Path, base, "", []string{"internal/tools"}, nil, nil)
	if !res.OK || !res.Committed {
		t.Fatalf("open-leaf land must be unaffected, got %+v", res)
	}
	body, _ := os.ReadFile(filepath.Join(leaf, "a.go"))
	if !strings.Contains(string(body), "return 42") {
		t.Fatalf("leaf land did not carry the edit:\n%s", body)
	}
	_ = Reap(repo, prep.Path, nil)
}
