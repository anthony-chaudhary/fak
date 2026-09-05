package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// hooks_test.go — end-to-end CLI tests for `fak hooks` against a real temp git repo. Skipped if
// git is absent or under -short.

func gitHook(t *testing.T, repo string, args ...string) {
	t.Helper()
	c := exec.Command("git", append([]string{"-C", repo}, args...)...)
	c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	if out, err := c.CombinedOutput(); err != nil {
		t.Skipf("git %v: %s", args, out)
	}
}

func newRepoWith(t *testing.T, files map[string]string) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitHook(t, repo, "init", "-q", "-b", "main")
	gitHook(t, repo, "config", "user.email", "t@t")
	gitHook(t, repo, "config", "user.name", "t")
	for p, content := range files {
		full := filepath.Join(repo, filepath.FromSlash(p))
		_ = os.MkdirAll(filepath.Dir(full), 0o755)
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		gitHook(t, repo, "add", "--", p)
	}
	return repo
}

func hookLeakIP() string { return "100" + ".64.0.10" }

func TestRunHooks_preCommitClean(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("clean staged set should pass (0), got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_preCommitBlocksLeak(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the host is " + hookLeakIP() + " here\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 1 {
		t.Fatalf("a leaked needle should block (1), got %d; stderr=%s", code, errb.String())
	}
}

// #1455: a staged doc whose CONTENT carries a prose hardware tell is refused at commit
// time (not only post-hoc in make ci), and the refusal names the `scrub_hardware_names.py
// --apply <file>` recovery so the author can fix it before it reds the trunk fleet-wide.
func TestRunHooks_preCommitHardwareTellGivesApplyHint(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the run was on DGX overnight\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 1 {
		t.Fatalf("a prose hardware tell in doc content should block (1), got %d; stderr=%s", code, errb.String())
	}
	s := errb.String()
	if !bytes.Contains(errb.Bytes(), []byte("HARDWARE_TELL")) {
		t.Fatalf("refusal should name HARDWARE_TELL, got %s", s)
	}
	if !bytes.Contains(errb.Bytes(), []byte("scrub_hardware_names.py --apply docs/a.md")) {
		t.Fatalf("refusal should carry the --apply recovery hint naming the file, got %s", s)
	}
}

// #1455: the new gate must reuse the FALSE-POSITIVE-safe masking — a hardware token that
// appears ONLY as filename link-text is an identifier, not a prose tell, so the commit
// passes (this is the exact FP that motivated the issue).
func TestRunHooks_preCommitHardwareTellMasksLinkText(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{
		"docs/a.md": "see [DGX-OVERNIGHT-PLAN](notes/plan.md) for the schedule\n",
	})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("a hardware token only in filename link-text must not block, got %d; stderr=%s", code, errb.String())
	}
}

// #1455: FLEET_ALLOW_HW=1 escapes the doc-content gate once (the meta-case: a commit about
// the scrubber itself).
func TestRunHooks_preCommitHardwareTellEscapes(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the run was on DGX overnight\n"})
	t.Setenv("FLEET_ALLOW_HW", "1")
	var out, errb bytes.Buffer
	if code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo}); code != 0 {
		t.Fatalf("FLEET_ALLOW_HW=1 should escape the hardware doc gate, got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_preCommitJSON(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the host is " + hookLeakIP() + " here\n"})
	var out, errb bytes.Buffer
	_ = runHooks(&out, &errb, []string{"pre-commit", "--root", repo, "--json"})
	if !bytes.Contains(out.Bytes(), []byte("PUBLIC_LEAK")) {
		t.Fatalf("--json should carry the gate name; got %s", out.String())
	}
}

func TestRunHooks_preCommitOffEnvSkips(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the host is " + hookLeakIP() + " here\n"})
	t.Setenv("FLEET_SCRUB_GUARD", "off")
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("with the leak gate off, the commit should pass; got %d", code)
	}
}

// #3615: a bare (no-pathspec) commit that did NOT come through `fak commit` — a raw
// `git add <path> && git commit` — is refused in block mode with BARE_COMMIT_SWEEP naming
// the staged path it would sweep. src/x.go is the known-clean shape (see preCommitClean) so
// ONLY the bare-sweep gate fires, not a leak/index gate.
func TestRunHooks_preCommitBareSweepBlocks(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	t.Setenv("FLEET_BARE_COMMIT_GUARD", "block")
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 1 {
		t.Fatalf("an unvetted bare commit should block in block mode (1), got %d; stderr=%s", code, errb.String())
	}
	if !bytes.Contains(errb.Bytes(), []byte("BARE_COMMIT_SWEEP")) {
		t.Fatalf("refusal should name BARE_COMMIT_SWEEP, got %s", errb.String())
	}
	if !bytes.Contains(errb.Bytes(), []byte("src/x.go")) {
		t.Fatalf("refusal should name the staged path it would sweep, got %s", errb.String())
	}
}

// #3615: the FAK_SAFECOMMIT_VETTED handshake (set by safecommit on its own `git commit`)
// stands the gate down even in block mode — fak's vetted, path-scoped commit is not a sweep.
func TestRunHooks_preCommitBareSweepVettedMarkerPasses(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	t.Setenv("FLEET_BARE_COMMIT_GUARD", "block")
	t.Setenv("FAK_SAFECOMMIT_VETTED", "1")
	var out, errb bytes.Buffer
	if code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo}); code != 0 {
		t.Fatalf("a vetted commit (FAK_SAFECOMMIT_VETTED=1) must pass even in block mode, got %d; stderr=%s", code, errb.String())
	}
}

// #3615: out of the box the gate is ADVISORY — an unvetted bare commit does NOT block (warn
// default), but the finding still surfaces (here via --json) so a soak can measure it.
func TestRunHooks_preCommitBareSweepWarnDefault(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	var out, errb bytes.Buffer
	if code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo, "--json"}); code != 0 {
		t.Fatalf("warn-default: an unvetted bare commit must not block, got %d; stderr=%s", code, errb.String())
	}
	if !bytes.Contains(out.Bytes(), []byte("BARE_COMMIT_SWEEP")) {
		t.Fatalf("--json should carry the advisory BARE_COMMIT_SWEEP finding; got %s", out.String())
	}
}

// #3615: FAK_PRESTAGED_PATH_GUARD=off disables the whole prestaged family, this gate included,
// so a deliberate one-shot opt-out of the family silences the bare-sweep gate too.
func TestRunHooks_preCommitBareSweepFamilyOff(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	t.Setenv("FLEET_BARE_COMMIT_GUARD", "block")
	t.Setenv("FAK_PRESTAGED_PATH_GUARD", "off")
	var out, errb bytes.Buffer
	if code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo}); code != 0 {
		t.Fatalf("FAK_PRESTAGED_PATH_GUARD=off must disable the gate even in block mode, got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_commitMsgVerbShape(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	dir := t.TempDir()
	good := filepath.Join(dir, "good.txt")
	bad := filepath.Join(dir, "bad.txt")
	_ = os.WriteFile(good, []byte("feat(x): add a thing\n"), 0o644)
	_ = os.WriteFile(bad, []byte("docs: clean up stuff\n"), 0o644) // 'clean' not a verb; warn-only

	var out, errb bytes.Buffer
	// default mode is warn -> exit 0 even on a noun-led subject.
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 0 {
		t.Fatalf("commit-msg defaults to warn; should not block, got %d", code)
	}
	// block mode -> a bad subject exits 1.
	t.Setenv("FLEET_MSG_GUARD", "block")
	errb.Reset()
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 1 {
		t.Fatalf("FLEET_MSG_GUARD=block should block a noun-led subject, got %d", code)
	}
	errb.Reset()
	if code := runHooks(&out, &errb, []string{"commit-msg", good}); code != 0 {
		t.Fatalf("a gradeable subject should pass even in block mode, got %d; %s", code, errb.String())
	}
}

func TestRunHooks_commitMsgBlocksHardwareTell(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.txt")
	_ = os.WriteFile(bad, []byte("docs(nightrun): add the dgx3 decode (fak nightrun)\n"), 0o644)

	// The public-leak gate intentionally runs first and recognizes the same private
	// host alias. Escape that gate so this test reaches the distinct hardware-tell
	// refusal it is meant to witness.
	t.Setenv("FLEET_ALLOW_LEAK", "1")
	var out, errb bytes.Buffer
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 1 {
		t.Fatalf("hardware tell should block, got %d; stderr=%s", code, errb.String())
	}
	if !bytes.Contains(errb.Bytes(), []byte("HARDWARE_TELL")) {
		t.Fatalf("stderr should name HARDWARE_TELL, got %s", errb.String())
	}

	t.Setenv("FLEET_ALLOW_HW", "1")
	errb.Reset()
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 0 {
		t.Fatalf("FLEET_ALLOW_HW should escape the hardware gate, got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_commitMsgBlocksUnnamedFreshDeletion(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	p := "docs/notes/SLACK-CONTROL-FOUNDATION-2026-07-02.md"
	repo := newRepoWith(t, map[string]string{p: "# Slack control\n"})
	gitHook(t, repo, "commit", "-m", "docs(notes): add slack control foundation note")
	if err := os.Remove(filepath.Join(repo, filepath.FromSlash(p))); err != nil {
		t.Fatal(err)
	}
	gitHook(t, repo, "add", "--all", "--", p)

	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.txt")
	good := filepath.Join(dir, "good.txt")
	_ = os.WriteFile(bad, []byte("docs(notes): bind no-babysitting doctrine\n"), 0o644)
	_ = os.WriteFile(good, []byte("docs(notes): remove Slack Control Foundation note\n"), 0o644)
	t.Setenv("FLEET_FRESH_DELETE_GUARD", "block")

	var out, errb bytes.Buffer
	if code := runHooks(&out, &errb, []string{"commit-msg", "--root", repo, bad}); code != 1 {
		t.Fatalf("unnamed fresh deletion should block, got %d; stderr=%s", code, errb.String())
	}
	if !bytes.Contains(errb.Bytes(), []byte("FRESH_DELETION")) {
		t.Fatalf("stderr should name FRESH_DELETION, got %s", errb.String())
	}

	out.Reset()
	errb.Reset()
	if code := runHooks(&out, &errb, []string{"commit-msg", "--root", repo, good}); code != 0 {
		t.Fatalf("message naming the deleted note should pass, got %d; stderr=%s", code, errb.String())
	}
}

func TestEmitFindingsJSONOrdersBlockingBeforeAdvisory(t *testing.T) {
	findings := []hooks.Finding{
		{Gate: "PUBLIC_LEAK", File: "docs/leak.md", Line: 2, Detail: "redact", Advisory: true},
		{Gate: "DUPLICATION", File: "internal/x/new.go", Line: 9, Detail: "copied block"},
	}
	var out, errb bytes.Buffer
	orderFindingsForRepair(findings)
	emitFindingsJSON(&out, &errb, findings, nil, nil, runScope{})
	if errb.Len() != 0 {
		t.Fatalf("emitFindingsJSON stderr = %s", errb.String())
	}
	var report struct {
		Findings []hooks.Finding `json:"findings"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, out.String())
	}
	if len(report.Findings) != 2 {
		t.Fatalf("findings = %#v, want two", report.Findings)
	}
	if report.Findings[0].Gate != "DUPLICATION" || report.Findings[0].Advisory {
		t.Fatalf("findings[0] = %#v, want binding DUPLICATION", report.Findings[0])
	}
	if report.Findings[1].Gate != "PUBLIC_LEAK" || !report.Findings[1].Advisory {
		t.Fatalf("findings[1] = %#v, want advisory PUBLIC_LEAK", report.Findings[1])
	}
}

// TestHooksCommitMsg_rejectsSingleParentPseudoMerge proves issue #10882:
// `fak hooks commit-msg` rejects a single-parent commit with subject `Merge ...` with COMMIT_MSG / MERGE_WITNESS_FAIL.
func TestHooksCommitMsg_rejectsSingleParentPseudoMerge(t *testing.T) {
	repo := t.TempDir()
	gitHook(t, repo, "init", "-q")
	gitHook(t, repo, "config", "user.name", "Test")
	gitHook(t, repo, "config", "user.email", "test@example.com")
	gitHook(t, repo, "config", "commit.gpgsign", "false")
	gitHook(t, repo, "commit", "--allow-empty", "-m", "initial commit")

	msgFile := filepath.Join(t.TempDir(), "msg.txt")
	if err := os.WriteFile(msgFile, []byte("Merge branch 'main' into dev\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"commit-msg", "--root", repo, msgFile})
	if code != 1 {
		t.Fatalf("runHooks commit-msg for pseudo-merge want exit 1, got %d; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "COMMIT_MSG") {
		t.Errorf("stderr %q does not contain COMMIT_MSG", errb.String())
	}
	if !strings.Contains(errb.String(), "MERGE_WITNESS_FAIL") {
		t.Errorf("stderr %q does not contain MERGE_WITNESS_FAIL", errb.String())
	}
}

// TestHooksCommitMsg_rejectsConflictBanners proves issue #11306:
// `fak hooks commit-msg` rejects messages containing conflict templates or conflict markers.
func TestHooksCommitMsg_rejectsConflictBanners(t *testing.T) {
	msgFile := filepath.Join(t.TempDir(), "msg.txt")

	// 1. Conflict template
	if err := os.WriteFile(msgFile, []byte("Merge remote-tracking branch 'origin/main'\n\n# Conflicts:\n#\tcmd/fak/serve.go\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"commit-msg", msgFile})
	if code != 1 {
		t.Fatalf("runHooks commit-msg for conflict template want exit 1, got %d; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "MERGE_CONFLICT_TEMPLATE_FORBIDDEN") {
		t.Errorf("stderr %q does not contain MERGE_CONFLICT_TEMPLATE_FORBIDDEN", errb.String())
	}

	// 2. Conflict marker
	errb.Reset()
	out.Reset()
	if err := os.WriteFile(msgFile, []byte("feat(core): update\n\n<<<<<<< HEAD\nfoo\n=======\nbar\n>>>>>>> main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	code = runHooks(&out, &errb, []string{"commit-msg", msgFile})
	if code != 1 {
		t.Fatalf("runHooks commit-msg for conflict markers want exit 1, got %d; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "MERGE_CONFLICT_MARKERS_FORBIDDEN") {
		t.Errorf("stderr %q does not contain MERGE_CONFLICT_MARKERS_FORBIDDEN", errb.String())
	}
}

// TestHooksCommitMsg_rejectsSilentDropMerge proves issue #11306:
// `fak hooks commit-msg` rejects merge commits where the tree SHA matches parent 1 exactly
// while parent 2 contains non-empty unique commits.
func TestHooksCommitMsg_rejectsSilentDropMerge(t *testing.T) {
	repo := t.TempDir()
	gitHook(t, repo, "init", "-q")
	gitHook(t, repo, "config", "user.name", "Test")
	gitHook(t, repo, "config", "user.email", "test@example.com")
	gitHook(t, repo, "config", "commit.gpgsign", "false")

	if err := os.WriteFile(filepath.Join(repo, "a.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitHook(t, repo, "add", "a.txt")
	gitHook(t, repo, "commit", "-m", "feat(core): initial base")

	gitHook(t, repo, "checkout", "-q", "-b", "side-branch")
	if err := os.WriteFile(filepath.Join(repo, "b.txt"), []byte("side change\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitHook(t, repo, "add", "b.txt")
	gitHook(t, repo, "commit", "-m", "feat(side): side change")

	gitHook(t, repo, "checkout", "-q", "-")
	if err := os.WriteFile(filepath.Join(repo, "c.txt"), []byte("main change\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitHook(t, repo, "add", "c.txt")
	gitHook(t, repo, "commit", "-m", "feat(main): main change")

	// Merge with ours strategy
	gitHook(t, repo, "merge", "-s", "ours", "--no-commit", "side-branch")

	msgFile := filepath.Join(t.TempDir(), "msg.txt")
	if err := os.WriteFile(msgFile, []byte("Merge branch 'side-branch'\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"commit-msg", "--root", repo, msgFile})
	if code != 1 {
		t.Fatalf("runHooks commit-msg for silent drop merge want exit 1, got %d; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "SILENT_DROP_MERGE_FORBIDDEN") {
		t.Errorf("stderr %q does not contain SILENT_DROP_MERGE_FORBIDDEN", errb.String())
	}

	// Allowed with Merge-Strategy: ours trailer
	errb.Reset()
	out.Reset()
	if err := os.WriteFile(msgFile, []byte("Merge branch 'side-branch'\n\nMerge-Strategy: ours\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	code = runHooks(&out, &errb, []string{"commit-msg", "--root", repo, msgFile})
	if code != 0 {
		t.Fatalf("runHooks commit-msg with Merge-Strategy: ours trailer want exit 0, got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_preCommitBlocksCommittedRed(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package main\n"})

	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	commitBuildCheckGate = func(_ io.Writer, _ string, _ []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckFailed, "syntax error: unexpected identifier"
	}

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 1 {
		t.Fatalf("prospective build failure should block (1), got %d; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "COMMITTED_RED") {
		t.Fatalf("stderr should name COMMITTED_RED, got %s", errb.String())
	}
	if !strings.Contains(errb.String(), "syntax error: unexpected identifier") {
		t.Fatalf("stderr should carry diagnostic detail, got %s", errb.String())
	}
}

func TestRunHooks_preCommitCommittedRedJSON(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package main\n"})

	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	commitBuildCheckGate = func(_ io.Writer, _ string, _ []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckFailed, "deliberate syntax error"
	}

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo, "--json"})
	if code != 1 {
		t.Fatalf("prospective build failure should block in JSON mode (1), got %d; stderr=%s", code, errb.String())
	}
	var payload struct {
		Findings []hooks.Finding `json:"findings"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("unmarshal findings JSON: %v\n%s", err, out.String())
	}
	found := false
	for _, f := range payload.Findings {
		if f.Gate == "COMMITTED_RED" && strings.Contains(f.Detail, "deliberate syntax error") {
			found = true
			if f.Advisory {
				t.Fatalf("COMMITTED_RED finding should be binding (Advisory=false), got %+v", f)
			}
			break
		}
	}
	if !found {
		t.Fatalf("COMMITTED_RED finding not found in JSON payload: %+v", payload.Findings)
	}
}

func TestRunHooks_preCommitCommittedRedEscapes(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package main\n"})

	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	commitBuildCheckGate = func(_ io.Writer, _ string, _ []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckFailed, "syntax error"
	}

	t.Setenv("ALLOW_COMMITTED_RED", "1")
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("ALLOW_COMMITTED_RED=1 should escape buildcheck, got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_preCommitCommittedRedOff(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package main\n"})

	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	called := false
	commitBuildCheckGate = func(_ io.Writer, _ string, _ []string) (safecommit.BuildCheckOutcome, string) {
		called = true
		return safecommit.BuildCheckFailed, "syntax error"
	}

	t.Setenv("FLEET_BUILDCHECK_GUARD", "off")
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("FLEET_BUILDCHECK_GUARD=off should skip buildcheck, got %d; stderr=%s", code, errb.String())
	}
	if called {
		t.Fatal("commitBuildCheckGate should NOT be called when FLEET_BUILDCHECK_GUARD=off")
	}
}

func TestRunHooks_preCommitCommittedRedSkipsNonGo(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/readme.md": "# Readme\n"})

	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	called := false
	commitBuildCheckGate = func(_ io.Writer, _ string, _ []string) (safecommit.BuildCheckOutcome, string) {
		called = true
		return safecommit.BuildCheckFailed, "syntax error"
	}

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("non-Go staged diff should not trigger buildcheck, got %d; stderr=%s", code, errb.String())
	}
	if called {
		t.Fatal("commitBuildCheckGate should NOT be called when no .go files are staged")
	}
}
